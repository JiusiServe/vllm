# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import asyncio
import os
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Mapping
from typing import Optional, Union

import msgspec
import numpy as np
import random
import uvicorn
import zmq
import zmq.asyncio
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from vllm.config import DecodingConfig, ModelConfig, VllmConfig
from vllm.core.scheduler import SchedulerOutputs
from vllm.disaggregated.protocol import (FailureResponse, GenerationRequest,
                                         GenerationResponse, RequestType,
                                         ResponseType)
from vllm.engine.protocol import EngineClient
from vllm.inputs.data import PromptType
from vllm.inputs.preprocess import InputPreprocessor
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.outputs import CompletionOutput, PoolingRequestOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.transformers_utils.tokenizer_group import TokenizerGroup
from vllm.utils import Device
from vllm.v1.outputs import SamplerOutput

logger = init_logger(__name__)

app = FastAPI()


class Proxy(EngineClient):
    """
    Proxy
    """

    def __init__(
        self,
        proxy_addr: str,
        encode_addr_list: list[str],
        pd_addr_list: list[str],
        model_name: str,
    ):
        self.queues: dict[str, asyncio.Queue] = {}

        self.encoder = msgspec.msgpack.Encoder()

        self.ctx = zmq.asyncio.Context()
        self.proxy_addr = f"ipc://{proxy_addr}"
        self.encode_addr_list = [f"ipc://{addr}" for addr in encode_addr_list]
        self.pd_addr_list = [f"ipc://{addr}" for addr in pd_addr_list]
        self.to_encode_sockets = []
        for addr in self.encode_addr_list:
            socket = self.ctx.socket(zmq.constants.PUSH)
            socket.connect(addr)
            self.to_encode_sockets.append(socket)
        self.to_pd_sockets = []
        for addr in self.pd_addr_list:
            socket = self.ctx.socket(zmq.constants.PUSH)
            socket.connect(addr)
            self.to_pd_sockets.append(socket)

        self.output_handler: Optional[asyncio.Task] = None

        # Dummy: needed for EngineClient Protocol.
        self.model_config = ModelConfig(
            model=model_name,
            tokenizer=model_name,
            tokenizer_mode="auto",
            trust_remote_code=False,
            dtype="auto",
            task="generate",
            seed=42,
        )

        # Dummy: needed for EngineClient Protocol.
        # TODO: refactor OAI Server to avoid needing this.
        self.tokenizer = TokenizerGroup(**dict(
            tokenizer_id=self.model_config.tokenizer,
            enable_lora=False,
            max_num_seqs=1024,
            max_loras=0,
            max_input_length=None,
            tokenizer_mode=self.model_config.tokenizer_mode,
            trust_remote_code=self.model_config.trust_remote_code,
            revision=self.model_config.tokenizer_revision,
            truncation_side=self.model_config.truncation_side,
        ))

    def shutdown(self):
        self.ctx.destroy()
        if (task := self.output_handler) is not None:
            task.cancel()

        socket_path = self.proxy_addr.replace("ipc://", "")
        if os.path.exists(socket_path):
            os.remove(socket_path)

    async def _run_encode(
        self,
        request: GenerationRequest,
        q: asyncio.Queue[Union[Exception, GenerationResponse]],
    ) -> None:
        """
        Send the encode request to one encoder worker.
        The encoder worker is selected based on hashing the request ID.
        """
        if not self.to_encode_sockets:
            raise RuntimeError(
                "No encode workers configured: encode_addr_list is empty.")

        try:
            payload = self.encoder.encode(request)
        except Exception as e:
            raise RuntimeError("Failed to serialize GenerationRequest") from e

        msg = (RequestType.ENCODE, payload)
        idx = random.randint(0, len(self.to_encode_sockets) - 1)
        socket = self.to_encode_sockets[idx]
        await socket.send_multipart(msg, copy=False)

        response = await q.get()
        logger.info("Encode response: %s", response)
        if isinstance(response, Exception):
            raise response

    async def _run_pd(
        self,
        request: GenerationRequest,
        q: asyncio.Queue[Union[Exception, GenerationResponse]],
    ):
        """
        Send the generation request to a PD worker and yield its response.
        The PD worker is selected based on hashing the request ID.
        """
        if not self.to_pd_sockets:
            raise RuntimeError(
                "No PD workers configured: pd_addr_list is empty.")

        try:
            payload = self.encoder.encode(request)
        except Exception as e:
            raise RuntimeError("Failed to serialize GenerationRequest") from e

        msg = (RequestType.GENERATION, payload)
        idx = random.randint(0, len(self.to_pd_sockets) - 1)
        socket = self.to_pd_sockets[idx]
        await socket.send_multipart(msg, copy=False)

        finished = False
        while not finished:
            response = await q.get()
            if isinstance(response, Exception):
                raise response
            finished = response.finish_reason is not None
            yield response

    def _to_request_output(self, resp: GenerationResponse) -> RequestOutput:
        """Convert a PD/Generate response to vLLM RequestOutput.

        This creates a single CompletionOutput. If the response includes
        text/token_ids attributes, they are used; otherwise defaults are used.
        """
        text = getattr(resp, "text", "")
        token_ids = getattr(resp, "token_ids", [])

        completion = CompletionOutput(
            index=0,
            text=text,
            token_ids=token_ids,
            cumulative_logprob=None,
            logprobs=None,
            finish_reason=resp.finish_reason,
            stop_reason=resp.stop_reason,
        )

        return RequestOutput(
            request_id=resp.request_id,
            prompt=None,
            prompt_token_ids=resp.prompt_token_ids,
            prompt_logprobs=None,
            outputs=[completion],
            finished=resp.finish_reason is not None,
        )

    async def generate(
        self,
        prompt: PromptType,
        sampling_params: SamplingParams,
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        priority: int = 0,
        data_parallel_rank: Optional[int] = None,
    ):
        if self.output_handler is None:
            self.output_handler = asyncio.create_task(
                self._run_output_handler())
        if not request_id:
            request_id = uuid.uuid4().hex

        q = asyncio.Queue()
        self.queues[request_id] = q

        # Support both raw string prompts and dict prompts with multimodal data
        prompt_text = prompt["prompt"] if isinstance(prompt, dict) else prompt

        request = GenerationRequest(
            request_id=request_id,
            prompt=prompt_text,
            sampling_params=sampling_params,
        )

        if _has_mm_data(prompt):
            request.multi_modal_data = _encode_mm_data(
                prompt["multi_modal_data"])
            await self._run_encode(request, q)

        # TODO: support pd separation
        async for pd_response in self._run_pd(request, q):
            yield self._to_request_output(pd_response)

    async def _run_output_handler(self) -> None:
        """Background task to pull responses and dispatch to request queues.

        Binds a PULL socket on proxy_addr and receives multipart messages of
        the form (response_type, payload). Decodes payload into a
        GenerationResponse and enqueues it into the corresponding request queue
        keyed by request_id.
        """
        socket: Optional[zmq.asyncio.Socket] = None
        decoder = msgspec.msgpack.Decoder(GenerationResponse)
        failure_decoder = msgspec.msgpack.Decoder(FailureResponse)

        try:
            socket = self.ctx.socket(zmq.constants.PULL)
            socket.bind(self.proxy_addr)

            while True:
                resp_type, payload = await socket.recv_multipart()
                if (resp_type == ResponseType.GENERATION
                        or resp_type == ResponseType.ENCODE):
                    resp = decoder.decode(payload)
                    self.queues[resp.request_id].put_nowait(resp)
                elif resp_type == ResponseType.FAILURE:
                    resp = failure_decoder.decode(payload)
                    raise RuntimeError(f"Worker error: {resp.error_message}")
                else:
                    raise RuntimeError(
                        f"Unknown response type from worker: {resp_type}")
        except Exception as e:
            # TODO: maybe there is a more fine-grained way to handle errors.
            # For now, if there is any error, we terminate all requests.
            for q in self.queues.values():
                q.put_nowait(e)
        finally:
            if socket is not None:
                socket.close(linger=0)

    def encode(
        self,
        prompt: PromptType,
        pooling_params: PoolingParams,
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        priority: int = 0,
    ) -> AsyncGenerator[PoolingRequestOutput, None]:
        raise NotImplementedError

    async def abort(self, request_id: str) -> None:
        raise NotImplementedError

    async def get_model_config(self) -> ModelConfig:
        return self.model_config

    async def get_decoding_config(self) -> DecodingConfig:
        raise NotImplementedError

    async def get_input_preprocessor(self) -> InputPreprocessor:
        raise NotImplementedError

    async def get_tokenizer(
        self,
        lora_request: Optional[LoRARequest] = None,
    ) -> AnyTokenizer:
        if lora_request is not None:
            raise NotImplementedError("LoRA is not yet supported.")
        return self.tokenizer.get_lora_tokenizer(None)

    async def is_tracing_enabled(self) -> bool:
        return False

    async def do_log_stats(
        self,
        scheduler_outputs: Optional[SchedulerOutputs] = None,
        model_output: Optional[list[SamplerOutput]] = None,
    ) -> None:
        pass

    async def check_health(self) -> None:
        pass

    async def start_profile(self) -> None:
        raise NotImplementedError

    async def stop_profile(self) -> None:
        raise NotImplementedError

    async def reset_prefix_cache(self,
                                 device: Optional[Device] = None) -> None:
        raise NotImplementedError

    async def sleep(self, level: int = 1) -> None:
        raise NotImplementedError

    async def wake_up(self) -> None:
        raise NotImplementedError

    async def is_sleeping(self) -> bool:
        return False

    async def add_lora(self, lora_request: LoRARequest) -> None:
        raise NotImplementedError

    @property
    def errored(self) -> bool:
        return False

    def dead_error(self) -> Exception:
        return Exception("PDController has failed.")

    def is_running(self) -> bool:
        return True

    def is_stopped(self) -> bool:
        return False

    async def get_vllm_config(self) -> VllmConfig:
        raise NotImplementedError

    async def reset_mm_cache(self) -> None:
        raise NotImplementedError


# Helper functions
def _has_mm_data(prompt: PromptType) -> bool:
    if isinstance(prompt, dict):
        return "multi_modal_data" in prompt
    return False

def _encode_mm_data(mm_data: dict[str, any]) -> dict[str, any]:
    images = mm_data.get("image", [])
    if not isinstance(images, list):
        images = [images]
    encoded_images = []
    for img in images:
        if isinstance(img, np.ndarray):
            encoded_img = {
                "type": "ndarray",
                "data": img.tobytes(),
                "shape": img.shape,
                "dtype": str(img.dtype),
            }
            encoded_images.append(encoded_img)
    return {"image": encoded_images}


# FastAPI event handlers
@app.on_event("startup")
async def startup_event():
    # Initialize the proxy instance in the app state
    if not hasattr(app.state, "proxy"):
        # Use default values for testing, will be overridden by command line args
        app.state.proxy = Proxy(
            proxy_addr="/tmp/vllm_proxy.ipc",
            encode_addr_list=["/tmp/vllm_encode_0.ipc"],
            pd_addr_list=["/tmp/vllm_pd_0.ipc"],
            model_name="unknown-model",
        )


@app.on_event("shutdown")
async def shutdown_event():
    if hasattr(app.state, "proxy"):
        app.state.proxy.shutdown()


# FastAPI routes
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        request_data = await request.json()
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        is_streaming = request_data.get("stream", False)
        
        # Extract parameters from request
        prompt = request_data.get("messages", [])
        # For simplicity, we'll use the last message content as the prompt
        if prompt and isinstance(prompt, list):
            prompt_text = prompt[-1].get("content", "")
        else:
            prompt_text = ""
        
        # Create sampling params
        sampling_params = SamplingParams(
            temperature=request_data.get("temperature", 0.7),
            top_p=request_data.get("top_p", 1.0),
            max_tokens=request_data.get("max_tokens", 100),
            stop=request_data.get("stop", None),
            seed=request_data.get("seed", 77),
            repetition_penalty=request_data.get("repetition_penalty", 1.0),
            stop_token_ids=request_data.get("stop_token_ids", None),
        )
        
        if is_streaming:
            async def stream_generator():
                async for output in app.state.proxy.generate(
                    prompt=prompt_text,
                    sampling_params=sampling_params,
                    request_id=request_id,
                ):
                    prompt_tokens = len(output.prompt_token_ids)
                    completion_tokens = len(output.outputs[0].token_ids)
                    total_tokens = prompt_tokens + completion_tokens
                    # Format according to OpenAI's streaming format
                    chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": int(asyncio.get_event_loop().time()),
                        "model": app.state.proxy.model_config.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": output.outputs[0].text},
                                "finish_reason": output.outputs[0].finish_reason
                            }
                        ],
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": total_tokens
                        }
                    }
                    yield f"data: {msgspec.json.encode(chunk).decode()}\n\n"
                # End of stream
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(stream_generator(), media_type="text/event-stream")
        else:
            # For non-streaming, collect all outputs
            final_output = None
            async for output in app.state.proxy.generate(
                prompt=prompt_text,
                sampling_params=sampling_params,
                request_id=request_id,
            ):
                final_output = output
            
            if final_output:
                prompt_tokens = len(final_output.prompt_token_ids)
                completion_tokens = len(final_output.outputs[0].token_ids)
                total_tokens = prompt_tokens + completion_tokens
                response = {
                    "id": request_id,
                    "object": "chat.completion",
                    "created": int(asyncio.get_event_loop().time()),
                    "model": app.state.proxy.model_config.model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": final_output.outputs[0].text},
                            "finish_reason": final_output.outputs[0].finish_reason
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens
                    }
                }
                return JSONResponse(content=response)
            else:
                raise HTTPException(status_code=500, detail="No response from proxy")
    except Exception as e:
        logger.error("Error processing chat completion request: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/v1/completions")
async def completions(request: Request):
    try:
        request_data = await request.json()
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        is_streaming = request_data.get("stream", False)
        
        # Extract parameters from request
        prompt = request_data.get("prompt", "")
        
        # Create sampling params
        sampling_params = SamplingParams(
            temperature=request_data.get("temperature", 0.7),
            top_p=request_data.get("top_p", 1.0),
            max_tokens=request_data.get("max_tokens", 100),
            stop=request_data.get("stop", None),
            seed=request_data.get("seed", 77),
            repetition_penalty=request_data.get("repetition_penalty", 1.0),
            stop_token_ids=request_data.get("stop_token_ids", None),
        )
        
        if is_streaming:
            async def stream_generator():
                async for output in app.state.proxy.generate(
                    prompt=prompt,
                    sampling_params=sampling_params,
                    request_id=request_id,
                ):
                    prompt_tokens = len(output.prompt_token_ids)
                    completion_tokens = len(output.outputs[0].token_ids)
                    total_tokens = prompt_tokens + completion_tokens
                    # Format according to OpenAI's streaming format
                    chunk = {
                        "id": request_id,
                        "object": "text_completion.chunk",
                        "created": int(asyncio.get_event_loop().time()),
                        "model": app.state.proxy.model_config.model,
                        "choices": [
                            {
                                "index": 0,
                                "text": output.outputs[0].text,
                                "finish_reason": output.outputs[0].finish_reason
                            }
                        ],
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": total_tokens
                        }
                    }
                    yield f"data: {msgspec.json.encode(chunk).decode()}\n\n"
                # End of stream
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(stream_generator(), media_type="text/event-stream")
        else:
            # For non-streaming, collect all outputs
            final_output = None
            async for output in app.state.proxy.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=request_id,
            ):
                final_output = output
            
            if final_output:
                prompt_tokens = len(final_output.prompt_token_ids)
                completion_tokens = len(final_output.outputs[0].token_ids)
                total_tokens = prompt_tokens + completion_tokens
                response = {
                    "id": request_id,
                    "object": "text_completion",
                    "created": int(asyncio.get_event_loop().time()),
                    "model": app.state.proxy.model_config.model,
                    "choices": [
                        {
                            "index": 0,
                            "text": final_output.outputs[0].text,
                            "finish_reason": final_output.outputs[0].finish_reason
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens
                    }
                }
                return JSONResponse(content=response)
            else:
                raise HTTPException(status_code=500, detail="No response from proxy")
    except Exception as e:
        logger.error("Error processing completion request: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/health")
async def health_check():
    try:
        if hasattr(app.state, "proxy"):
            return JSONResponse(content={"status": "healthy"})
        else:
            return JSONResponse(content={"status": "unhealthy", "reason": "Proxy not initialized"}, status_code=503)
    except Exception as e:
        logger.error("Health check failed: %s", e)
        return JSONResponse(content={"status": "unhealthy", "reason": str(e)}, status_code=503)


# Main entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vLLM Disaggregated Proxy")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Proxy host")
    parser.add_argument("--port", type=int, default=8000, help="Proxy port")
    parser.add_argument("--proxy-addr", type=str, default="/tmp/vllm_proxy.ipc", help="Proxy IPC address")
    parser.add_argument("--encode-addrs", nargs='+', type=str, required=True, help="Comma-separated list of encode worker IPC addresses")
    parser.add_argument("--pd-addrs", nargs='+', type=str, required=True, help="Comma-separated list of PD worker IPC addresses")
    parser.add_argument("--model-name", type=str, required=True, help="Model name")
    
    args = parser.parse_args()
    
    # Initialize the proxy with the provided arguments
    app.state.proxy = Proxy(
        proxy_addr=args.proxy_addr,
        encode_addr_list=args.encode_addrs,
        pd_addr_list=args.pd_addrs,
        model_name=args.model_name,
    )
    
    logger.info(f"Starting vLLM Disaggregated Proxy on {args.host}:{args.port}")
    
    # Run the server with uvicorn
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        access_log=False,
        loop="asyncio",
    )
