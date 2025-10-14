# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import os
import uuid
from collections.abc import AsyncGenerator, Mapping
from typing import Optional, Union

import msgspec
import numpy as np
import zmq
import zmq.asyncio

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
        idx = (hash(request.request_id) & 0x7FFFFFFF) % len(
            self.to_encode_sockets)
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
        idx = (hash(request.request_id) & 0x7FFFFFFF) % len(self.to_pd_sockets)
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
            prompt_token_ids=None,
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
