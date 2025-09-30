# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import os

import msgspec
import numpy as np
import zmq
import zmq.asyncio

from vllm.disaggregated.protocol import (GenerationRequest, GenerationResponse,
                                         RequestType, ResponseType)
from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger

logger = init_logger(__name__)


class DisaggWorker:

    def __init__(
        self,
        engine: EngineClient,
        address: str,
        proxy_addr: str,
    ):
        self.engine = engine

        self.worker_addr = f"ipc://{address}"
        self.proxy_addr = f"ipc://{proxy_addr}"
        self.ctx = zmq.asyncio.Context()
        self.from_proxy = self.ctx.socket(zmq.constants.PULL)
        self.from_proxy.bind(self.worker_addr)
        self.to_proxy = self.ctx.socket(zmq.constants.PUSH)
        self.to_proxy.connect(self.proxy_addr)

        self.decoder_generate = msgspec.msgpack.Decoder(GenerationRequest)
        self.decoder_abort = msgspec.msgpack.Decoder(GenerationRequest)
        self.encoder = msgspec.msgpack.Encoder()

        self.running_requests: set[asyncio.Task] = set()

    def shutdown(self):
        self.ctx.destroy()

        for running_request in self.running_requests:
            running_request.cancel()

        socket_path = self.worker_addr.replace("ipc://", "")
        if os.path.exists(socket_path):
            os.remove(socket_path)

    async def run_busy_loop(self):
        logger.info("DisaggWorker is ready To handle requests.")

        poller = zmq.asyncio.Poller()
        poller.register(self.from_proxy, zmq.POLLIN)

        while True:
            req_type, req_data = await self.from_proxy.recv_multipart()
            await self._handle_request(req_type, req_data)

    async def _handle_request(self, req_type: bytes, req_data: bytes):
        if req_type == RequestType.ENCODE:
            req = self.decoder_generate.decode(req_data)
            req.sampling_params.max_tokens = 1
            await self._encode_handler(req)
        elif req_type == RequestType.GENERATION:
            req = self.decoder_generate.decode(req_data)
            await self._generation_handler(req)
        elif req_type == RequestType.ABORT:
            req = self.decoder_abort.decode(req_data)
            await self._abort_handler(req)
        else:
            raise Exception(f"Unknown Request Type: {req_type}.")

    async def _encode_handler(self, req: GenerationRequest):
        task = asyncio.create_task(
            self._generate(req, lambda b: (ResponseType.ENCODE, b)))
        self.running_requests.add(task)
        task.add_done_callback(self.running_requests.discard)

    async def _generation_handler(self, req: GenerationRequest):
        task = asyncio.create_task(
            self._generate(req, lambda b: (ResponseType.GENERATION, b)))
        self.running_requests.add(task)
        task.add_done_callback(self.running_requests.discard)

    async def _abort_handler(self, req: GenerationRequest):
        self.engine.abort(request_id=req.request_id)

    async def _generate(
        self,
        req: GenerationRequest,
        make_msg_func,
    ):
        request_id = req.request_id

        generator = self.engine.generate(
            prompt={
                "prompt": req.prompt,
                "multi_modal_data": _decode_mm_data(req.multi_modal_data),
            },
            sampling_params=req.sampling_params,
            request_id=request_id,
        )

        async for request_output in generator:
            response = GenerationResponse.from_request_output(request_output)

            response_bytes = self.encoder.encode(response)
            msg = make_msg_func(response_bytes)
            await self.to_proxy.send_multipart(msg, copy=False)


def _decode_mm_data(mm_data: dict[str, any]) -> dict[str, any]:
    images = mm_data.get("image", [])
    if not isinstance(images, list):
        images = [images]
    decoded_images = []
    for img in images:
        if img["type"] == "ndarray":
            decoded_img = np.frombuffer(bytes(
                img["data"]), dtype=img["dtype"]).reshape(img["shape"])
            decoded_images.append(decoded_img)
    return {"image": decoded_images}
