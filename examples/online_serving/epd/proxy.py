# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# api_proxy.py
import argparse
import asyncio
import copy
import logging
import os
import random
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from enum import Enum, auto

import aiohttp
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

keepalive_timeout = int(os.getenv("CLIENT_HTTP_TIMEOUT_KEEP_ALIVE", 0))


class ServerType(Enum):
    E_INSTANCE = auto()
    PD_INSTANCE = auto()


EXCEPT_ERRORS = (
    aiohttp.ClientConnectorError,
    aiohttp.ServerDisconnectedError,
    aiohttp.ClientOSError,
)


class ServerState:
    def __init__(self, url, server_type):
        self.url = url
        self.server_type = server_type
        self.is_healthy = True

        # work load relative
        self.in_flight = 0

    async def init_session(self):
        self.session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(
                limit=0, keepalive_timeout=keepalive_timeout
            ),
            timeout=aiohttp.ClientTimeout(total=100000),
        )

    async def gather_workload(self):
        return self.in_flight

    @asynccontextmanager
    async def request_context(self):
        self._record_request_load()
        try:
            yield
        except Exception:
            logger.error("Failed to send request to %s.", self.url)
            raise
        finally:
            self._release_request_load()

    async def forward_non_streaming_request(
        self,
        request_data: dict,
        headers: dict,
    ) -> dict:
        request_data = copy.deepcopy(request_data)
        if self.server_type == ServerType.E_INSTANCE:
            request_data["max_tokens"] = 1
            request_data["stream"] = False
            request_data.pop("stream_options", None)
            if "max_completion_tokens" in request_data:
                request_data["max_completion_tokens"] = 1
        async with self.request_context():
            response = await self.session.post(
                f"{self.url}/v1/chat/completions",
                json=request_data,
                headers=headers,
            )
            response.raise_for_status()
            return await response.json()

    async def forward_streaming_request(
        self,
        request_data: dict,
        headers: dict,
    ) -> AsyncIterator[str]:
        async with (
            self.request_context(),
            self.session.post(
                f"{self.url}/v1/chat/completions", json=request_data, headers=headers
            ) as response,
        ):
            response.raise_for_status()
            async for chunk in response.content.iter_chunked(128):
                if chunk:
                    yield chunk.decode("utf-8", errors="ignore")

    async def healthy_check(self):
        try:
            async with self.session.get(f"{self.url}/health") as response:
                response.raise_for_status()
            self.is_healthy = True
        except Exception as e:
            logger.error("Health check failed for %s: %s", self.url, e)
            self.is_healthy = False
        return self.is_healthy

    async def stop(self):
        await self.session.close()

    def _record_request_load(self):
        self.in_flight += 1

    def _release_request_load(self):
        self.in_flight -= 1


class ServerScheduler:
    def __init__(self, instances, server_type, strategy: str = "random"):
        self.instances: list[ServerState] = [
            ServerState(url, server_type) for url in instances.split(",")
        ]
        self.server_type = server_type
        self.strategy_map = {
            "random": self._random_select,
            "round_robin": self._round_robin_select,
            "least_inflight": self._in_flight_select,
        }

        if strategy not in self.strategy_map:
            valid = ", ".join(self.strategy_map.keys())
            raise ValueError(
                f"Unknown strategy: {strategy}. Available strategies: {valid}"
            )
        self.select_instance = self.strategy_map[strategy]
        self.round_robin_indx = 0

    async def stream_retry_wrap(
        self, forward_func, max_retries: int = 3, delay: float = 0.1
    ):
        last_exc = None
        first_chunk_sent = False
        for attempt in range(max_retries):
            try:
                async for chunk in forward_func():
                    first_chunk_sent = True
                    yield chunk
                return
            except EXCEPT_ERRORS as e:
                if first_chunk_sent:
                    raise
                last_exc = e
                logger.warning(
                    "[%s] attempt %s / %s failed retrying... ",
                    self.server_type,
                    attempt + 1,
                    max_retries,
                )
                await asyncio.sleep(delay * (attempt + 1))

        raise RuntimeError(
            f"[{self.server_type}] all {max_retries} retries failed."
        ) from last_exc

    async def non_stream_retry_wrap(
        self, forward_func, max_retries: int = 3, delay: float = 0.1
    ):
        last_exc = None
        for attempt in range(max_retries):
            try:
                result = await forward_func()
                return result
            except EXCEPT_ERRORS as e:
                last_exc = e
                logger.warning(
                    "[%s] attempt %s / %s failed retrying... ",
                    self.server_type,
                    attempt + 1,
                    max_retries,
                )
                await asyncio.sleep(delay * (attempt + 1))
        raise RuntimeError(
            f"[{self.server_type}] all {max_retries} retries failed."
        ) from last_exc

    async def init_session(self):
        await asyncio.gather(*(ins.init_session() for ins in self.instances))

    async def gather_workload(self):
        result = await asyncio.gather(
            *(ins.gather_workload() for ins in self.instances)
        )
        return result

    async def healthy_check(self):
        tasks = [asyncio.create_task(ins.healthy_check()) for ins in self.instances]
        await asyncio.gather(*tasks, return_exceptions=True)

        unhealthy_url = [ins.url for ins in self.instances if not ins.is_healthy]
        return unhealthy_url

    async def stop_instances(self):
        await asyncio.gather(
            *(ins.stop() for ins in self.instances), return_exceptions=True
        )

    async def _random_select(self):
        healthy_instances = self._get_healthy_instances()
        return random.choice(healthy_instances)

    async def _round_robin_select(self):
        healthy_instances = self._get_healthy_instances()
        idx = self.round_robin_indx % len(healthy_instances)
        self.round_robin_indx += 1
        return healthy_instances[idx]

    async def _in_flight_select(self):
        healthy_instances = self._get_healthy_instances()
        return min(healthy_instances, key=lambda ins: ins.in_flight)

    def _get_healthy_instances(self):
        healthy_instances = [ins for ins in self.instances if ins.is_healthy]
        if not healthy_instances:
            raise LookupError(
                f"No healthy {self.server_type} instances available, "
                "please use '/health' to check"
            )
        return healthy_instances


@app.on_event("startup")
async def startup_event():
    await app.state.e_scheduler.init_session()
    await app.state.pd_scheduler.init_session()


@app.on_event("shutdown")
async def shutdown_event():
    if getattr(app.state, "e_scheduler", None):
        await app.state.e_scheduler.stop_instances()
    if getattr(app.state, "pd_scheduler", None):
        await app.state.pd_scheduler.stop_instances()


def has_mm_input(request_data: dict):
    if "messages" not in request_data:
        return False
    for message in request_data["messages"]:
        if not isinstance(message.get("content"), list):
            continue
        for content_item in message["content"]:
            if content_item.get("type") in ["image_url", "audio_url", "input_audio"]:
                return True
    return False


async def forward_streaming_request(
    request_data: dict,
    request_id: str,
) -> AsyncIterator[str]:
    headers = {"x-request-id": request_id}
    # Skip request to encoder instance if we don't have mm input
    if has_mm_input(request_data):

        async def non_stream_call():
            e_instance = await app.state.e_scheduler.select_instance()
            await e_instance.forward_non_streaming_request(request_data, headers)

        await app.state.e_scheduler.non_stream_retry_wrap(non_stream_call)

    async def stream_call():
        pd_instance = await app.state.pd_scheduler.select_instance()
        async for chunk in pd_instance.forward_streaming_request(request_data, headers):
            yield chunk

    async for chunk in app.state.pd_scheduler.stream_retry_wrap(stream_call):
        yield chunk


async def forward_non_streaming_request(
    request_data: dict,
    request_id: str,
) -> dict:
    headers = {"x-request-id": request_id}
    # Skip request to encoder instance if we don't have mm input
    if has_mm_input(request_data):

        async def e_non_stream_call():
            e_instance = await app.state.e_scheduler.select_instance()
            await e_instance.forward_non_streaming_request(request_data, headers)

        await app.state.e_scheduler.non_stream_retry_wrap(e_non_stream_call)

    async def pd_non_stream_call():
        pd_instance = await app.state.pd_scheduler.select_instance()
        return await pd_instance.forward_non_streaming_request(request_data, headers)

    return await app.state.pd_scheduler.non_stream_retry_wrap(pd_non_stream_call)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Handle chat completion requests."""
    try:
        request_data = await request.json()
        request_id = request.headers.get("x-request-id")
        if not request_id:
            request_id = str(uuid.uuid4())
        is_streaming = request_data.get("stream", False)
        if is_streaming:
            return StreamingResponse(
                forward_streaming_request(request_data, request_id),
                media_type="text/event-stream",
            )
        else:
            result = await forward_non_streaming_request(
                request_data,
                request_id,
            )
            return JSONResponse(content=result)
    except Exception as e:
        logger.error("Error processing request: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/workload_debug")
async def gather_workload():
    e_workload = await app.state.e_scheduler.gather_workload()
    pd_workload = await app.state.pd_scheduler.gather_workload()
    workload_status = {
        "encoder workload": e_workload,
        "prefill_decode workload": pd_workload,
    }
    return workload_status


@app.get("/v1/models")
async def list_models():
    try:
        async with app.state.pd_scheduler.instances[0].session.get(
            f"{app.state.pd_scheduler.instances[0].url}/v1/models"
        ) as response:
            response.raise_for_status()
            return await response.json()
    except Exception as e:
        logger.error("Error fetching models: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        unhealthy_encoder_url = await app.state.e_scheduler.healthy_check()
        unhealthy_pd_url = await app.state.pd_scheduler.healthy_check()

        health_status = {
            "proxy": "healthy",
            "unhealthy encode_servers": unhealthy_encoder_url,
            "unhealthy prefill_decode_servers": unhealthy_pd_url,
        }

        if unhealthy_encoder_url or unhealthy_pd_url:
            return JSONResponse(content=health_status, status_code=503)

        return health_status

    except Exception as e:
        logger.error("Health check error: %s", e)
        return JSONResponse(
            content={"proxy": "unhealthy", "error": str(e)}, status_code=503
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="API Proxy for distributed vLLM servers"
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Proxy host")
    parser.add_argument("--port", type=int, default=8000, help="Proxy port")

    parser.add_argument(
        "--encode-servers-urls",
        type=str,
        required=True,
        help="URLs of the encode server in comma separated format"
        '(e.g., "http://localhost:8001,http://localhost:8002")',
    )

    parser.add_argument(
        "--prefill-decode-servers-urls",
        type=str,
        required=True,
        help="URLs of the prefill/decode servers in comma separated format"
        '(e.g., "http://localhost:8003,http://localhost:8004")',
    )

    parser.add_argument(
        "--scheduling-proxy",
        type=str,
        default="random",
        help="instances scheduling proxy: "
        "choose from {random, round_robin, least_inflight}. Default: random",
    )

    args = parser.parse_args()
    app.state.e_scheduler = ServerScheduler(
        args.encode_servers_urls, ServerType.E_INSTANCE, args.scheduling_proxy
    )
    app.state.pd_scheduler = ServerScheduler(
        args.prefill_decode_servers_urls, ServerType.PD_INSTANCE, args.scheduling_proxy
    )

    logger.info("Starting API proxy on %s:%s with 1 worker", args.host, args.port)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        access_log=False,
        loop="uvloop",
    )
