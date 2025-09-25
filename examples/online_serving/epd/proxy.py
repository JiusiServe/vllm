# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# api_proxy.py
import argparse
import asyncio
import copy
import logging
import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from enum import Enum, auto
from typing import List

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


class ServerState:
    def __init__(self, url, server_type, id):
        self.url = url
        self.server_type = server_type
        self.session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(
                limit=0, keepalive_timeout=keepalive_timeout
            ),
            timeout=aiohttp.ClientTimeout(total=100000),
        )

        # work load relative
        self.scores = 0

    @asynccontextmanager
    async def request_context(self, log_prefix: str):
        self._record_request_load()
        try:
            yield
        except Exception:
            logger.error("Error in %s", log_prefix)
            raise
        finally:
            self._release_request_load()

    async def forward_non_streaming_request(
        self,
        request_data: dict,
        headers: dict,
    ) -> dict:
        self._record_request_load()
        request_data = copy.deepcopy(request_data)
        if self.server_type == ServerType.E_INSTANCE:
            request_data["max_tokens"] = 1
            if "max_completion_tokens" in request_data:
                request_data["max_completion_tokens"] = 1
        async with self.request_context("non-stream"):
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
        async with self.request_context("stream"):
            async with self.session.post(
                f"{self.url}/v1/chat/completions", json=request_data, headers=headers
            ) as response:
                response.raise_for_status()
                async for chunk in response.content.iter_chunked(128):
                    if chunk:
                        yield chunk.decode("utf-8", errors="ignore")

    def _record_request_load(self):
        self.scores += 1

    def _release_request_load(self):
        self.scores -= 1

    def stop(self):
        self.session.close()


class ServerScheduler:
    def __init__(self, instances, server_type):
        self.instances: List[ServerState] = [
            ServerState(url, server_type, i)
            for i, url in enumerate(instances.split(","))
        ]
        self.server_type = server_type

    def select_instance(self):
        server = min(self.instances, key=lambda ins: ins.scores)
        return server

    def stop_instances(self):
        for ins in self.instances:
            ins.close()


@app.on_event("shutdown")
async def shutdown_event():
    if app.state.e_scheduler:
        await app.state.e_scheduler.close()
    if app.state.pd_scheduler:
        await app.state.pd_scheduler.close()


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
        e_instance = app.state.e_scheduler.select_instance()
        await e_instance.forward_non_streaming_request(request_data, headers)

    pd_instance = app.state.pd_scheduler.select_instance()
    async for chunk in pd_instance.forward_streaming_request(request_data, headers):
        yield chunk


async def forward_non_streaming_request(
    request_data: dict,
    request_id: str,
) -> dict:
    headers = {"x-request-id": request_id}
    # Skip request to encoder instance if we don't have mm input
    if has_mm_input(request_data):
        e_instance = app.state.e_scheduler.select_instance()
        await e_instance.forward_non_streaming_request(request_data, headers)

    pd_instance = app.state.pd_scheduler.select_instance()
    return await pd_instance.forward_non_streaming_request(request_data, headers)


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


@app.get("/v1/models")
async def list_models():
    try:
        async with decode_session.get(f"{app.state.pd_urls[0]}/v1/models") as response:
            response.raise_for_status()
            return await response.json()
    except Exception as e:
        logger.error("Error fetching models: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:

        async def check_encode():
            try:
                for e_url in app.state.e_urls:
                    async with encode_session.get(f"{e_url}/health") as response:
                        response.raise_for_status()
                return True
            except Exception:
                return False

        async def check_decode():
            try:
                for pd_url in app.state.pd_urls:
                    async with encode_session.get(f"{pd_url}/health") as response:
                        response.raise_for_status()
                return True
            except Exception:
                return False

        encode_healthy, decode_healthy = await asyncio.gather(
            check_encode(), check_decode(), return_exceptions=True
        )

        health_status = {
            "proxy": "healthy",
            "encode_servers": "healthy" if encode_healthy is True else "unhealthy",
            "prefill_decode_servers": "healthy"
            if decode_healthy is True
            else "unhealthy",
        }

        if not (encode_healthy is True and decode_healthy is True):
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

    args = parser.parse_args()
    app.state.e_scheduler = ServerScheduler(args.encode_servers_urls)
    app.state.pd_scheduler = ServerScheduler(args.prefill_decode_servers_urls)

    logger.info("Starting API proxy on %s:%s with 1 worker", args.host, args.port)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        access_log=False,
        loop="uvloop",
    )
