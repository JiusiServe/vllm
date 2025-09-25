# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# api_proxy.py
import copy
import logging
import os
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from enum import Enum, auto

import aiohttp

logger = logging.getLogger(__name__)

keepalive_timeout = int(os.getenv("CLIENT_HTTP_TIMEOUT_KEEP_ALIVE", 0))


class ServerType(Enum):
    E_INSTANCE = auto()
    PD_INSTANCE = auto()


class ServerState:
    def __init__(self, url, server_type):
        self.url = url
        self.server_type = server_type
        self.is_healthy = True

        # work load relative
        self._lock = threading.Lock()
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
        api: str,
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
                f"{self.url}{api}",
                json=request_data,
                headers=headers,
            )
            response.raise_for_status()
            return await response.json()

    async def forward_streaming_request(
        self,
        api: str,
        request_data: dict,
        headers: dict,
    ) -> AsyncIterator[str]:
        async with (
            self.request_context(),
            self.session.post(
                f"{self.url}{api}", json=request_data, headers=headers
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
        with self._lock:
            self.in_flight = max(0, self.in_flight - 1)
