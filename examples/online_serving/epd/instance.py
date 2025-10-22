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
    def __init__(
        self,
        url,
        server_type,
        health_check_interval: int = 5,
        health_threshold: int = 3,
    ):
        self.url = url
        self.server_type = server_type
        self.is_healthy = True
        self._fail_count = 0
        self._success_count = 0
        self.health_threshold = health_threshold
        self.health_check_interval = health_check_interval

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

    async def healthy_check(self, use_threshold: bool = True) -> bool:
        try:
            timeout_cfg = aiohttp.ClientTimeout(total=self.health_check_interval)
            async with self.session.get(
                f"{self.url}/health", timeout=timeout_cfg
            ) as response:
                response.raise_for_status()
            healthy = True
        except Exception as e:
            logger.error("Health check failed for %s: %s", self.url, e)
            healthy = False

        if use_threshold:
            if healthy:
                self._success_count = min(
                    self.health_threshold, self._success_count + 1
                )
                self._fail_count = 0
            else:
                self._fail_count = min(self.health_threshold, self._fail_count + 1)
                self._success_count = 0

            if self.is_healthy and self._fail_count >= self.health_threshold:
                self.is_healthy = False
                logger.warning("%s marked as unhealthy", self.url)
            elif not self.is_healthy and self._success_count >= self.health_threshold:
                self.is_healthy = True
                logger.info("%s marked as healthy", self.url)
            return self.is_healthy
        else:
            return healthy

    async def stop(self):
        await self.session.close()

    def _record_request_load(self):
        self.in_flight += 1

    def _release_request_load(self):
        with self._lock:
            self.in_flight = max(0, self.in_flight - 1)
