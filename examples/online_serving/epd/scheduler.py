# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# api_proxy.py
import asyncio
import logging
import random

import aiohttp
from instance import ServerState

logger = logging.getLogger(__name__)


EXCEPT_ERRORS = (
    aiohttp.ClientConnectorError,
    aiohttp.ServerDisconnectedError,
    aiohttp.ClientOSError,
)


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
        tried_instances = set()
        for attempt in range(max_retries):
            try:
                async for chunk in forward_func(tried_instances):
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
        tried_instances = set()
        for attempt in range(max_retries):
            try:
                result = await forward_func(tried_instances)
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

    async def _random_select(self, exclude_instances: set):
        healthy_instances = self._get_healthy_instances(exclude_instances)
        return random.choice(healthy_instances)

    async def _round_robin_select(self, exclude_instances: set):
        healthy_instances = self._get_healthy_instances(exclude_instances)
        idx = self.round_robin_indx % len(healthy_instances)
        self.round_robin_indx = idx + 1
        return healthy_instances[idx]

    async def _in_flight_select(self, exclude_instances: set):
        healthy_instances = self._get_healthy_instances(exclude_instances)
        return min(healthy_instances, key=lambda ins: ins.in_flight)

    def _get_healthy_instances(self, exclude_instances: set):
        healthy_instances = [
            ins
            for ins in self.instances
            if ins.is_healthy and ins not in exclude_instances
        ]
        if not healthy_instances:
            raise LookupError(
                f"No healthy {self.server_type} instances available, "
                "please use '/health' to check"
            )
        return healthy_instances
