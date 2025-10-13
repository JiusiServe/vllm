# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# service_discovery.py

import asyncio
import time
from abc import ABC, abstractmethod

from vllm.disaggregated.protocol import ServerType
from vllm.logger import init_logger

logger = init_logger(__name__)


class ServiceDiscovery(ABC):

    @abstractmethod
    def get_health_endpoints(self) -> list[int]:
        """
        Retrieve a list of available instances for the given service name.

        Args:
            service_name (str): The name of the service to discover.
        Returns:
            list[int]: A list of available instance IDs.
        """
        pass

    @abstractmethod
    def get_unhealth_endpoints(self) -> list[int]:
        """
        Retrieve a list of available instances for the given service name.

        Args:
            service_name (str): The name of the service to discover.

        Returns:
            list[int]: A list of available instance IDs.
        """
        pass


class HealthCheckServiceDiscovery(ServiceDiscovery):

    def __init__(self, server_type: ServerType, instances: list[int],
                 enable_health_monitor: bool, health_check_interval: float,
                 health_threshold: int, health_check_func):
        self.server_type = server_type
        self._instances = {iid: True for iid in instances}
        self._cached_health_instances = [iid for iid in instances]
        self._cached_unhealth_instances = []
        self.enable_health_monitor = enable_health_monitor
        self._health_check_interval = health_check_interval
        self._health_threshold = health_threshold
        self._succ_count = {iid: 0 for iid in instances}
        self._fail_count = {iid: 0 for iid in instances}
        self._health_check_func = health_check_func
        self._health_monitor_handler = None

    def should_launch_health_monitor(self) -> bool:
        return (self.enable_health_monitor
                and self._health_monitor_handler is None)

    def launch_health_monitor(self):
        self._health_monitor_handler = asyncio.create_task(
            self.run_health_check_loop())
        logger.info("Health monitor for %s launched.", self.server_type)

    def get_health_endpoints(self) -> list[int]:
        return self._cached_health_instances

    def get_unhealth_endpoints(self) -> list[int]:
        return self._cached_unhealth_instances

    async def run_health_check_loop(self):
        while True:
            start_time = time.monotonic()
            tasks = [
                asyncio.create_task(
                    asyncio.wait_for(self._health_check_func(
                        self.server_type, iid),
                                     timeout=self._health_check_interval))
                for iid in self._instances
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for iid, result in zip(self._instances.keys(), results):
                if isinstance(result, bool) and result:
                    self._update_health_counts(iid, True)
                else:
                    self._update_health_counts(iid, False)
                    logger.warning(
                        "Health check for %s %s failed, reason is (%s).",
                        self.server_type, iid, "timeout" if isinstance(
                            result, asyncio.TimeoutError) else result)

            self._update_health_status()

            elapsed = time.monotonic() - start_time
            sleep_time = max(0, self._health_check_interval - elapsed)
            await asyncio.sleep(sleep_time)

    def _update_health_counts(self, iid: int, is_succ: bool):
        if is_succ:
            self._succ_count[iid] = min(self._health_threshold,
                                        self._succ_count.get(iid, 0) + 1)
            self._fail_count[iid] = 0
        else:
            self._fail_count[iid] = min(self._health_threshold,
                                        self._fail_count.get(iid, 0) + 1)
            self._succ_count[iid] = 0

    def _update_health_status(self):
        for iid in self._instances:
            if self._instances[iid] and self._fail_count.get(
                    iid, 0) >= self._health_threshold:
                self._instances[iid] = False
                logger.info("Instance %s %s marked as unhealthy.",
                            self.server_type, iid)
            elif self._instances[iid] is False and self._succ_count.get(
                    iid, 0) >= self._health_threshold:
                self._instances[iid] = True
                logger.info("Instance %s %s marked as healthy.",
                            self.server_type, iid)

            tmp_health_instances = [
                iid for iid, healthy in self._instances.items() if healthy
            ]
            tmp_unhealthy_instances = [
                iid for iid, healthy in self._instances.items() if not healthy
            ]

            self._cached_health_instances = tmp_health_instances
            self._cached_unhealth_instances = tmp_unhealthy_instances
