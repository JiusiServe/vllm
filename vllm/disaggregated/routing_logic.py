# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# routing_logic.py

import random

from vllm.disaggregated.reqeust_stats import RequestStats


class RoutingInterface:

    def route_request(self, endpoints: list[int], request_stats: dict) -> int:
        """
        Route the request to a specific instance based on the request stats.
        It can also be based on engine stats in the future.

        Args:
            endpoints (list[int]): The list of instance IDs.
            request_stats (dict): The incoming request stats.

        Returns:
            int: The ID of the selected instance.
        """

        # Implement your routing logic here
        raise NotImplementedError("Subclasses should implement this method.")


class RandomRouter(RoutingInterface):

    def route_request(self, endpoints: list[int], request_stats: dict) -> int:
        if not endpoints:
            raise RuntimeError("No healthy endpoints available for routing.")
        return random.choice(endpoints)


class RoundRobinRouter(RoutingInterface):

    def __init__(self):
        self.current_index = 0

    def route_request(self, endpoints: list[int], request_stats: dict) -> int:
        if not endpoints:
            raise RuntimeError("No healthy endpoints available for routing.")
        selected_index = self.current_index % len(endpoints)
        self.current_index = selected_index + 1
        return endpoints[selected_index]


class LatestInFlightRouter(RoutingInterface):

    def route_request(self, endpoints: list[int], request_stats: dict) -> int:
        if not endpoints:
            raise RuntimeError("No healthy endpoints available for routing.")
        selected_instance = endpoints[0]
        min_in_flight = float('inf')

        for endpoint in endpoints:
            stats: RequestStats = request_stats.get(endpoint)
            if stats is None:
                selected_instance = endpoint
                min_in_flight = 0
            elif stats and len(stats.in_flight_requests) < min_in_flight:
                min_in_flight = len(stats.in_flight_requests)
                selected_instance = endpoint

        return selected_instance
