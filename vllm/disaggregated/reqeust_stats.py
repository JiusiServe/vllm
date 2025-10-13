# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# request_stats.py
from dataclasses import dataclass


@dataclass
class RequestStats:
    # Inflight request count
    in_flight_requests: set[str]
    # Other stats can be added for more complex scheduling


class RequestStatsMonitor:
    """
    Monitors and records request statistics for all instances.
    """

    def __init__(self, instances: list[int]):
        # Key: instance id
        self.request_stats: dict[int, RequestStats] = {
            iid: RequestStats(in_flight_requests=set())
            for iid in instances
        }

    def on_new_request(self, instance_id: int, request_id: str):
        self.request_stats[instance_id].in_flight_requests.add(request_id)

    def on_request_completed(self, instance_id: int, request_id: str):
        self.request_stats[instance_id].in_flight_requests.discard(request_id)

    def get_request_stats(self):
        return self.request_stats
