# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import os
from typing import Optional

from prometheus_client import Histogram

EPD_TIMECOUNT_ENABLED = os.environ.get("EPD_TIMECOUNT_ENABLED", "0")

# seconds 直方图桶
_seconds_BUCKETS = [
    0.001,
    0.002,
    0.005,
    0.01,
    0.02,
    0.05,
    0.1,
    0.2,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    20.0,
    50.0,
    100.0,
    200.0,
    500.0,
]

# Per-stage histograseconds
ENC_COMPUTE = Histogram(
    "vllm_execute_mm_encoder_seconds",
    "Encoder compute latency (seconds)",
    ["model", "ec_role", "mm"],
    buckets=_seconds_BUCKETS,
)
LOAD_E_CACHE = Histogram(
    "vllm_load_Encoder_cache_seconds",
    "Encoder cache transfer latency (seconds)",
    ["model", "ec_role", "mm"],
    buckets=_seconds_BUCKETS,
)
PREFILL_FORWARD = Histogram(
    "vllm_prefill_forward_seconds",
    "Prefill forward latency to first token (seconds)",
    ["model", "ec_role", "mm"],
    buckets=_seconds_BUCKETS,
)
PROXY_TRANSFER_TO_ENCODE = Histogram(
    "vllm_proxy_transfer_to_encode_seconds",
    "proxy transfer to encode latency (seconds)",
    ["model", "ec_role", "mm"],
    buckets=_seconds_BUCKETS,
)
PROXY_TRANSFER_TO_PD = Histogram(
    "vllm_proxy_transfer_to_pd_seconds",
    "proxy transfer to pd latency (seconds)",
    ["model", "ec_role", "mm"],
    buckets=_seconds_BUCKETS,
)


def _labels(model_name: str, ec_role: Optional[str], is_mm: bool):
    return {
        "model": model_name,
        "ec_role": ec_role or "na",
        "mm": "yes" if is_mm else "no",
    }


def observe_enc_compute(seconds: float, model_name: str,
                        ec_role: Optional[str], is_mm: bool):
    if EPD_TIMECOUNT_ENABLED and seconds is not None:
        ENC_COMPUTE.labels(
            **_labels(model_name, ec_role, is_mm)).observe(seconds)


def observe_load_e_cache(seconds: float, model_name: str,
                         ec_role: Optional[str], is_mm: bool):
    if EPD_TIMECOUNT_ENABLED and seconds is not None:
        LOAD_E_CACHE.labels(
            **_labels(model_name, ec_role, is_mm)).observe(seconds)


def observe_prefill_compute(seconds: float, model_name: str,
                            ec_role: Optional[str], is_mm: bool):
    if EPD_TIMECOUNT_ENABLED and seconds is not None:
        PREFILL_FORWARD.labels(
            **_labels(model_name, ec_role, is_mm)).observe(seconds)


def observe_proxy_transfer_to_encode(seconds: float, model_name: str,
                                     ec_role: Optional[str], is_mm: bool):
    if EPD_TIMECOUNT_ENABLED and seconds is not None:
        PROXY_TRANSFER_TO_ENCODE.labels(
            **_labels(model_name, ec_role, is_mm)).observe(seconds)


def observe_proxy_transfer_to_pd(seconds: float, model_name: str,
                                 ec_role: Optional[str], is_mm: bool):
    if EPD_TIMECOUNT_ENABLED and seconds is not None:
        PROXY_TRANSFER_TO_PD.labels(
            **_labels(model_name, ec_role, is_mm)).observe(seconds)
