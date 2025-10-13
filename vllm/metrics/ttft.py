# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import os
from typing import Optional

from prometheus_client import Histogram

from vllm.v1.metrics.loggers import build_1_2_5_buckets

TTFT_ENABLED = os.environ.get("TTFT_ENABLED", "0")

# ms 直方图桶
_MS_BUCKETS = build_1_2_5_buckets(5000)

# Per-stage histograms
TTFT_ENC_QUEUE = Histogram(
    "vllm_ttft_enc_queue_ms",
    "Encoder queue latency (ms)",
    ["model", "ec_role", "mm"],
    buckets=_MS_BUCKETS,
)
TTFT_ENC_COMPUTE = Histogram(
    "vllm_ttft_enc_compute_ms",
    "Encoder compute latency (ms)",
    ["model", "ec_role", "mm"],
    buckets=_MS_BUCKETS,
)
TTFT_E_CACHE_TRANSFER = Histogram(
    "vllm_E_cache_transfer_to_PD_ms",
    "Encoder cache transfer latency (ms)",
    ["model", "ec_role", "mm"],
    buckets=_MS_BUCKETS,
)
TTFT_PREFILL_QUEUE = Histogram(
    "vllm_ttft_prefill_queue_ms",
    "Prefill queue latency (ms)",
    ["model", "ec_role", "mm"],
    buckets=_MS_BUCKETS,
)
TTFT_PREFILL_COMPUTE = Histogram(
    "vllm_ttft_prefill_compute_ms",
    "Prefill compute latency to first token (ms)",
    ["model", "ec_role", "mm"],
    buckets=_MS_BUCKETS,
)
TTFT_PROXY_TRANSFER_TO_ENCODE = Histogram(
    "vllm_proxy_transfer_to_encode_ms",
    "proxy transfer to encode latency (ms)",
    ["model", "ec_role", "mm"],
    buckets=_MS_BUCKETS,
)
TTFT_PROXY_TRANSFER_TO_PD = Histogram(
    "vllm_proxy_transfer_to_pd_ms",
    "proxy transfer to pd latency (ms)",
    ["model", "ec_role", "mm"],
    buckets=_MS_BUCKETS,
)


def _labels(model_name: str, ec_role: Optional[str], is_mm: bool):
    return {
        "model": model_name,
        "ec_role": ec_role or "na",
        "mm": "yes" if is_mm else "no"
    }


def observe_enc_queue(ms: float, model_name: str, ec_role: Optional[str],
                      is_mm: bool):
    if TTFT_ENABLED and ms is not None:
        TTFT_ENC_QUEUE.labels(
            **_labels(model_name, ec_role, is_mm)).observe(ms)


def observe_enc_compute(ms: float, model_name: str, ec_role: Optional[str],
                        is_mm: bool):
    if TTFT_ENABLED and ms is not None:
        TTFT_ENC_COMPUTE.labels(
            **_labels(model_name, ec_role, is_mm)).observe(ms)


def observe_emb_cache_transfer(ms: float, model_name: str,
                               ec_role: Optional[str], is_mm: bool):
    if TTFT_ENABLED and ms is not None:
        TTFT_E_CACHE_TRANSFER.labels(
            **_labels(model_name, ec_role, is_mm)).observe(ms)


def observe_prefill_queue(ms: float, model_name: str, ec_role: Optional[str],
                          is_mm: bool):
    if TTFT_ENABLED and ms is not None:
        TTFT_PREFILL_QUEUE.labels(
            **_labels(model_name, ec_role, is_mm)).observe(ms)


def observe_prefill_compute(ms: float, model_name: str, ec_role: Optional[str],
                            is_mm: bool):
    if TTFT_ENABLED and ms is not None:
        TTFT_PREFILL_COMPUTE.labels(
            **_labels(model_name, ec_role, is_mm)).observe(ms)


def observe_proxy_transfer_to_encode(ms: float, model_name: str,
                                     ec_role: Optional[str], is_mm: bool):
    if TTFT_ENABLED and ms is not None:
        TTFT_PROXY_TRANSFER_TO_ENCODE.labels(
            **_labels(model_name, ec_role, is_mm)).observe(ms)


def observe_proxy_transfer_to_pd(ms: float, model_name: str,
                                 ec_role: Optional[str], is_mm: bool):
    if TTFT_ENABLED and ms is not None:
        TTFT_PROXY_TRANSFER_TO_PD.labels(
            **_labels(model_name, ec_role, is_mm)).observe(ms)
