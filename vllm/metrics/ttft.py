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
    ["model", "engine_id", "mm"],
    buckets=_MS_BUCKETS,
)
TTFT_ENC_COMPUTE = Histogram(
    "vllm_ttft_enc_compute_ms",
    "Encoder compute latency (ms)",
    ["model", "engine_id", "mm"],
    buckets=_MS_BUCKETS,
)
TTFT_EMB_CACHE_TRANSFER = Histogram(
    "vllm_ttft_emb_cache_transfer_ms",
    "Embedding/encoder cache transfer latency (ms)",
    ["model", "engine_id", "mm"],
    buckets=_MS_BUCKETS,
)
TTFT_PREFILL_QUEUE = Histogram(
    "vllm_ttft_prefill_queue_ms",
    "Prefill queue latency (ms)",
    ["model", "engine_id", "mm"],
    buckets=_MS_BUCKETS,
)
TTFT_PREFILL_COMPUTE = Histogram(
    "vllm_ttft_prefill_compute_ms",
    "Prefill compute latency to first token (ms)",
    ["model", "engine_id", "mm"],
    buckets=_MS_BUCKETS,
)
TTFT_TOTAL = Histogram(
    "vllm_ttft_total_ms",
    "TTFT (sum of stages) (ms)",
    ["model", "engine_id", "mm"],
    buckets=_MS_BUCKETS,
)


def _labels(model_name: str, engine_id: Optional[str], is_mm: bool):
    return {
        "model": model_name,
        "engine_id": engine_id or "na",
        "mm": "yes" if is_mm else "no"
    }


def observe_enc_queue(ms: float, model_name: str, engine_id: Optional[str],
                      is_mm: bool):
    if TTFT_ENABLED and ms is not None:
        TTFT_ENC_QUEUE.labels(
            **_labels(model_name, engine_id, is_mm)).observe(ms)


def observe_enc_compute(ms: float, model_name: str, engine_id: Optional[str],
                        is_mm: bool):
    if TTFT_ENABLED and ms is not None:
        TTFT_ENC_COMPUTE.labels(
            **_labels(model_name, engine_id, is_mm)).observe(ms)


def observe_emb_cache_transfer(ms: float, model_name: str,
                               engine_id: Optional[str], is_mm: bool):
    if TTFT_ENABLED and ms is not None:
        TTFT_EMB_CACHE_TRANSFER.labels(
            **_labels(model_name, engine_id, is_mm)).observe(ms)


def observe_prefill_queue(ms: float, model_name: str, engine_id: Optional[str],
                          is_mm: bool):
    if TTFT_ENABLED and ms is not None:
        TTFT_PREFILL_QUEUE.labels(
            **_labels(model_name, engine_id, is_mm)).observe(ms)


def observe_prefill_compute(ms: float, model_name: str,
                            engine_id: Optional[str], is_mm: bool):
    if TTFT_ENABLED and ms is not None:
        TTFT_PREFILL_COMPUTE.labels(
            **_labels(model_name, engine_id, is_mm)).observe(ms)


def observe_total(ttft_ms: float, model_name: str, engine_id: Optional[str],
                  is_mm: bool):
    if TTFT_ENABLED and ttft_ms is not None:
        TTFT_TOTAL.labels(
            **_labels(model_name, engine_id, is_mm)).observe(ttft_ms)
