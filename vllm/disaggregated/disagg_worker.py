# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import copy
import os
import re
import time
from typing import Optional, Union

import msgspec
import numpy as np
import zmq
import zmq.asyncio
from prometheus_client import generate_latest

from vllm.disaggregated.protocol import (FailureResponse, GenerationRequest,
                                         GenerationResponse, HeartbeatRequest,
                                         HeartbeatResponse, RequestType,
                                         ResponseType)
from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger
from vllm.v1.metrics.prometheus import get_prometheus_registry

TIMECOUNT_ENABLED = os.getenv("TIMECOUNT_ENABLED",
                              "0") in ("1", "true", "True")
VLLM_LOG_STATS_INTERVAL = float(os.getenv("VLLM_LOG_STATS_INTERVAL", "10"))

logger = init_logger(__name__)


class DisaggWorker:

    def __init__(
        self,
        engine: EngineClient,
        address: str,
        proxy_addr: str,
        ec_role: str,
    ):
        self.engine = engine

        self.worker_addr = f"ipc://{address}"
        self.proxy_addr = f"ipc://{proxy_addr}"
        self.ctx = zmq.asyncio.Context()
        self.from_proxy = self.ctx.socket(zmq.constants.PULL)
        self.from_proxy.bind(self.worker_addr)
        self.to_proxy = self.ctx.socket(zmq.constants.PUSH)
        self.to_proxy.connect(self.proxy_addr)

        self.decoder_generate = msgspec.msgpack.Decoder(GenerationRequest)
        self.decoder_heartbeat = msgspec.msgpack.Decoder(HeartbeatRequest)
        self.decoder_abort = msgspec.msgpack.Decoder(GenerationRequest)
        self.encoder = msgspec.msgpack.Encoder()
        self.ec_role = ec_role
        self.running_requests: set[asyncio.Task] = set()
        # store the histograms log reported last time
        self.past_histograms_log: dict[str, dict] = dict()

    def shutdown(self):
        self.ctx.destroy()

        for running_request in self.running_requests:
            running_request.cancel()

        socket_path = self.worker_addr.replace("ipc://", "")
        if os.path.exists(socket_path):
            os.remove(socket_path)

    async def _do_log_stats(self) -> None:
        while True:
            await self.engine.do_log_stats()
            await asyncio.sleep(VLLM_LOG_STATS_INTERVAL)

    async def _force_log(self,
                         filter_keys: Optional[list[str]] = None) -> None:
        while True:
            try:
                metrics_text = generate_latest(
                    registry=get_prometheus_registry()).decode("utf-8")
                parse_result = parse_histograms(metrics_text,
                                                filter_keys=filter_keys)
                parse_result_diff: dict[str, dict[str, Union[str, float]]] = {}

                if self.past_histograms_log:
                    # compute diff
                    for k, cur in parse_result.items():
                        past = self.past_histograms_log.get(k, {})
                        cur_count = cur.get("count", 0)
                        past_count = past.get("count", 0)
                        # diff
                        diff_count = cur_count - past_count
                        # compute mean diff (weighted average)
                        if diff_count > 0:
                            cur_sum = cur.get("mean", 0) * cur_count
                            past_sum = past.get("mean", 0) * past_count
                            diff_mean = round(
                                (cur_sum - past_sum) / diff_count, 2)
                        else:
                            diff_mean = float('nan')
                        parse_result_diff[k] = {
                            "count": diff_count,
                            "mean_ms": f"{diff_mean} ms"
                        }
                else:
                    # first time log, just log current values
                    for k, v in parse_result.items():
                        mean_ms = round(v.get("mean", 0), 2)
                        parse_result_diff[k] = {
                            "count": v.get("count", 0),
                            "mean_ms": f"{mean_ms} ms"
                        }
                # refresh past log
                self.past_histograms_log = copy.deepcopy(parse_result)
                logger.info("DisaggWorker metrics: %s", parse_result_diff)
            except Exception as e:
                logger.error("Error in force_log: %s", e)
            await asyncio.sleep(VLLM_LOG_STATS_INTERVAL)

    async def run_busy_loop(self):
        logger.info("DisaggWorker is ready To handle requests.")

        poller = zmq.asyncio.Poller()
        poller.register(self.from_proxy, zmq.POLLIN)
        if TIMECOUNT_ENABLED:
            filter_keys = [
                "e2e_request_latency_seconds", "request_queue_time_seconds",
                "encoder_consume_time_seconds",
                "time_per_output_token_seconds", "request_prefill_time_seconds"
            ]
            self._add_managed_task(self._do_log_stats())
            self._add_managed_task(self._force_log(filter_keys=filter_keys))
        while True:
            req_type, req_data = await self.from_proxy.recv_multipart()
            await self._handle_request(req_type, req_data)

    async def _handle_request(self, req_type: bytes, req_data: bytes):
        if req_type == RequestType.ENCODE:
            req = self.decoder_generate.decode(req_data)
            if TIMECOUNT_ENABLED:
                proxy_to_encoder_time_end = time.perf_counter()
                logger.info("encode received proxy request: %s at time: %.3f",
                            req.request_id, proxy_to_encoder_time_end)
            req.sampling_params.max_tokens = 1
            await self._encode_handler(req)
        elif req_type == RequestType.GENERATION:
            req = self.decoder_generate.decode(req_data)
            if TIMECOUNT_ENABLED:
                proxy_to_pd_time_end = time.perf_counter()
                logger.info(
                    "generation received proxy request: %s at time: %.3f",
                    req.request_id, proxy_to_pd_time_end)
            await self._generation_handler(req)
        elif req_type == RequestType.ABORT:
            req = self.decoder_abort.decode(req_data)
            await self._abort_handler(req)
        elif req_type == RequestType.HEARTBEAT:
            req = self.decoder_heartbeat.decode(req_data)
            await self._heartbeat_handler(req)
        else:
            raise Exception(f"Unknown Request Type: {req_type}.")

    async def _encode_handler(self, req: GenerationRequest):
        task = asyncio.create_task(
            self._generate(req, lambda b: (ResponseType.ENCODE, b)))
        self.running_requests.add(task)
        task.add_done_callback(self.running_requests.discard)

    async def _generation_handler(self, req: GenerationRequest):
        task = asyncio.create_task(
            self._generate(req, lambda b: (ResponseType.GENERATION, b)))
        self.running_requests.add(task)
        task.add_done_callback(self.running_requests.discard)

    async def _abort_handler(self, req: GenerationRequest):
        self.engine.abort(request_id=req.request_id)

    async def _heartbeat_handler(self, req: HeartbeatRequest):
        msg = (ResponseType.HEARTBEAT,
               self.encoder.encode(
                   HeartbeatResponse(request_id=req.request_id, status="OK")))
        await self.to_proxy.send_multipart(msg, copy=False)

    async def _generate(
        self,
        req: GenerationRequest,
        make_msg_func,
    ):
        request_id = req.request_id

        try:
            generator = self.engine.generate(
                prompt={
                    "prompt": req.prompt,
                    "multi_modal_data": _decode_mm_data(req.multi_modal_data),
                },
                sampling_params=req.sampling_params,
                request_id=request_id,
            )

            async for request_output in generator:
                response = GenerationResponse.from_request_output(
                    request_output)

                response_bytes = self.encoder.encode(response)
                msg = make_msg_func(response_bytes)
                await self.to_proxy.send_multipart(msg, copy=False)
        except Exception as e:
            logger.exception("Generation failed for request %s", request_id)
            response = FailureResponse(request_id=request_id,
                                       error_message=str(e)
                                       or type(e).__name__)
            response_bytes = self.encoder.encode(response)
            msg = (ResponseType.FAILURE, response_bytes)
            await self.to_proxy.send_multipart(msg, copy=False)

    def _add_managed_task(self, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self.running_requests.add(task)
        task.add_done_callback(self.running_requests.discard)
        return task


def _decode_mm_data(mm_data: dict[str, any]) -> dict[str, any]:
    images = mm_data.get("image", [])
    if not isinstance(images, list):
        images = [images]
    decoded_images = []
    for img in images:
        if img["type"] == "ndarray":
            decoded_img = np.frombuffer(bytes(
                img["data"]), dtype=img["dtype"]).reshape(img["shape"])
            decoded_images.append(decoded_img)
    if len(decoded_images) == 1:
        decoded_images = decoded_images[0]
    return {"image": decoded_images}


def parse_histograms(
        metrics_text: str,
        filter_keys: Optional[list[str]] = None
) -> dict[str, dict[str, float]]:
    """
    Parse Prometheus metrics text, extract only histogram type data,
    and optionally filter by metric name (supporting multiple filter keys).
    Calculates mean value for each histogram: *_sum / *_count

    Args:
        metrics_text (str): Prometheus metrics raw text

        filter_keys (Optional[List[str]]): 
        Only keep histograms containing any string in this list (optional)

    Returns:
        dict: {histogram_name: {'sum': float, 'count': float, 'mean': float}}
    """
    histograms = {}  # name -> {sum, count, mean}

    # 1. Find all histogram metric names
    hist_type_pat = re.compile(r"# TYPE ([\w:]+) histogram")
    all_hist_names = set(hist_type_pat.findall(metrics_text))

    # 2. For each histogram, find _sum and _count lines
    for hist_name in all_hist_names:
        if filter_keys and not any(key in hist_name for key in filter_keys):
            continue
        # Match sum, count
        sum_pat = re.compile(
            rf"^{re.escape(hist_name)}_sum(\{{[^\}}]*\}})? ([\d\.eE\+-]+)",
            re.MULTILINE)
        count_pat = re.compile(
            rf"^{re.escape(hist_name)}_count(\{{[^\}}]*\}})? ([\d\.eE\+-]+)",
            re.MULTILINE)
        sum_match = sum_pat.findall(metrics_text)
        count_match = count_pat.findall(metrics_text)

        # Support multiple label sets for same histogram name
        for (sum_labels,
             sum_value), (count_labels,
                          count_value) in zip(sum_match, count_match):
            labels = {}
            if sum_labels:
                # Convert to dict
                for pair in sum_labels.strip("{}").split(","):
                    if pair:
                        k, v = pair.split("=")
                        if k.strip() != "model_name":
                            labels[k.strip()] = v.strip('"')
            # Calculate mean
            sum_value = float(sum_value)
            count_value = float(count_value)
            mean = sum_value / count_value if count_value > 0 else float("nan")
            # convert to milliseconds
            mean_ms = mean * 1000

            key = hist_name
            # If labels exist, use labels as part of the key
            if labels:
                key = (f"{hist_name}|"
                       f"{'|'.join([f'{k}={v}' for k, v in labels.items()])}")

            histograms[key] = {
                # 'sum': sum_value,
                'count': count_value,
                'mean': mean_ms,
            }
    return histograms
