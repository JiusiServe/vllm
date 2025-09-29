# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utils for model executor."""
from __future__ import annotations
import copy
import socket
import sys
import time
from typing import Any, Optional
import torch
import json
import os
import urllib.request
import urllib.error

_DEFAULT_URL = "http://127.0.0.1:5580/ttft_report"
_URL = os.getenv("VLLM_TTFT_REPORT_URL", _DEFAULT_URL)

def set_random_seed(seed: int) -> None:
    from vllm.platforms import current_platform
    current_platform.seed_everything(seed)


def set_weight_attrs(
    weight: torch.Tensor,
    weight_attrs: Optional[dict[str, Any]],
):
    """Set attributes on a weight tensor.

    This method is used to set attributes on a weight tensor. This method
    will not overwrite existing attributes.

    Args:
        weight: The weight tensor.
        weight_attrs: A dictionary of attributes to set on the weight tensor.
    """
    if weight_attrs is None:
        return
    for key, value in weight_attrs.items():
        assert not hasattr(
            weight, key), (f"Overwriting existing tensor attribute: {key}")

        # NOTE(woosuk): During weight loading, we often do something like:
        # narrowed_tensor = param.data.narrow(0, offset, len)
        # narrowed_tensor.copy_(real_weight)
        # expecting narrowed_tensor and param.data to share the same storage.
        # However, on TPUs, narrowed_tensor will lazily propagate to the base
        # tensor, which is param.data, leading to the redundant memory usage.
        # This sometimes causes OOM errors during model loading. To avoid this,
        # we sync the param tensor after its weight loader is called.
        # TODO(woosuk): Remove this hack once we have a better solution.
        from vllm.platforms import current_platform
        if current_platform.is_tpu() and key == "weight_loader":
            value = _make_synced_weight_loader(value)
        setattr(weight, key, value)


def _make_synced_weight_loader(original_weight_loader):

    def _synced_weight_loader(param, *args, **kwargs):
        original_weight_loader(param, *args, **kwargs)
        # torch._sync doesn't support, is not needed for CPU tensors.
        if param.device != torch.device("cpu"):
            torch._sync(param)

    return _synced_weight_loader


def get_packed_modules_mapping(model: torch.nn.Module) -> dict[str, list[str]]:
    parent_map = copy.deepcopy(getattr(model, "packed_modules_mapping", {}))

    # don't infer mapping if the model has defined it explicitly.
    if parent_map:
        return parent_map

    # We only check main components instead of whole model submodules
    for child in model.children():
        child_map = getattr(child, "packed_modules_mapping", {})
        if any((k in parent_map and parent_map[k] != v)
               for k, v in child_map.items()):
            raise ValueError(
                f"Can't update {type(model).__name__}'s packed_modules_mapping "
                f"safely because of conflicts from {type(child).__name__}.")
        else:
            parent_map.update(child_map)
    return parent_map

def _parse(u: str):
    assert u.startswith("http://")
    rest = u[7:]
    if "/" in rest:
        hostport, p = rest.split("/", 1)
        p = "/" + p
    else:
        hostport, p = rest, "/"
    if ":" in hostport:
        h, pt = hostport.split(":", 1)
        try:
            pt = int(pt)
        except:
            raise ValueError(f"bad port in URL {u}")
    else:
        h, pt = hostport, 80
    return h, pt, p

try:
    HOST, PORT, PATH = _parse(_URL)
except Exception as e:
    print(f"[TTFT][PARSE_ERR] {_URL} {e}", file=sys.stderr, flush=True)
    HOST = PORT = PATH = None


def send_ttft_report(payload: dict) -> None:
    # 调用入口
    print(f"[TTFT][REPORTER_CALL] host={HOST} port={PORT} path={PATH} payload={payload}", flush=True)

    if HOST is None:
        return
    if not payload or "request_id" not in payload:
        return

    # 序列化
    try:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    except Exception as e:
        print(f"[TTFT][SERIALIZE_FAIL] {e} payload={payload}", file=sys.stderr, flush=True)
        return

    # 连接
    t0 = time.perf_counter()
    try:
        sock = socket.create_connection((HOST, PORT), timeout=0.2)
    except Exception as e:
        print(f"[TTFT][CONN_FAIL] {type(e).__name__}:{e} payload={payload}", file=sys.stderr, flush=True)
        return
    t_conn = (time.perf_counter() - t0) * 1000

    # 发送
    try:
        req = (
            f"POST {PATH} HTTP/1.1\r\n"
            f"Host: {HOST}:{PORT}\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n"
            "Connection: close\r\n"
            "\r\n"
        ).encode("utf-8") + body

        sock.sendall(req)
    except Exception as e:
        print(f"[TTFT][SEND_FAIL] {type(e).__name__}:{e} payload={payload}", file=sys.stderr, flush=True)
        try:
            sock.close()
        except:
            pass
        return

    # 直接关闭（不读响应）
    try:
        sock.close()
    except:
        pass

    print(f"[TTFT][SENT] conn_ms={t_conn:.2f} size={len(body)}B rid={payload.get('request_id')}", flush=True)
