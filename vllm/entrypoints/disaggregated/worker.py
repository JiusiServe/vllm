# SPDX-License-Identifier: Apache-2.0

import os
import socket
from contextlib import closing

import requests
import uvloop
from prometheus_client import start_http_server

from vllm.disaggregated.disagg_worker import DisaggWorker
from vllm.engine.async_llm_engine import AsyncEngineArgs
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.api_server import build_async_engine_client
from vllm.logger import init_logger
from vllm.utils import FlexibleArgumentParser
from vllm.v1.metrics.prometheus import get_prometheus_registry
from vllm.version import __version__ as VLLM_VERSION

TIMECONUT_ENABLED = os.getenv("TIMECONUT_ENABLED", "0") == "1"

logger = init_logger(__name__)


async def run(args, engine: EngineClient):
    logger.info("Initializing disaggregated worker")

    worker = DisaggWorker(
        engine=engine,
        address=args.worker_addr,
        proxy_addr=args.proxy_addr,
    )

    try:
        await worker.run_busy_loop()
    finally:
        if TIMECONUT_ENABLED and args.metrics_port is not None:
            ec_role = args.ec_transfer_config.ec_role
            url = f"http://{args.metrics_host}:{args.metrics_port}/metrics"
            try:
                r = requests.get(url, timeout=3)
                print(f"=== GET ec_role:{ec_role} url:{url} ===")
                print(r.text)
            except Exception:
                pass
        worker.shutdown()


async def main(args) -> None:
    logger.info("Disaggregated Worker Server, vLLM ver. %s", VLLM_VERSION)
    logger.info("Args: %s", args)
    # metrics
    if TIMECONUT_ENABLED:
        ec_role = args.ec_transfer_config.ec_role
        args.metrics_port = _find_free_port(args.metrics_port)
        if args.metrics_port is not None:
            start_http_server(args.metrics_port,
                              addr=args.metrics_host,
                              registry=get_prometheus_registry())
            print(f"Started {ec_role} prometheus metrics server on"
                  f"{args.metrics_host}:{args.metrics_port}")
        else:
            print("No free port found. Metrics exporter disabled.")

    async with build_async_engine_client(args) as engine:
        await run(args, engine)


def _find_free_port(start_port: int,
                    max_tries: int = 50,
                    host: str = "0.0.0.0") -> int | None:
    """from start_port, find a free port, return the port or None."""
    port = start_port
    for _ in range(max_tries):
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, port))
                return port
            except OSError:
                port += 1
    return None


if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    parser.add_argument(
        "--proxy-addr",
        type=str,
        required=True,
        help="The address of the proxy.",
    )
    parser.add_argument(
        "--worker-addr",
        type=str,
        required=True,
        help="The address of the worker.",
    )
    parser.add_argument(
        "--disable-frontend-multiprocessing",
        action="store_true",
        help="Disable MQLLMEngine for AsyncLLMEngine.",
    )
    parser.add_argument(
        "--metrics-host",
        type=str,
        default="0.0.0.0",
        help="The host of the metrics server.",
    )
    parser.add_argument(
        "--metrics-port",
        type=int,
        default=9979,
        help="The port of the metrics server.",
    )
    AsyncEngineArgs.add_cli_args(parser)
    uvloop.run(main(parser.parse_args()))
