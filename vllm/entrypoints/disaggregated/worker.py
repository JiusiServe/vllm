# SPDX-License-Identifier: Apache-2.0

import uvloop

from vllm.disaggregated.disagg_worker import DisaggWorker
from vllm.engine.async_llm_engine import AsyncEngineArgs
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.api_server import build_async_engine_client
from vllm.logger import init_logger
from vllm.utils import FlexibleArgumentParser
from vllm.version import __version__ as VLLM_VERSION

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
        worker.shutdown()


async def main(args) -> None:
    logger.info("Disaggregated Worker Server, vLLM ver. %s", VLLM_VERSION)
    logger.info("Args: %s", args)

    async with build_async_engine_client(args) as engine:
        await run(args, engine)


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
    AsyncEngineArgs.add_cli_args(parser)
    uvloop.run(main(parser.parse_args()))
