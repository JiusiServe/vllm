# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import asyncio
import uuid

import numpy as np
from PIL import Image
from prometheus_client import start_http_server

from vllm import SamplingParams
from vllm.disaggregated.proxy import Proxy
from vllm.entrypoints.disaggregated.worker import _find_free_port
from vllm.logger import init_logger
from vllm.v1.metrics.prometheus import get_prometheus_registry

logger = init_logger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--proxy-addr", required=True, help="Proxy address")
parser.add_argument(
    "--encode-addr-list",
    required=True,
    nargs="+",
    help="List of encode addresses",
)
parser.add_argument(
    "--pd-addr-list",
    required=True,
    nargs="+",
    help="List of pd addresses",
)
parser.add_argument("--model-name", required=True, help="Model name")
parser.add_argument("--image-path", required=True, help="Path to the image")
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
args = parser.parse_args()

# new proxy
p = Proxy(
    proxy_addr=args.proxy_addr,
    encode_addr_list=args.encode_addr_list,
    pd_addr_list=args.pd_addr_list,
    model_name=args.model_name,
)

# prepare image
image = Image.open(args.image_path)
image_array = np.array(image)


async def main():
    # metrics
    metrics_port = _find_free_port(args.metrics_port)
    if metrics_port is not None:
        start_http_server(
            metrics_port, addr=args.metrics_host, registry=get_prometheus_registry()
        )
        logger.info(
            "Started prometheus metrics server on %s:%d",
            args.metrics_host,
            metrics_port,
        )
    else:
        logger.warning(
            "No free port found in range [%d, %d). Metrics exporter disabled.",
            args.metrics_port,
            args.metrics_port + 50,
        )

    prompt = (
        "<|im_start|> system\n"
        "You are a helpful assistant.<|im_end|> \n"
        "<|im_start|> user\n"
        "<image> \n"
        "What is the text in the illustrate?<|im_end|> \n"
        "<|im_start|> assistant\n"
    )
    sampling_params = SamplingParams(max_tokens=128)

    outputs = p.generate(
        prompt={
            "prompt": prompt,
            "multi_modal_data": {"image": image_array},
        },
        sampling_params=sampling_params,
        request_id=str(uuid.uuid4()),
    )
    async for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
