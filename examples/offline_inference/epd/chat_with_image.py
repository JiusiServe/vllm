# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import asyncio
import uuid

import numpy as np
from PIL import Image

from vllm import SamplingParams
from vllm.disaggregated.proxy import Proxy

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
args = parser.parse_args()


# prepare image
image = Image.open(args.image_path)
image_array = np.array(image)


async def run_single_request(i, prompt, image_array, sampling_params, p):
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
        print(f"Request({i}) generated_text: {generated_text}", flush=True)


async def main():
    # new proxy
    p = Proxy(
        proxy_addr=args.proxy_addr,
        encode_addr_list=args.encode_addr_list,
        pd_addr_list=args.pd_addr_list,
        model_name=args.model_name,
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

    tasks = [
        asyncio.create_task(
            run_single_request(i, prompt, image_array, sampling_params, p)
        )
        for i in range(10)
    ]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
