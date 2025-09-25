# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# api_proxy.py
import argparse
import asyncio
import logging
import uuid
from collections.abc import AsyncIterator

import aiohttp
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from instance import ServerType
from scheduler import ServerScheduler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

EXCEPT_ERRORS = (
    aiohttp.ClientConnectorError,
    aiohttp.ServerDisconnectedError,
    aiohttp.ClientOSError,
)


@app.on_event("startup")
async def startup_event():
    await app.state.e_scheduler.init_session()
    await app.state.pd_scheduler.init_session()
    if app.state.enable_health_daemon:
        asyncio.create_task(periodic_health_check(app.state.health_check_interval))


@app.on_event("shutdown")
async def shutdown_event():
    await app.state.e_scheduler.stop_instances()
    await app.state.pd_scheduler.stop_instances()


def has_mm_input(request_data: dict):
    if "messages" not in request_data:
        return False
    for message in request_data["messages"]:
        if not isinstance(message.get("content"), list):
            continue
        for content_item in message["content"]:
            if content_item.get("type") in ["image_url", "audio_url", "input_audio"]:
                return True
    return False


async def forward_streaming_request(
    api: str,
    request_data: dict,
    request_id: str,
) -> AsyncIterator[str]:
    headers = {"x-request-id": request_id}
    # Skip request to encoder instance if we don't have mm input
    if has_mm_input(request_data):

        async def non_stream_call(tried_instances):
            e_instance = await app.state.e_scheduler.select_instance(
                exclude_instances=tried_instances
            )
            tried_instances.add(e_instance)
            await e_instance.forward_non_streaming_request(api, request_data, headers)

        await app.state.e_scheduler.non_stream_retry_wrap(non_stream_call)

    async def stream_call(tried_instances):
        pd_instance = await app.state.pd_scheduler.select_instance(
            exclude_instances=tried_instances
        )
        tried_instances.add(pd_instance)
        async for chunk in pd_instance.forward_streaming_request(
            api, request_data, headers
        ):
            yield chunk

    async for chunk in app.state.pd_scheduler.stream_retry_wrap(stream_call):
        yield chunk


async def forward_non_streaming_request(
    api: str,
    request_data: dict,
    request_id: str,
) -> dict:
    headers = {"x-request-id": request_id}
    # Skip request to encoder instance if we don't have mm input
    if has_mm_input(request_data):

        async def e_non_stream_call(tried_instances):
            e_instance = await app.state.e_scheduler.select_instance(
                exclude_instances=tried_instances
            )
            tried_instances.add(e_instance)
            await e_instance.forward_non_streaming_request(api, request_data, headers)

        await app.state.e_scheduler.non_stream_retry_wrap(e_non_stream_call)

    async def pd_non_stream_call(tried_instances):
        pd_instance = await app.state.pd_scheduler.select_instance(
            exclude_instances=tried_instances
        )
        tried_instances.add(pd_instance)
        return await pd_instance.forward_non_streaming_request(
            api, request_data, headers
        )

    return await app.state.pd_scheduler.non_stream_retry_wrap(pd_non_stream_call)


async def _handle_completions(api: str, request: Request):
    """Handle chat completion requests."""
    try:
        request_data = await request.json()
        request_id = request.headers.get("x-request-id")
        if not request_id:
            request_id = str(uuid.uuid4())
        is_streaming = request_data.get("stream", False)
        if is_streaming:
            return StreamingResponse(
                forward_streaming_request(api, request_data, request_id),
                media_type="text/event-stream",
            )
        else:
            result = await forward_non_streaming_request(
                api,
                request_data,
                request_id,
            )
            return JSONResponse(content=result)
    except Exception as e:
        logger.error("Error processing request: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await _handle_completions("/v1/chat/completions", request)


@app.post("/v1/completions")
async def completions(request: Request):
    return await _handle_completions("/v1/completions", request)


@app.get("/workload_debug")
async def gather_workload():
    e_workload = await app.state.e_scheduler.gather_workload()
    pd_workload = await app.state.pd_scheduler.gather_workload()
    workload_status = {
        "encoder workload": e_workload,
        "prefill_decode workload": pd_workload,
    }
    return workload_status


@app.get("/v1/models")
async def list_models():
    try:
        async with app.state.pd_scheduler.instances[0].session.get(
            f"{app.state.pd_scheduler.instances[0].url}/v1/models"
        ) as response:
            response.raise_for_status()
            return await response.json()
    except Exception as e:
        logger.error("Error fetching models: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


async def do_health_check():
    unhealthy_encoder_url = await app.state.e_scheduler.healthy_check()
    unhealthy_pd_url = await app.state.pd_scheduler.healthy_check()

    health_status = {
        "proxy": "healthy",
        "unhealthy encode_servers": unhealthy_encoder_url,
        "unhealthy prefill_decode_servers": unhealthy_pd_url,
    }
    unhealthy = unhealthy_encoder_url or unhealthy_pd_url
    return unhealthy, health_status


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    unhealthy, health_status = await do_health_check()

    if unhealthy:
        return JSONResponse(content=health_status, status_code=503)

    return health_status


async def periodic_health_check(health_check_interval):
    while True:
        result = await do_health_check()
        logger.info("Periodic health check result: %s", result)
        await asyncio.sleep(health_check_interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="API Proxy for distributed vLLM servers"
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Proxy host")
    parser.add_argument("--port", type=int, default=8000, help="Proxy port")

    parser.add_argument(
        "--encode-servers-urls",
        type=str,
        required=True,
        help="URLs of the encode server in comma separated format"
        '(e.g., "http://localhost:8001,http://localhost:8002")',
    )

    parser.add_argument(
        "--prefill-decode-servers-urls",
        type=str,
        required=True,
        help="URLs of the prefill/decode servers in comma separated format"
        '(e.g., "http://localhost:8003,http://localhost:8004")',
    )

    parser.add_argument(
        "--scheduling-proxy",
        type=str,
        default="random",
        help="Instances scheduling proxy: "
        "choose from {random, round_robin, least_inflight}. Default: random",
    )

    parser.add_argument(
        "--enable-health-daemon",
        action="store_true",
        help="Enable background health check daemon",
    )
    parser.add_argument(
        "--health-daemon-interval",
        type=int,
        default=5,
        help="Interval (seconds) for background health check. Default: 5",
    )

    args = parser.parse_args()
    app.state.e_scheduler = ServerScheduler(
        args.encode_servers_urls, ServerType.E_INSTANCE, args.scheduling_proxy
    )
    app.state.pd_scheduler = ServerScheduler(
        args.prefill_decode_servers_urls, ServerType.PD_INSTANCE, args.scheduling_proxy
    )
    app.state.enable_health_daemon = args.enable_health_daemon
    app.state.health_check_interval = args.health_daemon_interval

    logger.info("Starting API proxy on %s:%s with 1 worker", args.host, args.port)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        access_log=False,
        loop="uvloop",
    )
