#!/usr/bin/env bash

set -euo pipefail

CURRENT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

MAX_NUM_SEQS_ENCODER="${MAX_NUM_SEQS_ENCODER:-1}"
MAX_NUM_SEQS_PD="${MAX_NUM_SEQS_PD:-128}"
ENCODER_ADDR_PREFIX="${ENCODER_ADDR_PREFIX:-/tmp/encoder}"
PD_ADDR_PREFIX="${PD_ADDR_PREFIX:-/tmp/prefill_decode}"
PROXY_ADDR="${PROXY_ADDR:-/tmp/proxy}"
PID_FILE="${PID_FILE:-${CURRENT_DIR}/pid.txt}"

MODEL=""
SHARED_STORAGE_PATH="/dev/shm/epd"
GPU_UTILIZATION_ENCODER=0.0
GPU_UTILIZATION_PD=0.95
ENCODER_DEVICE_ID_BASE=0
ENCODER_NUMBER=1
PD_DEVICE_ID_BASE=1
PD_NUMBER=1
LOG_PATH="${CURRENT_DIR}/logs"
IMAGE_FILE_PATH=""

function start_encoder() {
    local dev_id=$1
    local address=$2
    local proxy_address=$3
    local log_file=$4

    VLLM_USE_V1=1 ASCEND_RT_VISIBLE_DEVICES=$dev_id python -m vllm.entrypoints.disaggregated.worker \
        --proxy-addr $proxy_address \
        --worker-addr $address \
        --model $MODEL \
        --gpu-memory-utilization $GPU_UTILIZATION_ENCODER \
        --max-num-seqs $MAX_NUM_SEQS_ENCODER \
        --enforce-eager \
        --no-enable-prefix-caching \
        --ec-transfer-config '{
            "ec_connector": "ECSharedStorageConnector",
            "ec_role": "ec_producer",
            "ec_connector_extra_config": {
                "shared_storage_path": "'"$SHARED_STORAGE_PATH"'"
            }
        }' \
        >"$log_file" 2>&1 &
    echo $! >> "$PID_FILE"
}

function start_pd() {
    local dev_id=$1
    local address=$2
    local proxy_address=$3
    local log_file=$4

    VLLM_USE_V1=1 ASCEND_RT_VISIBLE_DEVICES=$dev_id python -m vllm.entrypoints.disaggregated.worker \
        --proxy-addr $proxy_address \
        --worker-addr $address \
        --model $MODEL \
        --gpu-memory-utilization $GPU_UTILIZATION_PD \
        --max-num-seqs $MAX_NUM_SEQS_PD \
        --enforce-eager \
        --ec-transfer-config '{
            "ec_connector": "ECSharedStorageConnector",
            "ec_role": "ec_consumer",
            "ec_connector_extra_config": {
                "shared_storage_path": "'"$SHARED_STORAGE_PATH"'"
            }
        }' \
        >"$log_file" 2>&1 &
    echo $! >> "$PID_FILE"
}

function start_all() {
    mkdir -p "$LOG_PATH"
    if [ -f "$PID_FILE" ]; then
        rm "$PID_FILE"
    fi

    if [ -d "$SHARED_STORAGE_PATH" ]; then
        rm -rf "$SHARED_STORAGE_PATH"
    fi
    mkdir -p "$SHARED_STORAGE_PATH"

    echo "Starting encoder workers..."
    for ((i=0; i<ENCODER_NUMBER; i++)); do
        dev_id=$((ENCODER_DEVICE_ID_BASE + i))
        address="${ENCODER_ADDR_PREFIX}_$i"
        log_file="$LOG_PATH/encoder_$i.log"
        start_encoder $dev_id $address $PROXY_ADDR $log_file
        echo "  Encoder worker $i starting on device $dev_id, address: $address, log: $log_file"
    done

    echo "Starting prefill/decode workers..."
    for ((i=0; i<PD_NUMBER; i++)); do
        dev_id=$((PD_DEVICE_ID_BASE + i))
        address="${PD_ADDR_PREFIX}_$i"
        log_file="$LOG_PATH/prefill_decode_$i.log"
        start_pd $dev_id $address $PROXY_ADDR $log_file
        echo "  Prefill/decode worker $i starting on device $dev_id, address: $address, log: $log_file"
    done

    echo "All workers starting. PIDs are stored in $PID_FILE."
}

function stop_all() {
    if [ -f "$PID_FILE" ]; then
        while read -r pid; do
            if kill -0 "$pid" > /dev/null 2>&1; then
                echo "Stopping process $pid"
                kill "$pid"
                for i in {1..5}; do
                    sleep 1
                    if ! kill -0 "$pid" > /dev/null 2>&1; then
                        break
                    fi
                done
                if kill -0 "$pid" > /dev/null 2>&1; then
                    echo "Process $pid did not exit, killing with -9"
                    kill -9 "$pid"
                fi
            fi
        done < "$PID_FILE"
        rm "$PID_FILE"
    else
        echo "No PID file found. Are the workers running?"
    fi

    if [ -d "$SHARED_STORAGE_PATH" ]; then
        rm -rf "$SHARED_STORAGE_PATH"
        echo "Removed shared storage at $SHARED_STORAGE_PATH"
    fi
}

function print_help() {
    echo "Usage: $0 [--model MODEL] [--shared-storage-path PATH]
              [--gpu-utilization-encoder FLOAT] [--gpu-utilization-pd FLOAT]
              [--encoder-device-id-base INT] [--encoder-number INT]
              [--pd-device-id-base INT] [--pd-number INT]
              [--image-file-path PATH] [--log-path PATH]
              [--stop] [--help]"
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift ;;
        --shared-storage-path) SHARED_STORAGE_PATH="$2"; shift ;;
        --gpu-utilization-encoder) GPU_UTILIZATION_ENCODER="$2"; shift ;;
        --gpu-utilization-pd) GPU_UTILIZATION_PD="$2"; shift ;;
        --encoder-device-id-base) ENCODER_DEVICE_ID_BASE="$2"; shift ;;
        --encoder-number) ENCODER_NUMBER="$2"; shift ;;
        --pd-device-id-base) PD_DEVICE_ID_BASE="$2"; shift ;;
        --pd-number) PD_NUMBER="$2"; shift ;;
        --log-path) LOG_PATH="$2"; shift ;;
        --image-file-path) IMAGE_FILE_PATH="$2"; shift ;;
        --stop) stop_all; exit 0 ;;
        --help) print_help; exit 0 ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$MODEL" ]; then
    echo "Error: --model is required."
    exit 1
fi

if [ -z "$IMAGE_FILE_PATH" ]; then
    echo "Error: --image-file-path is required."
    exit 1
fi

start_all

chat_with_image() {
    python $CURRENT_DIR/chat_with_image.py \
        --proxy-addr $PROXY_ADDR \
        --encode-addr-list $(for ((i=0; i<ENCODER_NUMBER; i++)); do echo -n "${ENCODER_ADDR_PREFIX}_$i "; done) \
        --pd-addr-list $(for ((i=0; i<PD_NUMBER; i++)); do echo -n "${PD_ADDR_PREFIX}_$i "; done) \
        --model-name $MODEL \
        --image-path $IMAGE_FILE_PATH
}

chat_with_image