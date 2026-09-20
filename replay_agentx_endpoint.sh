#!/usr/bin/env bash
# Replay InferenceX AgentX traces against an already-running engine endpoint.
#
# Unlike the checked-in agentic launchers, this helper never starts the serving
# engine. It runs the same AIPerf `inferencex-agentx-mvp` replay, but points the
# client directly at a caller-owned HTTP endpoint.
#
# Required environment inputs (validated below):
#   CONC                     AgentX lane/tree concurrency.
#   DURATION                 Profiling duration in seconds. The scenario rejects
#                            values below 900 unless --unsafe-override mode is
#                            explicitly enabled via AIPERF_UNSAFE_OVERRIDE=true.
#   KV_OFFLOADING            Deployment KV-offload tier metadata for the result
#                            aggregate: `none`, or `dram` together with
#                            KV_OFFLOAD_BACKEND[_METADATA] required by the
#                            result builder.
#   MODEL                    Real HF model id used as the tokenizer.
#   MODEL_PREFIX             InferenceX model prefix selecting the trace loader.
#   RESULT_DIR               Directory receiving replay logs and artifacts.
#   RESULT_FILENAME          Aggregate JSON filename (without `.json`).
#   TP                       Server tensor-parallel size (per-GPU throughput
#                            is divided by the emitted GPU count).
#
# Strongly recommended optional inputs:
#   AIPERF_SERVER_URL        http(s)://<host>:<port> engine endpoint. Required
#                            for this helper; PORT is derived from it.
#   AIPERF_SERVER_METRICS_URLS
#                            Comma-separated Prometheus /metrics URL(s). Keep
#                            unset only when the endpoint has no metrics.
#   SERVED_MODEL_NAME        Wire name reported by `/v1/models`. Defaults to
#                            $MODEL inside the shared helper.
#   MAX_MODEL_LEN            Engine context limit; traces longer than this are
#                            filtered instead of becoming deterministic 4xx.
#   FRAMEWORK                Result/framework metadata, e.g. `sglang`.
#   EP_SIZE, DP_ATTENTION    Remaining topology metadata for the result aggregate.
#   WEKA_LOADER_OVERRIDE     Pin a non-default trace corpus loader.
#   AIPERF_EXPERIMENTAL_FAST Explicit `0` (canonical) or `1` (one-warmup/lane
#                            1200s smoke mode). The shared helper requires this
#                            input, so this script rejects an unset value.
#   AIPERF_REQUIRED_SERVER_METRIC_PREFIX
#                            e.g. `sglang:` to require engine-metrics capture.
#   HF_TOKEN, HF_HUB_CACHE   Forwarded automatically by the environment.
#
# Example:
#   AIPERF_SERVER_URL=http://10.0.0.1:30000 \
#   AIPERF_SERVER_METRICS_URLS=http://10.0.0.1:30000/metrics \
#   AIPERF_EXPERIMENTAL_FAST=0 \
#   MODEL=deepseek-ai/DeepSeek-V4-Pro-0813 SERVED_MODEL_NAME=DeepSeek-V4-Pro \
#   MODEL_PREFIX=dsv4 FRAMEWORK=sglang TP=8 EP_SIZE=8 DP_ATTENTION=true \
#   KV_OFFLOADING=none CONC=4 DURATION=1800 MAX_MODEL_LEN=131072 \
#   RESULT_DIR="$PWD/agentx_results" RESULT_FILENAME=agentx_dsv4_c4 \
#   bash replay_agentx_endpoint.sh

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$SCRIPT_DIR/benchmarks/benchmark_lib.sh" --validation-only

check_env_vars \
    AIPERF_EXPERIMENTAL_FAST \
    AIPERF_SERVER_URL \
    CONC \
    DURATION \
    FRAMEWORK \
    KV_OFFLOADING \
    MODEL \
    MODEL_PREFIX \
    RESULT_DIR \
    RESULT_FILENAME \
    TP

if ! [[ "$DURATION" =~ ^[0-9]+$ && "$DURATION" -ge 900 ]]; then
    echo "ERROR: DURATION must be an integer >= 900 seconds for a valid replay; got '$DURATION'." >&2
    echo "       Shorter diagnostic runs require AIPERF_UNSAFE_OVERRIDE=true and are marked invalid." >&2
    exit 1
fi

if ! [[ "$CONC" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: CONC must be a positive integer; got '$CONC'." >&2
    exit 1
fi

# Derive the shared helper's required PORT from the caller's endpoint URL.
PORT="$(python3 - "$AIPERF_SERVER_URL" <<'PYPORT'
import sys
from urllib.parse import urlparse

url = urlparse(sys.argv[1])
if url.scheme not in ("http", "https") or not url.hostname:
    raise SystemExit(
        "ERROR: AIPERF_SERVER_URL must be http(s)://<host>[:<port>]; got "
        f"{sys.argv[1]!r}"
    )
print(url.port if url.port is not None else (443 if url.scheme == "https" else 80))
PYPORT
)"
export PORT

# Fail fast on unreachable endpoints and wire-name mismatches instead of
# spending trace-loading time first.
echo "Preflighting endpoint: $AIPERF_SERVER_URL/v1/models"
if ! curl -fsS --max-time 10 "$AIPERF_SERVER_URL/v1/models"; then
    echo "ERROR: could not fetch $AIPERF_SERVER_URL/v1/models; check reachability and auth." >&2
    exit 1
fi
echo

if [[ -z "${SERVED_MODEL_NAME:-}" ]]; then
    echo "NOTE: SERVED_MODEL_NAME is unset; replay requests will use MODEL ('$MODEL')." >&2
    echo "      Set it to the endpoint's wire name when it differs from the HF id." >&2
fi
if [[ -z "${AIPERF_SERVER_METRICS_URLS:-}" ]]; then
    echo "NOTE: AIPERF_SERVER_METRICS_URLS is unset; server-metric plots will be empty." >&2
fi
if [[ -z "${MAX_MODEL_LEN:-}" ]]; then
    echo "NOTE: MAX_MODEL_LEN is unset; corpus traces longer than the engine limit become 4xx failures." >&2
    echo "      Set it to the engine's max-model-len so long traces are filtered instead." >&2
fi

# Workflow-owned thresholds, windows, and warmups for AgentX replays.
source "$SCRIPT_DIR/benchmarks/runtime_settings.sh"

# Direct-endpoint mode cannot observe the remote engine's GPUs, so this tool
# is single-node and disables the local GPU power monitor and power contract.
export ENABLE_AGENTX_POWER=0
export IS_MULTINODE=false
export REQUIRE_POWER=0

export AGENTIC_OUTPUT_DIR="$RESULT_DIR"
mkdir -p "$RESULT_DIR"

# infx package roots and the aiperf/agentic-benchmark checkouts are resolved
# from this workspace. install_agentic_deps and trace helpers are registered
# when benchmark_lib.sh is sourced, so set it first.
if [[ -z "${INFMAX_CONTAINER_WORKSPACE:-}" ]]; then
    export INFMAX_CONTAINER_WORKSPACE="$SCRIPT_DIR"
fi

source "$SCRIPT_DIR/benchmarks/benchmark_lib.sh"

echo "Building isolated AIPerf environment..."
install_agentic_deps

echo "Resolving and downloading AgentX trace corpus..."
resolve_trace_source

echo "Assembling AgentX replay command..."
build_replay_cmd "$RESULT_DIR"

echo "Replaying AgentX traces against $AIPERF_SERVER_URL (CONC=$CONC, DURATION=$DURATION)..."
run_agentic_replay_and_write_outputs "$RESULT_DIR"

echo "Aggregate result: $RESULT_DIR/$RESULT_FILENAME.json"
echo "Replay command record: $RESULT_DIR/benchmark_command.txt"
echo "Raw AIPerf artifacts: $RESULT_DIR/aiperf_artifacts/"
