#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found on PATH" >&2
  exit 1
fi

EPISODES=""
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --episodes)
      if [[ $# -ge 2 ]]; then
        EPISODES="$2"
        shift 2
      else
        echo "--episodes expects a value" >&2
        exit 1
      fi
      ;;
    --episodes=*)
      EPISODES="${1#*=}"
      shift 1
      ;;
    *)
      POSITIONAL+=("$1")
      shift 1
      ;;
  esac
done

if [[ ${#POSITIONAL[@]} -gt 0 ]]; then
  echo "Unexpected arguments: ${POSITIONAL[*]}" >&2
  exit 1
fi

CONFIGS=(
  "td3 configs/experiment_gaplock_td3.yaml 0"
  "ppo configs/experiment_gaplock_ppo.yaml 1"
  "dqn configs/experiment_gaplock_dqn.yaml 2"
)

mkdir -p "$ROOT_DIR/logs"
log_prefix="$ROOT_DIR/logs/$(date +%Y%m%d_%H%M%S)"

pids=()

cleanup() {
  for pid in "${pids[@]:-}"; do
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill "$pid" >/dev/null 2>&1 || true
    fi
  done
}

trap cleanup INT TERM

for entry in "${CONFIGS[@]}"; do
  read -r name cfg gpu <<<"$entry"
  log_file="${log_prefix}_${name}.log"
  echo "Launching $name (config=$cfg) on GPU $gpu -> $log_file"

  cmd=(python3 experiments/main.py)
  if [[ -n "$EPISODES" ]]; then
    cmd+=(--episodes "$EPISODES")
  fi

  CUDA_VISIBLE_DEVICES="$gpu" \
  F110_CONFIG="$ROOT_DIR/$cfg" \
  PYTHONUNBUFFERED=1 \
  "${cmd[@]}" >"$log_file" 2>&1 &

  pids+=("$!")
  sleep 1
done

wait
