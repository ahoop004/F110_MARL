#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <TD3_SWEEP_ID> <PPO_SWEEP_ID> <DQN_SWEEP_ID> [WANDB_API_KEY]" >&2
  exit 1
fi

TD3_SWEEP_ID=$1
PPO_SWEEP_ID=$2
DQN_SWEEP_ID=$3

if [[ $# -ge 4 ]]; then
  export WANDB_API_KEY=$4
fi

run_agent() {
  local gpu=$1
  local config=$2
  local sweep=$3
  CUDA_VISIBLE_DEVICES="$gpu" F110_CONFIG="$config" wandb agent "$sweep" &
}

run_agent 0 configs/experiment_gaplock_td3.yaml "$TD3_SWEEP_ID"
run_agent 1 configs/experiment_gaplock_ppo.yaml "$PPO_SWEEP_ID"
run_agent 2 configs/experiment_gaplock_dqn.yaml "$DQN_SWEEP_ID"

wait
