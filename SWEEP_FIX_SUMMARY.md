# WandB Sweep Fix Summary

## Problem
- Single runs using `python3 run.py scenarios/sac.yaml` worked perfectly (70%+ success)
- WandB sweeps using `run_sb3_baseline.py` got 0% success (100% timeout)
- Success metrics weren't being logged correctly

## Root Cause
The codebase had TWO different paths for running SB3 algorithms:

1. **run_v2.py path** (single runs - WORKED):
   - Used `SB3SACAgent` wrappers from `src/agents/sb3_agents.py`
   - Integrated SB3 into the proven custom training loop
   - Manually called `agent.act()`, `agent.store_transition()`, `agent.update()`

2. **run_sb3_baseline.py path** (sweeps - BROKEN):
   - Used native SB3 `model.learn()` method
   - Wrapped PettingZoo env with `SB3SingleAgentWrapper`
   - Had subtle issues with the environment wrapper or training setup

## Solution
Added WandB sweep parameter support to `run_v2.py` so we can use the working path for BOTH single runs and sweeps:

### Changes Made:

1. **run_v2.py** (lines 311-368):
   - Added WandB sweep parameter detection after logger initialization
   - Extracts sweep params from `wandb.config`
   - Applies them to `agents.{agent_id}.params.*` in the scenario dict
   - Same logic as run_sb3_baseline.py but in the working training path

2. **sweeps/sac_sweep.yaml**:
   - Changed `program: run.py` → `program: run_v2.py`
   - Removed `--runner sb3` and `--algo sac` flags
   - Now uses run_v2 directly

3. **sweeps/ppo_sweep.yaml**:
   - Changed `program: run.py` → `program: run_v2.py`
   - Removed `--runner sb3` and `--algo ppo` flags
   - Now uses run_v2 directly

### Files Modified:
- ✅ `run_v2.py` - Added sweep parameter support
- ✅ `sweeps/sac_sweep.yaml` - Updated to use run_v2.py
- ✅ `sweeps/ppo_sweep.yaml` - Updated to use run_v2.py

### Files No Longer Needed:
- ❌ `run_sb3_baseline.py` - Can be removed (was causing the issue)
- ❌ `run_sb3_sweep.py` - Can be removed (redundant)
- ❌ `run_sweep.py` - Can be removed (redundant)
- ⚠️ `run.py` - Optional to keep as convenience dispatcher

## How to Use

### Single Run:
```bash
python3 run_v2.py --scenario scenarios/sac.yaml
```

### Sweep:
```bash
# Initialize sweep
wandb sweep sweeps/sac_sweep.yaml

# Run agents (on single machine)
wandb agent <sweep-id>

# Or run agents in parallel (multiple terminals/machines)
wandb agent <sweep-id>  # Terminal 1
wandb agent <sweep-id>  # Terminal 2
wandb agent <sweep-id>  # Terminal 3
```

## Expected Results
- Sweeps should now achieve similar success rates as single runs (70%+ success)
- All sweep parameters (learning_rate, gamma, tau, batch_size, hidden_dims) are applied correctly
- Success metrics are logged correctly to WandB
- Training uses the proven run_v2 training loop that already worked

## Verification
Run the debug script to verify sweep parameters are applied correctly:
```bash
python3 debug_sweep_params.py
```

Expected output: "✓ All sweep parameters applied correctly!"
