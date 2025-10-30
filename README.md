# F110_MARL

Multi-agent reinforcement learning stack for F1TENTH-style racing. The project wraps a high-fidelity simulator, PettingZoo-compatible environment, and a roster of training agents so you can stage adversarial racing scenarios (e.g., attacker vs. defender) and iterate on policies with PPO, TD3, DQN, or classic heuristics.

## Federated TD3 Quickstart

The `scenarios/convoy_lock_td3.yaml` scenario enables round-based federated averaging out of the box. When you launch multiple replicas with `run.py`, the harness sets the required environment variables (`FED_TOTAL_CLIENTS`, `FED_CLIENT_ID`, `FED_ROUND_INTERVAL`) automatically so each worker shares weights at the configured interval.

For manual launches (calling `experiments.main` directly), export the following before starting each replica:

- `FED_TOTAL_CLIENTS`: total number of parallel learners participating in federated averaging.
- `FED_CLIENT_ID`: zero-based index identifying the current learner (unique per replica).
- `FED_ROOT`: shared filesystem path where per-round checkpoints are exchanged.
- `FED_ROUND_INTERVAL` *(optional)*: override the episode interval between averaging rounds.
- `FED_CHECKPOINT_AFTER_SYNC` *(optional)*: set to `0`/`false` to skip saving checkpoints after syncs.
- `FED_OPTIMIZER_STRATEGY` *(optional)*: choose how to handle optimizer state (`local` keeps each client’s optimizer, `average` blends momentum buffers, `reset` reinitialises optimizers after sync).

### Upcoming collector options

Collector-based episode parallelism is still under construction, but the CLI already accepts the corresponding flags so you can experiment safely:

```bash
python run.py \
  --scenario scenarios/convoy_lock_td3.yaml \
  --collect-workers 4 \
  --collect-prefetch 8 \
  --collect-seed-stride 17
```

These options simply populate the configuration/environment at the moment; once the collector ships they will activate concurrent episode collection without further changes.

### Example: three parallel replicas

```bash
python run.py \
  --scenario scenarios/convoy_lock_td3.yaml \
  --repeat 3 \
  --max-parallel 3 \
  --auto-seed \
  --seed-base 12345 \
  --wandb-prefix fed-
```

The command above spawns three coordinated runs (sharing the federated root declared in the scenario). Override `FED_ROOT` when you want to isolate experiments:

```bash
FED_ROOT=outputs/fed_trial_$(date +%Y%m%d_%H%M%S) \
python run.py --scenario scenarios/convoy_lock_td3.yaml --repeat 3 --max-parallel 3

### Local smoke test

Use `tools/federated_smoke.py` to spin up a short two-client run (or just preview the command with `--dry-run`). The script reports how many rounds were produced and where artefacts were written:

```bash
python tools/federated_smoke.py --episodes 1 --eval-episodes 0 --max-parallel 2
```
```
