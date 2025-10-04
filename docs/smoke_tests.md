# Phase 0 Smoke Tests

Quick commands to confirm the legacy training/evaluation entry points still behave as expected after the initial extractions. Each snippet caps execution at five episodes per the current guardrails.

## Training loop (5 episodes)

```bash
python3 - <<'PY'
from experiments.train import create_training_session, run_training

session = create_training_session()
run_training(session, episodes=5)
PY
```

## Evaluation loop (5 episodes)

```bash
python3 - <<'PY'
from experiments.eval import create_evaluation_session, run_evaluation

session = create_evaluation_session(auto_load=False)
run_evaluation(session, episodes=5, force_render=False)
PY
```

## Runner adapters (bridging legacy loops)

```bash
python3 - <<'PY'
import sys
from pathlib import Path

sys.path.append(str(Path.cwd() / "src"))

from f110x.engine import build_runner_context
from f110x.runner import TrainRunner, EvalRunner
from f110x.utils.config import load_config

cfg, _, _ = load_config(None, default_path="scenarios/gaplock_dqn.yaml")
runner_ctx = build_runner_context(cfg)

TrainRunner(runner_ctx).run(episodes=5)
EvalRunner(runner_ctx).run(episodes=5)
PY
```

These smoke checks should remain green throughout Phase 0 so we can catch regressions while more invasive refactors land.
