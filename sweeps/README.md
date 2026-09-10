# PPO and MAPPO sweeps

All sweeps use `run.py` and pass their parameters through `${args}`. The runner
accepts scenario paths, seeds, and episode budgets; it does not accept arbitrary
hyperparameter CLI flags. Put model/reward changes in scenario configuration.

| Sweep | Experiments |
|---|---|
| `ppo_seed_sweep.yaml` | Existing seed list for PPO gaplock against FTG |
| `ppo_sweep.yaml` | PPO against pure pursuit, Stanley, and hybrid PP/FTG, across seeds |
| `mappo_sweep.yaml` | Four-car MAPPO individual/team reward and critic configurations, across seeds |

```bash
wandb sweep sweeps/mappo_sweep.yaml
wandb agent <sweep-id>
```

These are grid sweeps. They do not optimize an unavailable metric or compare
incompatible shaped rewards. Training logs include `episode/reward` and per-agent
MAPPO metrics. Compare checkpoints using deterministic racing evaluation with
matched maps, seeds, race lengths, and opponents.

Hyperparameter variants can use the same scenario parameter mechanism when an
experiment requires them. Retired algorithm sweeps remain recoverable from the
historical revision documented in the root README.
