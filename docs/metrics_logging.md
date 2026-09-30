Training logs use compact defaults. W&B chooses attack or racing metrics from the scenario; optimization and checkpoint selection still receive the complete metrics dictionaries.

W&B records the four PPO diagnostics (policy loss, value loss, entropy, approximate KL), learning rate, episode return/length/outcomes, task evaluation results, and available throughput/update/evaluation timing. Single-learner runs omit duplicate per-agent rewards and team means. Attack runs show actual successes, eligible crashes, target crashes, and ego failures; opponent finish statistics are hidden. Self-play shows rolling team results and evaluation outcomes.

Set `wandb.logging.profile` to `auto` (default), `attack`, `racing`, or `debug`. Debug includes detailed diagnostics and reward components. Group overrides take precedence over the profile:

```yaml
wandb:
  logging:
    profile: auto
    groups:
      train: true          # episode/*, train/*, selfplay/*
      eval: true           # eval/*, selfplay_eval/*
      performance: true    # perf/*
      curriculum: true
      collector: false     # worker/collector diagnostics; off by default
      reward_components: false  # off by default; requires train too
      define_metrics: true
```

Unspecified groups use profile defaults. The repository scenarios leave the two diagnostic overrides unspecified so switching to `debug` enables them. An optional `wandb.logging.metrics` dictionary is a wildcard allowlist that replaces the profile's metric selection; explicit false matches exclude a metric. Groups still apply. Required x-axis values accompany allowed measurements automatically. For example:

```yaml
wandb:
  logging:
    metrics:
      train/policy_loss: true
      eval/attack_*: true
      eval/attack_target_crashes: false
```

Use `--set wandb.logging.profile=debug` for a detailed dashboard, or `--set wandb.logging.groups.reward_components=true` for only reward diagnostics. W&B uses environment steps for optimizer/evaluation plots and completed episode number for episode plots. Disabling W&B does not disable local artifacts.

Local MAPPO episodes have one canonical record in `race_metrics.jsonl`, including agent outcomes, rewards, components, attack facts, and run identity. `update_metrics.csv` holds optimizer diagnostics. PPO retains `episode_metrics.csv` because it has no race-record export. Self-play retains its team and evaluation JSONL records. Configuration, physics provenance, checkpoints, and evaluation history remain available. Existing run review reads the canonical race/update artifacts.

Optional local settings:

```yaml
logging:
  csv_exports: false           # additional MAPPO episode/agent CSV copies
  collector_progress: false   # additional collector_progress.csv diagnostics
  flush_every: 64
  flush_interval_seconds: 10.0
```

Files flush after 64 records, on the first logging event after 10 seconds, and at normal shutdown. An abrupt process kill may lose buffered rows; use `--set logging.flush_every=1` when immediate visibility is needed. CSV headers expand once per flush batch when new fields appear. CSV exports can be restored with `--set logging.csv_exports=true`; old run files are not migrated or deleted.

W&B consumes episode facts already accumulated by MAPPO. Enabling it no longer creates per-step dataset transitions or a second reward accumulator. Physics provenance is emitted at episode start in PPO and MAPPO, including episodes interrupted by the training budget. Dataset recording still requests full transition records when explicitly enabled.
