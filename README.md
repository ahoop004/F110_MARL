# F110_MARL

Multi-agent reinforcement learning stack for F1TENTH-style racing. The project wraps a high-fidelity simulator, PettingZoo-compatible environment, and a roster of training agents so you can stage adversarial racing scenarios (e.g., attacker vs. defender) and iterate on policies with PPO, TD3, DQN, or classic heuristics.

## Centerline Assets

- Drop a CSV named `<map>_centerline.csv` next to the map YAML to expose a track centerline. Each row should follow `x,y,theta` (trailing commas are ignored) expressed in map coordinates.
- The loader auto-discovers the file when `centerline_autoload: true` (default) and exposes the data to both heuristics and learning agents.
- Rendering overlays the centerline when `centerline_render: true`, and observation wrappers append `[lateral_error, longitudinal_error, heading_error, progress]` features when `centerline_features: true`.

## Reward Strategies

- Configure `reward.mode` in the experiment YAML to choose between `gaplock`, `progress`, `fastest_lap`, or `composite` strategies.
- Gaplock rewards sparse pursuit outcomes (defender crashes vs. ego collisions) with optional scaling.
- Progress shaping uses the loaded centerline to pay out forward motion and penalise lateral/heading error.
- Fastest-lap mode incentivises lap completions with optional best-lap bonuses and per-step penalties.
- Composite mode blends multiple strategies with configurable weights; inspect `reward.components` for examples.
