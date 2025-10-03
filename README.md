# F110_MARL

Multi-agent reinforcement learning stack for F1TENTH-style racing. The project wraps a high-fidelity simulator, PettingZoo-compatible environment, and a roster of training agents so you can stage adversarial racing scenarios (e.g., attacker vs. defender) and iterate on policies with PPO, TD3, DQN, or classic heuristics.

## Centerline Assets

- Drop a CSV named `<map>_centerline.csv` next to the map YAML to expose a track centerline. Each row should follow `x,y,theta` (trailing commas are ignored) expressed in map coordinates.
- The loader auto-discovers the file when `centerline_autoload: true` (default) and exposes the data to both heuristics and learning agents.
- Rendering overlays the centerline when `centerline_render: true`, and observation wrappers append `[lateral_error, longitudinal_error, heading_error, progress]` features when `centerline_features: true`.
