# F110_MARL

Multi-agent reinforcement learning stack for F1TENTH-style racing. The project wraps a high-fidelity simulator, PettingZoo-compatible environment, and a roster of training agents so you can stage adversarial racing scenarios (e.g., attacker vs. defender) and iterate on policies with PPO, TD3, DQN, or classic heuristics.

## Centerline Assets

- Drop a CSV named `<map>_centerline.csv` next to the map YAML to expose a track centerline. Each row should follow `x,y,theta` (trailing commas are ignored) expressed in map coordinates.
- The loader auto-discovers the file when `centerline_autoload: true` (default) and exposes the data to both heuristics and learning agents.
- Rendering overlays the centerline when `centerline_render: true`, and observation wrappers append `[lateral_error, longitudinal_error, heading_error, progress]` features when `centerline_features: true`.

### Map Folder Layout

- Store each track under its own directory in `maps/`, e.g. `maps/shanghai/Shanghai_map.yaml`, `maps/shanghai/Shanghai_map.png`, and optional extras like `*_centerline.csv` or wall CSVs.
- Configs should reference the YAML relative to `map_dir`, e.g. `map_yaml: shanghai/Shanghai_map.yaml`.
- The loader preserves backwards compatibility by falling back to a recursive search, but the per-map folders keep assets self-contained and travel well with version control.

## Spawn Point Annotations

- Annotate maps with named spawn points under `annotations.spawn_points` (each entry needs a `name` and `[x, y, yaw]` pose array).
- Use `env.spawn_points` to assign a fixed spawn per agent slot (e.g., `spawn_points: [spawn_1, spawn_2]`).
- Provide `env.spawn_point_sets` to list reusable combinations that `reset_with_start_poses` will sample uniformly at reset time.
- Enable `env.spawn_point_randomize` (bool or dict with `pool`, `allow_reuse`) to draw agent spawns from the annotated pool each episode without hand-coding combinations.
- Training/eval logs now include `spawn_points` (per-agent names) and `spawn_option` so runs stay reproducible.

## Render Controls

- `T` toggles camera follow mode (auto-follow vs. free camera). Dragging with the mouse or using pan keys also switches to free mode.
- `TAB` cycles the tracked agent (hold `Shift` for previous). `PageUp/PageDown` are equivalent shortcuts.
- `+` / `-` (or mouse scroll) zoom in/out; hold the mouse over a point to zoom around it. Zoom obeys `Shift` to accelerate.
- Arrow keys or `WASD` pan the view; hold `Shift` for faster movement. Mouse drag also pans.
- `Space` or `Home` resets the camera to the default follow view.

## Reward Strategies

- Configure `reward.mode` in the experiment YAML to choose between `gaplock`, `progress`, `fastest_lap`, or `composite` strategies.
- Gaplock rewards sparse pursuit outcomes (defender crashes vs. ego collisions) with optional scaling.
- Progress shaping uses the loaded centerline to pay out forward motion and penalise lateral/heading error.
- Fastest-lap mode incentivises lap completions with optional best-lap bonuses and per-step penalties.
- Composite mode blends multiple strategies with configurable weights; inspect `reward.components` for examples.
