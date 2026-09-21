# Render the MPC controllers

Run these commands from the repository root in a graphical desktop session
(local desktop or the HPC remote desktop). They open a window; an ordinary
headless SSH/Jupyter terminal cannot display it. `env -u PYGLET_HEADLESS` removes
the headless setting used in training commands.

| Scenario | Cars | Duration |
|---|---|---|
| `racing_mpc_solo.yaml` | One blue MPC car | Three laps per map |
| `racing_mpc_2v2.yaml` | Blue/cyan MPC team vs orange/red hybrid PP+FTG team | Three laps per car; joint episode ends when all cars terminate |
| `racing_mpc_passing.yaml` | Blue MPC starts about 3 m behind an orange hybrid capped at 1.2 m/s | 60 simulated seconds per map |

```bash
env -u PYGLET_HEADLESS venv/bin/python run.py \
  --scenario scenarios/render/racing_mpc_solo.yaml --render

env -u PYGLET_HEADLESS venv/bin/python run.py \
  --scenario scenarios/render/racing_mpc_2v2.yaml --render

env -u PYGLET_HEADLESS venv/bin/python run.py \
  --scenario scenarios/render/racing_mpc_passing.yaml --render
```

Each scenario defaults to two episodes: **circle first, Budapest second**.
Use `--episodes 1` for circle only, or edit all three `map_bundles*` lists in
`configs/scenarios/racing_mpc_render_base.yaml` to choose different maps.
Use `--max-steps 400` for a shorter, 20-second simulated preview and `--seed 10042`
to change the named starting positions in solo/2v2. The passing demo uses a
deterministic centerline-relative spawn on each map instead of named starts;
it is a visual example, not the exact benchmark passing spawn.

Scroll to zoom, drag to pan, and press **F** to toggle camera follow. **T** cycles
telemetry; **1–4** select an agent's telemetry and **0** shows all agents.
Stop a run with Ctrl+C in the terminal.

All vehicles use fixed controllers. No checkpoint, neural-network training, or
`--eval` flag is needed. W&B is disabled. Nominal grip matches the initial
controller checks. Collision termination remains enabled; cars that crash stay
as obstacles, while finishers coast briefly and stop. A `TIMEOUT` is expected for
the short passing demo, and a joint `COLLISION` can describe a hybrid crash even
when both MPC cars finished. Per-agent outcomes are in the run's CSV outputs.

The render base's MPC settings and the second MPC car's settings in the 2v2 file
mirror `configs/controllers/racing_mpc.yaml`; keep these together when tuning.
See [controller details and benchmark results](../../docs/RACING_MPC_OPPONENTS.md).
These visual comparisons retain hybrid traffic; active MAPPO training scenarios
use two MPC opponents.
