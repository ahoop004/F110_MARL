# Render the MPC controllers

Use the standalone [racing_mpc.yaml](racing_mpc.yaml) file from a graphical desktop:

```bash
env -u PYGLET_HEADLESS python3 run.py --scenario scenarios/render/racing_mpc.yaml --render
```

The active configuration runs one MPC car for three laps, first on circle and
then Budapest. The commented `racing_mpc_2v2` and `racing_mpc_passing` recipes in
that file add the other cars, team assignments, colors and spawn settings.
Apply each recipe to the active solo configuration. The passing recipe uses a
60-second horizon and a slower hybrid car ahead of the MPC.

All controller parameters, vehicle settings and map lists are inline. Edit the
three `map_bundles*` lists together to change tracks. Use `--episodes 1` for one
map, `--max-steps 400` for a 20-second preview, and `--seed 10042` for a different
random start where random spawning is enabled. These fixed controllers need no
checkpoint or evaluation flag.

Scroll to zoom, drag to pan, **F** toggles follow, **T** cycles telemetry, and
**1–4** select a car (**0** shows all). Ctrl+C stops the process. Collisions and
finish clearance retain the selected scenario's physical behavior.
See [MPC details](../../docs/RACING_MPC_OPPONENTS.md) for sensing and solver limits.
