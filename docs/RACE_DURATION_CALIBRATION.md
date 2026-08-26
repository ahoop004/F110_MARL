# Race-duration calibration

Calibration date: 2026-08-26. All runs were headless, used seeds 11, 22, and
33, and cycled the eight configured maps once per seed. Each output directory
contains `config_snapshot.json`, `episode_metrics.csv`, and `agent_metrics.csv`.

## Commands

```bash
PYGLET_HEADLESS=true venv/bin/python run.py \
  --scenario scenarios/calibration/hybrid_pp_ftg_1lap.yaml \
  --no-wandb --seed <11|22|33> --output-dir outputs/calibration/1lap_seed<seed>

PYGLET_HEADLESS=true venv/bin/python run.py \
  --scenario scenarios/calibration/pure_pursuit_3lap.yaml \
  --no-wandb --seed <11|22|33> \
  --output-dir outputs/calibration/pure_pursuit_3lap_final_seed<seed>
```

## Results

One-lap Hybrid PP+FTG completed 22/24 trials. Successful duration was 165–619
decision steps (median 340; observed p95/max 619). Budapest, Melbourne,
Montreal, Shanghai, Silverstone, Spa, and Spielberg completed for every seed.
Circle completed for one seed and collided for two.

The conservative three-lap Pure Pursuit arm completed 6/24 trials. Budapest
completed in 51,645–51,872 steps and Shanghai in 64,019–64,121 steps. The
successful distribution was 51,645–64,121 (median 57,945.5; observed p95/max
64,121). Circle reached the 150,000-step calibration limit; the remaining maps
ended in controller collisions.

| Map | 1-lap completions | 1-lap successful steps | 3-lap completions | 3-lap successful steps |
| --- | ---: | --- | ---: | --- |
| Budapest | 3/3 | 175–288 | 3/3 | 51,645–51,872 |
| circle | 1/3 | 165 | 0/3 | — |
| Melbourne | 3/3 | 265–474 | 0/3 | — |
| Montreal | 3/3 | 175–521 | 0/3 | — |
| Shanghai | 3/3 | 169–226 | 3/3 | 64,019–64,121 |
| Silverstone | 3/3 | 175–618 | 0/3 | — |
| Spa | 3/3 | 285–619 | 0/3 | — |
| Spielberg | 3/3 | 395–611 | 0/3 | — |

## Decision

The successful three-lap p95 supports a limit well below 250,000 steps, but
the all-map fixed baseline is not reliable enough to declare a final bound.
Keep `max_steps: 250000` as an explicit safety ceiling for now. Do not treat
collision durations as censored completion samples. Final calibration requires
a fixed controller that completes three laps on every train and held-out map;
then select and document a percentile plus a fixed safety margin.

The run also exposed a closed-centerline seam bug in waypoint nearest/lookahead
helpers. The helpers now wrap on closed paths and retain clamping on open paths.
The conservative controller still fails several maps, so that geometry fix is
necessary but not sufficient to establish a strong multi-map baseline.
