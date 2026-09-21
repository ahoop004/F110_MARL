# Planar MF6.1 physics and pretraining tests

The existing `combined_slip_st` implementation now uses `model_version: 2` and
`tire_model: mf61_planar`. It replaces the former smooth friction-circle tire
law in the same vehicle, simulator, and training interfaces. Version 1 of that
nonlinear model is rejected; its checkpoints cannot be reused under new forces.
The independent `legacy_st` path remains available to existing legacy scenarios.

This is a **planar, steady-state MF6.1 force subset with synthetic coefficients**,
not a calibrated reproduction of arXiv:2504.02420v2. The paper does not provide
its complete identified parameter set. Software readiness is not evidence of
matching its vehicle or real-world performance.

## Run a bounded test

The canonical entry point is `scenarios/ppo_lap_completion_pretrain.yaml`.
The existing `_frenet` and `_combined_slip` filenames inherit its physics,
observation, action, and task configuration. Their worker counts and training
hyperparameters can still differ.

```bash
PYGLET_HEADLESS=true venv/bin/python run.py \
  --scenario scenarios/ppo_lap_completion_pretrain.yaml \
  --episodes 2 --num-envs 1 --max-steps 64 --no-wandb
```

`--max-steps` overrides the environment horizon and evaluation horizon. At the
configured action repeat of one, 64 steps are 3.2 simulated seconds per episode.
This tests collection, PPO updates, and checkpoint writing. Evaluation runs at
the configured checkpoint-selection interval (500 episodes by default), or via
`--eval` with an explicit saved checkpoint. A short untrained run is not a
lap-completion benchmark.

For a full run, omit the three testing overrides. Supply the new track bundle in
all three `environment.map_bundles*` lists when it is ready. No tire parameters
belong in the occupancy image, centerline CSV, or walls CSV.

## Configuration and calibration

`configs/vehicle/combined_slip.yaml` is the runtime profile. Its vehicle parameters
must be nested under `environment.vehicle_params`. The standalone component
profile `combined_slip_development.yaml` contains the same physics values for
numerical tests and is not a training-scenario include.

Parameters are explicitly marked `calibration.status: uncalibrated`. The current
values are development assumptions, including mass/inertia, geometry, wheel
radius, actuator time constants, and MF coefficients. The steering reference
bound of ±0.5 rad and wheel-reference acceleration of ±5 m/s² follow the paper;
the wheel-speed bound is still provisional. To reproduce the authors' vehicle,
obtain their parameters or identification data rather than treating these
synthetic values as measurements.

Each `front_tire` / `rear_tire` block describes **one effective axle**. `FNOMIN`
is the axle's nominal vertical load in newtons, not a full-size road-car tire
load. Do not directly insert per-wheel or full-scale tire data without reconciling
loads and force aggregation. Coefficient names follow the MF convention:

| Coefficients | Role |
|---|---|
| `FNOMIN` | Nominal axle load |
| `PCX/PDX/PEX/PKX/PHX/PVX` | Longitudinal shape, peak, curvature, stiffness, shifts |
| `PCY/PDY/PEY/PKY/PHY/PVY` | Lateral shape, peak, curvature, stiffness, shifts |
| `RBX/RCX/REX/RHX` | Combined-slip longitudinal reduction |
| `RBY/RCY/REY/RHY/RVY` | Combined-slip lateral reduction and induced lateral force |

The full supported key list is `MF61_KEYS` in `physics/tire_models.py`. Coefficients
must be explicit and finite. Unknown/missing keys, invalid signs, and incompatible
model versions fail validation. Forces also reject invalid load-dependent peaks,
stiffness, and singular weighting normalizations at runtime.

The synthetic profile has zero force shifts and a declining force beyond peak
slip. Friction scaling is `LMUX = LMUY = mu`; remaining scale factors are one.
MF combined-slip weighting is not an imposed radial friction-circle clamp.

## State, slip, and force conventions

The internal state is `[x, y, psi, vx, vy, r, delta, omega]` in SI units.
Body x points forward and y left; yaw and steering are positive to the left.
Steering is radians, wheel speed rad/s, and wheel-reference derivative rad/s².

For front/rear distances `lf, lr`, contact velocities in each tire frame are:

```text
uf = cos(delta)*vx + sin(delta)*(vy + lf*r)
vf = -sin(delta)*vx + cos(delta)*(vy + lf*r)
ur = vx
vr = vy - lr*r
```

The actual wheel actuator speed determines rolling speed `R*omega`, independent
of chassis velocity. One equivalent wheel speed is shared by the front/rear
axles (`shared_speed_awd`). Differential wheel speeds and torque split are not
modeled; this drivetrain assumption still requires comparison with the authors'
implementation.

```text
scale = max(abs(u), slip_speed_floor)
kappa = (R*omega - u) / scale
alpha = atan2(v, scale)
```

The contact-velocity slip-angle convention uses negative `PKY1` for restoring
lateral force. The ground-speed denominator allows slip ratios above one during
wheel spin. The absolute speed and explicit floor are a regularized extension
through rest/reverse; they are not an identified low-speed contact model.

The force kernel implements the pure-slip sine/arctangent relations and the
normalized combined-slip cosine weighting functions. It includes load-dependent
peaks/stiffness/curvature and horizontal/vertical force shifts. Zero load or zero
friction produces zero force. The domain is zero camber, nominal inflation
pressure, no turn slip, and steady state. Tire relaxation, aligning/rolling
moments, camber and pressure dynamics are not implemented or claimed.

Equation references: Pacejka, *Tire and Vehicle Dynamics*, third edition,
4.E9–4.E29 and 4.E50–4.E67, evaluated in this planar domain. Useful public
references are the [MFeval description](https://mfeval.wordpress.com/usingmfeval/)
and [TNO MF-Tyre/MF-Swift documentation](https://functionbay.com/documentation/onlinehelp/Documents/Tire/MFTyre-MFSwift_Help.pdf).
The project supplies its own numerical kernel; it does not vendor an external
tire-library implementation or depend on MATLAB.

## Load transfer and chassis integration

MF forces depend nonlinearly on axle load, so the former reduced model's algebraic
shortcut is no longer valid. At every RK4 stage the implementation solves:

```text
Fzf(ax) = m*(g*lr - h*ax)/(lf + lr)
Fzr(ax) = m*g - Fzf(ax)
ax = (Fx_front*cos(delta) - Fy_front*sin(delta) + Fx_rear) / m
```

A safeguarded scalar secant solve with periodic bisection finds the simultaneous
force/load equilibrium within the no-wheel-lift acceleration interval. For zero
center-of-mass height, loads are static and no nonlinear solve is needed. An
unbracketed solution, nonconvergence, or nonfinite state raises without committing
a partially advanced vehicle. There is no silent negative-load clipping beyond
roundoff at bracket endpoints.

The chassis retains body-frame Coriolis terms and front/rear yaw moment balance.
The maximum RK4 integration step remains 0.001 s, independent of the policy's
0.05 s decision interval. Steering/wheel actuator states are sampled at the RK4
stage times. Step partition convergence and independently driven stage integration
are covered by tests.

## Actuators and observations

The existing `WheelActuators` component integrates held-reference first-order
steering and wheel-speed responses. Its rate-limiting capability remains tested,
but the runtime profile uses limits that are nonbinding over its configured
state/reference bounds, matching the form of the paper's first-order equations.
The provisional time constants are 0.5 s (steering) and 0.15 s (wheel speed).

The action wrapper integrates wheel-reference acceleration once per policy
decision and clamps the stored reference to prevent windup. At R=0.05 m,
100 rad/s² corresponds to 5 m/s². This is a reference-change limit, not an
instantaneous chassis-acceleration constraint.

The pretraining observation is the paper's Eq. (2) layout with real simulated
wheel state and no LiDAR: 10 vehicle values plus 20 curvature and 20 width values.
Track samples are 0.3 m apart. N=20 and normalization maxima are provisional;
the paper does not give all corresponding numerical values. Normalization divides
by maxima without clipping. The derivative slot reflects the applied reference
change after saturation, which needs checking against the authors' convention.

## Friction protocol

Pretraining samples `mu = nominal_mu * Normal(1, 0.02)` at reset, independently of
spawn/sensor RNGs, shared across cars and held constant through the episode.
Evaluation uses fixed nominal friction. The existing `EpisodeFriction` service
supports `gaussian` alongside fixed/uniform/grid protocols; random evaluation
protocols are rejected. Negative/nonfinite Gaussian draws raise explicitly rather
than silently changing the distribution through clipping or rejection sampling.

Actual samples, seed/stream/draw metadata, and nominal configuration remain in
`physics_episodes.jsonl` and environment info. Friction is not exposed as a policy
observation. Map `surface` metadata is descriptive and does not override grip.

## Paper time-trial protocol

`ppo_lap_completion_pretrain.yaml` now configures 400 environments, 1,024
transitions per environment per rollout, and 120,000,000 aggregate transitions.
`params.n_steps` is pooled (409,600). Resets remain inside a rollout; the final
budget remainder is collected exactly. True terminals block value bootstrapping,
truncations use their final observation value, and both stop cross-episode GAE.
The learning-rate schedule uses collected transitions. The stated endpoints are
from the paper; linear interpolation is an explicit implementation choice.

Parallel startup initializes at most 16 new workers concurrently, waiting for
each batch's readiness before launching the next. All 400 workers then receive
a start signal; collection size and update boundaries are unchanged. Configure
`experiment.worker_startup_batch_size` and `worker_startup_timeout_s` (600 s)
separately from `worker_response_timeout_s` (120 s during collection). A startup
timeout does not identify its underlying cause: inspect scheduler logs for
resource limits, and check CPU, RAM, and filesystem contention. Staggering startup
reduces initialization contention but does not reduce steady-state worker memory.

The existing progress component computes signed, seam-corrected progress in
metres. If `info.track_limits.exceeded` is true, it returns only -1 instead.
There are no finish bonuses, time costs, collision penalties, or progress clamps.
Training has no lap or artificial time-limit termination. Boundary violations
reset to a random centerline position and resample friction.

`track_limits.enabled` selects a center-point boundary test, consistent with the
paper's Frenet lateral-distance criterion. Full wall-to-wall width is interpolated
at the current projection, then halved. This assumes a lane-bisecting centerline;
the supplied 1 m L-map satisfies this assumption. It does not use upcoming preview
widths or the vehicle footprint. Wall-contact stopping is disabled for this
single-vehicle task; otherwise the footprint could hit a wall before the vehicle
center reaches the geometric boundary. Ordinary race scenarios retain collisions.

Evaluation uses nominal grip, 20 completed laps, and an 800 s safety horizon.
It records excursions without resetting on them. The first accepted forward
crossing starts timing, excluding the random-spawn approach. Reports include
fastest valid lap, mean/std lap time, fraction of laps with violations, and
mean time-integrated off-track distance in m*s per complete lap. Unfinished laps
are excluded from these lap metrics; completion and timeout remain reported.
Timing and integration are sampled at the 0.05 s decision interval.

Checkpoint/evaluation cadence is measured in transitions, rounded up to a
completed PPO update. The default interval is ten full rollouts (4,096,000
transitions). `final_model.pt` saves the final update even when the budget ends
mid-episode. `lap_time` checkpoint selection ranks completion, valid-lap count,
fastest valid lap, then off-track error. This selection rule is an implementation
choice, not a claimed paper setting. A smaller worker count is an experimental
deviation; change pooled `n_steps` to keep 1,024 steps per worker.

## Remaining differences from the paper

Vehicle coefficients and actuator time constants remain synthetic. Enter fitted
values in `configs/vehicle/combined_slip.yaml` under `environment.vehicle_params`,
including its existing `calibration` metadata. No vehicle constants are added to
the scenario. The paper's identified dataset/parameters, observation sample count
and normalization maxima, and unspecified PPO settings are still required for an
exact reproduction. The L-map has the reported 17 m length and 1 m width but its
corner geometry is an approximation, not the authors' original track.

Legacy MAPPO scenarios require a compatible physics/observation/action contract
before receiving a new 50-value PPO actor. Experimental scenario aliases may use
smaller worker counts or different PPO settings; only the canonical scenario
selects the paper's stated parallelism and main training settings.

## Checks and checkpoint compatibility

```bash
PYGLET_HEADLESS=true venv/bin/python -m pytest \
  tests/test_combined_slip_physics.py tests/test_combined_slip_environment.py \
  tests/test_wheel_actuators.py tests/test_friction_protocol.py \
  tests/test_speed_actions.py tests/test_frenet_vehicle_track_observation.py -q
```

The tests cover closed-form force cases, near-zero stiffness, load sensitivity,
post-peak forces, combined slip, contact dissipation for the synthetic profile,
force/moment balance, load transfer, no-grip wheel spin, numerical convergence,
state isolation, actuator response, seeded sampling, and checkpoint contracts.
They establish software behavior, not hardware calibration.

Checkpoints record the full vehicle parameters, model version, timestep, friction
protocol, and observation/action contracts. Old reduced/legacy checkpoints fail
compatibility checks, even if dimensions happen to match. Train fresh. In
particular, a path previously saved in the transfer scenario still points to an
old checkpoint and must be replaced with a compatible new run before transfer.

Frozen legacy regression trajectories in `tests/fixtures/physics_baseline` remain
historical artifacts and are not regenerated as part of the MF6.1 migration.
