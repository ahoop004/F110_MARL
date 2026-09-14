# Physics model contract and baseline

Physics defaults to `legacy_st` version 1, with frozen regression trajectories.
The opt-in `combined_slip_st` version 1 now runs through the existing racing
environment and PPO/MAPPO training path. It couples reduced nonlinear tires,
planar chassis motion, and independent wheel/steering actuators. Its parameters
are uncalibrated development estimates; this is not an MF6.1 replication or a
validated hardware model. Optional map surface metadata remains descriptive.

## Selecting existing physics

The optional selector belongs inside the existing runtime vehicle block:

```yaml
environment:
  vehicle_params:
    model: legacy_st
    model_version: 1
    # Existing numeric vehicle overrides remain here.
```

Omitting both fields preserves the historical model and defaults. The fields do
not change the integration equations, public state vectors, observations,
actions, or rewards. Existing scenario files have not been edited to add them.
Unknown models, versions, and vehicle fields fail explicitly; specifying
`mf61`, `wheel_radius`, or a tire block does not silently enable different physics.
The top-level `vehicle_params` include is not the runtime override location;
use `environment.vehicle_params`.
Adding explicit selector fields changes the resolved configuration hash, so
loading an old checkpoint with edited YAML still follows existing provenance
checks even when the selected physical equations are equivalent.

Scenario loading validates supplied values before constructing the environment.
Environment and direct `RaceCar` construction/update also validate parameters.
Numbers must be finite, dimensions/mass/inertia/acceleration limits positive,
and friction, cornering coefficients, and center-of-mass height nonnegative.
Steering/rate/speed limits must straddle zero, and steering limits must lie
inside `(-pi/2, pi/2)`. The legacy speed controller divides by `v_max` and
`-v_min`, so a forward-only policy must use the action wrapper rather than set
`v_min: 0`. Physics timesteps must be positive and finite. These checks reject
invalid configurations; they do not clip or retune valid parameters.

## Legacy state and units

The internal state is `[x, y, delta, v, psi, r, beta]`:

| Field | Meaning | Units |
|---|---|---|
| `x`, `y` | Global position | m |
| `delta` | Actual equivalent front steering angle | rad |
| `v` | Signed speed along the modeled velocity direction | m/s |
| `psi` | Global yaw | rad |
| `r` | Yaw rate | rad/s |
| `beta` | Body slip angle | rad |

Positive yaw and steering turn left; body x points forward and body y left.
For the dynamic branch, body velocity is `vx = v*cos(beta)`,
`vy = v*sin(beta)`. Below `abs(v) < 0.5 m/s`, existing kinematic equations and
public velocity conversion use no-slip motion (`vx = v`, `vy = 0`), even if
an older beta value remains in the internal state. Preserve that legacy detail.

`lf` and `lr` are center-of-mass distances to the front/rear axles (m), `h` is
center-of-mass height (m), `m` is mass (kg), and `I` is yaw inertia (kg m²).
`mu` is dimensionless. `C_Sf`/`C_Sr` are normalized linear cornering coefficients
(per radian), not a Pacejka/MF6.1 parameter set. `a_max` is m/s²; `v_switch`
is a speed-dependent acceleration-limit threshold in m/s, not a wheel-spin state.

The environment receives physical `[steering_reference_rad, speed_reference_mps]`.
Its `step` advances one physics substep; trainers implement action repeat.
The action composer can instead accept a normalized acceleration channel and
integrate the speed reference once per decision interval. That remains a
chassis-speed reference; it does not apply wheel torque or tire force.
Legacy Frenet observations estimate `omega = vx / observation_wheel_radius`.
The estimate cannot report longitudinal tire slip.

## Frozen regression captures

`tests/fixtures/physics_baseline/manifest.json` freezes the expanded scenarios,
resolved observation/reward configs, normalized command sequences, reset seed,
actual spawn poses/IDs, physical parameters, action contracts, map asset hashes,
and original run provenance. Compressed NPZ files store every substep's internal
state, global vector, composed local observations, actions, and termination flags.

| Captured scenario | Map at capture | Physics substeps | Local observation dimensions |
|---|---|---:|---|
| `ppo.yaml` | line2 | 320 | 121 |
| `ppo_lap_completion_pretrain.yaml` | circle_map | 160 | 115 |
| `ppo_lap_completion_pretrain_frenet.yaml` | circle_map | 160 | 158 |
| `ppo_lap_completion_transfer.yaml` | Spielberg_map | 160 | 158 |
| `mappo_gaplock.yaml` | line2 | 320 | 121 per trainable agent |

The transfer capture reflects the local scenario at capture time; it does not
load its configured checkpoint. Replay uses frozen configurations rather than
requiring current experiment YAMLs to remain unchanged. All cars receive stored
scripted commands, including normally fixed-policy opponents. This is a physics
and composer regression, not a policy, reward-composer, training, or dataset
benchmark. It does not establish lap completion or sim-to-real fidelity.

PPO captures exercise speeds on both sides of the 0.5 m/s branch threshold.
The short gaplock captures exercise multiple cars and action repeat. They do
not cover the full operating envelope or terminal lifecycle; existing lifecycle,
action, reward, and dataset tests remain required. Nonlinear-model checks will
need additional physical tests and held-out maneuver data.

Run from the repository root:

```bash
PYGLET_HEADLESS=true venv/bin/python scripts/physics_baseline.py check
PYGLET_HEADLESS=true venv/bin/python scripts/physics_baseline.py check --explicit-model
PYGLET_HEADLESS=true venv/bin/python -m pytest tests/test_physics_contract.py -q
```

Replay uses stored spawn plans through `reset(options={"spawn_plan": ...})`.
Both omitted and explicit legacy selection must match saved values within
`rtol=1e-6`, `atol=1e-7`, with exact shapes and lifecycle flags. Map asset changes
fail before replay. These tolerances accommodate small floating-point differences;
they are not a license to regenerate expected results after an unexplained failure.

To capture a separately reviewed baseline, choose a new directory:

```bash
PYGLET_HEADLESS=true venv/bin/python scripts/physics_baseline.py capture --output /tmp/f110_physics_candidate
```

Capture refuses to overwrite an existing directory. Compare and explain changes
before replacing committed reference artifacts. Source hashes and the original
Git commit identify the physics used for the initial capture, including its
pre-validation equations; a dirty flag alone does not identify arbitrary edits.

## What track bundles can add

Existing bundles need no new geometry for this planar physics update. Keep the
occupancy image, resolution/origin, centerline, walls, spawns, and finish line.
The image describes obstacles/free space; its pixel intensity is not friction.
A circuit name identifies a layout, not the grip of the RC test surface.

Both `MapLoader` and direct environment map loading accept an optional root-level
`surface` block in the bundle YAML. Version 1 describes a uniform surface. No
existing bundle has been edited. If grip is unknown, record only an identity and
conditions, or omit the entire block:

```yaml
surface:
  version: 1
  id: rc_test_surface
  material: unknown
  condition: unknown
  # friction omitted: no measured or assumed coefficient is claimed
```

For a documented tire/surface pair, the optional friction block has this shape.
The values below are an illustration, not a recommended grip calibration:

```yaml
surface:
  version: 1
  id: synthetic_test_surface
  material: synthetic
  condition: dry
  friction:
    mu: 0.9
    reference_tire: development_reference_tire
    relative_std: 0.02
    calibration_id: synthetic-example-v1
    calibration_status: uncalibrated
    source: "Illustrative assumption; replace with identified tire/surface data"
```

`mu` and optional `relative_std` must be finite and nonnegative. Grip requires a
reference tire, calibration ID, status (`uncalibrated` or `measured`), and source.
Missing uncertainty stays unknown; it is not zero uncertainty. `relative_std`
describes the recorded estimate and does not automatically enable training
randomization. Future experiment configuration must select that distribution.

Parsed metadata is detached from the map cache. It can be read through
`MapData.metadata["surface"]` and runtime map metadata. These are descriptive
records only at this stage: legacy physics still uses
`environment.vehicle_params.mu`. No surface field changes tire forces or local
policy observations. Existing provenance hashes include the bundle YAML, so
adding/editing this block changes the map artifact hash even before force use.
Future force integration needs an explicit choice of grip source and a check
that the configured tire matches the measurement's reference tire. Do not
multiply two absolute tire/surface friction coefficients together.

Spatial grip zones could be useful later for low-grip patches, curbs, or off-track
surfaces. Prefer polygons in world coordinates or a separate raster with explicit
resolution/origin, units, missing-cell behavior, and front/rear contact sampling.
They are not implemented in version 1; unsupported `zones` fields are rejected.
Elevation, banking, roughness, temperature, and tire wear require additional model
equations or measured dependencies before such data affects physics. Do not add
unmeasured fields to every bundle merely to make the bundles look more detailed.

## Finding vehicle data and understanding provisional values

Provisional means an explicitly labeled development assumption, not a measured
property or a value copied from the paper. It lets us test equations while data
is unavailable. It must not be used as evidence of hardware fidelity.

| Information | Where to obtain it |
|---|---|
| Chassis/drivetrain layout and gearing | Chassis model/manual and the actual assembled drivetrain; record upgrades |
| Wheel radius | Tire specification as an initial estimate; check loaded rolling distance per wheel rotation |
| Mass and axle weight distribution | Weigh the assembled car, including sensors and batteries; measure front/rear loads |
| Axle distances and center of mass | Measure wheelbase and use weight distribution; document uncertainty in height/inertia estimates |
| Steering angle and response | Steering sensor/servo telemetry versus timestamped commands; if unavailable, collect an independent angle measurement |
| Wheel speed and response | Encoder or ESC telemetry versus commands; distinguish wheel RPM from motor RPM and electrical RPM, recording gearing/pole-pair conversion where needed |
| Tire/surface grip and force curves | Fit the vehicle/tire model to synchronized acceleration, braking, turning, and combined-slip data; retain separate validation runs |

The paper identifies its actuator response from measured commands and vehicle
motion; it does not provide ready-to-use parameters for our build. Start by
identifying the chassis, tire, servo, and ESC models, then inventory available
telemetry. Reuse existing log infrastructure for synchronized commands,
steering angle, rotational speed, pose, body velocity, and yaw rate. Record units,
clock alignment, tire/surface conditions, data IDs/hashes, and fitting uncertainty.
Manufacturer values are starting estimates; loaded geometry and dynamic response
still need validation. P1 remains open until such data exists.

## Implemented actuator component (P2 development slice)

`physics.vehicle.WheelActuators` stores actual and commanded
`[steering_angle_rad, wheel_speed_rad_per_s]` separately. Its configuration is
[wheel_actuator_development.yaml](../configs/vehicle/wheel_actuator_development.yaml).
That profile is explicitly `uncalibrated`: radius 0.05 m and steering settings
mirror existing assumptions; the 0.15 s wheel-speed time constant is synthetic.
It is a component input, not a scenario include. Scenario validation rejects
standalone top-level/environment `wheel_actuators` blocks. The runtime location
is explicitly `environment.vehicle_params.wheel_actuators` in the complete
[nonlinear runtime profile](../configs/vehicle/combined_slip.yaml).

The component implements the [paper's first-order steering/wheel-speed equations](https://arxiv.org/html/2504.02420v2#S3.SS4)
with explicit reference bounds and asymmetric rate limits. It integrates a held
command analytically: a constant-rate segment, when saturated, followed by the
exponential response. This is exact for the stated actuator model, independent
of timestep partition, and adds no unmeasured transport delay.

`sample(elapsed)` returns future actuator state without mutation. The coupled
vehicle RK4 step samples at 0, dt/2, and dt and then advance once at dt; it must
not reuse the final actuator state at every intermediate tire-force evaluation.
Because the first-order actuator equations in this approximation do not depend
on chassis state, exact actuator samples can accompany RK4 chassis integration.
This replaces the earlier proposal to integrate these two states numerically.

`reset(forward_speed=...)` initializes rolling speed using the physical radius
only once. An explicit `wheel_speed=...` can initialize slip. Subsequent wheel
motion is independent of chassis speed. Reset clears old references; callers
receive state/reference copies and immutable parameter/provenance mappings.
Tests cover analytical response, asymmetric rate saturation, reverse motion,
reset/instance isolation, reference clamping, stage sampling, and timestep
partition invariance.

The actuator component drives tire forces inside `CombinedSlipVehicle`, now
attached to `RaceCar` when explicitly selected. Environment reset, terminal
handling, state adapters, and versioned action/observation/checkpoint contracts
are described below. Legacy seven-element state and existing wheel-speed
observations retain their historical meaning.

## Implemented reduced wheel/tire vehicle (P3 development slice)

`physics.vehicle.CombinedSlipVehicle` implements `combined_slip_st` version 1,
with tire model `smooth_friction_circle` and a `shared_speed_awd` drivetrain.
Its profile is [combined_slip_development.yaml](../configs/vehicle/combined_slip_development.yaml).
It includes the actuator profile, with one authoritative wheel radius. Both
calibrations are marked uncalibrated; none of the new tire coefficients are
identified hardware values or conversions of legacy `C_Sf`/`C_Sr`.

This is a reduced project model, not MF6.1 or a Pacejka implementation. The
[paper, Section III-D](https://arxiv.org/html/2504.02420v2#S3.SS4) motivates combined
longitudinal/lateral slip and first-order actuators. The smooth tire equations
below are derived for this project to enforce bounded shared grip and dissipative
contact slip. They are not equations or fitted coefficients supplied by the paper.
For context, full empirical Magic Formula models require tire coefficients and
separate pure/combined-slip relations; see the [MathWorks combined-slip model
reference](https://www.mathworks.com/help/vdynblks/ref/combinedslipwheel2dof.html).
We implement the explicit reduced equations below, not that block's MF6.2 model.

### State and contact velocities

The state snapshot is `[x, y, psi, vx, vy, r, delta, omega]` in SI units. `vx`
and `vy` are body-frame velocity components; positive y/steering/yaw turn left.
`psi` is unwrapped internally. Commands are steering radians and wheel rad/s,
not chassis speed. Independent wheel speed can differ from ground speed.

One effective wheel speed is shared by front/rear axle tires, an explicit AWD
approximation. It does not simulate differential wheel speeds or torque split.
The actual drivetrain still needs to be checked during P1 identification.
Front/rear force allocation follows contact slip and normal load; there is no
extra fixed torque/braking multiplier.

With front/rear axle distances `lf`, `lr`, the tire-frame contact velocities are:

```text
u_front = cos(delta)*vx + sin(delta)*(vy + lf*r)
v_front = -sin(delta)*vx + cos(delta)*(vy + lf*r)
u_rear  = vx
v_rear  = vy - lr*r
rolling_speed = wheel_radius * omega
```

For either axle, using explicit `speed_floor > 0`:

```text
kappa = (rolling_speed - u) / max(abs(u), abs(rolling_speed), speed_floor)
alpha = atan2(v, max(abs(u), speed_floor))
```

Positive longitudinal slip pushes in tire x; positive lateral contact velocity
produces a restoring force in negative tire y. The symmetric longitudinal
normalization supports forward/reverse motion and locked/spinning wheels without
division by zero. At rest with zero wheel speed, force is zero. There is no
kinematic switch, static-contact solver, or imposed no-slip rolling constraint.
`speed_floor` controls the creep response and is part of the experiment contract,
not a hidden numerical constant. The development value is 0.5 m/s.

### Smooth combined-slip force law

Each axle has positive normalized longitudinal and cornering stiffnesses,
`Cx` and `Cy`. They multiply normal load to obtain small-slip force slopes.
With dimensionless grip `mu >= 0` and normal load `Fz` in newtons:

```text
qx = Cx * kappa
qy = -Cy * tan(alpha)
d = hypot(qx, qy)
[Fx, Fy] = Fz * mu * tanh(d / mu) * [qx, qy] / d
```

The implementation evaluates `tan(alpha)` directly as the regularized velocity
ratio and returns zero force when `mu == 0` or `d == 0`. This preserves the
linear small-slip slopes and approaches `hypot(Fx, Fy) <= mu*Fz` smoothly at
large slip. Acceleration/braking and cornering share that force budget.
For any contact velocity, `Fx*(u - rolling_speed) + Fy*v <= 0`: tire forces
oppose relative sliding. The wheel actuator can supply mechanical work; chassis
energy alone is not expected to decrease during acceleration.

This version has monotonic saturation, not a post-peak tire-force drop. It has
one isotropic friction circle, no camber/aligning moment, and no load-sensitive
coefficient changes beyond force scaling with normal load. Full MF6.1 fitting,
tire relaxation, temperature, wear, rolling resistance, aero, grade/banking,
lateral load transfer, and wheel inertia/torque control are outside its scope.

### Simultaneous load transfer and force balance

Because each tire force is proportional to its axle normal load, we can solve
longitudinal load transfer and acceleration algebraically at every RK4 stage.
Let `A` be the front body-x force per unit front normal load (after steering
rotation), `B` the rear x force per unit rear load, and `L = lf + lr`:

```text
ax = g*(lr*A + lf*B) / (L + h*(A - B))
Fz_front = m*(g*lr - h*ax) / L
Fz_rear  = m*(g*lf + h*ax) / L
```

`g = 9.81 m/s²`. This follows from `ax = sum(Fx_body)/m` and quasi-static pitch
moment balance. It avoids lagged previous-step acceleration and iterative load
clipping. Nonpositive denominators and negative axle loads raise errors;
wheel lift is outside the planar model's validity envelope.

Rotate front tire forces into the body frame, then apply Newton–Euler equations:

```text
ay = (Fy_front_body + Fy_rear) / m
r_dot = (lf*Fy_front_body - lr*Fy_rear) / I
vx_dot = ax + r*vy
vy_dot = ay - r*vx
x_dot = vx*cos(psi) - vy*sin(psi)
y_dot = vx*sin(psi) + vy*cos(psi)
psi_dot = r
```

The diagnostic body acceleration `[ax, ay]` is inertial acceleration resolved
in body axes, not simply `[vx_dot, vy_dot]`. Longitudinal acceleration comes
entirely from tire forces; no chassis-speed PID or direct acceleration command
is present in this new component. In zero grip, wheels can spin while a resting
chassis remains at rest.

### Integration, reset, and diagnostics

Chassis integration uses RK4 with exact actuator samples at each stage and a
configured maximum integration step (development default 0.001 s). A larger
requested interval is split deterministically into equal steps no larger than
that limit. This is distinct from policy action repeat and affects computation
cost. The limit must be validated by timestep convergence for changed tire
stiffness, mass/inertia, or low-speed regularization.

State changes commit only after all stages and the final endpoint validate.
A failed step raises and leaves both chassis and actuator state unchanged;
there is no NaN freezing or arbitrary clipping of speed, slip, or yaw rate.
`reset` validates a full initial condition before replacing either subsystem.
Default wheel speed initializes forward rolling once; explicit wheel speed can
initialize slip. Reset clears previous commands. State/reference snapshots and
force/slip diagnostics do not expose internal mutable arrays.

The component diagnostics report front/rear tire-frame contact velocity, slip
ratio/angle, `[Fx, Fy, Fz]`, body acceleration, and yaw acceleration. No per-step
logging or policy features are added. With `src` on `PYTHONPATH`, a local probe is:

```python
from core.scenario import load_yaml_config
from physics.vehicle import CombinedSlipVehicle

profile = load_yaml_config("configs/vehicle/combined_slip_development.yaml")
car = CombinedSlipVehicle(profile["combined_slip_vehicle"], profile["wheel_actuators"])
car.reset(velocity=(2.0, 0.0))
car.command(steering_angle=0.15, wheel_speed=80.0)
state = car.advance(0.01)
forces = car.diagnostics()["tire_forces"]
```

Tests cover force limits and contact dissipation, load/force/moment balance,
zero grip, free inertial motion, forward/reverse transitions, turn symmetry,
steering/yaw contact geometry, stage sampling, timestep convergence, a one-minute
maneuver sequence, parameter validation, reset isolation, and failed-step rollback.
These are mathematical/numerical checks, not held-out hardware validation.

### Environment and training integration

The complete runtime profile is
[configs/vehicle/combined_slip.yaml](../configs/vehicle/combined_slip.yaml).
It supplies `environment.vehicle_params.model: combined_slip_st`, version 1,
the reduced model's complete coefficients, collision dimensions, and nested
`wheel_actuators`. Nonlinear profiles do not merge legacy default parameters;
legacy coefficients in a nonlinear profile are rejected. Only RK4 is supported
for this model. Recreate the environment for parameter/model changes; existing
runtime `update_params` remains available for legacy physics.

`RaceCar.physics_state` returns a detached eight-element snapshot
`[x,y,psi,vx,vy,r,delta,omega]`. Its seven-element `state` is a geometry compatibility
view, not the integrated state. Body-frame velocity is read directly from the
nonlinear model, including lateral velocity at low speed and in reverse. Public
poses wrap yaw; the physical integrator keeps unwrapped yaw. `get_agent_state`
adds wheel speed in metadata. `physics_diagnostics()` exposes axle forces/slip
on demand without default per-step logging or adding true grip to actor inputs.

Public `reset(options={"velocities": ...})` initializes rolling wheel speed from
the authoritative actuator radius. Reset clears actuator history and terminal
freezes. Repeated chassis-speed locking is rejected for the nonlinear model,
since it would overwrite force-derived motion. A wall collision restores the
complete pre-step pose/steering and stops chassis/wheel motion. Terminal control
latches a stationary state for crashed/truncated cars; finished cars receive the
existing linearly decreasing clearance commands, then stop. Their hulls remain
collidable. Nonterminating collision configurations can resume on later commands.
LiDAR TTC uses both body velocity components for nonlinear physics, while retaining
the existing sensor geometry and contact threshold.

Raw nonlinear actions are `[steering_reference_rad, wheel_reference_rad_per_s]`.
Bounds come from the actuator profile. The environment clips commands before
recording their reference history; nonfinite inputs fail. In the action composer:

- `speed_control: wheel_speed` maps the normalized second channel to wheel rad/s.
- `speed_control: wheel_acceleration` integrates a bounded wheel reference once per
  policy decision. `max_wheel_acceleration` and `max_wheel_deceleration` are rad/s²;
  `decision_dt = timestep * action_repeat`. References start at zero on composer
  reset and clamp without windup. A rolling spawn initializes actual actuator
  state/reference, but does not seed the policy's integrated reference.
- `prevent_reverse` retains its existing default of true; `speed_index` must be 1.
  Physical actuator rate limits apply independently on every physics substep.
- Historical `direct` and `acceleration` remain chassis-speed modes and cannot
  select nonlinear scenario physics. No extra speed PID or legacy steering delay
  is applied to the new model.

[rl_racer_simulated_wheel.yaml](../configs/observations/rl_racer_simulated_wheel.yaml)
uses `wheel_speed_source: simulated_v1`. Its 158-value layout matches the legacy
Frenet layout, but the wheel state is actual simulated rotation. Reference rate,
reference, and wheel state are already in angular units; they are not divided by
radius again. Missing/nonfinite wheel values fail instead of falling back to
`vx/radius`. The observation radius must match the physical radius. Legacy
`rolling_estimate` configurations retain their no-slip estimate and are rejected
when paired with this nonlinear Frenet configuration.

Checkpoints from PPO, MAPPO, and PPO-to-MAPPO actor initialization compare explicit
physics and resolved observation contracts before loading weights. Physics identity
includes all coefficients/calibration records, model/state layout, timestep,
action repeat, and units. The observation contract records the resolved component
configuration/normalization and sensor/preview settings. Even equal tensor sizes
cannot bypass these checks. Historical missing contracts still load into legacy
agents, but cannot load into a nonlinear agent. Cross-physics transfer and altered
grip evaluation require a future explicit compatibility policy; the ordinary
provenance-mismatch flag does not bypass physics/observation checks.

Dataset schema remains 2.0. Run/dataset/checkpoint metadata record physics,
action, and observation contracts; physical actions are wheel references for the
new model. One transition is still recorded per active agent decision, including
the final collision/time-limit transition. MAPPO retains the existing 17-values-
per-agent global vector (34 for two agents); it does not gain wheel/steering state
or ground-truth friction implicitly. Actors remain local and critics centralized.
A fuller critic state would require a separate versioned change.

Development scenarios are
[ppo_combined_slip_development.yaml](../scenarios/ppo_combined_slip_development.yaml)
and [mappo_combined_slip_development.yaml](../scenarios/mappo_combined_slip_development.yaml).
Both use explicit circle-map selection, short horizons, fresh policies, and
uncalibrated parameters. They preserve existing reward component definitions and
exclude the initial finish-line crossing from lap completion. Example:

```bash
PYGLET_HEADLESS=true venv/bin/python run.py --scenario scenarios/ppo_combined_slip_development.yaml --no-wandb --episodes 1 --quiet
```

Fixed controllers retain physical chassis-speed output internally. Nonlinear
scenarios require explicit `action_adapter: rolling_speed_to_wheel_v1`; missing
adapters fail validation. No existing map assets or training scenarios are migrated
by this update. Adapter and grip protocol details follow below.

## Research implications and next delivery

The nonlinear model changes how actions produce motion: acceleration, braking,
and cornering share tire grip, and wheel rotation can differ from chassis speed.
The Frenet layout has the same size but different wheel-state meaning. New
experiments therefore need fresh or explicitly compatible checkpoints. Current
legacy trajectory regressions preserve the baseline for future comparisons.

P2 environment integration and the PPO/MAPPO portions of P4 are implemented.
Fixed-controller adapters and seeded P5 grip protocols are also implemented.
Next add optional force/slip time-series diagnostics and design controlled P6
comparisons using matched maps, seeds, action intervals, and tuning budgets.
P1 still requires vehicle measurements and P3 still requires held-out maneuver
validation. Synthetic parameter tests and short training runs cannot establish
physical fidelity or a sim-to-real improvement.

## P0 validation record

- `venv/bin/python -m compileall -q run.py src tests scripts`: passed.
- `PYGLET_HEADLESS=true venv/bin/python -m pytest tests/ -q`: 408 passed.
- `PYGLET_HEADLESS=true venv/bin/python -m pytest tests/test_physics_contract.py -q`:
  29 passed after the final capture-checker metadata checks were added.
- Headless `run.py --scenario scenarios/ppo.yaml --no-wandb --episodes 1 --quiet
  --output-dir /tmp/f110_p0_ppo_smoke`: passed (one time-limit episode).
- Headless `run.py --scenario scenarios/mappo_gaplock.yaml --no-wandb --episodes 1
  --quiet --output-dir /tmp/f110_p0_mappo_smoke`: passed (one time-limit episode).
- `physics_baseline.py capture --decisions 8` into a new temporary directory,
  followed by `check` with omitted and explicit model selection: passed.
- Dependency guard: no forbidden imports; `rg` returned the expected no-match
  status. `git diff --check`: passed.

The first full suite run exposed an existing stale transfer assertion comparing
the Frenet transfer scenario to base pretraining and requiring Budapest. The
corrected test compares Frenet contracts and consistent user-selected map lists.
An added capture-checker assertion also caught inconsistent non-bundle map naming;
normalizing that metadata to the YAML stem fixed the checker without changing
saved trajectory arrays. Both failures were resolved and rechecked. Remaining
warnings concern existing Pydantic field metadata and the RK4 selection notice.

## Actuator/surface development validation record

- `venv/bin/python -m compileall -q run.py src tests scripts`: passed.
- `PYGLET_HEADLESS=true venv/bin/python -m pytest tests/ -q`: 454 passed,
  including omitted/explicit legacy baseline replays.
- `PYGLET_HEADLESS=true venv/bin/python -m pytest tests/test_surface_metadata.py
  tests/test_wheel_actuators.py -q`: 47 passed after adding the final oversized
  numeric metadata rejection check.
- Headless `run.py --scenario scenarios/ppo.yaml --no-wandb --episodes 1 --quiet
  --output-dir /tmp/f110_wheel_component_ppo_smoke`: passed (time-limit episode).
- Headless `run.py --scenario scenarios/mappo_gaplock.yaml --no-wandb --episodes 1
  --quiet --output-dir /tmp/f110_wheel_component_mappo_smoke`: passed (time-limit episode).
- Dependency guard found no forbidden imports; `git diff --check` and document
  link/fence checks passed. Existing Pydantic/RK4 notices remain warnings.

These validate implementation consistency, not calibrated tire forces or physical
vehicle fidelity. No existing map assets, action/observation contracts, or reward
semantics changed in this slice. Real parameter identification and integrated
vehicle/terminal/reset checks remain pending.

## Combined-slip component validation record

- `venv/bin/python -m compileall -q run.py src tests scripts`: passed.
- `PYGLET_HEADLESS=true venv/bin/python -m pytest tests/test_combined_slip_physics.py
  tests/test_wheel_actuators.py tests/test_physics_contract.py -q`: 89 passed.
- `PYGLET_HEADLESS=true venv/bin/python -m pytest tests/ -q`: 490 passed,
  including the frozen omitted/explicit legacy trajectory checks.
- `PYGLET_HEADLESS=true venv/bin/python run.py --scenario scenarios/ppo.yaml
  --no-wandb --episodes 1 --quiet --output-dir /tmp/f110_combined_slip_ppo_smoke`:
  passed (one time-limit episode).
- `PYGLET_HEADLESS=true venv/bin/python run.py --scenario scenarios/mappo_gaplock.yaml
  --no-wandb --episodes 1 --quiet --output-dir /tmp/f110_combined_slip_mappo_smoke`:
  passed (one time-limit episode).
- Dependency guard found no forbidden imports. Existing Pydantic field metadata
  and RK4 selection notices remain warnings; no validation failures occurred.

The smoke runs exercise legacy training compatibility. New-model evidence comes
from component tests, including a deterministic one-minute maneuver sequence
and timestep convergence checks. Hardware calibration and environment/training
integration remain pending; these results do not establish sim-to-real fidelity.

## Environment/PPO/MAPPO integration validation record

- `venv/bin/python -m compileall -q run.py src tests scripts`: passed.
- `PYGLET_HEADLESS=true venv/bin/python -m pytest tests/test_combined_slip_environment.py
  tests/test_combined_slip_physics.py tests/test_physics_contract.py
  tests/test_wheel_actuators.py -q`: 110 passed, including frozen legacy replays.
- `PYGLET_HEADLESS=true venv/bin/python -m pytest tests/ -q`: final run
  **510 passed, 3 failed**. The remaining failures concern concurrent legacy
  pretraining configuration edits, not nonlinear physics:
  `test_pretraining_networks_and_physical_discount_contract` expects a 256×256
  actor (and batch size 1024), while the current pretraining YAML selects 512×512
  (and batch size 2048). Two parametrizations in `tests/test_pretrained_actor.py`
  correctly reject transfer from that 512×512 PPO actor to existing 256×256 MAPPO
  recipients. Those scenario choices were preserved; reconcile recipient/source
  architectures before using that legacy transfer workflow.
- An earlier full run exposed a legacy observation-composer test double without
  the new contract attribute. Limiting the required attribute to nonlinear runs
  fixed that regression; the affected CLI evaluation test passed afterward.
  The stale one-worker Frenet assertion was updated to the user's existing
  eight-worker setting; no scenario worker count was changed by this update.
- Both legacy smoke commands completed successfully (time-limit episodes):
  `PYGLET_HEADLESS=true venv/bin/python run.py --scenario scenarios/ppo.yaml
  --no-wandb --episodes 1 --quiet --output-dir /tmp/f110_nonlinear_integration_legacy_ppo`
  and the equivalent `scenarios/mappo_gaplock.yaml` command with output directory
  `/tmp/f110_nonlinear_integration_legacy_mappo`.
- New PPO training completed with
  `PYGLET_HEADLESS=true venv/bin/python run.py --scenario scenarios/ppo_combined_slip_development.yaml
  --no-wandb --episodes 1 --quiet --output-dir /tmp/f110_nonlinear_ppo_integration_v2
  --dataset-dir /tmp/f110_nonlinear_ppo_dataset`.
  New MAPPO training completed with the corresponding
  `scenarios/mappo_combined_slip_development.yaml`, output directory
  `/tmp/f110_nonlinear_mappo_integration`, and dataset directory
  `/tmp/f110_nonlinear_mappo_dataset`.
- Dataset audit passed: PPO recorded 198 decisions; MAPPO recorded 126 for the
  car that crashed and 250 for the car that reached the time limit. Both datasets
  closed as complete with schema 2.0, finite arrays, unique agent/episode/decision
  identities, consistent successive observations, terminal rows, no post-terminal
  decisions, and resolved wheel-action/observation/physics contracts.
- CLI evaluation loaded each new run's `checkpoint_ep000000.pt` using its matching
  scenario, `--eval --episodes 1 --eval-episodes 1 --no-wandb --quiet`, and separate
  output directories `/tmp/f110_nonlinear_ppo_eval` and
  `/tmp/f110_nonlinear_mappo_eval`. Both reported matching provenance and completed
  without a compatibility override. These minimally trained policies crashed;
  this validates the execution path, not racing competence or physical accuracy.
- Compilation, dependency guard (expected no-match status), `git diff --check`,
  and documentation link/fence checks passed. Existing Pydantic/RK4 warnings remain.

The first development smoke inherited initial-finish-crossing lap counting.
The new scenarios now explicitly exclude that crossing, and both algorithms
were rerun with the corrected development configuration. Existing map bundles,
legacy rewards, and the frozen baseline fixtures were not changed.

## Fixed-controller adapters and episode friction protocols

Fixed policies opt into nonlinear physics with an agent-level
`action_adapter: rolling_speed_to_wheel_v1`. AgentFactory still constructs the
original controller; `WheelReferenceAdapter` in the existing action module
converts its `[steering_rad, speed_mps]` output to `[steering_rad, speed_mps/radius]`.
The physical actuator radius is authoritative. The adapter clips to actuator
reference bounds, rejects malformed/nonfinite outputs, and delegates reset and
environment injection to the controller. If an action space is injected, the
controller receives speed bounds in m/s. Conversion assumes a requested rolling
speed, not zero actual tire slip or guaranteed chassis-speed tracking. Controller
settings, algorithms, and legacy scenarios are not retuned by the adapter.

Episode friction is opt-in under `environment.friction`:

```yaml
friction:
  version: 1
  scope: shared
  train:
    mode: uniform
    low: 0.8
    high: 1.1
  eval:
    mode: grid
    values: [0.8, 0.95, 1.1]
```

These example bounds are synthetic development values, not measured tire/surface
uncertainty or the paper's relative perturbation model. `fixed` uses an explicit
`mu`; `uniform` uses finite nonnegative `low < high`; `grid` requires a nonempty
list of finite nonnegative values. Evaluation rejects `uniform`, and both train
and eval protocols are required when a friction block is supplied. Without the
block, fixed nominal vehicle grip remains the default. Map surface metadata does
not automatically override this experiment configuration.

All cars receive the same coefficient for the whole episode, including fixed
opponents. Independent tire variation is not supported in this version. A new
reset reconstructs the nonlinear component with the selected coefficient;
nominal vehicle config/checkpoint parameters remain unchanged. No mid-step
parameter mutation or direct trainer access to physics internals is needed.

`EpisodeFriction` uses its own `default_rng(SeedSequence([seed, 0x46524943]))`.
It consumes no spawn, map, sensor, global NumPy, or policy RNG draws. Public
`reset(seed=s)` restarts this stream; `reset()` advances its recorded draw counter.
A grid uses `(seed + draw) % len(values)` so consecutive explicit evaluation
seeds cover the configured values. Use enough evaluation seeds to cover the grid;
record per-value results as well as aggregates. The setup mode selects the train
or eval protocol even for environments without map-bundle splits.

Nonlinear MAPPO reads the critic input dimension before resetting the environment.
It no longer consumes an unrecorded friction sample in a sizing reset. Legacy
MAPPO retains its historical reset sequence. PPO workers have distinct explicit
seeds and episode IDs, and their sampled parameters arrive with transition
records at the parent process.

Episode information and immutable global-state metadata contain `physics`, with
nominal/actual `mu`, protocol, phase, seed, RNG stream, draw index, and grid index.
These facts are not appended to actor observations or the critic vector. The
resolved train/eval protocol is part of strict checkpoint identity; train and
evaluation share that declaration, while their effective phase is recorded per
episode. Changing a grid or distribution in YAML does not silently bypass
checkpoint compatibility checks.

`PhysicsEpisodeHook` and the dataset writer reuse `PhysicsEpisodeLog` to write
`physics_episodes.jsonl`, one row per episode shared across agents. The record
includes the dataset episode ID and map ID; worker IDs remain part of episode
identity. Conflicting physics within one episode raises an error. Dataset schema
2.0 and transition arrays are unchanged; no dummy decisions are introduced.
This logging consumes ordinary transition records, so nonlinear training retains
them even without dataset recording. Optional time-series tire-force/slip logs
remain a separate task. CLI evaluation reports include per-episode physics, and
checkpoint-selection history includes the evaluated grid values and seeds.

New development scenarios:

- [PPO with randomized grip](../scenarios/ppo_combined_slip_friction_development.yaml)
- [PPO versus adapted FTG](../scenarios/ppo_combined_slip_vs_ftg_development.yaml)
- [MAPPO with randomized grip](../scenarios/mappo_combined_slip_friction_development.yaml)

These short scenarios test execution and provenance. They are not calibrated
comparisons or evidence that friction randomization improves racing performance.

### Adapter/friction validation record

- `venv/bin/python -m compileall -q run.py src tests`: passed.
- `PYGLET_HEADLESS=true venv/bin/python -m pytest tests/test_friction_protocol.py -q`:
  20 passed. Coverage includes physical command conversion/reset for FTG, pure
  pursuit, Stanley, hybrid PP/FTG, and kinematic MPC; invalid protocols; shared
  grip; zero-grip force behavior; independent/reproducible RNGs; unchanged
  spawn/LiDAR draws; evaluation grids; actor isolation; and episode logging.
- Full `PYGLET_HEADLESS=true venv/bin/python -m pytest tests/ -q`:
  **533 passed, 3 failed**. The same existing 512×512 PPO versus 256×256 MAPPO
  transfer configuration mismatch described above remains. No additional
  implementation regression was reported.
- Legacy PPO and MAPPO one-episode headless smoke runs passed using
  `scenarios/ppo.yaml` and `scenarios/mappo_gaplock.yaml` with
  `--no-wandb --episodes 1 --quiet` and separate `/tmp/f110_friction_legacy_*`
  output directories; their previous rewards/time-limit outcomes were retained.
- New PPO-versus-FTG and MAPPO two-episode smoke runs passed with datasets. A
  two-worker PPO run completed five episodes and checkpoint selection across
  three fixed grid values. Commands used the new scenarios, `--no-wandb --quiet`,
  explicit `--episodes`, `--output-dir`, and `--dataset-dir`; the worker run also
  used `--num-envs 2`.
- Run and dataset episode sidecars matched exactly, with one row per recorded
  episode and worker identity preserved. The corrected MAPPO run starts at draw
  zero and matches the single-worker PPO grip sequence under seed 42. The initial
  smoke revealed the sizing-reset draw consumption; removing that reset for
  nonlinear MAPPO and rerunning verified the fix.
- CLI evaluation of `/tmp/f110_friction_workers/best_model.pt` with the matching
  PPO-versus-FTG scenario, `--eval --eval-episodes 3 --episodes 5 --num-envs 2`,
  reported matching provenance and recorded `[0.8, 0.95, 1.1]` as actual grip.
  Selection history recorded its separate fixed seeds/grid values. Short-run
  evaluation crashes are policy outcomes, not numerical failures.
- After the MAPPO sizing-reset correction, targeted verification with
  `PYGLET_HEADLESS=true venv/bin/python -m pytest tests/test_friction_protocol.py
  tests/test_combined_slip_environment.py tests/test_physics_contract.py
  tests/test_mappo_terminal_handling.py -q` passed: **95 tests**.
- Final compilation, dependency guard, `git diff --check`, and documentation
  link/fence checks passed. Existing Pydantic/RK4 warnings remain.
