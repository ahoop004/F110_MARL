# F110 MARL Physics Update Plan

## Objective and scope

Add an optional, calibrated single-track physics model that represents tire
saturation, combined longitudinal/lateral slip, independent wheel speed, and
actuator response. Preserve the existing physics as the reproducible baseline.
Deliver each phase as a separate, reviewable behavior change.

Reference: [On learning racing policies with reinforcement learning,
arXiv:2504.02420v2](https://arxiv.org/html/2504.02420v2), especially Sections
III-D, III-E, and IV-A/D. The paper uses MF6.1 combined-slip tires and first-order
steering/wheel-speed actuators. It does not supply a complete numerical tire
parameter set for our vehicle. The implementation steps below are project
recommendations; a simplified tire model must not be labeled an MF6.1 replication.

Completed engineering history lives in [done.md](done.md). Existing performance
workflows live in [docs/PERFORMANCE.md](docs/PERFORMANCE.md). Outstanding work
from the previous roadmap is retained in the backlog below; this rewrite does
not mark it complete.

## Current gaps in the training environment

Opt-in nonlinear physics is integrated into PPO/MAPPO. The default legacy model
retains the following approximations for reproducible comparisons.

- `src/physics/dynamic_models.py` uses a linear lateral tire formulation with
  `mu`, `C_Sf`, and `C_Sr`; longitudinal acceleration is applied directly.
- Mass, yaw inertia, axle distances, center-of-mass height, and approximate
  longitudinal load-transfer terms already exist and should be reused.
- There is no independent wheel-speed state, nonlinear tire saturation, or
  shared longitudinal/lateral tire-force constraint.
- `src/physics/vehicle.py` already has a steering command delay and proportional
  steering/speed control. Calibrate or replace those paths explicitly rather
  than adding duplicate actuator delays.
- `src/wrappers/observations/track.py` computes `omega = vx / wheel_radius`.
  This is a no-slip estimate, not measured or simulated wheel rotation.
- The Frenet acceleration action integrates a vehicle-speed reference. A wheel
  speed-reference action needs an explicit units and semantics contract.

## Compatibility and research gates

- Keep `run.py`, the current setup/composer/trainer path, and the public `reset`,
  `step`, `get_global_state`, and `get_agent_state` contracts.
- Keep PPO/MAPPO supported; reject unsupported algorithms and roles explicitly.
  Preserve MAPPO CTDE and independent per-agent reward composers.
- Keep known-good scenarios and old checkpoint interpretation unchanged. Add
  separate physics scenarios with unique names and explicit maps.
- Version physics, observation semantics, action semantics, and normalization.
  Equal tensor dimensions alone do not establish checkpoint compatibility.
- Keep map files, spawn semantics, reward definitions, and opponent controller
  settings fixed while isolating physics effects.
- Use explicit seeds, resolved parameters, model versions, and calibration-data
  identifiers in run metadata. Keep W&B optional and respect `--no-wandb`.
- Reuse existing modules and dependencies; ask before adding a dependency.
- Review the current README findings before starting new learning comparisons.

## P0 - Freeze the baseline and define the model contract

- [x] Capture the commit, expanded configs, observation/action contracts, seeds,
  maps, spawn plans, and scripted-action trajectories for existing PPO/MAPPO.
- [x] Include `ppo_lap_completion_pretrain.yaml`,
  `ppo_lap_completion_pretrain_frenet.yaml`, and the existing track-transfer
  workflow in the compatibility audit.
- [x] Define an explicit model selector and version in environment vehicle
  configuration. Omitted selection must continue to use current physics.
- [x] Define state coordinates, sign conventions, SI units, wheel/axle speed
  representation, and the mapping into existing public state fields.
- [x] Choose an initial wheel-speed representation and document the drivetrain
  assumptions that P1 must verify against hardware.
- [x] Specify proposed parameters: rolling radius, tire coefficients, grip,
  actuator time constants, drive/braking distribution, and parameter provenance.
- [x] Validate finite values, positive physical quantities, coefficient domains,
  and incompatible model/config combinations before simulation begins.

Completed: [model/state contract and capture protocol](docs/PHYSICS_MODEL.md).
The five frozen cases replay under omitted and explicit `legacy_st` selection.
Nonlinear training requires its own complete, explicitly selected parameter
profile and wheel-action contracts. The shared-speed AWD design is provisional
pending P1 hardware identification. See the contract for capture coverage and limitations.

## P1 - Collect and identify vehicle, tire, and actuator parameters

Development support is available; hardware identification remains pending.
See [data sources and provisional-value guidance](docs/PHYSICS_MODEL.md).

- [x] Add a labeled uncalibrated actuator development profile with explicit units.
- [x] Validate optional uniform tire/surface metadata in bundle YAMLs, with tire
  identity and calibration provenance; keep missing grip unknown and legacy
  physics unchanged. No existing map bundles need migration.

- [ ] Measure effective rolling radius, mass, wheelbase, front/rear static weight
  distribution, and center-of-mass location; estimate yaw inertia with a
  documented method and uncertainty.
- [ ] Record tire type/condition, track surface, drive layout, gearing, and the
  conversion from measured motor RPM to wheel angular speed.
- [ ] Collect synchronized commands, measured steering angle, wheel/motor speed,
  position, heading, body-frame velocity, and yaw rate. Include acceleration,
  braking, steady turns, and combined braking/acceleration through turns.
- [ ] Fit steering and wheel-speed response constants from command/response
  data, distinguishing transport delay from first-order lag and saturation.
- [ ] Identify front/rear longitudinal and lateral tire behavior, including
  small-slip stiffness, peak force, post-peak response, load dependence, and
  combined-slip coupling. Do not reinterpret `C_Sf`/`C_Sr` as MF6.1 coefficients.
- [ ] Fit nominal tire/surface grip and quantify uncertainty; mark provisional
  simulator parameters as uncalibrated when hardware data is unavailable.
- [ ] Split identification and validation data by driving sequence/session.
  Check multi-step trajectory error on held-out maneuvers, not just one-step fit.
- [ ] Store units, fit method, parameter bounds, data IDs/hashes, and validation
  errors alongside the calibration configuration.

Exit gate: a traceable parameter set exists. Synthetic parameters may support
implementation tests but cannot support a sim-to-real accuracy claim.

## P2 - Add wheel state and calibrated actuator dynamics

Primary files: `src/physics/vehicle.py`, `src/physics/dynamic_models.py`, and the
existing environment state/reset adapters under `src/env/`.

- [x] Implement an independent steering/wheel actuator component with actual and
  commanded states, explicit reference bounds, and asymmetric rate limits.
- [x] Implement first-order response with exact held-command integration and
  non-mutating intermediate-stage sampling for future coupled RK4 integration.
- [x] Verify analytical step response, saturation, reverse motion, timestep
  partition invariance, component reset behavior, and independent instances.
- [ ] Calibrate the time constants/limits against hardware data and introduce
  transport delay only if measurements justify it.
- [x] Couple the component into the new vehicle model with P3 tire forces;
  sample actuator state at each chassis integrator stage and advance only once.
- [x] Integrate independent wheel speed into the new component's eight-state
  vehicle model, preserving the legacy model's state ordering.
- [x] Adapt the new component into RaceCar and public environment state views.
- [x] Wire rolling-start initialization through public reset/spawn options and
  verify terminal handling and concurrent-environment isolation end to end.

The nonlinear model now runs through RaceCar, public reset/spawn, collisions,
and terminal handling. P2 engineering integration is complete; hardware actuator
calibration remains pending. Repeated chassis-speed locking is rejected for this
model. Existing legacy scenarios retain their original physics.

## P3 - Implement nonlinear tires and combined-slip vehicle dynamics

Primary files: `src/physics/dynamic_models.py` and `src/physics/vehicle.py`.
The reduced force law lives in `src/physics/tire_models.py`; equations and
limitations are documented in [the model contract](docs/PHYSICS_MODEL.md).

- [x] Select and document either MF6.1 for paper replication or a named reduced
  nonlinear model with combined-slip coupling for an initial approximation.
  Identify the equation source and exact subset implemented.
- [x] Compute front/rear contact velocities in the tire frame, including yaw
  rate and front steering; derive slip angles and longitudinal slip ratios.
- [x] Define stable slip behavior at rest, during braking to rest, and in reverse.
  Document low-speed regularization and any kinematic blending explicitly.
- [x] Compute front/rear normal loads using existing mass/geometry and a
  consistent acceleration/load-transfer treatment; handle invalid loads clearly.
- [x] Evaluate longitudinal and lateral tire forces with nonlinear saturation
  and combined-slip coupling. Acceleration/braking must reduce remaining
  cornering capacity according to the selected tire model.
- [x] Apply drive/braking distribution consistently with the chosen wheel-state
  approximation. Rotate tire forces into the vehicle frame and derive chassis
  acceleration and yaw acceleration from force/moment balance.
- [x] Remove direct commanded chassis acceleration only for the new model;
  retain the existing baseline path.
- [x] Verify force signs, zero-slip behavior, saturation, straight-line symmetry,
  left/right turn symmetry, wheelspin, braking slip, and grip sensitivity.
- [x] Check finite outputs and timestep convergence through low-speed crossings,
  aggressive maneuvers, and long rollouts. State the model's validity envelope.

- [ ] Validate against held-out hardware maneuver data after P1 identification.

The coupled `combined_slip_st` / `smooth_friction_circle` component passes
deterministic physical checks, including a one-minute maneuver sequence. It is
an uncalibrated reduced approximation, not MF6.1. The numerical portion is
complete; the held-out-data exit gate remains open. PPO/MAPPO development runs
are available, but calibrated learning comparisons remain outstanding.

## P4 - Integrate actions, observations, checkpoints, and datasets

Primary areas: `src/wrappers/actions/composer.py`,
`src/wrappers/observations/track.py`, `src/env/`, existing checkpoint handling,
`src/training/hooks.py`, and `src/replay/dataset_writer.py`.

- [x] Add an explicit wheel-reference action mode through the existing action
  composer. Specify whether the reference derivative uses rad/s² or tire
  circumferential m/s², and convert with the configured physical radius.
- [x] Preserve old vehicle-speed action semantics. Apply reference integration
  once per intended interval and actuator integration per physics substep;
  verify behavior with action repeat and reference clamping.
- [x] Expose actual wheel speed through a versioned observation option; retain
  the old `vx / radius` estimate for legacy observations/checkpoints.
- [x] Use one authoritative physical rolling radius for new-model conversions;
  validate observation configuration against it and document normalization.
- [x] Record layout, units, physics model, calibration ID, and action/observation
  versions in checkpoint provenance; reject incompatible loads by default.
- [x] Keep any transfer across physics or observation semantics an explicit
  experiment with documented initialization scope, not a silent resume.
- [x] Preserve actor-local observations and MAPPO global-state compatibility;
  version any intentionally expanded critic input rather than changing its size
  implicitly. Do not leak randomized ground-truth grip into actor observations.
- [ ] Log diagnostic wheel speed, slip, forces, and sampled physics parameters
  through optional hooks/metadata without adding noisy default per-step logs.
- [x] Preserve one complete `TransitionRecord` per active agent decision for
  PPO and MAPPO, including normalized/physical action and required lifecycle,
  map/spawn/episode/step/agent fields and global state when available.
- [x] Preserve the dataset schema unless a deliberate migration is necessary;
  document any PPO/MAPPO logging differences and test terminal transitions.

PPO/MAPPO contracts now agree end to end. New checkpoints reject mismatched
physics, wheel semantics, and observation normalization even at equal dimensions.
Dataset schema remains 2.0; metadata records the physical action units and resolved
contracts. Cross-physics checkpoint transfer is rejected until an explicit transfer
policy is implemented. The 17-value-per-agent critic vector remains unchanged.

- [x] Add and validate explicit wheel-command adapters for fixed-policy agents;
  opt in with `action_adapter: rolling_speed_to_wheel_v1`.

Fixed-controller conversion and per-episode grip logging are implemented.
Optional time-series force/slip diagnostics remain before detailed model analysis.

## P5 - Add reproducible friction randomization

- [x] Keep fixed nominal parameters as the default; hardware calibration remains P1.
- [x] Add opt-in episode-level friction randomization through the public reset
  configuration path, using an explicit RNG stream independent of sensor noise.
- [x] Define physically valid sampling bounds and rejection/clipping behavior.
  Record nominal grip, distribution, seed, and actual sampled grip per episode.
- [x] Start with friction-only randomization. The paper's relative perturbation
  level of 0.02 is an experimental starting point, not a measured uncertainty
  for our platform or a universal optimum.
- [x] Define which surface variation is shared across cars and which tire
  variation is car-specific; avoid accidental baseline/opponent asymmetry.
- [x] Keep evaluation on fixed parameter grids and held-out seeds, separate from
  training randomization. Verify repeated resets reproduce sampled parameters.
- [ ] Add broader parameter randomization only as a separate calibrated ablation.

Implemented: explicit train/eval grip protocols, shared grip across all cars,
independent seeded sampling, and deterministic evaluation grids. Run/dataset
`physics_episodes.jsonl` records nominal/sample values, seed, draw, and protocol.
Vectorized PPO and MAPPO development runs verify recording and isolation. Synthetic
ranges are not calibrated uncertainty; per-car tire variation remains unsupported.

## P6 - Validate learning and transfer under the new physics

- [ ] Add separate scenarios for new-physics pretraining, track transfer, and
  MAPPO/fixed-controller comparisons. Preserve the existing scenario files.
- [ ] Define controlled arms: legacy physics, calibrated nonlinear physics,
  and calibrated nonlinear physics with friction randomization. Keep policies,
  rewards, maps, opponents, decision intervals, and budgets matched where possible.
- [ ] Separate physics changes from action/observation changes with ablations;
  record unavoidable contract differences rather than claiming identical inputs.
- [ ] Evaluate fixed-policy controllers and learned agents under the same test
  physics, maps, spawn plans, and friction grid. Freeze controller tuning or give
  each comparison arm the same documented tuning budget.
- [ ] Train fresh policies for primary comparisons. Evaluate old checkpoints
  separately as transfer experiments, with contract compatibility checked first.
- [ ] Report completion, collision, lap time, progress, slip/traction diagnostics,
  actuator tracking error, and dispersion across multiple seeds.
- [ ] Keep checkpoint selection separate from final held-out evaluation. Report
  track transfer separately from transfer across physics or to real hardware.
- [ ] Measure physics substeps/s, decisions/s, and memory using fixed work after
  correctness passes; report the fidelity cost without reducing physics silently.

Exit gate: results identify which model/action/observation changed and distinguish
simulation performance, robustness, and any measured sim-to-real evidence.

## Validation sequence for implementation

Run focused physical/contract checks after each phase, then the repository gates:

```bash
venv/bin/python -m compileall -q run.py src tests
PYGLET_HEADLESS=true venv/bin/python -m pytest tests/ -q
rg "stable_baselines3|from gymnasium|from pettingzoo" run.py src configs scenarios
PYGLET_HEADLESS=true venv/bin/python run.py --scenario scenarios/ppo.yaml --no-wandb --episodes 1 --quiet
PYGLET_HEADLESS=true venv/bin/python run.py --scenario scenarios/mappo_gaplock.yaml --no-wandb --episodes 1 --quiet
```

The dependency guard expects no matches (`rg` exit code 1). Add headless smoke
commands for the new physics scenarios when those files exist. Compare legacy
trajectories against P0 and test new-model physics against physical expectations
and held-out data; identical trajectories across different models are not a gate.
Report commands, failures, likely causes, limitations, and experiment-validity
implications for each implementation delivery.

## Retained backlog outside the physics update

Re-audit these items against current code before implementation; unchecked items
from the old roadmap remain pending and may have advanced independently.

### Performance validation

- [ ] Establish fixed-work benchmark variance sufficient to detect a 5% change.
- [ ] Finish end-to-end checks for `complete_4.yaml`, `complete_4_frenet.yaml`,
  and `complete_4_frenet_neighbors.yaml`, including at least five fixed seeds,
  multiple map cycles/updates, and dataset-enabled/disabled measurements.
- [ ] Compare legacy-path observations (115/158/173 dimensions), transitions,
  lifecycle outcomes, rewards, datasets, function counts, throughput, and memory.
- [ ] Confirm W&B-disabled/dataset-disabled benchmarks do no external I/O.
- [ ] Record results in `docs/PERFORMANCE.md`; evaluate the prior optimization
  targets (25% throughput improvement for `complete_4`, no >5% Frenet throughput
  regression, no >5% memory increase without justification) on unchanged physics.5,
  Do not apply those equivalence targets to the new physics model.

Historical variable-episode baseline (2026-09-02, Quadro RTX 5000): 4,497
policy decisions, 8,993 physics substeps, 59.02 s wall time, 1,834,940 KiB peak
RSS, approximately 76 decisions/s and 152 substeps/s. Use the existing fixed-work
benchmark scripts for new comparisons; this old timing alone is insufficient.

### Architecture and pretraining

- [ ] Recheck the scan-isolation test's assumption about the pretraining map.
- [ ] Add immutable observation layout/normalization metadata and checkpoint
  provenance; coordinate with P4 rather than duplicating contract machinery.
- [ ] Add a backward-compatible configurable MLP factory, followed by residual
  MLP and LiDAR fusion CNN, with independent actor/critic config and CTDE checks.
- [ ] Audit PPO/MAPPO datasets before behavior-cloning or masked-LiDAR encoder
  pretraining; version datasets and support encoder-only save/load and freezing.
- [ ] Strengthen full/actor/encoder-only checkpoint modes, compatible PPO-to-MAPPO
  transfer, architecture metadata, and legacy checkpoint validation.
- [ ] Compare architecture/initialization arms under matched online budgets,
  held-out evaluation, multiple seeds, and explicit parameter/latency reporting.
- [ ] Consider a small LiDAR Transformer after CNN validation; defer recurrent
  policies until hidden-state, rollout, batching, and reset contracts are tested.
- [ ] Consider external PyTorch/Hugging Face weights only for compatible input
  modalities, with pinned revisions/hashes, offline support, and approval before
  adding dependencies. Run focused, full, and headless architecture checks.

### Other research and tooling

- [ ] Calibrate race duration from controller timing distributions.
- [ ] Finish per-agent/team terminal-reason logging across console, CSV, and W&B.
- [ ] Complete deterministic multi-map lap validation and multi-seed comparisons.
- [ ] Add stronger fixed-policy opponents through AgentFactory after evaluation
  is reliable; keep algorithm additions separate from this physics work.
- [ ] Consider `--verbose` and `--debug` CLI flags as a separate tooling change.

## Next delivery

Fixed-controller wheel-command adapters and P5 seeded friction protocols are
implemented, with separate PPO/MAPPO development scenarios and episode records.
Next add optional force/slip time-series diagnostics and prepare controlled P6
comparisons. P1 hardware data and held-out model validation remain outstanding.
See the [model contract](docs/PHYSICS_MODEL.md) for units and compatibility rules.
