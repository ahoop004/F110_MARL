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

## Current gaps

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

- [ ] Capture the commit, expanded configs, observation/action contracts, seeds,
  maps, spawn plans, and scripted-action trajectories for existing PPO/MAPPO.
- [ ] Include `ppo_lap_completion_pretrain.yaml`,
  `ppo_lap_completion_pretrain_frenet.yaml`, and the existing track-transfer
  workflow in the compatibility audit.
- [ ] Define an explicit model selector and version in environment vehicle
  configuration. Omitted selection must continue to use current physics.
- [ ] Define state coordinates, sign conventions, SI units, wheel/axle speed
  representation, and the mapping into existing public state fields.
- [ ] Decide whether a shared effective wheel speed or separate front/rear axle
  speeds match the target drivetrain; document the approximation.
- [ ] Specify proposed parameters: rolling radius, tire coefficients, grip,
  actuator time constants, drive/braking distribution, and parameter provenance.
- [ ] Validate finite values, positive physical quantities, coefficient domains,
  and incompatible model/config combinations before simulation begins.

Exit gate: old scenarios reproduce baseline trajectories; the new model and
its state/action contract are documented before implementation.

## P1 - Collect and identify vehicle, tire, and actuator parameters

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

- [ ] Add wheel angular speed independently of chassis speed for the new model.
- [ ] Implement steering response `delta_dot = (delta_ref - delta) / T_delta`
  and wheel-speed response `omega_dot = (omega_ref - omega) / T_omega`, with
  explicit calibrated limits and any separately justified transport delay.
- [ ] Keep actual and commanded steering/wheel speeds distinct. Integrate
  actuator state at the physics timestep, including RK4 intermediate states.
- [ ] Define rolling-start initialization through public reset/spawn options;
  clear actuator history consistently on reset and preserve terminal behavior.
- [ ] Verify step response, steady-state tracking, saturation, reset isolation,
  and repeatability across multiple cars and concurrent environments.

Exit gate: wheel motion can differ from chassis motion and follows the defined
actuator response. Keep this intermediate model out of learning comparisons
until tire-force coupling is complete.

## P3 - Implement nonlinear tires and combined-slip vehicle dynamics

Primary files: `src/physics/dynamic_models.py` and `src/physics/vehicle.py`.
Extract a focused tire helper only if the implemented equations justify it.

- [ ] Select and document either MF6.1 for paper replication or a named reduced
  nonlinear model with combined-slip coupling for an initial approximation.
  Identify the equation source and exact subset implemented.
- [ ] Compute front/rear contact velocities in the tire frame, including yaw
  rate and front steering; derive slip angles and longitudinal slip ratios.
- [ ] Define stable slip behavior at rest, during braking to rest, and in reverse.
  Document low-speed regularization and any kinematic blending explicitly.
- [ ] Compute front/rear normal loads using existing mass/geometry and a
  consistent acceleration/load-transfer treatment; handle invalid loads clearly.
- [ ] Evaluate longitudinal and lateral tire forces with nonlinear saturation
  and combined-slip coupling. Acceleration/braking must reduce remaining
  cornering capacity according to the selected tire model.
- [ ] Apply drive/braking distribution consistently with the chosen wheel-state
  approximation. Rotate tire forces into the vehicle frame and derive chassis
  acceleration and yaw acceleration from force/moment balance.
- [ ] Remove direct commanded chassis acceleration only for the new model;
  retain the existing baseline path.
- [ ] Verify force signs, zero-slip behavior, saturation, straight-line symmetry,
  left/right turn symmetry, wheelspin, braking slip, and grip sensitivity.
- [ ] Check finite outputs and timestep convergence through low-speed crossings,
  aggressive maneuvers, and long rollouts. State the model's validity envelope.

Exit gate: deterministic physical checks pass, and held-out maneuver validation
shows where the model is accurate and where its approximation breaks down.

## P4 - Integrate actions, observations, checkpoints, and datasets

Primary areas: `src/wrappers/actions/composer.py`,
`src/wrappers/observations/track.py`, `src/env/`, existing checkpoint handling,
`src/training/hooks.py`, and `src/replay/dataset_writer.py`.

- [ ] Add an explicit wheel-reference action mode through the existing action
  composer. Specify whether the reference derivative uses rad/s² or tire
  circumferential m/s², and convert with the configured physical radius.
- [ ] Preserve old vehicle-speed action semantics. Apply reference integration
  once per intended interval and actuator integration per physics substep;
  verify behavior with action repeat and reference clamping.
- [ ] Expose actual wheel speed through a versioned observation option; retain
  the old `vx / radius` estimate for legacy observations/checkpoints.
- [ ] Use one authoritative physical rolling radius for new-model conversions;
  validate observation configuration against it and document normalization.
- [ ] Record layout, units, physics model, calibration ID, and action/observation
  versions in checkpoint provenance; reject incompatible loads by default.
- [ ] Keep any transfer across physics or observation semantics an explicit
  experiment with documented initialization scope, not a silent resume.
- [ ] Preserve actor-local observations and MAPPO global-state compatibility;
  version any intentionally expanded critic input rather than changing its size
  implicitly. Do not leak randomized ground-truth grip into actor observations.
- [ ] Log diagnostic wheel speed, slip, forces, and sampled physics parameters
  through optional hooks/metadata without adding noisy default per-step logs.
- [ ] Preserve one complete `TransitionRecord` per active agent decision for
  PPO and MAPPO, including normalized/physical action and required lifecycle,
  map/spawn/episode/step/agent fields and global state when available.
- [ ] Preserve the dataset schema unless a deliberate migration is necessary;
  document any PPO/MAPPO logging differences and test terminal transitions.

Exit gate: action units, observation meaning, checkpoint loading, and recorded
transitions agree end to end for both trainable and fixed-policy agents.

## P5 - Add reproducible friction randomization

- [ ] Keep fixed calibrated parameters as the default for deterministic checks.
- [ ] Add opt-in episode-level friction randomization through the public reset
  configuration path, using an explicit RNG stream independent of sensor noise.
- [ ] Define physically valid sampling bounds and rejection/clipping behavior.
  Record nominal grip, distribution, seed, and actual sampled grip per episode.
- [ ] Start with friction-only randomization. The paper's relative perturbation
  level of 0.02 is an experimental starting point, not a measured uncertainty
  for our platform or a universal optimum.
- [ ] Define which surface variation is shared across cars and which tire
  variation is car-specific; avoid accidental baseline/opponent asymmetry.
- [ ] Keep evaluation on fixed parameter grids and held-out seeds, separate from
  training randomization. Verify repeated resets reproduce sampled parameters.
- [ ] Add broader parameter randomization only as a separate calibrated ablation.

Exit gate: train/evaluation physics distributions are explicit and reproducible,
including vectorized PPO and multi-agent runs.

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
  regression, no >5% memory increase without justification) on unchanged physics.
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

## Suggested first delivery

Complete P0: preserve scripted baseline trajectories and specify model selection,
physical state, units, parameter provenance, and compatibility behavior. Begin
P1 data collection in parallel with local implementation preparation; do not
block numerical tests on unavailable hardware or present provisional parameters
as a calibrated model.
