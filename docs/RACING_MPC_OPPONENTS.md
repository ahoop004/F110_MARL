# Racing MPC opponents

Scenario settings are now inline. Select parameter choices in the canonical YAML
file or use `--set KEY=YAML`; there are no scenario inheritance files in `configs`.

`racing_mpc` is a non-neural controller registered with `AgentFactory`.
The active fixed-opponent MAPPO workflows configure two identical MPC opponents
inline in each scenario. `scenarios/mappo_2v2_race.yaml` is the finite-race workflow. Their settings match
`configs/controllers/racing_mpc.yaml`, including the 5 m/s speed cap, in both
scratch and pretrained arms.

The shared vehicle wheel-reference ceiling is 100 rad/s, equivalent to 5 m/s
at the 0.05 m wheel radius. Learner action integrators and MPC wheel adapters
both enforce this limit. Active pretraining, transfer, self-play and MPC render
scenarios use the same ceiling. Tire slip still determines actual chassis speed.
Corner-speed planning may select lower speeds; the shared value is a maximum.

To select another shared ceiling, pass `--max-speed 5`, `10`, `15`, or `20` to
`run.py`, or set `environment.max_speed` in the scenario YAML. The preset sets
both learner wheel-reference bounds and every MPC's `params.max_speed`; at the
current radius the four choices correspond to 100/200/300/400 rad/s. It works
for single-car pretraining, PPO pursuit, MAPPO teams and fixed-controller renders.
Use the same preset for pretraining, transfer and checkpoint evaluation.

Changing the wheel-reference ceiling changes the checkpoint physics/action-bound
contract. Existing checkpoints with the old 400 rad/s ceiling are rejected by
the strict loader. New pretraining uses the matching limit; old checkpoints need
retraining or an explicit migration before use with these scenarios.

For interactive inspection, use the [render scenarios](../scenarios/render/README.md):
solo MPC, two MPC cars against two hybrids, and a slower-car passing demo.

## Controller and information contract

The controller optimizes six steering/speed command knots over 30 decisions
(1.5 simulated seconds at the current 20 Hz rate). L-BFGS-B minimizes contour
error, heading error, clearance costs, and speed error while rewarding forward
arc-length progress. It warm-starts from the preceding plan and tries additional
left/right initial guesses around traffic. It executes only the first command,
then replans from the next measured state.
Additional pass-side starts run when the warm plan is unsafe or periodically
while moving slowly, rather than on every decision with traffic in sensor range.
Each solve has both iteration and objective-evaluation budgets (SciPy may finish
the current numerical gradient past the evaluation budget). These are work
limits, not a guaranteed wall-clock deadline.

The prediction state includes longitudinal/lateral velocity and yaw rate, so
existing sideways motion and yaw momentum persist through replanning. The model
uses the vehicle profile's MF6.1 combined-slip tire forces at **static axle loads**;
it omits the plant's dynamic load-transfer solve. It uses midpoint integration
with speed-dependent substeps (at most 4 ms near rest, 20 ms at speed) and
exact steering/wheel actuator responses.
Dimensions, tire coefficients and actuator parameters come from the environment.
Simulator rollouts, not predicted costs, establish driving quality.

The controller reads the current episode's friction from public global-state
metadata on every decision, including after resets. This is privileged grip
information, in addition to its perfect traffic sensing. Curvature-based speed
targets reserve 20% of nominal lateral grip (`grip_utilization: 0.8`) and include
a backward braking pass. Costs also discourage sideslip and excessive lateral
acceleration. These targets and penalties are approximate, not stability guarantees.

Steering references change by at most `max_steering_reference_rate: 1.5` rad/s
(default 0.075 rad per 50 ms decision). The same limiter applies inside prediction
and to the executed command. Steering-change costs include the transition from
the measured previous reference to the first command and all subsequent ramps.

The occupancy map supplies signed clearance. A grid covering the full rectangular
body is checked at every predicted decision. Vehicle avoidance uses conservative
oriented ellipses enclosing both footprints, with a growing prediction margin.
Candidates that violate the configured **wall margin**, rather than just overlap
a wall, are excluded when a feasible candidate exists. Traffic margins are already
included in the ellipses and are not added twice. If no candidate meets these
requirements, the controller chooses the braking candidate with the best minimum
safety slack, considering planned, held, and straightening steering. Braking still
obeys the reference-rate limits. This sampled, approximate prediction and fallback
do not guarantee collision avoidance.

`last_plan.predicted_safety_slack_m` reports the minimum slack across the wall
margin and inflated traffic envelopes (negative means a violation). This replaces
`predicted_clearance_m`, which mixed raw wall clearance with traffic-envelope slack.
Diagnostics also include `friction_mu`, `target_speed_mps` and the eight-state
trajectory (`x, y, yaw, vx, steering, rolling wheel speed, vy, yaw rate`).

The output is physical steering radians and rolling-speed reference m/s. The
existing `rolling_speed_to_wheel_v1` adapter converts speed to wheel rad/s.
The reference changes at most 5 m/s per second by default, matching the current
learners' 100 rad/s² wheel-reference limit with a 0.05 m radius.

Traffic sensing uses perfect current simulator poses and body velocities for
other cars within 10 m, including stationary crashed cars. Velocities are rotated
to world coordinates and extrapolated at constant velocity for the horizon.
There is no occlusion, measurement noise, future-action access, or opponent
reward access. This is privileged sensing compared with the hybrid's LiDAR and
must be disclosed in comparisons. Neither controller is a learned policy.

## Reproducible checks

```bash
PYGLET_HEADLESS=true OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  venv/bin/python scripts/benchmark_racing_opponents.py \
  --seeds 10042 10043 10044 --output outputs/racing_opponents.jsonl

# Two identical candidate controllers alongside two fixed hybrid cars.
PYGLET_HEADLESS=true OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  venv/bin/python scripts/benchmark_racing_opponents.py \
  --controllers racing_mpc --modes pair --seeds 10042 10043 10044 \
  --output outputs/racing_opponents_pair.jsonl

# Repeat on lower grip (applied to every car in evaluation).
PYGLET_HEADLESS=true OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python3 scripts/benchmark_racing_opponents.py \
  --controllers racing_mpc --modes solo --friction-mu 0.8 \
  --seeds 10044 --laps 1 --max-steps 5000 \
  --output outputs/racing_mpc_low_grip.jsonl

# Deliberately start behind a slower car; distinguish passes from passing wrecks.
PYGLET_HEADLESS=true OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  venv/bin/python scripts/benchmark_racing_opponents.py \
  --modes passing --max-steps 1200 --seeds 10042 10043 10044 \
  --output outputs/racing_opponents_passing.jsonl
```

Files are created exclusively to prevent overwriting prior results. Each race is
flushed immediately. Reports include seed, configuration/hash, source hash,
completion, collisions, measured lap times, active-car passes, fallback count,
and mean/p95 controller latency. Compilation and map initialization are excluded
from decision latency. Pair tests continue until both candidate cars terminate.
Other tests stop when the focal car terminates. A bounded passing test can end
at its time limit despite successfully passing; it is not a lap-completion test.

The benchmark loads its historical hybrid baseline explicitly from
`configs/controllers/hybrid_pp_ftg.yaml`, independently of the experiment matrix.
The MPC defaults to 5 m/s and the historical hybrid to 2.5 m/s. Results
compare those complete configurations, including differing information access;
they do not isolate the algorithm at equal speed and sensing limits. The
benchmark defaults to nominal grip; `--friction-mu` overrides evaluation grip.
Robustness to randomized training grip and aggressive learned opponents needs
separate testing.

Before interpreting a large study, check repeatable clean completion across more
starts and randomized grip on both training maps, traffic handling against learned
policies, two-controller decision cost, and pair completion. Keep held-out maps
out of this tuning process. Freeze the same opponent configuration for every
scratch/pretrained arm, and keep earlier hybrid-opponent results separate.

## Shared 5 m/s cap validation

The [5 m/s report](benchmarks/racing_mpc_shared_5mps_validation.json) records six
one-lap checks across Circle and Budapest: solo and paired MPC at nominal grip
(seed 10043), plus solo at friction 0.8 (seed 10044). Every focal MPC completed
without collision; both MPC cars finished each paired race. Nominal solo lap
times were 70.7 s on Circle and 82.9 s on Budapest. These are single-seed checks.

All 47 focused MPC/cap/integration tests passed. Transfer, LoRA and speed-action
checks produced 69 passes and one pre-existing evaluation-count failure, reproduced
against the original code. Four previously confirmed observation-layout failures
were excluded. The shared cap test exercises learner integrator saturation and
braking, MPC adapter saturation, and consistency across active workflows.

## Stability revision validation at the previous 3.5 m/s cap

The [stability report](benchmarks/racing_mpc_stability_validation.json) records
commands, configuration/source hashes and all eight simulator runs before the
shared speed cap was raised to 5 m/s:

- Both maps: one clean solo lap and both MPC cars finishing a paired one-lap race
  at nominal grip, seed 10043.
- Both maps: one clean solo lap at friction 0.8, seed 10044.
- Both maps: an active slower car passed during a 60-second check, seed 10044,
  without an MPC collision. The slower hybrid later crashed on Budapest; this
  does not establish collision-free behavior or fault for every traffic participant.
  Both hybrid traffic cars also crashed in the Budapest paired check.
- No focal MPC brake fallbacks in these runs.

These are bounded checks with one seed per configuration. Mean focal decision
latency was 12.9–23.0 ms and p95 was 16.2–53.3 ms across these runs;
concurrent local workloads and paired-controller costs preclude a deadline claim.
All 45 focused MPC/integration tests passed. Broader relevant checks returned
181 passes and four pre-existing observation-layout failures, reproduced unchanged
against the original commit; details are in the report.

## Historical initial local validation

The [saved report](benchmarks/racing_mpc_initial_validation.json) predates the
slip-aware prediction, steering-continuity and hard-wall-margin changes above.
It contains historical bounded-solver results and configuration/source hashes;
its timings and completion results do not qualify the revised controller. At that
time the full suite passed 668 tests, with 48 focused tests after tuning.

| Map / test | Hybrid result | Racing MPC result | MPC mean timed lap |
|---|---|---|---|
| Circle solo, seed 10042 | 3 clean laps; 216.37 s mean | 3 clean laps | 100.08 s |
| Circle traffic, seed 10042 | Collision before a full lap | 3 clean laps | 103.67 s |
| Budapest solo, seed 10042 | Collision before a full lap | 3 clean laps | 113.90 s |
| Budapest traffic, seed 10042 | Collision before a full lap | 3 clean laps | 113.85 s |

With two identical MPC cars plus two hybrid cars (seed 10043), **both MPC cars
completed three laps without collisions on both maps**. Separate 60-second
passing checks overtook an active slower car on both maps without the MPC car
colliding. Some hybrid cars subsequently crashed, including the Budapest passing
test's slower car; these records do not establish fault or collision-free driving
for every traffic participant.

Final focal-controller p95 latency was approximately 10–22 ms in the full races
and 23–27 ms in the passing checks. The complete paired loops averaged about
23–24 ms per simulated decision including both MPC controllers, hybrid cars,
simulation, and logging; this average is not a joint p95 or a deadline guarantee.

These are initial checks with one seed per full-race configuration, at nominal
grip and with distinct named spawns (`allow_reuse: false`). The matrix currently
uses `allow_reuse: true`, which changes the seeded spawn-sampling stream even
though overlapping choices are rejected. Broader seeds, randomized grip, and
the exact matrix spawn policy remain qualification work before a large study.
MPC is now enabled in the active experiment scenarios. Bounded integration checks
with the actual training spawn and friction settings do not replace this broader
qualification. Render comparisons intentionally retain hybrid traffic.
