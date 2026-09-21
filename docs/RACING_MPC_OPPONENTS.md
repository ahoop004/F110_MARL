# Candidate racing MPC opponents

`racing_mpc` is an opt-in, non-neural controller registered with `AgentFactory`.
Its parameters live in `configs/controllers/racing_mpc.yaml`. The A/B scenarios
still use their original hybrid opponents; no experiment controller was replaced.

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

The reduced prediction model includes steering response, steering-rate limits,
wheel response, limited acceleration, and a friction-limited bicycle yaw rate.
Dimensions, wheel radius, steering bounds and time constants come from the
environment vehicle profile. The prediction model is **not** the plant's MF6.1
tire model. Simulator rollouts, not predicted costs, establish driving quality.

The occupancy map supplies signed clearance. A grid covering the full rectangular
body is checked at every predicted decision. Vehicle avoidance uses conservative
oriented ellipses enclosing both footprints, with a growing prediction margin.
Candidates with predicted overlap are excluded when a feasible candidate exists;
otherwise the controller requests bounded braking. This sampled, approximate
prediction and fallback do not guarantee collision avoidance.

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

The candidate defaults to 3.5 m/s and the historical hybrid to 2.5 m/s. Results
compare those complete configurations, including differing information access;
they do not isolate the algorithm at equal speed and sensing limits. The
benchmark evaluates nominal grip. Robustness to randomized training grip and
aggressive learned opponents needs separate testing.

Before promotion into the experiment matrix, require repeatable clean completion
on both training maps, successful collision-free traffic handling, acceptable
two-controller decision cost, and no regression in pair completion. Keep held-out
maps out of this tuning process. Freeze the accepted opponent configuration for
every scratch/pretrained arm.

## Initial local validation

The [saved report](benchmarks/racing_mpc_initial_validation.json) contains the
final bounded-solver results and configuration/source hashes. The full suite
passed 668 tests; 48 focused tests passed after the final controller tuning.

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
No experiment scenarios have been switched by this implementation.
