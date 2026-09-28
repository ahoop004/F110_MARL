# Matched 1v1 attack training

`scenarios/mappo_1v1_attack.yaml` fine-tunes the full pretrained actor.
`scenarios/mappo_1v1_attack_lora.yaml` freezes that actor and trains rank-4 LoRA
adapters plus the exploration parameters. Both train a fresh centralized critic
and use the same single-learner MAPPO implementation, fixed racing MPC target,
maps, seeds, rewards, limits and collection/evaluation budgets.

Both initialize the actor from `outputs/pretrain/best_model2.pt`. The original
PPO critic is not transferred because MAPPO's critic consumes the joint state.
The default training budget is 20 million joint decisions; each episode lasts
at most 1,200 physics steps (60 simulated seconds), ending earlier on ego failure.

```bash
venv/bin/python run.py --scenario scenarios/mappo_1v1_attack.yaml --no-wandb
venv/bin/python run.py --scenario scenarios/mappo_1v1_attack_lora.yaml --no-wandb
```

For parallel collection, add `--num-envs 8` (and optionally
`--set experiment.num_workers=4`). Both arms use the same
`training_defaults.rollout_steps_per_env: 256` in parallel mode. Use the same
collection settings and seeds for both arms. This does not reproduce the serial
rollout size, so compare runs within the same collection protocol.

For render profiling, MPC optimizations and a 128-core launch/benchmark recipe,
see [the 1v1 performance review](1v1_mpc_performance.md).

Evaluate with the corresponding scenario and saved checkpoint:

```bash
venv/bin/python run.py --scenario scenarios/mappo_1v1_attack_lora.yaml \
  --eval --checkpoint outputs/YOUR_RUN/best_model.pt --eval-episodes 8 --no-wandb
```

Saved checkpoints include the actor base and adapters; evaluation does not need
the original pretraining file. Add `--render` to visualize either arm locally.

## Equal vehicle and command limits

The simulator applies one shared vehicle model, footprint, mass, tire model,
friction draw and actuator configuration to both cars. These remain compatible
with the pretrained checkpoint.

| Quantity | Ego and MPC target |
|---|---|
| Forward rolling-speed reference | 0–5 m/s |
| Rolling-speed reference acceleration / braking | −5 to +5 m/s² |
| Wheel-speed reference used while driving | 0–100 rad/s |
| Wheel-reference acceleration / braking | −100 to +100 rad/s² |
| Steering reference and physical angle bounds | −0.4189 to +0.4189 rad |
| Physical steering rate bounds | −3.2 to +3.2 rad/s |
| Physical wheel actuator rate bounds | −10,000 to +10,000 rad/s² |
| Steering / wheel actuator time constants | 0.5 s / 0.15 s |
| Physics and decision interval | 0.05 s |
| Wheel radius | 0.05 m |

The MPC's `max_acceleration: 5` limits both acceleration and braking symmetrically;
its optimization bounds prohibit negative speed references. The ego integrates
its action with `max_wheel_acceleration: 100`, `max_wheel_deceleration: 100` and
`prevent_reverse: true`. These are the same limits after multiplying by wheel radius.

The MPC's extra steering-reference smoother is set to 17 rad/s, exceeding the
entire allowed reference span per decision (16.756 rad/s). This makes it
nonbinding, matching the pretrained ego's ability to request any allowed angle
each decision. Both physical steering actuators still obey the same ±3.2 rad/s
bound and lag. Scenario validation rejects mismatched command limits.

The shared underlying wheel model retains its pretrained −400 rad/s lower
bound, but both controllers restrict **commands** to nonnegative wheel speeds.
These are rolling-reference bounds, not clamps on measured chassis velocity or
acceleration: tire slip and actuator response determine actual motion. Changing
physics/action limits requires a compatible source checkpoint.

## Observations and transfer

The original 158 inputs are retained exactly: 108 normalized LiDAR beams and
50 vehicle/track Frenet values. Five inputs are appended in this order:
`delta_s`, `delta_d`, `delta_vs`, `delta_vd`, `present`, with fixed scales
30 m, 5 m, 20 m/s, 20 m/s and 1. Target sensing uses simulator state, including
outside LiDAR range. `delta_s` is signed shortest wrapped track distance.

The first actor layer gains five zero columns. All original actor parameters
are copied from the source. LoRA residuals initially contribute zero, so both
arms start with the same action distribution for the same driving observations.
The LoRA base's new zero columns remain frozen; its first-layer adapter learns
to use the target inputs. Both critics train on the same joint-state contract.

## Crash events and target respawning

The episode starts at a random track location, with both cars stationary and
the target approximately 3 m ahead. A target wall collision or boundary exit
causes an immediate target-only respawn, uniformly sampling forward arc gaps
from 3 to 10 m and rejecting candidates without vehicle/wall clearance. The
target is track-aligned and restarts at 2 m/s. Ego state and ongoing episode
progress are preserved. The controller plan, collision memory and target
progress tracker are reset; relocation earns no distance or lap progress.

All gaps must remain below half the track length, including across the finish
seam, so the actual target-relative observation reports a positive forward gap.
Nearest-centerline recovery in existing scenarios is unchanged.

Ego collision or boundary exit ends the episode. Contact between the cars
normally produces a mutual collision in this simulator; this task therefore
encourages forcing errors through positioning and pressure. A mutual crash is
never an attack success.

For credit, ego must have been moving forward faster than 0.5 m/s and within
3 m of the target in relative Frenet distance during the preceding second.
Ego must then survive for another 0.5 s after the target crash. Every qualifying
crash can earn credit, including multiple crashes within one episode. Pending
credit is cancelled by ego failure and is not carried across episode resets;
credit still pending at the time limit is unawarded. Recent proximity is an
interaction proxy, not proof of causation.

## Reward and checkpoint selection

- Confirmed attack success: +10.
- Ego collision or boundary exit: −20, including mutual crashes.
- Signed approach-distance change: weight 0.1, capped at ±1 m per step.
- Signed change in the target's normalized distance from track center while
  nearby: weight 0.5.
- Signed ego distance progress: 0.01 per metre.

Approach/edge history resets at crashes so teleportation creates no shaping
bonus. Neither staying near the target nor remaining parked earns a standing
proximity bonus. The legacy once-per-episode crash rewards are not used.

Evaluation records raw target crashes, eligible crashes, survival-qualified
successes, successes per simulated minute, and ego crash rate (including
boundary exits). The primary checkpoint score is
`(successes - 2 * ego_failures) / scheduled_evaluation_minutes`.
An early ego crash keeps its full scheduled time budget in that denominator,
so early termination cannot inflate the primary score. Ties prefer fewer ego
failures, then successes per actual simulated minute, then earned progress.
The failure weight matches the 20:10 terminal reward ratio.

Evaluate the unadapted policy and target's ordinary driving as baselines before
interpreting increased target crashes as effective attacks. The setup and smoke
tests establish correct execution; attack effectiveness needs training and
held-out evaluation.
