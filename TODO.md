# TODO

- [ ] Expand README and supporting docs to explain env/map builders, agent roster wiring, and trainer handoff (PPO/TD3/DQN) so newcomers can launch experiments without diving into code.
- [ ] Add automated tests exercising trainer/policy update loops and config parsing (builders, wrappers, reward wiring) to catch regressions beyond the current env smoke tests.
- [ ] Document RewardWrapper lifecycle expectations (reset usage, crash bookkeeping) and ensure trainers call it consistently to avoid state leakage.
- [ ] Break up F110ParallelEnv and physics helpers with clearer module boundaries or shape/type docstrings, improving readability around lidar/collision pipelines.
- [ ] Prepare CLI guidance for the trainer registry workflow once doc/tests land, including example commands for common training/eval flows.
- [ ] Relocate checkpoints/eval artifacts to an ignored outputs/ directory and add helper script to bundle config + git SHA per run.
- [ ] Document reward normalization (reward_horizon/reward_clip) behaviour and guard against accidental max_step changes across sweeps.
- [ ] Let the map validator whitelist known-good assets or tune thresholds so sweep dashboards aren't flooded with wall-band warnings.
- [ ] Consider wiring the `lidar_beams` option into PPO/TD3 wrappers (or document why they keep full scans) for consistent obs shapes.
- [ ] DQN: derive the discrete `action_set` from vehicle steering/accel bounds (`src/f110x/utils/builders.py`, `src/f110x/policies/dqn/dqn.py`) by meshing steering bins (`params['s_min']`, `params['s_max']`, `Δs`) with accel/brake bins `[-params['a_max'], 0.0, 0.4*params['a_max'], params['a_max']]` to produce 20 actions and expose a `decode_action(idx) -> (steer_delta, accel)` helper.
- [ ] DQN: add K-step action hold inside `src/f110x/envs/f110ParallelEnv.py` (track `hold_left`, replay last command until it expires, enforce steering rate limits and keep velocities within `[params['v_min'], params['v_max']]`).
- [ ] DQN: extend the observation adapter (likely `obs` wrapper in `src/f110x/wrappers/observation.py`) to append an action-mask vector keyed to the discrete table (respect hold window sizes `w_s`, `w_a`, and physics guards when steering/velocity saturate).
