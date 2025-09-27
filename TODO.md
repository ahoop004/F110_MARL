# TODO Backlog

1. Refactor the finish-line state machine in `src/f110x/envs/f110ParallelEnv.py` into a helper and add tests so the forward-velocity logic is clearer and easier to maintain.
2. Expose/document `lap_forward_vel_epsilon` in the shared config schema (and sample YAML) so headless runs can tune the new lap filter.
3. Extract the reward curriculum factory from `train.py` into a reusable utility that both training and evaluation paths can share.
4. Route episode telemetry printing in `train.py` through a configurable logger keyed off YAML settings to keep headless logging consistent.
