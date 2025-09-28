# TODO

- [ ] Expand README and supporting docs to explain env/map builders, agent roster wiring, and trainer handoff (PPO/TD3/DQN) so newcomers can launch experiments without diving into code.
- [ ] Add automated tests exercising trainer/policy update loops and config parsing (builders, wrappers, reward wiring) to catch regressions beyond the current env smoke tests.
- [ ] Document RewardWrapper lifecycle expectations (reset usage, crash bookkeeping) and ensure trainers call it consistently to avoid state leakage.
- [ ] Break up F110ParallelEnv and physics helpers with clearer module boundaries or shape/type docstrings, improving readability around lidar/collision pipelines.
- [ ] Prepare CLI guidance for the trainer registry workflow once doc/tests land, including example commands for common training/eval flows.
