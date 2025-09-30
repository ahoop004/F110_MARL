# TODO


- [ ] Create shared CLI utilities (yaml loading, logging, manifest helpers) for scripts/run.py/map_validator.py.

- [ ] Relocate checkpoints/eval artifacts to an ignored outputs/ directory and add helper script to bundle config + git SHA per run.
    - [ ] Add manifest writer (CSV/JSON) capturing run_id, algo, map, seed, output_dir, git_sha.
    - [x] Update `.gitignore` for outputs/ and retrofit existing scripts to use the new directory.


- [ ] Resource planning for massive sweeps.
    - [ ] Draft compute concurrency plan (runs per GPU/CPU, reserved debug slot).
    - [ ] Specify storage budget + retention policy (outputs/<algo>/<map>/<seed>/ layout).
    - [ ] Add submission helper (scripts/launch_array.py or similar) with retry/status tracking.
    - [ ] Set up monitoring/alerting (W&B dashboards, log watchdog).


- [ ] Extend attacker algorithm roster.
    - [ ] Stand up SAC baseline alongside PPO/TD3/DQN.
    - [ ] Evaluate multi-agent options (MADDPG/MATD3) for co-learning defenders.
    - [ ] Prototype Rainbow-style upgrades for discrete attackers (distributional Q, noisy nets).
    - [ ] Recurrent RL (RNN-PPO, DRQN) 

