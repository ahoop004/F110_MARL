# Tickets Backlog

## Ticket: Extract shared env/agent factory module
- Goal: centralize environment construction, map loading, start-pose management, and agent/wrapper instantiation so different algorithms can be plugged in easily.
- Rationale: train.py, eval.py, and main.py currently duplicate setup logic and maintain global state, making algorithm swaps brittle.
- Tasks:
  1. Map out duplicated code paths (map metadata loading, start pose adjustments, PPO config extraction).
  2. Design a factory API (e.g., `build_experiment(cfg)` returning env, agents, wrappers, helper functions).
  3. Refactor train/eval/main to call the shared factory, ensuring backwards compatibility.
  4. Add smoke tests covering factory resets and start pose randomization.
  5. Document how to extend the factory for additional algorithms.
- Status: pending.
