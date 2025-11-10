# F110_MARL

Lightweight research stack for adversarial F1TENTH racing.

- **Scenarios** live under `scenarios/*.yaml`. Each file wires the simulator, agents, rewards, and logging. Tweak these when you want new spawn curricula, reward shaping, or TD3/PPO configs.
- **Training** is launched with `python run.py --scenario <file>`. The runner builds the PettingZoo environment, attaches the configured agents (trainable or scripted), and streams rollouts to wandb/log files.
- **TD3 agent** supports Adam/AdamW optimizers, cosine/step schedulers, PER, and the usual TD3 knobs (noise, policy delay, target smoothing). All of these are exposed via the scenario YAML.
- **Trajectories** are exported to `plots/<scenario>/<run_id>/paths_<run_id>.csv` along with a JSON config snapshot. Use `plots_viewer.ipynb` to overlay paths, render heat maps, or inspect episode causes.
- **Defender policies** include upgraded Follow-The-Gap heuristics with curriculum support so you can stage increasingly difficult blockers without touching training code.

