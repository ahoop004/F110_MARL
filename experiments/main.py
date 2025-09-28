#!/usr/bin/env python3
import os
os.environ.setdefault('PYGLET_HEADLESS', 'true')
import yaml
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent / 'src'))

import pyglet
pyglet.options['headless'] = True
pyglet.options['shadow_window'] = False
pyglet.options['vsync'] = False

try:
    import wandb
except ImportError:  # optional dependency
    wandb = None

import train as train_module
import eval as eval_module

def _wandb_init(cfg, mode):
    wandb_cfg = cfg.get("main", {}).get("wandb", {})
    enabled = wandb_cfg if isinstance(wandb_cfg, bool) else wandb_cfg.get("enabled", False)
    if not enabled or wandb is None:
        return None

    project = wandb_cfg.get("project", "f110-ppo") if isinstance(wandb_cfg, dict) else "f110-ppo"
    entity = wandb_cfg.get("entity") if isinstance(wandb_cfg, dict) else None

    return wandb.init(project=project, entity=entity, config=cfg, reinit=True, tags=[mode])


def _log_train_results(run, results, ppo_id, gap_id):
    if run is None:
        return
    for idx, totals in enumerate(results, 1):
        payload = {
            "train/episode": idx,
            f"train/return_{ppo_id}": totals.get(ppo_id, 0.0),
        }
        if gap_id is not None:
            payload[f"train/return_{gap_id}"] = totals.get(gap_id, 0.0)
        run.log(payload)


def _log_eval_results(run, results):
    if run is None:
        return
    for res in results:
        payload = {"eval/episode": res["episode"]}
        for key, value in res.items():
            if key.startswith("return_"):
                payload[f"eval/{key}"] = value
        run.log(payload)


def main():
    cfg_path = Path("configs/config.yaml")
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)

    main_cfg = cfg.get("main", {})
    mode = main_cfg.get("mode", "train").lower()
    checkpoint_dir = Path(cfg["ppo"].get("save_dir", "checkpoints"))
    checkpoint_name = cfg["ppo"].get("checkpoint_name", "ppo_best.pt")
    checkpoint_path = checkpoint_dir / checkpoint_name

    wandb_run = _wandb_init(cfg, mode)

    if mode == "train":
        episodes = cfg["ppo"].get("train_episodes", 10)
        results = train_module.run_training(train_module.CTX, episodes=episodes)
        _log_train_results(wandb_run, results, train_module.PPO_AGENT, train_module.GAP_AGENT)
        train_module.CTX.ppo_agent.save(str(checkpoint_path))

    elif mode == "eval":
        episodes = cfg["ppo"].get("eval_episodes", 5)
        # Rebuild evaluation context so explicit checkpoint overrides take effect
        eval_ctx = eval_module.create_evaluation_context(cfg_path)
        results = eval_module.evaluate(eval_ctx, episodes=episodes)
        _log_eval_results(wandb_run, results)

    elif mode == "train_eval":
        train_episodes = cfg["ppo"].get("train_episodes", 10)
        results = train_module.run_training(train_module.CTX, episodes=train_episodes)
        _log_train_results(wandb_run, results, train_module.PPO_AGENT, train_module.GAP_AGENT)
        train_module.CTX.ppo_agent.save(str(checkpoint_path))

        eval_ctx = eval_module.create_evaluation_context(cfg_path)
        eval_episodes = cfg["ppo"].get("eval_episodes", 5)
        results = eval_module.evaluate(eval_ctx, episodes=eval_episodes)
        _log_eval_results(wandb_run, results)

    else:
        raise ValueError(f"Unsupported mode '{mode}'. Expected train/eval/train_eval.")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
