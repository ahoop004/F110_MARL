#!/usr/bin/env python3
import os
import sys
from typing import Optional

_RENDER_FLAG = False
_EP_OVERRIDE: Optional[int] = None
_EVAL_EP_OVERRIDE: Optional[int] = None

argv = list(sys.argv)
i = 0
while i < len(argv):
    arg = argv[i]
    if arg == "--render":
        _RENDER_FLAG = True
        argv.pop(i)
        continue
    if arg.startswith("--episodes"):
        if arg == "--episodes" and i + 1 < len(argv):
            _EP_OVERRIDE = int(argv[i + 1])
            argv.pop(i + 1)
            argv.pop(i)
            continue
        else:
            _, value = arg.split("=", 1)
            _EP_OVERRIDE = int(value)
            argv.pop(i)
            continue
    if arg.startswith("--eval-episodes"):
        if arg == "--eval-episodes" and i + 1 < len(argv):
            _EVAL_EP_OVERRIDE = int(argv[i + 1])
            argv.pop(i + 1)
            argv.pop(i)
            continue
        else:
            _, value = arg.split("=", 1)
            _EVAL_EP_OVERRIDE = int(value)
            argv.pop(i)
            continue
    i += 1

sys.argv = argv

if not _RENDER_FLAG:
    env_flag = os.environ.get("F110_RENDER", "").lower()
    _RENDER_FLAG = env_flag in {"1", "true", "yes", "on"}

if _RENDER_FLAG:
    os.environ["PYGLET_HEADLESS"] = "false"
else:
    os.environ.setdefault("PYGLET_HEADLESS", "true")

import yaml
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(_ROOT / 'src'))

import pyglet
pyglet.options['headless'] = not _RENDER_FLAG
pyglet.options['shadow_window'] = _RENDER_FLAG
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


def _log_train_results(run, results, ppo_id=None, gap_id=None):
    if run is None:
        return
    for idx, totals in enumerate(results, 1):
        payload = {"train/episode": idx}
        for aid, value in totals.items():
            payload[f"train/return_{aid}"] = value
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
        train_ctx = train_module.create_training_context(cfg_path)
        if _RENDER_FLAG:
            train_ctx.render_interval = 1
            train_ctx.env.render_mode = "human"
        episodes = _EP_OVERRIDE or cfg["ppo"].get("train_episodes", 10)
        results = train_module.run_training(train_ctx, episodes=episodes)
        gap_id = train_ctx.opponent_ids[0] if train_ctx.opponent_ids else None
        _log_train_results(wandb_run, results, train_ctx.ppo_agent_id, gap_id)
        train_ctx.ppo_agent.save(str(train_ctx.best_path))

    elif mode == "eval":
        episodes = _EVAL_EP_OVERRIDE or cfg["ppo"].get("eval_episodes", 5)
        # Rebuild evaluation context so explicit checkpoint overrides take effect
        eval_ctx = eval_module.create_evaluation_context(cfg_path)
        if _RENDER_FLAG:
            eval_ctx.env.render_mode = "human"
        results = eval_module.evaluate(eval_ctx, episodes=episodes, force_render=_RENDER_FLAG)
        _log_eval_results(wandb_run, results)

    elif mode == "train_eval":
        train_ctx = train_module.create_training_context(cfg_path)
        if _RENDER_FLAG:
            train_ctx.render_interval = 1
            train_ctx.env.render_mode = "human"
        train_episodes = _EP_OVERRIDE or cfg["ppo"].get("train_episodes", 10)
        results = train_module.run_training(train_ctx, episodes=train_episodes)
        gap_id = train_ctx.opponent_ids[0] if train_ctx.opponent_ids else None
        _log_train_results(wandb_run, results, train_ctx.ppo_agent_id, gap_id)
        train_ctx.ppo_agent.save(str(train_ctx.best_path))

        eval_ctx = eval_module.create_evaluation_context(cfg_path)
        if _RENDER_FLAG:
            eval_ctx.env.render_mode = "human"
        eval_episodes = _EVAL_EP_OVERRIDE or cfg["ppo"].get("eval_episodes", 5)
        results = eval_module.evaluate(eval_ctx, episodes=eval_episodes, force_render=_RENDER_FLAG)
        _log_eval_results(wandb_run, results)

    else:
        raise ValueError(f"Unsupported mode '{mode}'. Expected train/eval/train_eval.")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
