#!/usr/bin/env python3
import os
import sys
import tempfile
import random
from typing import Any, Dict, Optional

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

import numpy as np
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


def _deep_update(base: Dict, updates: Dict) -> Dict:
    for key, value in updates.items():
        if key.startswith("_") or key in {"wandb_version"}:
            continue
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _seed_everything(seed: int) -> None:
    """Seed python, numpy, and torch (if available) for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - optional CUDA path
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover - torch optional
        pass


def _apply_run_seed(cfg: Dict[str, Any]) -> Optional[int]:
    """Override config seed from RUN_SEED environment variable if provided."""

    run_seed_value = os.environ.get("RUN_SEED")
    if run_seed_value is None:
        return None

    try:
        seed = int(run_seed_value)
    except (TypeError, ValueError):
        print(f"[WARN] Ignoring invalid RUN_SEED='{run_seed_value}'")
        return None

    env_cfg = cfg.setdefault("env", {})
    env_cfg["seed"] = seed
    cfg.setdefault("main", {})["seed"] = seed
    return seed

def _wandb_init(cfg, mode):
    wandb_cfg = cfg.get("main", {}).get("wandb", {})
    enabled = wandb_cfg if isinstance(wandb_cfg, bool) else wandb_cfg.get("enabled", False)
    if not enabled or wandb is None:
        return None

    project = wandb_cfg.get("project", "f110-ppo") if isinstance(wandb_cfg, dict) else "f110-ppo"
    entity = wandb_cfg.get("entity") if isinstance(wandb_cfg, dict) else None
    group = wandb_cfg.get("group") if isinstance(wandb_cfg, dict) else None

    tags = [mode]
    if isinstance(wandb_cfg, dict):
        extra_tags = wandb_cfg.get("tags")
        if isinstance(extra_tags, (list, tuple)):
            tags.extend(str(tag) for tag in extra_tags if tag)
        elif isinstance(extra_tags, str) and extra_tags:
            tags.append(extra_tags)
    # Deduplicate while preserving order
    tags = list(dict.fromkeys(tags))

    return wandb.init(project=project, entity=entity, config=cfg, group=group, reinit=True, tags=tags)


def _log_train_results(run, results, ppo_id=None, gap_id=None, tb_writer=None):
    if run is None and tb_writer is None:
        return

    for idx, record in enumerate(results, 1):
        if isinstance(record, dict):
            episode = int(record.get("episode", idx))
            payload: Dict[str, Any] = {"train/episode": episode}

            returns = record.get("returns")
            if isinstance(returns, dict):
                for aid, value in returns.items():
                    payload[f"train/return_{aid}"] = float(value)
            else:
                try:
                    for aid, value in record.items():
                        if isinstance(value, (int, float)):
                            payload[f"train/return_{aid}"] = float(value)
                except AttributeError:
                    pass

            if ppo_id and f"train/return_{ppo_id}" in payload:
                payload["train/return_attacker"] = payload[f"train/return_{ppo_id}"]
            if gap_id and f"train/return_{gap_id}" in payload:
                payload["train/return_defender"] = payload[f"train/return_{gap_id}"]

            scalar_keys = {
                "steps",
                "collisions_total",
                "defender_survival_steps",
            }
            for key in scalar_keys:
                value = record.get(key)
                if value is not None:
                    payload[f"train/{key}"] = float(value)

            bool_keys = {"success", "defender_crashed", "attacker_crashed"}
            for key in bool_keys:
                if key in record and record[key] is not None:
                    payload[f"train/{key}"] = int(bool(record[key]))

            for key, value in record.items():
                if key.startswith("collision_count_") or key.startswith("collision_step_"):
                    if value is not None:
                        payload[f"train/{key}"] = float(value)

            for key, value in record.items():
                if key.startswith("avg_speed_") and value is not None:
                    payload[f"train/{key}"] = float(value)

            for key, value in record.items():
                if key.startswith("reward_component_") and value is not None:
                    payload[f"train/{key}"] = float(value)

            if record.get("reward_mode"):
                payload["train/reward_mode"] = record["reward_mode"]
            if record.get("cause"):
                payload["train/cause"] = record["cause"]
        else:
            payload = {"train/episode": idx}
            try:
                for aid, value in record.items():
                    payload[f"train/return_{aid}"] = value
            except AttributeError:
                pass

        if run is not None:
            run.log(payload)
        if tb_writer is not None:
            step = payload["train/episode"]
            for key, value in payload.items():
                if key == "train/episode":
                    continue
                if isinstance(value, (int, float)):
                    tb_writer.add_scalar(key, value, step)


def _log_eval_results(run, results, tb_writer=None):
    if run is None and tb_writer is None:
        return

    success_count = 0
    defender_survival_total = 0.0
    defender_survival_count = 0
    for res in results:
        payload = {"eval/episode": res["episode"]}
        for key, value in res.items():
            if isinstance(value, (int, float)):
                payload[f"eval/{key}"] = value
        if run is not None:
            run.log(payload)
        if tb_writer is not None:
            step = payload["eval/episode"]
            for key, value in payload.items():
                if key == "eval/episode":
                    continue
                if isinstance(value, (int, float)):
                    tb_writer.add_scalar(key, value, step)

        defender_crashed = res.get("defender_crashed", False)
        attacker_crashed = res.get("attacker_crashed", False)
        if defender_crashed and not attacker_crashed:
            success_count += 1
        survival = res.get("defender_survival_steps")
        if survival is not None:
            defender_survival_total += float(survival)
            defender_survival_count += 1

    if not results:
        return

    summary = {"eval/success_rate": success_count / len(results)}
    if defender_survival_count:
        summary["eval/avg_defender_survival_steps"] = defender_survival_total / defender_survival_count

    if run is not None:
        run.log(summary)
    if tb_writer is not None:
        step = len(results)
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                tb_writer.add_scalar(key, value, step)


def main():
    cfg_env = os.environ.get("F110_CONFIG")
    default_cfg = Path("configs/experiment_gaplock_dqn.yaml")
    cfg_path = Path(cfg_env) if cfg_env else default_cfg
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)

    config_dirty = False
    run_seed = _apply_run_seed(cfg)
    if run_seed is not None:
        config_dirty = True
        print(f"[INFO] Using RUN_SEED={run_seed}")
        _seed_everything(run_seed)

    main_cfg = cfg.setdefault("main", {})
    mode = main_cfg.get("mode", "train").lower()
    wandb_run = _wandb_init(cfg, mode)
    if wandb_run is not None:
        overrides = wandb_run.config.as_dict()
        if overrides:
            _deep_update(cfg, overrides)
            config_dirty = True

    tmp_cfg_path: Optional[Path] = None
    if config_dirty:
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml", mode="w")
        with tmp_file:
            yaml.safe_dump(cfg, tmp_file)
        tmp_cfg_path = Path(tmp_file.name)
        cfg_path = tmp_cfg_path

    tb_writer = None
    tb_dir = main_cfg.get("tensorboard_dir")
    if tb_dir:
        try:
            from torch.utils.tensorboard import SummaryWriter

            tb_writer = SummaryWriter(log_dir=tb_dir)
        except ImportError:
            print("[WARN] torch.utils.tensorboard not available; skipping TensorBoard logging")
            tb_writer = None

    def update_logger(metrics: Dict[str, Any]):
        if wandb_run is not None:
            wandb_run.log(metrics)
        if tb_writer is not None:
            if "train/update" in metrics:
                step = int(metrics["train/update"])
            elif "train/episode" in metrics:
                step = int(metrics["train/episode"])
            else:
                step = 0
            for key, value in metrics.items():
                if key == "train/update":
                    continue
                if isinstance(value, (int, float)):
                    tb_writer.add_scalar(key, value, step)

    update_cb = update_logger if (wandb_run is not None or tb_writer is not None) else None

    if mode == "train":
        train_ctx = train_module.create_training_context(cfg_path)
        if _RENDER_FLAG:
            train_ctx.render_interval = 1
            train_ctx.env.render_mode = "human"
        bundle_cfg = train_ctx.ppo_bundle.metadata.get("config", {})
        episodes = _EP_OVERRIDE or bundle_cfg.get("train_episodes") or cfg.get("main", {}).get("train_episodes", 10)
        results = train_module.run_training(
            train_ctx,
            episodes=episodes,
            update_callback=update_cb,
        )
        gap_id = train_ctx.opponent_ids[0] if train_ctx.opponent_ids else None
        _log_train_results(wandb_run, results, train_ctx.ppo_agent_id, gap_id, tb_writer=tb_writer)
        train_ctx.ppo_agent.save(str(train_ctx.best_path))
        print(f"[INFO] Saved final model to {train_ctx.best_path}")

    elif mode == "eval":
        # Rebuild evaluation context so explicit checkpoint overrides take effect
        eval_ctx = eval_module.create_evaluation_context(cfg_path)
        if _RENDER_FLAG:
            eval_ctx.env.render_mode = "human"
        bundle_cfg = eval_ctx.ppo_bundle.metadata.get("config", {})
        episodes = _EVAL_EP_OVERRIDE or bundle_cfg.get("eval_episodes") or cfg.get("main", {}).get("eval_episodes", 5)
        results = eval_module.evaluate(eval_ctx, episodes=episodes, force_render=_RENDER_FLAG)
        _log_eval_results(wandb_run, results, tb_writer=tb_writer)

    elif mode == "train_eval":
        train_ctx = train_module.create_training_context(cfg_path)
        if _RENDER_FLAG:
            train_ctx.render_interval = 1
            train_ctx.env.render_mode = "human"
        bundle_cfg = train_ctx.ppo_bundle.metadata.get("config", {})
        train_episodes = _EP_OVERRIDE or bundle_cfg.get("train_episodes") or cfg.get("main", {}).get("train_episodes", 10)
        results = train_module.run_training(
            train_ctx,
            episodes=train_episodes,
            update_callback=update_cb,
        )
        gap_id = train_ctx.opponent_ids[0] if train_ctx.opponent_ids else None
        _log_train_results(wandb_run, results, train_ctx.ppo_agent_id, gap_id, tb_writer=tb_writer)
        train_ctx.ppo_agent.save(str(train_ctx.best_path))
        print(f"[INFO] Saved final model to {train_ctx.best_path}")

        eval_ctx = eval_module.create_evaluation_context(cfg_path)
        if _RENDER_FLAG:
            eval_ctx.env.render_mode = "human"
        eval_bundle_cfg = eval_ctx.ppo_bundle.metadata.get("config", {})
        eval_episodes = _EVAL_EP_OVERRIDE or eval_bundle_cfg.get("eval_episodes") or cfg.get("main", {}).get("eval_episodes", 5)
        results = eval_module.evaluate(eval_ctx, episodes=eval_episodes, force_render=_RENDER_FLAG)
        _log_eval_results(wandb_run, results, tb_writer=tb_writer)

    else:
        raise ValueError(f"Unsupported mode '{mode}'. Expected train/eval/train_eval.")

    if tb_writer is not None:
        tb_writer.flush()
        tb_writer.close()

    if wandb_run is not None:
        wandb_run.finish()

    if tmp_cfg_path is not None and tmp_cfg_path.exists():
        tmp_cfg_path.unlink()


if __name__ == "__main__":
    main()
