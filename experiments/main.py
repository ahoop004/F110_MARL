#!/usr/bin/env python3
import os
import sys
import tempfile
import random
from typing import Any, Dict, Optional

_RENDER_FLAG = False
_EP_OVERRIDE: Optional[int] = None
_EVAL_EP_OVERRIDE: Optional[int] = None
_MAP_OVERRIDE: Optional[str] = None
_CFG_OVERRIDE: Optional[str] = None
_ALGO_OVERRIDE: Optional[str] = None

_DEFAULT_CONFIGS = {
    "dqn": "configs/experiment_gaplock_dqn.yaml",
    "ppo": "configs/experiment_gaplock_ppo.yaml",
    "td3": "configs/experiment_gaplock_td3.yaml",
}

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
    if arg == "--map":
        if i + 1 >= len(argv):
            print("[ERROR] --map requires a value", file=sys.stderr)
            sys.exit(2)
        _MAP_OVERRIDE = argv[i + 1]
        argv.pop(i + 1)
        argv.pop(i)
        continue
    if arg.startswith("--map="):
        _, value = arg.split("=", 1)
        _MAP_OVERRIDE = value
        argv.pop(i)
        continue
    if arg == "--config":
        if i + 1 >= len(argv):
            print("[ERROR] --config requires a value", file=sys.stderr)
            sys.exit(2)
        _CFG_OVERRIDE = argv[i + 1]
        argv.pop(i + 1)
        argv.pop(i)
        continue
    if arg.startswith("--config="):
        _, value = arg.split("=", 1)
        _CFG_OVERRIDE = value
        argv.pop(i)
        continue
    if arg == "--algo":
        if i + 1 >= len(argv):
            print("[ERROR] --algo requires a value", file=sys.stderr)
            sys.exit(2)
        _ALGO_OVERRIDE = argv[i + 1]
        argv.pop(i + 1)
        argv.pop(i)
        continue
    if arg.startswith("--algo="):
        _, value = arg.split("=", 1)
        _ALGO_OVERRIDE = value
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


def _log_train_results(run, results, ppo_id=None, gap_id=None):
    if run is None:
        return

    focus_ids = {aid for aid in (ppo_id, gap_id) if aid}

    for idx, record in enumerate(results, 1):
        episode = idx
        payload: Dict[str, Any] = {}

        def _is_focus_reward(key: str) -> bool:
            return any(key.startswith(f"reward_component_{aid}_") for aid in focus_ids)

        def _add_metric(name: str, value: Any) -> None:
            if value is None:
                return
            if isinstance(value, (int, float, bool)):
                payload[name] = float(value)
            else:
                payload[name] = value

        if isinstance(record, dict):
            episode = int(record.get("episode", idx))

            duplicate_fields = {
                "steps",
                "collisions_total",
                "defender_survival_steps",
                "success",
                "defender_crashed",
                "attacker_crashed",
                "idle_truncated",
                "epsilon",
            }

            for key, value in record.items():
                if key in {"episode", "returns"}:
                    continue
                if key in duplicate_fields:
                    continue
                if key.startswith("return_"):
                    continue
                if key.startswith("collision_count_") or key.startswith("collision_step_"):
                    continue
                if key.startswith("avg_speed_"):
                    continue
                if key.startswith("reward_component_"):
                    continue
                if key == "reward_mode":
                    continue
                if key == "cause":
                    if value:
                        payload["train/cause"] = value
                    continue
                if isinstance(value, (int, float)):
                    _add_metric(f"train/{key}", value)
                elif value is not None:
                    payload[f"train/{key}"] = value
        if payload:
            run.log(payload, step=episode)


def _log_eval_results(run, results):
    if run is None:
        return

    success_count = 0
    for res in results:
        payload: Dict[str, Any] = {}

        avg_speed_attacker = res.get("avg_speed_attacker")
        if avg_speed_attacker is not None:
            payload["eval/avg_speed_attacker"] = float(avg_speed_attacker)

        avg_speed_defender = res.get("avg_speed_defender")
        if avg_speed_defender is not None:
            payload["eval/avg_speed_defender"] = float(avg_speed_defender)

        if payload:
            run.log(payload)

        defender_crashed = res.get("defender_crashed", False)
        attacker_crashed = res.get("attacker_crashed", False)
        if defender_crashed and not attacker_crashed:
            success_count += 1

    if not results:
        return

    run.log({"eval/success_rate": success_count / len(results)})


def main():
    cfg_env = os.environ.get("F110_CONFIG")

    cfg_path: Optional[Path] = None
    if _CFG_OVERRIDE:
        cfg_path = Path(_CFG_OVERRIDE)
    elif _ALGO_OVERRIDE:
        algo_key = _ALGO_OVERRIDE.strip().lower()
        cfg_candidate = _DEFAULT_CONFIGS.get(algo_key)
        if cfg_candidate is None:
            print(
                f"[ERROR] Unknown algorithm '{_ALGO_OVERRIDE}'. Available: {sorted(_DEFAULT_CONFIGS)}",
                file=sys.stderr,
            )
            sys.exit(2)
        cfg_path = Path(cfg_candidate)
    elif cfg_env:
        cfg_path = Path(cfg_env)
    else:
        cfg_path = Path(_DEFAULT_CONFIGS["dqn"])

    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)

    config_dirty = False
    env_cfg = cfg.setdefault("env", {})
    if _MAP_OVERRIDE is not None:
        map_choice = _MAP_OVERRIDE.strip()
        if not map_choice:
            print("[ERROR] --map value cannot be empty", file=sys.stderr)
            sys.exit(2)
        env_cfg["map"] = map_choice
        env_cfg["map_yaml"] = map_choice if Path(map_choice).suffix else f"{map_choice}.yaml"
        print(f"[INFO] Using map override '{map_choice}'")
        config_dirty = True
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

    def update_logger(metrics: Dict[str, Any]):
        if wandb_run is not None:
            wandb_run.log(metrics)

    update_cb = update_logger if wandb_run is not None else None

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
        _log_train_results(wandb_run, results, train_ctx.ppo_agent_id, gap_id)
        train_ctx.ppo_agent.save(str(train_ctx.best_path))
        print(f"[INFO] Saved final model to {train_ctx.best_path}")

    elif mode == "eval":
        # Rebuild evaluation context so explicit checkpoint overrides take effect
        eval_ctx = eval_module.create_evaluation_context(cfg_path, auto_load=True)
        if _RENDER_FLAG:
            eval_ctx.env.render_mode = "human"
        bundle_cfg = eval_ctx.ppo_bundle.metadata.get("config", {})
        episodes = _EVAL_EP_OVERRIDE or bundle_cfg.get("eval_episodes") or cfg.get("main", {}).get("eval_episodes", 5)
        results = eval_module.evaluate(eval_ctx, episodes=episodes, force_render=_RENDER_FLAG)
        _log_eval_results(wandb_run, results)

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
        _log_train_results(wandb_run, results, train_ctx.ppo_agent_id, gap_id)
        train_ctx.ppo_agent.save(str(train_ctx.best_path))
        print(f"[INFO] Saved final model to {train_ctx.best_path}")

        eval_ctx = eval_module.create_evaluation_context(cfg_path, auto_load=True)
        if _RENDER_FLAG:
            eval_ctx.env.render_mode = "human"
        eval_bundle_cfg = eval_ctx.ppo_bundle.metadata.get("config", {})
        eval_episodes = _EVAL_EP_OVERRIDE or eval_bundle_cfg.get("eval_episodes") or cfg.get("main", {}).get("eval_episodes", 5)
        results = eval_module.evaluate(eval_ctx, episodes=eval_episodes, force_render=_RENDER_FLAG)
        _log_eval_results(wandb_run, results)

    else:
        raise ValueError(f"Unsupported mode '{mode}'. Expected train/eval/train_eval.")

    if wandb_run is not None:
        wandb_run.finish()

    if tmp_cfg_path is not None and tmp_cfg_path.exists():
        tmp_cfg_path.unlink()


if __name__ == "__main__":
    main()
