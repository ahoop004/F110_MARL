#!/usr/bin/env python3
import os
import sys
import tempfile
import random
from typing import Any, Dict, List, Optional, Tuple

_RENDER_FLAG = False
_EP_OVERRIDE: Optional[int] = None
_EVAL_EP_OVERRIDE: Optional[int] = None
_MAP_OVERRIDE: Optional[str] = None
_CFG_OVERRIDE: Optional[str] = None
_ALGO_OVERRIDE: Optional[str] = None
_EXP_OVERRIDE: Optional[str] = None

_DEFAULT_CONFIGS: Dict[str, Tuple[str, str]] = {
    "dqn": ("scenarios/gaplock_dqn.yaml", "gaplock_dqn"),

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
    if arg == "--experiment":
        if i + 1 >= len(argv):
            print("[ERROR] --experiment requires a value", file=sys.stderr)
            sys.exit(2)
        _EXP_OVERRIDE = argv[i + 1]
        argv.pop(i + 1)
        argv.pop(i)
        continue
    if arg.startswith("--experiment="):
        _, value = arg.split("=", 1)
        _EXP_OVERRIDE = value
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
from f110x.utils.logger import ConsoleSink, Logger, WandbSink


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


def _coerce_positive_int(value: Any, default: int) -> int:
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return default
    return coerced if coerced > 0 else default

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

    run = wandb.init(project=project, entity=entity, config=cfg, group=group, reinit=True, tags=tags)

    if run is not None:
        wandb.define_metric("train/episode")
        wandb.define_metric("train/return_*", step_metric="train/episode")
        wandb.define_metric("eval/episode")
        wandb.define_metric("eval/return_*", step_metric="eval/episode")

    return run

def main():
    cfg_env = os.environ.get("F110_CONFIG")

    cfg_path: Optional[Path] = None
    cfg_experiment: Optional[str] = _EXP_OVERRIDE

    if _CFG_OVERRIDE:
        cfg_path = Path(_CFG_OVERRIDE)
    elif _ALGO_OVERRIDE:
        algo_key = _ALGO_OVERRIDE.strip().lower()
        cfg_info = _DEFAULT_CONFIGS.get(algo_key)
        if cfg_info is None:
            print(
                f"[ERROR] Unknown algorithm '{_ALGO_OVERRIDE}'. Available: {sorted(_DEFAULT_CONFIGS)}",
                file=sys.stderr,
            )
            sys.exit(2)
        cfg_candidate, cfg_exp = cfg_info
        cfg_path = Path(cfg_candidate)
        cfg_experiment = cfg_experiment or cfg_exp
    elif cfg_env:
        cfg_path = Path(cfg_env)
    else:
        cfg_candidate, cfg_exp = _DEFAULT_CONFIGS["dqn"]
        cfg_path = Path(cfg_candidate)
        cfg_experiment = cfg_experiment or cfg_exp

    with cfg_path.open() as f:
        cfg_doc = yaml.safe_load(f) or {}

    if not isinstance(cfg_doc, dict):
        print("[ERROR] Configuration root must be a mapping", file=sys.stderr)
        sys.exit(2)

    pending_logs: List[Tuple[str, str, Dict[str, Any]]] = []

    experiments_section = cfg_doc.get("experiments") if isinstance(cfg_doc.get("experiments"), dict) else None
    if experiments_section is not None:
        if not cfg_experiment:
            cfg_experiment = cfg_doc.get("default_experiment")
        if not cfg_experiment:
            print("[ERROR] Experiment must be specified for multi-experiment configs", file=sys.stderr)
            sys.exit(2)
        if cfg_experiment not in experiments_section:
            print(
                f"[ERROR] Experiment '{cfg_experiment}' not found. Available: {sorted(experiments_section)}",
                file=sys.stderr,
            )
            sys.exit(2)
        cfg = experiments_section[cfg_experiment] or {}
        if not isinstance(cfg, dict):
            print(f"[ERROR] Experiment '{cfg_experiment}' must be a mapping", file=sys.stderr)
            sys.exit(2)
    else:
        cfg = cfg_doc
        if cfg_experiment and _EXP_OVERRIDE:
            print("[WARN] --experiment ignored because config has no experiments section.")
        cfg_experiment = None

    config_dirty = False
    env_cfg = cfg.setdefault("env", {})
    if _MAP_OVERRIDE is not None:
        map_choice = _MAP_OVERRIDE.strip()
        if not map_choice:
            print("[ERROR] --map value cannot be empty", file=sys.stderr)
            sys.exit(2)
        env_cfg["map"] = map_choice
        env_cfg["map_yaml"] = map_choice if Path(map_choice).suffix else f"{map_choice}.yaml"
        pending_logs.append(("info", "Using map override", {"map": map_choice}))
        config_dirty = True
    run_seed = _apply_run_seed(cfg)
    if run_seed is not None:
        config_dirty = True
        pending_logs.append(("info", "Using RUN_SEED", {"seed": run_seed}))
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
            if experiments_section is not None and cfg_experiment:
                experiments_section[cfg_experiment] = cfg
                cfg_doc["experiments"] = experiments_section
                yaml.safe_dump(cfg_doc, tmp_file)
            else:
                yaml.safe_dump(cfg, tmp_file)
        tmp_cfg_path = Path(tmp_file.name)
        cfg_path = tmp_cfg_path

    # Console sink provides structured CLI output; Wandb sink respects cfg['main']['wandb']
    # (project/entity/run-id) and logs against the standard train/episode and eval/episode metrics.
    sinks = [ConsoleSink()]
    if wandb_run is not None:
        sinks.append(WandbSink(wandb_run))

    logger = Logger(sinks)
    logger.start({
        "mode": mode,
        "config_path": str(cfg_path),
        "experiment": cfg_experiment,
    })
    for level, message, extra in pending_logs:
        if level == "info":
            logger.info(message, extra=extra)
        else:
            logger.warning(message, extra=extra)
    pending_logs.clear()

    if mode == "train":
        train_session = train_module.create_training_session(
            cfg_path,
            experiment=cfg_experiment,
            logger=logger,
        )
        if _RENDER_FLAG:
            train_session.enable_render()
        fallback_train = _coerce_positive_int(cfg.get("main", {}).get("train_episodes"), 10)
        episodes = train_module.resolve_train_episodes(
            train_session,
            override=_EP_OVERRIDE,
            fallback=fallback_train,
        )
        results = train_module.run_training(
            train_session,
            episodes=episodes,
        )
        final_path = train_session.save_final_model()
        if final_path is not None:
            logger.info(
                "Saved final model",
                extra={"path": str(final_path)},
            )

    elif mode == "eval":
        eval_session = eval_module.create_evaluation_session(
            cfg_path,
            auto_load=True,
            experiment=cfg_experiment,
            logger=logger,
        )
        if _RENDER_FLAG:
            eval_session.enable_render()
        fallback_eval = _coerce_positive_int(cfg.get("main", {}).get("eval_episodes"), 5)
        episodes = eval_module.resolve_eval_episodes(
            eval_session,
            override=_EVAL_EP_OVERRIDE,
            fallback=fallback_eval,
        )
        results = eval_module.run_evaluation(
            eval_session,
            episodes=episodes,
            force_render=_RENDER_FLAG,
            auto_load=False,
        )

    elif mode == "train_eval":
        train_session = train_module.create_training_session(
            cfg_path,
            experiment=cfg_experiment,
            logger=logger,
        )
        if _RENDER_FLAG:
            train_session.enable_render()
        fallback_train = _coerce_positive_int(cfg.get("main", {}).get("train_episodes"), 10)
        train_episodes = train_module.resolve_train_episodes(
            train_session,
            override=_EP_OVERRIDE,
            fallback=fallback_train,
        )
        train_results = train_module.run_training(
            train_session,
            episodes=train_episodes,
        )
        final_path = train_session.save_final_model()
        if final_path is not None:
            logger.info(
                "Saved final model",
                extra={"path": str(final_path)},
            )

        eval_session = eval_module.create_evaluation_session(
            cfg_path,
            auto_load=True,
            experiment=cfg_experiment,
            logger=logger,
        )
        if _RENDER_FLAG:
            eval_session.enable_render()
        fallback_eval = _coerce_positive_int(cfg.get("main", {}).get("eval_episodes"), 5)
        eval_episodes = eval_module.resolve_eval_episodes(
            eval_session,
            override=_EVAL_EP_OVERRIDE,
            fallback=fallback_eval,
        )
        eval_results = eval_module.run_evaluation(
            eval_session,
            episodes=eval_episodes,
            force_render=_RENDER_FLAG,
            auto_load=False,
        )

    else:
        raise ValueError(f"Unsupported mode '{mode}'. Expected train/eval/train_eval.")

    logger.stop()

    if tmp_cfg_path is not None and tmp_cfg_path.exists():
        tmp_cfg_path.unlink()


if __name__ == "__main__":
    main()
