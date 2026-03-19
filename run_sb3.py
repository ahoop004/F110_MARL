#!/usr/bin/env python3
"""On-policy SB3 training runner (PPO/A2C) using the v2 scenario format."""

import argparse
import math
import os
import sys
import warnings
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

# Allow running from repo root without installing the package.
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if SRC_DIR.is_dir() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from baselines.sb3_curriculum_callback import CurriculumCallback
from baselines.sb3_eval_callback import SB3EvaluationCallback
from baselines.sb3_train_callback import SB3TrainLoggingCallback
from baselines.sb3_wrapper import SB3SingleAgentWrapper
from core.adaptive_controller import AdaptiveScalarController
from core.evaluator import EvaluationConfig
from core.obs_flatten import flatten_observation
from core.run_id import resolve_run_id, set_run_id_env
from core.scenario import ScenarioError, load_and_expand_scenario
from core.setup import create_training_setup
from loggers import ConsoleLogger, WandbLogger

try:
    from stable_baselines3 import A2C, PPO
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList
    from stable_baselines3.common.monitor import Monitor
    try:
        from wandb.integration.sb3 import WandbCallback as _WandbCallback
    except ImportError:
        _WandbCallback = None
except ImportError as exc:
    print(
        "stable-baselines3 is required for on-policy runs. "
        "Install with: pip install stable-baselines3",
        file=sys.stderr,
    )
    raise

try:
    from rich.panel import Panel
    from rich.table import Table
    RICH_PANEL_AVAILABLE = True
except Exception:
    RICH_PANEL_AVAILABLE = False


ON_POLICY_ALGOS = {"sb3_ppo", "sb3_a2c", "ppo", "a2c"}


@dataclass
class GatedScheduleSpec:
    """Metadata for a linear schedule gated by rolling training success."""

    key: str
    schedule: "GatedLinearSchedule"
    success_threshold: float
    window_episodes: int


@dataclass
class AdaptiveHyperparamSpec:
    """Metadata for success-adaptive hyperparameter control."""

    key: str
    controller: AdaptiveScalarController


class GatedLinearSchedule:
    """Linear schedule that stays flat until activated, then anneals to final value."""

    def __init__(self, initial_value: float, final_value: float):
        self.initial_value = float(initial_value)
        self.final_value = float(final_value)
        self._activated = False
        self._start_progress_remaining: Optional[float] = None

    @property
    def activated(self) -> bool:
        return self._activated

    def activate(self, progress_remaining: Optional[float] = None) -> None:
        self._activated = True
        if progress_remaining is not None:
            self._start_progress_remaining = max(0.0, float(progress_remaining))

    def __call__(self, progress_remaining: float) -> float:
        if not self._activated:
            return self.initial_value

        if self._start_progress_remaining is None:
            self._start_progress_remaining = max(0.0, float(progress_remaining))

        start = float(self._start_progress_remaining)
        if start <= 0.0:
            return self.final_value

        current = max(0.0, float(progress_remaining))
        progress = (start - current) / start
        progress = min(max(progress, 0.0), 1.0)
        return self.initial_value + (self.final_value - self.initial_value) * progress


def _linear_schedule(initial_value: float, final_value: float):
    """Create a linear schedule callable for SB3 (progress_remaining: 1 -> 0)."""
    initial_value = float(initial_value)
    final_value = float(final_value)

    def func(progress_remaining: float) -> float:
        return final_value + (initial_value - final_value) * float(progress_remaining)

    return func


def _resolve_scheduled_param(
    params: Dict[str, Any],
    key: str,
    default_value: float,
    default_linear_final: Optional[float] = None,
) -> tuple[Any, Optional[GatedScheduleSpec]]:
    """Resolve scalar/scheduled SB3 parameter from scenario params.

    Supports either:
    - `<key>: <float>` (fixed)
    - `<key>_schedule: {type: linear, initial: x, final: y}`
    - `<key>_schedule: linear` (uses current `<key>` as initial; final defaults if provided)
    """
    base_value = float(params.get(key, default_value))
    schedule_cfg = params.get(f"{key}_schedule")

    if schedule_cfg is None:
        return base_value, None

    if isinstance(schedule_cfg, str):
        schedule_type = schedule_cfg.strip().lower()
        if schedule_type == "linear":
            final_value = 0.0 if default_linear_final is None else float(default_linear_final)
            return _linear_schedule(base_value, final_value), None
        raise ValueError(
            f"Unsupported {key}_schedule '{schedule_cfg}'. Use 'linear' or a schedule dict."
        )

    if not isinstance(schedule_cfg, dict):
        raise ValueError(f"{key}_schedule must be a string or dict.")

    schedule_type = str(schedule_cfg.get("type", "linear")).strip().lower()
    if schedule_type != "linear":
        raise ValueError(f"Unsupported {key}_schedule type '{schedule_type}'. Only 'linear' is supported.")

    initial_value = float(schedule_cfg.get("initial", base_value))
    if "final" in schedule_cfg:
        final_value = float(schedule_cfg["final"])
    else:
        final_value = 0.0 if default_linear_final is None else float(default_linear_final)

    gate_cfg = schedule_cfg.get("gate")
    if gate_cfg is None:
        return _linear_schedule(initial_value, final_value), None
    if not isinstance(gate_cfg, dict):
        raise ValueError(f"{key}_schedule.gate must be a dict when provided.")

    success_threshold = float(gate_cfg.get("success_rate", gate_cfg.get("threshold", 0.85)))
    window_episodes = int(gate_cfg.get("window_episodes", gate_cfg.get("window", 100)))
    if window_episodes <= 0:
        raise ValueError(f"{key}_schedule.gate.window_episodes must be > 0.")

    gated_schedule = GatedLinearSchedule(initial_value, final_value)
    spec = GatedScheduleSpec(
        key=key,
        schedule=gated_schedule,
        success_threshold=success_threshold,
        window_episodes=window_episodes,
    )
    return gated_schedule, spec


def _is_adaptive_curriculum_enabled(params: Dict[str, Any], key: str) -> bool:
    cfg = params.get(f"{key}_curriculum")
    return isinstance(cfg, dict) and bool(cfg.get("enabled", False))


def _build_adaptive_controller(
    *,
    cfg: Dict[str, Any],
    default_start: float,
    default_final: float,
) -> AdaptiveScalarController:
    start = float(cfg.get("start", cfg.get("initial", default_start)))
    final = float(cfg.get("final", default_final))

    window = max(1, int(cfg.get("window_episodes", 100)))
    update_every = max(1, int(cfg.get("update_every_episodes", window)))
    sr_low = float(cfg.get("sr_lower_bound", 0.85))
    sr_high = float(cfg.get("sr_upper_bound", 0.95))

    span = abs(final - start)
    default_inc_min = max(span * 0.02, 1e-9)
    default_inc_max = max(span * 0.10, default_inc_min)

    inc_min_raw = float(cfg.get("increment_min", default_inc_min))
    inc_max_raw = float(cfg.get("increment_max", default_inc_max))
    inc_small_mag = abs(inc_min_raw)
    inc_large_mag = abs(inc_max_raw)
    if inc_large_mag < inc_small_mag:
        inc_small_mag, inc_large_mag = inc_large_mag, inc_small_mag

    regress_enabled = bool(cfg.get("regression_enabled", False))
    regress_trigger = float(cfg.get("regression_sr_trigger", max(0.0, sr_low - 0.20)))
    regress_floor = float(cfg.get("regression_sr_floor", max(0.0, regress_trigger - 0.20)))
    regress_patience = max(1, int(cfg.get("regression_patience_updates", 1)))
    regress_cooldown = max(0, int(cfg.get("regression_cooldown_updates", 0)))

    regress_inc_min_raw = float(
        cfg.get("regression_increment_min", cfg.get("regression_step_min", inc_small_mag))
    )
    regress_inc_max_raw = float(
        cfg.get("regression_increment_max", cfg.get("regression_step_max", inc_large_mag))
    )
    regress_small_mag = abs(regress_inc_min_raw)
    regress_large_mag = abs(regress_inc_max_raw)
    if regress_large_mag < regress_small_mag:
        regress_small_mag, regress_large_mag = regress_large_mag, regress_small_mag

    direction = 1.0 if final >= start else -1.0
    advance_small = direction * inc_small_mag
    advance_large = direction * inc_large_mag
    regress_small = -direction * regress_small_mag
    regress_large = -direction * regress_large_mag

    return AdaptiveScalarController(
        initial_value=start,
        value_min=min(start, final),
        value_max=max(start, final),
        window_size=window,
        update_every=update_every,
        advance_sr_low=sr_low,
        advance_sr_high=sr_high,
        advance_step_min=advance_small,
        advance_step_max=advance_large,
        regression_enabled=regress_enabled,
        regression_sr_trigger=regress_trigger,
        regression_sr_floor=regress_floor,
        regression_step_min=regress_small,
        regression_step_max=regress_large,
        regression_patience_updates=regress_patience,
        regression_cooldown_updates=regress_cooldown,
        progress_start=start,
        progress_end=final,
    )


def _resolve_adaptive_hparam_specs(
    algorithm: str,
    params: Dict[str, Any],
) -> List[AdaptiveHyperparamSpec]:
    specs: List[AdaptiveHyperparamSpec] = []
    algo_key = str(algorithm).strip().lower()
    defaults: Dict[str, Tuple[float, float]] = {
        "learning_rate": (float(params.get("learning_rate", 3e-4)), 1e-5),
        "ent_coef": (float(params.get("ent_coef", 0.02 if "ppo" in algo_key else 0.0)), 0.0),
    }
    if "ppo" in algo_key:
        defaults["clip_range"] = (float(params.get("clip_range", 0.2)), 0.02)

    for key, (default_start, default_final) in defaults.items():
        cfg = params.get(f"{key}_curriculum")
        if not (isinstance(cfg, dict) and bool(cfg.get("enabled", False))):
            continue
        mode = str(cfg.get("mode", "success_adaptive")).strip().lower()
        if mode != "success_adaptive":
            continue
        try:
            controller = _build_adaptive_controller(
                cfg=cfg,
                default_start=default_start,
                default_final=default_final,
            )
        except (TypeError, ValueError):
            continue
        specs.append(AdaptiveHyperparamSpec(key=key, controller=controller))

    return specs


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="F110 on-policy SB3 training (PPO/A2C)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--scenario",
        type=str,
        required=True,
        help="Path to scenario YAML file",
    )

    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging (overrides scenario config)",
    )

    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging (overrides scenario config)",
    )

    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable rendering (default: False)",
    )

    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable rendering",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides scenario config)",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Number of episodes (overrides scenario config)",
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "auto"],
        default=None,
        help="Compute device (overrides scenario config): cpu, cuda, or auto",
    )

    parser.add_argument(
        "--cuda-device",
        type=int,
        default=None,
        help="CUDA device index to use with --device cuda/auto (e.g., 0, 1)",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable console output",
    )

    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Custom run ID for logging alignment",
    )

    return parser.parse_args()


def resolve_cli_overrides(scenario: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Apply CLI argument overrides to scenario."""
    if args.seed is not None:
        scenario.setdefault("experiment", {})["seed"] = args.seed

    if args.episodes is not None:
        scenario.setdefault("experiment", {})["episodes"] = args.episodes

    if args.wandb:
        scenario.setdefault("wandb", {})["enabled"] = True
    elif args.no_wandb:
        scenario.setdefault("wandb", {})["enabled"] = False

    if args.render:
        scenario.setdefault("environment", {})["render"] = True
    elif args.no_render:
        scenario.setdefault("environment", {})["render"] = False

    if args.device is not None:
        # Apply to all on-policy trainable agents to keep behavior consistent.
        for agent_cfg in scenario.get("agents", {}).values():
            algo = str(agent_cfg.get("algorithm", "")).lower()
            if algo in ON_POLICY_ALGOS:
                if args.device == "auto":
                    if torch.cuda.is_available():
                        if args.cuda_device is not None:
                            resolved_device = f"cuda:{int(args.cuda_device)}"
                        else:
                            resolved_device = "cuda"
                    else:
                        resolved_device = "cpu"
                elif args.device == "cuda" and args.cuda_device is not None:
                    resolved_device = f"cuda:{int(args.cuda_device)}"
                else:
                    resolved_device = args.device
                agent_cfg["device"] = resolved_device
                agent_cfg.setdefault("params", {})["device"] = resolved_device

    return scenario


def initialize_loggers(
    scenario: Dict[str, Any], args: argparse.Namespace, run_id: Optional[str] = None
) -> Tuple[Optional[WandbLogger], ConsoleLogger]:
    """Initialize W&B and console loggers."""
    console_logger = ConsoleLogger(verbose=not args.quiet)

    wandb_config = scenario.get("wandb", {})
    wandb_enabled = wandb_config.get("enabled", False)

    default_algo = "unknown"
    for agent_cfg in scenario.get("agents", {}).values():
        algo = agent_cfg.get("algorithm", "").lower()
        if algo and algo not in ["ftg", "pp", "pure_pursuit"]:
            default_algo = algo
            break
    default_group = scenario.get("experiment", {}).get("name")

    sweep_mode = bool(os.environ.get("WANDB_SWEEP_ID"))

    if wandb_enabled:
        console_logger.print_info("Initializing Weights & Biases...")
        wandb_logger = WandbLogger(
            project=wandb_config.get("project", "f110-marl"),
            name=wandb_config.get("name", scenario["experiment"]["name"]),
            config=None if sweep_mode else scenario,
            tags=wandb_config.get("tags", []),
            group=wandb_config.get("group", default_group),
            job_type=wandb_config.get("job_type", default_algo),
            entity=wandb_config.get("entity", None),
            notes=wandb_config.get("notes", None),
            mode=wandb_config.get("mode", "online"),
            run_id=run_id,
            logging_config=wandb_config.get("logging"),
        )
    else:
        wandb_logger = None

    return wandb_logger, console_logger


def apply_wandb_sweep_overrides(scenario: Dict[str, Any], console_logger: ConsoleLogger) -> None:
    """Apply WandB sweep parameters to the scenario if in sweep mode."""
    try:
        import wandb

        sweep_mode = bool(os.environ.get("WANDB_SWEEP_ID") or getattr(wandb.run, "sweep_id", None))
        if not sweep_mode or not wandb.config or len(dict(wandb.config)) == 0:
            return

        console_logger.print_info("Detected WandB sweep mode - applying sweep parameters...")

        def set_nested_value(d: dict, path: str, value: Any) -> None:
            keys = path.split(".")
            for key in keys[:-1]:
                if key not in d:
                    d[key] = {}
                d = d[key]
            d[keys[-1]] = value

        sb3_agent_id = None
        for agent_id, agent_cfg in scenario.get("agents", {}).items():
            algo = agent_cfg.get("algorithm", "").lower()
            if algo in ON_POLICY_ALGOS:
                sb3_agent_id = agent_id
                break

        if not sb3_agent_id:
            console_logger.print_warning("Sweep mode active but no on-policy agent found to apply params.")
            return

        sweep_params_applied = {}
        wandb_params = {k: wandb.config[k] for k in wandb.config}
        agent_params = scenario.get("agents", {}).get(sb3_agent_id, {}).get("params", {})
        allowed_param_keys = set(agent_params.keys())
        skipped_keys = []

        for key, value in wandb_params.items():
            if (
                key.startswith("_")
                or "/" in key
                or key in ["method", "metric", "program", "algorithm", "scenario"]
            ):
                continue

            if key == "episodes":
                scenario.setdefault("experiment", {})["episodes"] = value
                sweep_params_applied[key] = value
                continue
            if key == "seed":
                scenario.setdefault("experiment", {})["seed"] = value
                sweep_params_applied[key] = value
                continue

            if key not in allowed_param_keys and "." not in key:
                skipped_keys.append(key)
                continue

            override_path = key if "." in key else f"agents.{sb3_agent_id}.params.{key}"
            set_nested_value(scenario, override_path, value)
            sweep_params_applied[key] = value

        if sweep_params_applied:
            console_logger.print_success(
                f"Applied {len(sweep_params_applied)} sweep parameter(s) to {sb3_agent_id}"
            )
            for key, value in sweep_params_applied.items():
                console_logger.print_info(f"  {key} = {value}")
            if skipped_keys:
                skipped_str = ", ".join(sorted(skipped_keys))
                console_logger.print_warning(
                    f"Skipped {len(skipped_keys)} sweep key(s) not in agent params: {skipped_str}"
                )
    except Exception as exc:
        console_logger.print_warning(f"Failed to apply sweep parameters: {exc}")


def select_on_policy_agent(scenario: Dict[str, Any]) -> Tuple[str, str]:
    """Select the on-policy agent from the scenario."""
    for agent_id, agent_cfg in scenario.get("agents", {}).items():
        algo = agent_cfg.get("algorithm", "").lower()
        if algo in ON_POLICY_ALGOS:
            return agent_id, algo
    raise ValueError("No on-policy agent found (expected sb3_ppo or sb3_a2c).")


def infer_observation_preset(agent_config: Dict[str, Any]) -> Optional[str]:
    """Infer observation preset name from agent config."""
    obs_config = agent_config.get("observation")
    if isinstance(obs_config, dict):
        if "preset" in obs_config:
            return obs_config["preset"]
        if "_preset" in obs_config:
            return obs_config["_preset"]
        if obs_config.get("speed", {}).get("enabled") or obs_config.get("prev_action", {}).get("enabled"):
            return "centerline"
        if obs_config.get("target_state", {}).get("enabled"):
            return "gaplock"
        if len(obs_config) > 0:
            return "gaplock"
    return None


def resolve_action_stack(agent_config: Dict[str, Any]) -> int:
    """Resolve action history stack size from observation config.

    Returns:
        Number of previous actions to append to each observation.
        Returns 0 when disabled or unspecified.
    """
    obs_config = agent_config.get("observation")
    if not isinstance(obs_config, dict):
        return 0

    cfg = obs_config.get("action_stack")
    if not isinstance(cfg, dict):
        return 0

    if not bool(cfg.get("enabled", False)):
        return 0

    try:
        size = int(cfg.get("size", 1))
    except (TypeError, ValueError):
        size = 1
    return max(0, size)


def get_space_dim(space) -> int:
    """Get total dimension of a gym space."""
    if isinstance(space, spaces.Dict):
        return sum(get_space_dim(s) for s in space.spaces.values())
    if isinstance(space, spaces.Box):
        return int(np.prod(space.shape))
    if isinstance(space, spaces.Discrete):
        return 1
    if isinstance(space, spaces.MultiDiscrete):
        return len(space.nvec)
    return 1


def parse_action_repeat(env_config: Dict[str, Any]) -> int:
    """Parse action repeat (step skip) from environment config."""
    value = None
    for key in ("action_repeat", "step_repeat", "step_skip", "frame_skip"):
        if key in env_config:
            value = env_config.get(key)
            break
    if value is None:
        return 1
    try:
        repeat = int(value)
    except (TypeError, ValueError):
        repeat = 1
    return max(1, repeat)


def compute_obs_dim(
    obs_space,
    action_space,
    preset: Optional[str],
    target_id: Optional[str],
    frame_stack: int,
    action_stack: int = 0,
) -> int:
    """Compute flattened observation dimension."""
    if preset:
        dummy_obs = obs_space.sample()
        if target_id:
            dummy_obs["central_state"] = obs_space.sample()
        flat_dummy = flatten_observation(dummy_obs, preset=preset, target_id=target_id)
        obs_dim = int(flat_dummy.shape[0])
    else:
        obs_dim = get_space_dim(obs_space)

    if action_stack > 0:
        action_dim = get_space_dim(action_space)
        obs_dim += action_dim * action_stack

    if frame_stack > 1:
        obs_dim *= frame_stack
    return obs_dim


def build_policy_kwargs(params: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """Build policy_kwargs with net_arch and activation_fn."""
    hidden_dims = params.get("hidden_dims", [256, 256])
    pi_dims = params.get("pi_hidden_dims")
    qf_dims = params.get("qf_hidden_dims")
    vf_dims = params.get("vf_hidden_dims")
    supports_split = model_name in {"SAC", "TD3", "TQC", "DDPG"}
    supports_vf_split = model_name in {"PPO", "A2C"}

    if supports_vf_split and (pi_dims is not None or vf_dims is not None):
        if pi_dims is None:
            pi_dims = hidden_dims
        if vf_dims is None:
            vf_dims = hidden_dims
        net_arch = {"pi": pi_dims, "vf": vf_dims}
    elif supports_split and (pi_dims is not None or qf_dims is not None):
        if pi_dims is None:
            pi_dims = hidden_dims
        if qf_dims is None:
            qf_dims = hidden_dims
        net_arch = {"pi": pi_dims, "qf": qf_dims}
    else:
        net_arch = hidden_dims

    policy_kwargs: Dict[str, Any] = {"net_arch": net_arch}

    activation = params.get("activation")
    if activation is not None:
        if isinstance(activation, str):
            activation_key = activation.strip().lower()
            activation_map = {
                "relu": nn.ReLU,
                "silu": nn.SiLU,
                "swish": nn.SiLU,
                "tanh": nn.Tanh,
            }
            if activation_key not in activation_map:
                raise ValueError(
                    f"Unsupported activation '{activation}'. Use: relu, silu/swish, tanh."
                )
            policy_kwargs["activation_fn"] = activation_map[activation_key]
        else:
            policy_kwargs["activation_fn"] = activation

    return policy_kwargs


def build_model(
    algorithm: str,
    params: Dict[str, Any],
    env,
    policy_kwargs: Dict[str, Any],
    device: str,
    seed: Optional[int],
) -> tuple[Any, List[GatedScheduleSpec]]:
    """Create SB3 on-policy model."""
    gated_specs: List[GatedScheduleSpec] = []

    if _is_adaptive_curriculum_enabled(params, "learning_rate"):
        learning_rate = float(params.get("learning_rate", 3e-4))
    else:
        learning_rate, lr_gate = _resolve_scheduled_param(
            params,
            key="learning_rate",
            default_value=3e-4,
            default_linear_final=0.0,
        )
        if lr_gate is not None:
            gated_specs.append(lr_gate)
    gamma = params.get("gamma", 0.995)

    if algorithm in {"sb3_ppo", "ppo"}:
        if _is_adaptive_curriculum_enabled(params, "clip_range"):
            clip_range = float(params.get("clip_range", 0.2))
        else:
            clip_range, clip_gate = _resolve_scheduled_param(
                params,
                key="clip_range",
                default_value=0.2,
                default_linear_final=0.02,
            )
            if clip_gate is not None:
                gated_specs.append(clip_gate)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*primarily intended to run on the CPU.*", category=UserWarning)
            model = PPO(
                policy="MlpPolicy",
                env=env,
                learning_rate=learning_rate,
                n_steps=params.get("n_steps", 2048),
                batch_size=params.get("batch_size", 256),
                n_epochs=params.get("n_epochs", 10),
                gamma=gamma,
                gae_lambda=params.get("gae_lambda", 0.95),
                clip_range=clip_range,
                clip_range_vf=params.get("clip_range_vf", None),
                ent_coef=float(params.get("ent_coef", 0.02)),
                vf_coef=params.get("vf_coef", 0.5),
                max_grad_norm=params.get("max_grad_norm", 0.5),
                target_kl=params.get("target_kl", None),
                normalize_advantage=params.get("normalize_advantage", True),
                policy_kwargs=policy_kwargs,
                device=device,
                verbose=0,
                seed=seed,
            )
    elif algorithm in {"sb3_a2c", "a2c"}:
        model = A2C(
            policy="MlpPolicy",
            env=env,
            learning_rate=learning_rate,
            n_steps=params.get("n_steps", 5),
            gamma=gamma,
            gae_lambda=params.get("gae_lambda", 1.0),
            ent_coef=float(params.get("ent_coef", 0.0)),
            vf_coef=params.get("vf_coef", 0.5),
            policy_kwargs=policy_kwargs,
            device=device,
            verbose=0,
            seed=seed,
        )
    else:
        raise ValueError(f"Unsupported on-policy algorithm: {algorithm}")

    return model, gated_specs


def _resolve_model_path(path_value: Any) -> Optional[Path]:
    """Resolve warm-start path, accepting with/without .zip suffix."""
    if path_value is None:
        return None
    raw = str(path_value).strip()
    if not raw:
        return None
    expanded = os.path.expandvars(raw)
    path = Path(expanded).expanduser()
    if path.exists():
        return path
    zip_path = path if path.suffix == ".zip" else Path(f"{path}.zip")
    if zip_path.exists():
        return zip_path
    return None


def maybe_apply_warm_start(
    *,
    model,
    algorithm: str,
    train_params: Dict[str, Any],
    console_logger: ConsoleLogger,
) -> None:
    """Warm-start from a saved SB3 checkpoint by copying policy weights only.

    Copying only policy weights intentionally resets optimizer/LR schedule state
    while preserving the learned policy/value function parameters.
    """
    warm_start_cfg = train_params.get("warm_start")
    if isinstance(warm_start_cfg, dict):
        raw_enabled = warm_start_cfg.get("enabled", False)
        path_value = warm_start_cfg.get("path", train_params.get("warm_start_model_path"))
        reset_lr = bool(warm_start_cfg.get("reset_lr", train_params.get("warm_start_reset_lr", True)))
    else:
        raw_enabled = train_params.get("warm_start_enabled")
        path_value = train_params.get("warm_start_model_path")
        reset_lr = bool(train_params.get("warm_start_reset_lr", True))

    warm_path = _resolve_model_path(path_value)
    enabled = bool(raw_enabled) if raw_enabled is not None else (warm_path is not None)

    if not enabled:
        if warm_path is not None:
            console_logger.print_info("Warm-start path provided but loading is disabled (warm_start_enabled=false).")
        return
    if warm_path is None:
        console_logger.print_warning(
            "Warm-start enabled but no valid model path found "
            "(set warm_start_model_path or warm_start.path)."
        )
        return

    algo_key = algorithm.lower()
    if algo_key in {"sb3_ppo", "ppo"}:
        model_class = PPO
    elif algo_key in {"sb3_a2c", "a2c"}:
        model_class = A2C
    else:
        raise ValueError(f"Warm start unsupported for algorithm: {algorithm}")

    console_logger.print_info(f"Warm-start loading weights from: {warm_path}")
    source_model = model_class.load(str(warm_path), device=model.device)
    model.policy.load_state_dict(source_model.policy.state_dict(), strict=True)

    # Optional compatibility mode: restore optimizer states too (not recommended).
    if not reset_lr:
        try:
            model.policy.optimizer.load_state_dict(source_model.policy.optimizer.state_dict())
            console_logger.print_warning(
                "Loaded optimizer state from warm-start model (warm_start_reset_lr=false)."
            )
        except Exception as exc:
            console_logger.print_warning(f"Failed to load warm-start optimizer state: {exc}")

    del source_model
    if reset_lr:
        console_logger.print_success("Warm-start applied with fresh optimizer/LR schedule.")


def build_spawn_curriculum(env, scenario: Dict[str, Any], console_logger: ConsoleLogger):
    """Create spawn curriculum (or sampler) if configured."""
    env_config = scenario.get("environment", {})
    spawn_curriculum = None
    phased_curriculum_enabled = scenario.get("curriculum", {}).get("type") == "phased"

    spawn_configs = env_config.get("spawn_configs", {})
    spawn_config = env_config.get("spawn_curriculum", {})
    if not spawn_configs:
        spawn_configs = spawn_config.get("spawn_configs", {})

    if spawn_config.get("enabled", False):
        from core.spawn_curriculum import SpawnCurriculumManager

        if spawn_configs:
            console_logger.print_info("Creating spawn curriculum...")
            try:
                spawn_curriculum = SpawnCurriculumManager(
                    config=spawn_config, available_spawn_points=spawn_configs
                )
                env.spawn_configs = spawn_configs
                console_logger.print_success(
                    f"Spawn curriculum: {len(spawn_curriculum.stages)} stages, "
                    f"starting at '{spawn_curriculum.current_stage.name}'"
                )
                if phased_curriculum_enabled:
                    console_logger.print_info(
                        "Phased curriculum active: spawn curriculum progression disabled"
                    )
            except Exception as exc:
                console_logger.print_warning(f"Failed to create spawn curriculum: {exc}")
                spawn_curriculum = None
        else:
            console_logger.print_warning("Spawn curriculum enabled but no spawn_configs provided")
    elif phased_curriculum_enabled and spawn_configs:
        from core.spawn_curriculum import SpawnCurriculumManager

        console_logger.print_info("Creating spawn sampler for phased curriculum...")
        try:
            spawn_curriculum = SpawnCurriculumManager(
                config={
                    "window": 1,
                    "activation_samples": 1,
                    "min_episode": 0,
                    "enable_patience": 1,
                    "disable_patience": 1,
                    "cooldown": 0,
                    "lock_speed_steps": 0,
                    "stages": [
                        {
                            "name": "phase_sampler",
                            "spawn_points": "all",
                            "speed_range": [0.0, 0.0],
                            "enable_rate": 1.0,
                        }
                    ],
                },
                available_spawn_points=spawn_configs,
            )
            env.spawn_configs = spawn_configs
            console_logger.print_success("Spawn sampler ready for phased curriculum")
        except Exception as exc:
            console_logger.print_warning(f"Failed to create spawn sampler: {exc}")
            spawn_curriculum = None

    return spawn_curriculum, spawn_configs


def build_ftg_curricula(
    scenario: Dict[str, Any],
    train_agent_id: str,
) -> Dict[str, Dict[str, Any]]:
    """Collect independently toggled FTG curricula from scenario agent configs."""
    curricula: Dict[str, Dict[str, Any]] = {}
    for agent_id, agent_cfg in scenario.get("agents", {}).items():
        if agent_id == train_agent_id:
            continue
        if str(agent_cfg.get("algorithm", "")).strip().lower() != "ftg":
            continue
        ftg_curriculum = agent_cfg.get("ftg_curriculum")
        if not isinstance(ftg_curriculum, dict):
            continue
        if not bool(ftg_curriculum.get("enabled", False)):
            continue
        curricula[str(agent_id)] = dict(ftg_curriculum)
    return curricula


class StopOnEpisodeCallback(BaseCallback):
    """Stop training after a fixed number of episodes."""

    def __init__(self, max_episodes: int, console_logger: Optional[ConsoleLogger] = None):
        super().__init__()
        self.max_episodes = max_episodes
        self.console_logger = console_logger
        self.episode_count = 0

    def _count_episode_ends(self) -> int:
        dones = self.locals.get("dones")
        if dones is None:
            terminated = self.locals.get("terminateds")
            truncated = self.locals.get("truncateds")
            if terminated is None or truncated is None:
                return 0
            done_flags = np.logical_or(terminated, truncated)
        else:
            done_flags = dones
        if isinstance(done_flags, (list, tuple, np.ndarray)):
            return int(np.sum(done_flags))
        return int(bool(done_flags))

    def _on_step(self) -> bool:
        if self.max_episodes <= 0:
            return True
        done_count = self._count_episode_ends()
        if done_count > 0:
            self.episode_count += done_count
            if self.episode_count >= self.max_episodes:
                if self.console_logger:
                    self.console_logger.print_success(
                        f"Reached {self.episode_count} episodes; stopping training."
                    )
                return False
        return True


class EpisodeProgressCallback(BaseCallback):
    """Periodic episode progress logging."""

    def __init__(
        self,
        log_every: int,
        console_logger: Optional[ConsoleLogger] = None,
        window_size: int = 100,
    ):
        super().__init__()
        self.log_every = max(1, int(log_every))
        self.console_logger = console_logger
        self.window_size = max(1, int(window_size))
        self.episode_count = 0
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        self.episode_rewards: Deque[float] = deque(maxlen=self.window_size)
        self.episode_successes: Deque[bool] = deque(maxlen=self.window_size)
        self.episode_lengths: Deque[int] = deque(maxlen=self.window_size)
        self.curriculum_callback: Optional[CurriculumCallback] = None
        self.orchestrator_callback: Optional["OrderedCurriculumOrchestratorCallback"] = None
        self.anneal_callback: Optional["AnnealOnSuccessRateCallback"] = None
        self.adaptive_hparam_callback: Optional["AdaptiveHyperparamCallback"] = None
        self.spawn_x_curriculum_config: Optional[Dict[str, Any]] = None
        self.spawn_y_curriculum_config: Optional[Dict[str, Any]] = None
        self.target_spawn_x_curriculum_config: Optional[Dict[str, Any]] = None
        self.target_spawn_y_curriculum_config: Optional[Dict[str, Any]] = None
        self.last_spawn_x_min: Optional[float] = None
        self.last_spawn_x_max: Optional[float] = None
        self.last_spawn_x_progress: Optional[float] = None
        self.last_spawn_x_window_sr: Optional[float] = None
        self.last_spawn_x_adjustment: Optional[str] = None
        self.last_spawn_y_min: Optional[float] = None
        self.last_spawn_y_max: Optional[float] = None
        self.last_spawn_y_progress: Optional[float] = None
        self.last_spawn_y_window_sr: Optional[float] = None
        self.last_spawn_y_adjustment: Optional[str] = None
        self.last_target_spawn_x_min: Optional[float] = None
        self.last_target_spawn_x_max: Optional[float] = None
        self.last_target_spawn_x_progress: Optional[float] = None
        self.last_target_spawn_x_window_sr: Optional[float] = None
        self.last_target_spawn_x_adjustment: Optional[str] = None
        self.last_target_spawn_y_min: Optional[float] = None
        self.last_target_spawn_y_max: Optional[float] = None
        self.last_target_spawn_y_progress: Optional[float] = None
        self.last_target_spawn_y_window_sr: Optional[float] = None
        self.last_target_spawn_y_adjustment: Optional[str] = None
        self.last_ftg_curriculum_agent: Optional[str] = None
        self.last_ftg_curriculum_mode: Optional[str] = None
        self.last_ftg_curriculum_progress: Optional[float] = None
        self.last_ftg_curriculum_window_sr: Optional[float] = None
        self.last_ftg_curriculum_adjustment: Optional[str] = None
        self.last_ftg_curriculum_params: Dict[str, float] = {}

    def set_curriculum_callback(self, callback: Optional[CurriculumCallback]) -> None:
        self.curriculum_callback = callback

    def set_orchestrator_callback(
        self,
        callback: Optional["OrderedCurriculumOrchestratorCallback"],
    ) -> None:
        self.orchestrator_callback = callback

    def set_anneal_callback(self, callback: Optional["AnnealOnSuccessRateCallback"]) -> None:
        self.anneal_callback = callback

    def set_adaptive_hparam_callback(self, callback: Optional["AdaptiveHyperparamCallback"]) -> None:
        self.adaptive_hparam_callback = callback

    def set_spawn_x_curriculum_config(self, cfg: Optional[Dict[str, Any]]) -> None:
        self.spawn_x_curriculum_config = dict(cfg) if isinstance(cfg, dict) else None

    def set_spawn_y_curriculum_config(self, cfg: Optional[Dict[str, Any]]) -> None:
        self.spawn_y_curriculum_config = dict(cfg) if isinstance(cfg, dict) else None

    def set_target_spawn_x_curriculum_config(self, cfg: Optional[Dict[str, Any]]) -> None:
        self.target_spawn_x_curriculum_config = dict(cfg) if isinstance(cfg, dict) else None

    def set_target_spawn_y_curriculum_config(self, cfg: Optional[Dict[str, Any]]) -> None:
        self.target_spawn_y_curriculum_config = dict(cfg) if isinstance(cfg, dict) else None

    def _count_episode_ends(self) -> int:
        dones = self.locals.get("dones")
        if dones is None:
            terminated = self.locals.get("terminateds")
            truncated = self.locals.get("truncateds")
            if terminated is None or truncated is None:
                return 0
            done_flags = np.logical_or(terminated, truncated)
        else:
            done_flags = dones
        if isinstance(done_flags, (list, tuple, np.ndarray)):
            return int(np.sum(done_flags))
        return int(bool(done_flags))

    def _collect_done_flags(self) -> List[bool]:
        dones = self.locals.get("dones")
        if dones is None:
            terminated = self.locals.get("terminateds")
            truncated = self.locals.get("truncateds")
            if terminated is None or truncated is None:
                return []
            terminated_arr = np.array(terminated, dtype=bool).reshape(-1)
            truncated_arr = np.array(truncated, dtype=bool).reshape(-1)
            return np.logical_or(terminated_arr, truncated_arr).tolist()
        return np.array(dones, dtype=bool).reshape(-1).tolist()

    def _collect_infos(self) -> List[Dict[str, Any]]:
        infos = self.locals.get("infos")
        if infos is None:
            return []
        if isinstance(infos, list):
            return [entry if isinstance(entry, dict) else {} for entry in infos]
        if isinstance(infos, tuple):
            return [entry if isinstance(entry, dict) else {} for entry in infos]
        if isinstance(infos, dict):
            return [infos]
        return []

    def _safe_metric(self, key: str) -> Optional[float]:
        logger = getattr(self.model, "logger", None)
        if logger is None:
            return None
        values = getattr(logger, "name_to_value", None)
        if not isinstance(values, dict):
            return None
        raw = values.get(key)
        if raw is None:
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    def _format_optional(self, value: Optional[float], fmt: str) -> str:
        if value is None:
            return "n/a"
        return format(value, fmt)

    def _curriculum_status(self) -> str:
        orchestrator = self.orchestrator_callback
        if orchestrator is not None and orchestrator.has_order():
            return orchestrator.status_text()

        callback = self.curriculum_callback
        if callback is None:
            parts: List[str] = []
            cfg_x = self.spawn_x_curriculum_config
            if cfg_x and bool(cfg_x.get("enabled", False)):
                mode_x = str(cfg_x.get("mode", "linear"))
                pct_x = self._spawn_x_curriculum_percent()
                if pct_x is None:
                    parts.append(f"spawn_x:{mode_x}")
                else:
                    parts.append(f"spawn_x:{mode_x} {pct_x:.1f}%")
            cfg_y = self.spawn_y_curriculum_config
            if cfg_y and bool(cfg_y.get("enabled", False)):
                mode_y = str(cfg_y.get("mode", "linear"))
                pct_y = self._spawn_y_curriculum_percent()
                if pct_y is None:
                    parts.append(f"spawn_y:{mode_y}")
                else:
                    parts.append(f"spawn_y:{mode_y} {pct_y:.1f}%")
            cfg_tx = self.target_spawn_x_curriculum_config
            if cfg_tx and bool(cfg_tx.get("enabled", False)):
                mode_tx = str(cfg_tx.get("mode", "linear"))
                pct_tx = self._target_spawn_x_curriculum_percent()
                if pct_tx is None:
                    parts.append(f"target_spawn_x:{mode_tx}")
                else:
                    parts.append(f"target_spawn_x:{mode_tx} {pct_tx:.1f}%")
            cfg_ty = self.target_spawn_y_curriculum_config
            if cfg_ty and bool(cfg_ty.get("enabled", False)):
                mode_ty = str(cfg_ty.get("mode", "linear"))
                pct_ty = self._target_spawn_y_curriculum_percent()
                if pct_ty is None:
                    parts.append(f"target_spawn_y:{mode_ty}")
                else:
                    parts.append(f"target_spawn_y:{mode_ty} {pct_ty:.1f}%")
            if not parts:
                return "off"
            return " | ".join(parts)
        try:
            phase_idx = callback.get_current_phase_index()
            phase_cfg = callback.get_current_phase_config()
            if phase_idx is None or not phase_cfg:
                return "on"
            phase_name = str(phase_cfg.get("name", "unknown"))
            return f"on p{int(phase_idx)}:{phase_name}"
        except Exception:
            return "on"

    def _anneal_status(self) -> str:
        callback = self.anneal_callback
        if callback is None:
            return "off"
        activation = callback.activation_episode
        if activation is None:
            return "waiting"
        min_rate = callback.post_anneal_min_rate
        if min_rate is None:
            return f"on@{activation}"
        return f"on@{activation},min={min_rate:.1%}"

    def _schedule_status(self) -> str:
        callback = self.adaptive_hparam_callback
        if callback is not None:
            return callback.status_text()
        return self._anneal_status()

    def _spawn_x_curriculum_percent(self) -> Optional[float]:
        cfg = self.spawn_x_curriculum_config
        if not cfg or not bool(cfg.get("enabled", False)):
            return None

        if self.last_spawn_x_progress is not None:
            try:
                progress = float(self.last_spawn_x_progress)
                return float(np.clip(progress, 0.0, 1.0) * 100.0)
            except (TypeError, ValueError):
                pass

        if self.last_spawn_x_min is None:
            return None

        mode = str(cfg.get("mode", "linear")).strip().lower()
        try:
            if mode == "success_adaptive":
                x_start = float(cfg.get("x_lower_start", cfg.get("x_start", self.last_spawn_x_min)))
                x_bound = float(cfg.get("x_lower_bound", cfg.get("x_final", x_start)))
            else:
                x_start = float(cfg.get("x_start", self.last_spawn_x_min))
                x_bound = float(cfg.get("x_final", x_start))
        except (TypeError, ValueError):
            return None

        denom = x_start - x_bound
        if abs(denom) <= 1e-9:
            return 100.0

        progress = (x_start - float(self.last_spawn_x_min)) / denom
        return float(np.clip(progress, 0.0, 1.0) * 100.0)

    def _spawn_x_status(self) -> str:
        cfg = self.spawn_x_curriculum_config
        if not cfg or not bool(cfg.get("enabled", False)):
            return "n/a"
        progress_pct = self._spawn_x_curriculum_percent()
        pct_txt = f"{progress_pct:.1f}%" if progress_pct is not None else "n/a"
        if self.last_spawn_x_min is None or self.last_spawn_x_max is None:
            return f"{pct_txt} (warming)"
        sr_txt = (
            f",sr={self.last_spawn_x_window_sr:.1%}"
            if self.last_spawn_x_window_sr is not None
            else ""
        )
        adj_txt = (
            f",adj={self.last_spawn_x_adjustment}"
            if self.last_spawn_x_adjustment
            else ""
        )
        return f"{pct_txt} x=[{self.last_spawn_x_min:.3f},{self.last_spawn_x_max:.3f}]{sr_txt}{adj_txt}"

    def _spawn_y_curriculum_percent(self) -> Optional[float]:
        cfg = self.spawn_y_curriculum_config
        if not cfg or not bool(cfg.get("enabled", False)):
            return None

        if self.last_spawn_y_progress is not None:
            try:
                progress = float(self.last_spawn_y_progress)
                return float(np.clip(progress, 0.0, 1.0) * 100.0)
            except (TypeError, ValueError):
                pass

        if self.last_spawn_y_min is None:
            return None

        mode = str(cfg.get("mode", "linear")).strip().lower()
        try:
            if mode == "success_adaptive":
                y_start = float(cfg.get("y_lower_start", cfg.get("y_start", self.last_spawn_y_min)))
                y_bound = float(cfg.get("y_lower_bound", cfg.get("y_final", y_start)))
            else:
                y_start = float(cfg.get("y_start", self.last_spawn_y_min))
                y_bound = float(cfg.get("y_final", y_start))
        except (TypeError, ValueError):
            return None

        denom = y_start - y_bound
        if abs(denom) <= 1e-9:
            return 100.0

        progress = (y_start - float(self.last_spawn_y_min)) / denom
        return float(np.clip(progress, 0.0, 1.0) * 100.0)

    def _spawn_y_status(self) -> str:
        cfg = self.spawn_y_curriculum_config
        if not cfg or not bool(cfg.get("enabled", False)):
            return "n/a"
        progress_pct = self._spawn_y_curriculum_percent()
        pct_txt = f"{progress_pct:.1f}%" if progress_pct is not None else "n/a"
        if self.last_spawn_y_min is None or self.last_spawn_y_max is None:
            return f"{pct_txt} (warming)"
        sr_txt = (
            f",sr={self.last_spawn_y_window_sr:.1%}"
            if self.last_spawn_y_window_sr is not None
            else ""
        )
        adj_txt = (
            f",adj={self.last_spawn_y_adjustment}"
            if self.last_spawn_y_adjustment
            else ""
        )
        return f"{pct_txt} y=[{self.last_spawn_y_min:.3f},{self.last_spawn_y_max:.3f}]{sr_txt}{adj_txt}"

    def _target_spawn_x_curriculum_percent(self) -> Optional[float]:
        cfg = self.target_spawn_x_curriculum_config
        if not cfg or not bool(cfg.get("enabled", False)):
            return None

        if self.last_target_spawn_x_progress is not None:
            try:
                progress = float(self.last_target_spawn_x_progress)
                return float(np.clip(progress, 0.0, 1.0) * 100.0)
            except (TypeError, ValueError):
                pass

        if self.last_target_spawn_x_min is None:
            return None

        mode = str(cfg.get("mode", "linear")).strip().lower()
        try:
            if mode == "success_adaptive":
                x_start = float(
                    cfg.get("x_lower_start", cfg.get("x_start", self.last_target_spawn_x_min))
                )
                x_bound = float(cfg.get("x_lower_bound", cfg.get("x_final", x_start)))
            else:
                x_start = float(cfg.get("x_start", self.last_target_spawn_x_min))
                x_bound = float(cfg.get("x_final", x_start))
        except (TypeError, ValueError):
            return None

        denom = x_start - x_bound
        if abs(denom) <= 1e-9:
            return 100.0

        progress = (x_start - float(self.last_target_spawn_x_min)) / denom
        return float(np.clip(progress, 0.0, 1.0) * 100.0)

    def _target_spawn_x_status(self) -> str:
        cfg = self.target_spawn_x_curriculum_config
        if not cfg or not bool(cfg.get("enabled", False)):
            return "n/a"
        progress_pct = self._target_spawn_x_curriculum_percent()
        pct_txt = f"{progress_pct:.1f}%" if progress_pct is not None else "n/a"
        if self.last_target_spawn_x_min is None or self.last_target_spawn_x_max is None:
            return f"{pct_txt} (warming)"
        sr_txt = (
            f",sr={self.last_target_spawn_x_window_sr:.1%}"
            if self.last_target_spawn_x_window_sr is not None
            else ""
        )
        adj_txt = (
            f",adj={self.last_target_spawn_x_adjustment}"
            if self.last_target_spawn_x_adjustment
            else ""
        )
        return (
            f"{pct_txt} x=[{self.last_target_spawn_x_min:.3f},{self.last_target_spawn_x_max:.3f}]"
            f"{sr_txt}{adj_txt}"
        )

    def _target_spawn_y_curriculum_percent(self) -> Optional[float]:
        cfg = self.target_spawn_y_curriculum_config
        if not cfg or not bool(cfg.get("enabled", False)):
            return None

        if self.last_target_spawn_y_progress is not None:
            try:
                progress = float(self.last_target_spawn_y_progress)
                return float(np.clip(progress, 0.0, 1.0) * 100.0)
            except (TypeError, ValueError):
                pass

        if self.last_target_spawn_y_min is None:
            return None

        mode = str(cfg.get("mode", "linear")).strip().lower()
        try:
            if mode == "success_adaptive":
                y_start = float(
                    cfg.get("y_lower_start", cfg.get("y_start", self.last_target_spawn_y_min))
                )
                y_bound = float(cfg.get("y_lower_bound", cfg.get("y_final", y_start)))
            else:
                y_start = float(cfg.get("y_start", self.last_target_spawn_y_min))
                y_bound = float(cfg.get("y_final", y_start))
        except (TypeError, ValueError):
            return None

        denom = y_start - y_bound
        if abs(denom) <= 1e-9:
            return 100.0

        progress = (y_start - float(self.last_target_spawn_y_min)) / denom
        return float(np.clip(progress, 0.0, 1.0) * 100.0)

    def _target_spawn_y_status(self) -> str:
        cfg = self.target_spawn_y_curriculum_config
        if not cfg or not bool(cfg.get("enabled", False)):
            return "n/a"
        progress_pct = self._target_spawn_y_curriculum_percent()
        pct_txt = f"{progress_pct:.1f}%" if progress_pct is not None else "n/a"
        if self.last_target_spawn_y_min is None or self.last_target_spawn_y_max is None:
            return f"{pct_txt} (warming)"
        sr_txt = (
            f",sr={self.last_target_spawn_y_window_sr:.1%}"
            if self.last_target_spawn_y_window_sr is not None
            else ""
        )
        adj_txt = (
            f",adj={self.last_target_spawn_y_adjustment}"
            if self.last_target_spawn_y_adjustment
            else ""
        )
        return (
            f"{pct_txt} y=[{self.last_target_spawn_y_min:.3f},{self.last_target_spawn_y_max:.3f}]"
            f"{sr_txt}{adj_txt}"
        )

    def _ftg_curriculum_status(self) -> str:
        if self.last_ftg_curriculum_agent is None:
            return "n/a"

        mode_txt = (
            f"{self.last_ftg_curriculum_mode} "
            if self.last_ftg_curriculum_mode
            else ""
        )
        pct_txt = (
            f"{self.last_ftg_curriculum_progress:.1%}"
            if self.last_ftg_curriculum_progress is not None
            else "n/a"
        )
        sr_txt = (
            f",sr={self.last_ftg_curriculum_window_sr:.1%}"
            if self.last_ftg_curriculum_window_sr is not None
            else ""
        )
        adj_txt = (
            f",adj={self.last_ftg_curriculum_adjustment}"
            if self.last_ftg_curriculum_adjustment
            else ""
        )
        params_txt = ""
        if self.last_ftg_curriculum_params:
            param_parts = []
            for key in sorted(self.last_ftg_curriculum_params.keys()):
                value = self.last_ftg_curriculum_params[key]
                param_parts.append(f"{key}={value:.3f}")
            params_txt = " " + ",".join(param_parts[:3])

        return f"{self.last_ftg_curriculum_agent} {mode_txt}{pct_txt}{sr_txt}{adj_txt}{params_txt}"

    def _on_step(self) -> bool:
        if "rewards" in self.locals:
            rewards = self.locals["rewards"]
            if isinstance(rewards, (list, np.ndarray)):
                step_reward = float(rewards[0])
            else:
                step_reward = float(rewards)
            self.current_episode_reward += step_reward
            self.current_episode_length += 1

        done_flags = self._collect_done_flags()
        if not done_flags:
            return True
        infos = self._collect_infos()
        done_count = 0
        for idx, done in enumerate(done_flags):
            if not done:
                continue
            done_count += 1
            info = infos[idx] if idx < len(infos) else {}
            success = bool(info.get("is_success", False))
            x_min = info.get("spawn_x_min")
            x_max = info.get("spawn_x_max")
            x_prog = info.get("spawn_x_progress")
            x_sr = info.get("spawn_x_window_sr")
            x_adj = info.get("spawn_x_last_adjustment")
            y_min = info.get("spawn_y_min")
            y_max = info.get("spawn_y_max")
            y_prog = info.get("spawn_y_progress")
            y_sr = info.get("spawn_y_window_sr")
            y_adj = info.get("spawn_y_last_adjustment")
            tx_min = info.get("target_spawn_x_min")
            tx_max = info.get("target_spawn_x_max")
            tx_prog = info.get("target_spawn_x_progress")
            tx_sr = info.get("target_spawn_x_window_sr")
            tx_adj = info.get("target_spawn_x_last_adjustment")
            ty_min = info.get("target_spawn_y_min")
            ty_max = info.get("target_spawn_y_max")
            ty_prog = info.get("target_spawn_y_progress")
            ty_sr = info.get("target_spawn_y_window_sr")
            ty_adj = info.get("target_spawn_y_last_adjustment")
            ftg_agent = info.get("ftg_curriculum_agent")
            ftg_mode = info.get("ftg_curriculum_mode")
            ftg_prog = info.get("ftg_curriculum_progress")
            ftg_sr = info.get("ftg_curriculum_window_sr")
            ftg_adj = info.get("ftg_curriculum_last_adjustment")
            if x_min is not None:
                try:
                    self.last_spawn_x_min = float(x_min)
                except (TypeError, ValueError):
                    pass
            if x_max is not None:
                try:
                    self.last_spawn_x_max = float(x_max)
                except (TypeError, ValueError):
                    pass
            if x_prog is not None:
                try:
                    self.last_spawn_x_progress = float(x_prog)
                except (TypeError, ValueError):
                    pass
            if x_sr is not None:
                try:
                    self.last_spawn_x_window_sr = float(x_sr)
                except (TypeError, ValueError):
                    pass
            if x_adj is not None:
                self.last_spawn_x_adjustment = str(x_adj)
            if y_min is not None:
                try:
                    self.last_spawn_y_min = float(y_min)
                except (TypeError, ValueError):
                    pass
            if y_max is not None:
                try:
                    self.last_spawn_y_max = float(y_max)
                except (TypeError, ValueError):
                    pass
            if y_prog is not None:
                try:
                    self.last_spawn_y_progress = float(y_prog)
                except (TypeError, ValueError):
                    pass
            if y_sr is not None:
                try:
                    self.last_spawn_y_window_sr = float(y_sr)
                except (TypeError, ValueError):
                    pass
            if y_adj is not None:
                self.last_spawn_y_adjustment = str(y_adj)
            if tx_min is not None:
                try:
                    self.last_target_spawn_x_min = float(tx_min)
                except (TypeError, ValueError):
                    pass
            if tx_max is not None:
                try:
                    self.last_target_spawn_x_max = float(tx_max)
                except (TypeError, ValueError):
                    pass
            if tx_prog is not None:
                try:
                    self.last_target_spawn_x_progress = float(tx_prog)
                except (TypeError, ValueError):
                    pass
            if tx_sr is not None:
                try:
                    self.last_target_spawn_x_window_sr = float(tx_sr)
                except (TypeError, ValueError):
                    pass
            if tx_adj is not None:
                self.last_target_spawn_x_adjustment = str(tx_adj)
            if ty_min is not None:
                try:
                    self.last_target_spawn_y_min = float(ty_min)
                except (TypeError, ValueError):
                    pass
            if ty_max is not None:
                try:
                    self.last_target_spawn_y_max = float(ty_max)
                except (TypeError, ValueError):
                    pass
            if ty_prog is not None:
                try:
                    self.last_target_spawn_y_progress = float(ty_prog)
                except (TypeError, ValueError):
                    pass
            if ty_sr is not None:
                try:
                    self.last_target_spawn_y_window_sr = float(ty_sr)
                except (TypeError, ValueError):
                    pass
            if ty_adj is not None:
                self.last_target_spawn_y_adjustment = str(ty_adj)
            if ftg_agent is not None:
                self.last_ftg_curriculum_agent = str(ftg_agent)
            if ftg_mode is not None:
                self.last_ftg_curriculum_mode = str(ftg_mode)
            if ftg_prog is not None:
                try:
                    self.last_ftg_curriculum_progress = float(ftg_prog)
                except (TypeError, ValueError):
                    pass
            if ftg_sr is not None:
                try:
                    self.last_ftg_curriculum_window_sr = float(ftg_sr)
                except (TypeError, ValueError):
                    pass
            if ftg_adj is not None:
                self.last_ftg_curriculum_adjustment = str(ftg_adj)
            ftg_params: Dict[str, float] = {}
            if isinstance(info, dict):
                for metric_key, metric_value in info.items():
                    if not isinstance(metric_key, str):
                        continue
                    if not metric_key.startswith("ftg_"):
                        continue
                    if metric_key.startswith("ftg_curriculum_"):
                        continue
                    try:
                        ftg_params[metric_key[len("ftg_"):]] = float(metric_value)
                    except (TypeError, ValueError):
                        continue
            if ftg_params:
                self.last_ftg_curriculum_params = ftg_params
            self.episode_count += 1
            self.episode_rewards.append(float(self.current_episode_reward))
            self.episode_successes.append(success)
            self.episode_lengths.append(int(self.current_episode_length))
            self.current_episode_reward = 0.0
            self.current_episode_length = 0

        if done_count > 0 and self.console_logger and self.episode_count % self.log_every == 0:
            sr = (
                float(sum(self.episode_successes) / len(self.episode_successes))
                if self.episode_successes
                else 0.0
            )
            r_mean = float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0
            l_mean = float(np.mean(self.episode_lengths)) if self.episode_lengths else 0.0
            approx_kl = self._safe_metric("train/approx_kl")
            clip_frac = self._safe_metric("train/clip_fraction")
            val_loss = self._safe_metric("train/value_loss")
            pol_loss = self._safe_metric("train/policy_gradient_loss")
            ent_loss = self._safe_metric("train/entropy_loss")
            msg = (
                f"Ep {self.episode_count} | SR{self.window_size} {sr:.1%} | "
                f"R{self.window_size} {r_mean:.2f} | L{self.window_size} {l_mean:.1f} | "
                f"KL {self._format_optional(approx_kl, '.4f')} | "
                f"Clip {self._format_optional(clip_frac, '.3f')} | "
                f"VLoss {self._format_optional(val_loss, '.3f')} | "
                f"PLoss {self._format_optional(pol_loss, '.3f')} | "
                f"Ent {self._format_optional(ent_loss, '.3f')} | "
                f"Curr {self._curriculum_status()} | "
                f"SpawnX {self._spawn_x_status()} | "
                f"SpawnY {self._spawn_y_status()} | "
                f"TSpawnX {self._target_spawn_x_status()} | "
                f"TSpawnY {self._target_spawn_y_status()} | "
                f"FTG {self._ftg_curriculum_status()} | "
                f"Sched {self._schedule_status()}"
            )
            printed_rich = False
            if RICH_PANEL_AVAILABLE and hasattr(self.console_logger, "console"):
                try:
                    progress_table = Table.grid(expand=True)
                    progress_table.add_column(justify="left")
                    progress_table.add_column(justify="left")
                    progress_table.add_column(justify="left")
                    progress_table.add_row(
                        f"[bold cyan]Episode[/bold cyan] {self.episode_count}",
                        f"[bold cyan]SR{self.window_size}[/bold cyan] {sr:.1%}",
                        f"[bold cyan]R{self.window_size}[/bold cyan] {r_mean:.2f}",
                    )
                    progress_table.add_row(
                        f"[bold cyan]L{self.window_size}[/bold cyan] {l_mean:.1f}",
                        f"[bold cyan]Curriculum[/bold cyan] {self._curriculum_status()}",
                        f"[bold cyan]Sched[/bold cyan] {self._schedule_status()}",
                    )
                    progress_table.add_row(
                        f"[bold cyan]SpawnX[/bold cyan] {self._spawn_x_status()}",
                        f"[bold cyan]SpawnY[/bold cyan] {self._spawn_y_status()}",
                        f"[bold cyan]FTG[/bold cyan] {self._ftg_curriculum_status()}",
                    )
                    progress_table.add_row(
                        f"[bold cyan]TargetSpawnX[/bold cyan] {self._target_spawn_x_status()}",
                        f"[bold cyan]TargetSpawnY[/bold cyan] {self._target_spawn_y_status()}",
                        "",
                    )

                    ppo_table = Table.grid(expand=True)
                    ppo_table.add_column(justify="left")
                    ppo_table.add_column(justify="left")
                    ppo_table.add_column(justify="left")
                    ppo_table.add_column(justify="left")
                    ppo_table.add_column(justify="left")
                    ppo_table.add_row(
                        f"[bold magenta]KL[/bold magenta] {self._format_optional(approx_kl, '.4f')}",
                        f"[bold magenta]Clip[/bold magenta] {self._format_optional(clip_frac, '.3f')}",
                        f"[bold magenta]VLoss[/bold magenta] {self._format_optional(val_loss, '.3f')}",
                        f"[bold magenta]PLoss[/bold magenta] {self._format_optional(pol_loss, '.3f')}",
                        f"[bold magenta]Ent[/bold magenta] {self._format_optional(ent_loss, '.3f')}",
                    )

                    container = Table.grid(expand=True)
                    container.add_row(progress_table)
                    container.add_row(ppo_table)

                    self.console_logger.console.print(
                        Panel(
                            container,
                            title="SB3 Snapshot",
                            border_style="blue",
                            padding=(0, 1),
                        )
                    )
                    printed_rich = True
                except Exception:
                    printed_rich = False
            if not printed_rich:
                self.console_logger.print_info(msg)
        return True


class OrderedCurriculumOrchestratorCallback(BaseCallback):
    """Gate curriculum/schedule stages by ordered completion with optional backtracking."""

    STAGE_ALIASES: Dict[str, str] = {
        "x": "x",
        "spawn_x": "x",
        "x_spawn": "x",
        "y": "y",
        "spawn_y": "y",
        "y_spawn": "y",
        "tx": "tx",
        "target_x": "tx",
        "x_target": "tx",
        "target_spawn_x": "tx",
        "spawn_target_x": "tx",
        "ty": "ty",
        "target_y": "ty",
        "y_target": "ty",
        "target_spawn_y": "ty",
        "spawn_target_y": "ty",
        "ftg": "ftg",
        "learning_rate": "learning_rate",
        "lr": "learning_rate",
        "clip": "clip_range",
        "clip_range": "clip_range",
        "ent": "ent_coef",
        "ent_coef": "ent_coef",
    }

    INFO_PROGRESS_KEYS: Dict[str, str] = {
        "x": "spawn_x_progress",
        "y": "spawn_y_progress",
        "tx": "target_spawn_x_progress",
        "ty": "target_spawn_y_progress",
        "ftg": "ftg_curriculum_progress",
    }
    SUPPORTED_STAGES = [
        "x",
        "y",
        "tx",
        "ty",
        "ftg",
        "learning_rate",
        "clip_range",
        "ent_coef",
    ]

    def __init__(
        self,
        *,
        order: List[str],
        trigger_threshold: float = 1.0,
        stable_updates_required: int = 2,
        allow_backtrack: bool = False,
        backtrack_success_threshold: float = 0.65,
        backtrack_window_episodes: int = 100,
        backtrack_patience_updates: int = 2,
        console_logger: Optional[ConsoleLogger] = None,
        adaptive_hparam_callback: Optional["AdaptiveHyperparamCallback"] = None,
    ):
        super().__init__()
        self.requested_order = [str(item) for item in order]
        self.order: List[str] = []
        self.trigger_threshold = float(trigger_threshold)
        self.stable_updates_required = max(1, int(stable_updates_required))
        self.allow_backtrack = bool(allow_backtrack)
        self.backtrack_success_threshold = float(backtrack_success_threshold)
        self.backtrack_window_episodes = max(1, int(backtrack_window_episodes))
        self.backtrack_patience_updates = max(1, int(backtrack_patience_updates))
        self.console_logger = console_logger
        self.adaptive_hparam_callback = adaptive_hparam_callback

        self.active_index = 0
        self.stable_counter = 0
        self.backtrack_low_streak = 0
        self.episode_count = 0
        self.success_history: Deque[bool] = deque(maxlen=self.backtrack_window_episodes)

    def has_order(self) -> bool:
        return bool(self.order)

    def _normalize_stage(self, stage: str) -> Optional[str]:
        key = str(stage).strip().lower()
        normalized = self.STAGE_ALIASES.get(key)
        if normalized in {"x", "y", "tx", "ty", "ftg", "learning_rate", "clip_range", "ent_coef"}:
            return normalized
        return None

    def _collect_done_flags(self):
        dones = self.locals.get("dones")
        if dones is None:
            terminated = self.locals.get("terminateds")
            truncated = self.locals.get("truncateds")
            if terminated is None or truncated is None:
                return []
            terminated_arr = np.array(terminated, dtype=bool).reshape(-1)
            truncated_arr = np.array(truncated, dtype=bool).reshape(-1)
            return np.logical_or(terminated_arr, truncated_arr).tolist()
        return np.array(dones, dtype=bool).reshape(-1).tolist()

    def _collect_infos(self):
        infos = self.locals.get("infos")
        if infos is None:
            return []
        if isinstance(infos, list):
            return infos
        if isinstance(infos, tuple):
            return list(infos)
        if isinstance(infos, dict):
            return [infos]
        return []

    def _env_method_first(self, method: str, *args):
        try:
            result = self.training_env.env_method(method, *args)
        except Exception:
            return None
        if isinstance(result, list):
            return result[0] if result else None
        return result

    def _stage_available(self, stage: str) -> bool:
        if stage in {"learning_rate", "clip_range", "ent_coef"}:
            callback = self.adaptive_hparam_callback
            return bool(callback and callback.has_stage(stage))
        available = self._env_method_first("has_curriculum_stage", stage)
        return bool(available)

    def _set_stage_mode(self, stage: str, mode: str) -> None:
        if stage in {"learning_rate", "clip_range", "ent_coef"}:
            callback = self.adaptive_hparam_callback
            if callback is not None:
                callback.set_stage_mode(stage, mode)
            return
        _ = self._env_method_first("set_curriculum_stage_mode", stage, mode)

    def _apply_stage_modes(self) -> None:
        stage_mode: Dict[str, str] = {}
        for stage in self.SUPPORTED_STAGES:
            if not self._stage_available(stage):
                continue
            stage_mode[stage] = "off"
        for idx, stage in enumerate(self.order):
            if stage not in stage_mode:
                continue
            if idx < self.active_index:
                mode = "frozen"
            elif idx == self.active_index:
                mode = "active"
            else:
                mode = "off"
            stage_mode[stage] = mode
        for stage, mode in stage_mode.items():
            self._set_stage_mode(stage, mode)

    def _stage_progress(self, stage: str, info: Optional[Dict[str, Any]]) -> Optional[float]:
        if stage in self.INFO_PROGRESS_KEYS and isinstance(info, dict):
            raw = info.get(self.INFO_PROGRESS_KEYS[stage])
            if raw is not None:
                try:
                    return float(np.clip(float(raw), 0.0, 1.0))
                except (TypeError, ValueError):
                    pass

        if stage in {"learning_rate", "clip_range", "ent_coef"}:
            callback = self.adaptive_hparam_callback
            if callback is None:
                return None
            progress = callback.stage_progress(stage)
            if progress is None:
                return None
            return float(np.clip(float(progress), 0.0, 1.0))

        progress = self._env_method_first("get_curriculum_stage_progress", stage)
        if progress is None:
            return None
        try:
            return float(np.clip(float(progress), 0.0, 1.0))
        except (TypeError, ValueError):
            return None

    def _current_stage(self) -> Optional[str]:
        if not self.order:
            return None
        if self.active_index < 0 or self.active_index >= len(self.order):
            return None
        return self.order[self.active_index]

    def _try_backtrack(self) -> bool:
        if not self.allow_backtrack:
            return False
        if self.active_index <= 0:
            return False
        if len(self.success_history) < self.backtrack_window_episodes:
            return False
        if self.episode_count % self.backtrack_window_episodes != 0:
            return False

        rolling_sr = float(sum(self.success_history) / len(self.success_history))
        if rolling_sr < self.backtrack_success_threshold:
            self.backtrack_low_streak += 1
        else:
            self.backtrack_low_streak = 0

        if self.backtrack_low_streak < self.backtrack_patience_updates:
            return False

        prev_stage = self._current_stage()
        self.active_index = max(0, self.active_index - 1)
        self.stable_counter = 0
        self.backtrack_low_streak = 0
        self._apply_stage_modes()
        if self.console_logger:
            self.console_logger.print_warning(
                f"Orchestrator backtrack: {prev_stage} -> {self._current_stage()} "
                f"(rolling_sr={rolling_sr:.1%})"
            )
        return True

    def _try_advance(self, info: Optional[Dict[str, Any]]) -> bool:
        stage = self._current_stage()
        if stage is None:
            return False

        progress = self._stage_progress(stage, info)
        if progress is None:
            return False
        if progress >= self.trigger_threshold:
            self.stable_counter += 1
        else:
            self.stable_counter = 0

        if self.stable_counter < self.stable_updates_required:
            return False
        if self.active_index >= len(self.order) - 1:
            return False

        prev_stage = stage
        self.active_index += 1
        self.stable_counter = 0
        self.backtrack_low_streak = 0
        self._apply_stage_modes()
        if self.console_logger:
            self.console_logger.print_success(
                f"Orchestrator advance: {prev_stage} -> {self._current_stage()} "
                f"(progress={progress:.1%})"
            )
        return True

    def _on_training_start(self) -> None:
        seen = set()
        resolved: List[str] = []
        for raw_stage in self.requested_order:
            stage = self._normalize_stage(raw_stage)
            if stage is None or stage in seen:
                continue
            if not self._stage_available(stage):
                if self.console_logger:
                    self.console_logger.print_warning(
                        f"Orchestrator skipping unavailable stage '{raw_stage}'."
                    )
                continue
            seen.add(stage)
            resolved.append(stage)

        self.order = resolved
        if not self.order:
            return

        self.active_index = 0
        self.stable_counter = 0
        self.backtrack_low_streak = 0
        self.episode_count = 0
        self.success_history.clear()
        self._apply_stage_modes()
        if self.console_logger:
            chain = " > ".join(self.order)
            self.console_logger.print_info(f"Orchestrator order: {chain}")

    def _on_step(self) -> bool:
        done_flags = self._collect_done_flags()
        if not done_flags or not self.order:
            return True

        infos = self._collect_infos()
        for idx, done in enumerate(done_flags):
            if not done:
                continue
            info = infos[idx] if idx < len(infos) and isinstance(infos[idx], dict) else {}
            success = bool(info.get("is_success", False))
            self.episode_count += 1
            self.success_history.append(success)
            if self._try_backtrack():
                continue
            self._try_advance(info)
        return True

    def status_text(self) -> str:
        if not self.order:
            return "off"
        stage = self._current_stage()
        if stage is None:
            return "off"
        pos = self.active_index + 1
        total = len(self.order)
        stage_progress = self._stage_progress(stage, None)
        progress_txt = f"{stage_progress:.0%}" if stage_progress is not None else "n/a"
        suffix = ""
        if self.allow_backtrack and self.success_history:
            rolling_sr = float(sum(self.success_history) / len(self.success_history))
            suffix = f",bk_sr={rolling_sr:.0%}"
        return f"order {pos}/{total}:{stage} p={progress_txt}{suffix}"


class RenderCallback(BaseCallback):
    """Render the environment during training."""

    def __init__(self, render_every: int = 1):
        super().__init__()
        self.render_every = max(1, int(render_every))

    def _on_step(self) -> bool:
        if self.n_calls % self.render_every == 0:
            try:
                self.training_env.env_method("render")
            except Exception:
                return True
        return True


class _MutableScalarSchedule:
    """Callable schedule that always returns an externally updated scalar."""

    def __init__(self, value: float):
        self.value = float(value)

    def set(self, value: float) -> None:
        self.value = float(value)

    def __call__(self, progress_remaining: float) -> float:
        _ = progress_remaining
        return self.value


class AdaptiveHyperparamCallback(BaseCallback):
    """Update optimizer hyperparameters using success-adaptive controllers."""

    def __init__(
        self,
        specs: List[AdaptiveHyperparamSpec],
        console_logger: Optional[ConsoleLogger] = None,
    ):
        super().__init__()
        self.specs = specs
        self.console_logger = console_logger
        self.episode_count = 0
        self._lr_schedule: Optional[_MutableScalarSchedule] = None
        self._clip_schedule: Optional[_MutableScalarSchedule] = None
        self._spec_by_key: Dict[str, AdaptiveHyperparamSpec] = {spec.key: spec for spec in specs}
        self._stage_mode: Dict[str, str] = {spec.key: "active" for spec in specs}
        self._initial_value: Dict[str, float] = {
            spec.key: float(spec.controller.value) for spec in specs
        }

    def _collect_done_flags(self):
        dones = self.locals.get("dones")
        if dones is None:
            terminated = self.locals.get("terminateds")
            truncated = self.locals.get("truncateds")
            if terminated is None or truncated is None:
                return []
            terminated_arr = np.array(terminated, dtype=bool).reshape(-1)
            truncated_arr = np.array(truncated, dtype=bool).reshape(-1)
            return np.logical_or(terminated_arr, truncated_arr).tolist()
        return np.array(dones, dtype=bool).reshape(-1).tolist()

    def _collect_infos(self):
        infos = self.locals.get("infos")
        if infos is None:
            return []
        if isinstance(infos, list):
            return infos
        if isinstance(infos, tuple):
            return list(infos)
        if isinstance(infos, dict):
            return [infos]
        return []

    def _apply_value(self, key: str, value: float) -> None:
        model = self.model
        if key == "learning_rate":
            if self._lr_schedule is None:
                self._lr_schedule = _MutableScalarSchedule(value)
                model.lr_schedule = self._lr_schedule
            else:
                self._lr_schedule.set(value)
            optimizer = getattr(getattr(model, "policy", None), "optimizer", None)
            if optimizer is not None and hasattr(model, "_update_learning_rate"):
                try:
                    model._update_learning_rate(optimizer)
                except Exception:
                    pass
            return

        if key == "clip_range":
            if not hasattr(model, "clip_range"):
                return
            if self._clip_schedule is None:
                self._clip_schedule = _MutableScalarSchedule(value)
                model.clip_range = self._clip_schedule
            else:
                self._clip_schedule.set(value)
            return

        if key == "ent_coef":
            if hasattr(model, "ent_coef"):
                model.ent_coef = float(value)

    def _on_training_start(self) -> None:
        for spec in self.specs:
            mode = self._stage_mode.get(spec.key, "active")
            if mode == "off":
                spec.controller.reset(value=self._initial_value.get(spec.key))
            self._apply_value(spec.key, spec.controller.value)

    def _on_step(self) -> bool:
        done_flags = self._collect_done_flags()
        if not done_flags:
            return True

        infos = self._collect_infos()
        for idx, done in enumerate(done_flags):
            if not done:
                continue
            info = infos[idx] if idx < len(infos) and isinstance(infos[idx], dict) else {}
            success = bool(info.get("is_success", False))
            self.episode_count += 1
            for spec in self.specs:
                mode = self._stage_mode.get(spec.key, "active")
                if mode != "active":
                    continue
                updated_cycle = spec.controller.observe(success)
                if not updated_cycle:
                    continue
                self._apply_value(spec.key, spec.controller.value)
                if self.console_logger and spec.controller.last_adjustment != "hold":
                    sr = spec.controller.last_window_sr
                    sr_txt = f"{sr:.1%}" if sr is not None else "n/a"
                    self.console_logger.print_info(
                        f"{spec.key} {spec.controller.last_adjustment} -> {spec.controller.value:.6g} "
                        f"(sr={sr_txt}, ep={self.episode_count})"
                    )
        return True

    def has_stage(self, stage: str) -> bool:
        return stage in self._spec_by_key

    def stage_progress(self, stage: str) -> Optional[float]:
        spec = self._spec_by_key.get(stage)
        if spec is None:
            return None
        return spec.controller.progress()

    def set_stage_mode(self, stage: str, mode: str) -> None:
        spec = self._spec_by_key.get(stage)
        if spec is None:
            return
        mode_key = str(mode).strip().lower()
        if mode_key not in {"active", "frozen", "off"}:
            mode_key = "off"
        self._stage_mode[stage] = mode_key
        if mode_key == "off":
            spec.controller.reset(value=self._initial_value.get(stage))
        self._apply_value(stage, spec.controller.value)

    def status_text(self) -> str:
        if not self.specs:
            return "off"
        parts = []
        for spec in self.specs:
            key = spec.key
            short = "lr" if key == "learning_rate" else ("clip" if key == "clip_range" else "ent")
            value = spec.controller.value
            mode = self._stage_mode.get(key, "active")
            if key == "learning_rate":
                value_txt = f"{value:.2e}"
            else:
                value_txt = f"{value:.4f}"
            sr = spec.controller.last_window_sr
            sr_txt = f",sr={sr:.0%}" if sr is not None else ""
            adj_txt = f",adj={spec.controller.last_adjustment}" if mode == "active" else ""
            parts.append(f"{short}={value_txt}[{mode}]{sr_txt}{adj_txt}")
        return " | ".join(parts)


class AnnealOnSuccessRateCallback(BaseCallback):
    """Activate gated schedules once rolling training success meets configured thresholds."""

    def __init__(
        self,
        gated_schedules: List[GatedScheduleSpec],
        console_logger: Optional[ConsoleLogger] = None,
    ):
        super().__init__()
        self.gated_schedules = gated_schedules
        self.console_logger = console_logger
        max_window = max((spec.window_episodes for spec in gated_schedules), default=1)
        self.success_history: Deque[bool] = deque(maxlen=max_window)
        self.episode_count = 0
        self._activation_episode: Optional[int] = None
        self._post_anneal_breached = False
        self._post_anneal_min_rate: Optional[float] = None

        self._monitor_spec: Optional[GatedScheduleSpec] = None
        for spec in gated_schedules:
            if spec.key == "learning_rate":
                self._monitor_spec = spec
                break
        if self._monitor_spec is None and gated_schedules:
            self._monitor_spec = gated_schedules[0]

    def _collect_done_flags(self):
        dones = self.locals.get("dones")
        if dones is None:
            terminated = self.locals.get("terminateds")
            truncated = self.locals.get("truncateds")
            if terminated is None or truncated is None:
                return []
            terminated_arr = np.array(terminated, dtype=bool).reshape(-1)
            truncated_arr = np.array(truncated, dtype=bool).reshape(-1)
            return np.logical_or(terminated_arr, truncated_arr).tolist()
        return np.array(dones, dtype=bool).reshape(-1).tolist()

    def _collect_infos(self):
        infos = self.locals.get("infos")
        if infos is None:
            return []
        if isinstance(infos, list):
            return infos
        if isinstance(infos, tuple):
            return list(infos)
        if isinstance(infos, dict):
            return [infos]
        return []

    def _on_step(self) -> bool:
        done_flags = self._collect_done_flags()
        if not done_flags:
            return True

        infos = self._collect_infos()
        for idx, done in enumerate(done_flags):
            if not done:
                continue
            info = infos[idx] if idx < len(infos) and isinstance(infos[idx], dict) else {}
            success = bool(info.get("is_success", False))
            self.success_history.append(success)
            self.episode_count += 1

        if not self.success_history:
            return True

        for spec in self.gated_schedules:
            if spec.schedule.activated:
                continue
            if len(self.success_history) < spec.window_episodes:
                continue

            recent = list(self.success_history)[-spec.window_episodes :]
            success_rate = float(sum(recent) / len(recent))
            if success_rate < spec.success_threshold:
                continue

            progress_remaining = getattr(self.model, "_current_progress_remaining", None)
            spec.schedule.activate(progress_remaining=progress_remaining)
            if self._monitor_spec is spec and self._activation_episode is None:
                self._activation_episode = int(self.episode_count)
            if self.console_logger:
                self.console_logger.print_success(
                    f"Activated {spec.key} annealing at episode {self.episode_count} "
                    f"(rolling success={success_rate:.1%} over {spec.window_episodes} episodes)."
                )

        self._update_post_anneal_guard()
        return True

    def _update_post_anneal_guard(self) -> None:
        spec = self._monitor_spec
        if spec is None:
            return
        if self._activation_episode is None:
            return
        if len(self.success_history) < spec.window_episodes:
            return

        recent = list(self.success_history)[-spec.window_episodes :]
        rolling_rate = float(sum(recent) / len(recent))
        if self._post_anneal_min_rate is None or rolling_rate < self._post_anneal_min_rate:
            self._post_anneal_min_rate = rolling_rate
        if rolling_rate < spec.success_threshold:
            self._post_anneal_breached = True

    @property
    def activation_episode(self) -> Optional[int]:
        return self._activation_episode

    @property
    def post_anneal_min_rate(self) -> Optional[float]:
        return self._post_anneal_min_rate

    def is_post_anneal_stable(self) -> bool:
        spec = self._monitor_spec
        if spec is None:
            return False
        if self._activation_episode is None:
            return False
        if self._post_anneal_breached:
            return False
        if self._post_anneal_min_rate is None:
            return False
        return self._post_anneal_min_rate >= spec.success_threshold

    def monitor_threshold(self) -> Optional[float]:
        if self._monitor_spec is None:
            return None
        return float(self._monitor_spec.success_threshold)


def main() -> None:
    args = parse_args()

    try:
        scenario = load_and_expand_scenario(args.scenario)
    except ScenarioError as exc:
        print(f"Error loading scenario: {exc}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print(f"Scenario file not found: {args.scenario}", file=sys.stderr)
        sys.exit(1)

    scenario = resolve_cli_overrides(scenario, args)

    train_agent_id, algorithm = select_on_policy_agent(scenario)

    run_id = args.run_id or resolve_run_id(
        scenario_name=scenario.get("experiment", {}).get("name"),
        algorithm=algorithm,
        seed=scenario.get("experiment", {}).get("seed"),
    )
    set_run_id_env(run_id)

    wandb_logger, console_logger = initialize_loggers(scenario, args, run_id=run_id)
    if wandb_logger is not None:
        apply_wandb_sweep_overrides(scenario, console_logger)

    console_logger.print_header(
        f"Training: {scenario['experiment']['name']}",
        f"Episodes: {scenario['experiment'].get('episodes', 'N/A')}",
    )

    env, agents, reward_strategies = create_training_setup(scenario, mode="train")

    train_agent_cfg = scenario["agents"][train_agent_id]
    train_params = train_agent_cfg.get("params", {})
    action_constraints = train_agent_cfg.get("action_constraints", {})

    frame_stack = int(train_agent_cfg.get("frame_stack", 1) or 1)
    if frame_stack < 1:
        frame_stack = 1

    env_config = scenario.get("environment", {})
    action_repeat = parse_action_repeat(env_config)

    target_id = train_agent_cfg.get("target_id")
    observation_preset = infer_observation_preset(train_agent_cfg)
    action_stack = resolve_action_stack(train_agent_cfg)

    obs_space = env.observation_spaces.get(train_agent_id)
    action_space = env.action_spaces.get(train_agent_id)
    if obs_space is None or action_space is None:
        raise ValueError(f"Agent '{train_agent_id}' not found in environment spaces.")

    obs_dim = compute_obs_dim(
        obs_space,
        action_space,
        observation_preset,
        target_id,
        frame_stack,
        action_stack=action_stack,
    )

    if not isinstance(action_space, spaces.Box):
        raise ValueError("On-policy SB3 runner expects continuous action spaces.")
    action_low = action_space.low
    action_high = action_space.high

    spawn_curriculum, spawn_configs = build_spawn_curriculum(env, scenario, console_logger)
    ftg_curricula = build_ftg_curricula(scenario, train_agent_id)

    reward_strategy = reward_strategies.get(train_agent_id)
    sb3_env = SB3SingleAgentWrapper(
        env,
        agent_id=train_agent_id,
        obs_dim=obs_dim,
        action_low=action_low,
        action_high=action_high,
        observation_preset=observation_preset,
        target_id=target_id,
        reward_strategy=reward_strategy,
        spawn_curriculum=spawn_curriculum,
        frame_stack=frame_stack,
        action_stack=action_stack,
        action_repeat=action_repeat,
        action_constraints=action_constraints,
        spawn_x_curriculum=env_config.get("spawn_x_curriculum"),
        spawn_y_curriculum=env_config.get("spawn_y_curriculum"),
        target_spawn_x_curriculum=env_config.get("target_spawn_x_curriculum"),
        target_spawn_y_curriculum=env_config.get("target_spawn_y_curriculum"),
        ftg_curriculum=ftg_curricula,
    )

    other_agents = {aid: agent for aid, agent in agents.items() if aid != train_agent_id}
    sb3_env.set_other_agents(other_agents)

    monitor_env = Monitor(
        sb3_env,
        info_keywords=("is_success", "outcome", "target_finished", "target_collision", "collision"),
    )

    device = train_agent_cfg.get("device", train_params.get("device", "cuda"))
    policy_kwargs = build_policy_kwargs(train_params, "PPO" if "ppo" in algorithm else "A2C")
    model, gated_schedules = build_model(
        algorithm,
        train_params,
        monitor_env,
        policy_kwargs,
        device,
        scenario["experiment"].get("seed"),
    )
    adaptive_hparam_specs = _resolve_adaptive_hparam_specs(algorithm, train_params)
    adaptive_hparam_keys = {spec.key for spec in adaptive_hparam_specs}
    if adaptive_hparam_keys and console_logger:
        keys_text = ", ".join(sorted(adaptive_hparam_keys))
        console_logger.print_info(f"Adaptive hyperparameter control enabled: {keys_text}")
    if adaptive_hparam_keys:
        gated_schedules = [spec for spec in gated_schedules if spec.key not in adaptive_hparam_keys]
    maybe_apply_warm_start(
        model=model,
        algorithm=algorithm,
        train_params=train_params,
        console_logger=console_logger,
    )

    episodes = int(scenario["experiment"].get("episodes", 0) or 0)
    if episodes <= 0:
        raise ValueError("Scenario must define a positive experiment.episodes for on-policy runs.")

    max_steps = int(env_config.get("max_steps", 2500))
    decision_steps = max_steps
    if action_repeat > 1:
        decision_steps = int(math.ceil(max_steps / action_repeat))
    total_timesteps = decision_steps * episodes

    callbacks = [StopOnEpisodeCallback(episodes, console_logger)]
    orchestrator_callback: Optional[OrderedCurriculumOrchestratorCallback] = None
    adaptive_hparam_callback: Optional[AdaptiveHyperparamCallback] = None
    if adaptive_hparam_specs:
        adaptive_hparam_callback = AdaptiveHyperparamCallback(adaptive_hparam_specs, console_logger)

    orchestrator_cfg = env_config.get("curriculum_orchestrator", {})
    if isinstance(orchestrator_cfg, dict) and bool(orchestrator_cfg.get("enabled", False)):
        order_cfg = orchestrator_cfg.get("order", [])
        if isinstance(order_cfg, list):
            order = [str(item) for item in order_cfg]
        else:
            order = []
        orchestrator_callback = OrderedCurriculumOrchestratorCallback(
            order=order,
            trigger_threshold=float(orchestrator_cfg.get("trigger_threshold", 1.0)),
            stable_updates_required=int(orchestrator_cfg.get("stable_updates_required", 2)),
            allow_backtrack=bool(orchestrator_cfg.get("allow_backtrack", False)),
            backtrack_success_threshold=float(
                orchestrator_cfg.get("backtrack_success_threshold", 0.65)
            ),
            backtrack_window_episodes=int(orchestrator_cfg.get("backtrack_window_episodes", 100)),
            backtrack_patience_updates=int(orchestrator_cfg.get("backtrack_patience_updates", 2)),
            console_logger=console_logger,
            adaptive_hparam_callback=adaptive_hparam_callback,
        )
        callbacks.append(orchestrator_callback)

    if adaptive_hparam_callback is not None:
        callbacks.append(adaptive_hparam_callback)

    anneal_guard_callback: Optional[AnnealOnSuccessRateCallback] = None
    if gated_schedules:
        anneal_guard_callback = AnnealOnSuccessRateCallback(gated_schedules, console_logger)
        callbacks.append(anneal_guard_callback)

    progress_callback: Optional[EpisodeProgressCallback] = None
    log_every_env = os.environ.get("F110_LOG_EVERY_EPISODES")
    if log_every_env is None:
        log_every = 25
    else:
        try:
            log_every = int(log_every_env)
        except ValueError:
            log_every = 25
    if log_every > 0:
        progress_callback = EpisodeProgressCallback(log_every, console_logger)
        if orchestrator_callback is not None:
            progress_callback.set_orchestrator_callback(orchestrator_callback)
        progress_callback.set_spawn_x_curriculum_config(env_config.get("spawn_x_curriculum"))
        progress_callback.set_spawn_y_curriculum_config(env_config.get("spawn_y_curriculum"))
        progress_callback.set_target_spawn_x_curriculum_config(
            env_config.get("target_spawn_x_curriculum")
        )
        progress_callback.set_target_spawn_y_curriculum_config(
            env_config.get("target_spawn_y_curriculum")
        )
        if adaptive_hparam_callback is not None:
            progress_callback.set_adaptive_hparam_callback(adaptive_hparam_callback)
        if anneal_guard_callback is not None:
            progress_callback.set_anneal_callback(anneal_guard_callback)
        callbacks.append(progress_callback)
    if env_config.get("render"):
        render_every_env = os.environ.get("F110_RENDER_EVERY_STEPS")
        if render_every_env is None:
            render_every = 1
        else:
            try:
                render_every = int(render_every_env)
            except (TypeError, ValueError):
                render_every = 1
        callbacks.append(RenderCallback(render_every))

    ftg_agents = {}
    ftg_schedules = {}
    for agent_id, agent_cfg in scenario.get("agents", {}).items():
        if agent_id == train_agent_id:
            continue
        if agent_cfg.get("algorithm", "").lower() == "ftg":
            if agent_id in agents:
                ftg_agents[agent_id] = agents[agent_id]
            schedule = agent_cfg.get("ftg_schedule")
            if isinstance(schedule, dict):
                ftg_schedules[agent_id] = schedule

    eval_cfg = scenario.get("evaluation", {})
    curriculum_config = scenario.get("curriculum")
    curriculum_callback = None
    if curriculum_config or spawn_curriculum:
        eval_gate_schedule_enabled = bool(eval_cfg.get("gate_eval_schedule", True))
        eval_gate_advancement_enabled = bool(eval_cfg.get("gate_phase_advancement", True))
        curriculum_callback = CurriculumCallback(
            curriculum_config=curriculum_config,
            spawn_curriculum=spawn_curriculum,
            ftg_agents=ftg_agents,
            ftg_schedules=ftg_schedules,
            env_wrapper=sb3_env,
            wandb_run=wandb_logger.run if wandb_logger else None,
            wandb_logging=scenario.get("wandb", {}).get("logging"),
            algo_name=algorithm,
            eval_gate_enabled=bool(
                eval_cfg.get("enabled", False)
                and eval_cfg.get("frequency", 0)
                and eval_gate_advancement_enabled
            ),
            eval_gate_schedule_enabled=bool(
                eval_cfg.get("enabled", False)
                and eval_cfg.get("frequency", 0)
                and eval_gate_schedule_enabled
            ),
        )
        callbacks.append(curriculum_callback)
        if progress_callback is not None:
            progress_callback.set_curriculum_callback(curriculum_callback)

    if eval_cfg.get("enabled", False):
        eval_config = EvaluationConfig(
            num_episodes=eval_cfg.get("num_episodes", 10),
            deterministic=eval_cfg.get("deterministic", True),
            spawn_points=eval_cfg.get("spawn_points", ["spawn_pinch_left", "spawn_pinch_right"]),
            spawn_speeds=eval_cfg.get("spawn_speeds", [0.44, 0.44]),
            lock_speed_steps=eval_cfg.get("lock_speed_steps", 0),
            ftg_override=eval_cfg.get("ftg_override", {}),
            max_steps=max_steps,
            rolling_window=eval_cfg.get("rolling_window"),
        )
        eval_env_raw, eval_agents, _ = create_training_setup(scenario, mode="eval")
        eval_other_agents = {aid: agent for aid, agent in eval_agents.items() if aid != train_agent_id}

        eval_env = SB3SingleAgentWrapper(
            eval_env_raw,
            agent_id=train_agent_id,
            obs_dim=obs_dim,
            action_low=action_low,
            action_high=action_high,
            observation_preset=observation_preset,
            target_id=target_id,
            reward_strategy=reward_strategy,
            frame_stack=frame_stack,
            action_stack=action_stack,
            action_repeat=action_repeat,
            spawn_x_curriculum=env_config.get("spawn_x_curriculum"),
            spawn_y_curriculum=env_config.get("spawn_y_curriculum"),
            target_spawn_x_curriculum=env_config.get("target_spawn_x_curriculum"),
            target_spawn_y_curriculum=env_config.get("target_spawn_y_curriculum"),
        )
        eval_env.set_other_agents(eval_other_agents)

        eval_every = eval_cfg.get("frequency", 100)
        if eval_every:
            callbacks.append(
                SB3EvaluationCallback(
                    eval_env=eval_env,
                    evaluation_config=eval_config,
                    spawn_configs=spawn_configs,
                    eval_every_n_episodes=eval_every,
                    wandb_run=wandb_logger.run if wandb_logger else None,
                    wandb_logging=scenario.get("wandb", {}).get("logging"),
                    curriculum_callback=curriculum_callback,
                    verbose=1,
                )
            )

    if wandb_logger and wandb_logger.run and curriculum_callback is None:
        callbacks.append(
            SB3TrainLoggingCallback(
                wandb_run=wandb_logger.run,
                wandb_logging=scenario.get("wandb", {}).get("logging"),
            )
        )

    if wandb_logger and wandb_logger.run and _WandbCallback is not None:
        if wandb_logger.should_log("sb3_callbacks"):
            callbacks.append(_WandbCallback(verbose=0))

    callback = CallbackList(callbacks) if callbacks else None

    console_logger.print_info("Starting SB3 on-policy training...")
    try:
        model.learn(total_timesteps=total_timesteps, callback=callback)
        experiment_cfg = scenario.get("experiment", {})
        save_model_at_end = bool(experiment_cfg.get("save_model_at_end", True))
        if save_model_at_end:
            out_dir = ROOT_DIR / "outputs" / "best_sb3" / scenario["experiment"]["name"]
            out_dir.mkdir(parents=True, exist_ok=True)
            model_name_raw = experiment_cfg.get("save_model_name")
            if isinstance(model_name_raw, str) and model_name_raw.strip():
                model_name = model_name_raw.strip()
            else:
                model_name = f"{run_id}_final"
            if model_name.endswith(".zip"):
                model_name = model_name[:-4]
            model_path = out_dir / model_name
            model.save(str(model_path))
            console_logger.print_success(
                f"Saved end-of-run model: {model_path}.zip"
            )
        else:
            console_logger.print_info("Skipping end-of-run model save: experiment.save_model_at_end=false.")
    finally:
        if wandb_logger:
            wandb_logger.finish()


if __name__ == "__main__":
    main()
