#!/usr/bin/env python3
"""Average per-phase checkpoints across runs (score-weighted).

Example:
  python tools/average_phase_checkpoints.py \
    --checkpoint-root outputs/checkpoints \
    --output-dir outputs/averaged \
    --agent-id car_0
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch


NON_TRAINABLE = {"ftg", "pp", "pure_pursuit"}


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r") as handle:
        return json.load(handle)


def find_trainable_agent(config: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
    for agent_id, agent_cfg in config.get("agents", {}).items():
        algo = str(agent_cfg.get("algorithm", "")).lower()
        if algo and algo not in NON_TRAINABLE:
            return agent_id, agent_cfg
    return None


def build_fingerprint(config: Dict[str, Any]) -> Optional[str]:
    agent_entry = find_trainable_agent(config)
    if not agent_entry:
        return None

    agent_id, agent_cfg = agent_entry
    fingerprint = {
        "agent_id": agent_id,
        "algorithm": agent_cfg.get("algorithm"),
        "params": agent_cfg.get("params", {}),
        "observation": agent_cfg.get("observation", {}),
        "action_set": agent_cfg.get("action_set") or agent_cfg.get("params", {}).get("action_set"),
        "reward": agent_cfg.get("reward", {}),
        "normalize_observations": config.get("experiment", {}).get("normalize_observations"),
    }

    return json.dumps(fingerprint, sort_keys=True)


def average_state_dicts(
    state_dicts: List[Dict[str, torch.Tensor]],
    weights: List[float],
) -> Dict[str, torch.Tensor]:
    if not state_dicts:
        raise ValueError("No state dicts provided for averaging")

    keys = list(state_dicts[0].keys())
    for state in state_dicts[1:]:
        if set(state.keys()) != set(keys):
            raise ValueError("State dict keys do not match across checkpoints")

    weight_sum = sum(weights)
    if weight_sum <= 0.0:
        weights = [1.0 for _ in weights]
        weight_sum = float(len(weights))

    avg_state: Dict[str, torch.Tensor] = {}
    for key in keys:
        accum = None
        for state, weight in zip(state_dicts, weights):
            tensor = state[key]
            if not isinstance(tensor, torch.Tensor):
                tensor = torch.as_tensor(tensor)
            tensor = tensor.to(dtype=torch.float32, device="cpu")
            if accum is None:
                accum = tensor * weight
            else:
                accum = accum + tensor * weight
        orig_value = state_dicts[0][key]
        if isinstance(orig_value, torch.Tensor):
            avg_state[key] = (accum / weight_sum).to(dtype=orig_value.dtype)
        else:
            avg_state[key] = accum / weight_sum

    return avg_state


def collect_phase_checkpoints(
    root: Path,
    agent_id: Optional[str],
    strict_config: bool,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    candidates: List[Dict[str, Any]] = []
    expected_fingerprint: Optional[str] = None
    metadata_template: Optional[Dict[str, Any]] = None

    for ckpt_path in root.rglob("*_eval_phase*_best_ep*.pt"):
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        checkpoint_type = checkpoint.get("checkpoint_type", "")
        if not str(checkpoint_type).startswith("best_eval_phase"):
            continue

        training_state = checkpoint.get("training_state", {})
        phase_index = training_state.get("phase_index")
        if phase_index is None:
            continue

        run_metadata_path = ckpt_path.parent / "run_metadata.json"
        if not run_metadata_path.exists():
            continue
        run_metadata = load_json(run_metadata_path)
        config_snapshot = run_metadata.get("config_snapshot", {})
        fingerprint = build_fingerprint(config_snapshot)
        if fingerprint is None:
            continue

        if expected_fingerprint is None:
            expected_fingerprint = fingerprint
            metadata_template = run_metadata
        elif strict_config and fingerprint != expected_fingerprint:
            continue

        agent_states = checkpoint.get("agent_states", {})
        if not agent_states:
            continue

        target_agent_id = agent_id or next(iter(agent_states.keys()))
        if target_agent_id not in agent_states:
            continue

        agent_state = agent_states[target_agent_id]
        policy_state = agent_state.get("policy")
        if policy_state is None:
            continue

        metric_value = checkpoint.get("metric_value")
        if metric_value is None:
            metric_value = training_state.get("eval_result", {}).get("success_rate", 0.0)

        candidates.append({
            "path": ckpt_path,
            "phase_index": int(phase_index),
            "phase_name": training_state.get("phase_name"),
            "metric_value": float(metric_value) if metric_value is not None else 0.0,
            "policy_state": policy_state,
            "agent_id": target_agent_id,
        })

    return candidates, metadata_template


def main() -> None:
    parser = argparse.ArgumentParser(description="Average per-phase checkpoints across runs.")
    parser.add_argument("--checkpoint-root", type=str, default="outputs/checkpoints")
    parser.add_argument("--output-dir", type=str, default="outputs/averaged")
    parser.add_argument("--agent-id", type=str, default=None)
    parser.add_argument("--phase-index", type=int, default=None)
    parser.add_argument("--min-runs", type=int, default=2)
    parser.add_argument("--allow-mismatch", action="store_true", default=False)
    args = parser.parse_args()

    root = Path(args.checkpoint_root)
    if not root.exists():
        raise SystemExit(f"Checkpoint root not found: {root}")

    candidates, metadata_template = collect_phase_checkpoints(
        root=root,
        agent_id=args.agent_id,
        strict_config=not args.allow_mismatch,
    )

    if not candidates:
        raise SystemExit("No phase-best checkpoints found to average.")

    scenario_name = "unknown"
    if metadata_template:
        scenario_name = metadata_template.get("scenario_name", "unknown")

    by_phase: Dict[int, List[Dict[str, Any]]] = {}
    for item in candidates:
        if args.phase_index is not None and item["phase_index"] != args.phase_index:
            continue
        by_phase.setdefault(item["phase_index"], []).append(item)

    out_dir = Path(args.output_dir) / scenario_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if metadata_template:
        metadata_path = out_dir / "run_metadata.json"
        if not metadata_path.exists():
            with metadata_path.open("w") as handle:
                json.dump(metadata_template, handle, indent=2)

    for phase_index, items in sorted(by_phase.items()):
        state_dicts = [item["policy_state"] for item in items]
        weights = [item["metric_value"] for item in items]
        if len(state_dicts) < args.min_runs:
            print(f"Skipping phase {phase_index}: only {len(state_dicts)} runs (min {args.min_runs})")
            continue
        avg_state = average_state_dicts(state_dicts, weights)

        phase_name = next(
            (item["phase_name"] for item in items if item.get("phase_name")), None
        )

        agent_id = items[0]["agent_id"]
        weight_sum = sum(weights)
        if weight_sum > 0.0:
            weighted_metric = sum(w * m for w, m in zip(weights, weights)) / weight_sum
        else:
            weighted_metric = sum(weights) / max(1, len(weights))

        checkpoint = {
            "episode": -1,
            "agent_states": {
                agent_id: {"policy": avg_state}
            },
            "checkpoint_type": "avg_eval_phase",
            "metric_value": weighted_metric,
            "training_state": {
                "phase_index": phase_index,
                "phase_name": phase_name,
                "averaging": "score_weighted",
                "source_checkpoints": [str(item["path"]) for item in items],
                "source_weights": weights,
            },
        }

        output_path = out_dir / f"avg_eval_phase{phase_index}_weighted.pt"
        torch.save(checkpoint, output_path)
        print(f"Wrote {output_path} ({len(items)} checkpoints)")


if __name__ == "__main__":
    main()
