"""Import one 1v1 LoRA policy into a heterogeneous team without its critic."""
from copy import deepcopy
import hashlib
from pathlib import Path

import torch

from utils.torch_io import safe_load


def observation_layout(contract):
    if not isinstance(contract, dict):
        raise ValueError("Adapter transfer requires explicit observation contracts")
    common = deepcopy(contract)
    obs = common["observation"]
    allowed = {"lidar", "frenet_vehicle_track", "frenet_neighbors", "target_frenet"}
    if {k for k, v in obs.items() if isinstance(v, dict) and v.get("enabled")} - allowed:
        raise ValueError("Adapter transfer requires the LiDAR/Frenet driving layout")
    if not all(obs.get(k, {}).get("enabled") for k in ("lidar", "frenet_vehicle_track")):
        raise ValueError("Adapter transfer requires the LiDAR/Frenet driving layout")
    driving = int(common["lidar_beams"]) + 10 + 2 * int(obs["frenet_vehicle_track"].get("points", 20))
    neighbors = obs.pop("frenet_neighbors", {})
    target = obs.pop("target_frenet", {})
    extra = 0
    if neighbors.get("enabled"):
        ids = len(neighbors.get("agent_ids") or [])
        extra = int(neighbors.get("max_neighbors", 1)) * (5 + int(neighbors.get("include_team", False)) + ids) + ids
    target_start = driving + extra
    width = target_start + (5 + len(target.get("agent_ids") or []) if target.get("enabled") else 0)
    return common, driving, target_start, width, target


def import_adapter(agent, path, *, source_agent, target_agent):
    if not agent.per_agent_adapters or not agent.lora_config.get("per_agent_log_std"):
        raise ValueError("Adapter transfer requires per_agent LoRA with per_agent_log_std")
    checkpoint = safe_load(path, map_location=agent.device)
    source_lora = checkpoint.get("lora_contract") or {}
    if checkpoint.get("algorithm") != "mappo" or source_agent not in source_lora.get("agent_to_adapter", {}):
        raise ValueError("Adapter source must be a MAPPO LoRA checkpoint containing source_agent")
    source_contract = checkpoint.get("observation_contracts", {}).get(source_agent, checkpoint.get("observation_contract"))
    common, driving, source_target, source_width, target_fields = observation_layout(source_contract)
    if source_target != driving or not target_fields.get("enabled"):
        raise ValueError("Adapter source must use the 1v1 driving + target_frenet layout")
    layouts = {}
    for aid in agent.agent_ids:
        layout = observation_layout(agent.observation_contracts[aid])
        layouts[aid] = layout
    target_start = layouts[target_agent][2]

    source = checkpoint["actor"]
    state = deepcopy(agent.actor.state_dict())
    first = source["net.0.weight"]
    # The common driving base is frozen. Other observation columns have zero
    # base weights; each learner's adapter sees its own unpadded input layout.
    for key in agent.actor.base_state_dict():
        if key == "log_std":
            continue  # Racer exploration starts at its configured initialization.
        value = source.get(key)
        if key == "net.0.weight":
            value = torch.zeros_like(state[key])
            value[:, :driving] = first[:, :driving]
        if value is None or value.shape != state[key].shape:
            raise ValueError(f"Incompatible frozen base tensor {key}")
        state[key] = value.clone()
    source_bank = source_lora["agent_to_adapter"][source_agent]
    target_bank = agent._agent_index[target_agent]
    for layer in agent.actor.target_layers:
        for name in ("A", "B"):
            key = f"adapters.{target_bank}.{layer}.{name}"
            value = source[f"adapters.{source_bank}.{layer}.{name}"]
            if layer == 0 and name == "A":
                if value.shape != (agent.lora_config["rank"], source_width):
                    raise ValueError("Incompatible source adapter input width")
                expanded = torch.zeros_like(state[key])
                expanded[:, :driving] = value[:, :driving]
                expanded[:, target_start:target_start + 5] = value[:, source_target:source_target + 5]
                value = expanded
            if value.shape != state[key].shape:
                raise ValueError(f"Incompatible adapter tensor {key}")
            state[key] = value.clone()
    std = source[f"log_stds.{source_bank}"] if source_lora.get("per_agent_log_std") else source["log_std"]
    if std.shape != state[f"log_stds.{target_bank}"].shape:
        raise ValueError("Incompatible source exploration shape")
    state[f"log_stds.{target_bank}"] = std.clone()
    agent.actor.load_state_dict(state, strict=True)
    agent.pretrained_actor_source = dict(
        path=str(Path(path).resolve()), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        algorithm="mappo", transfer="single_adapter_v1", source_agent=source_agent,
        target_agent=target_agent, original_source=checkpoint.get("pretrained_actor_source"))
    agent._lora_ready = True
