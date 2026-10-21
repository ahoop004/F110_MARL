"""Evaluate one independent MAPPO actor without loading its team critic."""
from __future__ import annotations

import numpy as np
import torch

from agents.common.networks import Actor
from utils.torch_io import resolve_device, validate_checkpoint_compatibility


class SoloActorObservations:
    """Keep the source vehicle identity while the solo environment uses car_0."""

    def __init__(self, composer, source_agent: str):
        self.composer = composer
        self.source_agent = source_agent

    def __getattr__(self, name):
        return getattr(self.composer, name)

    def wrap(self, raw_obs, info=None):
        info = dict(info or {})
        if info.get("frenet_neighbors"):
            raise ValueError("Solo actor evaluation requires empty neighbor slots")
        info["agent_id"] = self.source_agent
        return self.composer.wrap(raw_obs, info)


class SoloMAPPOActor:
    """Deterministic local-policy inference; no critic, optimizer or rollout buffer."""

    def __init__(self, checkpoint, source_agent, composer, action_low, action_high, params):
        validate_checkpoint_compatibility(checkpoint, {
            "algorithm": "mappo", "actor_mode": "independent", "lora_contract": None,
            "action_dim": len(action_low), "action_low": action_low, "action_high": action_high,
            "action_contract": params["_action_contract"],
            "physics_contract": params["_physics_contract"],
            "actor_hidden_dims": list(params.get("pi_hidden_dims", params.get("hidden_dims", [256, 256]))),
            "activation": params.get("activation", "tanh"),
        })
        actors = checkpoint.get("actors") or {}
        if source_agent not in actors or source_agent not in checkpoint.get("agent_ids", []):
            raise ValueError(f"Checkpoint has no independent actor {source_agent!r}; available: {list(actors)}")
        contract = checkpoint.get("observation_contracts", {}).get(source_agent)
        validate_checkpoint_compatibility({
            "observation_contract": contract,
            "obs_dim": checkpoint.get("obs_dims", {}).get(source_agent),
        }, {"observation_contract": composer.contract, "obs_dim": composer.obs_dim})
        neighbors = contract["observation"].get("frenet_neighbors", {})
        if not neighbors.get("enabled") or source_agent not in neighbors.get("agent_ids", []):
            raise ValueError("Solo actor evaluation requires Frenet neighbors with the original vehicle IDs")

        self.device = resolve_device([params.get("device", "cpu")])
        self.actor = Actor(composer.obs_dim, len(action_low), checkpoint["actor_hidden_dims"],
                           checkpoint["activation"]).to(self.device)
        validate_checkpoint_compatibility(checkpoint, {}, states=[
            ("actor", actors[source_agent], self.actor.state_dict())])
        self.actor.load_state_dict(actors[source_agent], strict=True)
        self.actor.eval()
        self.observations = SoloActorObservations(composer, source_agent)

    def predict(self, observation):
        with torch.inference_mode():
            obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
            action, _ = self.actor.get_action(obs, deterministic=True)
        return action[0].cpu().numpy().astype(np.float32, copy=False)
