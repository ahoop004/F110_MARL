"""Minimal PPO implementation tailored for the F110 MARL experiments."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from gymnasium.spaces import Box
from gymnasium.spaces import utils as space_utils

from algos.ppo.net import PPOActorCritic


@dataclass
class PPOConfig:
    rollout_steps: int = 4096
    num_epochs: int = 10
    mini_batch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    lr: float = 3e-4
    adam_eps: float = 1e-5
    hidden_sizes: Tuple[int, ...] = (256, 256)
    device: str | None = None
    seed: int | None = None


class RolloutBuffer:
    """Container for on-policy data and advantage computation."""

    def __init__(self) -> None:
        self.observations: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.log_probs: List[float] = []
        self.advantages: List[float] = []
        self.returns: List[float] = []
        self.values: List[float] = []

    def __len__(self) -> int:
        return len(self.observations)

    def clear(self) -> None:
        self.observations.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.advantages.clear()
        self.returns.clear()
        self.values.clear()

    def add_paths(
        self,
        paths: Dict[str, List[Dict[str, Any]]],
        gamma: float,
        gae_lambda: float,
    ) -> Tuple[int, Dict[str, float]]:
        steps_added = 0
        episode_returns: Dict[str, float] = {}

        for agent_id, transitions in paths.items():
            if not transitions:
                continue

            episode_returns[agent_id] = float(sum(t["reward"] for t in transitions))

            advantages, returns = _compute_gae(transitions, gamma, gae_lambda)
            for idx, transition in enumerate(transitions):
                self.observations.append(np.asarray(transition["obs"], dtype=np.float32))
                self.actions.append(np.asarray(transition["action"], dtype=np.float32))
                self.log_probs.append(float(transition["log_prob"]))
                self.values.append(float(transition["value"]))
                self.advantages.append(float(advantages[idx]))
                self.returns.append(float(returns[idx]))
                steps_added += 1

        return steps_added, episode_returns

    def as_tensors(self, device: torch.device) -> Dict[str, torch.Tensor]:
        obs = torch.as_tensor(np.asarray(self.observations, dtype=np.float32), device=device)
        actions = torch.as_tensor(np.asarray(self.actions, dtype=np.float32), device=device)
        log_probs = torch.as_tensor(np.asarray(self.log_probs, dtype=np.float32), device=device)
        returns = torch.as_tensor(np.asarray(self.returns, dtype=np.float32), device=device)
        advantages = torch.as_tensor(np.asarray(self.advantages, dtype=np.float32), device=device)
        values = torch.as_tensor(np.asarray(self.values, dtype=np.float32), device=device)
        return {
            "obs": obs,
            "actions": actions,
            "log_probs": log_probs,
            "returns": returns,
            "advantages": advantages,
            "values": values,
        }


def _compute_gae(
    transitions: List[Dict[str, Any]],
    gamma: float,
    gae_lambda: float,
) -> Tuple[List[float], List[float]]:
    advantages: List[float] = []
    returns: List[float] = []
    gae = 0.0
    next_value = 0.0

    for step in reversed(transitions):
        reward = float(step["reward"])
        value = float(step["value"])
        done = bool(step["done"])
        mask = 1.0 - float(done)
        delta = reward + gamma * next_value * mask - value
        gae = delta + gamma * gae_lambda * mask * gae
        advantages.append(gae)
        returns.append(gae + value)
        next_value = value

    advantages.reverse()
    returns.reverse()
    return advantages, returns


class PPOTrainer:
    """Collects rollouts and optimises the PPO objective."""

    def __init__(
        self,
        env,
        config: PPOConfig | None = None,
        *,
        render: bool = False,
        render_every: int = 1,
    ) -> None:
        if config is None:
            config = PPOConfig()
        self.config = config

        if config.seed is not None:
            torch.manual_seed(int(config.seed))
            np.random.seed(int(config.seed))

        self.env = env
        representative_agent = env.possible_agents[0]
        obs_space = env.observation_space(representative_agent)
        action_space = env.action_space(representative_agent)
        if not isinstance(action_space, Box):
            raise TypeError("PPOTrainer expects Box action spaces")

        device_str = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(device_str, str) and device_str.lower().startswith("cuda") and not torch.cuda.is_available():
            print("[WARN] CUDA requested but no GPU detected; falling back to CPU.")
            device_str = "cpu"
        try:
            self.device = torch.device(device_str)
        except (TypeError, RuntimeError) as exc:
            print(f"[WARN] Unable to use device '{device_str}': {exc}. Falling back to CPU.")
            self.device = torch.device("cpu")

        self.policy = PPOActorCritic(
            obs_space=obs_space,
            action_space=action_space,
            hidden_sizes=config.hidden_sizes,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=config.lr,
            eps=config.adam_eps,
        )

        self._flat_obs_space = obs_space
        self._flatten_obs = lambda observation: space_utils.flatten(
            self._flat_obs_space, observation
        ).astype(np.float32, copy=False)

        self._action_low = action_space.low
        self._action_high = action_space.high

        self._need_reset = True
        self._render = bool(render)
        self._render_every = max(1, int(render_every))
        self._step_counter = 0

    def collect_rollout(self) -> Tuple[RolloutBuffer, Dict[str, List[float]]]:
        buffer = RolloutBuffer()
        episode_returns: Dict[str, List[float]] = defaultdict(list)

        if self._need_reset:
            obs, _ = self.env.reset(seed=self.config.seed)
            self._need_reset = False
        else:
            obs, _ = self.env.reset()

        while len(buffer) < self.config.rollout_steps:
            paths: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

            done = False
            while not done:
                actions: Dict[str, np.ndarray] = {}
                cached: Dict[str, Dict[str, Any]] = {}
                for aid in list(self.env.agents):
                    agent_obs = obs.get(aid)
                    if agent_obs is None:
                        continue
                    obs_vec = self._flatten_obs(agent_obs)
                    obs_tensor = torch.as_tensor(obs_vec, device=self.device)
                    with torch.no_grad():
                        clipped, log_prob, value, raw_action = self.policy.act(obs_tensor)

                    cached[aid] = {
                        "obs": obs_vec,
                        "log_prob": log_prob.item(),
                        "value": value.item(),
                        "action": raw_action.detach().cpu().numpy(),
                    }
                    action_np = clipped.detach().cpu().numpy()
                    actions[aid] = np.clip(action_np, self._action_low, self._action_high)

                next_obs, rewards, terminations, truncations, _ = self.env.step(actions)
                self._step_counter += 1
                if self._render and (self._step_counter % self._render_every == 0):
                    try:
                        self.env.render()
                    except Exception as exc:
                        print(f"[WARN] Render call failed: {exc}")

                for aid, data in cached.items():
                    reward = float(rewards.get(aid, 0.0))
                    terminated = bool(terminations.get(aid, False))
                    truncated = bool(truncations.get(aid, False))
                    transition = {
                        "obs": data["obs"],
                        "action": data["action"],
                        "log_prob": data["log_prob"],
                        "value": data["value"],
                        "reward": reward,
                        "done": terminated or truncated,
                    }
                    paths[aid].append(transition)

                obs = next_obs
                done = not self.env.agents

            _, episodic = buffer.add_paths(paths, self.config.gamma, self.config.gae_lambda)
            for aid, value in episodic.items():
                episode_returns[aid].append(value)

            if len(buffer) >= self.config.rollout_steps:
                break

            obs, _ = self.env.reset()

        return buffer, episode_returns

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        data = buffer.as_tensors(self.device)
        advantages = data["advantages"]
        if advantages.numel() == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        clip_coef = self.config.clip_coef
        ent_coef = self.config.ent_coef
        vf_coef = self.config.vf_coef
        batch_size = advantages.size(0)
        minibatch = min(self.config.mini_batch_size, batch_size)

        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
        }
        total_updates = 0

        for _ in range(self.config.num_epochs):
            indices = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, minibatch):
                end = start + minibatch
                mb_idx = indices[start:end]
                obs_mb = data["obs"][mb_idx]
                actions_mb = data["actions"][mb_idx]
                old_logprob_mb = data["log_probs"][mb_idx]
                returns_mb = data["returns"][mb_idx]
                adv_mb = advantages[mb_idx]

                logprob, entropy, values = self.policy.evaluate_actions(obs_mb, actions_mb)
                ratio = torch.exp(logprob - old_logprob_mb)
                surrogate1 = ratio * adv_mb
                surrogate2 = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * adv_mb
                policy_loss = -torch.min(surrogate1, surrogate2).mean()

                value_loss = torch.nn.functional.mse_loss(values, returns_mb)
                entropy_loss = -entropy.mean()

                loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                metrics["policy_loss"] += float(policy_loss.detach().cpu())
                metrics["value_loss"] += float(value_loss.detach().cpu())
                metrics["entropy"] += float(entropy.mean().detach().cpu())
                total_updates += 1

        if total_updates > 0:
            for key in metrics:
                metrics[key] /= total_updates

        return metrics

    def train(self, updates: int) -> List[Dict[str, Any]]:
        history: List[Dict[str, Any]] = []
        for _ in range(updates):
            buffer, ep_returns = self.collect_rollout()
            update_metrics = self.update(buffer)
            buffer.clear()
            history.append({
                "returns": {aid: np.mean(vals) if vals else 0.0 for aid, vals in ep_returns.items()},
                "metrics": update_metrics,
            })
        self._need_reset = True
        return history
