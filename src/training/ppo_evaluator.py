"""Deterministic, fact-based evaluation used for PPO checkpoint selection."""
from __future__ import annotations

import random
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch

from metrics.racing_eval import (
    aggregate_eval_episodes,
    create_episode_facts,
    finalize_episode_facts,
    update_agent_step_facts,
)


class DeterministicPPOEvaluator:
    """Evaluate one PPO racer without touching its rollout buffer or optimizer."""

    def __init__(
        self,
        *,
        env: Any,
        rl_agent_id: str,
        other_agents: Mapping[str, Any],
        obs_composer: Any,
        action_composer: Any,
        episodes: int,
        base_seed: int,
        action_repeat: int = 1,
    ) -> None:
        self.env = env
        self.rl_agent_id = rl_agent_id
        self.other_agents = dict(other_agents)
        self.obs_composer = obs_composer
        self.action_composer = action_composer
        self.episodes = max(1, int(episodes))
        self.base_seed = int(base_seed)
        self.action_repeat = max(1, int(action_repeat))

    def _actions(
        self,
        action_phys: np.ndarray,
        obs_dict: Dict[str, Any],
        active: set[str],
    ) -> Dict[str, np.ndarray]:
        actions: Dict[str, np.ndarray] = {}
        if self.rl_agent_id in active:
            actions[self.rl_agent_id] = action_phys
        for aid, controller in self.other_agents.items():
            if aid not in active:
                continue
            try:
                action = controller.act(obs_dict.get(aid, {}))
            except Exception:
                action = np.zeros(2, dtype=np.float32)
            actions[aid] = np.asarray(action, dtype=np.float32)
        return actions

    def _agent_states(self, agent_ids: Sequence[str]) -> Dict[str, Any]:
        states: Dict[str, Any] = {}
        for aid in agent_ids:
            try:
                states[aid] = self.env.get_agent_state(aid)
            except (KeyError, ValueError):
                continue
        return states

    def evaluate(self, agent: Any | None = None) -> Dict[str, Any]:
        # ``agent`` is accepted for easy use by callers and tests; production
        # code binds it once via ``bind_agent``.
        active_agent = agent or getattr(self, "agent", None)
        if active_agent is None:
            raise ValueError("DeterministicPPOEvaluator requires a bound PPO agent.")

        all_agent_ids = list(getattr(self.env, "possible_agents", [self.rl_agent_id]))
        opponent_ids = [aid for aid in all_agent_ids if aid != self.rl_agent_id]
        results = []
        actor_was_training = bool(active_agent.actor.training)
        numpy_rng_state = np.random.get_state()
        python_rng_state = random.getstate()
        torch_rng_state = torch.random.get_rng_state()
        cuda_rng_states = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        active_agent.actor.eval()
        try:
            with torch.no_grad():
                for episode in range(self.episodes):
                    obs_dict, info_dict = self.env.reset(seed=self.base_seed + episode)
                    self.obs_composer.reset()
                    for controller in self.other_agents.values():
                        if hasattr(controller, "reset"):
                            controller.reset()

                    obs = self.obs_composer.wrap(
                        obs_dict.get(self.rl_agent_id, {}),
                        info_dict.get(self.rl_agent_id, {}),
                    )
                    facts = create_episode_facts(
                        episode=episode,
                        agent_ids=all_agent_ids,
                        trainable_ids=[self.rl_agent_id],
                        opponent_ids=opponent_ids,
                    )
                    env_steps = 0

                    while True:
                        active = set(getattr(self.env, "agents", list(obs_dict)))
                        if self.rl_agent_id not in active:
                            break
                        action_norm = np.asarray(
                            active_agent.act(obs, deterministic=True)[0], dtype=np.float32
                        )
                        action_phys = self.action_composer.process(action_norm)
                        actions = self._actions(action_phys, obs_dict, active)
                        if not actions:
                            break

                        episode_done = False
                        for _ in range(self.action_repeat):
                            obs_dict, _, terms, truncs, info_dict = self.env.step(actions)
                            env_steps += 1
                            update_agent_step_facts(
                                facts,
                                step_idx=env_steps,
                                infos=info_dict,
                                terminations=terms,
                                truncations=truncs,
                                agent_states=self._agent_states(all_agent_ids),
                            )
                            active_after = set(getattr(self.env, "agents", []))
                            if self.rl_agent_id not in active_after:
                                episode_done = True
                                break
                        if episode_done:
                            break

                        self.obs_composer.update_prev_action(action_norm)
                        obs = self.obs_composer.wrap(
                            obs_dict.get(self.rl_agent_id, {}),
                            info_dict.get(self.rl_agent_id, {}),
                        )

                    results.append(finalize_episode_facts(facts))
        finally:
            active_agent.actor.train(actor_was_training)
            # Fixed evaluation controllers are permitted to use process-global
            # randomness, but evaluation must not perturb training trajectories.
            np.random.set_state(numpy_rng_state)
            random.setstate(python_rng_state)
            torch.random.set_rng_state(torch_rng_state)
            if cuda_rng_states is not None:
                torch.cuda.set_rng_state_all(cuda_rng_states)

        return aggregate_eval_episodes(results, focal_agent_id=self.rl_agent_id)

    def bind_agent(self, agent: Any) -> "DeterministicPPOEvaluator":
        self.agent = agent
        return self

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if callable(close):
            close()
