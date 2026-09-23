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
    update_agent_step_facts, episode_race_record,
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
        self.recording = None

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
        results, episode_records, by_map = [], [], {}
        protocol = dict(name='selection', seeds=list(range(self.base_seed, self.base_seed+self.episodes)),
            max_steps=getattr(self.env, 'max_steps', None), timestep_s=getattr(self.env, 'timestep', None),
            target_laps=getattr(self.env, 'target_laps', None), action_repeat=self.action_repeat)
        physics_episodes = []
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
                    obs_dict, info_dict = self.env.reset(
                        seed=self.base_seed + episode,
                        options={"map_episode_index": episode},
                    )
                    record_context = self.recording.start(episode, info_dict, protocol=protocol) if self.recording else {}
                    reset_actions = getattr(self.action_composer, "reset", None)
                    if reset_actions is not None:
                        reset_actions()
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
                    decision = 0

                    while True:
                        active = set(getattr(self.env, "agents", list(obs_dict)))
                        if self.rl_agent_id not in active:
                            break
                        action_norm = active_agent.predict(obs)
                        action_phys = self.action_composer.process(action_norm)
                        actions = self._actions(action_phys, obs_dict, active)
                        if not actions:
                            break

                        episode_done = False
                        for substep in range(self.action_repeat):
                            if self.recording:
                                self.recording.before_step(info_dict, obs_dict, env_steps)
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
                            if self.recording:
                                self.recording.step(infos=info_dict, obs=obs_dict, physical=actions,
                                    normalized={self.rl_agent_id: action_norm}, wrapped={self.rl_agent_id: obs},
                                    physics_index=env_steps-1, decision_index=decision, substep=substep,
                                    terminated=terms, truncated=truncs)
                            active_after = set(getattr(self.env, "agents", []))
                            if self.rl_agent_id not in active_after:
                                episode_done = True
                                break
                            if not set(actions).issubset(active_after):
                                break
                        decision += 1
                        if episode_done:
                            break

                        self.obs_composer.update_prev_action(action_norm)
                        obs = self.obs_composer.wrap(
                            obs_dict.get(self.rl_agent_id, {}),
                            info_dict.get(self.rl_agent_id, {}),
                        )

                    if self.recording:
                        self.recording.end()
                    result = finalize_episode_facts(facts)
                    results.append(result)
                    map_id = getattr(self.env, '_map_bundle_active', None) or getattr(self.env, 'map_name', None)
                    by_map.setdefault(map_id or 'unknown', []).append(result)
                    episode_records.append({**episode_race_record(result,
                        timestep=getattr(self.env, 'timestep', None), include_rewards=False),
                        'phase': 'evaluation', 'map_id': map_id, 'environment_episode': episode,
                        'seed': self.base_seed+episode, **record_context})
                    physics = info_dict.get(self.rl_agent_id, {}).get("physics")
                    if physics is not None:
                        physics_episodes.append({"seed": self.base_seed + episode, "physics": physics})
        except BaseException:
            if self.recording:
                self.recording.failed = True
            raise
        finally:
            active_agent.actor.train(actor_was_training)
            # Fixed evaluation controllers are permitted to use process-global
            # randomness, but evaluation must not perturb training trajectories.
            np.random.set_state(numpy_rng_state)
            random.setstate(python_rng_state)
            torch.random.set_rng_state(torch_rng_state)
            if cuda_rng_states is not None:
                torch.cuda.set_rng_state_all(cuda_rng_states)

        summary = aggregate_eval_episodes(
            results, focal_agent_id=self.rl_agent_id,
            timestep=getattr(self.env, "timestep", None),
        )
        summary['episode_results'] = episode_records
        summary['per_map'] = {name: aggregate_eval_episodes(rows, focal_agent_id=self.rl_agent_id,
            timestep=getattr(self.env, 'timestep', None)) for name, rows in by_map.items()}
        summary["evaluation_protocol"] = protocol
        if physics_episodes:
            summary["evaluation_protocol"]["physics_episodes"] = physics_episodes
        return summary

    def bind_agent(self, agent: Any) -> "DeterministicPPOEvaluator":
        self.agent = agent
        return self

    def close(self) -> None:
        if self.recording:
            self.recording.close()
        close = getattr(self.env, "close", None)
        if callable(close):
            close()
