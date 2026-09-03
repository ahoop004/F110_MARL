"""Step-based off-policy training loop for SAC, TD3, DQN."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from env.types import GlobalState, TransitionRecord
from metrics.outcomes import determine_outcome
from src.replay.replay_buffer import ReplayBuffer
from training.hooks import TrainingHook
from training.reward_context import build_reward_context, transition_lifecycle_fields
from wrappers.actions.composer import ActionComposer
from wrappers.observations.composer import ObservationComposer
from wrappers.rewards.composer import RewardComposer


class OffPolicyTrainer:
    """Step-based training loop for off-policy RL agents.

    Flow per step:
      1. act (random if step < learning_starts, else agent.act)
      2. step env (with action_repeat)
      3. compute reward via RewardComposer
      4. add transition to ReplayBuffer
      5. every train_freq steps after learning_starts: sample + agent.update
      6. on episode end: call hooks.on_episode_end, reset env
    """

    def __init__(
        self,
        env: Any,
        rl_agent_id: str,
        agent: Any,
        other_agents: Dict[str, Any],
        obs_composer: ObservationComposer,
        reward_composer: RewardComposer,
        action_composer: ActionComposer,
        replay_buffer: ReplayBuffer,
        action_repeat: int = 1,
        hooks: Optional[List[TrainingHook]] = None,
        render: bool = False,
        run_id: str = "run",
        spawn_plan_fn: Optional[Callable] = None,
    ) -> None:
        self.env = env
        self.rl_agent_id = rl_agent_id
        self.agent = agent
        self.other_agents = other_agents
        self.obs_composer = obs_composer
        self.reward_composer = reward_composer
        self.action_composer = action_composer
        self.replay_buffer = replay_buffer
        self.action_repeat = max(1, int(action_repeat))
        self.hooks = hooks or []
        self.render = render
        self.run_id = run_id
        self.spawn_plan_fn = spawn_plan_fn

    def _episode_id(self, episode: int) -> str:
        return f"{self.run_id}_ep{episode:06d}"

    def _map_id(self) -> Optional[str]:
        return getattr(self.env, "_map_bundle_active", None) or getattr(
            self.env, "map_name", None
        )

    def _spawn_id(self) -> Optional[str]:
        sm = getattr(self.env, "_spawn_manager", None)
        if sm is None:
            return None
        meta = getattr(sm, "last_spawn_metadata", {}) or {}
        spawn_ids = meta.get("spawn_ids", {})
        return spawn_ids.get(self.rl_agent_id) or meta.get("spawn_id")

    def _build_actions(self, rl_action_phys: np.ndarray, obs_dict: Dict) -> Dict[str, np.ndarray]:
        active = set(getattr(self.env, "agents", obs_dict))
        actions: Dict[str, np.ndarray] = {}
        if self.rl_agent_id in active:
            actions[self.rl_agent_id] = rl_action_phys
        for aid, other_agent in self.other_agents.items():
            if aid in active:
                try:
                    act = other_agent.act(obs_dict[aid])
                except Exception:
                    act = np.zeros(2, dtype=np.float32)
                actions[aid] = np.asarray(act, dtype=np.float32)
        return actions

    def _reset_env(self) -> tuple:
        """Reset env, injecting a curriculum spawn plan when one is available."""
        spawn_plan = self.spawn_plan_fn() if self.spawn_plan_fn is not None else None
        options = {"spawn_plan": spawn_plan} if spawn_plan is not None else None
        return self.env.reset(options=options)

    def _reward_context(
        self,
        *,
        agent_id: str,
        info_dict: Dict,
        obs_dict: Dict,
        actions: Dict[str, np.ndarray],
        global_state: Optional[GlobalState] = None,
    ) -> Dict[str, Any]:
        return build_reward_context(
            env=self.env,
            agent_id=agent_id,
            info_dict=info_dict,
            obs_dict=obs_dict,
            actions=actions,
            global_state=global_state,
        )

    def train(
        self,
        total_steps: int,
        learning_starts: int,
        train_freq: int,
        gradient_steps: int,
        batch_size: int,
    ) -> None:
        obs_dict, info_dict = self._reset_env()
        self.obs_composer.reset()
        self.reward_composer.reset()
        obs = self.obs_composer.wrap(obs_dict.get(self.rl_agent_id, {}), info_dict.get(self.rl_agent_id, {}))

        episode = 0
        episode_reward = 0.0
        episode_truncated = False
        last_info: Dict = {}
        update_metrics: Dict = {}
        step_idx = 0  # within current episode
        episode_id = self._episode_id(episode)
        map_id = self._map_id()
        spawn_id = self._spawn_id()

        for step in range(total_steps):
            # Choose action
            if step < learning_starts:
                action = self.agent.explore()
            else:
                action = self.agent.act(obs)

            action_phys = self.action_composer.process(action)
            actions = self._build_actions(action_phys, obs_dict)
            acted_agents = set(actions)

            # Step env (with action_repeat).
            # Reward is accumulated across sub-steps so that progress-based
            # components (centerline delta, etc.) are not lost between repeats.
            done = False
            rl_term = False
            rl_trunc = False
            reward = 0.0
            post_step_global_snapshot: Optional[GlobalState] = None
            for _ in range(self.action_repeat):
                obs_dict, _, term_dict, trunc_dict, info_dict = self.env.step(actions)
                step_facts = getattr(self.env, "last_step_facts", None)
                post_step_global_snapshot = getattr(
                    step_facts, "global_state", None
                )
                if post_step_global_snapshot is None:
                    post_step_global_snapshot = self.env.get_global_state()
                if self.render:
                    try:
                        self.env.render()
                    except Exception:
                        pass
                rl_term = bool(term_dict.get(self.rl_agent_id, False))
                rl_trunc = bool(trunc_dict.get(self.rl_agent_id, False))
                sub_done = rl_term or rl_trunc
                sub_info = info_dict.get(self.rl_agent_id, {})
                sub_step_info = {
                    "obs": obs,
                    "next_obs": obs_dict.get(self.rl_agent_id, {}),
                    "info": sub_info,
                    "done": sub_done,
                    "terminated": rl_term,
                    "truncated": rl_trunc,
                    "action": action,
                    "timestep": 0.01,
                }
                sub_step_info.update(
                    self._reward_context(
                        agent_id=self.rl_agent_id,
                        info_dict=info_dict,
                        obs_dict=obs_dict,
                        actions=actions,
                        global_state=post_step_global_snapshot,
                    )
                )
                sub_reward, _ = self.reward_composer.compute(sub_step_info)
                reward += sub_reward
                membership_changed = not acted_agents.issubset(
                    set(getattr(self.env, "agents", []))
                )
                if sub_done or membership_changed:
                    done = True
                    episode_truncated = bool(rl_trunc)
                    if sub_done:
                        break
                    done = False
                    break

            last_info = info_dict.get(self.rl_agent_id, {})
            episode_reward += reward

            next_obs = self.obs_composer.wrap(obs_dict.get(self.rl_agent_id, {}), last_info)

            # --- Emit transition record ---
            global_state = (
                post_step_global_snapshot.vector
                if post_step_global_snapshot is not None
                else np.zeros(0, dtype=np.float32)
            )
            record = TransitionRecord(
                obs=obs,
                action_norm=action,
                action_phys=action_phys,
                reward=reward,
                reward_components={},
                next_obs=next_obs,
                terminated=rl_term if done else False,
                truncated=rl_trunc if done else False,
                info=last_info,
                global_state=global_state,
                map_id=map_id,
                spawn_id=spawn_id,
                episode_id=episode_id,
                step_idx=step_idx,
                agent_id=self.rl_agent_id,
                **transition_lifecycle_fields(
                    self.env,
                    last_info,
                    global_state=post_step_global_snapshot,
                ),
            )
            for hook in self.hooks:
                hook.on_step(record)
            step_idx += 1

            self.replay_buffer.add(obs, action, reward, next_obs, float(done))

            # Update
            if step >= learning_starts and (step - learning_starts) % train_freq == 0:
                for _ in range(gradient_steps):
                    if len(self.replay_buffer) >= batch_size:
                        batch = self.replay_buffer.sample(batch_size)
                        update_metrics = self.agent.update(batch)
                        for hook in self.hooks:
                            hook.on_update(update_metrics)

            if done:
                outcome = determine_outcome(last_info, truncated=episode_truncated)
                last_info["outcome"] = outcome.value
                update_metrics = dict(update_metrics)
                update_metrics["episode_steps"] = step_idx
                for hook in self.hooks:
                    hook.on_episode_end(episode, episode_reward, last_info, update_metrics)

                obs_dict, info_dict = self._reset_env()
                self.obs_composer.reset()
                self.reward_composer.reset()
                obs = self.obs_composer.wrap(obs_dict.get(self.rl_agent_id, {}), info_dict.get(self.rl_agent_id, {}))
                episode_reward = 0.0
                episode_truncated = False
                episode += 1
                step_idx = 0
                episode_id = self._episode_id(episode)
                map_id = self._map_id()
                spawn_id = self._spawn_id()
            else:
                obs = next_obs

        for hook in self.hooks:
            hook.on_training_end()
