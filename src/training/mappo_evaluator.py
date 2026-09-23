"""Isolated deterministic MAPPO evaluation for checkpoint selection."""
from copy import deepcopy
import random

import numpy as np
import torch

from metrics.racing_eval import (
    aggregate_eval_episodes, create_episode_facts, finalize_episode_facts,
    update_agent_step_facts,
    episode_race_record, capture_spawn_context,
)


class DeterministicMAPPOEvaluator:
    def __init__(self, *, env, trainable_ids, other_agents, obs_composers,
                 action_composer, episodes, base_seed, action_repeat=1):
        self.env = env
        self.trainable_ids = list(trainable_ids)
        self.other_agents = dict(other_agents)
        self.obs_composers = obs_composers
        self.actions = {aid: deepcopy(action_composer) for aid in trainable_ids}
        self.episodes = int(episodes)
        self.base_seed = int(base_seed)
        self.action_repeat = int(action_repeat)
        self.recording = None

    def bind_agent(self, agent):
        self.agent = agent
        return self

    def evaluate(self):
        numpy_state, python_state = np.random.get_state(), random.getstate()
        torch_state = torch.random.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        was_training = self.agent.actor.training
        raw_actions = deepcopy(self.agent.last_raw_actions)
        self.agent.actor.eval()
        results, by_map, physics_episodes, episode_records = [], {}, [], []
        protocol = dict(name='selection', seeds=list(range(self.base_seed, self.base_seed+self.episodes)),
            max_steps=self.env.max_steps, timestep_s=self.env.timestep,
            target_laps=getattr(self.env, 'target_laps', None), action_repeat=self.action_repeat)
        try:
            with torch.no_grad():
                for episode in range(self.episodes):
                    obs, infos = self.env.reset(seed=self.base_seed + episode,
                                               options={"map_episode_index": episode})
                    spawn_context = capture_spawn_context(self.env, self.env.possible_agents)
                    record_context = self.recording.start(episode, infos, protocol=protocol) if self.recording else {}
                    for item in [*self.obs_composers.values(), *self.actions.values(),
                                 *self.other_agents.values()]:
                        if hasattr(item, "reset"):
                            item.reset()
                    facts = create_episode_facts(
                        episode=episode, agent_ids=self.env.possible_agents,
                        trainable_ids=self.trainable_ids,
                        opponent_ids=list(self.other_agents),
                    )
                    steps = 0
                    decision = 0
                    while self.env.agents:
                        ids = [aid for aid in self.trainable_ids if aid in self.env.agents]
                        normalized, physical, wrapped_rows = {}, {}, {}
                        if ids:
                            wrapped = np.stack([self.obs_composers[aid].wrap(
                                obs.get(aid, {}), infos.get(aid, {})) for aid in ids])
                            wrapped_rows = dict(zip(ids, wrapped))
                            normalized, _ = self.agent.act_batch(ids, wrapped, deterministic=True)
                            physical = {aid: self.actions[aid].process(normalized[aid]) for aid in ids}
                        for aid, controller in self.other_agents.items():
                            if aid in self.env.agents:
                                physical[aid] = controller.act(obs.get(aid, {}))
                        for substep in range(self.action_repeat):
                            if self.recording:
                                self.recording.before_step(infos, obs, steps)
                            obs, _, terms, truncs, infos = self.env.step(physical)
                            steps += 1
                            update_agent_step_facts(facts, step_idx=steps, infos=infos,
                                                    terminations=terms, truncations=truncs,
                                                    agent_states={aid: self.env.get_agent_state(aid)
                                                                  for aid in self.env.possible_agents})
                            if self.recording:
                                self.recording.step(infos=infos, obs=obs, physical=physical, normalized=normalized,
                                    wrapped=wrapped_rows, physics_index=steps-1, decision_index=decision, substep=substep,
                                    terminated=terms, truncated=truncs)
                            if not set(physical).issubset(self.env.agents):
                                break
                        decision += 1
                        for aid in ids:
                            self.obs_composers[aid].update_prev_action(normalized[aid])
                    if self.recording:
                        self.recording.end()
                    result = finalize_episode_facts(facts)
                    results.append(result)
                    map_name = getattr(self.env, "_map_bundle_active", None) or self.env.map_name
                    by_map.setdefault(str(map_name), []).append(result)
                    episode_records.append({
                        **episode_race_record(result, timestep=self.env.timestep, include_rewards=False),
                        "phase": "evaluation", "environment_episode": episode,
                        "seed": self.base_seed + episode, "map_id": map_name,
                        "spawn_configuration": spawn_context, **record_context,
                    })
                    physics = infos.get(self.trainable_ids[0], {}).get("physics")
                    if physics is not None:
                        physics_episodes.append({"seed": self.base_seed + episode,
                                                 "map_bundle": map_name, "physics": physics})
        except BaseException:
            if self.recording:
                self.recording.failed = True
            raise
        finally:
            self.agent.actor.train(was_training)
            self.agent.last_raw_actions = raw_actions
            np.random.set_state(numpy_state)
            random.setstate(python_state)
            torch.random.set_rng_state(torch_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)
        summary = aggregate_eval_episodes(results, timestep=self.env.timestep)
        summary["per_map"] = {name: aggregate_eval_episodes(rows, timestep=self.env.timestep)
                              for name, rows in by_map.items()}
        summary["episode_results"] = episode_records
        summary["evaluation_protocol"] = {
            "name": "selection", "seeds": list(range(self.base_seed, self.base_seed + self.episodes)),
            "max_steps": self.env.max_steps, "timestep_s": self.env.timestep,
            "target_laps": getattr(self.env, "target_laps", None),
            "action_repeat": self.action_repeat,
        }
        if physics_episodes:
            summary["physics_episodes"] = physics_episodes
        # This evaluator measures race facts; reward is not computed here.
        for row in [summary, *summary["per_map"].values()]:
            for key in ("mean_episode_reward", "per_agent_rewards_mean",
                        "per_agent_individual_rewards_mean", "per_agent_reward_components_mean"):
                row.pop(key, None)
        return summary

    def close(self):
        if self.recording:
            self.recording.close()
        self.env.close()
