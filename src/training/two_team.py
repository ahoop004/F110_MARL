"""Simultaneous two-team MAPPO with independent policies and joint team credit.

The race deadline is part of the task. Its remaining fraction is appended to
both actor and critic inputs; actors also receive their remaining lap fraction.
Individual deaths never terminate either team's return before the race ends.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from contextlib import contextmanager
import random

import numpy as np


@contextmanager
def preserve_rng():
    """Keep evaluation and its environment construction out of training RNGs."""
    import torch

    numpy_state, python_state = np.random.get_state(), random.getstate()
    torch_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)
        torch.random.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def evaluate_pair(trainer, protocol):
    """Fixed seeds/maps, deterministic actors, no updates or rollout mutation."""
    modes = {team: agent.actor.training for team, agent in trainer.agents.items()}
    raw = {team: deepcopy(agent.last_raw_actions) for team, agent in trainer.agents.items()}
    rows = []
    with preserve_rng():
        try:
            for agent in trainer.agents.values():
                agent.actor.eval()
            for index in range(protocol["episodes"]):
                row = trainer.episode(training=False, seed=protocol["seed"] + index,
                                      map_episode_index=index)
                row["map"] = getattr(trainer.env, "_map_bundle_active", None)
                row["seed"] = protocol["seed"] + index
                rows.append(row)
        finally:
            for team, agent in trainer.agents.items():
                agent.actor.train(modes[team])
                agent.last_raw_actions = raw[team]
    keys = sorted({key for row in rows for key in row} - {"map", "seed"})
    summary = {key: float(np.mean([row.get(key, 0.0) for row in rows])) for key in keys}
    return summary, rows


def selection_score(summary, teams):
    """Favor completion of both teams; reward is never the selection metric."""
    return (min(summary[f"{t}/both_finished"] for t in teams),
            sum(summary[f"{t}/finish_rate"] for t in teams),
            sum(summary[f"{t}/progress_laps"] for t in teams),
            -sum(summary[f"{t}/crash_count"] for t in teams))


def update_inactive_values(agent, states, returns):
    """Train value targets after both teammates stop, without actor samples."""
    import torch

    if not len(states):
        return {}
    states = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=agent.device)
    returns = torch.as_tensor(returns, dtype=torch.float32, device=agent.device)
    losses = []
    for _ in range(agent.n_epochs):
        for batch in torch.randperm(len(states), device=agent.device).split(agent.batch_size):
            loss = torch.nn.functional.mse_loss(agent.critic(states[batch]), returns[batch])
            agent.optimizer.zero_grad(set_to_none=True)
            (agent.vf_coef * loss).backward()
            torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), agent.max_grad_norm)
            agent.optimizer.step()
            losses.append(loss.detach())
    return {"inactive_value_loss": float(torch.stack(losses).mean().cpu())}


def resolve_teams(scenario):
    from core.agent_builder import get_trainable_agent_ids

    ids = get_trainable_agent_ids(scenario["agents"])
    mapping = scenario["environment"].get("agent_teams", {})
    teams = {}
    for aid in ids:
        if aid not in mapping:
            raise ValueError(f"Two-team training requires agent_teams[{aid}]")
        teams.setdefault(str(mapping[aid]), []).append(aid)
    if len(ids) != 4 or len(teams) != 2 or any(len(v) != 2 for v in teams.values()):
        raise ValueError("Two-team training requires exactly two teams of two trainable cars")
    if set(ids) != set(scenario["agents"]):
        raise ValueError("Two-team training does not accept additional fixed agents")
    return teams


class TeamEvents:
    """Award each opponent collision and both-finished event exactly once."""

    def __init__(self, teams, *, opponent_crash_bonus=1.0, both_finished_bonus=2.0):
        self.teams = teams
        self.opponent_crash_bonus = float(opponent_crash_bonus)
        self.both_finished_bonus = float(both_finished_bonus)
        self.reset()

    def reset(self):
        self.crashed = set()
        self.finished = set()
        self.both_finished = set()

    def step(self, infos):
        crashed = {aid for aid, info in infos.items()
                   if info.get("terminal_reason") == "collision"}
        newly_crashed = crashed - self.crashed
        self.crashed.update(crashed)
        self.finished.update(aid for aid, info in infos.items()
                             if info.get("status") == "finished")
        result = {}
        for team, ids in self.teams.items():
            opponents = {aid for other, members in self.teams.items()
                         if other != team for aid in members}
            both = set(ids) <= self.finished and team not in self.both_finished
            if both:
                self.both_finished.add(team)
            result[team] = {
                "opponent_crash": self.opponent_crash_bonus * len(newly_crashed & opponents),
                "both_finished": self.both_finished_bonus * int(both),
            }
        return result


def race_metrics(teams, infos, progress):
    """Completion-first result, independent of the shaped learning reward.

    Compare finisher count, then earned signed progress (in laps). Equal results
    are draws. Opponent crashes remain a separate measured objective.
    """
    result, scores = {}, {}
    for team, ids in teams.items():
        finished = sum(infos[aid].get("status") == "finished" for aid in ids)
        crashes = sum(infos[aid].get("terminal_reason") == "collision" for aid in ids)
        net_progress = sum(progress[aid] for aid in ids)
        scores[team] = (finished, round(net_progress, 6))
        result.update({
            f"{team}/finish_count": finished,
            f"{team}/finish_rate": finished / len(ids),
            f"{team}/both_finished": float(finished == len(ids)),
            f"{team}/crash_count": crashes,
            f"{team}/any_crash": float(crashes > 0),
            f"{team}/eliminated": float(crashes == len(ids)),
            f"{team}/timeout_count": sum(infos[a].get("terminal_reason") == "time_limit" for a in ids),
            f"{team}/laps": sum(infos[a].get("lap_count", 0) for a in ids),
            f"{team}/progress_laps": net_progress,
        })
    first, second = teams
    for team, other in ((first, second), (second, first)):
        result[f"{team}/win"] = float(scores[team] > scores[other])
        result[f"{team}/draw"] = float(scores[team] == scores[other])
        result[f"{team}/opponent_crash_count"] = result[f"{other}/crash_count"]
    for aid, info in infos.items():
        result.update({f"{aid}/finished": float(info.get("status") == "finished"),
                       f"{aid}/crashed": float(info.get("terminal_reason") == "collision"),
                       f"{aid}/laps": info.get("lap_count", 0),
                       f"{aid}/progress_laps": progress[aid]})
    return result


class TwoTeamTrainer:
    def __init__(self, *, env, teams, agents, observations, rewards, actions,
                 event_config=None, render=False, on_update=None, run_id='run', environment_id=0,
                 race_recorder=None, race_writer=None):
        self.env, self.teams, self.agents = env, teams, agents
        self.observations, self.rewards, self.actions = observations, rewards, actions
        self.events = TeamEvents(teams, **(event_config or {}))
        self.render = render
        self.on_update = on_update
        self.environment_steps = 0
        self.agent_steps = 0
        self.updates = 0
        self.run_id, self.environment_id = run_id, environment_id
        self.race_recorder, self.race_writer = race_recorder, race_writer
        self.recording_stop = False
        self.training_episode = 0
        self._global_rollout = []
        if any(composer.team_contract for composer in rewards.values()):
            raise ValueError("Two-team rewards use local composers plus explicit team events")

    def global_state(self, steps):
        return np.concatenate((self.env.get_global_state().vector,
                               [max(0.0, 1.0 - steps / self.env.max_steps)])).astype(np.float32)

    def wrap(self, aid, obs, info, steps):
        return np.concatenate((self.observations[aid].wrap(obs, info), [
            max(0.0, 1.0 - steps / self.env.max_steps),
            max(0.0, 1.0 - info.get("lap_count", 0) / self.env.target_laps),
        ])).astype(np.float32)

    def update(self, state):
        metrics = {}
        for team, agent in self.agents.items():
            # With both teammates inactive, no actor samples exist, but their
            # value function still needs targets for later opponent events.
            # Otherwise a fragment ending after elimination bootstraps from
            # states on which the critic is never trained.
            actor_steps = {index for indices in agent._team_step_indices.values() for index in indices}
            inactive_steps = [i for i in range(len(self._global_rollout)) if i not in actor_steps]
            tail_returns = None
            if inactive_steps:
                tail_returns = agent.compute_team_gae(agent.evaluate_state(state, agent.agent_ids[0]))[1][inactive_steps]
            metrics.update({f"{team}/{key.removeprefix('train/')}": value
                            for key, value in agent.update(state).items()})
            if inactive_steps:
                metrics.update({f"{team}/{key}": value for key, value in update_inactive_values(
                    agent, [self._global_rollout[i] for i in inactive_steps], tail_returns).items()})
            agent.clear_buffers()
        self._global_rollout.clear()
        self.updates += 1
        if self.on_update:
            self.on_update({"train/update": self.updates,
                            "train/environment_steps": self.environment_steps,
                            "train/agent_steps": self.agent_steps,
                            **{f"train/{key}": value for key, value in metrics.items()}})

    def episode(self, *, training=True, seed=None, step_budget=None, map_episode_index=0):
        generator = self.iter_episode(training=training, seed=seed, step_budget=step_budget,
                                      map_episode_index=map_episode_index)
        try:
            next(generator)
        except StopIteration as result:
            return result.value
        raise RuntimeError("Serial two-team episode unexpectedly requested remote inference")

    def finish_fragment(self, next_values):
        """Flush both CPU collectors against their own bootstrap values."""
        if self._global_rollout:
            for team, agent in self.agents.items():
                agent.finish_team_fragment(next_values[team], self._global_rollout)
            self._global_rollout.clear()

    def iter_episode(self, *, training=True, seed=None, step_budget=None,
                     map_episode_index=0, parallel=False):
        options = {"map_episode_index": map_episode_index} if seed is not None else None
        obs, infos = self.env.reset(seed=seed, options=options)
        recorder = self.race_recorder if training else None
        episode = self.training_episode
        episode_id = f'{self.run_id}_env{self.environment_id:04d}_ep{episode:06d}'
        agent_teams = {aid: team for team, ids in self.teams.items() for aid in ids}
        start_policy = self.updates
        policy_version = start_policy
        if training:
            self.training_episode += 1
        if recorder is not None:
            from src.replay.race_recorder import capture_state, plain
            from metrics.racing_eval import capture_spawn_context
            recorder.start(episode_id=episode_id, episode=episode,
                map_id=getattr(self.env, '_map_bundle_active', None) or self.env.map_name,
                seed=self.env.seed, timestep=self.env.timestep, action_repeat=1,
                trainable_ids=list(agent_teams), agent_ids=list(self.env.possible_agents),
                track_length=self.env.centerline_track_length, policy_version=start_policy,
                spawn=capture_spawn_context(self.env, self.env.possible_agents),
                physics=infos[next(iter(agent_teams))].get('physics'),
                termination=dict(mode=self.env.episode_termination_mode, finish_on_laps=True),
                agent_teams=agent_teams, team_policy_versions={t: start_policy for t in self.teams})
        for composer in (*self.observations.values(), *self.rewards.values(), *self.actions.values()):
            composer.reset()
        self.events.reset()
        steps = 0
        progress = defaultdict(float)
        totals = defaultdict(float)
        component_totals = {team: defaultdict(float) for team in self.teams}
        final_infos = deepcopy(infos)
        state = self.global_state(steps)
        while self.env.agents and not self.env.episode_done:
            active = set(self.env.agents)
            wrapped = {aid: self.wrap(aid, obs[aid], infos[aid], steps) for aid in active}
            normalized, log_probs, values, team_values = {}, {}, {}, {}
            if parallel:
                normalized, log_probs, values, raw_actions, team_values = yield ("act", (wrapped, state))
                for agent in self.agents.values():
                    agent.last_raw_actions = {aid: raw_actions[aid] for aid in agent.agent_ids if aid in raw_actions}
            else:
                for team, agent in self.agents.items():
                    ids = [aid for aid in self.teams[team] if aid in active]
                    if ids:
                        acts, logs = agent.act_batch(ids, np.stack([wrapped[aid] for aid in ids]),
                                                     deterministic=not training)
                        normalized.update(acts)
                        log_probs.update(logs)
                    # Team values remain defined after both cars become inactive.
                    team_values[team] = agent.evaluate_state(state, self.teams[team][0])
                    values.update({aid: team_values[team] for aid in ids})
            physical = {aid: self.actions[aid].process(action) for aid, action in normalized.items()}
            policy_version = self.updates
            if recorder is not None and (self.recording_stop or
                    (self.race_writer is not None and self.race_writer.storage_full)) and not recorder.stopped:
                recorder.stop()
            recording_active = recorder is not None and recorder.capturing
            self.env.record_applied_commands = recording_active
            if recording_active:
                recorded_pre = capture_state(self.env, infos, obs, self.env.possible_agents)
            next_obs, _, terminated, truncated, infos = self.env.step(physical)
            steps += 1
            if training:
                self.environment_steps += 1
                self.agent_steps += len(active)
            final_infos.update(deepcopy(infos))
            breakdowns = self.events.step(infos)
            local_rewards = {}
            local_components = {}
            for aid in active:
                info = infos[aid]
                progress[aid] += float(info.get("centerline", {}).get("progress_delta", 0.0))
                local_rewards[aid], components = self.rewards[aid].compute({
                    "info": info, "obs": wrapped[aid], "next_obs": next_obs.get(aid, {}),
                    "action": normalized[aid], "timestep": self.env.timestep,
                    "done": terminated[aid] or truncated[aid],
                    "terminated": terminated[aid], "truncated": truncated[aid],
                    "track_length": self.env.centerline_track_length,
                })
                team = next(team for team, members in self.teams.items() if aid in members)
                if recording_active:
                    local_components[aid] = dict(components)
                for key, value in components.items():
                    breakdowns[team][key] = breakdowns[team].get(key, 0.0) + value
                self.observations[aid].update_prev_action(normalized[aid])
            done = bool(self.env.episode_done or not self.env.agents)
            next_state = self.global_state(steps)
            if training:
                self._global_rollout.append(state)
            for team, agent in self.agents.items():
                ids = [aid for aid in self.teams[team] if aid in active]
                reward = sum(breakdowns[team].values())
                totals[team] += reward
                for key, value in breakdowns[team].items():
                    component_totals[team][key] += value
                if training:
                    agent.store_batch(ids, observations=wrapped, global_state=state,
                                      actions=normalized, rewards={aid: reward for aid in ids},
                                      log_probs=log_probs, values=values, terminated=terminated,
                                      truncated=truncated, raw_actions=agent.last_raw_actions)
                    agent.store_team_step(ids, reward=reward, value=team_values[team], terminal=done)
            if recording_active:
                applied = getattr(self.env, '_recorded_applied_commands', None)
                recorder.step(plain(dict(
                    physics_index=steps-1, physics_index_end=steps, decision_index=steps-1, substep_index=0,
                    policy_version=policy_version, team_policy_versions={t: policy_version for t in self.teams},
                    timestep_s=self.env.timestep, action_repeat=1,
                    simulation_time_s=(steps-1)*self.env.timestep, simulation_time_end_s=steps*self.env.timestep,
                    pre_state=recorded_pre, post_state=capture_state(self.env, infos, next_obs, self.env.possible_agents),
                    agent_teams=agent_teams,
                    commands={aid: dict(requested=physical.get(aid),
                        applied=applied[i] if applied is not None else None,
                        source='learner' if aid in physical else 'terminal_controller', team=agent_teams[aid])
                        for i, aid in enumerate(self.env.possible_agents)},
                    learners={aid: dict(observation=wrapped[aid], action_normalized=normalized[aid],
                        reward=sum(breakdowns[agent_teams[aid]].values()), individual_reward=local_rewards[aid],
                        reward_components=local_components[aid], team=agent_teams[aid]) for aid in active},
                    team_rewards={team: sum(parts.values()) for team, parts in breakdowns.items()},
                    team_reward_components={f'{team}/{key}': value for team, parts in breakdowns.items()
                                            for key, value in parts.items()},
                    terminated=terminated, truncated=truncated)))
            if self.render:
                self.env.render()
            budget_cut = step_budget is not None and steps >= step_budget and not done
            if training and (done or budget_cut or any(a.any_buffer_full() for a in self.agents.values())):
                if parallel:
                    next_values = yield ("bootstrap", next_state)
                    self.finish_fragment(next_values)
                else:
                    self.update(next_state)
            state, obs = next_state, next_obs
            if parallel:
                yield ("step", next_state)
            if done or budget_cut:
                break
        result = race_metrics(self.teams, final_infos, progress)
        if recorder is not None:
            recorder.end(bool(self.env.episode_done))
        if training:
            result.update(run_id=self.run_id, episode_id=episode_id, environment_id=self.environment_id,
                environment_episode=episode, map_id=getattr(self.env, '_map_bundle_active', None),
                policy_version_start=start_policy, policy_version_end=policy_version)
        result.update({"steps": steps, "duration_s": steps * self.env.timestep,
                       "completed": float(self.env.episode_done),
                       "budget_cut": float(not self.env.episode_done)})
        for team in self.teams:
            result[f"{team}/reward"] = totals[team]
            result.update({f"{team}/reward_components/{key}": value
                           for key, value in component_totals[team].items()})
        return result
