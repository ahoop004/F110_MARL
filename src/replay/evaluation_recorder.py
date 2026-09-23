"""Shared-frame adapter for deterministic PPO and fixed-opponent MAPPO evaluation."""
from pathlib import Path

from .dataset_writer import RaceDatasetWriter
from .race_recorder import RaceRecorder, capture_state, plain, recording_config


def evaluation_recording_config(scenario, *, requested=False):
    override = scenario.get('evaluation', {}).get('recording', {})
    enabled = override.get('enabled', requested or scenario.get('recording', {}).get('enabled', False))
    if not enabled:
        return None
    return recording_config({**scenario.get('recording', {}), 'sample_probability': 1.,
                             'windows': [], **override, 'enabled': True})


class EvaluationRecording:
    def __init__(self, directory, *, config, env, trainable_ids, run_id, action_repeat, metadata):
        self.env, self.ids, self.run_id = env, list(trainable_ids), run_id
        self.action_repeat = action_repeat
        self.writer = RaceDatasetWriter(Path(directory), config=config,
            metadata=dict(metadata, run_id=run_id, phase='evaluation', trainable_agents=self.ids))
        self.recorder = RaceRecorder(config, self.writer.add_event, run_id=run_id, environment_id='evaluation')
        self.context = None
        self.failed = False
        self.pre = None

    def begin(self, context):
        self.context = dict(context)

    def start(self, episode, infos, *, protocol):
        from metrics.racing_eval import capture_spawn_context
        context = dict(self.context, phase='evaluation', protocol=protocol['name'], evaluation_protocol=protocol,
            run_id=self.run_id, environment_id='evaluation', environment_episode=episode,
            episode_id=f"{self.run_id}_{self.context['evaluation_id']}_ep{episode:06d}")
        self.recorder.start(episode_id=context['episode_id'], episode=episode,
            map_id=getattr(self.env, '_map_bundle_active', None) or self.env.map_name,
            seed=self.env.seed, timestep=self.env.timestep, action_repeat=self.action_repeat,
            trainable_ids=self.ids, agent_ids=list(self.env.possible_agents),
            track_length=self.env.centerline_track_length, policy_version=context.get('policy_version'),
            spawn=capture_spawn_context(self.env, self.env.possible_agents),
            physics=infos.get(self.ids[0], {}).get('physics'),
            termination=dict(mode=self.env.episode_termination_mode,
                             finish_on_laps=self.env.lifecycle.finish_on_laps), evaluation=context)
        return context

    def before_step(self, infos, obs, physics_index):
        if self.writer.storage_full and not self.recorder.stopped:
            self.recorder.stop()
        self.recorder.prepare(self.context.get('environment_steps'), physics_index,
            self.context.get('policy_version'), exhausted_windows=self.writer.exhausted_windows,
            clock='checkpoint_training_steps')
        self.env.record_applied_commands = self.recorder.capturing
        self.pre = capture_state(self.env, infos, obs, self.env.possible_agents) if self.recorder.capturing else None

    def step(self, *, infos, obs, physical, normalized, wrapped, physics_index, decision_index, substep,
             terminated, truncated, rewards=None, individual_rewards=None, components=None, team_components=None):
        if self.pre is None:
            return
        applied = getattr(self.env, '_recorded_applied_commands', None)
        availability = 'computed' if rewards is not None else 'not_computed_by_selection_evaluator'
        self.recorder.step(plain(dict(
            physics_index=physics_index, physics_index_end=physics_index+1,
            decision_index=decision_index, substep_index=substep, action_repeat=self.action_repeat,
            policy_version=self.context.get('policy_version'), timestep_s=self.env.timestep,
            simulation_time_s=physics_index*self.env.timestep,
            simulation_time_end_s=(physics_index+1)*self.env.timestep,
            pre_state=self.pre, post_state=capture_state(self.env, infos, obs, self.env.possible_agents),
            commands={aid: dict(requested=physical.get(aid), applied=applied[i] if applied is not None else None,
                source=('learner' if aid in normalized else 'fixed_policy' if aid in physical else 'terminal_controller'))
                for i, aid in enumerate(self.env.possible_agents)},
            learners={aid: dict(observation=wrapped[aid], action_normalized=action,
                reward=(rewards or {}).get(aid), individual_reward=(individual_rewards or {}).get(aid),
                reward_components=(components or {}).get(aid, {}), reward_availability=availability)
                for aid, action in normalized.items()},
            team_reward_components=team_components or {}, reward_availability=availability,
            terminated=terminated, truncated=truncated)))

    def end(self):
        complete = bool(self.env.episode_done)
        self.recorder.end(complete, reason=None if complete else 'evaluator_boundary')
        self.env.record_applied_commands = False

    def close(self, *, complete=True):
        self.writer.close(complete=complete and not self.failed)
