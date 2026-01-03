"""Curriculum learning callback for Stable Baselines3."""

from collections import deque
from typing import Any, Dict, Optional, List
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from src.metrics.outcomes import EpisodeOutcome

class CurriculumCallback(BaseCallback):
    """Callback to handle curriculum learning progression for SB3.

    Supports both spawn curriculum and phased curriculum from scenario configs.

    Args:
        curriculum_config: Curriculum configuration dict from scenario
        spawn_curriculum: SpawnCurriculumManager instance (optional)
        ftg_agents: Dict of FTG agents to update parameters (optional)
        ftg_schedules: Dict of FTG schedules by agent_id (optional)
        env_wrapper: SB3SingleAgentWrapper instance to access environment
        wandb_run: WandB run instance for logging (optional)
        verbose: Verbosity level
    """

    def __init__(
        self,
        curriculum_config: Optional[Dict[str, Any]] = None,
        spawn_curriculum: Optional[Any] = None,
        ftg_agents: Optional[Dict[str, Any]] = None,
        ftg_schedules: Optional[Dict[str, Dict[str, Any]]] = None,
        env_wrapper: Optional[Any] = None,
        wandb_run: Optional[Any] = None,
        rich_console: Optional[Any] = None,
        algo_name: Optional[str] = None,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.curriculum_config = curriculum_config
        self.spawn_curriculum = spawn_curriculum
        self.ftg_agents = ftg_agents or {}
        self.ftg_schedules = ftg_schedules or {}
        self.env_wrapper = env_wrapper
        self.wandb_run = wandb_run
        self.rich_console = rich_console
        self.algo_name = algo_name

        # Phased curriculum state
        self.current_phase = 0
        self.phase_episodes = 0
        self.phase_successes = 0
        self.phase_total_reward = 0.0
        self.patience_counter = 0

        # Episode tracking
        self.episode_count = 0
        self.episode_rewards: List[float] = []
        self.episode_successes: List[bool] = []
        self.episode_lengths: List[int] = []
        self.window_size = 100
        self.episode_outcomes: deque = deque(maxlen=self.window_size)

        # Parse curriculum config
        self._parse_curriculum()

    def _parse_curriculum(self):
        """Parse curriculum configuration."""
        if not self.curriculum_config:
            self.phases = []
            self.curriculum_type = None
            return

        self.curriculum_type = self.curriculum_config.get('type', 'phased')
        self.phases = self.curriculum_config.get('phases', [])
        start_phase = self.curriculum_config.get('start_phase', 0)
        self.current_phase = min(start_phase, len(self.phases) - 1) if self.phases else 0

        if self.verbose > 0 and self.phases:
            print(f"\nCurriculum initialized:")
            print(f"  Type: {self.curriculum_type}")
            print(f"  Phases: {len(self.phases)}")
            print(f"  Starting phase: {self.current_phase} - {self.phases[self.current_phase]['name']}")

    def _init_callback(self) -> None:
        """Initialize callback at start of training."""
        if self.phases and self.current_phase < len(self.phases):
            self._apply_phase(self.current_phase)

    def _on_training_start(self) -> None:
        """Start Rich console if enabled."""
        if self.rich_console:
            self.rich_console.start()

    def _on_training_end(self) -> None:
        """Stop Rich console if enabled."""
        if self.rich_console:
            self.rich_console.stop()

    def _on_step(self) -> bool:
        """Called after each environment step."""
        # Check for new episode info in the logger
        if len(self.model.ep_info_buffer) > 0:
            # Get the most recent episode
            for ep_info in self.model.ep_info_buffer:
                # Only process if we haven't seen this episode yet
                if 'r' in ep_info and 'l' in ep_info:
                    # Check if this is a new episode (basic deduplication)
                    reward = ep_info['r']
                    # Process this episode
                    self._process_episode(
                        reward=reward,
                        length=ep_info['l'],
                        success=ep_info.get('is_success', False),
                        target_finished=ep_info.get('target_finished', False),
                        target_collision=ep_info.get('target_collision', False),
                        outcome=ep_info.get('outcome')
                    )
            # Clear the buffer to avoid reprocessing
            self.model.ep_info_buffer.clear()
        return True

    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout (for on-policy algorithms)."""
        # Episodes are tracked in _on_step via ep_info_buffer
        pass

    def _process_episode(
        self,
        reward: float,
        length: int,
        success: bool,
        target_finished: bool = False,
        target_collision: bool = False,
        outcome: Optional[str] = None,
    ):
        """Process completed episode for curriculum progression."""
        self.episode_count += 1
        self.episode_rewards.append(float(reward))
        self.episode_successes.append(bool(success))
        self.episode_lengths.append(int(length))

        # Keep sliding window
        if len(self.episode_rewards) > self.window_size:
            self.episode_rewards.pop(0)
            self.episode_successes.pop(0)
            self.episode_lengths.pop(0)

        allowed_outcomes = {outcome.value for outcome in EpisodeOutcome}
        outcome_value = outcome if outcome in allowed_outcomes else None
        if outcome_value is None:
            if target_finished:
                outcome_value = EpisodeOutcome.TARGET_FINISH.value
            elif target_collision or success:
                outcome_value = EpisodeOutcome.TARGET_CRASH.value
            else:
                outcome_value = EpisodeOutcome.TIMEOUT.value

        self.episode_outcomes.append(outcome_value)
        outcome_counts = {value: 0 for value in allowed_outcomes}
        for value in self.episode_outcomes:
            if value in outcome_counts:
                outcome_counts[value] += 1
        total_outcomes = len(self.episode_outcomes)
        outcome_rates = {
            key: (count / total_outcomes if total_outcomes > 0 else 0.0)
            for key, count in outcome_counts.items()
        }

        # Unified W&B logging for comparisons
        if self.wandb_run:
            count = len(self.episode_successes)
            success_rate = sum(self.episode_successes) / count if count else 0.0
            reward_mean = float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0
            steps_mean = float(np.mean(self.episode_lengths)) if self.episode_lengths else 0.0

            self.wandb_run.log({
                "train/outcome": outcome_value,
                "train/success": int(success),
                "train/episode_reward": float(reward),
                "train/episode_steps": int(length),
                "train/success_rate": success_rate,
                "train/reward_mean": reward_mean,
                "train/steps_mean": steps_mean,
                "target/success": int(bool(target_finished)),
                "target/crash": int(bool(target_collision)),
            }, step=self.episode_count)

        spawn_state = None
        # Update spawn curriculum if available
        if self.spawn_curriculum:
            spawn_state = self.spawn_curriculum.observe(self.episode_count, success)
            if spawn_state['changed'] and self.verbose > 0:
                print(f"\nSpawn curriculum: {spawn_state['stage']} "
                      f"(success rate: {spawn_state['success_rate']:.2%})")

                # Apply FTG schedule if available
                if self.ftg_agents and self.ftg_schedules:
                    self._apply_ftg_schedule(
                        stage_name=spawn_state['stage'],
                        stage_index=spawn_state['stage_index']
                    )

            # Log spawn curriculum metrics
            if self.wandb_run:
                self.wandb_run.log({
                    'curriculum/spawn/stage_index': spawn_state['stage_index'],
                    'curriculum/spawn/success_rate': spawn_state['success_rate'] or 0.0,
                    'curriculum/spawn/stage_success_rate': spawn_state['stage_success_rate'] or 0.0,
                })

        # Update phased curriculum
        if self.phases:
            self._update_phased_curriculum(reward, success)

        # Update Rich console dashboard
        if self.rich_console:
            count = len(self.episode_successes)
            success_rate = sum(self.episode_successes) / count if count else 0.0
            reward_mean = float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0
            steps_mean = float(np.mean(self.episode_lengths)) if self.episode_lengths else 0.0

            rolling_stats = {
                "success_rate": success_rate,
                "avg_reward": reward_mean,
                "avg_steps": steps_mean,
                "total_episodes": count,
                "outcome_counts": outcome_counts,
                "outcome_rates": outcome_rates,
            }

            curriculum_state = {}
            if spawn_state:
                curriculum_state.update({
                    "stage": spawn_state.get("stage"),
                    "stage_index": spawn_state.get("stage_index"),
                    "stage_success_rate": spawn_state.get("stage_success_rate"),
                })
            if self.phases and self.current_phase < len(self.phases):
                phase_name = self.phases[self.current_phase].get("name")
                phase_success = (
                    self.phase_successes / self.phase_episodes
                    if self.phase_episodes > 0
                    else 0.0
                )
                curriculum_state.update({
                    "phase_index": self.current_phase,
                    "phase_name": phase_name,
                    "phase_success_rate": phase_success,
                })

            self.rich_console.update_episode(
                episode=self.episode_count,
                outcome=outcome_value,
                reward=reward,
                steps=length,
                outcome_stats=rolling_stats,
                curriculum_state=curriculum_state or None,
                algo_name=self.algo_name,
            )

    def _update_phased_curriculum(self, reward: float, success: bool):
        """Update phased curriculum progression."""
        if self.current_phase >= len(self.phases):
            return  # Already at final phase

        phase = self.phases[self.current_phase]
        self.phase_episodes += 1
        self.phase_successes += int(success)
        self.phase_total_reward += reward

        # Check if criteria met
        criteria = phase.get('criteria', {})
        min_episodes = criteria.get('min_episodes', 50)

        if self.phase_episodes >= min_episodes:
            # Calculate metrics over window
            recent_successes = sum(self.episode_successes[-self.window_size:])
            recent_episodes = len(self.episode_successes)
            success_rate = recent_successes / recent_episodes if recent_episodes > 0 else 0.0
            avg_reward = np.mean(self.episode_rewards[-self.window_size:]) if self.episode_rewards else 0.0

            target_success_rate = criteria.get('success_rate', 0.5)
            target_reward = criteria.get('avg_reward', float('-inf'))

            criteria_met = (
                success_rate >= target_success_rate and
                avg_reward >= target_reward
            )

            if criteria_met:
                self.patience_counter = 0
                # Advance to next phase
                if self.current_phase + 1 < len(self.phases):
                    self.current_phase += 1
                    self.phase_episodes = 0
                    self.phase_successes = 0
                    self.phase_total_reward = 0.0

                    if self.verbose > 0:
                        next_phase = self.phases[self.current_phase]
                        print(f"\n{'='*60}")
                        print(f"Advancing to phase {self.current_phase}: {next_phase['name']}")
                        print(f"  Success rate: {success_rate:.2%} >= {target_success_rate:.2%}")
                        print(f"  Avg reward: {avg_reward:.1f} >= {target_reward:.1f}")
                        print(f"{'='*60}\n")

                    self._apply_phase(self.current_phase)
            else:
                # Increment patience counter
                patience = criteria.get('patience', 500)
                self.patience_counter += 1

                if self.patience_counter >= patience:
                    # Force advance despite not meeting criteria
                    if self.verbose > 0:
                        print(f"\nPatience limit reached ({patience}), forcing phase advance")

                    if self.current_phase + 1 < len(self.phases):
                        self.current_phase += 1
                        self.phase_episodes = 0
                        self.phase_successes = 0
                        self.phase_total_reward = 0.0
                        self.patience_counter = 0
                        self._apply_phase(self.current_phase)

        # Log phased curriculum metrics
        if self.wandb_run:
            recent_successes = sum(self.episode_successes[-self.window_size:])
            recent_episodes = len(self.episode_successes)
            success_rate = recent_successes / recent_episodes if recent_episodes > 0 else 0.0

            self.wandb_run.log({
                'curriculum/phased/phase': self.current_phase,
                'curriculum/phased/phase_episodes': self.phase_episodes,
                'curriculum/phased/success_rate': success_rate,
                'curriculum/phased/patience': self.patience_counter,
            })

    def _apply_phase(self, phase_idx: int):
        """Apply configuration for a curriculum phase."""
        if phase_idx >= len(self.phases):
            return

        phase = self.phases[phase_idx]

        # Update spawn configuration if available
        if self.env_wrapper and hasattr(self.env_wrapper, 'env'):
            env = self.env_wrapper.env

            # Update spawn points
            spawn_config = phase.get('spawn', {})
            if spawn_config and hasattr(env, 'set_spawn_points'):
                spawn_points = spawn_config.get('points', [])
                speed_range = spawn_config.get('speed_range', [0.3, 1.0])
                env.set_spawn_points(spawn_points, speed_range)

            # Update speed lock
            lock_speed_steps = phase.get('lock_speed_steps', 0)
            if hasattr(env, 'set_speed_lock'):
                env.set_speed_lock(lock_speed_steps)

        # Update FTG parameters
        ftg_config = phase.get('ftg', {})
        if ftg_config and self.ftg_agents:
            for agent_id, agent in self.ftg_agents.items():
                if hasattr(agent, 'apply_config'):
                    agent.apply_config(ftg_config)
                    if self.verbose > 1:
                        print(f"  Updated FTG agent {agent_id}: {ftg_config}")

    def _apply_ftg_schedule(self, stage_name: str, stage_index: int):
        """Apply FTG schedule for spawn curriculum stage."""
        for agent_id, schedule in self.ftg_schedules.items():
            if agent_id not in self.ftg_agents:
                continue

            agent = self.ftg_agents[agent_id]
            by_stage = schedule.get('by_stage', {})

            if stage_name in by_stage:
                config = by_stage[stage_name]
                if hasattr(agent, 'apply_config'):
                    agent.apply_config(config)
                    if self.verbose > 1:
                        print(f"  Applied FTG schedule to {agent_id} for stage {stage_name}")


__all__ = ['CurriculumCallback']
