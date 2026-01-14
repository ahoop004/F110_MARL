"""Curriculum learning callback for Stable Baselines3."""

from collections import deque
from typing import Any, Dict, Optional, List
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from metrics.outcomes import EpisodeOutcome

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
        wandb_logging: Optional[Dict[str, Any]] = None,
        rich_console: Optional[Any] = None,
        algo_name: Optional[str] = None,
        eval_gate_enabled: bool = True,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.curriculum_config = curriculum_config
        self.spawn_curriculum = spawn_curriculum
        self.ftg_agents = ftg_agents or {}
        self.ftg_schedules = ftg_schedules or {}
        self.env_wrapper = env_wrapper
        self.wandb_run = wandb_run
        self.wandb_logging = wandb_logging if isinstance(wandb_logging, dict) else None
        self.rich_console = rich_console
        self.algo_name = algo_name
        self.eval_gate_enabled = bool(eval_gate_enabled)

        # Eval gate configuration (hard gate: 4 consecutive 100% eval runs)
        self.eval_required_streak = 4
        self.eval_success_threshold = 1.0
        self._phase_eval_streaks: Dict[int, int] = {}
        self._phase_training_criteria_met = False

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
        self._reset_phase_histories(self.window_size)

        # Direct episode tracking (more reliable than ep_info_buffer)
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        self.last_obs = None
        self.processed_episodes = set()  # Track processed episode IDs to avoid duplicates

        # Parse curriculum config once during initialization
        self._parse_curriculum()

    def _should_log(self, key: str) -> bool:
        if not self.wandb_run:
            return False
        group_config = self._get_group_config()
        if group_config is None:
            return True
        if not group_config.get("sb3_callbacks", False):
            return False
        return bool(group_config.get(key, False))

    def _get_group_config(self) -> Optional[Dict[str, Any]]:
        if not isinstance(self.wandb_logging, dict):
            return None
        if "groups" in self.wandb_logging:
            groups = self.wandb_logging.get("groups")
            return groups if isinstance(groups, dict) else {}
        return self.wandb_logging

    def _get_metrics_config(self) -> Optional[Dict[str, Any]]:
        if not isinstance(self.wandb_logging, dict):
            return None
        metrics = self.wandb_logging.get("metrics")
        if metrics is None:
            return None
        if isinstance(metrics, dict):
            return metrics
        if isinstance(metrics, (list, tuple, set)):
            return {name: True for name in metrics}
        return None

    def _filter_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        metrics_config = self._get_metrics_config()
        if metrics_config is None:
            return metrics
        return {key: value for key, value in metrics.items() if metrics_config.get(key, False)}

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

    def _get_eval_streak(self, phase_idx: int) -> int:
        return int(self._phase_eval_streaks.get(int(phase_idx), 0))

    def _set_eval_streak(self, phase_idx: int, streak: int) -> None:
        self._phase_eval_streaks[int(phase_idx)] = int(streak)

    def _get_eval_requirements(self, phase_idx: int) -> Dict[str, float]:
        criteria = {}
        if self.phases and 0 <= phase_idx < len(self.phases):
            criteria = self.phases[phase_idx].get("criteria", {}) or {}
        required = int(criteria.get("eval_required_runs", self.eval_required_streak))
        threshold = float(criteria.get("eval_success_rate", self.eval_success_threshold))
        return {"required": required, "threshold": threshold}

    def get_current_phase_config(self) -> Optional[Dict[str, Any]]:
        if not self.phases or self.current_phase >= len(self.phases):
            return None
        return self.phases[self.current_phase]

    def get_current_phase_index(self) -> Optional[int]:
        if not self.phases:
            return None
        return int(self.current_phase)

    def should_run_eval(self) -> bool:
        if not self.eval_gate_enabled or not self.phases:
            return False
        if not self._phase_training_criteria_met:
            return False
        required = self._get_eval_requirements(self.current_phase)["required"]
        return self._get_eval_streak(self.current_phase) < required

    def record_eval_result(self, success_rate: float, phase_index: Optional[int] = None) -> None:
        if not self.eval_gate_enabled or not self.phases:
            return
        phase_idx = self.current_phase if phase_index is None else int(phase_index)
        if phase_idx != self.current_phase:
            # Ignore stale eval results from earlier phases.
            return

        reqs = self._get_eval_requirements(phase_idx)
        if success_rate >= reqs["threshold"]:
            new_streak = self._get_eval_streak(phase_idx) + 1
        else:
            new_streak = 0
        self._set_eval_streak(phase_idx, new_streak)

        if self.verbose > 0:
            print(
                f"[Eval gate] Phase {phase_idx}: "
                f"success_rate={success_rate:.2%}, "
                f"streak={new_streak}/{reqs['required']}"
            )

    def _init_callback(self) -> None:
        """Initialize callback at start of training."""
        if self.phases and self.current_phase < len(self.phases):
            self._set_eval_streak(self.current_phase, 0)
            self._phase_training_criteria_met = False
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
        """Called after each environment step.

        Uses direct tracking of rewards and dones from locals instead of
        unreliable ep_info_buffer which can lose data.
        """
        # Get current step info from locals
        if 'rewards' in self.locals:
            step_reward = self.locals['rewards'][0] if isinstance(self.locals['rewards'], (list, np.ndarray)) else self.locals['rewards']
            self.current_episode_reward += step_reward
            self.current_episode_length += 1

        # Check if episode ended
        if 'dones' in self.locals:
            done = self.locals['dones'][0] if isinstance(self.locals['dones'], (list, np.ndarray)) else self.locals['dones']

            if done:
                # Extract episode info from the environment wrapper's last info
                info = {}
                if 'infos' in self.locals and len(self.locals['infos']) > 0:
                    info = self.locals['infos'][0] if isinstance(self.locals['infos'], list) else self.locals['infos']

                # Process the completed episode
                self._process_episode(
                    reward=self.current_episode_reward,
                    length=self.current_episode_length,
                    success=info.get('is_success', False),
                    target_finished=info.get('target_finished', False),
                    target_collision=info.get('target_collision', False),
                    outcome=info.get('outcome')
                )

                # Reset tracking for next episode
                self.current_episode_reward = 0.0
                self.current_episode_length = 0

        # Also check ep_info_buffer as fallback (for compatibility)
        # but with deduplication to avoid double-counting
        if len(self.model.ep_info_buffer) > 0:
            for ep_info in self.model.ep_info_buffer:
                if 'r' in ep_info and 'l' in ep_info:
                    # Create unique ID for this episode to avoid reprocessing
                    ep_id = (ep_info['r'], ep_info['l'], self.num_timesteps)
                    if ep_id not in self.processed_episodes:
                        self.processed_episodes.add(ep_id)
                        # Only process if we haven't tracked it via direct method
                        # (direct method resets current_episode_reward to 0)
                        if self.current_episode_reward == 0.0 and self.current_episode_length == 0:
                            self._process_episode(
                                reward=ep_info['r'],
                                length=ep_info['l'],
                                success=ep_info.get('is_success', False),
                                target_finished=ep_info.get('target_finished', False),
                                target_collision=ep_info.get('target_collision', False),
                                outcome=ep_info.get('outcome')
                            )
            # Clear after processing
            self.model.ep_info_buffer.clear()

        # Clean up old episode IDs to prevent memory growth
        if len(self.processed_episodes) > 10000:
            self.processed_episodes.clear()

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
        if self._should_log("train") or self._should_log("target"):
            count = len(self.episode_successes)
            success_rate = sum(self.episode_successes) / count if count else 0.0
            reward_mean = float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0
            steps_mean = float(np.mean(self.episode_lengths)) if self.episode_lengths else 0.0

            log_dict = {}
            if self._should_log("train"):
                log_dict.update({
                    "train/outcome": outcome_value,
                    "train/success": int(success),
                    "train/episode": int(self.episode_count),
                    "train/episode_reward": float(reward),
                    "train/episode_steps": int(length),
                    "train/success_rate": success_rate,
                    "train/reward_mean": reward_mean,
                    "train/steps_mean": steps_mean,
                })
            if self._should_log("target"):
                log_dict.update({
                    "target/success": int(bool(target_finished)),
                    "target/crash": int(bool(target_collision)),
                })
            if log_dict:
                log_dict = self._filter_metrics(log_dict)
                if log_dict:
                    self.wandb_run.log(
                        log_dict,
                        step=self.episode_count,
                    )  # FIXED: Use episode_count instead of num_timesteps

        spawn_state = None
        # Update spawn curriculum if available (skip when phased curriculum is active)
        if self.spawn_curriculum and not self.phases:
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

            # Log minimal curriculum metrics if no phased curriculum
            if self._should_log("curriculum") and not self.phases:
                curriculum_log = self._filter_metrics({
                    'train/episode': int(self.episode_count),
                    'curriculum/stage': spawn_state['stage'],
                    'curriculum/stage_success_rate': spawn_state['stage_success_rate'] or 0.0,
                })
                if curriculum_log:
                    self.wandb_run.log(
                        curriculum_log,
                        step=self.episode_count,
                    )  # FIXED: Use episode_count instead of num_timesteps

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
                if self.phase_success_history:
                    phase_success = sum(self.phase_success_history) / len(self.phase_success_history)
                else:
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
        if self.phase_success_history is not None:
            self.phase_success_history.append(bool(success))
        if self.phase_reward_history is not None:
            self.phase_reward_history.append(float(reward))

        # Check if criteria met
        criteria = phase.get('criteria', {})
        min_episodes = criteria.get('min_episodes', 50)
        self._phase_training_criteria_met = False

        if self.phase_episodes >= min_episodes:
            # Calculate metrics over phase-local window (respects criteria.window_size)
            window_size = int(criteria.get('window_size', self.window_size))
            if self.phase_success_history is None or self.phase_reward_history is None:
                self._reset_phase_histories(window_size)
            elif self.phase_success_history.maxlen != window_size:
                self._reset_phase_histories(window_size)

            recent_episodes = len(self.phase_success_history)
            success_rate = (
                sum(self.phase_success_history) / recent_episodes
                if recent_episodes > 0
                else 0.0
            )
            avg_reward = (
                float(np.mean(self.phase_reward_history))
                if self.phase_reward_history
                else 0.0
            )

            target_success_rate = criteria.get('success_rate', 0.5)
            target_reward = criteria.get('avg_reward', float('-inf'))

            criteria_met = (
                success_rate >= target_success_rate and
                avg_reward >= target_reward
            )
            self._phase_training_criteria_met = bool(criteria_met)
            reqs = self._get_eval_requirements(self.current_phase)
            eval_gate_satisfied = (
                not self.eval_gate_enabled
                or self._get_eval_streak(self.current_phase) >= reqs["required"]
            )

            if criteria_met and eval_gate_satisfied:
                self.patience_counter = 0
                # Advance to next phase
                if self.current_phase + 1 < len(self.phases):
                    self.current_phase += 1
                    self.phase_episodes = 0
                    self.phase_successes = 0
                    self.phase_total_reward = 0.0
                    self._reset_phase_histories(criteria.get('window_size', self.window_size))
                    self._set_eval_streak(self.current_phase, 0)
                    self._phase_training_criteria_met = False

                    if self.verbose > 0:
                        next_phase = self.phases[self.current_phase]
                        print(f"\n{'='*60}")
                        print(f"Advancing to phase {self.current_phase}: {next_phase['name']}")
                        print(f"  Success rate: {success_rate:.2%} >= {target_success_rate:.2%}")
                        print(f"  Avg reward: {avg_reward:.1f} >= {target_reward:.1f}")
                        print(f"{'='*60}\n")

                    self._apply_phase(self.current_phase)
            elif criteria_met and not eval_gate_satisfied:
                # Hold phase while eval gate is not satisfied.
                self.patience_counter = 0
            elif not criteria_met:
                # Increment patience counter
                patience = criteria.get('patience', 500)
                self.patience_counter += 1

                if self.patience_counter >= patience:
                    if self.eval_gate_enabled:
                        if self.verbose > 0:
                            print(
                                f"\nPatience limit reached ({patience}) "
                                "but eval gate is enabled; holding phase."
                            )
                    else:
                        # Force advance despite not meeting criteria
                        if self.verbose > 0:
                            print(f"\nPatience limit reached ({patience}), forcing phase advance")

                        if self.current_phase + 1 < len(self.phases):
                            self.current_phase += 1
                            self.phase_episodes = 0
                            self.phase_successes = 0
                            self.phase_total_reward = 0.0
                            self.patience_counter = 0
                            self._reset_phase_histories(criteria.get('window_size', self.window_size))
                            self._set_eval_streak(self.current_phase, 0)
                            self._phase_training_criteria_met = False
                            self._apply_phase(self.current_phase)

        # Log minimal phased curriculum metrics
        if self._should_log("curriculum_phase") and self.phases:
            phase_name = None
            if 0 <= self.current_phase < len(self.phases):
                phase_name = self.phases[self.current_phase].get("name")
            reqs = self._get_eval_requirements(self.current_phase)

            phase_success_rate = (
                sum(self.phase_success_history) / len(self.phase_success_history)
                if self.phase_success_history
                else (
                    self.phase_successes / self.phase_episodes
                    if self.phase_episodes > 0
                    else 0.0
                )
            )

            self.wandb_run.log({
                'train/episode': int(self.episode_count),
                'curriculum/phase_idx': int(self.current_phase),
                'curriculum/phase_name': phase_name,
                'curriculum/phase_success_rate': phase_success_rate,
                'curriculum/stage': phase_name,
                'curriculum/stage_success_rate': phase_success_rate,
                'curriculum/eval_success_streak': self._get_eval_streak(self.current_phase),
                'curriculum/criteria_eval_success_rate': reqs["threshold"],
                'curriculum/criteria_eval_required_runs': reqs["required"],
            }, step=self.episode_count)  # FIXED: Use episode_count instead of num_timesteps

    def _apply_phase(self, phase_idx: int):
        """Apply configuration for a curriculum phase."""
        if phase_idx >= len(self.phases):
            return

        phase = self.phases[phase_idx]
        criteria = phase.get('criteria', {})
        window_size = int(criteria.get('window_size', self.window_size))
        self._reset_phase_histories(window_size)

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
        if self.spawn_curriculum:
            from curriculum.curriculum_env import apply_curriculum_to_spawn_curriculum
            apply_curriculum_to_spawn_curriculum(self.spawn_curriculum, phase)

        # Update FTG parameters
        ftg_config = phase.get('ftg', {})
        if ftg_config and self.ftg_agents:
            for agent_id, agent in self.ftg_agents.items():
                if hasattr(agent, 'apply_config'):
                    agent.apply_config(ftg_config)
                    if self.verbose > 1:
                        print(f"  Updated FTG agent {agent_id}: {ftg_config}")

    def _reset_phase_histories(self, window_size: int) -> None:
        """Reset per-phase rolling histories."""
        history_len = max(1, int(window_size))
        self.phase_success_history = deque(maxlen=history_len)
        self.phase_reward_history = deque(maxlen=history_len)

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
                else:
                    if self.verbose > 0:
                        print(f"  Warning: Agent {agent_id} doesn't have apply_config method")
            else:
                # Stage not in schedule - could be intentional (not all stages need updates)
                if self.verbose > 1:
                    available_stages = ', '.join(by_stage.keys())
                    print(f"  Note: Stage '{stage_name}' not in FTG schedule for {agent_id}")
                    if available_stages:
                        print(f"        Available stages: {available_stages}")


__all__ = ['CurriculumCallback']
