"""Stable-Baselines3 agent wrappers for F110 training system.

Wraps SB3 algorithms (SAC, TD3, PPO) to work with the existing
agent interface and training loop.
"""

from typing import Any, Dict, Optional
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from gymnasium import spaces

try:
    from stable_baselines3 import SAC, TD3, PPO, DDPG, A2C
    from stable_baselines3.common.buffers import ReplayBuffer
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.utils import polyak_update
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("Warning: stable-baselines3 not installed. Run: pip install stable-baselines3")

# TQC and QR-DQN are in sb3-contrib
try:
    from sb3_contrib import TQC, QRDQN
    from sb3_contrib.common.utils import quantile_huber_loss
    TQC_AVAILABLE = True
    QRDQN_AVAILABLE = True
except ImportError:
    TQC_AVAILABLE = False
    QRDQN_AVAILABLE = False
    quantile_huber_loss = None
    # Don't warn - contrib algorithms are optional

# Try to import DQN (for discrete actions)
try:
    from stable_baselines3 import DQN
    DQN_AVAILABLE = True
except ImportError:
    DQN_AVAILABLE = False

from replay import PrioritizedReplayBuffer


class DummyEnv(gym.Env):
    """Dummy environment for SB3 initialization (continuous actions).

    SB3 requires a gym.Env to initialize models, but we handle
    the actual environment interaction in the training loop.
    This dummy env just provides the observation/action spaces.
    """

    def __init__(self, obs_dim: int, action_low: np.ndarray, action_high: np.ndarray):
        super().__init__()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=action_low, high=action_high, dtype=np.float32
        )

    def reset(self, **kwargs):
        return self.observation_space.sample(), {}

    def step(self, action):
        obs = self.observation_space.sample()
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info


class DummyDiscreteEnv(gym.Env):
    """Dummy environment for SB3 initialization (discrete actions).

    Used for DQN-based algorithms that operate on discretized action spaces.
    """

    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(n_actions)

    def reset(self, **kwargs):
        return self.observation_space.sample(), {}

    def step(self, action):
        obs = self.observation_space.sample()
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info


def _parse_per_config(params: Dict[str, Any]) -> Dict[str, Any]:
    per_cfg = params.get('per', {}) or {}
    return {
        'enabled': bool(per_cfg.get('enabled', False)),
        'alpha': float(per_cfg.get('alpha', 0.6)),
        'beta_start': float(per_cfg.get('beta', per_cfg.get('beta_start', 0.4))),
        'beta_final': float(per_cfg.get('beta_final', per_cfg.get('beta_end', 1.0))),
        'beta_anneal_steps': int(per_cfg.get('beta_anneal_steps', 100_000)),
        'eps': float(per_cfg.get('epsilon', per_cfg.get('eps', 1e-6))),
        'max_priority': float(per_cfg.get('max_priority', 1.0)),
        'normalize_weights': bool(per_cfg.get('normalize_weights', True)),
    }


def _resolve_per_weights(replay_data: Any, batch_size: int) -> tuple[th.Tensor, Optional[np.ndarray]]:
    weights = getattr(replay_data, "weights", None)
    indices = getattr(replay_data, "indices", None)
    if weights is None:
        device = replay_data.observations.device
        weights = th.ones((batch_size, 1), device=device)
    return weights, indices


def _update_priorities(replay_buffer: Any, indices: Optional[np.ndarray], priorities: th.Tensor) -> None:
    if indices is None:
        return
    update_fn = getattr(replay_buffer, "update_priorities", None)
    if not callable(update_fn):
        return
    priorities_np = priorities.detach().cpu().numpy().reshape(-1)
    update_fn(indices, priorities_np)


class SB3AgentBase:
    """Base wrapper for SB3 agents to match F110 agent interface.

    This wrapper adapts Stable-Baselines3 algorithms to work with the F110 training system.
    Key responsibilities:
    1. Translate between F110's agent protocol and SB3's API
    2. Handle observation/action space configuration
    3. Manage replay buffer and gradient updates
    4. Provide checkpointing/loading functionality

    Protocol methods (required by enhanced_training.py):
    - act(obs, deterministic): Select action given observation
    - store_transition(...): Store transition in replay buffer (off-policy only)
    - update(): Perform gradient step and return training metrics
    - get_state() / load_state(): Checkpointing support
    """

    def __init__(self, cfg: Dict[str, Any], model_class):
        """Initialize base agent with common SB3 hyperparameters.

        Args:
            cfg: Configuration dict with obs_dim, params, device, etc.
            model_class: SB3 algorithm class (SAC, TD3, TQC, PPO, etc.)
        """
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 required. Install with: pip install stable-baselines3")

        # ========================================
        # CONFIGURATION
        # ========================================

        self.cfg = cfg
        self.model_class = model_class
        self.model = None  # Created by subclass after setting algorithm-specific params
        self._setup_done = False
        self._steps = 0  # Track total steps for learning_starts threshold

        # Observation and action dimensions
        self.obs_dim = cfg['obs_dim']  # e.g., 119 for gaplock (108 LiDAR + 11 state)
        self.act_dim = cfg.get('act_dim', 2)  # [steering, velocity]
        self.device = cfg.get('device', 'cuda')

        # Action bounds for continuous control
        # Default: steering [-0.46, 0.46], velocity [-1.0, 1.0]
        self.action_low = np.array(cfg.get('action_low', [-0.46, -1.0]))
        self.action_high = np.array(cfg.get('action_high', [0.46, 1.0]))

        # ========================================
        # COMMON SB3 HYPERPARAMETERS
        # ========================================

        params = cfg.get('params', {})

        # Optimization
        self.learning_rate = params.get('learning_rate', 3e-4)  # Adam LR

        # Off-policy learning (replay buffer)
        self.buffer_size = params.get('buffer_size', 1_000_000)  # Replay buffer capacity
        self.batch_size = params.get('batch_size', 256)  # SGD minibatch size
        self.learning_starts = params.get('learning_starts', 1000)  # Random exploration steps

        # Temporal difference learning
        self.gamma = params.get('gamma', 0.995)  # Discount factor (0.995 for long horizon)
        self.tau = params.get('tau', 0.005)  # Polyak averaging for target networks

        # Network architecture
        # Default: 2-layer MLP with 256 units each
        # Applied to both actor and critic networks
        hidden_dims = params.get('hidden_dims', [256, 256])
        pi_dims = params.get('pi_hidden_dims')
        qf_dims = params.get('qf_hidden_dims')
        model_name = self.model_class.__name__.replace("WithPER", "")
        supports_split = model_name in {"SAC", "TD3", "TQC", "DDPG"}
        if supports_split and (pi_dims is not None or qf_dims is not None):
            if pi_dims is None:
                pi_dims = hidden_dims
            if qf_dims is None:
                qf_dims = hidden_dims
            net_arch = {'pi': pi_dims, 'qf': qf_dims}
        else:
            net_arch = hidden_dims
        self.policy_kwargs = {'net_arch': net_arch}

        activation = params.get('activation')
        if activation is not None:
            if isinstance(activation, str):
                activation_key = activation.strip().lower()
                activation_map = {
                    "relu": nn.ReLU,
                    "silu": nn.SiLU,
                    "swish": nn.SiLU,
                    "tanh": nn.Tanh,
                }
                if activation_key not in activation_map:
                    raise ValueError(f"Unsupported activation '{activation}'. Use: relu, silu/swish, tanh.")
                self.policy_kwargs['activation_fn'] = activation_map[activation_key]
            else:
                self.policy_kwargs['activation_fn'] = activation

        # Hindsight Experience Replay (HER) configuration (optional)
        her_cfg = params.get('her', {})
        if isinstance(her_cfg, bool):
            her_cfg = {'enabled': her_cfg}
        if her_cfg is None:
            her_cfg = {}
        self.her_enabled = bool(her_cfg.get('enabled', False))
        self.her_reward_mode = str(her_cfg.get('reward_mode', 'replace')).lower()
        success_mode = her_cfg.get('success_mode') or her_cfg.get('success_signal') or 'distance'
        self.her_success_mode = str(success_mode).lower()
        self.her_distance_threshold = float(her_cfg.get('distance_threshold', 1.0))
        self.her_success_reward = float(her_cfg.get('success_reward', 1.0))
        self.her_failure_reward = float(her_cfg.get('failure_reward', 0.0))
        self.her_distance_scale = float(her_cfg.get('distance_scale', 1.0))
        self.her_distance_offset = float(her_cfg.get('distance_offset', 0.0))
        self.her_done_on_success = bool(her_cfg.get('done_on_success', False))
        self.her_probability = float(her_cfg.get('probability', 1.0))
        self.her_probability = max(0.0, min(1.0, self.her_probability))

        distance_clip = her_cfg.get('distance_clip')
        if isinstance(distance_clip, (list, tuple)) and len(distance_clip) == 2:
            self.her_distance_clip = (float(distance_clip[0]), float(distance_clip[1]))
        else:
            self.her_distance_clip = None

        # Prioritized Experience Replay (optional, off-policy only)
        per_cfg = _parse_per_config(params)
        self.per_enabled = bool(per_cfg['enabled'])
        self.per_alpha = float(per_cfg['alpha'])
        self.per_beta_start = float(per_cfg['beta_start'])
        self.per_beta_final = float(per_cfg['beta_final'])
        self.per_beta_anneal_steps = int(per_cfg['beta_anneal_steps'])
        self.per_eps = float(per_cfg['eps'])
        self.per_max_priority = float(per_cfg['max_priority'])
        self.per_normalize_weights = bool(per_cfg['normalize_weights'])

        # Model creation deferred to subclass (after algorithm-specific params set)

    def _create_model(self, env):
        """Create SB3 model (called on first act())."""
        raise NotImplementedError

    def _get_per_replay_buffer(self) -> tuple[Optional[type], Optional[Dict[str, Any]]]:
        if not self.per_enabled:
            return None, None
        replay_buffer_kwargs = {
            "alpha": self.per_alpha,
            "beta": self.per_beta_start,
            "beta_final": self.per_beta_final,
            "beta_anneal_steps": self.per_beta_anneal_steps,
            "eps": self.per_eps,
            "max_priority": self.per_max_priority,
            "normalize_weights": self.per_normalize_weights,
        }
        return PrioritizedReplayBuffer, replay_buffer_kwargs

    def _setup_logger(self):
        """Set up a basic logger for the model."""
        if self.model is not None:
            # Create a minimal logger (no output to avoid clutter)
            logger = configure(None, [])
            self.model.set_logger(logger)

    def act(self, obs: np.ndarray, deterministic: bool = False, info: Optional[Dict] = None) -> np.ndarray:
        """Select action using SB3 policy.

        Args:
            obs: Flattened, normalized observation vector (shape: obs_dim,)
            deterministic: If True, use mean action (no exploration noise)
                          If False, sample from policy distribution
            info: Optional info dict (unused, for API compatibility)

        Returns:
            Action array (shape: act_dim,) bounded by [action_low, action_high]

        Action Selection:
        - Stochastic (training): a ~ π(·|s) = N(μ_θ(s), σ_θ(s))
        - Deterministic (eval): a = μ_θ(s)
        """
        if not self._setup_done:
            raise RuntimeError("Model not initialized. Check __init__ implementation.")

        # SB3's predict() handles both stochastic and deterministic modes
        # Returns (action, _state) tuple; we ignore _state for stateless policies
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return action

    def store_transition(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        info: Optional[Dict] = None
    ) -> None:
        """Store transition in replay buffer for off-policy learning.

        Args:
            obs: Current observation (s_t)
            action: Action taken (a_t)
            reward: Reward received (r_t)
            next_obs: Next observation (s_{t+1})
            done: Terminal flag (episode ended)
            info: Optional info dict (unused)

        Replay Buffer:
        - Stores (s, a, r, s', done) transitions
        - Circular buffer with capacity buffer_size
        - Sampled uniformly during update() for SGD
        """
        if self.model is None or not hasattr(self.model, 'replay_buffer'):
            return

        # SB3 replay buffer expects specific signature:
        # add(obs, next_obs, action, reward, done, infos)
        infos = [info] if info is not None else [{}]
        self.model.replay_buffer.add(
            obs, next_obs, action, reward, done, infos
        )
        self._steps += 1  # Track steps for learning_starts threshold

    def _get_replay_action(self, action: np.ndarray) -> np.ndarray:
        """Return the action to store in the replay buffer."""
        return action

    def _derive_her_success(self, distance: float, info: Optional[Dict]) -> bool:
        """Determine HER success condition from config and info."""
        if self.her_success_mode in ("target_crash", "target_collision"):
            if not isinstance(info, dict):
                return False
            if info.get("target_crash", False) or info.get("success", False):
                return True
            target_collision = bool(info.get("target_collision", False))
            if self.her_success_mode == "target_collision":
                return target_collision
            attacker_collision = bool(info.get("collision", False))
            return target_collision and not attacker_collision
        return distance <= self.her_distance_threshold

    def _compute_her_reward(self, base_reward: float, distance: float, success: bool) -> float:
        """Compute HER reward from success flag and distance."""
        base_reward = float(base_reward) if base_reward is not None else 0.0

        if self.her_reward_mode in ("replace", "sparse"):
            reward = self.her_success_reward if success else self.her_failure_reward
        elif self.her_reward_mode == "bonus":
            bonus = self.her_success_reward if success else self.her_failure_reward
            reward = base_reward + bonus
        elif self.her_reward_mode == "distance":
            reward = (-distance * self.her_distance_scale) + self.her_distance_offset
            if self.her_distance_clip is not None:
                reward = float(np.clip(reward, self.her_distance_clip[0], self.her_distance_clip[1]))
        else:
            reward = base_reward

        return float(reward)

    def store_hindsight_transition(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        distance: float,
        info: Optional[Dict] = None,
    ) -> None:
        """Store an optional HER relabeled transition (off-policy only)."""
        if not self.her_enabled:
            return
        if self.model is None or not hasattr(self.model, 'replay_buffer'):
            return

        try:
            distance_val = float(distance)
        except (TypeError, ValueError):
            return
        if not np.isfinite(distance_val):
            return
        if self.her_probability < 1.0 and np.random.random() > self.her_probability:
            return

        success = self._derive_her_success(distance_val, info)
        her_reward = self._compute_her_reward(reward, distance_val, success)
        her_done = bool(done) or (self.her_done_on_success and success)

        her_info = {}
        if isinstance(info, dict):
            her_info.update(info)
        her_info['her'] = True
        her_info['her_distance'] = distance_val
        her_info['her_success'] = success

        infos = [her_info]
        replay_action = self._get_replay_action(action)
        self.model.replay_buffer.add(
            obs, next_obs, replay_action, her_reward, her_done, infos
        )

    def update(self) -> Optional[Dict[str, float]]:
        """Perform one gradient step using minibatch from replay buffer.

        Update Procedure (for TQC/SAC/TD3):
        1. Sample minibatch of size batch_size from replay buffer
        2. Compute critic loss: L_Q = E[(Q(s,a) - y)²] where y = r + γQ_target(s',a')
        3. Update critics via gradient descent: θ_Q ← θ_Q - α∇L_Q
        4. Compute actor loss: L_π = -E[Q(s, π(s))] (maximize Q-value)
        5. Update actor via gradient ascent: θ_π ← θ_π - α∇L_π
        6. Update target networks via Polyak averaging: θ_target ← (1-τ)θ_target + τθ
        7. (If ent_coef='auto') Update entropy coefficient

        Returns:
            None (SB3 doesn't expose per-step metrics in a consistent way)
            Training metrics logged internally by SB3 if logger configured

        Note:
        - Only called for OFF-POLICY agents (SAC, TD3, TQC, DQN)
        - On-policy agents (PPO, A2C) handle updates differently
        """
        if self.model is None or not hasattr(self.model, 'replay_buffer'):
            return None

        # Don't train until we've collected enough random exploration data
        if self._steps < self.learning_starts:
            return None

        # Ensure replay buffer has enough samples for a minibatch
        if self.model.replay_buffer.size() < self.batch_size:
            return None

        try:
            # Perform one gradient step
            # gradient_steps=1: Single SGD update per environment step
            self.model.train(gradient_steps=1, batch_size=self.batch_size)
        except Exception as e:
            print(f"Warning: SB3 training step failed: {e}")
            return None

        # Return None to avoid W&B logging conflicts
        # (Training loop handles logging separately)
        return None

    def save(self, path: str) -> None:
        """Save model."""
        if self.model is not None:
            self.model.save(path)

    def load(self, path: str) -> None:
        """Load model."""
        self.model = self.model_class.load(path)
        self._setup_done = True

    def get_state(self) -> Dict[str, Any]:
        """Return model state for checkpointing."""
        if self.model is None:
            return {}

        state: Dict[str, Any] = {}

        policy = getattr(self.model, "policy", None)
        if policy is not None:
            try:
                state["policy"] = {
                    key: value.detach().cpu()
                    for key, value in policy.state_dict().items()
                }
            except Exception:
                pass

        if hasattr(self.model, "get_parameters"):
            try:
                state["parameters"] = self.model.get_parameters()
            except Exception:
                pass

        return state

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load model state from checkpoint."""
        if self.model is None or not state:
            return

        if "parameters" in state and hasattr(self.model, "set_parameters"):
            try:
                self.model.set_parameters(state["parameters"], exact_match=True)
                return
            except Exception:
                pass

        policy_state = state.get("policy")
        if policy_state is not None and hasattr(self.model, "policy"):
            self.model.policy.load_state_dict(policy_state)

    def get_optimizer_state(self) -> Optional[Dict[str, Any]]:
        """Return optimizer state for checkpointing."""
        if self.model is None:
            return None

        optim_state: Dict[str, Any] = {}

        policy = getattr(self.model, "policy", None)
        if policy is not None:
            optimizer = getattr(policy, "optimizer", None)
            if optimizer is not None:
                optim_state["policy_optimizer"] = optimizer.state_dict()

        ent_coef_optimizer = getattr(self.model, "ent_coef_optimizer", None)
        if ent_coef_optimizer is not None:
            optim_state["ent_coef_optimizer"] = ent_coef_optimizer.state_dict()

        if not optim_state:
            return None
        return optim_state

    def load_optimizer_state(self, state: Dict[str, Any]) -> None:
        """Load optimizer state from checkpoint."""
        if self.model is None or not state:
            return

        policy = getattr(self.model, "policy", None)
        if policy is not None:
            optimizer = getattr(policy, "optimizer", None)
            if optimizer is not None and "policy_optimizer" in state:
                optimizer.load_state_dict(state["policy_optimizer"])

        ent_coef_optimizer = getattr(self.model, "ent_coef_optimizer", None)
        if ent_coef_optimizer is not None and "ent_coef_optimizer" in state:
            ent_coef_optimizer.load_state_dict(state["ent_coef_optimizer"])


def _quantile_huber_loss_per_sample(
    current_quantiles: th.Tensor,
    target_quantiles: th.Tensor,
) -> th.Tensor:
    """Compute per-sample quantile huber loss (no reduction across batch)."""
    if current_quantiles.ndim not in (2, 3):
        raise ValueError("current_quantiles must be 2D or 3D for quantile huber loss.")

    n_quantiles = current_quantiles.shape[-1]
    cum_prob = (th.arange(n_quantiles, device=current_quantiles.device, dtype=th.float) + 0.5) / n_quantiles
    if current_quantiles.ndim == 2:
        cum_prob = cum_prob.view(1, -1, 1)
    else:
        cum_prob = cum_prob.view(1, 1, -1, 1)

    pairwise_delta = target_quantiles.unsqueeze(-2) - current_quantiles.unsqueeze(-1)
    abs_pairwise_delta = th.abs(pairwise_delta)
    huber_loss = th.where(abs_pairwise_delta > 1, abs_pairwise_delta - 0.5, pairwise_delta**2 * 0.5)
    loss = th.abs(cum_prob - (pairwise_delta.detach() < 0).float()) * huber_loss
    reduce_dims = tuple(range(1, loss.ndim))
    return loss.mean(dim=reduce_dims)


if SB3_AVAILABLE and polyak_update is not None:
    class SACWithPER(SAC):
        """SAC variant that applies PER weights and priority updates."""

        def train(self, gradient_steps: int, batch_size: int = 64) -> None:
            self.policy.set_training_mode(True)
            optimizers = [self.actor.optimizer, self.critic.optimizer]
            if self.ent_coef_optimizer is not None:
                optimizers += [self.ent_coef_optimizer]

            self._update_learning_rate(optimizers)

            ent_coef_losses, ent_coefs = [], []
            actor_losses, critic_losses = [], []

            for gradient_step in range(gradient_steps):
                replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
                weights, indices = _resolve_per_weights(replay_data, batch_size)
                discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

                if self.use_sde:
                    self.actor.reset_noise()

                actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
                log_prob = log_prob.reshape(-1, 1)

                ent_coef_loss = None
                if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                    ent_coef = th.exp(self.log_ent_coef.detach())
                    ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                    ent_coef_losses.append(ent_coef_loss.item())
                else:
                    ent_coef = self.ent_coef_tensor

                ent_coefs.append(ent_coef.item())

                if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                    self.ent_coef_optimizer.zero_grad()
                    ent_coef_loss.backward()
                    self.ent_coef_optimizer.step()

                with th.no_grad():
                    next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                    next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                    next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                    next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                    target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

                current_q_values = self.critic(replay_data.observations, replay_data.actions)

                td_errors = []
                per_losses = []
                for current_q in current_q_values:
                    td_error = target_q_values - current_q
                    td_errors.append(td_error)
                    per_loss = F.mse_loss(current_q, target_q_values, reduction="none")
                    per_losses.append(per_loss)

                critic_loss = 0.5 * sum((per_loss * weights).mean() for per_loss in per_losses)
                critic_losses.append(critic_loss.item())

                self.critic.optimizer.zero_grad()
                critic_loss.backward()
                self.critic.optimizer.step()

                q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
                min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
                actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
                actor_losses.append(actor_loss.item())

                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                if gradient_step % self.target_update_interval == 0:
                    polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                    polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

                if td_errors:
                    td_stack = th.stack([th.abs(td) for td in td_errors], dim=1)
                    td_priority = td_stack.mean(dim=1)
                    _update_priorities(self.replay_buffer, indices, td_priority)

            self._n_updates += gradient_steps
            self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
            self.logger.record("train/ent_coef", np.mean(ent_coefs))
            self.logger.record("train/actor_loss", np.mean(actor_losses))
            self.logger.record("train/critic_loss", np.mean(critic_losses))
            if len(ent_coef_losses) > 0:
                self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))


    class TD3WithPER(TD3):
        """TD3 variant that applies PER weights and priority updates."""

        def train(self, gradient_steps: int, batch_size: int = 100) -> None:
            self.policy.set_training_mode(True)
            self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

            actor_losses, critic_losses = [], []
            for _ in range(gradient_steps):
                self._n_updates += 1
                replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
                weights, indices = _resolve_per_weights(replay_data, batch_size)
                discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

                with th.no_grad():
                    noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                    noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                    next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)

                    next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                    next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                    target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

                current_q_values = self.critic(replay_data.observations, replay_data.actions)

                td_errors = []
                per_losses = []
                for current_q in current_q_values:
                    td_error = target_q_values - current_q
                    td_errors.append(td_error)
                    per_loss = F.mse_loss(current_q, target_q_values, reduction="none")
                    per_losses.append(per_loss)

                critic_loss = sum((per_loss * weights).mean() for per_loss in per_losses)
                critic_losses.append(critic_loss.item())

                self.critic.optimizer.zero_grad()
                critic_loss.backward()
                self.critic.optimizer.step()

                if self._n_updates % self.policy_delay == 0:
                    actor_loss = -self.critic.q1_forward(
                        replay_data.observations,
                        self.actor(replay_data.observations)
                    ).mean()
                    actor_losses.append(actor_loss.item())

                    self.actor.optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor.optimizer.step()

                    polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                    polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
                    polyak_update(self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0)
                    polyak_update(self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0)

                if td_errors:
                    td_stack = th.stack([th.abs(td) for td in td_errors], dim=1)
                    td_priority = td_stack.mean(dim=1)
                    _update_priorities(self.replay_buffer, indices, td_priority)

            self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
            if len(actor_losses) > 0:
                self.logger.record("train/actor_loss", np.mean(actor_losses))
            self.logger.record("train/critic_loss", np.mean(critic_losses))


    class DDPGWithPER(DDPG):
        """DDPG variant that applies PER weights and priority updates."""

        def train(self, gradient_steps: int, batch_size: int = 100) -> None:
            self.policy.set_training_mode(True)
            self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

            actor_losses, critic_losses = [], []
            for _ in range(gradient_steps):
                self._n_updates += 1
                replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
                weights, indices = _resolve_per_weights(replay_data, batch_size)
                discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

                with th.no_grad():
                    noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                    noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                    next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)

                    next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                    next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                    target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

                current_q_values = self.critic(replay_data.observations, replay_data.actions)

                td_errors = []
                per_losses = []
                for current_q in current_q_values:
                    td_error = target_q_values - current_q
                    td_errors.append(td_error)
                    per_loss = F.mse_loss(current_q, target_q_values, reduction="none")
                    per_losses.append(per_loss)

                critic_loss = sum((per_loss * weights).mean() for per_loss in per_losses)
                critic_losses.append(critic_loss.item())

                self.critic.optimizer.zero_grad()
                critic_loss.backward()
                self.critic.optimizer.step()

                if self._n_updates % self.policy_delay == 0:
                    actor_loss = -self.critic.q1_forward(
                        replay_data.observations,
                        self.actor(replay_data.observations)
                    ).mean()
                    actor_losses.append(actor_loss.item())

                    self.actor.optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor.optimizer.step()

                    polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                    polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
                    polyak_update(self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0)
                    polyak_update(self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0)

                if td_errors:
                    td_stack = th.stack([th.abs(td) for td in td_errors], dim=1)
                    td_priority = td_stack.mean(dim=1)
                    _update_priorities(self.replay_buffer, indices, td_priority)

            self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
            if len(actor_losses) > 0:
                self.logger.record("train/actor_loss", np.mean(actor_losses))
            self.logger.record("train/critic_loss", np.mean(critic_losses))


    if DQN_AVAILABLE:
        class DQNWithPER(DQN):
            """DQN variant that applies PER weights and priority updates."""

            def train(self, gradient_steps: int, batch_size: int = 100) -> None:
                self.policy.set_training_mode(True)
                self._update_learning_rate(self.policy.optimizer)

                losses = []
                for _ in range(gradient_steps):
                    replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
                    weights, indices = _resolve_per_weights(replay_data, batch_size)
                    discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

                    with th.no_grad():
                        next_q_values = self.q_net_target(replay_data.next_observations)
                        next_q_values, _ = next_q_values.max(dim=1)
                        next_q_values = next_q_values.reshape(-1, 1)
                        target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

                    current_q_values = self.q_net(replay_data.observations)
                    current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

                    per_loss = F.smooth_l1_loss(current_q_values, target_q_values, reduction="none")
                    loss = (per_loss * weights).mean()
                    losses.append(loss.item())

                    td_errors = th.abs(target_q_values - current_q_values)
                    _update_priorities(self.replay_buffer, indices, td_errors)

                    self.policy.optimizer.zero_grad()
                    loss.backward()
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.policy.optimizer.step()

                self._n_updates += gradient_steps
                self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
                self.logger.record("train/loss", np.mean(losses))


    if QRDQN_AVAILABLE:
        class QRDQNWithPER(QRDQN):
            """QR-DQN variant that applies PER weights and priority updates."""

            def train(self, gradient_steps: int, batch_size: int = 100) -> None:
                if not hasattr(self, "quantile_net") or not hasattr(self, "quantile_net_target"):
                    return super().train(gradient_steps, batch_size=batch_size)

                self.policy.set_training_mode(True)
                self._update_learning_rate(self.policy.optimizer)

                losses = []
                for _ in range(gradient_steps):
                    replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
                    weights, indices = _resolve_per_weights(replay_data, batch_size)
                    discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

                    quantiles = self.quantile_net(replay_data.observations)
                    next_quantiles = self.quantile_net_target(replay_data.next_observations)

                    n_quantiles = getattr(self, "n_quantiles", None)
                    if n_quantiles is None:
                        if quantiles.ndim == 3:
                            n_quantiles = quantiles.shape[1]
                        else:
                            n_quantiles = quantiles.shape[-1]

                    if quantiles.ndim == 3:
                        if quantiles.shape[1] == n_quantiles:
                            action_idx = replay_data.actions.long().unsqueeze(1).expand(-1, n_quantiles, -1)
                            current_quantiles = quantiles.gather(2, action_idx).squeeze(2)
                        elif quantiles.shape[2] == n_quantiles:
                            action_idx = replay_data.actions.long().unsqueeze(-1).expand(-1, -1, n_quantiles)
                            current_quantiles = quantiles.gather(1, action_idx).squeeze(1)
                        else:
                            current_quantiles = quantiles.mean(dim=-1)
                    elif quantiles.ndim == 2:
                        current_quantiles = quantiles
                    else:
                        return super().train(gradient_steps, batch_size=batch_size)

                    if next_quantiles.ndim == 3:
                        if next_quantiles.shape[1] == n_quantiles:
                            next_q_values = next_quantiles.mean(dim=1)
                            next_actions = next_q_values.max(dim=1)[1]
                            next_quantiles = next_quantiles.gather(
                                2, next_actions.view(-1, 1, 1).expand(-1, n_quantiles, 1)
                            ).squeeze(2)
                        elif next_quantiles.shape[2] == n_quantiles:
                            next_q_values = next_quantiles.mean(dim=2)
                            next_actions = next_q_values.max(dim=1)[1]
                            next_quantiles = next_quantiles.gather(
                                1, next_actions.view(-1, 1, 1).expand(-1, 1, n_quantiles)
                            ).squeeze(1)
                    elif next_quantiles.ndim == 2:
                        next_quantiles = next_quantiles

                    target_quantiles = replay_data.rewards + (1 - replay_data.dones) * discounts * next_quantiles

                    current_quantiles = current_quantiles.unsqueeze(1)
                    target_quantiles = target_quantiles.unsqueeze(1)
                    per_sample_loss = _quantile_huber_loss_per_sample(current_quantiles, target_quantiles)
                    loss = (per_sample_loss * weights.squeeze(-1)).mean()
                    losses.append(loss.item())

                    td_errors = th.abs(target_quantiles - current_quantiles).mean(dim=(1, 2))
                    _update_priorities(self.replay_buffer, indices, td_errors)

                    self.policy.optimizer.zero_grad()
                    loss.backward()
                    max_grad_norm = getattr(self, "max_grad_norm", None)
                    if max_grad_norm is not None:
                        th.nn.utils.clip_grad_norm_(self.policy.parameters(), max_grad_norm)
                    self.policy.optimizer.step()

                self._n_updates += gradient_steps
                self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
                self.logger.record("train/loss", np.mean(losses))


if TQC_AVAILABLE and quantile_huber_loss is not None and polyak_update is not None:
    class TQCWithPER(TQC):
        """TQC variant that supports prioritized replay buffers."""

        def train(self, gradient_steps: int, batch_size: int = 64) -> None:
            # Switch to train mode (this affects batch norm / dropout)
            self.policy.set_training_mode(True)
            # Update optimizers learning rate
            optimizers = [self.actor.optimizer, self.critic.optimizer]
            if self.ent_coef_optimizer is not None:
                optimizers += [self.ent_coef_optimizer]

            # Update learning rate according to lr schedule
            self._update_learning_rate(optimizers)
            actor_lr = getattr(self, "_actor_lr", None)
            critic_lr = getattr(self, "_critic_lr", None)
            if actor_lr is not None:
                for param_group in self.actor.optimizer.param_groups:
                    param_group["lr"] = actor_lr
            if critic_lr is not None:
                for param_group in self.critic.optimizer.param_groups:
                    param_group["lr"] = critic_lr

            ent_coef_losses, ent_coefs = [], []
            actor_losses, critic_losses = [], []

            for gradient_step in range(gradient_steps):
                # Sample replay buffer
                replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
                weights = getattr(replay_data, "weights", None)
                indices = getattr(replay_data, "indices", None)
                if weights is None:
                    weights = th.ones((batch_size, 1), device=replay_data.observations.device)

                # For n-step replay, discount factor is gamma**n_steps (when no early termination)
                discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

                # We need to sample because `log_std` may have changed between two gradient steps
                if self.use_sde:
                    self.actor.reset_noise()

                # Action by the current actor for the sampled state
                actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
                log_prob = log_prob.reshape(-1, 1)

                ent_coef_loss = None
                if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                    # Important: detach the variable from the graph
                    ent_coef = th.exp(self.log_ent_coef.detach())
                    ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()  # type: ignore[operator]
                    ent_coef_losses.append(ent_coef_loss.item())
                else:
                    ent_coef = self.ent_coef_tensor

                ent_coefs.append(ent_coef.item())

                # Optimize entropy coefficient, also called entropy temperature or alpha in the paper
                if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                    self.ent_coef_optimizer.zero_grad()
                    ent_coef_loss.backward()
                    self.ent_coef_optimizer.step()

                with th.no_grad():
                    # Select action according to policy
                    next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                    # Compute and cut quantiles at the next state
                    # batch x nets x quantiles
                    next_quantiles = self.critic_target(replay_data.next_observations, next_actions)

                    # Sort and drop top k quantiles to control overestimation.
                    n_target_quantiles = (
                        self.critic.quantiles_total - self.top_quantiles_to_drop_per_net * self.critic.n_critics
                    )
                    next_quantiles, _ = th.sort(next_quantiles.reshape(batch_size, -1))
                    next_quantiles = next_quantiles[:, :n_target_quantiles]

                    # td error + entropy term
                    target_quantiles = next_quantiles - ent_coef * next_log_prob.reshape(-1, 1)
                    target_quantiles = replay_data.rewards + (1 - replay_data.dones) * discounts * target_quantiles
                    # Make target_quantiles broadcastable to (batch_size, n_critics, n_target_quantiles).
                    target_quantiles.unsqueeze_(dim=1)

                # Get current Quantile estimates using action from the replay buffer
                current_quantiles = self.critic(replay_data.observations, replay_data.actions)

                # Compute per-sample critic loss for PER weighting and priority updates
                per_sample_loss = _quantile_huber_loss_per_sample(current_quantiles, target_quantiles)
                critic_loss = (per_sample_loss * weights.squeeze(-1)).mean()
                critic_losses.append(critic_loss.item())

                # Optimize the critic
                self.critic.optimizer.zero_grad()
                critic_loss.backward()
                self.critic.optimizer.step()

                # Update priorities if buffer supports it
                if indices is not None and hasattr(self.replay_buffer, "update_priorities"):
                    td_errors = per_sample_loss.detach().cpu().numpy()
                    self.replay_buffer.update_priorities(indices, td_errors)

                # Compute actor loss
                qf_pi = self.critic(replay_data.observations, actions_pi).mean(dim=2).mean(dim=1, keepdim=True)
                actor_loss = (ent_coef * log_prob - qf_pi).mean()
                actor_losses.append(actor_loss.item())

                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                # Update target networks
                if gradient_step % self.target_update_interval == 0:
                    polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                    # Copy running stats, see https://github.com/DLR-RM/stable-baselines3/issues/996
                    polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

            self._n_updates += gradient_steps

            self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
            self.logger.record("train/ent_coef", np.mean(ent_coefs))
            self.logger.record("train/actor_loss", np.mean(actor_losses))
            self.logger.record("train/critic_loss", np.mean(critic_losses))
            if len(ent_coef_losses) > 0:
                self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))


class SB3SACAgent(SB3AgentBase):
    """SAC agent using Stable-Baselines3."""

    def __init__(self, cfg: Dict[str, Any]):
        # Set SAC-specific params BEFORE calling super().__init__
        params = cfg.get('params', {})
        self.ent_coef = params.get('ent_coef', 'auto')
        self.target_entropy = params.get('target_entropy', 'auto')
        per_enabled = _parse_per_config(params)['enabled']

        # Now initialize base class
        model_class = SACWithPER if per_enabled else SAC
        super().__init__(cfg, model_class)

        # Create model after all attributes are set
        dummy_env = DummyEnv(self.obs_dim, self.action_low, self.action_high)
        self._create_model(dummy_env)

    def _create_model(self, env):
        """Create SAC model."""
        replay_buffer_class, replay_buffer_kwargs = self._get_per_replay_buffer()
        self.model = self.model_class(
            policy='MlpPolicy',
            env=env,
            learning_rate=self.learning_rate,
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            tau=self.tau,
            gamma=self.gamma,
            learning_starts=self.learning_starts,
            ent_coef=self.ent_coef,
            target_entropy=self.target_entropy,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=self.policy_kwargs,
            device=self.device,
            verbose=0,
        )
        self._setup_logger()
        self._setup_done = True


class SB3TD3Agent(SB3AgentBase):
    """TD3 (Twin Delayed Deep Deterministic Policy Gradient) agent using SB3.

    TD3 is a deterministic policy gradient algorithm that addresses
    overestimation bias in DDPG through three key innovations:

    1. **Clipped Double Q-Learning**: Uses min of two critics to reduce overestimation
       - Maintains two independent Q-networks
       - Takes minimum for Bellman target: y = r + γ * min(Q1(s',a'), Q2(s',a'))

    2. **Delayed Policy Updates**: Updates actor less frequently than critics
       - Default: Update policy every 2 critic updates (policy_delay=2)
       - Reduces variance in policy gradients
       - Allows critics to converge before policy changes

    3. **Target Policy Smoothing**: Adds noise to target actions for regularization
       - Smooths Q-function by averaging over actions
       - Prevents exploitation of critic errors
       - Noise: a' = π_target(s') + clip(ε, -c, c), ε ~ N(0, σ²)

    Architecture:
    - Actor: Deterministic policy π(s) → a
    - Critics: Two Q-networks Q1(s,a), Q2(s,a)
    - Target networks for stable learning (actor_target, Q1_target, Q2_target)

    Key Differences from SAC:
    - TD3: Deterministic policy, uses exploration noise during training
    - SAC: Stochastic policy (Gaussian), no separate exploration noise
    - TD3: Generally more sample efficient but less robust
    """

    def __init__(self, cfg: Dict[str, Any]):
        """Initialize TD3 agent.

        Args:
            cfg: Configuration dict with:
                - params:
                    - policy_delay: Update policy every N critic updates (default: 2)
                    - target_policy_noise: Std dev of noise added to target policy (default: 0.2)
                    - target_noise_clip: Clipping range for target noise (default: 0.5)
                    - learning_rate, gamma, tau, etc. (inherited from base)
        """
        # Set TD3-specific params BEFORE calling super().__init__
        params = cfg.get('params', {})

        # Delayed policy updates: Update actor every N critic updates
        self.policy_delay = params.get('policy_delay', 2)

        # Target policy smoothing noise parameters
        self.target_policy_noise = params.get('target_policy_noise', 0.2)  # σ for ε ~ N(0, σ²)
        self.target_noise_clip = params.get('target_noise_clip', 0.5)  # Clip range [-c, c]
        self.action_noise_sigma = params.get('action_noise_sigma', 0.1)
        self.action_noise = None
        per_enabled = _parse_per_config(params)['enabled']

        # Initialize base class (sets up common parameters)
        model_class = TD3WithPER if per_enabled else TD3
        super().__init__(cfg, model_class)

        # Create TD3 model with dummy environment
        dummy_env = DummyEnv(self.obs_dim, self.action_low, self.action_high)
        self._create_model(dummy_env)

    def _create_model(self, env):
        """Create TD3 model with configured hyperparameters.

        TD3 Update Algorithm:
        1. Sample minibatch from replay buffer
        2. Compute target actions with noise: a' = π_target(s') + clip(ε, -c, c)
        3. Compute target Q-value: y = r + γ * min(Q1_target(s',a'), Q2_target(s',a'))
        4. Update both critics: minimize (Q_i(s,a) - y)²
        5. (Every policy_delay steps) Update actor: maximize Q1(s, π(s))
        6. (Every policy_delay steps) Update target networks via Polyak averaging

        Args:
            env: Dummy Gym environment for SB3 initialization
        """
        action_noise = None
        if self.action_noise_sigma and float(self.action_noise_sigma) > 0.0:
            from stable_baselines3.common.noise import NormalActionNoise
            n_actions = env.action_space.shape[0]
            action_noise = NormalActionNoise(
                mean=np.zeros(n_actions),
                sigma=float(self.action_noise_sigma) * np.ones(n_actions),
            )
        self.action_noise = action_noise

        replay_buffer_class, replay_buffer_kwargs = self._get_per_replay_buffer()
        self.model = self.model_class(
            policy='MlpPolicy',  # Deterministic policy network
            env=env,
            learning_rate=self.learning_rate,  # Adam LR for both actor and critics
            buffer_size=self.buffer_size,  # Replay buffer capacity
            batch_size=self.batch_size,  # SGD minibatch size
            tau=self.tau,  # Polyak averaging coefficient
            gamma=self.gamma,  # Discount factor
            learning_starts=self.learning_starts,  # Random exploration steps
            policy_delay=self.policy_delay,  # Actor update frequency (relative to critic)
            target_policy_noise=self.target_policy_noise,  # Target smoothing noise std
            target_noise_clip=self.target_noise_clip,  # Target noise clipping range
            action_noise=action_noise,  # Exploration noise (unused in predict path)
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=self.policy_kwargs,  # Network architecture
            device=self.device,  # 'cuda' or 'cpu'
            verbose=0,
        )
        self._setup_logger()
        self._setup_done = True

    def act(self, obs: np.ndarray, deterministic: bool = False, info: Optional[Dict] = None) -> np.ndarray:
        """Select action and apply exploration noise for TD3."""
        action = super().act(obs, deterministic=deterministic, info=info)
        if not deterministic and self.action_noise is not None:
            action = np.clip(action + self.action_noise(), self.action_low, self.action_high)
        return action


class SB3DDPGAgent(SB3AgentBase):
    """DDPG agent using Stable-Baselines3."""

    def __init__(self, cfg: Dict[str, Any]):
        # Set DDPG-specific params BEFORE calling super().__init__
        params = cfg.get('params', {})
        self.action_noise_sigma = params.get('action_noise_sigma', 0.1)
        per_enabled = _parse_per_config(params)['enabled']

        # Now initialize base class
        model_class = DDPGWithPER if per_enabled else DDPG
        super().__init__(cfg, model_class)

        # Create model after all attributes are set
        dummy_env = DummyEnv(self.obs_dim, self.action_low, self.action_high)
        self._create_model(dummy_env)

    def _create_model(self, env):
        """Create DDPG model."""
        # DDPG uses Ornstein-Uhlenbeck noise by default, but we'll use normal action noise
        from stable_baselines3.common.noise import NormalActionNoise

        n_actions = env.action_space.shape[0]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=self.action_noise_sigma * np.ones(n_actions)
        )

        replay_buffer_class, replay_buffer_kwargs = self._get_per_replay_buffer()
        self.model = self.model_class(
            policy='MlpPolicy',
            env=env,
            learning_rate=self.learning_rate,
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            tau=self.tau,
            gamma=self.gamma,
            learning_starts=self.learning_starts,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=self.policy_kwargs,
            device=self.device,
            verbose=0,
        )
        self._setup_logger()
        self._setup_done = True


class SB3PPOAgent(SB3AgentBase):
    """PPO (Proximal Policy Optimization) agent using Stable-Baselines3.

    PPO is an ON-POLICY policy gradient algorithm that uses clipped surrogate
    objective for stable, sample-efficient learning.

    Key Innovations:
    1. **Clipped Surrogate Objective**: Prevents large policy updates
       - Ratio: r(θ) = π_θ(a|s) / π_θ_old(a|s)
       - Loss: L^CLIP = min(r(θ)A, clip(r, 1-ε, 1+ε)A)
       - Conservative updates within [1-ε, 1+ε] (default ε=0.2)

    2. **Generalized Advantage Estimation (GAE)**: Reduces variance
       - A^GAE = Σ(γλ)^t δ_t where δ_t = r_t + γV(s_{t+1}) - V(s_t)
       - λ balances bias-variance tradeoff (default: 0.95)

    3. **Multiple Epochs**: Reuses rollout data for sample efficiency
       - Collect n_steps of experience (default: 2048)
       - Train for n_epochs on this data (default: 10)
       - Prevents catastrophic forgetting via clipping

    Architecture:
    - Shared backbone or separate networks for policy and value function
    - Policy: Gaussian distribution π(a|s) = N(μ_θ(s), σ_θ(s))
    - Value: State-value function V(s)

    ON-POLICY vs OFF-POLICY:
    - PPO: Requires fresh rollouts, cannot use replay buffer
    - SAC/TD3/TQC: Can reuse old transitions from replay buffer
    - PPO: Generally more stable, lower sample efficiency
    - OFF-POLICY: Higher sample efficiency, more prone to instability

    Training Flow (enhanced_training.py integration):
    - Collects transitions in rollout buffer during episode
    - Updates at episode end using full trajectory
    - Does NOT update every step like off-policy algorithms
    """

    def __init__(self, cfg: Dict[str, Any]):
        """Initialize PPO agent.

        Args:
            cfg: Configuration dict with:
                - params:
                    - n_steps: Rollout buffer size (default: 2048)
                    - n_epochs: Training epochs per rollout (default: 10)
                    - clip_range: PPO clipping epsilon (default: 0.2)
                    - gae_lambda: GAE lambda for advantage estimation (default: 0.95)
                    - ent_coef: Entropy bonus coefficient (default: 0.02)
                    - batch_size: Minibatch size for SGD (default: 64)
                    - learning_rate, gamma (inherited from base)
        """
        # Set PPO-specific params BEFORE calling super().__init__
        params = cfg.get('params', {})

        # Rollout collection
        self.n_steps = params.get('n_steps', 2048)  # Steps per rollout

        # Training configuration
        self.n_epochs = params.get('n_epochs', 10)  # Epochs per rollout
        self.clip_range = params.get('clip_range', 0.2)  # PPO clipping epsilon

        # Advantage estimation
        self.gae_lambda = params.get('gae_lambda', 0.95)  # GAE λ parameter

        # Exploration
        self.ent_coef = params.get('ent_coef', 0.02)  # Entropy bonus weight

        # Initialize base class (PPO doesn't use replay buffer)
        super().__init__(cfg, PPO)

        # Create PPO model
        dummy_env = DummyEnv(self.obs_dim, self.action_low, self.action_high)
        self._create_model(dummy_env)

    def _create_model(self, env):
        """Create PPO model with configured hyperparameters.

        PPO Update Algorithm (executed at episode end):
        1. Collect n_steps of experience in rollout buffer
        2. Compute advantages using GAE: A = Σ(γλ)^t δ_t
        3. Normalize advantages (zero mean, unit variance)
        4. For n_epochs:
           a. Sample minibatches from rollout buffer
           b. Compute policy loss: L^CLIP = -min(r·A, clip(r, 1-ε, 1+ε)·A)
           c. Compute value loss: L^VF = (V(s) - V_target)²
           d. Compute entropy bonus: H = -Σ π(a|s) log π(a|s)
           e. Total loss: L = L^CLIP + c1·L^VF - c2·H
           f. Update parameters via gradient descent
        5. Clear rollout buffer

        Args:
            env: Dummy Gym environment for SB3 initialization
        """
        self.model = PPO(
            policy='MlpPolicy',  # Gaussian policy with value head
            env=env,
            learning_rate=self.learning_rate,  # Adam LR (often higher than off-policy)
            n_steps=self.n_steps,  # Rollout buffer size (experience per update)
            batch_size=self.batch_size,  # Minibatch size for SGD
            n_epochs=self.n_epochs,  # Training epochs per rollout
            gamma=self.gamma,  # Discount factor
            gae_lambda=self.gae_lambda,  # GAE λ for advantage estimation
            clip_range=self.clip_range,  # PPO clipping epsilon
            ent_coef=self.ent_coef,  # Entropy bonus coefficient
            policy_kwargs=self.policy_kwargs,  # Network architecture
            device=self.device,  # 'cuda' or 'cpu'
            verbose=0,
        )
        self._setup_logger()
        self._setup_done = True

    def store_transition(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        info: Optional[Dict] = None
    ) -> None:
        """PPO doesn't use store_transition() - uses internal rollout buffer.

        Note: This method is a no-op for API compatibility with enhanced_training.py.
        PPO handles transition storage internally via SB3's RolloutBuffer.
        """
        # PPO uses a different buffering mechanism (RolloutBuffer)
        # Transitions are collected via env.step() during model.learn()
        # For compatibility with run_v2.py custom training loop, we pass here
        pass

    def update(self) -> Optional[Dict[str, float]]:
        """PPO update - requires full rollout collection.

        Note: This method returns None for compatibility with enhanced_training.py.
        PPO's update happens automatically when n_steps is reached in model.learn().

        For proper PPO training, use SB3's built-in model.learn() method or
        ensure rollout buffer is full before calling update.
        """
        # PPO requires collecting full rollouts before updating
        # The custom training loop in enhanced_training.py calls update() every episode
        # but PPO needs n_steps (e.g., 2048) before updating
        # Use run_sb3_baseline.py for proper PPO training with model.learn()
        return None


class SB3A2CAgent(SB3AgentBase):
    """A2C (Advantage Actor-Critic) agent using Stable-Baselines3.

    A2C is a synchronous, ON-POLICY actor-critic algorithm. It's essentially
    the synchronous version of A3C (Asynchronous Advantage Actor-Critic).

    Key Characteristics:
    1. **Actor-Critic Architecture**: Learns policy and value function jointly
       - Actor (policy): π(a|s) outputs action distribution
       - Critic (value): V(s) estimates state value

    2. **Advantage Function**: Uses advantage for policy gradient
       - A(s,a) = Q(s,a) - V(s) = r + γV(s') - V(s)
       - Reduces variance compared to raw returns

    3. **Synchronous Updates**: Updates after every n_steps (typically 5-10)
       - Faster than PPO (no multiple epochs)
       - Less sample efficient than PPO (no rollout reuse)

    4. **No Clipping**: Unlike PPO, uses vanilla policy gradient
       - Can have larger policy updates (less stable)
       - Requires careful learning rate tuning

    Architecture:
    - Shared backbone: Conv/MLP → [policy_head, value_head]
    - Policy: Gaussian distribution π(a|s) = N(μ_θ(s), σ_θ(s))
    - Value: State-value function V(s)

    A2C vs PPO:
    - A2C: Faster updates, fewer hyperparameters, less stable
    - PPO: More stable, sample efficient, but slower

    A2C vs A3C:
    - A2C: Synchronous (deterministic), easier to debug
    - A3C: Asynchronous parallel workers, faster wall-clock time

    Training Flow:
    - Collects n_steps transitions (default: 5)
    - Computes advantages using n-step returns
    - Single gradient update on policy and value function
    - Repeat (no replay buffer, no multiple epochs)
    """

    def __init__(self, cfg: Dict[str, Any]):
        """Initialize A2C agent.

        Args:
            cfg: Configuration dict with:
                - params:
                    - n_steps: Rollout length before update (default: 5)
                    - gae_lambda: GAE lambda (default: 1.0, i.e., n-step returns)
                    - ent_coef: Entropy bonus coefficient (default: 0.0)
                    - vf_coef: Value function loss coefficient (default: 0.5)
                    - learning_rate, gamma (inherited from base)
        """
        # Set A2C-specific params BEFORE calling super().__init__
        params = cfg.get('params', {})

        # Rollout configuration
        self.n_steps = params.get('n_steps', 5)  # A2C uses shorter rollouts than PPO

        # Advantage estimation
        self.gae_lambda = params.get('gae_lambda', 1.0)  # 1.0 = n-step returns (no GAE)

        # Loss coefficients
        self.ent_coef = params.get('ent_coef', 0.0)  # Entropy bonus (0 = no bonus)
        self.vf_coef = params.get('vf_coef', 0.5)  # Value function loss weight

        # Initialize base class (A2C doesn't use replay buffer)
        super().__init__(cfg, A2C)

        # Create A2C model
        dummy_env = DummyEnv(self.obs_dim, self.action_low, self.action_high)
        self._create_model(dummy_env)

    def _create_model(self, env):
        """Create A2C model with configured hyperparameters.

        A2C Update Algorithm (executed every n_steps):
        1. Collect n_steps of experience (s, a, r, s')
        2. Compute n-step returns or GAE advantages:
           - If λ=1.0: R_t = r_t + γr_{t+1} + ... + γ^n V(s_{t+n})
           - If λ<1.0: A^GAE = Σ(γλ)^k δ_{t+k}
        3. Compute policy loss: L^π = -E[log π(a|s) * A(s,a)]
        4. Compute value loss: L^V = E[(V(s) - R_t)²]
        5. Compute entropy bonus: H = E[-Σ π(a|s) log π(a|s)]
        6. Total loss: L = L^π + vf_coef * L^V - ent_coef * H
        7. Single gradient update
        8. Clear rollout buffer

        Args:
            env: Dummy Gym environment for SB3 initialization
        """
        self.model = A2C(
            policy='MlpPolicy',  # Shared network with policy + value heads
            env=env,
            learning_rate=self.learning_rate,  # Adam LR (typically higher than off-policy)
            n_steps=self.n_steps,  # Rollout length (shorter than PPO)
            gamma=self.gamma,  # Discount factor
            gae_lambda=self.gae_lambda,  # GAE λ (1.0 = no GAE)
            ent_coef=self.ent_coef,  # Entropy regularization weight
            vf_coef=self.vf_coef,  # Value loss coefficient
            policy_kwargs=self.policy_kwargs,  # Network architecture
            device=self.device,  # 'cuda' or 'cpu'
            verbose=0,
        )
        self._setup_logger()
        self._setup_done = True

    def store_transition(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        info: Optional[Dict] = None
    ) -> None:
        """A2C doesn't use store_transition() - uses internal rollout buffer.

        Note: This method is a no-op for API compatibility with enhanced_training.py.
        A2C handles transition storage internally via SB3's RolloutBuffer.
        """
        # A2C uses RolloutBuffer (like PPO)
        # Transitions collected automatically during model.learn()
        # For compatibility with custom training loop, we pass here
        pass

    def update(self) -> Optional[Dict[str, float]]:
        """A2C update - requires rollout collection.

        Note: This method returns None for compatibility with enhanced_training.py.
        A2C's update happens automatically when n_steps is reached in model.learn().

        For proper A2C training, use SB3's built-in model.learn() method.
        """
        # A2C requires collecting n_steps before updating
        # Custom training loop calls update() every episode, but A2C needs n_steps
        # Use run_sb3_baseline.py for proper A2C training with model.learn()
        return None


class SB3TQCAgent(SB3AgentBase):
    """TQC (Truncated Quantile Critics) agent using SB3-Contrib.

    TQC is a distributional RL algorithm that extends SAC by:
    1. Modeling full Q-value distributions (not just expected values)
    2. Using quantile regression with multiple critics (25 quantiles × 5 critics)
    3. Dropping top quantiles for conservative (risk-averse) value estimates

    Key advantages over SAC:
    - Better sample efficiency (learns faster)
    - More stable training (less prone to overestimation)
    - Risk-sensitive policy (conservative action selection)

    Architecture:
    - Actor: Gaussian policy network (outputs μ, σ for action distribution)
    - Critics: 5 quantile networks, each outputting 25 quantiles
    - Drops top 2 quantiles per network → uses 23×5=115 quantiles for Bellman update
    """

    def __init__(self, cfg: Dict[str, Any]):
        """Initialize TQC agent.

        Args:
            cfg: Configuration dict containing:
                - obs_dim: Observation dimension (e.g., 119 for gaplock)
                - params: Hyperparameters dict with:
                    - learning_rate: Adam LR (default: 3e-4)
                    - gamma: Discount factor (default: 0.995)
                    - tau: Polyak averaging for target networks (default: 0.005)
                    - ent_coef: Entropy regularization ('auto' for auto-tuning)
                    - top_quantiles_to_drop_per_net: Risk-aversion parameter (default: 2)
                    - hidden_dims: Network architecture (default: [256, 256])
                    - buffer_size: Replay buffer capacity (default: 1M)
                    - batch_size: Minibatch size for SGD (default: 256)
                    - learning_starts: Random exploration steps before training (default: 1000)
        """
        if not TQC_AVAILABLE:
            raise ImportError("sb3-contrib required for TQC. Install with: pip install sb3-contrib")

        # ========================================
        # TQC-SPECIFIC PARAMETERS
        # ========================================

        params = cfg.get('params', {})

        # Entropy regularization coefficient (encourages exploration)
        # 'auto': Automatically tuned to maintain target entropy
        self.ent_coef = params.get('ent_coef', 'auto')

        # Target entropy for automatic tuning (usually -dim(action_space))
        # 'auto': Sets target_entropy = -dim(A) ≈ -2 for F110
        self.target_entropy = params.get('target_entropy', 'auto')

        # Number of quantiles to drop per critic network (risk-aversion)
        # Higher = more conservative (drops more optimistic estimates)
        # Default: 2 → uses 23/25 quantiles per critic
        self.top_quantiles_to_drop_per_net = params.get('top_quantiles_to_drop_per_net', 2)

        # Optional separate learning rates for actor/critic
        self.actor_learning_rate = params.get('actor_learning_rate')
        self.critic_learning_rate = params.get('critic_learning_rate')

        # ========================================
        # INITIALIZATION
        # ========================================

        # Initialize base class (sets up common parameters, replay buffer, etc.)
        super().__init__(cfg, TQC)

        # Create TQC model with dummy environment (SB3 requirement)
        # Dummy env provides obs/action spaces but isn't used for actual training
        dummy_env = DummyEnv(self.obs_dim, self.action_low, self.action_high)
        self._create_model(dummy_env)

    def _create_model(self, env):
        """Create TQC model with configured hyperparameters.

        Model components:
        - Actor network (Gaussian policy): obs → (μ, log_σ) → action
        - 5 Critic networks (quantile regressors): (obs, action) → 25 quantiles each
        - Target networks for stable learning (soft-updated via Polyak averaging)
        - Replay buffer for off-policy learning
        - Entropy coefficient optimizer (if ent_coef='auto')

        Args:
            env: Dummy Gym environment providing obs/action spaces
        """
        model_class = TQC
        use_custom_model = self.per_enabled or self.actor_learning_rate is not None or self.critic_learning_rate is not None
        if use_custom_model and "TQCWithPER" in globals():
            model_class = TQCWithPER  # type: ignore[name-defined]
        elif use_custom_model:
            print("Warning: custom TQC features unavailable; using standard TQC.")

        self.model_class = model_class
        replay_buffer_class, replay_buffer_kwargs = self._get_per_replay_buffer()

        self.model = self.model_class(
            policy='MlpPolicy',  # Use MLP networks for actor and critics
            env=env,
            learning_rate=self.learning_rate,  # Adam optimizer learning rate
            buffer_size=self.buffer_size,  # Replay buffer capacity (transitions)
            batch_size=self.batch_size,  # SGD minibatch size
            tau=self.tau,  # Polyak averaging: θ_target ← (1-τ)θ_target + τθ
            gamma=self.gamma,  # Discount factor for future rewards
            learning_starts=self.learning_starts,  # Random exploration before training
            ent_coef=self.ent_coef,  # Entropy regularization coefficient
            target_entropy=self.target_entropy,  # Target entropy for auto-tuning
            top_quantiles_to_drop_per_net=self.top_quantiles_to_drop_per_net,  # Risk-aversion
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=self.policy_kwargs,  # Network architecture {'net_arch': [256, 256]}
            device=self.device,  # 'cuda' or 'cpu'
            verbose=0,  # Suppress SB3 logging (we handle logging separately)
        )
        if self.model is not None:
            actor_lr = self.actor_learning_rate or self.learning_rate
            critic_lr = self.critic_learning_rate or self.learning_rate
            setattr(self.model, "_actor_lr", float(actor_lr))
            setattr(self.model, "_critic_lr", float(critic_lr))
        self._setup_logger()
        self._setup_done = True


class SB3DQNAgent(SB3AgentBase):
    """DQN agent using Stable-Baselines3 with discretized actions."""

    def __init__(self, cfg: Dict[str, Any]):
        if not DQN_AVAILABLE:
            raise ImportError("DQN not available. Make sure stable-baselines3 is installed.")

        # Parse action set for discretization (can be in cfg or params)
        params = cfg.get('params', {})
        action_set = cfg.get('action_set') or params.get('action_set')
        if action_set is None:
            raise ValueError("SB3DQNAgent requires 'action_set' parameter for action discretization")

        self.action_set = np.asarray(action_set, dtype=np.float32)
        if self.action_set.ndim != 2:
            raise ValueError("action_set must be 2D array of shape (n_actions, act_dim)")

        self.n_discrete_actions = self.action_set.shape[0]

        # Set DQN-specific params BEFORE calling super().__init__
        params = cfg.get('params', {})
        self.exploration_fraction = params.get('exploration_fraction', 0.1)
        self.exploration_final_eps = params.get('exploration_final_eps', 0.05)
        self.exploration_initial_eps = params.get('exploration_initial_eps', 1.0)
        per_enabled = _parse_per_config(params)['enabled']

        # Now initialize base class
        model_class = DQNWithPER if per_enabled and DQN_AVAILABLE else DQN
        super().__init__(cfg, model_class)

        # Create model after all attributes are set
        dummy_env = DummyDiscreteEnv(self.obs_dim, self.n_discrete_actions)
        self._create_model(dummy_env)

    def _create_model(self, env):
        """Create DQN model."""
        replay_buffer_class, replay_buffer_kwargs = self._get_per_replay_buffer()
        self.model = self.model_class(
            policy='MlpPolicy',
            env=env,
            learning_rate=self.learning_rate,
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            tau=self.tau,
            gamma=self.gamma,
            learning_starts=self.learning_starts,
            exploration_fraction=self.exploration_fraction,
            exploration_final_eps=self.exploration_final_eps,
            exploration_initial_eps=self.exploration_initial_eps,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=self.policy_kwargs,
            device=self.device,
            verbose=0,
        )
        self._setup_logger()
        self._setup_done = True

    def act(self, obs: np.ndarray, deterministic: bool = False, info: Optional[Dict] = None) -> np.ndarray:
        """Select action - returns continuous action from discrete index."""
        if not self._setup_done:
            raise RuntimeError("Model not initialized.")

        # Get discrete action index from DQN
        action_idx, _ = self.model.predict(obs, deterministic=deterministic)

        # Store action index for later use in store_transition
        self._last_action_idx = int(action_idx)

        # Convert to continuous action using action set
        continuous_action = self.action_set[self._last_action_idx]

        # Store action index in info for tracking
        if info is not None:
            info['action_index'] = self._last_action_idx

        return continuous_action

    def store_transition(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        info: Optional[Dict] = None
    ) -> None:
        """Store transition using action index instead of continuous action."""
        if self.model is None or not hasattr(self.model, 'replay_buffer'):
            return

        # Use the stored action index instead of the continuous action
        action_idx = np.array([self._last_action_idx], dtype=np.int64)

        # SB3 replay buffer expects specific format
        infos = [info] if info is not None else [{}]
        self.model.replay_buffer.add(
            obs, next_obs, action_idx, reward, done, infos
        )
        self._steps += 1

    def _get_replay_action(self, action: np.ndarray) -> np.ndarray:
        if hasattr(self, '_last_action_idx'):
            return np.array([self._last_action_idx], dtype=np.int64)
        return np.array([0], dtype=np.int64)


class SB3QRDQNAgent(SB3AgentBase):
    """QR-DQN agent using SB3-Contrib with discretized actions."""

    def __init__(self, cfg: Dict[str, Any]):
        if not QRDQN_AVAILABLE:
            raise ImportError("sb3-contrib required for QR-DQN. Install with: pip install sb3-contrib")

        # Parse action set for discretization (can be in cfg or params)
        params = cfg.get('params', {})
        action_set = cfg.get('action_set') or params.get('action_set')
        if action_set is None:
            raise ValueError("SB3QRDQNAgent requires 'action_set' parameter for action discretization")

        self.action_set = np.asarray(action_set, dtype=np.float32)
        if self.action_set.ndim != 2:
            raise ValueError("action_set must be 2D array of shape (n_actions, act_dim)")

        self.n_discrete_actions = self.action_set.shape[0]

        # Set QR-DQN-specific params BEFORE calling super().__init__
        params = cfg.get('params', {})
        self.exploration_fraction = params.get('exploration_fraction', 0.1)
        self.exploration_final_eps = params.get('exploration_final_eps', 0.05)
        self.exploration_initial_eps = params.get('exploration_initial_eps', 1.0)
        per_enabled = _parse_per_config(params)['enabled']

        # Now initialize base class
        model_class = QRDQNWithPER if per_enabled and QRDQN_AVAILABLE else QRDQN
        super().__init__(cfg, model_class)

        # Create model after all attributes are set
        dummy_env = DummyDiscreteEnv(self.obs_dim, self.n_discrete_actions)
        self._create_model(dummy_env)

    def _create_model(self, env):
        """Create QR-DQN model."""
        replay_buffer_class, replay_buffer_kwargs = self._get_per_replay_buffer()
        self.model = self.model_class(
            policy='MlpPolicy',
            env=env,
            learning_rate=self.learning_rate,
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            tau=self.tau,
            gamma=self.gamma,
            learning_starts=self.learning_starts,
            exploration_fraction=self.exploration_fraction,
            exploration_final_eps=self.exploration_final_eps,
            exploration_initial_eps=self.exploration_initial_eps,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=self.policy_kwargs,
            device=self.device,
            verbose=0,
        )
        self._setup_logger()
        self._setup_done = True

    def act(self, obs: np.ndarray, deterministic: bool = False, info: Optional[Dict] = None) -> np.ndarray:
        """Select action - returns continuous action from discrete index."""
        if not self._setup_done:
            raise RuntimeError("Model not initialized.")

        # Get discrete action index from QR-DQN
        action_idx, _ = self.model.predict(obs, deterministic=deterministic)

        # Store action index for later use in store_transition
        self._last_action_idx = int(action_idx)

        # Convert to continuous action using action set
        continuous_action = self.action_set[self._last_action_idx]

        # Store action index in info for tracking
        if info is not None:
            info['action_index'] = self._last_action_idx

        return continuous_action

    def store_transition(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        info: Optional[Dict] = None
    ) -> None:
        """Store transition using action index instead of continuous action."""
        if self.model is None or not hasattr(self.model, 'replay_buffer'):
            return

        # Use the stored action index instead of the continuous action
        action_idx = np.array([self._last_action_idx], dtype=np.int64)

        # SB3 replay buffer expects specific format
        infos = [info] if info is not None else [{}]
        self.model.replay_buffer.add(
            obs, next_obs, action_idx, reward, done, infos
        )
        self._steps += 1

    def _get_replay_action(self, action: np.ndarray) -> np.ndarray:
        if hasattr(self, '_last_action_idx'):
            return np.array([self._last_action_idx], dtype=np.int64)
        return np.array([0], dtype=np.int64)


__all__ = [
    'SB3SACAgent',
    'SB3TD3Agent',
    'SB3DDPGAgent',
    'SB3PPOAgent',
    'SB3A2CAgent',
    'SB3TQCAgent',
    'SB3DQNAgent',
    'SB3QRDQNAgent',
]
