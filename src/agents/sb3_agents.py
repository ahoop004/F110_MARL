"""Stable-Baselines3 agent wrappers for F110 training system.

Wraps SB3 algorithms (SAC, TD3, PPO) to work with the existing
agent interface and training loop.
"""

from typing import Any, Dict, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    from stable_baselines3 import SAC, TD3, PPO, DDPG, A2C
    from stable_baselines3.common.buffers import ReplayBuffer
    from stable_baselines3.common.logger import configure
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("Warning: stable-baselines3 not installed. Run: pip install stable-baselines3")

# TQC and QR-DQN are in sb3-contrib
try:
    from sb3_contrib import TQC, QRDQN
    TQC_AVAILABLE = True
    QRDQN_AVAILABLE = True
except ImportError:
    TQC_AVAILABLE = False
    QRDQN_AVAILABLE = False
    # Don't warn - contrib algorithms are optional

# Try to import DQN (for discrete actions)
try:
    from stable_baselines3 import DQN
    DQN_AVAILABLE = True
except ImportError:
    DQN_AVAILABLE = False


class DummyEnv(gym.Env):
    """Dummy environment for SB3 initialization.

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


class SB3AgentBase:
    """Base wrapper for SB3 agents to match F110 agent interface."""

    def __init__(self, cfg: Dict[str, Any], model_class):
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 required. Install with: pip install stable-baselines3")

        self.cfg = cfg
        self.model_class = model_class
        self.model = None
        self._setup_done = False
        self._steps = 0

        # Extract config
        self.obs_dim = cfg['obs_dim']
        self.act_dim = cfg.get('act_dim', 2)
        self.device = cfg.get('device', 'cuda')

        # Action bounds
        self.action_low = np.array(cfg.get('action_low', [-0.46, -1.0]))
        self.action_high = np.array(cfg.get('action_high', [0.46, 1.0]))

        # SB3 hyperparameters
        params = cfg.get('params', {})
        self.learning_rate = params.get('learning_rate', 3e-4)
        self.buffer_size = params.get('buffer_size', 1_000_000)
        self.batch_size = params.get('batch_size', 256)
        self.tau = params.get('tau', 0.005)
        self.gamma = params.get('gamma', 0.995)
        self.learning_starts = params.get('learning_starts', 1000)

        # Network architecture
        hidden_dims = params.get('hidden_dims', [256, 256])
        self.policy_kwargs = {'net_arch': hidden_dims}

        # Don't create model yet - subclasses will do it after setting their specific params

    def _create_model(self, env):
        """Create SB3 model (called on first act())."""
        raise NotImplementedError

    def _setup_logger(self):
        """Set up a basic logger for the model."""
        if self.model is not None:
            # Create a minimal logger (no output to avoid clutter)
            logger = configure(None, [])
            self.model.set_logger(logger)

    def act(self, obs: np.ndarray, deterministic: bool = False, info: Optional[Dict] = None) -> np.ndarray:
        """Select action."""
        if not self._setup_done:
            raise RuntimeError("Model not initialized. Check __init__ implementation.")

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
        """Store transition in replay buffer."""
        if self.model is None or not hasattr(self.model, 'replay_buffer'):
            return

        # SB3 replay buffer expects specific format
        # add(obs, next_obs, action, reward, done, infos)
        infos = [info] if info is not None else [{}]
        self.model.replay_buffer.add(
            obs, next_obs, action, reward, done, infos
        )
        self._steps += 1

    def update(self) -> Optional[Dict[str, float]]:
        """Update agent by sampling from replay buffer and training."""
        if self.model is None or not hasattr(self.model, 'replay_buffer'):
            return None

        # Only update if we have enough samples
        if self._steps < self.learning_starts:
            return None

        # Check if replay buffer has enough samples
        if self.model.replay_buffer.size() < self.batch_size:
            return None

        try:
            # Train for one gradient step
            self.model.train(gradient_steps=1, batch_size=self.batch_size)
        except Exception as e:
            print(f"Warning: SB3 training step failed: {e}")
            return None

        # Return dummy metrics (SB3 doesn't expose per-step metrics easily)
        return {'steps': self._steps}

    def save(self, path: str) -> None:
        """Save model."""
        if self.model is not None:
            self.model.save(path)

    def load(self, path: str) -> None:
        """Load model."""
        self.model = self.model_class.load(path)
        self._setup_done = True


class SB3SACAgent(SB3AgentBase):
    """SAC agent using Stable-Baselines3."""

    def __init__(self, cfg: Dict[str, Any]):
        # Set SAC-specific params BEFORE calling super().__init__
        params = cfg.get('params', {})
        self.ent_coef = params.get('ent_coef', 'auto')
        self.target_entropy = params.get('target_entropy', 'auto')

        # Now initialize base class
        super().__init__(cfg, SAC)

        # Create model after all attributes are set
        dummy_env = DummyEnv(self.obs_dim, self.action_low, self.action_high)
        self._create_model(dummy_env)

    def _create_model(self, env):
        """Create SAC model."""
        self.model = SAC(
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
            policy_kwargs=self.policy_kwargs,
            device=self.device,
            verbose=0,
        )
        self._setup_logger()
        self._setup_done = True


class SB3TD3Agent(SB3AgentBase):
    """TD3 agent using Stable-Baselines3."""

    def __init__(self, cfg: Dict[str, Any]):
        # Set TD3-specific params BEFORE calling super().__init__
        params = cfg.get('params', {})
        self.policy_delay = params.get('policy_delay', 2)
        self.target_policy_noise = params.get('target_policy_noise', 0.2)
        self.target_noise_clip = params.get('target_noise_clip', 0.5)

        # Now initialize base class
        super().__init__(cfg, TD3)

        # Create model after all attributes are set
        dummy_env = DummyEnv(self.obs_dim, self.action_low, self.action_high)
        self._create_model(dummy_env)

    def _create_model(self, env):
        """Create TD3 model."""
        self.model = TD3(
            policy='MlpPolicy',
            env=env,
            learning_rate=self.learning_rate,
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            tau=self.tau,
            gamma=self.gamma,
            learning_starts=self.learning_starts,
            policy_delay=self.policy_delay,
            target_policy_noise=self.target_policy_noise,
            target_noise_clip=self.target_noise_clip,
            policy_kwargs=self.policy_kwargs,
            device=self.device,
            verbose=0,
        )
        self._setup_logger()
        self._setup_done = True


class SB3DDPGAgent(SB3AgentBase):
    """DDPG agent using Stable-Baselines3."""

    def __init__(self, cfg: Dict[str, Any]):
        # Set DDPG-specific params BEFORE calling super().__init__
        params = cfg.get('params', {})
        self.action_noise_sigma = params.get('action_noise_sigma', 0.1)

        # Now initialize base class
        super().__init__(cfg, DDPG)

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

        self.model = DDPG(
            policy='MlpPolicy',
            env=env,
            learning_rate=self.learning_rate,
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            tau=self.tau,
            gamma=self.gamma,
            learning_starts=self.learning_starts,
            action_noise=action_noise,
            policy_kwargs=self.policy_kwargs,
            device=self.device,
            verbose=0,
        )
        self._setup_logger()
        self._setup_done = True


class SB3PPOAgent(SB3AgentBase):
    """PPO agent using Stable-Baselines3."""

    def __init__(self, cfg: Dict[str, Any]):
        # Set PPO-specific params BEFORE calling super().__init__
        params = cfg.get('params', {})
        self.n_steps = params.get('n_steps', 2048)
        self.n_epochs = params.get('n_epochs', 10)
        self.clip_range = params.get('clip_range', 0.2)
        self.gae_lambda = params.get('gae_lambda', 0.95)
        self.ent_coef = params.get('ent_coef', 0.02)

        # Now initialize base class (PPO doesn't use replay buffer)
        super().__init__(cfg, PPO)

        # Create model after all attributes are set
        dummy_env = DummyEnv(self.obs_dim, self.action_low, self.action_high)
        self._create_model(dummy_env)

    def _create_model(self, env):
        """Create PPO model."""
        self.model = PPO(
            policy='MlpPolicy',
            env=env,
            learning_rate=self.learning_rate,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_range=self.clip_range,
            ent_coef=self.ent_coef,
            policy_kwargs=self.policy_kwargs,
            device=self.device,
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
        """PPO doesn't use replay buffer - transitions stored in rollout buffer."""
        # PPO uses a different buffering mechanism
        # For compatibility with run_v2.py, we'll just pass
        # The proper way to use PPO is via model.learn() which handles rollouts
        pass

    def update(self) -> Optional[Dict[str, float]]:
        """PPO update - not compatible with step-by-step training."""
        # PPO requires collecting full rollouts before updating
        # Use run_sb3_baseline.py for proper PPO training
        return None


class SB3A2CAgent(SB3AgentBase):
    """A2C agent using Stable-Baselines3."""

    def __init__(self, cfg: Dict[str, Any]):
        # Set A2C-specific params BEFORE calling super().__init__
        params = cfg.get('params', {})
        self.n_steps = params.get('n_steps', 5)  # A2C typically uses fewer steps than PPO
        self.gae_lambda = params.get('gae_lambda', 1.0)
        self.ent_coef = params.get('ent_coef', 0.0)
        self.vf_coef = params.get('vf_coef', 0.5)

        # Now initialize base class (A2C doesn't use replay buffer)
        super().__init__(cfg, A2C)

        # Create model after all attributes are set
        dummy_env = DummyEnv(self.obs_dim, self.action_low, self.action_high)
        self._create_model(dummy_env)

    def _create_model(self, env):
        """Create A2C model."""
        self.model = A2C(
            policy='MlpPolicy',
            env=env,
            learning_rate=self.learning_rate,
            n_steps=self.n_steps,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            ent_coef=self.ent_coef,
            vf_coef=self.vf_coef,
            policy_kwargs=self.policy_kwargs,
            device=self.device,
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
        """A2C doesn't use replay buffer - transitions stored in rollout buffer."""
        # A2C uses a rollout buffer mechanism
        # For compatibility with run_v2.py, we'll just pass
        pass

    def update(self) -> Optional[Dict[str, float]]:
        """A2C update - not compatible with step-by-step training."""
        # A2C requires collecting rollouts before updating
        # Use run_sb3_baseline.py for proper A2C training
        return None


class SB3TQCAgent(SB3AgentBase):
    """TQC agent using SB3-Contrib (distributional SAC variant)."""

    def __init__(self, cfg: Dict[str, Any]):
        if not TQC_AVAILABLE:
            raise ImportError("sb3-contrib required for TQC. Install with: pip install sb3-contrib")

        # Set TQC-specific params BEFORE calling super().__init__
        params = cfg.get('params', {})
        self.ent_coef = params.get('ent_coef', 'auto')
        self.target_entropy = params.get('target_entropy', 'auto')
        self.top_quantiles_to_drop_per_net = params.get('top_quantiles_to_drop_per_net', 2)

        # Now initialize base class
        super().__init__(cfg, TQC)

        # Create model after all attributes are set
        dummy_env = DummyEnv(self.obs_dim, self.action_low, self.action_high)
        self._create_model(dummy_env)

    def _create_model(self, env):
        """Create TQC model."""
        self.model = TQC(
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
            top_quantiles_to_drop_per_net=self.top_quantiles_to_drop_per_net,
            policy_kwargs=self.policy_kwargs,
            device=self.device,
            verbose=0,
        )
        self._setup_logger()
        self._setup_done = True


__all__ = [
    'SB3SACAgent',
    'SB3TD3Agent',
    'SB3DDPGAgent',
    'SB3PPOAgent',
    'SB3A2CAgent',
    'SB3TQCAgent',
]
