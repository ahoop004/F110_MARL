"""Wavelet-based Episodic Memory Agent for continuous control."""

from typing import Dict, Optional, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from agents.buffers.chronological import ChronologicalBuffer
from agents.buffers.episodic import EpisodicBuffer2D, EpisodicChunk
from agents.episodic.wavelet import build_wavelet
from agents.episodic.encoder import build_chunk_encoder
from agents.episodic.heads import build_heads


class WaveletEpisodicAgent:
    """RL agent using wavelet-transformed episodic memory for action selection.

    Architecture:
        - ChronologicalBuffer stores transitions in temporal order
        - Sliding window (stride=12, size=25) extracts chunks
        - Chunks reshaped to 5×5 grid with 5 channels (obs, action, reward, next_obs, done)
        - Wavelet transform preprocesses chunks
        - EpisodicBuffer2D stores transformed chunks
        - ChunkEncoder (Conv2D + RNN) encodes chunks to latents
        - Multi-task heads: Policy, Value, Reconstruction, Forward model
        - Training: weighted combination of multi-task losses

    Args:
        cfg: Configuration dictionary
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

        # Dimensions
        self.obs_dim = cfg['obs_dim']
        self.act_dim = cfg.get('act_dim', 2)  # steering + velocity
        self.chunk_size = cfg.get('chunk_size', 25)
        self.chunk_stride = cfg.get('chunk_stride', 12)
        self.grid_shape = tuple(cfg.get('grid_shape', [5, 5]))
        self.n_channels = cfg.get('n_channels', 5)

        # Device
        self.device = torch.device(cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

        # Action bounds
        self.action_low = np.array(cfg.get('action_low', [-0.4, 0.0]))
        self.action_high = np.array(cfg.get('action_high', [0.4, 8.0]))

        # Buffers
        chrono_config = cfg.get('chronological_buffer', {})
        self.chronological_buffer = ChronologicalBuffer(
            max_capacity=chrono_config.get('max_capacity', 10000),
            obs_shape=(self.obs_dim,),
            act_shape=(self.act_dim,),
            store_actions=True,
            store_action_indices=False,
        )

        episodic_config = cfg.get('episodic_buffer', {})
        self.episodic_buffer = EpisodicBuffer2D(
            capacity=episodic_config.get('capacity', 1000),
            chunk_size=self.chunk_size,
            grid_shape=self.grid_shape,
            n_channels=self.n_channels,
            selection_mode=episodic_config.get('selection_mode', 'uniform'),
            alpha=episodic_config.get('alpha', 0.6),
            beta=episodic_config.get('beta', 0.4),
            beta_increment=episodic_config.get('beta_increment', 0.001),
        )

        # Wavelet transform
        wavelet_config = cfg.get('wavelet', {})
        self.wavelet = build_wavelet(wavelet_config)

        # Compute wavelet feature dimension (depends on transform type)
        # For simplicity, assume same size as input after transform
        # In practice, this may vary; adjust as needed
        self.wavelet_feature_dim = self.obs_dim  # Placeholder

        # Networks
        encoder_config = cfg.get('encoder', {})
        encoder_config['input_feature_dim'] = self.wavelet_feature_dim
        encoder_config['latent_dim'] = cfg.get('latent_dim', 128)
        self.encoder = build_chunk_encoder(encoder_config).to(self.device)

        self.latent_dim = encoder_config['latent_dim']
        chunk_shape = (*self.grid_shape, self.wavelet_feature_dim)

        heads_config = cfg.get('heads', {})
        self.heads = build_heads(
            latent_dim=self.latent_dim,
            act_dim=self.act_dim,
            chunk_shape=chunk_shape,
            config=heads_config,
        )
        for head in self.heads.values():
            head.to(self.device)

        # Optimizers
        lr = cfg.get('learning_rate', 3e-4)
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.policy_opt = torch.optim.Adam(self.heads['policy'].parameters(), lr=lr)
        self.value_opt = torch.optim.Adam(self.heads['value'].parameters(), lr=lr)
        self.reconstruction_opt = torch.optim.Adam(self.heads['reconstruction'].parameters(), lr=lr)
        self.forward_opt = torch.optim.Adam(self.heads['forward'].parameters(), lr=lr)

        # Loss weights
        loss_weights = cfg.get('loss_weights', {})
        self.policy_loss_weight = loss_weights.get('policy', 1.0)
        self.value_loss_weight = loss_weights.get('value', 0.5)
        self.reconstruction_loss_weight = loss_weights.get('reconstruction', 0.1)
        self.forward_loss_weight = loss_weights.get('forward', 0.1)

        # Training parameters
        self.batch_size = cfg.get('batch_size', 32)
        self.warmup_chunks = cfg.get('warmup_chunks', 100)
        self.update_freq = cfg.get('update_freq', 4)
        self.max_grad_norm = cfg.get('max_grad_norm', 1.0)
        self.gamma = cfg.get('gamma', 0.99)

        # State
        self.step_count = 0
        self.update_count = 0
        self.last_chunk_creation_step = 0

        print(f"Initialized WaveletEpisodicAgent:")
        print(f"  Chunk size: {self.chunk_size} ({self.grid_shape} grid)")
        print(f"  Wavelet: {self.wavelet}")
        print(f"  Encoder: {self.encoder}")
        print(f"  Device: {self.device}")

    def act(self, obs: np.ndarray, deterministic: bool = False, info: Optional[Dict] = None) -> np.ndarray:
        """Select action for current observation.

        Strategy:
            - Use model immediately (no random warmup)
            - If buffer < chunk_size: pad with current obs to create partial chunk
            - Override velocity with locked speed if curriculum is active

        Args:
            obs: Current observation
            deterministic: Whether to act deterministically (for evaluation)
            info: Optional info dict containing locked_velocity and lock_speed_active

        Returns:
            action: (act_dim,) action vector [steering, velocity]
        """
        # Extract recent chunk or create partial chunk
        buffer_size = len(self.chronological_buffer)

        if buffer_size == 0:
            # First step: create chunk from repeated current observation
            chunk_data = {
                'observations': np.tile(obs, (self.chunk_size, 1)),
                'actions': np.zeros((self.chunk_size, self.act_dim)),
                'rewards': np.zeros(self.chunk_size),
                'next_observations': np.tile(obs, (self.chunk_size, 1)),
                'dones': np.zeros(self.chunk_size, dtype=bool),
            }
        elif buffer_size < self.chunk_size:
            # Partial chunk: get what we have and pad with current obs
            try:
                chunk_data = self.chronological_buffer.get_recent_window(buffer_size)
                # Pad to chunk_size by repeating last observation
                padding_needed = self.chunk_size - buffer_size
                chunk_data['observations'] = np.vstack([
                    chunk_data['observations'],
                    np.tile(obs, (padding_needed, 1))
                ])
                chunk_data['actions'] = np.vstack([
                    chunk_data['actions'],
                    np.zeros((padding_needed, self.act_dim))
                ])
                chunk_data['rewards'] = np.concatenate([
                    chunk_data['rewards'],
                    np.zeros(padding_needed)
                ])
                chunk_data['next_observations'] = np.vstack([
                    chunk_data['next_observations'],
                    np.tile(obs, (padding_needed, 1))
                ])
                chunk_data['dones'] = np.concatenate([
                    chunk_data['dones'],
                    np.zeros(padding_needed, dtype=bool)
                ])
            except ValueError:
                # Fallback: repeat current obs
                chunk_data = {
                    'observations': np.tile(obs, (self.chunk_size, 1)),
                    'actions': np.zeros((self.chunk_size, self.act_dim)),
                    'rewards': np.zeros(self.chunk_size),
                    'next_observations': np.tile(obs, (self.chunk_size, 1)),
                    'dones': np.zeros(self.chunk_size, dtype=bool),
                }
        else:
            # Full chunk available
            chunk_data = self.chronological_buffer.get_recent_window(self.chunk_size)

        # Create and transform chunk
        chunk_grid = self._create_chunk_grid(chunk_data)
        wavelet_chunk = self._apply_wavelet(chunk_grid)

        # Convert to tensor
        wavelet_tensor = torch.as_tensor(wavelet_chunk, dtype=torch.float32, device=self.device).unsqueeze(0)

        # Encode and predict action
        with torch.no_grad():
            latent = self.encoder(wavelet_tensor)  # (1, latent_dim)
            action_tensor = self.heads['policy'](latent)  # (1, act_dim)

        action = action_tensor.squeeze(0).cpu().numpy()

        # Scale action from [-1, 1] (tanh output) to [action_low, action_high]
        action_scaled = self.action_low + (action + 1.0) * 0.5 * (self.action_high - self.action_low)

        # Override velocity with locked speed if curriculum is active
        if info is not None and info.get('lock_speed_active', False):
            locked_velocity = info.get('locked_velocity')
            if locked_velocity is not None:
                # Keep steering from model, override velocity (index 1)
                action_scaled[1] = locked_velocity

        return action_scaled

    def store_transition(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        info: Optional[Dict] = None,
    ) -> None:
        """Store transition in chronological buffer and create chunks.

        Args:
            obs: Observation
            action: Action taken
            reward: Reward received
            next_obs: Next observation
            done: Whether episode terminated
            info: Optional metadata
        """
        # Add to chronological buffer
        self.chronological_buffer.add(
            obs=obs,
            action=action,
            reward=reward,
            next_obs=next_obs,
            done=done,
            info=info,
        )

        self.step_count += 1

        # Create chunk every chunk_stride steps (50% overlap with stride=12, size=25)
        if (
            len(self.chronological_buffer) >= self.chunk_size
            and self.step_count - self.last_chunk_creation_step >= self.chunk_stride
        ):
            self._create_and_store_chunk()
            self.last_chunk_creation_step = self.step_count

    def _create_chunk_grid(self, chunk_data: Dict[str, np.ndarray]) -> np.ndarray:
        """Reshape chunk into 5×5 grid.

        Args:
            chunk_data: Dictionary from ChronologicalBuffer.get_window()

        Returns:
            grid: (5, 5, feature_dim) grid where features are stacked tuple elements
        """
        # For now, we'll use observations as the primary feature
        # TODO: Incorporate actions, rewards, next_obs, dones as additional channels
        obs = chunk_data['observations']  # (25, obs_dim)

        # Reshape to 5×5 grid
        grid = obs.reshape(*self.grid_shape, -1)  # (5, 5, obs_dim)

        return grid

    def _apply_wavelet(self, chunk_grid: np.ndarray) -> np.ndarray:
        """Apply wavelet transform to chunk grid.

        Args:
            chunk_grid: (H, W, D) grid

        Returns:
            wavelet_grid: (H, W, D') transformed grid
        """
        H, W, D = chunk_grid.shape

        # Flatten to (H*W, D) for wavelet transform
        flat_chunk = chunk_grid.reshape(H * W, D)

        # Apply wavelet transform
        wavelet_coeffs = self.wavelet.transform(flat_chunk)  # (N_coeffs, D)

        # Reshape back to grid (may need padding/truncation)
        # For simplicity, pad or truncate to match original grid size
        target_size = H * W
        if wavelet_coeffs.shape[0] < target_size:
            # Pad with zeros
            pad_size = target_size - wavelet_coeffs.shape[0]
            wavelet_coeffs = np.vstack([wavelet_coeffs, np.zeros((pad_size, D))])
        elif wavelet_coeffs.shape[0] > target_size:
            # Truncate
            wavelet_coeffs = wavelet_coeffs[:target_size]

        # Reshape to grid
        wavelet_grid = wavelet_coeffs.reshape(H, W, D)

        return wavelet_grid

    def _create_and_store_chunk(self) -> None:
        """Extract chunk from chronological buffer and store in episodic buffer."""
        # Get last chunk_size transitions
        chunk_data = self.chronological_buffer.get_recent_window(self.chunk_size)

        # Create 5×5 grid
        chunk_grid = self._create_chunk_grid(chunk_data)

        # Apply wavelet transform
        wavelet_grid = self._apply_wavelet(chunk_grid)

        # Create episodic chunk
        episodic_chunk = EpisodicChunk(
            chunk_id=len(self.episodic_buffer),
            wavelet_coefficients=wavelet_grid,
            raw_observations=chunk_data['observations'],
            raw_actions=chunk_data['actions'],
            raw_rewards=chunk_data['rewards'],
            raw_next_observations=chunk_data['next_observations'],
            raw_dones=chunk_data['dones'],
            metadata={'timestamp': self.step_count},
            selection_weight=1.0,
        )

        self.episodic_buffer.add_chunk(episodic_chunk)

    def update(self) -> Optional[Dict[str, float]]:
        """Perform learning update with multi-task training.

        Returns:
            Training statistics or None if not ready to update
        """
        # Check if ready to update
        if len(self.episodic_buffer) < self.warmup_chunks:
            return None

        if self.step_count % self.update_freq != 0:
            return None

        # Sample batch of chunks
        try:
            batch = self.episodic_buffer.sample(self.batch_size)
        except ValueError:
            return None

        # Convert to tensors
        wavelet_chunks = torch.as_tensor(batch['wavelet_chunks'], dtype=torch.float32, device=self.device)
        raw_actions = torch.as_tensor(batch['raw_actions'], dtype=torch.float32, device=self.device)
        raw_rewards = torch.as_tensor(batch['raw_rewards'], dtype=torch.float32, device=self.device)

        # Forward pass through encoder
        latent = self.encoder(wavelet_chunks)  # (B, latent_dim)

        # Policy head
        predicted_actions = self.heads['policy'](latent)  # (B, act_dim)

        # Value head
        predicted_values = self.heads['value'](latent)  # (B, 1)

        # Reconstruction head
        reconstructed_chunks = self.heads['reconstruction'](latent)  # (B, 5, 5, D)

        # Forward model head
        # Use mean action from each chunk as input
        mean_actions = raw_actions.mean(dim=1)  # (B, act_dim)
        predicted_next_chunks = self.heads['forward'](latent, mean_actions)  # (B, 5, 5, D)

        # Compute losses
        # 1. Policy loss: imitation learning on actual actions
        target_actions = raw_actions.mean(dim=1)  # (B, act_dim) - average action in chunk
        # Normalize target actions to [-1, 1] range (matching tanh output)
        target_actions_norm = 2.0 * (target_actions - torch.tensor(self.action_low, device=self.device)) / \
                             torch.tensor(self.action_high - self.action_low, device=self.device) - 1.0
        policy_loss = F.mse_loss(predicted_actions, target_actions_norm)

        # 2. Value loss: predict cumulative reward in chunk
        target_values = raw_rewards.sum(dim=1, keepdim=True)  # (B, 1)
        value_loss = F.mse_loss(predicted_values, target_values)

        # 3. Reconstruction loss
        reconstruction_loss = F.mse_loss(reconstructed_chunks, wavelet_chunks)

        # 4. Forward model loss
        # Ideally, we'd predict the next chunk, but for simplicity, predict same chunk (identity)
        # TODO: Track consecutive chunks in buffer for proper forward modeling
        forward_loss = F.mse_loss(predicted_next_chunks, wavelet_chunks)

        # Combined loss
        total_loss = (
            self.policy_loss_weight * policy_loss +
            self.value_loss_weight * value_loss +
            self.reconstruction_loss_weight * reconstruction_loss +
            self.forward_loss_weight * forward_loss
        )

        # Backward pass
        self.encoder_opt.zero_grad()
        self.policy_opt.zero_grad()
        self.value_opt.zero_grad()
        self.reconstruction_opt.zero_grad()
        self.forward_opt.zero_grad()

        total_loss.backward()

        # Compute gradient norms before clipping (for monitoring)
        encoder_grad_norm = nn.utils.clip_grad_norm_(self.encoder.parameters(), float('inf'))
        policy_grad_norm = nn.utils.clip_grad_norm_(self.heads['policy'].parameters(), float('inf'))
        value_grad_norm = nn.utils.clip_grad_norm_(self.heads['value'].parameters(), float('inf'))
        recon_grad_norm = nn.utils.clip_grad_norm_(self.heads['reconstruction'].parameters(), float('inf'))
        forward_grad_norm = nn.utils.clip_grad_norm_(self.heads['forward'].parameters(), float('inf'))

        # Gradient clipping
        nn.utils.clip_grad_norm_(self.encoder.parameters(), self.max_grad_norm)
        nn.utils.clip_grad_norm_(self.heads['policy'].parameters(), self.max_grad_norm)
        nn.utils.clip_grad_norm_(self.heads['value'].parameters(), self.max_grad_norm)
        nn.utils.clip_grad_norm_(self.heads['reconstruction'].parameters(), self.max_grad_norm)
        nn.utils.clip_grad_norm_(self.heads['forward'].parameters(), self.max_grad_norm)

        # Optimizer steps
        self.encoder_opt.step()
        self.policy_opt.step()
        self.value_opt.step()
        self.reconstruction_opt.step()
        self.forward_opt.step()

        # Update chunk priorities (if using priority mode)
        if self.episodic_buffer.selection_mode == "priority":
            # Use reconstruction error as priority signal
            with torch.no_grad():
                recon_errors = (reconstructed_chunks - wavelet_chunks).abs().sum(dim=(1, 2, 3))
            self.episodic_buffer.update_weights(
                batch['indices'],
                recon_errors.cpu().numpy()
            )

        self.update_count += 1

        # Compute additional metrics for monitoring
        with torch.no_grad():
            # Reconstruction quality metrics
            recon_mse = F.mse_loss(reconstructed_chunks, wavelet_chunks).item()
            recon_mae = (reconstructed_chunks - wavelet_chunks).abs().mean().item()
            recon_max_error = (reconstructed_chunks - wavelet_chunks).abs().max().item()

            # Forward model quality metrics
            forward_mse = F.mse_loss(predicted_next_chunks, next_wavelet_chunks).item()
            forward_mae = (predicted_next_chunks - next_wavelet_chunks).abs().mean().item()

            # Action statistics
            action_mean = predicted_actions.mean().item()
            action_std = predicted_actions.std().item()
            steering_mean = predicted_actions[:, 0].mean().item()
            steering_std = predicted_actions[:, 0].std().item()
            velocity_mean = predicted_actions[:, 1].mean().item()
            velocity_std = predicted_actions[:, 1].std().item()

            # Value statistics
            value_mean = predicted_values.mean().item()
            value_std = predicted_values.std().item()
            value_min = predicted_values.min().item()
            value_max = predicted_values.max().item()

            # Buffer statistics
            chronological_size = len(self.chronological_buffer)
            episodic_size = len(self.episodic_buffer)
            episodic_utilization = episodic_size / self.episodic_buffer.capacity

            # Priority statistics (if using priority mode)
            if self.episodic_buffer.selection_mode == "priority":
                weights = self.episodic_buffer.get_all_weights()
                weight_mean = np.mean(weights) if len(weights) > 0 else 0.0
                weight_std = np.std(weights) if len(weights) > 0 else 0.0
                weight_max = np.max(weights) if len(weights) > 0 else 0.0
                weight_min = np.min(weights) if len(weights) > 0 else 0.0
            else:
                weight_mean = weight_std = weight_max = weight_min = 1.0

        # Comprehensive metrics dictionary
        metrics = {
            # === Loss Metrics ===
            'loss/total': total_loss.item(),
            'loss/policy': policy_loss.item(),
            'loss/value': value_loss.item(),
            'loss/reconstruction': reconstruction_loss.item(),
            'loss/forward': forward_loss.item(),

            # Weighted loss contributions
            'loss_weighted/policy': (self.policy_loss_weight * policy_loss).item(),
            'loss_weighted/value': (self.value_loss_weight * value_loss).item(),
            'loss_weighted/reconstruction': (self.reconstruction_loss_weight * reconstruction_loss).item(),
            'loss_weighted/forward': (self.forward_loss_weight * forward_loss).item(),

            # === Buffer Statistics ===
            'buffer/chronological_size': chronological_size,
            'buffer/episodic_size': episodic_size,
            'buffer/episodic_utilization': episodic_utilization,
            'buffer/episodic_capacity': self.episodic_buffer.capacity,

            # === Action Statistics ===
            'actions/mean': action_mean,
            'actions/std': action_std,
            'actions/steering_mean': steering_mean,
            'actions/steering_std': steering_std,
            'actions/velocity_mean': velocity_mean,
            'actions/velocity_std': velocity_std,

            # === Value Statistics ===
            'values/mean': value_mean,
            'values/std': value_std,
            'values/min': value_min,
            'values/max': value_max,

            # === Reconstruction Quality ===
            'reconstruction/mse': recon_mse,
            'reconstruction/mae': recon_mae,
            'reconstruction/max_error': recon_max_error,

            # === Forward Model Quality ===
            'forward_model/mse': forward_mse,
            'forward_model/mae': forward_mae,

            # === Gradient Norms ===
            'gradients/encoder_norm': encoder_grad_norm.item(),
            'gradients/policy_norm': policy_grad_norm.item(),
            'gradients/value_norm': value_grad_norm.item(),
            'gradients/reconstruction_norm': recon_grad_norm.item(),
            'gradients/forward_norm': forward_grad_norm.item(),
            'gradients/total_norm': (
                encoder_grad_norm.item() +
                policy_grad_norm.item() +
                value_grad_norm.item() +
                recon_grad_norm.item() +
                forward_grad_norm.item()
            ),

            # === Priority Sampling Statistics ===
            'priority/weight_mean': weight_mean,
            'priority/weight_std': weight_std,
            'priority/weight_max': weight_max,
            'priority/weight_min': weight_min,

            # === Training Progress ===
            'training/update_count': self.update_count,
            'training/batch_size': len(batch['indices']),
        }

        return metrics

    def save(self, path: str) -> None:
        """Save agent state to disk.

        Args:
            path: Path to save directory
        """
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save networks
        torch.save(self.encoder.state_dict(), save_path / "encoder.pt")
        torch.save(self.heads['policy'].state_dict(), save_path / "policy_head.pt")
        torch.save(self.heads['value'].state_dict(), save_path / "value_head.pt")
        torch.save(self.heads['reconstruction'].state_dict(), save_path / "reconstruction_head.pt")
        torch.save(self.heads['forward'].state_dict(), save_path / "forward_head.pt")

        # Save optimizers
        torch.save(self.encoder_opt.state_dict(), save_path / "encoder_opt.pt")
        torch.save(self.policy_opt.state_dict(), save_path / "policy_opt.pt")
        torch.save(self.value_opt.state_dict(), save_path / "value_opt.pt")
        torch.save(self.reconstruction_opt.state_dict(), save_path / "reconstruction_opt.pt")
        torch.save(self.forward_opt.state_dict(), save_path / "forward_opt.pt")

        # Save counters
        torch.save({
            'step_count': self.step_count,
            'update_count': self.update_count,
        }, save_path / "counters.pt")

        print(f"Saved agent to {save_path}")

    def load(self, path: str) -> None:
        """Load agent state from disk.

        Args:
            path: Path to save directory
        """
        load_path = Path(path)

        # Load networks
        self.encoder.load_state_dict(torch.load(load_path / "encoder.pt", map_location=self.device))
        self.heads['policy'].load_state_dict(torch.load(load_path / "policy_head.pt", map_location=self.device))
        self.heads['value'].load_state_dict(torch.load(load_path / "value_head.pt", map_location=self.device))
        self.heads['reconstruction'].load_state_dict(torch.load(load_path / "reconstruction_head.pt", map_location=self.device))
        self.heads['forward'].load_state_dict(torch.load(load_path / "forward_head.pt", map_location=self.device))

        # Load optimizers
        self.encoder_opt.load_state_dict(torch.load(load_path / "encoder_opt.pt", map_location=self.device))
        self.policy_opt.load_state_dict(torch.load(load_path / "policy_opt.pt", map_location=self.device))
        self.value_opt.load_state_dict(torch.load(load_path / "value_opt.pt", map_location=self.device))
        self.reconstruction_opt.load_state_dict(torch.load(load_path / "reconstruction_opt.pt", map_location=self.device))
        self.forward_opt.load_state_dict(torch.load(load_path / "forward_opt.pt", map_location=self.device))

        # Load counters
        counters = torch.load(load_path / "counters.pt", map_location=self.device)
        self.step_count = counters['step_count']
        self.update_count = counters['update_count']

        print(f"Loaded agent from {load_path}")
