"""Multi-task learning heads for wavelet episodic agent."""

import torch
import torch.nn as nn
from typing import List


def build_mlp(
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    activation: nn.Module = nn.ReLU(),
    output_activation: nn.Module = None,
) -> nn.Sequential:
    """Build a multi-layer perceptron.

    Args:
        input_dim: Input dimension
        hidden_dims: List of hidden layer dimensions
        output_dim: Output dimension
        activation: Activation function for hidden layers
        output_activation: Activation function for output layer (None = no activation)

    Returns:
        Sequential MLP network
    """
    layers = []
    prev_dim = input_dim

    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(activation)
        prev_dim = hidden_dim

    layers.append(nn.Linear(prev_dim, output_dim))

    if output_activation is not None:
        layers.append(output_activation)

    return nn.Sequential(*layers)


class PolicyHead(nn.Module):
    """Policy head for action prediction.

    Outputs continuous actions (steering angle + velocity).

    Args:
        latent_dim: Input latent dimension
        act_dim: Action dimension (typically 2 for steering + velocity)
        hidden_dims: Hidden layer dimensions
        activation: Activation function
        output_activation: Output activation (e.g., Tanh for bounded actions)
    """

    def __init__(
        self,
        latent_dim: int,
        act_dim: int = 2,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu",
        output_activation: str = "tanh",
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.act_dim = act_dim

        # Activation functions
        act_fn = nn.ReLU() if activation == "relu" else nn.Tanh()
        out_act = nn.Tanh() if output_activation == "tanh" else None

        self.net = build_mlp(
            latent_dim,
            hidden_dims,
            act_dim,
            activation=act_fn,
            output_activation=out_act,
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Predict actions from latent embedding.

        Args:
            latent: (B, latent_dim) latent embeddings

        Returns:
            actions: (B, act_dim) predicted actions
        """
        return self.net(latent)


class ValueHead(nn.Module):
    """Value head for state value estimation.

    Args:
        latent_dim: Input latent dimension
        hidden_dims: Hidden layer dimensions
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: List[int] = [256, 128],
    ):
        super().__init__()

        self.latent_dim = latent_dim

        self.net = build_mlp(
            latent_dim,
            hidden_dims,
            output_dim=1,
            activation=nn.ReLU(),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Estimate value from latent embedding.

        Args:
            latent: (B, latent_dim) latent embeddings

        Returns:
            values: (B, 1) state values
        """
        return self.net(latent)


class ReconstructionHead(nn.Module):
    """Reconstruction head for autoencoder-style learning.

    Reconstructs the original wavelet-transformed chunk from latent embedding.

    Args:
        latent_dim: Input latent dimension
        chunk_shape: Shape of chunk to reconstruct (e.g., (5, 5, feature_dim))
        hidden_dims: Hidden layer dimensions
    """

    def __init__(
        self,
        latent_dim: int,
        chunk_shape: tuple,  # (H, W, D) e.g., (5, 5, 107)
        hidden_dims: List[int] = [256, 512],
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.chunk_shape = chunk_shape
        self.output_dim = int(torch.prod(torch.tensor(chunk_shape)))

        self.net = build_mlp(
            latent_dim,
            hidden_dims,
            self.output_dim,
            activation=nn.ReLU(),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Reconstruct chunk from latent embedding.

        Args:
            latent: (B, latent_dim) latent embeddings

        Returns:
            reconstructed: (B, H, W, D) reconstructed chunks
        """
        B = latent.shape[0]
        flat_output = self.net(latent)  # (B, H*W*D)
        return flat_output.view(B, *self.chunk_shape)  # (B, H, W, D)


class ForwardModelHead(nn.Module):
    """Forward model head for predicting next chunk.

    Predicts the next chunk given current chunk embedding and action.

    Args:
        latent_dim: Input latent dimension
        act_dim: Action dimension
        chunk_shape: Shape of chunk to predict (e.g., (5, 5, feature_dim))
        hidden_dims: Hidden layer dimensions
    """

    def __init__(
        self,
        latent_dim: int,
        act_dim: int,
        chunk_shape: tuple,  # (H, W, D)
        hidden_dims: List[int] = [256, 512],
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.act_dim = act_dim
        self.chunk_shape = chunk_shape
        self.output_dim = int(torch.prod(torch.tensor(chunk_shape)))

        # Concatenate latent and action
        input_dim = latent_dim + act_dim

        self.net = build_mlp(
            input_dim,
            hidden_dims,
            self.output_dim,
            activation=nn.ReLU(),
        )

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict next chunk from latent and action.

        Args:
            latent: (B, latent_dim) latent embeddings
            action: (B, act_dim) actions taken

        Returns:
            predicted_next: (B, H, W, D) predicted next chunks
        """
        B = latent.shape[0]

        # Concatenate latent and action
        x = torch.cat([latent, action], dim=-1)  # (B, latent_dim + act_dim)

        # Predict next chunk
        flat_output = self.net(x)  # (B, H*W*D)
        return flat_output.view(B, *self.chunk_shape)  # (B, H, W, D)


def build_heads(
    latent_dim: int,
    act_dim: int,
    chunk_shape: tuple,
    config: dict,
) -> dict:
    """Factory function to build all task heads from config.

    Args:
        latent_dim: Latent embedding dimension
        act_dim: Action dimension
        chunk_shape: Shape of chunks (H, W, D)
        config: Configuration dictionary with keys:
            - policy_head: Dict with PolicyHead config
            - value_head: Dict with ValueHead config
            - reconstruction_head: Dict with ReconstructionHead config
            - forward_head: Dict with ForwardModelHead config

    Returns:
        Dictionary of head instances: {'policy': PolicyHead, 'value': ValueHead, ...}

    Example:
        >>> config = {
        ...     'policy_head': {'hidden_dims': [256, 256]},
        ...     'value_head': {'hidden_dims': [256, 128]},
        ...     'reconstruction_head': {'hidden_dims': [256, 512]},
        ...     'forward_head': {'hidden_dims': [256, 512]},
        ... }
        >>> heads = build_heads(128, 2, (5, 5, 107), config)
    """
    heads = {}

    # Policy head
    policy_config = config.get('policy_head', {})
    heads['policy'] = PolicyHead(
        latent_dim=latent_dim,
        act_dim=act_dim,
        hidden_dims=policy_config.get('hidden_dims', [256, 256]),
        activation=policy_config.get('activation', 'relu'),
        output_activation=policy_config.get('output_activation', 'tanh'),
    )

    # Value head
    value_config = config.get('value_head', {})
    heads['value'] = ValueHead(
        latent_dim=latent_dim,
        hidden_dims=value_config.get('hidden_dims', [256, 128]),
    )

    # Reconstruction head
    recon_config = config.get('reconstruction_head', {})
    heads['reconstruction'] = ReconstructionHead(
        latent_dim=latent_dim,
        chunk_shape=chunk_shape,
        hidden_dims=recon_config.get('hidden_dims', [256, 512]),
    )

    # Forward model head
    forward_config = config.get('forward_head', {})
    heads['forward'] = ForwardModelHead(
        latent_dim=latent_dim,
        act_dim=act_dim,
        chunk_shape=chunk_shape,
        hidden_dims=forward_config.get('hidden_dims', [256, 512]),
    )

    return heads
