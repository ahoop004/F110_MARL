"""Chunk encoder network: Conv2D + RNN hybrid for episodic memory."""

import torch
import torch.nn as nn
from typing import Dict, List, Optional


class ChunkEncoder(nn.Module):
    """Hybrid CNN-RNN encoder for processing episodic memory chunks.

    Takes wavelet-transformed 5×5 grids and encodes them into latent embeddings.

    Architecture:
        Input: (batch, 5, 5, feature_dim) - 5×5 grid of wavelet features
        → Conv2D layers: Extract spatial-temporal patterns
        → Flatten + Optional RNN: Model sequential dependencies
        → Linear projection: To latent space

    Args:
        input_feature_dim: Dimension of features at each grid cell
        latent_dim: Output latent embedding dimension
        cnn_channels: List of CNN channel dimensions (e.g., [32, 64, 128])
        kernel_size: Convolution kernel size (e.g., 3 for 3×3)
        pooling: List of pooling sizes for each layer (None = no pooling)
        use_rnn: Whether to use RNN after CNN
        rnn_type: 'lstm', 'gru', or 'rnn'
        rnn_hidden_size: RNN hidden dimension
        rnn_num_layers: Number of RNN layers
        rnn_dropout: Dropout rate for RNN
    """

    def __init__(
        self,
        input_feature_dim: int,
        latent_dim: int = 128,
        cnn_channels: List[int] = [32, 64, 128],
        kernel_size: int = 3,
        pooling: Optional[List[Optional[int]]] = None,
        use_rnn: bool = True,
        rnn_type: str = "lstm",
        rnn_hidden_size: int = 256,
        rnn_num_layers: int = 2,
        rnn_dropout: float = 0.1,
    ):
        super().__init__()

        self.input_feature_dim = input_feature_dim
        self.latent_dim = latent_dim
        self.use_rnn = use_rnn

        # Default pooling: 2x2 for first two layers, none for last
        if pooling is None:
            pooling = [2, 2, None]

        # Build CNN layers
        # Input: (B, 1, 5, 5, feature_dim) → treat as (B, feature_dim, 5, 5) for Conv2D
        cnn_layers = []
        in_channels = input_feature_dim

        self.cnn_channels = cnn_channels
        self.pooling = pooling

        for i, out_channels in enumerate(cnn_channels):
            # Conv2D layer
            cnn_layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,  # Same padding
                )
            )
            cnn_layers.append(nn.ReLU())

            # Optional pooling
            if pooling[i] is not None:
                cnn_layers.append(nn.MaxPool2d(kernel_size=pooling[i]))

            in_channels = out_channels

        self.cnn = nn.Sequential(*cnn_layers)

        # Compute output spatial size after CNN
        # Starting from 5×5, after pooling [2, 2, None]:
        # 5 → pool(2) → 2 → pool(2) → 1 → no pool → 1
        # So final spatial size is approximately 1×1 or 2×2 depending on exact pooling
        # For simplicity, we'll flatten and use linear layer

        # Placeholder for CNN output size (computed during first forward pass)
        self.cnn_output_size = None

        if use_rnn:
            # RNN for sequential modeling
            # Input: (B, seq_len, features) after reshaping CNN output
            rnn_class = {
                "lstm": nn.LSTM,
                "gru": nn.GRU,
                "rnn": nn.RNN,
            }[rnn_type.lower()]

            self.rnn = rnn_class(
                input_size=cnn_channels[-1],  # Use last CNN channel as input
                hidden_size=rnn_hidden_size,
                num_layers=rnn_num_layers,
                batch_first=True,
                dropout=rnn_dropout if rnn_num_layers > 1 else 0.0,
            )

            self.rnn_type = rnn_type.lower()
            self.rnn_hidden_size = rnn_hidden_size

            # Final projection from RNN hidden to latent
            self.projection = nn.Linear(rnn_hidden_size, latent_dim)
        else:
            # Direct projection from flattened CNN output to latent
            self.rnn = None
            # We'll create the projection layer after first forward pass
            self.projection = None

    def forward(self, chunk_batch: torch.Tensor) -> torch.Tensor:
        """Encode a batch of wavelet-transformed chunks.

        Args:
            chunk_batch: (B, 5, 5, feature_dim) tensor of wavelet coefficients

        Returns:
            latent: (B, latent_dim) latent embeddings
        """
        B, H, W, D = chunk_batch.shape
        assert H == 5 and W == 5, f"Expected 5×5 grid, got {H}×{W}"

        # Reshape for Conv2D: (B, feature_dim, H, W)
        x = chunk_batch.permute(0, 3, 1, 2)  # (B, D, 5, 5)

        # CNN forward
        x = self.cnn(x)  # (B, C, H', W') where H', W' are reduced by pooling

        # Initialize output size if first forward pass
        if self.cnn_output_size is None:
            _, C, H_out, W_out = x.shape
            self.cnn_output_size = C * H_out * W_out

            # Create projection if not using RNN
            if not self.use_rnn:
                self.projection = nn.Linear(self.cnn_output_size, self.latent_dim)
                self.projection = self.projection.to(x.device)

        if self.use_rnn:
            # Reshape for RNN: (B, seq_len, features)
            # Flatten spatial dimensions: (B, C, H', W') → (B, H'*W', C)
            _, C, H_out, W_out = x.shape
            x = x.view(B, C, H_out * W_out)  # (B, C, H'*W')
            x = x.permute(0, 2, 1)  # (B, H'*W', C) - sequence of spatial locations

            # RNN forward
            rnn_out, hidden = self.rnn(x)  # rnn_out: (B, seq_len, hidden_size)

            # Use final hidden state
            if self.rnn_type == "lstm":
                latent_input = hidden[0][-1]  # (B, hidden_size) - last layer hidden
            else:
                latent_input = hidden[-1]  # (B, hidden_size)

            # Project to latent space
            latent = self.projection(latent_input)  # (B, latent_dim)
        else:
            # Flatten CNN output
            x_flat = x.view(B, -1)  # (B, cnn_output_size)

            # Project to latent space
            latent = self.projection(x_flat)  # (B, latent_dim)

        return latent


def build_chunk_encoder(config: Dict) -> ChunkEncoder:
    """Factory function to build ChunkEncoder from config.

    Args:
        config: Configuration dictionary with keys:
            - input_feature_dim: int
            - latent_dim: int (default: 128)
            - cnn: Dict with CNN configuration
                - channels: List[int] (default: [32, 64, 128])
                - kernel_size: int (default: 3)
                - pooling: List[Optional[int]] (default: [2, 2, None])
            - rnn: Dict with RNN configuration
                - use: bool (default: True)
                - type: str (default: 'lstm')
                - hidden_size: int (default: 256)
                - num_layers: int (default: 2)
                - dropout: float (default: 0.1)

    Returns:
        ChunkEncoder instance

    Example:
        >>> config = {
        ...     'input_feature_dim': 107,
        ...     'latent_dim': 128,
        ...     'cnn': {'channels': [32, 64, 128]},
        ...     'rnn': {'type': 'lstm', 'hidden_size': 256}
        ... }
        >>> encoder = build_chunk_encoder(config)
    """
    input_feature_dim = config['input_feature_dim']
    latent_dim = config.get('latent_dim', 128)

    cnn_config = config.get('cnn', {})
    cnn_channels = cnn_config.get('channels', [32, 64, 128])
    kernel_size = cnn_config.get('kernel_size', 3)
    pooling = cnn_config.get('pooling', [2, 2, None])

    rnn_config = config.get('rnn', {})
    use_rnn = rnn_config.get('use', True)
    rnn_type = rnn_config.get('type', 'lstm')
    rnn_hidden_size = rnn_config.get('hidden_size', 256)
    rnn_num_layers = rnn_config.get('num_layers', 2)
    rnn_dropout = rnn_config.get('dropout', 0.1)

    return ChunkEncoder(
        input_feature_dim=input_feature_dim,
        latent_dim=latent_dim,
        cnn_channels=cnn_channels,
        kernel_size=kernel_size,
        pooling=pooling,
        use_rnn=use_rnn,
        rnn_type=rnn_type,
        rnn_hidden_size=rnn_hidden_size,
        rnn_num_layers=rnn_num_layers,
        rnn_dropout=rnn_dropout,
    )
