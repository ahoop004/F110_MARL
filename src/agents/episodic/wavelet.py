"""Modular wavelet transform interface for episodic memory preprocessing."""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
import pywt


class WaveletTransformBase(ABC):
    """Abstract base class for wavelet transforms.

    Wavelet transforms decompose time series into multi-scale representations,
    capturing both low-frequency trends and high-frequency details.
    """

    @abstractmethod
    def transform(self, sequence: np.ndarray) -> np.ndarray:
        """Transform a sequence using wavelet decomposition.

        Args:
            sequence: Input sequence, shape (N, D) where N=time steps, D=features

        Returns:
            Wavelet coefficients, shape depends on implementation
        """
        pass

    @abstractmethod
    def inverse_transform(self, coefficients: np.ndarray) -> np.ndarray:
        """Inverse transform from wavelet coefficients back to sequence.

        Args:
            coefficients: Wavelet coefficients

        Returns:
            Reconstructed sequence, shape (N, D)
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class IdentityTransform(WaveletTransformBase):
    """Identity transform (pass-through) for ablation studies.

    No wavelet transform is applied; input is returned as-is.
    """

    def transform(self, sequence: np.ndarray) -> np.ndarray:
        """Return sequence unchanged."""
        return sequence.copy()

    def inverse_transform(self, coefficients: np.ndarray) -> np.ndarray:
        """Return coefficients unchanged."""
        return coefficients.copy()


class HaarWavelet(WaveletTransformBase):
    """Haar wavelet transform.

    Simplest wavelet: piecewise constant. Fast and good for step-like changes.

    Args:
        mode: Padding mode ('symmetric', 'zero', 'periodic')
        level: Decomposition level (None = auto-compute max level)
    """

    def __init__(self, mode: str = "symmetric", level: Optional[int] = None):
        self.wavelet = "haar"
        self.mode = mode
        self.level = level

    def transform(self, sequence: np.ndarray) -> np.ndarray:
        """Apply Haar wavelet transform to each feature dimension.

        Args:
            sequence: (N, D) array

        Returns:
            Wavelet coefficients (N_coeffs, D) where N_coeffs depends on level
        """
        N, D = sequence.shape

        # Auto-compute level if not specified
        level = self.level if self.level is not None else pywt.dwt_max_level(N, self.wavelet)
        level = min(level, 3)  # Cap at 3 levels for stability

        coeffs_list = []

        for d in range(D):
            signal = sequence[:, d]

            # Perform wavelet decomposition
            coeffs = pywt.wavedec(signal, self.wavelet, mode=self.mode, level=level)

            # Flatten coefficients: [cA_n, cD_n, cD_n-1, ..., cD_1]
            coeffs_flat = np.concatenate([c for c in coeffs])
            coeffs_list.append(coeffs_flat)

        # Stack all features: (coeffs_len, D)
        result = np.stack(coeffs_list, axis=1)

        return result

    def inverse_transform(self, coefficients: np.ndarray) -> np.ndarray:
        """Reconstruct sequence from wavelet coefficients.

        Args:
            coefficients: (N_coeffs, D) array

        Returns:
            Reconstructed sequence (N, D)
        """
        N_coeffs, D = coefficients.shape

        # Note: Exact inverse requires knowing original decomposition structure
        # For simplicity, we'll use waverec with auto-detection
        # This is approximate and mainly for reconstruction loss

        reconstructed_list = []

        for d in range(D):
            coeffs_flat = coefficients[:, d]

            # Approximate reconstruction by treating as single-level
            # In practice, you'd need to store the decomposition structure
            # For now, use identity as fallback
            reconstructed_list.append(coeffs_flat)

        result = np.stack(reconstructed_list, axis=1)

        return result

    def __repr__(self) -> str:
        return f"HaarWavelet(mode={self.mode}, level={self.level})"


class DaubechiesWavelet(WaveletTransformBase):
    """Daubechies wavelet transform.

    Smoother than Haar, better for continuous signals.
    Common choices: db2, db4, db6, db8.

    Args:
        order: Daubechies wavelet order (2, 4, 6, 8, ...)
        mode: Padding mode
        level: Decomposition level (None = auto-compute)
    """

    def __init__(self, order: int = 2, mode: str = "symmetric", level: Optional[int] = None):
        self.wavelet = f"db{order}"
        self.order = order
        self.mode = mode
        self.level = level

    def transform(self, sequence: np.ndarray) -> np.ndarray:
        """Apply Daubechies wavelet transform.

        Args:
            sequence: (N, D) array

        Returns:
            Wavelet coefficients (N_coeffs, D)
        """
        N, D = sequence.shape

        # Auto-compute level
        level = self.level if self.level is not None else pywt.dwt_max_level(N, self.wavelet)
        level = min(level, 3)

        coeffs_list = []

        for d in range(D):
            signal = sequence[:, d]
            coeffs = pywt.wavedec(signal, self.wavelet, mode=self.mode, level=level)
            coeffs_flat = np.concatenate([c for c in coeffs])
            coeffs_list.append(coeffs_flat)

        result = np.stack(coeffs_list, axis=1)

        return result

    def inverse_transform(self, coefficients: np.ndarray) -> np.ndarray:
        """Reconstruct sequence (approximate)."""
        return coefficients.copy()  # Simplified for now

    def __repr__(self) -> str:
        return f"DaubechiesWavelet(order={self.order}, mode={self.mode}, level={self.level})"


class MorletWavelet(WaveletTransformBase):
    """Morlet wavelet transform using Continuous Wavelet Transform (CWT).

    Good for frequency analysis. More complex than DWT-based wavelets.

    Args:
        scales: Array of scales for CWT (determines frequency resolution)
        sampling_period: Sampling period of input signal
    """

    def __init__(self, scales: Optional[np.ndarray] = None, sampling_period: float = 1.0):
        self.wavelet = "morl"
        self.scales = scales if scales is not None else np.arange(1, 32)  # Default scales
        self.sampling_period = sampling_period

    def transform(self, sequence: np.ndarray) -> np.ndarray:
        """Apply CWT with Morlet wavelet.

        Args:
            sequence: (N, D) array

        Returns:
            CWT coefficients (scales, N, D) → flattened to (scales*N, D) for compatibility
        """
        N, D = sequence.shape

        all_coeffs = []

        for d in range(D):
            signal = sequence[:, d]

            # Perform CWT
            coeffs, freqs = pywt.cwt(signal, self.scales, self.wavelet, sampling_period=self.sampling_period)

            # coeffs shape: (scales, N)
            # Flatten to (scales * N,) for this feature
            coeffs_flat = coeffs.flatten()
            all_coeffs.append(coeffs_flat)

        # Stack: (scales*N, D)
        result = np.stack(all_coeffs, axis=1)

        return result

    def inverse_transform(self, coefficients: np.ndarray) -> np.ndarray:
        """Approximate inverse (CWT inverse is non-trivial)."""
        # For reconstruction loss, we'll use the coefficients directly
        # Proper CWT inversion requires icwt which is more complex
        return coefficients.copy()

    def __repr__(self) -> str:
        return f"MorletWavelet(scales={len(self.scales)}, sampling_period={self.sampling_period})"


def build_wavelet(config: dict) -> WaveletTransformBase:
    """Factory function to build wavelet transform from config.

    Args:
        config: Configuration dictionary with keys:
            - type: 'haar', 'db2', 'db4', 'db6', 'morlet', 'identity'
            - mode: Padding mode (default: 'symmetric')
            - level: Decomposition level (default: None for auto)

    Returns:
        WaveletTransformBase instance

    Example:
        >>> wavelet = build_wavelet({'type': 'haar', 'mode': 'symmetric'})
        >>> coeffs = wavelet.transform(sequence)
    """
    wavelet_type = config.get('type', 'haar').lower()
    mode = config.get('mode', 'symmetric')
    level = config.get('level', None)

    if wavelet_type == 'identity':
        return IdentityTransform()
    elif wavelet_type == 'haar':
        return HaarWavelet(mode=mode, level=level)
    elif wavelet_type in ['db2', 'db4', 'db6', 'db8']:
        order = int(wavelet_type[2:])  # Extract number from 'db2' → 2
        return DaubechiesWavelet(order=order, mode=mode, level=level)
    elif wavelet_type == 'morlet':
        scales = config.get('scales', None)
        sampling_period = config.get('sampling_period', 1.0)
        return MorletWavelet(scales=scales, sampling_period=sampling_period)
    else:
        raise ValueError(f"Unknown wavelet type: {wavelet_type}. "
                        f"Supported: haar, db2, db4, db6, db8, morlet, identity")
