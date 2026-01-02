"""Wavelet-based Episodic Memory Agent."""

from agents.episodic.wavelet_agent import WaveletEpisodicAgent
from agents.episodic.wavelet import (
    WaveletTransformBase,
    IdentityTransform,
    HaarWavelet,
    DaubechiesWavelet,
    MorletWavelet,
    build_wavelet,
)
from agents.episodic.encoder import ChunkEncoder, build_chunk_encoder
from agents.episodic.heads import (
    PolicyHead,
    ValueHead,
    ReconstructionHead,
    ForwardModelHead,
    build_heads,
)

__all__ = [
    'WaveletEpisodicAgent',
    'WaveletTransformBase',
    'IdentityTransform',
    'HaarWavelet',
    'DaubechiesWavelet',
    'MorletWavelet',
    'build_wavelet',
    'ChunkEncoder',
    'build_chunk_encoder',
    'PolicyHead',
    'ValueHead',
    'ReconstructionHead',
    'ForwardModelHead',
    'build_heads',
]
