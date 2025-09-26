# src/f110x/__init__.py
from f110x import envs, physics, render, utils, wrappers, tasks

from gymnasium.envs.registration import register, registry

from .envs.f110ParallelEnv import F110ParallelEnv

ENV_ID = "F110x-v0"

if ENV_ID not in registry:
    register(
        id=ENV_ID,
        entry_point="f110x.envs:F110ParallelEnv",
    )

__all__ = ["F110ParallelEnv"]
