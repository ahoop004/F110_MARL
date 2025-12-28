"""Shared integration modes for physics components."""
from enum import Enum


class Integrator(Enum):
    RK4 = "RK4"
    Euler = "Euler"


__all__ = ["Integrator"]
