"""Curriculum learning modules for F110 MARL training.

Provides structured curriculum learning systems that progressively increase
task difficulty across multiple dimensions.
"""

from .phased_curriculum import (
    PhaseBasedCurriculum,
    Phase,
    AdvancementCriteria,
    CurriculumState,
)

__all__ = [
    'PhaseBasedCurriculum',
    'Phase',
    'AdvancementCriteria',
    'CurriculumState',
]
