from __future__ import annotations

import pytest

from scripts.benchmark_complete4 import SUPPORTED_SCENARIOS, summarize_results


def _result(decisions_per_second: float, rss: int = 100) -> dict:
    return {
        "action_sequence_sha256": "same",
        "counts": {
            "joint_decisions": 10,
            "physics_substeps": 20,
            "transitions": 40,
            "optimizer_samples": 400,
            "ppo_updates": 1,
        },
        "throughput": {
            "decisions_per_second": decisions_per_second,
            "physics_substeps_per_second": decisions_per_second * 2,
            "update_samples_per_second": decisions_per_second * 10,
        },
        "seconds": {"total_measured": 10.0 / decisions_per_second},
        "memory": {"peak_rss_kib": rss, "peak_cuda_bytes": 0},
    }


def test_benchmark_summary_reports_median_spread_and_fixed_work() -> None:
    summary = summarize_results([_result(90.0), _result(100.0), _result(110.0)])

    assert summary["repetitions"] == 3
    assert summary["fixed_work_verified"] is True
    assert summary["action_sequence_verified"] is True
    assert summary["metrics"]["decisions_per_second"] == {
        "median": 100.0,
        "min": 90.0,
        "max": 110.0,
        "spread": 20.0,
    }


def test_benchmark_summary_detects_workload_or_action_drift() -> None:
    first = _result(100.0)
    second = _result(100.0)
    second["counts"]["physics_substeps"] = 19
    second["action_sequence_sha256"] = "different"

    summary = summarize_results([first, second])

    assert summary["fixed_work_verified"] is False
    assert summary["action_sequence_verified"] is False


def test_benchmark_summary_requires_results() -> None:
    with pytest.raises(ValueError, match="At least one"):
        summarize_results([])


def test_benchmark_supports_all_complete_4_observation_arms() -> None:
    assert SUPPORTED_SCENARIOS == (
        "scenarios/complete_4.yaml",
        "scenarios/complete_4_frenet.yaml",
        "scenarios/complete_4_frenet_neighbors.yaml",
    )
