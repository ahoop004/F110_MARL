"""Smoke checks for the active setup/factory path."""
from __future__ import annotations

import sys
from typing import Dict

sys.path.insert(0, ".")
sys.path.insert(0, "src")

from src.core.config import AgentFactory, register_builtin_agents
from src.core.scenario import load_and_expand_scenario
from src.core.setup import create_training_setup


class FactoryTester:
    """Small smoke-test runner for setup helpers."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results: Dict[str, bool] = {}

    def test_agent_factory(self) -> bool:
        try:
            register_builtin_agents()
            available = set(AgentFactory.available_agents())
            expected = {"ftg", "pure_pursuit", "stanley", "hybrid_pp_ftg"}
            missing = expected - available
            if missing:
                raise AssertionError(f"missing fixed-policy agents: {sorted(missing)}")
            if self.verbose:
                print(f"AgentFactory fixed-policy agents: {sorted(available)}")
            return True
        except Exception as exc:
            if self.verbose:
                print(f"AgentFactory smoke check failed: {exc}")
            return False

    def test_create_training_setup(self) -> bool:
        try:
            scenario = load_and_expand_scenario("scenarios/ppo.yaml")
            env, agents, reward_strategies = create_training_setup(scenario, mode="train")
            if "car_1" not in agents:
                raise AssertionError("expected FTG defender car_1 to be created")
            if reward_strategies != {}:
                raise AssertionError("reward strategies should be owned by run.py composers")
            if self.verbose:
                print(f"Created {type(env).__name__} with fixed agents: {sorted(agents)}")
            return True
        except Exception as exc:
            if self.verbose:
                print(f"create_training_setup smoke check failed: {exc}")
            return False

    def run_all_tests(self) -> Dict[str, bool]:
        tests = {
            "AgentFactory": self.test_agent_factory,
            "create_training_setup": self.test_create_training_setup,
        }
        self.results = {name: fn() for name, fn in tests.items()}
        return self.results

    def print_summary(self) -> None:
        passed = sum(1 for result in self.results.values() if result)
        total = len(self.results)
        for name, result in self.results.items():
            print(f"{name:24} {'PASS' if result else 'FAIL'}")
        print(f"OVERALL                  {passed}/{total}")


if __name__ == "__main__":
    tester = FactoryTester(verbose=True)
    results = tester.run_all_tests()
    tester.print_summary()
    sys.exit(0 if all(results.values()) else 1)
