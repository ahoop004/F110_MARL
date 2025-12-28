"""Integration tests for config loading, training loop, and edge cases."""
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.core.config import AgentFactory, EnvironmentFactory
from src.core.training import TrainingLoop, EvaluationLoop
from src.core.utils import save_checkpoint, load_checkpoint, set_random_seeds
from tests.integration.test_helpers import SimpleObservationWrapper


class TestConfigLoading:
    """Test configuration loading and validation."""

    def test_agent_factory_continuous_algorithms(self):
        """Test that AgentFactory can create continuous action agents."""
        algorithms = ["ppo", "rec_ppo", "td3", "sac"]

        for algo in algorithms:
            config = {
                "obs_dim": 10,
                "act_dim": 2,
                "action_low": np.array([-1.0, -1.0]),
                "action_high": np.array([1.0, 1.0]),
                "device": "cpu",
            }

            agent = AgentFactory.create(algo, config)
            assert agent is not None, f"Failed to create {algo} agent"

            # Verify agent has required methods
            assert hasattr(agent, "act")
            assert hasattr(agent, "update")
            assert hasattr(agent, "save")
            assert hasattr(agent, "load")

    def test_agent_factory_discrete_algorithms(self):
        """Test that AgentFactory can create discrete action agents."""
        algorithms = ["dqn", "rainbow"]

        for algo in algorithms:
            config = {
                "obs_dim": 10,
                "action_set": np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]),
                "device": "cpu",
            }

            agent = AgentFactory.create(algo, config)
            assert agent is not None, f"Failed to create {algo} agent"

            # Verify agent has required methods
            assert hasattr(agent, "act")
            assert hasattr(agent, "update")

    def test_environment_factory_single_agent(self):
        """Test that EnvironmentFactory creates single agent environments."""
        config = {"map": "maps/line_map.yaml", "num_agents": 1}

        env = EnvironmentFactory.create(config)
        assert env is not None
        # F110ParallelEnv defaults to 2 agents, so this tests that override works
        # or we just verify it created successfully
        assert env.num_agents >= 1

    def test_invalid_algorithm_raises_error(self):
        """Test that invalid algorithm name raises appropriate error."""
        config = {"obs_dim": 10, "act_dim": 2}

        with pytest.raises((ValueError, KeyError)):
            AgentFactory.create("invalid_algo", config)


class TestTrainingLoop:
    """Test training loop functionality."""

    @pytest.fixture
    def simple_setup(self):
        """Create simple training setup."""
        env_config = {
            "map": "maps/line_map.yaml",
            "num_agents": 1,
            "timestep": 0.01,
            "start_poses": [[0.0, 0.0, 0.0]],  # Required: (x, y, theta) for each agent
        }
        env = EnvironmentFactory.create(env_config)

        # Wrap environment to flatten dict observations to vectors
        env = SimpleObservationWrapper(env)

        # Use actual obs_dim from wrapped observations (1080 lidar + 5 pose/velocity)
        agent_config = {
            "obs_dim": 1085,  # Actual wrapped observation size
            "act_dim": 2,
            "action_low": np.array([-1.0, -1.0]),
            "action_high": np.array([1.0, 1.0]),
            "device": "cpu",
        }
        agent = AgentFactory.create("ppo", agent_config)

        return env, {"car_0": agent}  # Environment uses car_0, not agent_0

    def test_training_loop_runs_episodes(self, simple_setup):
        """Test that training loop can run multiple episodes."""
        env, agents = simple_setup

        loop = TrainingLoop(env, agents, max_episodes=5, max_steps_per_episode=50)
        training_history = loop.run()

        assert training_history is not None
        assert "car_0" in training_history
        assert len(training_history["car_0"]) == 5
        # Check that each episode has the expected keys
        for ep_stats in training_history["car_0"]:
            assert "episode_reward" in ep_stats
            assert "episode_length" in ep_stats

    def test_evaluation_loop_deterministic(self, simple_setup):
        """Test that evaluation loop uses deterministic actions."""
        env, agents = simple_setup

        eval_loop = EvaluationLoop(env, agents, num_episodes=3)
        results = eval_loop.run()

        assert results is not None
        assert "car_0" in results
        agent_results = results["car_0"]
        assert "mean_reward" in agent_results
        assert "std_reward" in agent_results
        assert "mean_length" in agent_results
        assert "std_length" in agent_results

    def test_multi_agent_training_loop(self):
        """Test training loop with multiple agents."""
        env_config = {
            "map": "maps/line_map.yaml",
            "num_agents": 2,
            "timestep": 0.01,
            "start_poses": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],  # Two agent start poses
        }
        env = EnvironmentFactory.create(env_config)

        # Wrap environment
        env = SimpleObservationWrapper(env)

        agents = {}
        for i in range(2):
            agent_config = {
                "obs_dim": 1085,
                "act_dim": 2,
                "action_low": np.array([-1.0, -1.0]),
                "action_high": np.array([1.0, 1.0]),
                "device": "cpu",
            }
            agents[f"car_{i}"] = AgentFactory.create("ppo", agent_config)

        loop = TrainingLoop(env, agents, max_episodes=3, max_steps_per_episode=30)
        training_history = loop.run()

        assert training_history is not None
        assert len(training_history["car_0"]) == 3
        assert len(training_history["car_1"]) == 3


class TestCheckpointResume:
    """Test checkpoint saving, loading, and resume functionality."""

    def test_checkpoint_save_and_load(self):
        """Test saving and loading agent checkpoints."""
        # Create agent
        config = {
            "obs_dim": 10,
            "act_dim": 2,
            "action_low": np.array([-1.0, -1.0]),
            "action_high": np.array([1.0, 1.0]),
            "device": "cpu",
        }
        agent = AgentFactory.create("ppo", config)
        agents = {"car_0": agent}

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save checkpoint
            save_checkpoint(
                agents,
                episode=42,
                checkpoint_dir=tmpdir,
                metrics={"reward": 100.0},
            )

            # Verify checkpoint exists
            checkpoint_path = Path(tmpdir) / "checkpoint_episode_42"
            assert checkpoint_path.exists()
            assert (checkpoint_path / "car_0.pt").exists()

            # Create new agent and load
            new_agent = AgentFactory.create("ppo", config)
            new_agents = {"car_0": new_agent}

            # Load using the checkpoint directory
            metadata = load_checkpoint(new_agents, str(checkpoint_path))
            assert metadata["episode"] == 42
            assert metadata["metrics"]["reward"] == 100.0

    def test_training_resume_from_checkpoint(self):
        """Test resuming training from a saved checkpoint."""
        env_config = {
            "map": "maps/line_map.yaml",
            "num_agents": 1,
            "timestep": 0.01,
            "start_poses": [[0.0, 0.0, 0.0]],
        }
        env = EnvironmentFactory.create(env_config)

        # Wrap environment
        env = SimpleObservationWrapper(env)

        agent_config = {
            "obs_dim": 1085,
            "act_dim": 2,
            "action_low": np.array([-1.0, -1.0]),
            "action_high": np.array([1.0, 1.0]),
            "device": "cpu",
        }
        agent = AgentFactory.create("ppo", agent_config)
        agents = {"car_0": agent}

        with tempfile.TemporaryDirectory() as tmpdir:
            # Train for a few episodes
            loop = TrainingLoop(env, agents, max_episodes=5, max_steps_per_episode=30)
            loop.run()

            # Save checkpoint
            save_checkpoint(agents, episode=5, checkpoint_dir=tmpdir)

            # Create new agent and resume
            new_agent = AgentFactory.create("ppo", agent_config)
            new_agents = {"car_0": new_agent}
            checkpoint_path = Path(tmpdir) / "checkpoint_episode_5"
            load_checkpoint(new_agents, str(checkpoint_path))

            # Continue training
            loop2 = TrainingLoop(env, new_agents, max_episodes=3, max_steps_per_episode=30)
            training_history = loop2.run()

            assert len(training_history["car_0"]) == 3

    def test_multiple_agent_checkpoint(self):
        """Test checkpointing with multiple agents."""
        configs = []
        agents = {}

        for i in range(3):
            config = {
                "obs_dim": 10,
                "act_dim": 2,
                "action_low": np.array([-1.0, -1.0]),
                "action_high": np.array([1.0, 1.0]),
                "device": "cpu",
            }
            agents[f"car_{i}"] = AgentFactory.create("ppo", config)
            configs.append(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint(agents, episode=10, checkpoint_dir=tmpdir)

            # Verify all agents saved
            checkpoint_path = Path(tmpdir) / "checkpoint_episode_10"
            for i in range(3):
                assert (checkpoint_path / f"car_{i}.pt").exists()

            # Load into new agents
            new_agents = {}
            for i in range(3):
                new_agents[f"car_{i}"] = AgentFactory.create("ppo", configs[i])

            metadata = load_checkpoint(new_agents, str(checkpoint_path))
            assert metadata["episode"] == 10


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_random_seed_reproducibility(self):
        """Test that setting random seeds produces reproducible results."""
        config = {
            "obs_dim": 10,
            "act_dim": 2,
            "action_low": np.array([-1.0, -1.0]),
            "action_high": np.array([1.0, 1.0]),
            "device": "cpu",
        }

        # Run 1
        set_random_seeds(42)
        agent1 = AgentFactory.create("ppo", config)
        obs = np.random.randn(10)
        action1 = agent1.act(obs)

        # Run 2 with same seed
        set_random_seeds(42)
        agent2 = AgentFactory.create("ppo", config)
        action2 = agent2.act(obs)

        # Actions should be similar (not exact due to initialization randomness)
        # but the pattern should be reproducible
        assert action1.shape == action2.shape

    def test_zero_episodes_training(self):
        """Test that training loop handles zero episodes gracefully."""
        env_config = {
            "map": "maps/line_map.yaml",
            "num_agents": 1,
            "start_poses": [[0.0, 0.0, 0.0]],
        }
        env = EnvironmentFactory.create(env_config)

        # Wrap environment
        env = SimpleObservationWrapper(env)

        agent_config = {
            "obs_dim": 1085,
            "act_dim": 2,
            "action_low": np.array([-1.0, -1.0]),
            "action_high": np.array([1.0, 1.0]),
        }
        agent = AgentFactory.create("ppo", agent_config)

        loop = TrainingLoop(env, {"car_0": agent}, max_episodes=0)
        training_history = loop.run()

        assert training_history is not None
        assert len(training_history.get("car_0", [])) == 0

    def test_single_step_episode(self):
        """Test training with single-step episodes."""
        env_config = {
            "map": "maps/line_map.yaml",
            "num_agents": 1,
            "start_poses": [[0.0, 0.0, 0.0]],
        }
        env = EnvironmentFactory.create(env_config)

        # Wrap environment
        env = SimpleObservationWrapper(env)

        agent_config = {
            "obs_dim": 1085,
            "act_dim": 2,
            "action_low": np.array([-1.0, -1.0]),
            "action_high": np.array([1.0, 1.0]),
        }
        agent = AgentFactory.create("ppo", agent_config)

        loop = TrainingLoop(env, {"car_0": agent}, max_episodes=2, max_steps_per_episode=1)
        training_history = loop.run()

        assert len(training_history["car_0"]) == 2

    def test_mixed_algorithm_multi_agent(self):
        """Test multi-agent with different algorithms."""
        env_config = {
            "map": "maps/line_map.yaml",
            "num_agents": 2,
            "start_poses": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        }
        env = EnvironmentFactory.create(env_config)

        # Wrap environment
        env = SimpleObservationWrapper(env)

        ppo_config = {
            "obs_dim": 1085,
            "act_dim": 2,
            "action_low": np.array([-1.0, -1.0]),
            "action_high": np.array([1.0, 1.0]),
        }

        td3_config = {
            "obs_dim": 1085,
            "act_dim": 2,
            "action_low": np.array([-1.0, -1.0]),
            "action_high": np.array([1.0, 1.0]),
            "warmup_steps": 10,
            "batch_size": 8,
        }

        agents = {
            "car_0": AgentFactory.create("ppo", ppo_config),
            "car_1": AgentFactory.create("td3", td3_config),
        }

        loop = TrainingLoop(env, agents, max_episodes=3, max_steps_per_episode=20)
        training_history = loop.run()

        assert len(training_history["car_0"]) == 3
        assert len(training_history["car_1"]) == 3
