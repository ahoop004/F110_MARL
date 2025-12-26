"""Tests for episode outcome classification."""

import pytest
from v2.metrics import EpisodeOutcome, determine_outcome


class TestEpisodeOutcome:
    """Test EpisodeOutcome enum."""

    def test_outcome_values(self):
        """Test that all outcome values are defined."""
        assert EpisodeOutcome.TARGET_CRASH.value == "target_crash"
        assert EpisodeOutcome.SELF_CRASH.value == "self_crash"
        assert EpisodeOutcome.COLLISION.value == "collision"
        assert EpisodeOutcome.TIMEOUT.value == "timeout"
        assert EpisodeOutcome.IDLE_STOP.value == "idle_stop"
        assert EpisodeOutcome.TARGET_FINISH.value == "target_finish"

    def test_is_success(self):
        """Test success detection."""
        assert EpisodeOutcome.TARGET_CRASH.is_success()
        assert not EpisodeOutcome.SELF_CRASH.is_success()
        assert not EpisodeOutcome.COLLISION.is_success()
        assert not EpisodeOutcome.TIMEOUT.is_success()
        assert not EpisodeOutcome.IDLE_STOP.is_success()
        assert not EpisodeOutcome.TARGET_FINISH.is_success()

    def test_is_failure(self):
        """Test failure detection."""
        assert not EpisodeOutcome.TARGET_CRASH.is_failure()
        assert EpisodeOutcome.SELF_CRASH.is_failure()
        assert EpisodeOutcome.COLLISION.is_failure()
        assert EpisodeOutcome.TIMEOUT.is_failure()
        assert EpisodeOutcome.IDLE_STOP.is_failure()
        assert EpisodeOutcome.TARGET_FINISH.is_failure()


class TestDetermineOutcome:
    """Test outcome determination logic."""

    def test_target_finish_priority(self):
        """Test that target finish has highest priority."""
        # Even if both crashed, target finish takes precedence
        info = {
            'target_finished': True,
            'collision': True,
            'target_collision': True,
        }
        assert determine_outcome(info) == EpisodeOutcome.TARGET_FINISH

    def test_collision(self):
        """Test collision detection."""
        info = {
            'collision': True,
            'target_collision': True,
        }
        assert determine_outcome(info) == EpisodeOutcome.COLLISION

    def test_target_crash(self):
        """Test target crash (success)."""
        info = {
            'collision': False,
            'target_collision': True,
        }
        assert determine_outcome(info) == EpisodeOutcome.TARGET_CRASH

    def test_self_crash(self):
        """Test self crash (failure)."""
        info = {
            'collision': True,
            'target_collision': False,
        }
        assert determine_outcome(info) == EpisodeOutcome.SELF_CRASH

    def test_idle_stop(self):
        """Test idle stop detection."""
        info = {
            'collision': False,
            'target_collision': False,
            'idle_truncation': True,
        }
        assert determine_outcome(info) == EpisodeOutcome.IDLE_STOP

    def test_timeout(self):
        """Test timeout detection."""
        info = {
            'collision': False,
            'target_collision': False,
        }
        assert determine_outcome(info, truncated=True) == EpisodeOutcome.TIMEOUT

    def test_default_timeout(self):
        """Test default to timeout if no clear outcome."""
        info = {}
        assert determine_outcome(info) == EpisodeOutcome.TIMEOUT

    def test_priority_order(self):
        """Test full priority order."""
        # Priority: finish > collision > target_crash > self_crash > idle > timeout

        # Finish beats everything
        info = {
            'target_finished': True,
            'collision': True,
            'target_collision': True,
            'idle_truncation': True,
        }
        assert determine_outcome(info, truncated=True) == EpisodeOutcome.TARGET_FINISH

        # Collision beats crashes
        info = {
            'collision': True,
            'target_collision': True,
        }
        assert determine_outcome(info) == EpisodeOutcome.COLLISION

        # Target crash beats self crash
        info = {
            'collision': False,
            'target_collision': True,
        }
        assert determine_outcome(info) == EpisodeOutcome.TARGET_CRASH

        # Self crash beats idle
        info = {
            'collision': True,
            'target_collision': False,
            'idle_truncation': True,
        }
        assert determine_outcome(info) == EpisodeOutcome.SELF_CRASH

        # Idle beats timeout
        info = {
            'collision': False,
            'target_collision': False,
            'idle_truncation': True,
        }
        assert determine_outcome(info, truncated=True) == EpisodeOutcome.IDLE_STOP


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
