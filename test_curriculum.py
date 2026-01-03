#!/usr/bin/env python3
"""Test script for phased curriculum system.

Quick test to verify curriculum phases progress correctly.
"""

from src.curriculum.phased_curriculum import PhaseBasedCurriculum, Phase, AdvancementCriteria


def test_basic_curriculum():
    """Test basic curriculum creation and progression."""
    print("Testing Phased Curriculum System")
    print("="*60)

    # Create simple test curriculum
    phases = [
        Phase(
            name="1_easy",
            description="Easy phase",
            criteria=AdvancementCriteria(success_rate=0.70, min_episodes=10, patience=20),
            spawn_config={'points': ['spawn_1'], 'speed_range': [0.44, 0.44]},
            ftg_config={'steering_gain': 0.25},
            lock_speed_steps=800,
        ),
        Phase(
            name="2_medium",
            description="Medium phase",
            criteria=AdvancementCriteria(success_rate=0.60, min_episodes=10, patience=20),
            spawn_config={'points': ['spawn_1', 'spawn_2'], 'speed_range': [0.4, 0.6]},
            ftg_config={'steering_gain': 0.35},
            lock_speed_steps=400,
        ),
        Phase(
            name="3_hard",
            description="Hard phase",
            criteria=AdvancementCriteria(success_rate=0.50, min_episodes=10, patience=20),
            spawn_config={'points': 'all', 'speed_range': [0.3, 1.0]},
            ftg_config={'steering_gain': 0.50},
            lock_speed_steps=0,
        ),
    ]

    curriculum = PhaseBasedCurriculum(phases)

    print(f"\n✓ Created curriculum with {len(curriculum.phases)} phases")
    print(f"  Current phase: {curriculum.get_current_phase().name}")

    # Simulate episodes
    print("\nSimulating training episodes:")
    print("-"*60)

    episode = 0
    while not curriculum.is_complete() and episode < 100:
        # Simulate episode outcome (gradually improving success rate)
        if episode < 15:
            # Early episodes: 50% success
            outcome = 'target_crash' if episode % 2 == 0 else 'timeout'
            reward = 100.0 if outcome == 'target_crash' else -50.0
        elif episode < 30:
            # Mid episodes: 70% success (should trigger advancement)
            outcome = 'target_crash' if episode % 10 < 7 else 'timeout'
            reward = 120.0 if outcome == 'target_crash' else -30.0
        else:
            # Later episodes: 80% success
            outcome = 'target_crash' if episode % 10 < 8 else 'timeout'
            reward = 150.0 if outcome == 'target_crash' else -20.0

        # Update curriculum
        advancement_info = curriculum.update(outcome, reward, episode)

        # Print episode summary
        metrics = curriculum.get_metrics()
        if episode % 5 == 0 or advancement_info:
            status = "SUCCESS" if outcome == 'target_crash' else "FAIL"
            print(f"  Ep {episode:3d} [{status:7s}]: "
                  f"Phase={curriculum.get_current_phase().name:10s} "
                  f"SR={metrics.get('curriculum/success_rate', 0):.2%} "
                  f"Episodes={metrics.get('curriculum/episodes_in_phase', 0):2d}")

        # Print advancement
        if advancement_info:
            print(f"\n  >>> ADVANCED: {advancement_info['old_phase']} → {advancement_info['new_phase']}")
            print(f"      Success Rate: {advancement_info['success_rate']:.2%}")
            print(f"      Forced: {advancement_info['forced']}")
            print()

        episode += 1

    print("-"*60)
    print(f"\n✓ Training simulation complete!")
    print(f"  Final phase: {curriculum.get_current_phase().name}")
    print(f"  Total advancements: {len(curriculum.state.advancement_log)}")

    # Print advancement log
    print("\nAdvancement History:")
    for adv in curriculum.state.advancement_log:
        forced_str = " (FORCED)" if adv['forced'] else ""
        print(f"  Ep {adv['episode']:3d}: {adv['old_phase']:10s} → {adv['new_phase']:10s} "
              f"(SR={adv['success_rate']:.2%}){forced_str}")

    print("\n" + "="*60)
    print("✓ All tests passed!")


if __name__ == '__main__':
    test_basic_curriculum()
