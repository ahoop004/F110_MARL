# Getting Started with Phase 8 Implementation

## Current Status

✅ **Architecture designed** - All components planned
✅ **Phase 7 complete** - Documentation, tests passing (69 tests)
✅ **Ready to implement** - Phase 8 breakdown complete

---

## Documentation Structure

### Master Documents (Read These First)

1. **[DESIGN_MASTER.md](DESIGN_MASTER.md)** - Complete overview
   - Key decisions
   - What's being removed/added
   - Architecture
   - Implementation plan

2. **[../REFACTOR_TODO.md](../REFACTOR_TODO.md)** - Implementation checklist
   - Phase 8 detailed breakdown (10 subtasks)
   - Time estimates
   - Success criteria

### Detailed Design Documents

3. **[ARCHITECTURE.md](ARCHITECTURE.md)** - System architecture
   - Component descriptions
   - Data flow
   - User workflow

4. **[V1_TO_V2_TRANSITION.md](V1_TO_V2_TRANSITION.md)** - Transition plan
   - What V1 has (7 layers, 5,000 lines)
   - What V2 will have (3 layers, 800 lines)
   - Detailed component breakdown

5. **[rewards/DESIGN.md](rewards/DESIGN.md)** - Reward system design
   - Component architecture
   - Presets
   - Implementation plan

6. **[OBSERVATIONS_REVISED.md](OBSERVATIONS_REVISED.md)** - Observation system
   - V1 configuration (738 dims)
   - Preset definitions
   - Integration plan

---

## Phase 8 Overview

**Goal**: Complete v2 pipeline
**Timeline**: 1-2 weeks (34 hours)
**Components**: 10 subtasks

### Quick Summary

| # | Task | Hours | Status |
|---|------|-------|--------|
| 8.1 | Rewards | 10 | ⬜ Not started |
| 8.2 | Metrics | 2 | ⬜ Not started |
| 8.3 | W&B | 2 | ⬜ Not started |
| 8.4 | Console | 2 | ⬜ Not started |
| 8.5 | Observations | 5 | ⬜ Not started |
| 8.6 | Scenarios | 4 | ⬜ Not started |
| 8.7 | CLI | 2 | ⬜ Not started |
| 8.8 | Training Loop | 3 | ⬜ Not started |
| 8.9 | Examples | 2 | ⬜ Not started |
| 8.10 | Docs | 2 | ⬜ Not started |

---

## Recommended Implementation Order

### Week 1: Core Functionality

**Day 1-2: Rewards (10 hrs)**
- Build foundation: protocols, composer, presets
- Implement components: terminal, pressure, distance, heading, speed, penalties, forcing
- Port v1 forcing rewards (complex)
- Tests

**Day 3: Metrics + Observations (7 hrs)**
- Outcome tracking (6 types)
- Rolling statistics
- Observation presets (738 dims)
- Auto-compute obs_dim
- Tests

**Day 4: Scenarios (4 hrs)**
- YAML parser
- Preset expansion
- Config validation
- Tests

### Week 2: Integration & Polish

**Day 5: Logging (4 hrs)**
- W&B integration
- Rich console output
- Tests

**Day 6-7: Integration (7 hrs)**
- CLI interface
- Training loop enhancement
- Wire all components
- End-to-end tests

**Day 8: Examples & Docs (4 hrs)**
- Example scenarios
- Documentation
- Final validation

---

## Starting Point: 8.1 Rewards

### Why Start Here?

1. **Core functionality** - Everything else depends on rewards
2. **Most complex** - Get the hard part done first
3. **Standalone** - Can test independently
4. **Well-defined** - Clear port from v1

### What to Build

**Step 1: Base Infrastructure** (2 hrs)

Create these files:
```
v2/rewards/
├── __init__.py
├── base.py              # Protocols
├── composer.py          # Composition
└── presets.py           # Presets
```

**Step 2: Gaplock Components** (7 hrs)

Create these files:
```
v2/rewards/gaplock/
├── __init__.py
├── gaplock.py           # Main class
├── terminal.py          # Terminal rewards
├── pressure.py          # Pressure shaping
├── distance.py          # Distance shaping
├── heading.py           # Heading alignment
├── speed.py             # Speed bonuses
├── forcing.py           # Forcing rewards (complex!)
└── penalties.py         # Behavior penalties
```

**Step 3: Tests** (1 hr)

Create:
```
tests/
├── test_reward_base.py       # Protocol tests
├── test_reward_composer.py   # Composition tests
└── test_gaplock_rewards.py   # Component tests
```

### Key Files to Reference

**V1 gaplock reward**: [src/f110x/tasks/reward/gaplock.py](../src/f110x/tasks/reward/gaplock.py:1)
- Line 863: Terminal rewards
- Lines 720-817: Pressure system
- Lines 759-776: Distance gradient
- Lines 782-784: Heading reward
- Lines 680-684: Speed bonus
- Lines 960-1212: Forcing rewards (pinch, clearance, turn)
- Line 637: Idle penalty
- Lines 692-704: Brake/reverse penalties

**Design**: [rewards/DESIGN.md](rewards/DESIGN.md:1)
- Complete component breakdown
- Code examples
- Testing strategy

---

## Command to Start

```bash
# Create reward directories
mkdir -p v2/rewards/gaplock

# Create test file
touch tests/test_reward_base.py

# Start with base.py
# See rewards/DESIGN.md for protocol definitions
```

---

## Success Criteria for 8.1 (Rewards)

When 8.1 is complete:

✅ All reward components implemented
✅ Unit tests passing for each component
✅ Integration test (full episode) passing
✅ Reward values match v1 (validated)
✅ Component dict returned (for logging)
✅ Presets defined (gaplock_simple, gaplock_full)

---

## Questions Before Starting?

1. **Implementation approach**: Sequential or parallel tasks?
2. **Testing strategy**: Write tests first or after?
3. **V1 compatibility**: Validate rewards match v1 at each step?
4. **Code style**: Any specific preferences?

---

## Ready to Start?

Once you confirm, I can begin implementing:

**Option A**: Start with 8.1 Rewards (recommended)
- Build foundation first
- Most complex component
- Everything else depends on this

**Option B**: Parallel implementation
- I work on Rewards (8.1)
- You work on Metrics (8.2) or Observations (8.5)
- Faster overall, but requires coordination

**Option C**: Different starting point
- Your choice of which component to start with

Let me know how you want to proceed!
