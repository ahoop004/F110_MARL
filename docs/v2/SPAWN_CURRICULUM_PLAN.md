# Spawn Curriculum Implementation Plan

## Goal

Implement curriculum learning for spawn points and start speeds, progressively increasing difficulty as the agent improves.

## Curriculum Progression

### Phase 1: Fixed Optimal Spawns (Easy)
- **Spawn Points**: Fixed positions near pinch pockets (1.2m ahead, ±0.7m lateral)
- **Start Speed**: Same for both agents (e.g., 0.5 m/s)
- **Goal**: Agent learns basic pinch pocket positioning in ideal scenarios

### Phase 2: Speed Variation (Medium)
- **Spawn Points**: Keep optimal positions
- **Start Speed**: Randomized for both agents (0.3-1.0 m/s range)
- **Goal**: Agent learns to handle different relative velocities

### Phase 3: Position Randomization (Hard)
- **Spawn Points**: Random from full track spawn pool
- **Start Speed**: Randomized
- **Goal**: Agent generalizes to any initial configuration

## Architecture

### 1. Spawn Point Configuration

```python
# scenarios/v2/gaplock_sac.yaml
environment:
  spawn_curriculum:
    enabled: true

    # Success tracking
    window: 200               # Episodes to track for success rate
    activation_samples: 50    # Min episodes before enabling curriculum
    min_episode: 100          # Min episode before first transition

    # Transition thresholds
    enable_patience: 5        # Consecutive successes to advance
    disable_patience: 3       # Consecutive failures to regress
    cooldown: 20              # Episodes between transitions

    # Stages
    stages:
      # Stage 0: Fixed optimal (baseline)
      - name: "optimal_fixed"
        spawn_points: [spawn_pinch_right, spawn_pinch_left]
        speed_range: [0.5, 0.5]  # Fixed speed
        enable_rate: 0.70        # Advance if success >= 70%

      # Stage 1: Speed variation
      - name: "optimal_varied_speed"
        spawn_points: [spawn_pinch_right, spawn_pinch_left, spawn_approach]
        speed_range: [0.3, 1.0]  # Randomized speed
        enable_rate: 0.65
        disable_rate: 0.50       # Regress if success < 50%

      # Stage 2: Full randomization
      - name: "full_random"
        spawn_points: "all"      # Use all available spawn points
        speed_range: [0.3, 1.0]
        disable_rate: 0.45
```

### 2. Core Components

#### SpawnCurriculumManager
Location: `v2/core/spawn_curriculum.py`

```python
class SpawnCurriculumManager:
    """Manages progressive spawn difficulty based on success rate."""

    def __init__(self, config: dict, spawn_points: dict, logger):
        self.stages = self._parse_stages(config['stages'])
        self.current_stage_idx = 0

        # Success tracking
        self.window = config.get('window', 200)
        self.success_history = deque(maxlen=self.window)
        self.stage_histories = {}  # Per-stage success tracking

        # Thresholds
        self.enable_patience = config.get('enable_patience', 5)
        self.disable_patience = config.get('disable_patience', 3)
        self.cooldown = config.get('cooldown', 20)

        # State
        self.promote_streak = 0
        self.regress_streak = 0
        self.last_transition_episode = 0

    def observe(self, episode: int, success: bool) -> dict:
        """Record episode outcome and check for stage transitions."""
        self.success_history.append(1.0 if success else 0.0)

        # Calculate success rate
        success_rate = self._success_rate()

        # Check for stage advancement
        if self._can_advance(episode, success_rate):
            self._advance_stage(episode)

        # Check for stage regression
        if self._can_regress(episode, success_rate):
            self._regress_stage(episode)

        return self._get_state()

    def sample_spawn(self) -> tuple[dict, np.ndarray]:
        """Sample spawn points and speeds for current stage."""
        stage = self.stages[self.current_stage_idx]

        # Sample spawn points
        spawn_names = self._sample_spawn_points(stage)
        spawn_poses = self._get_spawn_poses(spawn_names)

        # Sample start speeds
        speeds = self._sample_speeds(stage, len(spawn_poses))

        return spawn_names, spawn_poses, speeds
```

#### SpawnStage
```python
@dataclass
class SpawnStage:
    """Configuration for a curriculum stage."""
    name: str
    spawn_points: list[str] | str  # List of spawn IDs or "all"
    speed_range: tuple[float, float]
    enable_rate: float
    disable_rate: float = 0.0
    enable_patience: int | None = None
    disable_patience: int | None = None
```

### 3. Map Spawn Point Definitions

Define pinch-pocket-aligned spawn points in map YAML:

```yaml
# maps/line2/line2.yaml (enhanced)
spawn_points:
  # Baseline spawns (near optimal pinch pockets)
  spawn_pinch_right:
    attacker: [x, y, theta]  # 1.2m ahead, -0.7m right of defender
    defender: [x, y, theta]
    metadata:
      curriculum_stage: 0
      difficulty: "easy"

  spawn_pinch_left:
    attacker: [x, y, theta]  # 1.2m ahead, +0.7m left
    defender: [x, y, theta]
    metadata:
      curriculum_stage: 0
      difficulty: "easy"

  spawn_approach:
    attacker: [x, y, theta]  # Slightly behind, needs to catch up
    defender: [x, y, theta]
    metadata:
      curriculum_stage: 1
      difficulty: "medium"

  # Random pool (harder scenarios)
  spawn_1:
    attacker: [x, y, theta]
    defender: [x, y, theta]
    metadata:
      curriculum_stage: 2
      difficulty: "hard"
      random_pool: true
```

### 4. Integration with Training Loop

```python
# v2/core/enhanced_training.py

class EnhancedTrainingLoop:
    def __init__(self, ...):
        # Initialize spawn curriculum if enabled
        spawn_config = scenario['environment'].get('spawn_curriculum')
        if spawn_config and spawn_config.get('enabled'):
            self.spawn_manager = SpawnCurriculumManager(
                config=spawn_config,
                spawn_points=env.map_data.spawn_points,
                logger=console_logger
            )
        else:
            self.spawn_manager = None

    def train_episode(self, episode_idx: int) -> RolloutResult:
        # Sample spawn configuration
        if self.spawn_manager:
            spawn_names, spawn_poses, speeds = self.spawn_manager.sample_spawn()
            obs, info = env.reset(options={
                'poses': spawn_poses,
                'velocities': speeds
            })
        else:
            obs, info = env.reset()

        # ... run episode ...

        # Update curriculum based on outcome
        if self.spawn_manager:
            spawn_state = self.spawn_manager.observe(episode_idx, success)

            # Log curriculum metrics
            if spawn_state['changed']:
                console_logger.print_info(
                    f"Spawn curriculum: {spawn_state['stage']} "
                    f"(success rate: {spawn_state['success_rate']:.2%})"
                )
```

## Improvements Over V1

### 1. **Pinch-Pocket Alignment**
- V1 had generic spawn points
- V2 explicitly defines spawn points near Gaussian pinch pockets
- Makes Phase 1 much more effective for learning

### 2. **Speed Curriculum**
- V1 only varied positions
- V2 adds progressive speed variation
- Better handles velocity-dependent behaviors

### 3. **Clearer Stage Definitions**
- V1 used metadata tags that could be confusing
- V2 has explicit stage configurations in YAML
- Easier to tune and understand

### 4. **Per-Stage Thresholds**
- V1 had global thresholds
- V2 allows per-stage enable/disable rates
- More flexible curriculum design

### 5. **Integration with Heatmap**
- Can visualize pinch pockets during training
- Press H to see reward field and understand spawn placement
- Helps tune spawn point positions

## Implementation Steps

### Phase 1: Core Infrastructure
1. Create `v2/core/spawn_curriculum.py` with `SpawnCurriculumManager`
2. Add spawn point metadata to map YAML files
3. Calculate pinch-pocket spawn positions from reward parameters

### Phase 2: Integration
4. Add spawn curriculum config to scenario YAML schema
5. Integrate with `EnhancedTrainingLoop`
6. Add curriculum state to logging/WandB

### Phase 3: Spawn Point Generation
7. Create helper to auto-generate pinch pocket spawn points
8. Use forcing reward parameters (anchor_forward, anchor_lateral)
9. Validate spawn points don't collide with walls

### Phase 4: Testing
10. Test each stage independently
11. Verify stage transitions trigger correctly
12. Measure impact on learning efficiency

## Example Workflow

```bash
# 1. Train with curriculum
python3 v2/run.py scenarios/v2/gaplock_sac.yaml

# Episode 0-100: Stage 0 (optimal_fixed)
#   - Spawns near pinch pockets, fixed speed
#   - Success rate: 75% → Advance to Stage 1

# Episode 101-300: Stage 1 (optimal_varied_speed)
#   - Same spawns, randomized speeds
#   - Success rate: 68% → Advance to Stage 2

# Episode 301+: Stage 2 (full_random)
#   - Random spawns, randomized speeds
#   - Success rate: 55% → Full generalization

# 2. Visualize during training
#   - Press H to see pinch pocket heatmap
#   - Verify spawns align with green hotspots
```

## Success Metrics

- **Stage 0 Success**: Agent consistently reaches pinch pockets (>70%)
- **Stage 1 Success**: Agent handles speed variations (>65%)
- **Stage 2 Success**: Agent generalizes to all spawns (>50%)
- **Learning Efficiency**: Faster convergence than random spawns from start
- **Final Performance**: Better generalization than fixed spawns

## Future Enhancements

1. **Adaptive Thresholds**: Automatically tune enable/disable rates
2. **Difficulty Scoring**: Rank spawn points by historical success rate
3. **Targeted Practice**: Increase sampling of difficult spawn pairs
4. **Multi-Agent Curriculum**: Different curricula for attacker/defender
5. **Transfer Learning**: Use curriculum from one map on another

---

**Last Updated**: December 26, 2024
