# Weights & Biases Integration - Verification

## Summary
✅ **Weights & Biases logging is fully configured and wired in all v2 scenario files.**

## YAML Configuration

All v2 gaplock scenario files have wandb configuration:

### ✅ gaplock_sac.yaml
```yaml
wandb:
  enabled: false  # Set to true to enable
  project: f110-gaplock
  entity: ahoop004-old-dominion-university
  tags: [sac, comparison, gaplock]
  notes: "SAC comparison for gaplock adversarial task"
```

### ✅ gaplock_ppo.yaml
```yaml
wandb:
  enabled: false
  project: f110-gaplock
  entity: ahoop004-old-dominion-university
  tags: [ppo, baseline, gaplock]
  notes: "PPO baseline for gaplock adversarial task"
```

### ✅ gaplock_td3.yaml
```yaml
wandb:
  enabled: false
  project: f110-gaplock
  entity: ahoop004-old-dominion-university
  tags: [td3, comparison, gaplock]
  notes: "TD3 comparison for gaplock adversarial task"
```

### ✅ gaplock_limo.yaml
```yaml
wandb:
  enabled: false
  project: f110-gaplock
  entity: ahoop004-old-dominion-university
  tags: [sac, limo, realistic]
  notes: "SAC with Agilex Limo realistic parameters (v_max=1.0 m/s)"
```

### ✅ gaplock_simple.yaml
```yaml
wandb:
  enabled: false
  project: f110-gaplock
  entity: ahoop004-old-dominion-university
  tags: [ppo, simple_rewards, ablation]
  notes: "PPO with simplified rewards (ablation study)"
```

### ✅ gaplock_custom.yaml
```yaml
wandb:
  enabled: false
  project: marl-f110
  entity: ahoop004-old-dominion-university
  group: gaplock_custom
  tags: [custom, experimental]
```

## Code Wiring Verification

### 1. WandbLogger Implementation ✅
**File:** `src/loggers/wandb_logger.py`

- ✅ Full implementation with init, logging, and cleanup
- ✅ Supports nested config flattening
- ✅ Episode-level metrics logging
- ✅ Rolling statistics logging
- ✅ Component statistics logging
- ✅ Custom metrics logging
- ✅ Run metadata tracking (ID, name, URL)

### 2. Training Loop Integration ✅
**File:** `src/core/enhanced_training.py`

**Wandb logger is passed to EnhancedTrainingLoop:**
```python
def __init__(
    self,
    env,
    agents: Dict[str, Any],
    agent_rewards: Dict[str, RewardStrategy],
    wandb_logger: Optional[WandbLogger] = None,  # ✅ Accepts wandb_logger
    ...
):
    self.wandb_logger = wandb_logger  # ✅ Stores logger
```

**Logs trainer statistics every episode:**
```python
if self.wandb_logger:
    log_dict = {}
    for stat_name, stat_value in update_stats.items():
        log_dict[f'trainer/{agent_id}/{stat_name}'] = stat_value
    self.wandb_logger.log_metrics(log_dict, step=episode_num)
```

**Logs episode metrics with rolling statistics:**
```python
if self.wandb_logger:
    self.wandb_logger.log_episode(
        episode=episode_num,
        metrics=metrics,
        rolling_stats=rolling_stats,
        extra={'agent_id': agent_id},
    )
```

**Logs curriculum progression:**
```python
if self.wandb_logger:
    self.wandb_logger.log_metrics({
        'curriculum/stage': curriculum_state['stage'],
        'curriculum/stage_index': curriculum_state['stage_index'],
        'curriculum/success_rate': curriculum_state['success_rate'] or 0.0,
        'curriculum/stage_success_rate': curriculum_state['stage_success_rate'] or 0.0,
    }, step=episode_num)
```

### 3. Main Training Script ✅
**File:** `run_v2.py`

**Loads wandb config from scenario:**
```python
def initialize_loggers(scenario: dict, args, run_id: str = None):
    wandb_config = scenario.get('wandb', {})
    wandb_enabled = wandb_config.get('enabled', False)

    if wandb_enabled:
        wandb_logger = WandbLogger(
            project=wandb_config.get('project', 'f110-marl'),
            name=wandb_config.get('name', scenario['experiment']['name']),
            config=scenario,  # ✅ Passes full scenario as config
            tags=wandb_config.get('tags', []),
            group=wandb_config.get('group', None),
            entity=wandb_config.get('entity', None),
            notes=wandb_config.get('notes', None),
            mode=wandb_config.get('mode', 'online'),
            run_id=run_id,
        )
```

**Passes to training loop:**
```python
training_loop = EnhancedTrainingLoop(
    env=env,
    agents=agents,
    agent_rewards=reward_strategies,
    wandb_logger=wandb_logger,  # ✅ Passes logger
    ...
)
```

**Cleanup on finish:**
```python
if wandb_logger:
    console_logger.print_info("Finishing W&B run...")
    wandb_logger.finish()  # ✅ Proper cleanup
```

### 4. Metrics Logging ✅
**File:** `src/metrics/tracker.py`

**Reward components are logged with `reward/` prefix:**
```python
def to_dict(self) -> Dict:
    return {
        'episode': self.episode,
        'outcome': self.outcome.value,
        'total_reward': self.total_reward,
        'steps': self.steps,
        'success': self.success,
        **{f'reward/{k}': v for k, v in self.reward_components.items()},  # ✅ Logs all reward components
    }
```

## What Gets Logged to W&B

### Episode Metrics
- `episode` - Episode number
- `outcome` - Episode outcome (target_crash, self_crash, timeout, etc.)
- `total_reward` - Total accumulated reward
- `steps` - Number of steps taken
- `success` - Boolean success flag

### Reward Components (with `reward/` prefix)
- `reward/distance/near` - Distance reward when near target
- `reward/distance/far` - Distance penalty when far from target
- `reward/distance/gradient` - Distance gradient shaping
- `reward/forcing/pinch` - Pinch pocket Gaussian rewards
- `reward/forcing/clearance` - Clearance reduction rewards
- `reward/forcing/turn` - Turn shaping rewards
- `reward/forcing/potential_field` - **NEW!** Potential field rewards
- `reward/heading/alignment` - Heading alignment bonus
- `reward/speed/bonus` - Speed bonus
- `reward/penalties/step` - Step penalty
- `reward/terminal/success` - Terminal success reward
- `reward/terminal/timeout` - Terminal timeout penalty
- `reward/terminal/self_crash` - Terminal self-crash penalty

### Rolling Statistics (with `rolling/` prefix)
- `rolling/success_rate` - Rolling success rate
- `rolling/avg_reward` - Rolling average reward
- `rolling/avg_steps` - Rolling average steps
- `rolling/outcomes/{outcome}` - Count of each outcome type
- `rolling/outcome_rates/{outcome}` - Rate of each outcome type

### Trainer Statistics (with `trainer/{agent_id}/` prefix)
- PPO: `policy_loss`, `value_loss`, `entropy`, `approx_kl`, etc.
- SAC: `actor_loss`, `critic_loss`, `alpha_loss`, `alpha`, etc.
- TD3: `actor_loss`, `critic_loss`, etc.

### Curriculum Statistics (with `curriculum/` prefix)
- `curriculum/stage` - Current curriculum stage name
- `curriculum/stage_index` - Current stage index
- `curriculum/success_rate` - Overall success rate
- `curriculum/stage_success_rate` - Success rate in current stage

## How to Enable W&B Logging

### Option 1: Set in YAML
Edit your scenario file (e.g., `gaplock_sac.yaml`):
```yaml
wandb:
  enabled: true  # Change from false to true
  project: f110-gaplock
  entity: ahoop004-old-dominion-university
  tags: [sac, comparison, gaplock]
  notes: "SAC comparison for gaplock adversarial task"
```

### Option 2: Use CLI flag
```bash
python run_v2.py --scenario scenarios/v2/gaplock_sac.yaml --wandb
```

### Option 3: Set W&B API Key
If you haven't already, set your W&B API key:
```bash
export WANDB_API_KEY=your_api_key_here
# Or login interactively:
wandb login
```

## Configuration Parameters

### Required
- `enabled` - Enable/disable W&B logging (default: false)
- `project` - W&B project name

### Optional
- `entity` - W&B entity (username or team)
- `name` - Custom run name (default: uses scenario name)
- `tags` - List of tags for organizing runs
- `group` - Group name for organizing related runs
- `notes` - Description of the experiment
- `mode` - "online", "offline", or "disabled" (default: "online")

## Example W&B Dashboard Metrics

After training with W&B enabled, you'll see:

```
📊 Episode Metrics
  - episode: 0, 1, 2, ...
  - success: true/false
  - total_reward: 85.2, 92.1, ...
  - steps: 450, 523, ...

🎯 Reward Components
  - reward/forcing/potential_field: 0.32, -0.15, ...
  - reward/terminal/success: 60.0 (on success)
  - reward/terminal/timeout: -90.0 (on timeout)
  - reward/distance/near: 0.12, ...

📈 Rolling Statistics (window=100)
  - rolling/success_rate: 0.75, 0.78, ...
  - rolling/avg_reward: 85.2, 87.3, ...

🎓 Curriculum Progression
  - curriculum/stage: "optimal_fixed", "optimal_varied_speed", ...
  - curriculum/stage_index: 0, 1, 2
  - curriculum/success_rate: 0.70, 0.65, ...
```

## Verification Checklist

- ✅ All v2 scenario YAML files have `wandb` section
- ✅ `WandbLogger` class is fully implemented
- ✅ `run_v2.py` loads wandb config from YAML
- ✅ `run_v2.py` initializes `WandbLogger` when enabled
- ✅ `run_v2.py` passes `wandb_logger` to `EnhancedTrainingLoop`
- ✅ `EnhancedTrainingLoop` accepts and stores `wandb_logger`
- ✅ Episode metrics are logged to W&B
- ✅ Reward components are logged with `reward/` prefix
- ✅ Rolling statistics are logged with `rolling/` prefix
- ✅ Trainer statistics are logged with `trainer/` prefix
- ✅ Curriculum metrics are logged with `curriculum/` prefix
- ✅ W&B run is properly finished on cleanup
- ✅ CLI flags `--wandb` and `--no-wandb` override YAML config
- ✅ All reward components (including new `potential_field`) will be logged

## Status

🎉 **W&B integration is complete and ready to use!**

Simply set `enabled: true` in your scenario YAML or use the `--wandb` flag when running training.
