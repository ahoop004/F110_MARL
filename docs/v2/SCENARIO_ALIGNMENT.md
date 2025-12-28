# V2 Scenario Alignment Summary

All v2 gaplock scenarios have been aligned to use consistent configurations for fair algorithm comparison.

## Aligned Scenarios

- `gaplock_sac.yaml` - Soft Actor-Critic (SAC) - **Reference Configuration**
- `gaplock_ppo.yaml` - Proximal Policy Optimization (PPO)
- `gaplock_td3.yaml` - Twin Delayed DDPG (TD3)
- `gaplock_rainbow.yaml` - Rainbow DQN (NEW)

## Key Configuration Alignment

### Environment Parameters
All scenarios now use:
- **max_steps**: 2500 (reduced from 5000 in PPO/TD3 for consistency)
- **timestep**: 0.01
- **lidar_beams**: 720
- **vehicle_params**: Agilex Limo parameters (v_max=1.0 m/s)

### Spawn Curriculum
- **lock_speed_steps**: 175 (was 150 in PPO/TD3)
- **spawn_configs**: Consistent theta values (0.100/-0.100 for pinch pockets)
- **stages**: Identical 3-stage curriculum (optimal_fixed → optimal_varied_speed → full_random)

### FTG Defender
- **max_distance**: 12.0 (was 30.0 in PPO/TD3)
- All other FTG parameters aligned

### Reward Configuration

#### Forcing Rewards (Unified Pinch Pockets)
```yaml
forcing:
  enabled: true
  pinch_pockets:
    anchor_forward: 1.20
    anchor_lateral: 0.35  # Adjusted for narrow track
    sigma: 0.55
    weight: 0.80
    # Potential field mode
    peak: 1.0
    floor: -0.5
    power: 2.0
  clearance:
    weight: 0.80
    band_min: 0.30
    band_max: 3.20
    clip: 0.15  # Was 0.25 in PPO/TD3
    time_scaled: true
  turn:
    weight: 2.0
    clip: 0.15  # Was 0.35 in PPO/TD3
    time_scaled: true
```

#### Other Rewards
```yaml
distance:
  enabled: false  # Was true in PPO/TD3

heading:
  enabled: true
  coefficient: 0.08

speed:
  enabled: true
  bonus_coef: 0.05

step_reward: -0.01
```

#### Terminal Rewards
```yaml
terminal:
  target_crash: 60.0
  self_crash: -40.0
  timeout: -90.0  # Was 0.0 in PPO/TD3
```

## Rainbow DQN Specifics

Rainbow DQN is the only discrete action algorithm. Key differences:

### Discrete Action Set
9 discrete actions covering speed × steering combinations:
```python
action_set:
  - [1.0, 0.0]    # Fast straight
  - [1.0, 0.30]   # Fast left
  - [1.0, -0.30]  # Fast right
  - [0.7, 0.0]    # Medium straight
  - [0.7, 0.30]   # Medium left
  - [0.7, -0.30]  # Medium right
  - [0.5, 0.0]    # Slow straight
  - [0.5, 0.30]   # Slow left
  - [0.5, -0.30]  # Slow right
```

### Rainbow Components
All Rainbow DQN improvements enabled:
- **Noisy Networks**: `noisy_layers: true` (instead of epsilon-greedy)
- **Categorical DQN**: `atoms: 51`, `v_min: -100`, `v_max: 100`
- **N-step Learning**: `n_step: 3`
- **Prioritized Experience Replay**: `use_per: true`
- **Hard Target Updates**: `target_update_interval: 1000`

### Hyperparameters
```yaml
lr: 0.0005
gamma: 0.995
hidden_dims: [512, 256, 128]
buffer_size: 1000000
batch_size: 256
learning_starts: 10000
max_grad_norm: 10.0
```

## Algorithm-Specific Parameters

### SAC
```yaml
lr_actor: 0.0003
lr_critic: 0.0003
lr_alpha: 0.0003
tau: 0.005
alpha: 0.2
target_entropy: -1.0
```

### PPO
```yaml
lr: 0.0005
gae_lambda: 0.95
clip_epsilon: 0.2
entropy_coef: 0.01
value_loss_coef: 0.5
max_grad_norm: 0.5
n_epochs: 10
```

### TD3
```yaml
lr_actor: 0.0003
lr_critic: 0.0003
tau: 0.005
policy_noise: 0.2
noise_clip: 0.5
policy_delay: 2
exploration_noise: 0.1
```

## Changes from Previous Configuration

### PPO & TD3 Updated
- ✓ Reduced max_steps: 5000 → 2500
- ✓ Increased lock_speed_steps: 150 → 175
- ✓ Updated spawn theta values: 0.0 → 0.100/-0.100
- ✓ Reduced FTG max_distance: 30.0 → 12.0
- ✓ Changed timeout penalty: 0.0 → -90.0
- ✓ Updated forcing params: anchor_lateral 0.3→0.35, sigma 0.45→0.55
- ✓ Updated forcing peak/floor: 0.6/-0.25 → 1.0/-0.5
- ✓ Reduced forcing clips: clearance 0.25→0.15, turn 0.35→0.15
- ✓ Disabled distance reward (was enabled)

### SAC Minor Updates
- ✓ Changed render: true → false (for batch training)

## Validation

All scenarios tested for:
- ✓ YAML syntax validity
- ✓ Agent creation (Rainbow DQN verified)
- ✓ Configuration consistency
- ✓ Ready for comparative experiments

## Running Experiments

Each scenario can now be run with:
```bash
python3 run_v2.py --scenario scenarios/v2/gaplock_sac.yaml
python3 run_v2.py --scenario scenarios/v2/gaplock_ppo.yaml
python3 run_v2.py --scenario scenarios/v2/gaplock_td3.yaml
python3 run_v2.py --scenario scenarios/v2/gaplock_rainbow.yaml
```

Results should now be directly comparable as all scenarios use identical:
- Environment dynamics
- Reward shaping
- Spawn curriculum
- Terminal conditions
- Defender behavior
