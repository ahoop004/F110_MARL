# Vehicle and Policy Parameters Reference

## Overview

This document details all configurable parameters for vehicles and policies in the F110 MARL environment, including Agilex Limo specifications.

## Vehicle Dynamics Parameters

### Default (F1/10 Racecar - High Performance)
Used in original F1/10 competition. **NOT realistic for Agilex Limo robot.**

```yaml
vehicle_params:
  # Tire parameters
  mu: 1.0489          # Friction coefficient
  C_Sf: 4.718         # Front cornering stiffness
  C_Sr: 5.4562        # Rear cornering stiffness

  # Geometry
  lf: 0.15875         # Distance from CG to front axle (m)
  lr: 0.17145         # Distance from CG to rear axle (m)
  h: 0.074            # Height of CG (m)
  length: 0.32        # Vehicle length (m)
  width: 0.225        # Vehicle width (m)

  # Dynamics
  m: 3.74             # Mass (kg)
  I: 0.04712          # Moment of inertia (kg*m^2)

  # Control limits
  s_min: -0.4189      # Min steering angle (rad) ≈ -24°
  s_max: 0.4189       # Max steering angle (rad) ≈ +24°
  sv_min: -3.2        # Min steering velocity (rad/s)
  sv_max: 3.2         # Max steering velocity (rad/s)

  # Speed limits
  v_switch: 7.319     # Speed threshold for switching (m/s)
  a_max: 9.51         # Max acceleration (m/s²) - UNREALISTIC
  v_min: -5.0         # Min velocity (m/s) - reverse
  v_max: 20.0         # Max velocity (m/s) - UNREALISTIC (~45 mph)
```

### Agilex Limo (Realistic Robot Platform)
**Use these parameters for realistic sim-to-real transfer.**

Based on Agilex Limo specifications and real-world testing.

```yaml
vehicle_params:
  # Tire parameters (same as F1/10)
  mu: 1.0489
  C_Sf: 4.718
  C_Sr: 5.4562

  # Geometry (Limo-specific)
  lf: 0.15875
  lr: 0.17145
  h: 0.074
  length: 0.32        # Limo wheelbase ≈ 320mm
  width: 0.225        # Limo width ≈ 225mm

  # Dynamics (same as F1/10)
  m: 3.74
  I: 0.04712

  # Control limits (Limo-specific)
  s_min: -0.46        # Min steering angle (rad) ≈ -26.4°
  s_max: 0.46         # Max steering angle (rad) ≈ +26.4°
  sv_min: -3.2        # Min steering velocity (rad/s)
  sv_max: 3.2         # Max steering velocity (rad/s)

  # Speed limits (LIMO-SPECIFIC - CRITICAL FOR REALISM)
  v_switch: 0.8       # Speed threshold (m/s)
  a_max: 2.0          # Max acceleration (m/s²) - realistic for Limo
  v_min: -1.0         # Min velocity (m/s) - slow reverse
  v_max: 1.0          # Max velocity (m/s) - Limo top speed ~1.0-1.5 m/s
```

**Key Differences:**
- `v_max`: 20.0 → 1.0 m/s (20x slower, realistic)
- `a_max`: 9.51 → 2.0 m/s² (more realistic acceleration)
- `s_min/s_max`: -0.4189 → -0.46 rad (slightly wider steering range)

---

## Follow The Gap (FTG) Policy Parameters

### Complete Parameter List

```yaml
ftg_params:
  # Gap detection
  max_distance: 30.0          # LiDAR max range (m)
  window_size: 4              # Smoothing window for gap detection
  bubble_radius: 2            # Safety bubble around obstacles (beam count)
  gap_min_range: 0.65         # Minimum clearance for valid gap (m)

  # Steering
  max_steer: 0.32             # Max steering angle output (rad) ≈ 18.3°
  steering_gain: 0.6          # Steering responsiveness multiplier
  steer_smooth: 0.4           # Steering smoothing factor [0-1]

  # Speed control
  min_speed: 2.0              # Minimum speed (m/s) - F1/10 default
  max_speed: 20.0             # Maximum speed (m/s) - F1/10 default
  steering_speed_scale: 1.0   # Speed scaling based on steering angle
  crawl_steer_ratio: 0.6      # Speed ratio when crawling through tight gaps

  # LiDAR configuration
  fov: 4.71238898             # Field of view (rad) = 270°
  normalized: false           # Whether scan ranges are normalized
  mode: "lidar"               # Observation mode

  # Gap targeting
  target_mode: "farthest"     # "farthest" or "midgap"
  center_bias_gain: 0.0       # Bias toward center gaps
  inside_bias_gain: 0.0       # Bias toward inside of track

  # Disparity extender (obstacle growth)
  use_disparity_extender: true
  disparity_threshold: 0.35   # Disparity detection threshold (m)
  vehicle_width: 0.225        # Vehicle width for clearance (m)
  safety_margin: 0.08         # Additional safety margin (m)

  # No-cutback heuristic
  no_cutback_enabled: true
  cutback_clearance: 0.9      # Min clearance to avoid cutback (m)
  cutback_hold_steps: 8       # Steps to hold after cutback detection

  # Advanced features (usually disabled)
  preview_horizon: 0.0        # Lookahead distance (m)
  preview_samples: 0          # Number of preview samples
  dwa_samples: 0              # Dynamic window approach samples
  dwa_horizon: 0.5            # DWA planning horizon (s)
  dwa_heading_weight: 0.1     # DWA heading weight

  # Enhanced gap scoring (usually disabled)
  enhanced_gap_scoring: false
  gap_score_width_weight: 1.0
  gap_score_clearance_weight: 1.0
  gap_score_center_weight: 0.5
  gap_score_curvature_weight: 0.2

  # U-shape detection (usually disabled)
  u_shape_enabled: false
  u_shape_threshold: 0.5
  u_shape_crawl_speed: 0.3
```

### V1-Compatible FTG Configuration (for Limo)

**Critical: Adjust speed parameters to match Limo vehicle limits.**

```yaml
ftg_params:
  # Core parameters (same as v1)
  max_distance: 30.0
  window_size: 4
  bubble_radius: 2
  max_steer: 0.32
  steering_gain: 0.6
  fov: 4.71238898            # 270°
  normalized: false
  steer_smooth: 0.4
  mode: "lidar"

  # Speed parameters (MATCH LIMO VEHICLE PARAMS)
  min_speed: 0.2             # Reduced for Limo (was 2.0)
  max_speed: 1.0             # Match vehicle v_max for Limo

  # Gap detection (v1 defaults)
  gap_min_range: 0.65
  target_mode: "farthest"
  use_disparity_extender: true
  disparity_threshold: 0.35
  vehicle_width: 0.225
  safety_margin: 0.08
  no_cutback_enabled: true
  cutback_clearance: 0.9
  cutback_hold_steps: 8

  # Disabled features
  center_bias_gain: 0.0
  steering_speed_scale: 1.0
  inside_bias_gain: 0.0
  crawl_steer_ratio: 0.6
  preview_horizon: 0.0
  preview_samples: 0
  dwa_samples: 0
  enhanced_gap_scoring: false
  u_shape_enabled: false
```

---

## Environment Configuration

### LiDAR Parameters

```yaml
lidar_beams: 720           # Number of beams (360, 720, or 1080)
lidar_fov: 4.71238898      # Field of view (rad) = 270°
lidar_max_range: 30.0      # Max detection range (m)
```

### Simulation Parameters

```yaml
timestep: 0.01             # Physics timestep (s) = 100 Hz
max_steps: 5000            # Max steps per episode
integrator: "RK4"          # Integration method (RK4 or Euler)
```

---

## How to Configure for Different Scenarios

### High-Performance Racing (F1/10 Competition)

```yaml
environment:
  timestep: 0.01
  vehicle_params:
    v_max: 20.0
    a_max: 9.51

agents:
  car_0:
    algorithm: ftg
    params:
      min_speed: 2.0
      max_speed: 20.0
```

### Realistic Limo Robot (Sim-to-Real)

```yaml
environment:
  timestep: 0.01
  vehicle_params:
    v_max: 1.0          # Limo top speed
    v_min: -1.0
    a_max: 2.0          # Realistic acceleration
    s_min: -0.46
    s_max: 0.46

agents:
  car_0:
    algorithm: ftg
    params:
      min_speed: 0.2    # Lower minimum for Limo
      max_speed: 1.0    # Match vehicle v_max
```

---

## Parameter Impact on Behavior

### Vehicle Speed (`v_max`)
- **20.0 m/s**: Unrealistic for Limo, fast racing
- **1.0 m/s**: Realistic Limo speed, better for sim-to-real

### FTG Speed Range (`min_speed`, `max_speed`)
- **MUST match vehicle `v_max`** or FTG will command impossible speeds
- Mismatch causes poor performance and unrealistic behavior

### Steering Limits (`s_min`, `s_max`)
- Affects turning radius and maneuverability
- Limo: ≈ ±26.4° (0.46 rad)
- F1/10: ≈ ±24° (0.4189 rad)

### Acceleration (`a_max`)
- **9.51 m/s²**: Unrealistic for Limo (≈1g)
- **2.0 m/s²**: More realistic for small robot

---

## Quick Reference: Parameter Locations

| Parameter Type | File Location |
|---|---|
| Vehicle dynamics | `scenarios/v2/*.yaml` → `environment.vehicle_params` |
| FTG policy | `scenarios/v2/*.yaml` → `agents.car_X.params` |
| LiDAR config | `scenarios/v2/*.yaml` → `environment.lidar_*` |
| Physics sim | `scenarios/v2/*.yaml` → `environment.timestep` |

---

## Migration from V1

If you have v1 configs, map parameters as follows:

```python
# V1 format
env_config = {
    'vehicle_params': {...},
    'ftg_config': {...}
}

# V2 format
scenario = {
    'environment': {
        'vehicle_params': {...}  # Same structure
    },
    'agents': {
        'car_0': {
            'algorithm': 'ftg',
            'params': {...}      # ftg_config goes here
        }
    }
}
```

---

**Last Updated**: December 26, 2024
**Author**: Claude (Anthropic)
