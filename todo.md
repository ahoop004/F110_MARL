# F110 MARL Pretrain → 2v2 Experiment Cleanup

## Purpose

Prepare the `marl-rewiring` branch of `F110_MARL` for the HPC experiment comparing:

1. **2v2 MAPPO trained from scratch**
2. **2v2 MAPPO full fine-tuning from a pretrained single-agent PPO lap-completion policy**
3. **2v2 MAPPO using per-agent LoRA adapters on the same frozen pretrained PPO policy**

The main research question is whether LoRA adaptation can improve team performance and/or training efficiency compared with scratch training and full fine-tuning.

This cleanup should focus only on code/configuration required to make the comparison correct, reproducible, and easy to evaluate.

---

# Coding Guidelines

## Keep the implementation minimal

- Prefer **editing and reusing existing functions** over adding new abstractions.
- Reuse existing:
  - checkpoint dictionaries
  - provenance/contracts
  - scenario inheritance
  - checkpoint hooks
  - evaluation hooks
  - logging metrics
- Do not create a second checkpoint system.
- Do not create another logging framework.
- Do not create duplicated scenario files unless absolutely necessary.
- Avoid creating new helper modules when an existing checkpoint/provenance utility can reasonably hold the logic.

## Do not spend time on

- New documentation beyond code comments that are needed for clarity.
- New formal/unit/integration test suites.
- Long training runs.
- Convergence testing.
- Benchmark sweeps.
- Expensive multi-environment validation.
- Refactoring unrelated code.

Use only lightweight initialization/save/load/single-action smoke checks after the changes.

---

# Canonical Experiment Speed Contract

Use **8 m/s** as the common operational speed limit for the primary experiment.

With the current wheel radius:

```text
wheel_radius = 0.05 m
v_max = 8 m/s
wheel_speed_max = 8 / 0.05 = 160 rad/s
```

The following must therefore use the same speed contract:

- PPO lap-completion pretrain
- 2v2 scratch MAPPO
- 2v2 full fine-tuning
- 2v2 LoRA
- Racing MPC opponents

Canonical values:

```yaml
max_speed: 8.0
wheel_speed_max: 160.0
```

Where applicable, fixed Racing MPC opponents should also use:

```yaml
max_speed: 8.0
```

## Important: observation normalization

Do **not** automatically shrink observation normalization constants to 8 m/s.

If the current observation contract uses values such as:

```yaml
vx: 20.0
omega_ref: 400.0
omega: 400.0
```

leave those normalization bounds unchanged unless changing them is technically required.

The operational physical limit and the observation normalization range are separate concepts.

Keeping the wider normalization contract:

- avoids unnecessary checkpoint incompatibility
- preserves headroom for later higher-speed experiments
- keeps the PPO → MAPPO observation contract stable

---

# Priority 1 — Must Complete Before HPC Training

## Task 1 — Unify PPO pretrain and 2v2 physical/action contracts

### Primary files

- `scenarios/ppo_lap_completion_pretrain.yaml`
- `scenarios/mappo_2v2_completion_scratch.yaml`
- inherited:
  - `scenarios/mappo_2v2_completion_pretrained.yaml`
  - `scenarios/mappo_2v2_completion_lora.yaml`

### Required changes

Make the source PPO pretrain and destination 2v2 environment use the same:

- vehicle model
- vehicle model version
- tire model
- timestep
- action repeat
- steering bounds
- steering-rate bounds
- wheel radius
- wheel actuator dynamics
- wheel acceleration/deceleration limits
- friction protocol
- forward wheel-speed limit
- action-processing mode

Set the operational speed limit to:

```text
8 m/s
160 rad/s
```

The current mismatch between the PPO pretrain and 2v2 destination must be removed.

### Preserve

Do not change:

- reward objective
- MAPPO critic design
- team-return logic
- evaluation strategy
- observation normalization unless required for compatibility

### Acceptance

The PPO source and 2v2 destination must report compatible physics/action contracts.

No training run is required.

---

## Task 2 — Add reusable pretrained-checkpoint compatibility validation

### Primary code

- `src/agents/mappo/__init__.py`

Prefer reusing an existing checkpoint/provenance utility if appropriate.

### Goal

Before `load_pretrained_actor()` modifies any model weights, validate that the pretrained checkpoint is compatible with the destination policy.

### Validate at minimum

- checkpoint is a dictionary
- supported source algorithm
- actor state exists
- action dimension matches
- actor hidden dimensions match
- activation matches
- physics contract matches
- action contract matches
- source observation contract is compatible with the destination actor observation prefix

For the primary PPO → MAPPO experiment, the source should normally be:

```text
algorithm = ppo
```

Do not silently accept unrelated checkpoint types.

### Design constraint

Use one small reusable compatibility helper rather than placing duplicated comparison logic in multiple load methods.

Do not build a general checkpoint migration framework.

---

## Task 3 — Make the PPO → MAPPO observation extension explicit

### Primary code

- `src/agents/mappo/__init__.py`

### Existing configuration

The 2v2 scenario already declares:

```yaml
pretrained_actor_observation_extension: frenet_neighbors
```

### Required behavior

Three cases should exist:

#### Case A — exact observation width

Load normally.

#### Case B — destination is wider and extension is explicitly approved

If:

```text
pretrained_actor_observation_extension == frenet_neighbors
```

then:

1. copy the pretrained first-layer weights for the original PPO observation prefix
2. initialize new traffic/neighbor input columns to zero
3. copy all later compatible actor layers normally
4. preserve the pretrained exploration parameters

This ensures the initialized MAPPO policy initially behaves like the solo PPO driver and initially ignores traffic.

#### Case C — other width mismatch

Fail clearly.

### Important

Do not allow arbitrary first-layer expansion simply because the tensor dimensions happen to permit it.

Use the stored observation contracts to validate the relationship when possible.

---

## Task 4 — Save `final_model.pt` for PPO

### Primary code

- `run.py`
- existing `CheckpointHook`

### Current desired artifact convention

Every main training run should preserve:

```text
best_model.pt
final_model.pt
checkpoint_stepXXXXXXXXX.pt
evaluation_history.jsonl
```

Meaning:

- `best_model.pt` = deterministic evaluation-selected checkpoint
- `final_model.pt` = weights at the end of the complete training budget
- periodic checkpoints = learning-curve snapshots
- evaluation history = checkpoint-selection results over training

### Required change

Reuse the existing `CheckpointHook`.

Enable final checkpoint saving for PPO as well as MAPPO.

Do not add separate PPO checkpoint logic.

---

## Task 5 — Standardize the pretrained source checkpoint convention

### Primary file

- `scenarios/mappo_2v2_completion_pretrained.yaml`

### Required behavior

Use the canonical source filename:

```text
best_model.pt
```

Remove stale/default references such as:

```text
best_model2.pt
```

The command-line override must remain supported:

```text
--pretrained-actor PATH
```

### Experiment relationship

Scratch:

```yaml
pretrained_actor_checkpoint: null
```

Full fine-tuning:

```yaml
pretrained_actor_checkpoint: SAME_PPO_SOURCE
```

LoRA:

inherits the **same exact pretrained PPO source** as full fine-tuning.

Do not duplicate the source path in multiple child scenarios unless necessary.

---

## Task 6 — Strengthen MAPPO checkpoint load validation

### Primary code

- `src/agents/mappo/__init__.py`

### Goal

`MAPPOAgent.load()` should validate semantic checkpoint compatibility before loading weights.

Tensor shapes alone are not enough.

### Validate

At minimum:

- `algorithm == "mappo"`
- actor mode
- learner IDs
- learner ID order/routing
- actor hidden dimensions
- critic hidden dimensions
- activation
- critic mode
- global-state dimension
- global-state contract version
- observation dimensions
- observation contracts
- action dimension
- action contract
- physics contract
- LoRA presence/absence
- LoRA mode
- LoRA rank
- LoRA alpha
- `per_agent_log_std`
- adapter routing

### Critical LoRA mapping

Ensure the checkpoint still means:

```text
car_0 → adapter 0
car_1 → adapter 1
```

or whatever explicit mapping is stored.

Do not assume identical tensor shapes imply correct routing.

### Reuse

Reuse the checkpoint-contract comparison helper from Task 2 where applicable.

---

## Task 7 — Preserve fresh MAPPO critic initialization across all three arms

### Primary code

- `src/agents/mappo/__init__.py`
- `load_pretrained_actor()`

### Required behavior

All three destination conditions must start with a new centralized MAPPO critic:

```text
scratch      → fresh critic
fine-tuning  → fresh critic
LoRA         → fresh critic
```

The PPO critic must **not** initialize the MAPPO critic.

`load_pretrained_actor()` should modify actor state only.

### Experimental control

Preserve the current model-construction ordering if it causes identical seeds to produce equivalent critic initialization across:

- scratch
- full fine-tuning
- LoRA

No new code is needed if this is already correct.

---

## Task 8 — Keep destination training settings identical across the three arms

### Scenario structure

Use inheritance rather than duplicated YAML.

Expected chain:

```text
mappo_2v2_completion_scratch.yaml
            ↑
mappo_2v2_completion_pretrained.yaml
            ↑
mappo_2v2_completion_lora.yaml
```

### Intended differences only

#### Scratch

```text
random actor initialization
independent full actors
```

#### Full fine-tuning

```text
same pretrained PPO initialization
independent full actors
all actor parameters trainable
```

#### LoRA

```text
same pretrained PPO initialization
shared frozen base
per-agent LoRA adapters
per-agent exploration parameters
```

### Everything else should match

- environment-step budget
- physics
- speed
- maps
- spawn behavior
- friction
- fixed opponents
- rollout configuration
- batch size
- epochs
- gamma
- GAE lambda
- clip range
- entropy coefficient
- value coefficient
- reward
- team-return mode
- critic
- evaluation protocol
- evaluation seeds
- checkpoint cadence

Do not copy configuration into child files unnecessarily.

---

# Priority 2 — Verify / Minimal Cleanup

## Task 9 — Confirm LoRA checkpoint is fully self-contained

### Primary code

- `src/agents/mappo/__init__.py`
- `src/agents/common/lora.py`

### Desired checkpoint contents

A saved LoRA MAPPO checkpoint should contain:

- complete frozen pretrained base actor
- all LoRA adapters
- per-agent log-standard-deviation parameters when enabled
- centralized critic
- optimizer state
- LoRA contract
- agent-to-adapter routing
- pretrained source path
- pretrained source SHA-256
- observation/action/physics contracts

### Evaluation requirement

A saved LoRA MAPPO checkpoint must be evaluatable without loading the original PPO checkpoint.

Do not switch to adapter-only checkpoint files.

Only modify save logic if something required above is actually missing.

---

## Task 10 — Confirm full-finetune checkpoints preserve both actors independently

### Primary code

- `src/agents/mappo/__init__.py`
- `src/agents/common/independent.py`

### Desired representation

Continue storing:

```text
actors["car_0"]
actors["car_1"]
```

Each should be a complete actor state.

Do not collapse them into one shared state.

MAPPO loading must reconstruct the same actor-to-agent mapping.

Use Task 6 semantic validation rather than adding another save format.

---

## Task 11 — Keep the same destination learning rate across comparison arms

The PPO pretraining learning rate is unrelated to the downstream comparison.

For the three 2v2 conditions:

- scratch
- full fine-tuning
- LoRA

use the same destination learning-rate configuration for the baseline experiment.

If LoRA currently inherits:

```text
1e-4
```

from the same destination configuration, keep it that way.

Do not introduce a special LoRA learning rate during this cleanup.

Adapter-specific LR tuning can be a later ablation.

---

## Task 12 — Preserve existing experiment metadata

Do not create another metadata format.

Continue using the checkpoint payload/provenance already present.

Ensure saved checkpoints retain enough information to identify:

- environment steps
- policy/update version where available
- actor mode
- critic mode
- pretrained checkpoint path
- pretrained checkpoint SHA
- LoRA configuration
- physics contract
- action contract
- observation contract

Use existing:

```text
evaluation_history.jsonl
```

and existing W&B metrics.

---

## Task 13 — Preserve existing efficiency metrics

No new profiling subsystem.

Verify the three arms report the existing metrics needed for later comparison:

```text
train/environment_steps
train/agent_steps
train/updates
train/rollout_agent_samples
perf/collection_seconds
perf/update_seconds
perf/elapsed_seconds
perf/end_to_end_env_steps_per_second
```

These will later support:

### Sample efficiency

```text
team performance vs environment steps
```

### Compute efficiency

```text
team performance vs wall-clock time
```

### Effective actor experience

```text
team performance vs agent/rollout samples
```

Only add a metric if it is genuinely unavailable in one of the existing execution paths.

---

# Priority 3 — Optional Control

## Task 14 — Shared full-fine-tuning control

Only do this after the three primary arms are correct.

Add a minimal comparison condition using the same pretrained PPO source with:

```text
MAPPO actor_mode = shared
LoRA = null
```

Purpose:

Separate the effects of:

```text
pretraining
parameter sharing
LoRA specialization
```

Possible final comparison:

| Arm | Initialization | Actor organization |
|---|---|---|
| scratch-independent | random | one full actor per teammate |
| pretrained-independent | PPO | one full actor per teammate |
| pretrained-shared | PPO | one shared fully trainable actor |
| pretrained-LoRA | PPO | shared frozen base + adapter per teammate |

Do not alter the main three scenarios solely to support this optional control.

---

# Lightweight Verification Only

Do not perform long training.

After implementation, perform only enough smoke checking to verify checkpoint correctness.

## Suggested sequence

### 1. PPO checkpoint smoke check

- instantiate PPO pretrain agent
- save checkpoint
- reload checkpoint
- verify actor/critic load without error

### 2. PPO → MAPPO fine-tune initialization

- instantiate 2v2 pretrained MAPPO agent
- load PPO actor
- confirm:
  - both independent actors initialize from the same PPO source
  - centralized critic remains freshly initialized
  - traffic observation columns are initialized correctly

### 3. PPO → LoRA initialization

- instantiate LoRA MAPPO agent
- load the same PPO source
- confirm:
  - base actor frozen
  - LoRA parameters trainable
  - critic trainable
  - adapter routing correct
  - per-agent log std routing correct

### 4. MAPPO checkpoint round trip

For scratch/full-finetune/LoRA as practical:

- save checkpoint
- reload checkpoint
- run one deterministic actor action
- ensure output before and after reload agrees within normal floating-point tolerance

### 5. Compatibility failure check

Intentionally alter one relevant contract field or use an incompatible checkpoint.

Verify loading fails clearly before model weights are modified.

Examples:

- wrong speed/physics contract
- wrong action contract
- wrong LoRA rank
- wrong adapter routing
- unsupported observation expansion

No convergence test is required.

---

# Experiment Artifact Expectations

After cleanup, a normal training run should produce enough information for later evaluation without custom postprocessing infrastructure.

Expected important artifacts:

```text
best_model.pt
final_model.pt
checkpoint_step*.pt
evaluation_history.jsonl
existing W&B run metrics
```

For all three downstream arms, checkpoint evaluation should be possible using the matching scenario without reconstructing actor weights manually.

---

# Primary Experiment Definition

Once code cleanup is complete, the first HPC study should compare:

```text
A. MAPPO scratch
B. MAPPO full fine-tuning from PPO
C. MAPPO per-agent LoRA from the same PPO
```

All must use:

```text
speed cap:       8 m/s
wheel speed cap: 160 rad/s
same maps
same opponents
same physics
same action contract
same team reward
same critic architecture
same destination training budget
same destination hyperparameters
same evaluation protocol
same evaluation seeds
```

The full-finetuning and LoRA arms must use the **same exact PPO checkpoint SHA-256**.

Recommended destination seeds for the first replicated study:

```text
42
43
44
```

Do not begin large HPC runs until the compatibility checks above are complete.

---

# Scope Reminder

The coding task is complete when:

1. the PPO pretrain and all three 2v2 experiment arms share the canonical 8 m/s physical/action contract
2. pretrained actor transfer rejects incompatible checkpoints
3. intentional PPO → MAPPO traffic-observation expansion is explicit
4. PPO saves `final_model.pt`
5. MAPPO checkpoint loading validates semantic contracts and routing
6. LoRA and full-finetune checkpoints remain fully evaluatable
7. the three experiment arms differ only in their intended actor initialization/adaptation method
8. lightweight save/load smoke checks pass

Do not spend agent tokens on unrelated cleanup, documentation, formal testing infrastructure, or long-running experiments.