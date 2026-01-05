# F110 MARL Training Flow Diagram

This document provides visual diagrams of the complete training pipeline for the F110 Multi-Agent Reinforcement Learning system.

---

## **1. High-Level Training Architecture**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRAINING ORCHESTRATION                               │
│                         (run_v2.py → enhanced_training.py)                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    │                 │                 │
                    ▼                 ▼                 ▼
        ┌─────────────────┐ ┌──────────────┐ ┌──────────────────┐
        │   SCENARIO      │ │  CURRICULUM  │ │   EVALUATION     │
        │   LOADING       │ │  SYSTEM      │ │   SYSTEM         │
        │                 │ │              │ │                  │
        │ - YAML config   │ │ - Spawn      │ │ - Deterministic  │
        │ - Agent setup   │ │ - Phased     │ │ - Fixed spawns   │
        │ - Rewards       │ │ - FTG sched  │ │ - Best tracking  │
        └─────────────────┘ └──────────────┘ └──────────────────┘
                    │                 │                 │
                    └─────────────────┼─────────────────┘
                                      │
                                      ▼
                        ┌──────────────────────────┐
                        │   EPISODE EXECUTION      │
                        │   (Main Training Loop)   │
                        └──────────────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    │                 │                 │
                    ▼                 ▼                 ▼
        ┌─────────────────┐ ┌──────────────┐ ┌──────────────────┐
        │  ENVIRONMENT    │ │   AGENTS     │ │   LOGGING        │
        │                 │ │              │ │                  │
        │ - F110 Physics  │ │ - TQC/TD3    │ │ - W&B           │
        │ - Multi-agent   │ │ - PPO/A2C    │ │ - CSV           │
        │ - Observations  │ │ - FTG        │ │ - Checkpoints   │
        └─────────────────┘ └──────────────┘ └──────────────────┘
```

---

## **2. Main Training Loop (run_v2.py)**

```
┌─────────────────────────────────────────────────────────────────────────┐
│ START: python run_v2.py --scenario scenarios/tqc.yaml --wandb          │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
        ┌──────────────────────────────────────────────┐
        │ 1. INITIALIZATION                            │
        │    ├─ Load scenario YAML                     │
        │    ├─ Expand presets (obs, rewards, algos)   │
        │    ├─ Create F110 parallel environment       │
        │    ├─ Instantiate agents via AgentFactory    │
        │    ├─ Build reward strategies                │
        │    ├─ Initialize curriculum                  │
        │    ├─ Setup loggers (W&B, CSV, console)      │
        │    └─ Apply initial FTG schedule             │
        └──────────────────────────────────────────────┘
                                  │
                                  ▼
        ┌──────────────────────────────────────────────┐
        │ 2. TRAINING LOOP                             │
        │    FOR episode = 0 to max_episodes:          │
        │       │                                       │
        │       ├─ _run_episode(episode_num)           │
        │       │                                       │
        │       ├─ [Every eval_frequency episodes]     │
        │       │   └─ Run evaluation (deterministic)  │
        │       │                                       │
        │       ├─ Update curriculum                   │
        │       │                                       │
        │       └─ Checkpoint if best model            │
        └──────────────────────────────────────────────┘
                                  │
                                  ▼
        ┌──────────────────────────────────────────────┐
        │ 3. FINALIZATION                              │
        │    ├─ Save final checkpoint                  │
        │    ├─ Print training summary                 │
        │    ├─ Close loggers                          │
        │    └─ Return training statistics             │
        └──────────────────────────────────────────────┘
                                  │
                                  ▼
                              [ END ]
```

---

## **3. Episode Execution Flow (_run_episode)**

```
┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: ENVIRONMENT RESET                                              │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                ┌─────────────────┴─────────────────┐
                │                                   │
                ▼                                   ▼
    ┌──────────────────────┐          ┌──────────────────────┐
    │ With Curriculum      │          │ Without Curriculum   │
    │                      │          │                      │
    │ ├─ Sample spawn cfg  │          │ ├─ Random reset      │
    │ ├─ Set poses         │          │ └─ Default spawns    │
    │ ├─ Set velocities    │          │                      │
    │ └─ Set lock_steps    │          │                      │
    └──────────────────────┘          └──────────────────────┘
                │                                   │
                └─────────────────┬─────────────────┘
                                  │
                                  ▼
                    obs, info = env.reset(options)

┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 2: EPISODE INITIALIZATION                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
        ┌──────────────────────────────────────────────┐
        │ Initialize tracking variables:               │
        │  ├─ episode_rewards = {agent_id: 0.0}        │
        │  ├─ episode_reward_components = {}           │
        │  ├─ episode_steps = 0                        │
        │  ├─ done = {agent_id: False}                 │
        │  └─ success_transitions = []                 │
        └──────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 3: EPISODE EXECUTION LOOP                                         │
│ WHILE not all(done) AND steps < max_steps:                              │
└─────────────────────────────────────────────────────────────────────────┘
        │
        ├─► ┌───────────────────────────────────────────────┐
        │   │ 3a. ACTION SELECTION                          │
        │   │  FOR each agent:                              │
        │   │    ├─ Flatten obs (dict → vector)             │
        │   │    │    LiDAR[108] + ego_vel[3] +             │
        │   │    │    target_vel[3] + rel_pose[5] = 119D    │
        │   │    │                                           │
        │   │    ├─ Normalize obs (running mean/std)        │
        │   │    │    flat_obs = (obs - μ) / σ              │
        │   │    │                                           │
        │   │    └─ Select action                           │
        │   │        action = agent.act(flat_obs,           │
        │   │                          deterministic=False)  │
        │   └───────────────────────────────────────────────┘
        │                       │
        ├─► ┌─────────────────┴─────────────────────────────┐
        │   │ 3b. ENVIRONMENT STEP                          │
        │   │    next_obs, env_rewards, terminations,       │
        │   │    truncations, step_info = env.step(actions) │
        │   └───────────────────────────────────────────────┘
        │                       │
        ├─► ┌─────────────────┴─────────────────────────────┐
        │   │ 3c. REWARD COMPUTATION                        │
        │   │  FOR each agent:                              │
        │   │    IF custom reward configured:               │
        │   │      ├─ Build reward_info dict                │
        │   │      │    (obs, next_obs, info, done, etc.)   │
        │   │      │                                         │
        │   │      └─ Compute reward components             │
        │   │          reward, components =                 │
        │   │            reward_strategy.compute(info)      │
        │   │                                                │
        │   │          Components (gaplock_full):           │
        │   │          ├─ terminal (±200, sparse)           │
        │   │          ├─ distance (±0.002, dense)          │
        │   │          ├─ pressure (±0.001, dense)          │
        │   │          ├─ forcing  (±0.013, dense)          │
        │   │          ├─ heading  (±0.001, dense)          │
        │   │          └─ speed    (+0.0005, dense)         │
        │   │    ELSE:                                      │
        │   │      └─ Use environment reward                │
        │   └───────────────────────────────────────────────┘
        │                       │
        └─► ┌─────────────────┴─────────────────────────────┐
            │ 3d. AGENT UPDATES & STORAGE                   │
            │  FOR each agent:                              │
            │    ├─ Accumulate episode_rewards              │
            │    │                                           │
            │    ├─ Flatten & normalize next_obs            │
            │    │                                           │
            │    ├─ BRANCH: On-Policy vs Off-Policy         │
            │    │                                           │
            │    ├─► ON-POLICY (PPO, A2C):                  │
            │    │    ├─ agent.store(obs, action,           │
            │    │    │              reward, done, term)     │
            │    │    │   (stores in rollout buffer)        │
            │    │    │                                      │
            │    │    └─ [Update at episode end]            │
            │    │                                           │
            │    └─► OFF-POLICY (TQC, TD3, SAC):            │
            │         ├─ agent.store_transition(            │
            │         │     obs, action, reward,            │
            │         │     next_obs, done)                 │
            │         │   (stores in replay buffer)         │
            │         │                                      │
            │         ├─ [Optional] HER:                    │
            │         │   agent.store_hindsight_transition( │
            │         │     obs, action, reward, next_obs,  │
            │         │     done, distance)                 │
            │         │                                      │
            │         └─ agent.update()  ◄── EVERY STEP    │
            │             │                                  │
            │             ├─ Sample minibatch from buffer   │
            │             ├─ Compute TD targets             │
            │             ├─ Update critics (Q-networks)    │
            │             ├─ Update actor (policy)          │
            │             └─ Update target networks         │
            │                 (Polyak averaging)            │
            │                                                │
            └───────────────────────────────────────────────┘
                                  │
                                  ▼
                        obs = next_obs
                        episode_steps += 1
                                  │
            ┌─────────────────────┴─────────────────────┐
            │ LOOP UNTIL: all(done) OR                  │
            │             steps >= max_steps             │
            └────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 4: EPISODE FINALIZATION                                           │
└─────────────────────────────────────────────────────────────────────────┘
        │
        ├─► ┌───────────────────────────────────────────────┐
        │   │ 4a. FINAL AGENT UPDATES                       │
        │   │  FOR each ON-POLICY agent:                    │
        │   │    update_stats = agent.update()              │
        │   │      ├─ Compute GAE advantages                │
        │   │      ├─ Normalize advantages                  │
        │   │      ├─ Policy gradient update                │
        │   │      └─ Value function update                 │
        │   └───────────────────────────────────────────────┘
        │                       │
        ├─► ┌─────────────────┴─────────────────────────────┐
        │   │ 4b. OUTCOME DETERMINATION                     │
        │   │  FOR each agent:                              │
        │   │    outcome = determine_outcome(info, trunc)   │
        │   │                                                │
        │   │    Possible outcomes:                         │
        │   │    ├─ target_crash (SUCCESS)                  │
        │   │    ├─ self_crash                              │
        │   │    ├─ collision                               │
        │   │    ├─ timeout                                 │
        │   │    └─ target_finish                           │
        │   └───────────────────────────────────────────────┘
        │                       │
        ├─► ┌─────────────────┴─────────────────────────────┐
        │   │ 4c. METRICS TRACKING                          │
        │   │  FOR each agent:                              │
        │   │    ├─ metrics_tracker.add_episode(            │
        │   │    │     episode, outcome, reward, steps)     │
        │   │    │                                           │
        │   │    ├─ Compute rolling stats (success_rate,    │
        │   │    │   avg_reward, etc.) over window          │
        │   │    │                                           │
        │   │    └─ IF outcome.is_success():                │
        │   │        └─ Store all transitions in            │
        │   │            success replay buffer              │
        │   └───────────────────────────────────────────────┘
        │                       │
        ├─► ┌─────────────────┴─────────────────────────────┐
        │   │ 4d. CURRICULUM UPDATE                         │
        │   │  IF spawn_curriculum:                         │
        │   │    advancement = curriculum.update(           │
        │   │      outcome, reward, episode_num)            │
        │   │                                                │
        │   │    IF advanced:                               │
        │   │      ├─ Log stage transition                  │
        │   │      ├─ Update FTG schedule                   │
        │   │      └─ Save best_eval_phase_N checkpoint     │
        │   └───────────────────────────────────────────────┘
        │                       │
        └─► ┌─────────────────┴─────────────────────────────┐
            │ 4e. LOGGING                                   │
            │  ├─ Log to W&B (train/* metrics)              │
            │  ├─ Log to CSV (episode row)                  │
            │  └─ Update Rich console dashboard             │
            └───────────────────────────────────────────────┘
                                  │
                                  ▼
                        [ EPISODE COMPLETE ]
```

---

## **4. Algorithm-Specific Update Flows**

### **4.1 TQC Update (Off-Policy, Every Step)**

```
┌─────────────────────────────────────────────────────────────┐
│ TQC.update() - Called EVERY environment step                │
└─────────────────────────────────────────────────────────────┘
        │
        ├─► Check: steps >= learning_starts? ──NO──► return None
        │                │
        │               YES
        │                ▼
        ├─► Check: buffer_size >= batch_size? ──NO──► return None
        │                │
        │               YES
        │                ▼
        ├─► ┌─────────────────────────────────────────────┐
        │   │ 1. Sample minibatch from replay buffer      │
        │   │    (s, a, r, s', done) × batch_size         │
        │   └─────────────────────────────────────────────┘
        │                │
        ├─► ┌─────────────┴───────────────────────────────┐
        │   │ 2. Compute target actions with noise        │
        │   │    a' ~ π_target(s')                        │
        │   │    ε ~ N(0, σ²)  (Gaussian noise)           │
        │   │    a' = a' + ε                              │
        │   └─────────────────────────────────────────────┘
        │                │
        ├─► ┌─────────────┴───────────────────────────────┐
        │   │ 3. Compute target Q-values (distributional) │
        │   │    FOR each of 5 critic networks:           │
        │   │      ├─ Q_i(s', a') → 25 quantiles          │
        │   │      └─ Drop top 2 quantiles                │
        │   │                                              │
        │   │    Aggregate: 5 critics × 23 quantiles      │
        │   │               = 115 quantile estimates      │
        │   │                                              │
        │   │    Target: y = r + γ * quantile_dist(s',a') │
        │   └─────────────────────────────────────────────┘
        │                │
        ├─► ┌─────────────┴───────────────────────────────┐
        │   │ 4. Update critics (quantile regression)     │
        │   │    FOR each critic i = 1..5:                │
        │   │      Q_i(s, a) → 25 quantiles               │
        │   │      Loss_i = quantile_huber_loss(          │
        │   │        Q_i(s,a), y, τ)                      │
        │   │                                              │
        │   │    Total critic loss = Σ Loss_i             │
        │   │    Backprop & update θ_Q                    │
        │   └─────────────────────────────────────────────┘
        │                │
        ├─► ┌─────────────┴───────────────────────────────┐
        │   │ 5. Update actor (policy gradient)           │
        │   │    Sample actions: a ~ π(s)                 │
        │   │    Compute Q-value: Q = min_i(Q_i(s, a))    │
        │   │                                              │
        │   │    Actor loss = -E[Q(s, π(s))]              │
        │   │               + α * H(π(·|s))  (entropy)    │
        │   │                                              │
        │   │    Backprop & update θ_π                    │
        │   └─────────────────────────────────────────────┘
        │                │
        ├─► ┌─────────────┴───────────────────────────────┐
        │   │ 6. Update target networks (Polyak)          │
        │   │    θ_Q_target ← (1-τ)θ_Q_target + τθ_Q      │
        │   │    θ_π_target ← (1-τ)θ_π_target + τθ_π      │
        │   │                                              │
        │   │    Default τ = 0.005 (slow update)          │
        │   └─────────────────────────────────────────────┘
        │                │
        └─► ┌─────────────┴───────────────────────────────┐
            │ 7. Update entropy coefficient (if auto)     │
            │    IF ent_coef == 'auto':                   │
            │      target_entropy = -dim(action_space)    │
            │      α_loss = -α * (H - target_entropy)     │
            │      Update α via gradient descent          │
            └─────────────────────────────────────────────┘
                                │
                                ▼
                        return None
              (Metrics logged internally by SB3)
```

---

### **4.2 PPO Update (On-Policy, Episode End)**

```
┌─────────────────────────────────────────────────────────────┐
│ PPO.update() - Called at EPISODE END                        │
└─────────────────────────────────────────────────────────────┘
        │
        ├─► Check: rollout_buffer.full()? ──NO──► return None
        │                │                    (Need n_steps)
        │               YES
        │                ▼
        ├─► ┌─────────────────────────────────────────────┐
        │   │ 1. Compute advantages (GAE)                 │
        │   │    FOR t = T-1 down to 0:                   │
        │   │      δ_t = r_t + γV(s_{t+1}) - V(s_t)      │
        │   │      A_t = δ_t + γλA_{t+1}                 │
        │   │                                              │
        │   │    Returns: R_t = A_t + V(s_t)              │
        │   └─────────────────────────────────────────────┘
        │                │
        ├─► ┌─────────────┴───────────────────────────────┐
        │   │ 2. Normalize advantages                     │
        │   │    A_norm = (A - mean(A)) / std(A)          │
        │   └─────────────────────────────────────────────┘
        │                │
        ├─► ┌─────────────┴───────────────────────────────┐
        │   │ 3. FOR epoch = 1 to n_epochs (default: 10): │
        │   │    │                                         │
        │   │    ├─ Shuffle rollout buffer                │
        │   │    │                                         │
        │   │    └─ FOR each minibatch:                   │
        │   │         │                                    │
        │   │         ├─► Compute policy ratio            │
        │   │         │    r(θ) = π_θ(a|s) / π_old(a|s)   │
        │   │         │                                    │
        │   │         ├─► Compute clipped loss            │
        │   │         │    L^CLIP = -min(                 │
        │   │         │      r(θ) * A,                    │
        │   │         │      clip(r, 1-ε, 1+ε) * A        │
        │   │         │    )                              │
        │   │         │    where ε = clip_range (0.2)     │
        │   │         │                                    │
        │   │         ├─► Compute value loss              │
        │   │         │    L^VF = (V(s) - R)²             │
        │   │         │                                    │
        │   │         ├─► Compute entropy bonus           │
        │   │         │    H = -Σ π(a|s) log π(a|s)       │
        │   │         │                                    │
        │   │         ├─► Total loss                      │
        │   │         │    L = L^CLIP + 0.5*L^VF - 0.01*H │
        │   │         │                                    │
        │   │         └─► Backprop & update θ             │
        │   └─────────────────────────────────────────────┘
        │                │
        └─► ┌─────────────┴───────────────────────────────┐
            │ 4. Clear rollout buffer                     │
            │    (Ready for next n_steps collection)      │
            └─────────────────────────────────────────────┘
                                │
                                ▼
                        return None
```

---

## **5. Evaluation Flow**

```
┌─────────────────────────────────────────────────────────────┐
│ Evaluator.evaluate() - Called every eval_frequency episodes │
└─────────────────────────────────────────────────────────────┘
        │
        ├─► ┌─────────────────────────────────────────────┐
        │   │ 1. INITIALIZATION                           │
        │   │    ├─ Apply FTG override (full strength)    │
        │   │    │    max_speed: 1.0                       │
        │   │    │    bubble_radius: 3.0                   │
        │   │    │    steering_gain: 0.35                  │
        │   │    │                                          │
        │   │    └─ Prepare fixed spawn sequence          │
        │   │       [pinch_left, pinch_right, ...]        │
        │   └─────────────────────────────────────────────┘
        │                │
        ├─► ┌─────────────┴───────────────────────────────┐
        │   │ 2. EVALUATION LOOP                          │
        │   │    FOR ep = 0 to num_episodes (default: 12):│
        │   │       │                                      │
        │   │       ├─ Select spawn (sequential cycle)    │
        │   │       │    spawn_idx = ep % len(spawns)     │
        │   │       │                                      │
        │   │       ├─ Reset with fixed spawn             │
        │   │       │    obs = env.reset(                 │
        │   │       │      poses=spawn_poses,              │
        │   │       │      velocities=spawn_speeds,        │
        │   │       │      lock_speed_steps=0)            │
        │   │       │                                      │
        │   │       ├─ Run episode (DETERMINISTIC)        │
        │   │       │    WHILE not done:                  │
        │   │       │      ├─ Flatten & normalize obs     │
        │   │       │      │    (update_stats=False!)     │
        │   │       │      │                               │
        │   │       │      ├─ Select action               │
        │   │       │      │    a = agent.act(obs,        │
        │   │       │      │      deterministic=True)     │
        │   │       │      │    (Uses mean, no sampling)  │
        │   │       │      │                               │
        │   │       │      └─ Step environment            │
        │   │       │                                      │
        │   │       └─ Record episode result             │
        │   │          (outcome, reward, steps)           │
        │   └─────────────────────────────────────────────┘
        │                │
        └─► ┌─────────────┴───────────────────────────────┐
            │ 3. AGGREGATE RESULTS                        │
            │    ├─ Compute success_rate                  │
            │    ├─ Compute avg_reward, std_reward        │
            │    ├─ Compute avg_episode_length            │
            │    ├─ Compute outcome distribution          │
            │    │                                          │
            │    └─ Log to W&B (eval_agg/*)               │
            │       Log to CSV (eval_results.csv)          │
            └─────────────────────────────────────────────┘
                                │
                                ▼
                    return EvaluationResult

┌─────────────────────────────────────────────────────────────┐
│ CHECKPOINT MANAGEMENT                                        │
└─────────────────────────────────────────────────────────────┘
        │
        ├─► IF eval_success_rate > best_eval_success_rate:
        │      ├─ Save best_eval checkpoint
        │      └─ Update best_eval_success_rate
        │
        └─► IF phased_curriculum AND advanced_to_new_phase:
               └─ Save best_eval_phase_N checkpoint
```

---

## **6. Curriculum System Flow**

### **6.1 Spawn Curriculum**

```
┌─────────────────────────────────────────────────────────────┐
│ SpawnCurriculum.sample_spawn() - Called every episode       │
└─────────────────────────────────────────────────────────────┘
        │
        ├─► ┌─────────────────────────────────────────────┐
        │   │ 1. Determine difficulty level (mixture)     │
        │   │    rand = random()                          │
        │   │                                              │
        │   │    IF rand < keep_foundation:               │
        │   │      └─ Use stage 0 (easiest)               │
        │   │    ELIF rand < keep_foundation+keep_prev:   │
        │   │      └─ Use previous stage                  │
        │   │    ELSE:                                     │
        │   │      └─ Use current stage                   │
        │   └─────────────────────────────────────────────┘
        │                │
        ├─► ┌─────────────┴───────────────────────────────┐
        │   │ 2. Sample from selected stage               │
        │   │    ├─ Choose random spawn point             │
        │   │    ├─ Sample speed from range               │
        │   │    └─ Get lock_speed_steps                  │
        │   └─────────────────────────────────────────────┘
        │                │
        └─► ┌─────────────┴───────────────────────────────┐
            │ 3. Return spawn configuration               │
            │    {                                         │
            │      'poses': np.array([[x1,y1,θ1], ...]),  │
            │      'velocities': {agent_id: speed},       │
            │      'lock_speed_steps': int,               │
            │      'stage': stage_name,                   │
            │      'spawn_points': {agent_id: name}       │
            │    }                                         │
            └─────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ SpawnCurriculum.update() - Called after each episode        │
└─────────────────────────────────────────────────────────────┘
        │
        ├─► ┌─────────────────────────────────────────────┐
        │   │ 1. Update rolling statistics                │
        │   │    rolling_success = EMA(success,           │
        │   │      window=window_size, alpha=smoothing)   │
        │   └─────────────────────────────────────────────┘
        │                │
        ├─► ┌─────────────┴───────────────────────────────┐
        │   │ 2. Check advancement criteria               │
        │   │    IF rolling_success >= threshold AND      │
        │   │       episodes_in_stage >= min_episodes:    │
        │   │      └─► ADVANCE to next stage              │
        │   │                                              │
        │   │    ELIF episodes_in_stage >= patience:      │
        │   │      └─► FORCE ADVANCE (prevent stalling)   │
        │   │                                              │
        │   │    IF rolling_success < regress_threshold:  │
        │   │      └─► REGRESS to previous stage          │
        │   └─────────────────────────────────────────────┘
        │                │
        └─► ┌─────────────┴───────────────────────────────┐
            │ 3. Apply stage changes                      │
            │    IF stage changed:                        │
            │      ├─ Log stage transition                │
            │      ├─ Reset episode counter               │
            │      └─ Update FTG schedule                 │
            └─────────────────────────────────────────────┘
```

### **6.2 Phased Curriculum (26 Phases)**

```
┌─────────────────────────────────────────────────────────────┐
│ PHASE GROUPS (4 sequential groups, 26 total phases)         │
└─────────────────────────────────────────────────────────────┘

GROUP 1: Lock Speed Reduction (Phases 0-6)
│
├─ Phase 0: lock=800 steps, speed=0.44, spawns=pinch
├─ Phase 1: lock=600 steps
├─ Phase 2: lock=400 steps
├─ Phase 3: lock=300 steps
├─ Phase 4: lock=200 steps
├─ Phase 5: lock=100 steps
└─ Phase 6: lock=0 steps   ◄── Fully dynamic defender

GROUP 2: Speed Reduction (Phases 7-12)
│
├─ Phase 7:  speed=0.44 (high speed)
├─ Phase 8:  speed=0.35
├─ Phase 9:  speed=0.25
├─ Phase 10: speed=0.15
├─ Phase 11: speed=varied [0.0-0.10]
└─ Phase 12: speed=0.0  ◄── Standstill start (hardest)

GROUP 3: Spatial Introduction (Phases 13-17)
│
├─ Phase 13: spawns=pinch (2 scenarios)
├─ Phase 14: spawns=pinch + ahead (3 scenarios)
├─ Phase 15: spawns=pinch + ahead + center (4 scenarios)
├─ Phase 16: spawns=pinch + ahead + center + left (5 scenarios)
└─ Phase 17: spawns=all (6 scenarios)  ◄── Full spatial diversity

GROUP 4: FTG Hardening (Phases 18-25)
│
├─ Phase 18: FTG weak (gain=0.25, radius=2.0)
├─ Phase 19: FTG weak+ (gain=0.30, radius=2.2)
├─ Phase 20: FTG medium (gain=0.35, radius=2.5)
├─ Phase 21: FTG medium+ (gain=0.40, radius=2.8, mode=farthest)
├─ Phase 22: FTG medium++ (gain=0.45, radius=3.0)
├─ Phase 23: FTG strong (gain=0.50, radius=3.2)
├─ Phase 24: FTG strong+ (gain=0.55, radius=3.4)
└─ Phase 25: FTG expert (gain=0.60, radius=3.5)  ◄── Final difficulty

┌─────────────────────────────────────────────────────────────┐
│ Phase Advancement Logic                                     │
└─────────────────────────────────────────────────────────────┘

FOR each episode:
  ├─ Get current phase config
  ├─ Sample spawn (with mixture: foundation/previous/current)
  ├─ Run episode
  ├─ Track success rate (rolling window)
  │
  └─ Check advancement:
      IF success_rate >= threshold AND
         episodes >= min_episodes:
        ├─ Advance to next phase
        ├─ Save best_eval_phase_N checkpoint
        └─ Update FTG schedule

      ELIF episodes >= patience:
        └─ Force advance (prevent stalling)
```

---

## **7. Data Flow Diagram**

```
┌───────────────────────────────────────────────────────────────────────┐
│                        DATA FLOW PIPELINE                             │
└───────────────────────────────────────────────────────────────────────┘

ENVIRONMENT                    OBSERVATIONS                  AGENT
    │                               │                          │
    ├─► scans: [108]               │                          │
    ├─► pose: [x, y, θ]            │                          │
    ├─► velocity: [vx, vy]         │                          │
    ├─► angular_velocity: ω        │                          │
    └─► central_state: [14]        │                          │
         (multi-agent state)       │                          │
                │                  │                          │
                ▼                  │                          │
    ┌────────────────────┐        │                          │
    │ flatten_gaplock_   │        │                          │
    │ obs()              │        │                          │
    │                    │        │                          │
    │ ├─ LiDAR[108]      │        │                          │
    │ │   / max_range    │        │                          │
    │ │   → [0, 1]       │        │                          │
    │ │                  │        │                          │
    │ ├─ ego_vel[3]      │        │                          │
    │ │   / speed_scale  │        │                          │
    │ │   → [-1, 1]      │        │                          │
    │ │                  │        │                          │
    │ ├─ target_vel[3]   │        │                          │
    │ │   / speed_scale  │        │                          │
    │ │   → [-1, 1]      │        │                          │
    │ │                  │        │                          │
    │ └─ rel_pose[5]     │        │                          │
    │     ego → target   │        │                          │
    │     sin/cos(Δθ)    │        │                          │
    │     → [-1, 1]      │        │                          │
    └────────────────────┘        │                          │
                │                  │                          │
                ▼                  │                          │
         flat_obs[119] ───────────┼─────────────────────────►│
                                   │                          │
                                   │              ┌───────────▼──────────┐
                                   │              │ ObservationNormalizer│
                                   │              │  (running mean/std)  │
                                   │              │                      │
                                   │              │ obs_norm =           │
                                   │              │   (obs - μ) / σ      │
                                   │              └──────────────────────┘
                                   │                          │
                                   │                          ▼
                                   │              ┌──────────────────────┐
                                   │              │ Neural Network       │
                                   │              │  [119] → [256]       │
                                   │              │         ↓            │
                                   │              │        [256]         │
                                   │              │         ↓            │
                                   │              │    Actor/Critic      │
                                   │              └──────────────────────┘
                                   │                          │
                                   │                          ▼
                                   │                    action[2]
                                   │                  [steering, velocity]
                                   │                          │
    ┌──────────────────────────────┼──────────────────────────┘
    │                              │
    ▼                              │
ENV.step(action)                  │
    │                              │
    ├─► next_obs                   │
    ├─► env_reward                 │
    ├─► terminations               │
    ├─► truncations                │
    └─► info                       │
         │                         │
         ▼                         │
┌─────────────────────┐           │
│ RewardStrategy      │           │
│  (gaplock_full)     │           │
│                     │           │
│ ├─ terminal         │           │
│ │   +200 success    │           │
│ │   -100 timeout    │           │
│ │   -20 self_crash  │           │
│ │                   │           │
│ ├─ distance         │           │
│ │   ±0.002/step     │           │
│ │                   │           │
│ ├─ pressure         │           │
│ │   ±0.001/step     │           │
│ │                   │           │
│ ├─ forcing          │           │
│ │   ├─ pinch_pockets│           │
│ │   ├─ clearance    │           │
│ │   └─ turn         │           │
│ │   ±0.013/step     │           │
│ │                   │           │
│ ├─ heading          │           │
│ │   ±0.001/step     │           │
│ │                   │           │
│ └─ speed            │           │
│     +0.0005/step    │           │
└─────────────────────┘           │
         │                         │
         ▼                         │
  custom_reward ──────────────────┼────────────────────┐
                                   │                    │
                                   │                    ▼
                                   │        ┌──────────────────────┐
                                   │        │ Replay Buffer /      │
                                   │        │ Rollout Buffer       │
                                   │        │                      │
                                   │        │ Store:               │
                                   │        │  (s, a, r, s', done) │
                                   │        └──────────────────────┘
                                   │                    │
                                   │                    ▼
                                   │        ┌──────────────────────┐
                                   │        │ Agent.update()       │
                                   │        │  - Sample batch      │
                                   │        │  - Compute loss      │
                                   │        │  - Backprop          │
                                   │        │  - Update params     │
                                   │        └──────────────────────┘
                                   │                    │
                                   │                    ▼
                                   │              Updated Policy
                                   │
                                   └───────────► Next Iteration
```

---

## **8. Logging & Checkpointing Flow**

```
┌─────────────────────────────────────────────────────────────┐
│ LOGGING DESTINATIONS                                         │
└─────────────────────────────────────────────────────────────┘

EPISODE END
    │
    ├─► ┌─────────────────────────────────────────┐
    │   │ W&B Logger                              │
    │   │  ├─ train/success_rate                  │
    │   │  ├─ train/avg_reward                    │
    │   │  ├─ train/episode_length                │
    │   │  ├─ train/outcome_distribution          │
    │   │  ├─ train/reward_components/*           │
    │   │  ├─ curriculum/current_stage            │
    │   │  ├─ curriculum/rolling_success          │
    │   │  └─ curriculum/episodes_in_stage        │
    │   └─────────────────────────────────────────┘
    │
    ├─► ┌─────────────────────────────────────────┐
    │   │ CSV Logger                              │
    │   │  ├─ episode_results.csv                 │
    │   │  │   (episode, outcome, reward, steps)  │
    │   │  │                                       │
    │   │  └─ training_summary.csv                │
    │   │      (final aggregated statistics)      │
    │   └─────────────────────────────────────────┘
    │
    └─► ┌─────────────────────────────────────────┐
        │ Rich Console                            │
        │  └─ Live dashboard:                     │
        │      - Episode progress bar             │
        │      - Rolling success rate             │
        │      - Current stage                    │
        │      - Recent episode outcomes          │
        └─────────────────────────────────────────┘

EVALUATION END (every eval_frequency episodes)
    │
    ├─► ┌─────────────────────────────────────────┐
    │   │ W&B Logger                              │
    │   │  ├─ eval_agg/success_rate               │
    │   │  ├─ eval_agg/avg_reward                 │
    │   │  ├─ eval_agg/avg_episode_length         │
    │   │  └─ eval_agg/outcome_distribution       │
    │   └─────────────────────────────────────────┘
    │
    └─► ┌─────────────────────────────────────────┐
        │ CSV Logger                              │
        │  └─ eval_results.csv                    │
        │      (episode, spawn, outcome, reward)  │
        └─────────────────────────────────────────┘

CHECKPOINTING TRIGGERS
    │
    ├─► IF eval_success_rate > best_eval:
    │      └─ Save: best_eval.pt
    │          ├─ Model state_dict
    │          ├─ Optimizer state_dict
    │          ├─ Obs normalizer stats
    │          └─ Metadata (episode, success_rate)
    │
    ├─► IF eval_success_rate > best_eval_phase:
    │      └─ Save: best_eval_phase_N.pt
    │
    ├─► IF episode % periodic_interval == 0:
    │      └─ Save: checkpoint_ep_{episode}.pt
    │
    └─► IF training_reward > best_training:
           └─ Save: best_training.pt

CHECKPOINT STRUCTURE
┌─────────────────────────────────────────┐
│ checkpoint.pt                           │
│  ├─ model_state:                        │
│  │   └─ agent.get_state()               │
│  │       ├─ policy parameters           │
│  │       ├─ critic parameters           │
│  │       └─ target networks             │
│  │                                       │
│  ├─ optimizer_state:                    │
│  │   ├─ policy_optimizer                │
│  │   └─ ent_coef_optimizer (if auto)    │
│  │                                       │
│  ├─ normalizer_state:                   │
│  │   ├─ running_mean                    │
│  │   ├─ running_var                     │
│  │   └─ count                           │
│  │                                       │
│  └─ metadata:                           │
│      ├─ episode                         │
│      ├─ success_rate                    │
│      ├─ curriculum_stage                │
│      ├─ timestamp                       │
│      └─ scenario_name                   │
└─────────────────────────────────────────┘
```

---

## **9. Complete System Overview**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     F110 MARL TRAINING SYSTEM ARCHITECTURE                   │
└─────────────────────────────────────────────────────────────────────────────┘

USER INPUT: python run_v2.py --scenario scenarios/tqc.yaml --wandb
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ INITIALIZATION PHASE                                                         │
│  ├─ Load YAML → Expand Presets → Validate                                   │
│  ├─ Create F110 Environment (2 agents, parallel)                            │
│  ├─ Instantiate Agents (TQC attacker, FTG defender)                         │
│  ├─ Build Reward Strategies (gaplock_full components)                       │
│  ├─ Initialize Curriculum (26 phased stages)                                │
│  └─ Setup Logging (W&B, CSV, Rich console)                                  │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ TRAINING LOOP (2500 episodes)                                                │
│                                                                               │
│  FOR episode in range(2500):                                                 │
│    │                                                                          │
│    ├─► EPISODE EXECUTION                                                     │
│    │    ├─ Reset env with curriculum spawn config                            │
│    │    ├─ FOR step in episode:                                             │
│    │    │    ├─ Flatten obs (119D) → Normalize → Act                         │
│    │    │    ├─ Env.step(actions) → next_obs, rewards                       │
│    │    │    ├─ Compute custom rewards (components)                          │
│    │    │    └─ Store & Update                                               │
│    │    │        ├─ OFF-POLICY: store_transition() → update()               │
│    │    │        └─ ON-POLICY: store() → [update at episode end]            │
│    │    │                                                                     │
│    │    └─ Finalize: metrics, outcomes, curriculum update                   │
│    │                                                                          │
│    ├─► EVALUATION (every 100 episodes)                                       │
│    │    ├─ Run 12 deterministic episodes (fixed spawns)                     │
│    │    ├─ Full-strength FTG defender                                        │
│    │    ├─ Compute eval_agg/success_rate                                    │
│    │    └─ Save best_eval checkpoint if improved                             │
│    │                                                                          │
│    ├─► CURRICULUM UPDATE                                                     │
│    │    └─ Check advancement criteria → Maybe advance phase                 │
│    │                                                                          │
│    └─► LOGGING                                                               │
│         ├─ W&B: train/* and eval_agg/* metrics                              │
│         ├─ CSV: episode_results.csv                                          │
│         └─ Console: Live dashboard                                           │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ FINALIZATION                                                                 │
│  ├─ Save final checkpoint                                                    │
│  ├─ Print training summary                                                   │
│  ├─ Save CSV summary                                                         │
│  └─ Close all loggers                                                        │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
OUTPUT FILES:
  ├─ outputs/checkpoints/tqc/run_*/
  │   ├─ best_eval.pt
  │   ├─ best_eval_phase_0.pt ... best_eval_phase_25.pt
  │   └─ metadata.json
  │
  ├─ outputs/logs/tqc/run_*/
  │   ├─ episode_results.csv
  │   ├─ eval_results.csv
  │   └─ training_summary.csv
  │
  └─ wandb/run-*/
      └─ W&B cloud logs
```

---

## **Legend & Key Concepts**

### **Symbols**
- `│` Pipeline connection
- `├─`, `└─` Branch/tree structure
- `▼` Data flow direction
- `►` Process flow
- `┌─┐` Process box
- `[...]` Optional/conditional

### **Key Terms**

**Observation Dimensions (119D)**
- LiDAR: 108 dims (range measurements)
- Ego velocity: 3 dims (vx, vy, ω)
- Target velocity: 3 dims (vx, vy, ω)
- Relative pose: 5 dims (x, y, sin(θ), cos(θ), dist)

**Training Paradigms**
- **ON-POLICY**: PPO, A2C (fresh rollouts, episode-end updates)
- **OFF-POLICY**: TQC, TD3, SAC (replay buffer, every-step updates)

**Curriculum Stages**
- **Group 1**: Lock reduction (800→0 steps)
- **Group 2**: Speed reduction (0.44→0.0 m/s)
- **Group 3**: Spatial diversity (2→6 spawns)
- **Group 4**: FTG hardening (weak→expert)

**Reward Components (gaplock_full)**
- Terminal: ±200 (sparse, episode end)
- Shaping: ~0.015/step (dense, continuous)
  - Distance, pressure, forcing, heading, speed

**Evaluation**
- Deterministic actions (mean, no sampling)
- Fixed spawn sequence (reproducible)
- Full-strength FTG (max difficulty)
- Frozen normalization (no stats updates)

---

This diagram serves as a complete reference for understanding the training pipeline from start to finish!
