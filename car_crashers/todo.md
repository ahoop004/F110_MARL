# car_crashers — Reset-to-Start TODO

## State machine (both nodes)
```
RUNNING → (safety stop persists N ticks) → RESETTING → (arrived at home) → READY
```

---

## PPO.py

- [ ] Add reset parameters: `reset_dist_thresh`, `reset_max_speed`, `reset_steer_gain`, `reset_safety_ticks`
- [ ] Add state variables: `self.home_pose`, `self.mode`, `self._safety_ticks`
- [ ] Capture home pose in `on_primary` on first VICON message
- [ ] Add `_reset_cmd(pose) -> (Twist, arrived)` P-controller method
- [ ] Refactor `on_tick` with RUNNING / RESETTING / READY mode logic

## ftg_node.py

- [ ] Add imports: `TransformStamped`, `init_agent_state`, `update_agent_state` from gaplock_utils
- [ ] Add VICON subscription parameter (`vicon_topic`, default `/vicon/Limo_02/Limo_02`)
- [ ] Add `max_pose_age` parameter + stale pose guard
- [ ] Add same reset parameters as PPO
- [ ] Add state variables: `self.ego_state`, `self.home_pose`, `self.mode`, `self._safety_ticks`
- [ ] Add `on_vicon(msg)` callback (update ego_state, capture home pose, local clock stamp)
- [ ] Add `_reset_cmd(pose) -> (Twist, arrived)` (same as PPO)
- [ ] Refactor `update()` with RUNNING / RESETTING / READY mode logic

## Verification

- [ ] `colcon build --packages-select car_crashers && source install/setup.bash`
- [ ] Block lidar < 3 ticks → no mode change (debounce works)
- [ ] Block lidar > 3 ticks → "Resetting to home" log → robot drives to start → "READY" log
- [ ] Block path during reset → robot waits, resumes when clear
