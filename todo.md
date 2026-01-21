# TODO

## Spawn Randomization + Heatmaps (Gaplock Line2)
- Convert pinch left/right/ahead from fixed points to spawn regions with bounds.
- Add config for randomized spawn groups (center pose + dx/dy/dtheta + speed ranges).
- Sample spawn group per episode and generate target/ego poses within bounds.
- Record spawn_group + sampled pose + outcome in episode info.
- Add three heatmaps: all spawns, success spawns, failure spawns.
