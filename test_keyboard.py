"""Quick test script to diagnose keyboard input issues."""

import time
from env.f110ParallelEnv import F110ParallelEnv
from render import TelemetryHUD, RewardRingExtension

print("=" * 60)
print("Keyboard Test - Diagnosing keyboard input")
print("=" * 60)
print()
print("Instructions:")
print("1. Make sure the render window has focus (click on it)")
print("2. Try pressing T, R, H, F keys")
print("3. Watch the console for [Keyboard] messages")
print()
print("If you don't see [Keyboard] messages, the window may not have focus")
print("or events aren't being dispatched properly.")
print()
print("=" * 60)
print()

# Create environment
env = F110ParallelEnv(
    map_name='maps/line2/line2',
    num_agents=2,
    timestep=0.01,
    render_mode='human'
)

# Reset to create renderer
obs, _ = env.reset()

# Add extensions
print("Adding telemetry extension...")
telemetry = TelemetryHUD(env.renderer)
telemetry.configure(enabled=True, mode=TelemetryHUD.MODE_BASIC)
env.renderer.add_extension(telemetry)
print(f"  Extension added: {telemetry.__class__.__name__}")
print(f"  Enabled: {telemetry._enabled}")
print()

print("Adding reward ring extension...")
ring = RewardRingExtension(env.renderer)
ring.configure(
    enabled=True,
    target_agent='car_1',
    inner_radius=1.0,
    outer_radius=2.5
)
env.renderer.add_extension(ring)
print(f"  Extension added: {ring.__class__.__name__}")
print(f"  Enabled: {ring._enabled}")
print()

print(f"Total extensions: {len(env.renderer._extensions)}")
for i, ext in enumerate(env.renderer._extensions):
    print(f"  {i}: {ext.__class__.__name__}")
print()

print("=" * 60)
print("Press T, R, H, F keys now...")
print("You should see [Keyboard] messages appear below")
print("=" * 60)
print()

# Run a few frames to test keyboard
for i in range(100):
    # Just render without stepping
    env.render()
    time.sleep(0.05)  # 20 FPS

    if i % 20 == 0:
        print(f"Frame {i}/100 - Press keys now...")

print()
print("=" * 60)
print("Test complete. Did you see [Keyboard] messages?")
print("If NO: The keyboard handler is not being called")
print("If YES: Check if the extensions toggled correctly")
print("=" * 60)

env.close()
