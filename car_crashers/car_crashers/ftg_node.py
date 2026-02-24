import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

import numpy as np

from car_crashers.gap_follow import FollowTheGapPolicy


class gap_follower(Node):

    def __init__(self):
        super().__init__('gap_follower')

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('cmd_topic', '/cmd_vel')
        self.declare_parameter('min_speed', 0.1)
        self.declare_parameter('max_speed', 0.5)
        self.declare_parameter('max_steer', 1.0)
        self.declare_parameter('steering_gain', 20.0)
        # Safety stop: halt if anything in the forward arc is closer than this
        self.declare_parameter('min_safe_distance', 0.35)  # metres
        self.declare_parameter('safety_arc_deg', 45.0)     # ± degrees from front

        scan_topic   = self.get_parameter('scan_topic').value
        cmd_topic    = self.get_parameter('cmd_topic').value
        min_speed    = float(self.get_parameter('min_speed').value)
        max_speed    = float(self.get_parameter('max_speed').value)
        max_steer    = float(self.get_parameter('max_steer').value)
        steer_gain   = float(self.get_parameter('steering_gain').value)
        self.min_safe_dist  = float(self.get_parameter('min_safe_distance').value)
        self.safety_arc_rad = np.deg2rad(float(self.get_parameter('safety_arc_deg').value))

        # ── Publisher / subscriber ─────────────────────────────────────────────
        self.pub = self.create_publisher(Twist, cmd_topic, 10)
        self.timer = self.create_timer(0.2, self.update)
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.sub = self.create_subscription(LaserScan, scan_topic, self.onScan, qos)

        # ── Policy ────────────────────────────────────────────────────────────
        self.policy = FollowTheGapPolicy(
            min_speed=min_speed,
            max_speed=max_speed,
            max_steer=max_steer,
            fov=np.deg2rad(360),
            max_distance=12.0,
            bubble_radius=0.1,
            steering_gain=steer_gain,
        )

        self.scan = None
        self.current_velocity = np.array([0.0, 0.0, 0.0])

        self.get_logger().info(
            "FTG node ready | scan=%s cmd=%s | safety: %.2fm arc=±%.0f°"
            % (scan_topic, cmd_topic, self.min_safe_dist,
               float(self.get_parameter('safety_arc_deg').value))
        )

    # ── Safety check ──────────────────────────────────────────────────────────

    def _forward_clear(self):
        """Return True if the forward arc has no obstacle closer than min_safe_dist."""
        msg = self.scan
        if msg is None or len(msg.ranges) == 0:
            return True  # no data yet — let policy decide

        n = len(msg.ranges)
        # Beam indices that fall within ±safety_arc_rad of forward (angle = 0)
        idx_lo = int((-self.safety_arc_rad - msg.angle_min) / msg.angle_increment)
        idx_hi = int(( self.safety_arc_rad - msg.angle_min) / msg.angle_increment)
        idx_lo = max(0, min(idx_lo, n - 1))
        idx_hi = max(0, min(idx_hi, n - 1))
        if idx_lo > idx_hi:
            idx_lo, idx_hi = idx_hi, idx_lo

        forward = np.array(msg.ranges[idx_lo: idx_hi + 1], dtype=np.float32)
        valid = forward[np.isfinite(forward) & (forward > 0.0)]
        if len(valid) == 0:
            return True  # all inf/nan → open space

        return float(np.min(valid)) >= self.min_safe_dist

    # ── Main loop ─────────────────────────────────────────────────────────────

    def update(self):
        if self.scan is None or len(self.scan.ranges) == 0:
            self.get_logger().warn("Empty scan!", throttle_duration_sec=2.0)
            return

        # Safety check before running policy
        if not self._forward_clear():
            self.get_logger().warn(
                "Obstacle within %.2fm — stopping" % self.min_safe_dist,
                throttle_duration_sec=1.0,
            )
            self.pub.publish(Twist())
            return

        obs = {'scans': np.array(self.scan.ranges)}
        action = self.policy.get_action(None, obs)
        cmd = Twist()
        cmd.linear.x  = float(action[1])
        cmd.angular.z = float(action[0]) * 2
        self.pub.publish(cmd)

    def onScan(self, msg):
        self.scan = msg


def main(args=None):
    rclpy.init(args=args)
    robot = gap_follower()
    rclpy.spin(robot)
    robot.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
