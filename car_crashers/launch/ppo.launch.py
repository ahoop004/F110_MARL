from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """Launch the PPO attacker node for Limo_04.

    node_namespace='limo_04' makes all relative topics resolve under /limo_04/:
        scan    -> /limo_04/scan
        cmd_vel -> /limo_04/cmd_vel

    Absolute VICON topics (/vicon/Limo_04/...) are unaffected by the namespace.
    """
    return LaunchDescription([
        Node(
            package='car_crashers',
            node_executable='ppo_node',
            node_name='rl_actor_ppo_node',
            node_namespace='limo_04',
            parameters=[{
                'scan_topic':      'scan',
                'cmd_topic':       'cmd_vel',
                'primary_topic':   '/vicon/Limo_04/Limo_04',
                'secondary_topic': '/vicon/Limo_02/Limo_02',
                'rate_hz':         20.0,
                'throttle_scale':  0.5,
                'use_safety':      True,
                'hard_border':     1.0,
            }],
            output='screen',
        )
    ])
