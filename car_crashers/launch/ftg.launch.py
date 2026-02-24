from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """Launch the FTG defender node for Limo_02.

    node_namespace='limo_02' makes all relative topics resolve under /limo_02/:
        scan    -> /limo_02/scan
        cmd_vel -> /limo_02/cmd_vel

    Override parameters at the command line, e.g.:
        ros2 launch car_crashers ftg.launch.py
    """
    return LaunchDescription([
        Node(
            package='car_crashers',
            node_executable='ftg_node',
            node_name='gap_follower',
            node_namespace='limo_02',
            parameters=[{
                'scan_topic':    'scan',
                'cmd_topic':     'cmd_vel',
                'min_speed':     0.1,
                'max_speed':     0.5,
                'max_steer':     1.0,
                'steering_gain': 20.0,
            }],
            output='screen',
        )
    ])
