from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='dual_uav_ros2',
            executable='uav1_node',
            name='uav1_node',
            output='screen'
        ),
        Node(
            package='dual_uav_ros2',
            executable='uav2_node',
            name='uav2_node',
            output='screen'
        ),
    ])

