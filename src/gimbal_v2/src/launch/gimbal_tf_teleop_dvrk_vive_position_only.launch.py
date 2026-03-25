#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    arm_arg = DeclareLaunchArgument(
        'arm',
        default_value='PSM1',
        description='dVRK PSM arm name'
    )

    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='false',
        description='Launch RViz2 (set true to enable)'
    )

    arm = LaunchConfiguration('arm')
    use_rviz = LaunchConfiguration('use_rviz')

    rviz_config = PathJoinSubstitution([
        FindPackageShare('gimbal_v2'),
        'rviz',
        'gimbal_tf.rviz'
    ])

    teleop_node = Node(
        package='gimbal_v2',
        executable='dvrk_teleop_gimbal_v2_vive_position_V2_udp.py',
        output='screen',
        parameters=[{
            'dvrk_arm': arm,
            'period': 0.005,
            'vive_scale': 0.2,
            'vive_deadzone': 0.002,
        }],
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz',
        arguments=['-d', rviz_config],
        output='screen',
        condition=IfCondition(use_rviz),
    )

    return LaunchDescription([
        arm_arg,
        use_rviz_arg,
        teleop_node,
        rviz_node,
    ])
