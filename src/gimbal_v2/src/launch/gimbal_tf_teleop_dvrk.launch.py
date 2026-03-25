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

    gimbal_tf_node = Node(
        package='gimbal_v2',
        executable='gimbal_v2_tf_align_dvrk.py',
        name='dynamixel_gimbal_tf',
        output='screen',
        parameters=[{
            'dvrk_arm': arm,
            'device': '/dev/ttyUSB0',
            'baudrate': 57600,
            'protocol_version': 2.0,
            'dxl1_id': 1,
            'dxl2_id': 2,
            'dxl3_id': 6,
            'dxl4_id': 5,
            'zero_offset': 2048,
            'multi_turn': False,
            'base_frame': 'gimbal_base',
            'joint1_frame': 'joint1',
            'joint2_frame': 'joint2',
            'joint3_frame': 'joint3',
            'rcm_frame': 'rcm',
            'period': 0.005,
        }],
    )

    teleop_node = Node(
        package='gimbal_v2',
        executable='dvrk_teleop_gimbal_v2_orientation_vive_position.py',
        output='screen',
        parameters=[{
            'dvrk_arm': arm,
            'period': 0.005,
            'use_multithread': False,
            'executor_threads': 2,
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
        gimbal_tf_node,
        teleop_node,
        rviz_node,
    ])