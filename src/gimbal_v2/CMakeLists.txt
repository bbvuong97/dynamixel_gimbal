cmake_minimum_required(VERSION 3.8)
project(gimbal_v2)

find_package(ament_cmake REQUIRED)

install(
  PROGRAMS
    src/scripts/gimbal_v2_tf_align_dvrk.py
    src/scripts/gimbal_v2_tf_align_dvrk_jaw_trigger.py
    src/scripts/gimbal_v2_tf_align_dvrk_V2.py
    src/scripts/dvrk_teleop_gimbal_v2_orientation_vive_position.py
    src/scripts/dvrk_teleop_gimbal_v2_orientation_vive_position_jaw_trigger.py
    src/scripts/dvrk_teleop_gimbal_v2_orientation_vive_position_V2.py
    src/scripts/dvrk_teleop_gimbal_v2_vive_position.py
    src/scripts/dvrk_teleop_gimbal_v2_vive_position_V2.py
    src/scripts/dvrk_teleop_gimbal_v2_vive_position_V2_udp.py
    src/scripts/dvrk_teleop_gimbal_v2.py
    src/scripts/vive_tracker_udp_to_ros2.py
    src/scripts/teleop_gui.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY src/launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

install(
  DIRECTORY src/rviz/
  DESTINATION share/${PROJECT_NAME}/rviz
)

ament_package()
