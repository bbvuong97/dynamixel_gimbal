cmake_minimum_required(VERSION 3.0.2)
project(dynamixel_gimbal)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
)

catkin_package()

catkin_install_python(
  PROGRAMS
    scripts/gimbal_teleop_node.py
    scripts/dynamixel_interface.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

if(CATKIN_ENABLE_TESTING)
  find_package(rostest REQUIRED)
  catkin_add_nosetests(test/test_dynamixel_interface.py)
  catkin_add_nosetests(test/test_teleop_node.py)
endif()
