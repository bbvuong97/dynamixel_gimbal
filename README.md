# dynamixel_gimbal

A ROS package for **Dynamixel gimbal teleoperation** using the
[da Vinci Research Kit (dVRK)](https://github.com/jhu-dvrk/sawIntuitiveResearchKit/wiki).

The node subscribes to the MTM (Master Tool Manipulator) joint-state topic,
extracts the wrist orientation (pitch / yaw), and converts it to goal positions
for a two-axis Dynamixel pan/tilt gimbal.

---

## Table of Contents

- [Hardware Requirements](#hardware-requirements)
- [Software Dependencies](#software-dependencies)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Topics](#topics)
- [Parameters](#parameters)
- [Package Structure](#package-structure)
- [Running Tests](#running-tests)

---

## Hardware Requirements

| Component | Description |
|-----------|-------------|
| **Dynamixel servos** | Any X-series servo (XM430, XL430, XD540, …) configured with Protocol 2.0 |
| **U2D2 adapter** | ROBOTIS USB-to-Dynamixel interface |
| **dVRK MTM** | MTML or MTMR master arm |

The default configuration assumes:

- **Servo ID 1** — pan axis (yaw / horizontal rotation)
- **Servo ID 2** — tilt axis (pitch / vertical rotation)
- Both servos pre-configured for **Position Control Mode** and **baudrate 57600**

---

## Software Dependencies

| Dependency | Notes |
|------------|-------|
| ROS Noetic (or Melodic) | Tested on Ubuntu 20.04 |
| `dynamixel-sdk` ≥ 3.7 | `pip install dynamixel-sdk` |
| `python3-numpy` | Available via `apt` |
| `dvrk_ros` | Required to run the dVRK stack |

Install the Dynamixel SDK:

```bash
pip install dynamixel-sdk
```

---

## Installation

1. Clone this repository into your catkin workspace:

   ```bash
   cd ~/catkin_ws/src
   git clone https://github.com/bbvuong97/dynamixel_gimbal.git
   ```

2. Build:

   ```bash
   cd ~/catkin_ws
   catkin_make
   source devel/setup.bash
   ```

---

## Configuration

Default parameters are defined in [`config/config.yaml`](config/config.yaml).
Override any value on the command line or in a custom launch file.

Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `port_name` | `/dev/ttyUSB0` | Serial port for the U2D2 adapter |
| `baudrate` | `57600` | Dynamixel bus baudrate |
| `pan_servo_id` | `1` | Dynamixel ID of the pan servo |
| `tilt_servo_id` | `2` | Dynamixel ID of the tilt servo |
| `mtm_joint_topic` | `/dvrk/MTML/state_joint_current` | Input joint-state topic |
| `pan_scale` | `1.0` | Scale factor for pan command |
| `tilt_scale` | `1.0` | Scale factor for tilt command |
| `min_angle_rad` | `-1.5708` (−90°) | Lower servo angle limit |
| `max_angle_rad` | `1.5708` (+90°) | Upper servo angle limit |
| `deadband_rad` | `0.01` | Minimum change before a new command is sent |

---

## Usage

### Basic launch

```bash
roslaunch dynamixel_gimbal gimbal_teleop.launch
```

### Override MTM arm (MTMR instead of MTML)

```bash
roslaunch dynamixel_gimbal gimbal_teleop.launch mtm_name:=MTMR
```

### Custom serial port and servo IDs

```bash
roslaunch dynamixel_gimbal gimbal_teleop.launch \
    port_name:=/dev/ttyUSB1 \
    pan_servo_id:=3 \
    tilt_servo_id:=4
```

### Invert pan direction

```bash
roslaunch dynamixel_gimbal gimbal_teleop.launch pan_scale:=-1.0
```

---

## Topics

### Subscribed

| Topic | Type | Description |
|-------|------|-------------|
| `/dvrk/MTML/state_joint_current` *(configurable)* | `sensor_msgs/JointState` | MTM wrist joint positions — joint[5]=pitch (tilt), joint[6]=yaw (pan) |

### Published

| Topic | Type | Description |
|-------|------|-------------|
| `~gimbal_angles` | `std_msgs/Float64MultiArray` | `[pan_rad, tilt_rad]` of the last commanded angles |

---

## Parameters

All parameters are loaded under the private namespace of the node
(`~parameter_name`).  See [`config/config.yaml`](config/config.yaml) for
annotated defaults and unit information.

---

## Package Structure

```
dynamixel_gimbal/
├── config/
│   └── config.yaml              # Default parameter values
├── launch/
│   └── gimbal_teleop.launch     # Main launch file
├── scripts/
│   ├── dynamixel_interface.py   # Dynamixel SDK abstraction layer
│   └── gimbal_teleop_node.py    # ROS teleoperation node
├── test/
│   ├── test_dynamixel_interface.py  # Unit tests for Dynamixel interface
│   └── test_teleop_node.py          # Unit tests for the teleoperation node
├── CMakeLists.txt
└── package.xml
```

---

## Running Tests

No hardware or ROS master is required — all tests use mocks.

```bash
cd ~/catkin_ws/src/dynamixel_gimbal
python3 -m unittest discover -s test -v
```

Or via catkin:

```bash
catkin_make run_tests_dynamixel_gimbal
```
