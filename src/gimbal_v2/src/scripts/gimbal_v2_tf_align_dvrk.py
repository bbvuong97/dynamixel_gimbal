#!/usr/bin/env python3
import math
import numpy as np
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped, Quaternion, PoseStamped
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster

from pynput import keyboard

import argparse

import os

if os.name == 'nt':
    import msvcrt
    def getch():
        return msvcrt.getch().decode()
else:
    import sys, tty, termios
    def getch():
        if not sys.stdin.isatty():
            return ''
        fd = sys.stdin.fileno()
        old_settings = None
        try:
            old_settings = termios.tcgetattr(fd)
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        except termios.error:
            return ''
        finally:
            if old_settings is not None:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

from dynamixel_sdk import *  # PortHandler, PacketHandler, GroupBulkRead, COMM_SUCCESS, etc.
# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def signed_int32(n: int) -> int:
    """Interpret an unsigned 32-bit register value as signed int32."""
    return n - 2**32 if n > 2**31 - 1 else n

def counts_to_angle_rad(counts: int, zero_offset: int = 2048, multi_turn: bool = False) -> float:
    """
    Convert Dynamixel position counts (0..4095 typical) to radians.
    If multi_turn=False, wrap to [-pi, pi) around the zero_offset.
    """
    delta = counts - zero_offset
    if not multi_turn:
        # wrap to [-2048, 2048)
        delta = ((delta + 2048) % 4096) - 2048
    return delta * 2.0 * math.pi / 4096.0

def angle_rad_to_counts(theta, zero_offset=2048, multi_turn=False):
    delta = theta * 4096.0 / (2.0 * math.pi)

    if not multi_turn:
        # wrap delta to [-2048, 2048) counts
        delta = ((delta + 2048.0) % 4096.0) - 2048.0

    counts = int(round(delta + zero_offset))

    if not multi_turn:
        counts = counts % 4096  # enforce 0..4095

    return counts


def axis_angle_to_quaternion(axis, angle_rad: float):
    """
    Return quaternion (w, x, y, z) for rotation of angle_rad about axis.
    Axis is assumed normalized (for canonical axes this is true).
    """
    ax, ay, az = axis
    half = angle_rad / 2.0
    s = math.sin(half)
    return (math.cos(half), ax * s, ay * s, az * s)

def quat_multiply(q1, q2):
    """Hamilton product q = q1 * q2, both in (w,x,y,z)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    )

def quat_to_msg(q):
    w, x, y, z = q
    msg = Quaternion()
    msg.w = w
    msg.x = x
    msg.y = y
    msg.z = z
    return msg

def quat_to_rot(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]
    ])

def rot_to_quat(R):
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2,1] - R[1,2]) * s
        y = (R[0,2] - R[2,0]) * s
        z = (R[1,0] - R[0,1]) * s
    else:
        if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            s = 2.0 * math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
            w = (R[2,1] - R[1,2]) / s
            x = 0.25 * s
            y = (R[0,1] + R[1,0]) / s
            z = (R[0,2] + R[2,0]) / s
        elif R[1,1] > R[2,2]:
            s = 2.0 * math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
            w = (R[0,2] - R[2,0]) / s
            x = (R[0,1] + R[1,0]) / s
            y = 0.25 * s
            z = (R[1,2] + R[2,1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
            w = (R[1,0] - R[0,1]) / s
            x = (R[0,2] + R[2,0]) / s
            y = (R[1,2] + R[2,1]) / s
            z = 0.25 * s
    return (w, x, y, z)

# def solve_gimbal_ik(q_04, q_34, eps=1e-9):

#     R_04 = quat_to_rot(q_04)
#     R_34 = quat_to_rot(q_34)

#     R_43 = np.array([
#             [0, -1, 0],
#             [-1, 0, 0],
#             [0, 0, -1]
#         ])

#     R_03 = R_04 @ R_43
#     R = R_03

#     # # Remove fixed end rotation
#     # q_34_inv = (q_34[0], -q_34[1], -q_34[2], -q_34[3])
#     # q_03 = quat_multiply(q_04, q_34_inv)

#     # R = quat_to_rot(q_03)

#     # --- θ2: use atan2, NOT asin ---
#     s2 = R[2,2]
#     c2 = math.sqrt(R[0,2]**2 + R[1,2]**2)
#     # theta2 = math.atan2(s2, c2)
#     tan2 = s2 / c2
#     theta2 = math.atan(tan2)

#     if c2 < eps:
#         # true gimbal lock
#         theta1 = math.atan2(R[1,0], R[0,0])
#         theta3 = 0.0
#         return theta1, theta2, theta3

#     # --- θ1 ---
#     theta1 = math.atan2(R[1,2], R[0,2])

#     # --- θ3 ---
#     theta3 = math.atan2(R[2,0], R[2,1])

#     return theta1, theta2, theta3

def in_wizard_270_360_from_counts(p):
    p = p % 4096
    return 3072 <= p <= 4095   # 270°..just under 360°

def in_wizard_180_270_from_counts(p):
    p = p % 4096
    return 2048 <= p < 3072    # 180°..just under 270°

def in_wizard_90_180_from_counts(p):
    p = p % 4096
    return 1024 <= p < 2048    # 90°..just under 180°

def in_wizard_270_90_from_counts(p):
    p = p % 4096
    return (0 <= p < 1024) or (3072 <= p <= 4095)   # 270°..360° and 0°..just under 90°

def solve_gimbal_ik(q_04, q_34, eps=1e-9):
    R_04 = quat_to_rot(q_04)
    R_34 = quat_to_rot(q_34)
    R = R_04 @ R_34.T

    s2 = float(np.clip(R[2,2], -1.0, 1.0))
    c2_abs = math.sqrt(R[0,2]**2 + R[1,2]**2)

    if c2_abs < eps:
        theta2 = math.copysign(math.pi/2, s2)
        theta1 = math.atan2(R[1,0], R[0,0])
        theta3 = 0.0
        return theta1, theta2, theta3

    candidates = []
    for c2 in (+c2_abs, -c2_abs):
        theta2 = math.atan2(s2, c2)
        theta1 = math.atan2(R[1,2], R[0,2])
        theta3 = math.atan2(R[2,0], R[2,1])

        for (t1, t2, t3) in [
            (theta1, theta2, theta3),
            (theta1 + math.pi, math.pi - theta2, theta3 + math.pi),
        ]:
            p1 = angle_rad_to_counts(t1, zero_offset=2048, multi_turn=False)
            p2 = angle_rad_to_counts(t2, zero_offset=2048, multi_turn=False)

            if not in_wizard_270_90_from_counts(p1):
                continue
            if not in_wizard_270_360_from_counts(p2):
                continue

        # Reconstruct R_03 from FK (angles -> quats -> rot)
        q_01 = axis_angle_to_quaternion((0,0,1), theta1)
        q_12 = quat_multiply(axis_angle_to_quaternion((1,0,0), -math.pi/2),
                             axis_angle_to_quaternion((0,0,1), theta2))
        q_23 = quat_multiply(axis_angle_to_quaternion((0,1,0), -math.pi/2),
                             axis_angle_to_quaternion((0,0,1), theta3))
        q_03_pred = quat_multiply(quat_multiply(q_01, q_12), q_23)
        R_pred = quat_to_rot(q_03_pred)

        err = np.linalg.norm(R_pred - R)
        candidates.append((err, theta1, theta2, theta3))

    if not candidates:
        # No IK branch hits the Wizard 270..360 band; fall back to best unconstrained
        # (or raise/return None if you prefer)
        candidates = []
        for c2 in (+c2_abs, -c2_abs):
            theta2 = math.atan2(s2, c2)
            theta1 = math.atan2(R[1,2], R[0,2])
            theta3 = math.atan2(R[2,0], R[2,1])

            q_01 = axis_angle_to_quaternion((0,0,1), theta1)
            q_12 = quat_multiply(axis_angle_to_quaternion((1,0,0), -math.pi/2),
                                 axis_angle_to_quaternion((0,0,1), theta2))
            q_23 = quat_multiply(axis_angle_to_quaternion((0,1,0), -math.pi/2),
                                 axis_angle_to_quaternion((0,0,1), theta3))
            q_03_pred = quat_multiply(quat_multiply(q_01, q_12), q_23)
            R_pred = quat_to_rot(q_03_pred)
            err = np.linalg.norm(R_pred - R)
            candidates.append((err, theta1, theta2, theta3))

    candidates.sort(key=lambda t: t[0])
    _, theta1, theta2, theta3 = candidates[0]
    return theta1, theta2, theta3


# def solve_gimbal_ik(q_04, q_34, eps=1e-9):
#     # Solves for theta_1, theta_2, theta_3 given q_34 and desired q_04

#     R_04 = quat_to_rot(q_04)
#     R_34 = quat_to_rot(q_34)
#     R = R_04 @ R_34.T   # R_03 = R_04 *R_43

#     s2 = float(np.clip(R[2,2], -1.0, 1.0))
#     c2_abs = math.sqrt(R[0,2]**2 + R[1,2]**2)

#     if c2_abs < eps:
#         # gimbal lock: cos(theta2) ~ 0
#         theta2 = math.copysign(math.pi/2, s2)
#         theta1 = math.atan2(R[1,0], R[0,0])
#         theta3 = 0.0
#         return theta1, theta2, theta3

#     # Two possible signs for cos(theta2)
#     candidates = []
#     for c2 in (+c2_abs, -c2_abs):
#         theta2 = math.atan2(s2, c2)

#         theta1 = math.atan2(R[1,2], R[0,2])
#         theta3 = math.atan2(R[2,0], R[2,1])

#         # Reconstruct R_03 from FK (angles -> quats -> rot)
#         q_01 = axis_angle_to_quaternion((0,0,1), theta1)
#         q_12 = quat_multiply(axis_angle_to_quaternion((1,0,0), -math.pi/2),
#                              axis_angle_to_quaternion((0,0,1), theta2))
#         q_23 = quat_multiply(axis_angle_to_quaternion((0,1,0), -math.pi/2),
#                              axis_angle_to_quaternion((0,0,1), theta3))
#         q_03_pred = quat_multiply(quat_multiply(q_01, q_12), q_23)
#         R_pred = quat_to_rot(q_03_pred)

#         # Rotation-matrix error (Frobenius norm)
#         err = np.linalg.norm(R_pred - R)
#         candidates.append((err, theta1, theta2, theta3))

#     candidates.sort(key=lambda t: t[0])
#     _, theta1, theta2, theta3 = candidates[0]
#     return theta1, theta2, theta3


# ---------------------------------------------------------------------------
# Gimbal TF Node (ROS 2)
# ---------------------------------------------------------------------------

class DynamixelGimbalTF(Node):
    def __init__(self, arm_override=None):
        super().__init__("dynamixel_gimbal_tf")
        self._last_log_times = {}

        # ---------------- Parameters (ROS 2) ----------------
        self.devicename = self._param("device", "/dev/ttyUSB0")
        self.baudrate = int(self._param("baudrate", 57600))
        self.protocol_version = float(self._param("protocol_version", 2.0))

        self.dxl1_id = int(self._param("dxl1_id", 1))
        self.dxl2_id = int(self._param("dxl2_id", 2))
        self.dxl3_id = int(self._param("dxl3_id", 6))
        self.dxl4_id = int(self._param("dxl4_id", 5))

        # XL/X-series addresses (same as your ROS 2 script)
        self.ADDR_TORQUE_ENABLE    = 64
        self.TORQUE_DISABLE         = 0
        self.TORQUE_ENABLE          = 1
        self.ADDR_PROFILE_VELOCITY = 112
        self.ADDR_GOAL_POSITION    = 116
        self.ADDR_PRESENT_POSITION = 132
        self.LEN_PRESENT_POSITION  = 4
        self.ADDR_OPERATING_MODE   = 11
        self.POSITION_CONTROL_MODE = 3

        self.ADDR_LED_RED                = 65
        self.LEN_LED_RED                 = 1         # Data Byte Length
        self.LED_ENABLE                 = 1         # Value for enabling the LED
        self.LED_DISABLE                = 0         # Value for disabling the LED

        # Zero offset for angles
        self.zero_offset = int(self._param("zero_offset", 2048))
        self.multi_turn = bool(self._param("multi_turn", False))

        # TF frame names
        self.base_frame = self._param("base_frame", "gimbal_base")
        self.joint1_frame = self._param("joint1_frame", "joint1")
        self.joint2_frame = self._param("joint2_frame", "joint2")
        self.joint3_frame = self._param("joint3_frame", "joint3")
        self.joint4_frame = self._param("joint4_frame", "joint4")
        self.rcm_frame = self._param("rcm_frame", "rcm") # Frame 5

        # Timer rate
        self.period_s = float(self._param("period", 0.005))  # 200 Hz

        # ---------------- Dynamixel setup ----------------
        self.portHandler = PortHandler(self.devicename)
        self.packetHandler = PacketHandler(self.protocol_version)
        self.groupBulkRead = GroupBulkRead(self.portHandler, self.packetHandler)

        if not self.portHandler.openPort():
            self.get_logger().error(f"Failed to open port: {self.devicename}")
            raise RuntimeError("Port open failed")

        if not self.portHandler.setBaudRate(self.baudrate):
            self.get_logger().error(f"Failed to set baudrate: {self.baudrate}")
            raise RuntimeError("Baudrate set failed")

        # Ensure in position mode and torque disabled (so you can backdrive), then add to bulk read
        for dxl_id in [self.dxl1_id, self.dxl2_id, self.dxl3_id, self.dxl4_id]:
            dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(
                self.portHandler, dxl_id, self.ADDR_OPERATING_MODE, self.POSITION_CONTROL_MODE
            )
            if dxl_comm_result != COMM_SUCCESS or dxl_error != 0:
                self.get_logger().warning(
                    f"Failed to set operating mode for ID {dxl_id} "
                    f"(comm={self.packetHandler.getTxRxResult(dxl_comm_result)}, err={dxl_error})"
                )

            # Disable torque
            self.packetHandler.write1ByteTxRx(
                self.portHandler, dxl_id, self.ADDR_TORQUE_ENABLE, self.TORQUE_DISABLE
            )

            # Add to bulk read
            ok = self.groupBulkRead.addParam(dxl_id, self.ADDR_PRESENT_POSITION, self.LEN_PRESENT_POSITION)
            if not ok:
                self.get_logger().error(f"groupBulkRead.addParam failed for ID {dxl_id}")
                raise RuntimeError("Bulk read addParam failed")

        # TF broadcaster (ROS 1)
        self.br = TransformBroadcaster(self)

        # Periodic timer
        self.timer = self.create_timer(self.period_s, self.timer_callback)

        self.get_logger().info(
            f"DynamixelGimbalTF (ROS2) running. device={self.devicename} baud={self.baudrate} "
            f"ids=[{self.dxl1_id},{self.dxl2_id},{self.dxl3_id},{self.dxl4_id}] rate={1.0/self.period_s:.1f} Hz"
        )
        
        # ---------------- dVRK measured_cp subscriber ----------------

        # Subscribe to dVRK measured_cp topic corresponding to the selected arm (PSM)
        self.psm_name = arm_override if arm_override is not None else self._param("dvrk_arm", "PSM1")
        self.psm_cp_topic = f"/{self.psm_name}/measured_cp"
        self.psm_cp_sub = self.create_subscription(
            PoseStamped, self.psm_cp_topic, self.psm_cp_callback, 10
        )  
        self.psm_q = None  # Desired orientation from dVRK (in ECM frame)

        # Subscribe to dVRK ECM measured_cp topic for frame transformations
        self.ecm_cp_sub = self.create_subscription(
            PoseStamped, f"/ECM/measured_cp", self.ecm_cp_callback, 10
        )
        self.ecm_q = None  # Orientation of ECM (in Cart frame)

        # self.R_Cart_GimbalBase = np.array([
        #     [0, -1, 0],
        #     [-1, 0, 0],
        #     [0, 0, -1]
        # ])  # Known fixed rotation from dVRK Cart frame to gimbal_base frame

        # OLD ORIENTATION
        self.R_Cart_GimbalBase = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])  # Known fixed rotation from dVRK Cart frame to gimbal_base frame

        # # NEW ORIENTATION (added 01/2026)
        # self.R_Cart_GimbalBase = np.array([
        #     [0, 1, 0],
        #     [-1, 0, 0],
        #     [0, 0, 1]
        # ])  # Known fixed rotation from dVRK Cart frame to gimbal_base frame

        # self.R_Cart_GimbalBase = np.array([
        #     [1, 0, 0],
        #     [0, 1, 0],
        #     [0, 0, 1]
        # ])  # Known fixed rotation from dVRK Cart frame to gimbal_base frame

        # self.R_Cart_GimbalBase = np.array([
        #     [0, 1, 0],
        #     [1, 0, 0],
        #     [0, 0, -1]
        # ])  # Known fixed rotation from dVRK Cart frame to gimbal_base frame

        # self.R_Cart_GimbalBase = np.array([
        #     [0, -1, 0],
        #     [-1, 0, 0],
        #     [0, 0, -1]
        # ])  # Known fixed rotation from dVRK Cart frame to gimbal_base frame

        # ---------------- dVRK measured_js subscriber ----------------
        self.dvrk_js_topic = f"/{self.psm_name}/measured_js"
        self.dvrk_js_sub = self.create_subscription(
            JointState, self.dvrk_js_topic, self.psm_js_callback, 10
        )
        self.psm_js = None  # Measured joint states from dVRK PSM

        # ---------------- Keyboard listener ----------------
        self.command = None

        self.listener = keyboard.Listener(on_press=self.on_key_press)
        self.listener.start()

    def _param(self, name, default):
        self.declare_parameter(name, default)
        return self.get_parameter(name).value

    def _log_throttle(self, level, period_sec, key, message):
        now = time.monotonic()
        last = self._last_log_times.get(key)
        if last is None or (now - last) >= period_sec:
            self._last_log_times[key] = now
            if level == "warning":
                self.get_logger().warning(message)
            elif level == "info":
                self.get_logger().info(message)
            else:
                self.get_logger().error(message)
    
    def psm_cp_callback(self, msg):
        """
        Callback for dVRK measured_cp topic.
        Computes the required gimbal angles to align with the dVRK end-effector pose.
        """

        # Extract orientation from dVRK message
        self.psm_q = (
            msg.pose.orientation.w,
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z
        )

    def ecm_cp_callback(self, msg):

        # Extract orientation from dVRK message
        self.ecm_q = (
            msg.pose.orientation.w,
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z
        )

    def psm_js_callback(self, msg):
        """
        Callback for dVRK measured_js topic.
        Stores the measured joint states from dVRK PSM.
        """

        self.psm_js = msg.position  # List of joint positions  

    def _get_pos(self, dxl_id: int):
        if not self.groupBulkRead.isAvailable(dxl_id, self.ADDR_PRESENT_POSITION, self.LEN_PRESENT_POSITION):
            self._log_throttle("warning", 1.0, f"bulk_not_available_{dxl_id}", f"BulkRead data not available for ID {dxl_id}")
            return None

        raw = self.groupBulkRead.getData(dxl_id, self.ADDR_PRESENT_POSITION, self.LEN_PRESENT_POSITION)
        return signed_int32(raw)
    
    def rpm_to_profile_velocity(rpm):
        return int(rpm / 0.229)

    def timer_callback(self):
        # Read positions
        dxl_comm_result = self.groupBulkRead.txRxPacket()
        if dxl_comm_result != COMM_SUCCESS:
            self._log_throttle(
                "warning",
                1.0,
                "bulk_txrx_fail",
                f"BulkRead txRxPacket failed: {self.packetHandler.getTxRxResult(dxl_comm_result)}",
            )
            return

        p1 = self._get_pos(self.dxl1_id)
        p2 = self._get_pos(self.dxl2_id)
        p3 = self._get_pos(self.dxl3_id)
        p4 = self._get_pos(self.dxl4_id)
        if p1 is None or p2 is None or p3 is None or p4 is None:
            return

        # Convert to angles (rad)
        theta1 = counts_to_angle_rad(p1, zero_offset=self.zero_offset, multi_turn=self.multi_turn)
        theta2 = counts_to_angle_rad(p2, zero_offset=self.zero_offset, multi_turn=self.multi_turn)
        theta3 = counts_to_angle_rad(p3, zero_offset=self.zero_offset, multi_turn=self.multi_turn)
        theta4 = counts_to_angle_rad(p4, zero_offset=self.zero_offset, multi_turn=self.multi_turn)

        # Compose orientations (same structure as your ROS 2 script)
        # Change axes here if your physical gimbal axes differ.
        q_01 = axis_angle_to_quaternion((0.0, 0.0, 1.0), theta1)

        q_1_intermediate = axis_angle_to_quaternion((1.0, 0.0, 0.0), math.radians(-90))
        q_intermediate_2 = axis_angle_to_quaternion((0.0, 0.0, 1.0), theta2)
        q_12 = quat_multiply(q_1_intermediate, q_intermediate_2)

        q_2_intermediate = axis_angle_to_quaternion((0.0, 1.0, 0.0), math.radians(-90))
        q_intermediate_3 = axis_angle_to_quaternion((0.0, 0.0, 1.0), theta3)
        q_23 = quat_multiply(q_2_intermediate, q_intermediate_3)

        q_3_1st_intermediate = axis_angle_to_quaternion((0.0, 0.0, 1.0), math.radians(180))
        q_2nd_intermediate = axis_angle_to_quaternion((1.0, 0.0, 0.0), math.radians(-90))
        q_2nd_intermediate_4 = axis_angle_to_quaternion((0.0, 0.0, 1.0), theta4)
        q_3_intermediate = quat_multiply(q_3_1st_intermediate, q_2nd_intermediate)
        q_34 = quat_multiply(q_3_intermediate, q_2nd_intermediate_4)

        q_45 = axis_angle_to_quaternion((0.0, 1.0, 0.0), math.radians(180))

        if self.command == "zero":
            self.zero_gimbal_position()
            self.command = None

        elif self.command == "align":
            self.align_gimbal_to_dvrk()
            self.command = None

        elif self.command == "switch_torque":
            self.switch_torque_state()
            self.command = None

        now = self.get_clock().now().to_msg()

        # base -> joint1
        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = self.base_frame
        t.child_frame_id = self.joint1_frame
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 26.0 / 1000.0
        t.transform.rotation = quat_to_msg(q_01)
        self.br.sendTransform(t)

        # joint1 -> joint2
        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = self.joint1_frame
        t.child_frame_id = self.joint2_frame
        t.transform.translation.x = 0.0
        t.transform.translation.y = -66.0 / 1000.0
        t.transform.translation.z = 86.0 / 1000.0
        t.transform.rotation = quat_to_msg(q_12)
        self.br.sendTransform(t)

        # joint2 -> joint3
        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = self.joint2_frame
        t.child_frame_id = self.joint3_frame
        t.transform.translation.x = 43.0 / 1000.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 66.0 / 1000.0
        t.transform.rotation = quat_to_msg(q_23)
        self.br.sendTransform(t)

        # # joint3 -> rcm
        # t = TransformStamped()
        # t.header.stamp = now
        # t.header.frame_id = self.joint3_frame
        # t.child_frame_id = self.rcm_frame
        # t.transform.translation.x = 0.0
        # t.transform.translation.y = 0.0
        # t.transform.translation.z = 43.0 / 1000.0
        # t.transform.rotation = quat_to_msg(q_34)
        # self.br.sendTransform(t)

        # joint3 -> joint4
        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = self.joint3_frame
        t.child_frame_id = self.joint4_frame
        t.transform.translation.x = 0.0
        t.transform.translation.y = 43.0 / 1000.0
        t.transform.translation.z = 43.0 / 1000.0
        t.transform.rotation = quat_to_msg(q_34)
        self.br.sendTransform(t)

        #joint4 -> rcm (frame 5)
        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = self.joint4_frame
        t.child_frame_id = self.rcm_frame
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 26.0 / 1000.0
        t.transform.rotation = quat_to_msg(q_45)
        self.br.sendTransform(t)

        self._log_throttle(
            "info",
            1.0,
            "joint_angles",
            f"θ1={math.degrees(theta1):6.1f}°, θ2={math.degrees(theta2):6.1f}°, θ3={math.degrees(theta3):6.1f}°",
        )

    def switch_torque_state(self):
        """
        Switches the state of the torque (TORQUE.ENABLE vs TORQUE.DISABLE) depending on what the current state is
        """

        for dxl_id in [self.dxl1_id, self.dxl2_id, self.dxl3_id, self.dxl4_id]:
            # Read current torque state
            dxl_torque_state, dxl_comm_result, dxl_error = self.packetHandler.read1ByteTxRx(
                self.portHandler, dxl_id, self.ADDR_TORQUE_ENABLE
            )
            if dxl_error != 0:
                self.get_logger().error(f"Failed to read torque state for ID {dxl_id}")
                continue

            # Toggle torque state
            if dxl_torque_state == self.TORQUE_ENABLE:
                # Disable torque
                self.packetHandler.write1ByteTxRx(
                    self.portHandler, dxl_id, self.ADDR_TORQUE_ENABLE,
                    self.TORQUE_DISABLE
                )

                # Turn off red LED to indicate torque disabled
                self.packetHandler.write1ByteTxRx(
                    self.portHandler, dxl_id, self.ADDR_LED_RED,
                    self.LED_DISABLE
                )

                self.get_logger().info(f"Disabled torque for ID {dxl_id}")
            else:
                # Enable torque
                self.packetHandler.write1ByteTxRx(
                    self.portHandler, dxl_id, self.ADDR_TORQUE_ENABLE,
                    self.TORQUE_ENABLE
                )

                # Turn on red LED to indicate torque enabled
                self.packetHandler.write1ByteTxRx(
                    self.portHandler, dxl_id, self.ADDR_LED_RED,
                    self.LED_ENABLE
                )

                self.get_logger().info(f"Enabled torque for ID {dxl_id}")
   
    def zero_gimbal_position(self):
        """
        Reset gimbal to zero position (all angles to zero).
        This function sends commands to the Dynamixel motors to move them to
        their zero positions.
        """

        # Define zero positions for each motor
        zero_positions = {
            self.dxl1_id: self.zero_offset,
            self.dxl2_id: self.zero_offset,
            self.dxl3_id: self.zero_offset,
            self.dxl4_id: self.zero_offset
        }

        # Write zero positions to each motor
        for dxl_id, position in zero_positions.items():
            # Enable torque
            self.packetHandler.write1ByteTxRx(
                self.portHandler, dxl_id, self.ADDR_TORQUE_ENABLE,
                self.TORQUE_ENABLE
            )

            # Turn on red LED to indicate torque enabled
            self.packetHandler.write1ByteTxRx(
                self.portHandler, dxl_id, self.ADDR_LED_RED,
                self.LED_ENABLE
            )

            profile_velocity = 50   # ≈ 11.5 rpm for faster movement

            dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(
                self.portHandler,
                dxl_id,
                self.ADDR_PROFILE_VELOCITY,
                profile_velocity
            )

            dxl_comm_result, dxl_error = (
                self.packetHandler.write4ByteTxRx(
                    self.portHandler, dxl_id, self.ADDR_GOAL_POSITION,
                    position
                )
            )
            if dxl_comm_result != COMM_SUCCESS or dxl_error != 0:
                self.get_logger().error(f"Failed to write goal position for ID {dxl_id}")

        time.sleep(1)

        # for dxl_id in zero_positions.keys():
        #     # Disable torque
        #     self.packetHandler.write1ByteTxRx(
        #         self.portHandler, dxl_id, self.ADDR_TORQUE_ENABLE,
        #         self.TORQUE_DISABLE
        #     )

        #     # Turn off red LED to indicate torque disabled
        #     self.packetHandler.write1ByteTxRx(
        #         self.portHandler, dxl_id, self.ADDR_LED_RED,
        #         self.LED_DISABLE
        #     )

    def align_gimbal_to_dvrk(self):
        """
        Align gimbal orientation to match the desired orientation from dVRK.
        This function computes the required gimbal angles using inverse kinematics
        and sends commands to the Dynamixel motors to achieve the alignment.
        """

        if self.psm_q is None:
            self.get_logger().warning("No desired orientation from dVRK PSM available for alignment.")
            return

        # # Fixed end rotation from joint3 to end-effector
        # q_3_intermediate = axis_angle_to_quaternion((1.0, 0.0, 0.0), math.radians(-180))
        # q_intermediate_4 = axis_angle_to_quaternion((0.0, 0.0, 1.0), math.radians(90))
        # q_34 = quat_multiply(q_3_intermediate, q_intermediate_4)

        # q_3_1st_intermediate = axis_angle_to_quaternion((0.0, 0.0, 1.0), math.radians(-180))
        # q_2nd_intermediate = axis_angle_to_quaternion((1.0, 0.0, 0.0), math.radians(-90))
        # q_2nd_intermediate_4 = axis_angle_to_quaternion((0.0, 0.0, 1.0), theta4)
        # q_3_intermediate = quat_multiply(q_3_1st_intermediate, q_2nd_intermediate)
        # q_34 = quat_multiply(q_3_intermediate, q_2nd_intermediate_4)

        # Fixed end rotation from joint4 to rcm (frame 5)
        q_45 = axis_angle_to_quaternion((0.0, 1.0, 0.0), math.radians(180))

        # Take self.psm_q and transform to gimbal base frame
        if self.ecm_q is None:
            self.get_logger().warning("No ECM orientation from dVRK available for alignment.")
            return
        
        # R_PSM_ECM = quat_to_rot(self.psm_q)

        # R_ECM_Cart = quat_to_rot(self.ecm_q)

        # R_PSM_Cart = R_ECM_Cart @ R_PSM_ECM

        # R_PSM_GimbalBase = self.R_Cart_GimbalBase @ R_PSM_Cart

        # q_des = rot_to_quat(R_PSM_GimbalBase)


        R_PSM_ECM  = quat_to_rot(self.psm_q)      # ^ECM R_PSM
        R_ECM_Cart = quat_to_rot(self.ecm_q)      # ^Cart R_ECM

        R_PSM_Cart = R_ECM_Cart @ R_PSM_ECM       # ^Cart R_PSM

        # If your constant matrix is Cart <- GimbalBase, invert it
        R_GimbalBase_Cart = self.R_Cart_GimbalBase

        R_PSM_GimbalBase = R_GimbalBase_Cart @ R_PSM_Cart

        # Print for debugging: R_PSM_GimbalBase
        self.get_logger().info(f"R_PSM_GimbalBase:\n{R_PSM_GimbalBase}")

        q_des = rot_to_quat(R_PSM_GimbalBase)

        # Publish a tf2 Transform for debugging (optional)
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "gimbal_base"
        t.child_frame_id = "dvrk_desired"
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation = quat_to_msg(q_des)
        self.br.sendTransform(t)

        # Print for debugging: q_des
        self.get_logger().info(
            f"Desired quaternion from dVRK PSM: w={q_des[0]:.4f}, x={q_des[1]:.4f}, y={q_des[2]:.4f}, z={q_des[3]:.4f}"
        )
        
        ###

        # New steps to solve IK:

        # ***IMPORTANT: q_des is q_05***

        # 1. Get q_04 from q_04 = q_05 * q_54 = q_des * q_45_inv

        q_45_inv = (q_45[0], -q_45[1], -q_45[2], -q_45[3])
        q_04 = quat_multiply(q_des, q_45_inv)

        # 2. Get q_03 from q_03 = q_04 * q_43 = q_04 * q_34_inv

        # a. First, get q_43 = q_34_inv by plugging in a set theta4 value
        # theta4 defined using angle from dVRK PSM joint value
        # subtract from 2pi since gimbal 4 axis is opposite direction to dVRK PSM Joint 4 axis

        # theta4 is the value of the 4th joint position in self.psm_js subtracted from 2pi or 360 degrees
        if self.psm_js is None or len(self.psm_js) < 4:
            self.get_logger().warning("No dVRK PSM joint states available for theta4.")
            return
        theta4 = 2*math.pi - self.psm_js[3]  # Assuming the 4th joint corresponds to index 3
        # theta4 = self.psm_js[3]  # Assuming the 4th joint corresponds to index 3
        
        # b. Then compute q_43 = q_34_inv by plugging in theta4
        q_3_intermediate = axis_angle_to_quaternion((0.0, 0.0, 1.0), math.radians(180))
        q_2nd_intermediate = axis_angle_to_quaternion((1.0, 0.0, 0.0), math.radians(-90))
        q_2nd_intermediate_4 = axis_angle_to_quaternion((0.0, 0.0, 1.0), theta4)
        q_3_intermediate = quat_multiply(q_3_intermediate, q_2nd_intermediate)
        q_34 = quat_multiply(q_3_intermediate, q_2nd_intermediate_4)    

        q_34_inv = (q_34[0], -q_34[1], -q_34[2], -q_34[3])

        # # c. Compute q_03
        # q_03 = quat_multiply(q_04, q_34_inv)

        ###
        
        # Print for debugging: q_34
        self.get_logger().info(
            f"Fixed end quaternion q_34: w={q_34[0]:.4f}, x={q_34[1]:.4f}, y={q_34[2]:.4f}, z={q_34[3]:.4f}"
        )
        
        # Print for debugging: R_34
        R_34 = quat_to_rot(q_34)
        self.get_logger().info(f"Fixed end rotation R_34:\n{R_34}")

        # Solve IK for gimbal angles
        # theta1, theta2, theta3 = solve_gimbal_ik(q_03, q_34)
        theta1, theta2, theta3 = solve_gimbal_ik(q_04, q_34)

        # Print for debugging: computed angles
        self.get_logger().info(
            f"Computed gimbal angles for alignment: θ1={math.degrees(theta1):.2f}°, "
            f"θ2={math.degrees(theta2):.2f}°, θ3={math.degrees(theta3):.2f}°, θ4={math.degrees(theta4):.2f}°"
        )
        # Print for debugging: corresponding Dynamixel positions in angle degrees
        # convert theta angles to position counts with the zero offset and multi_turn and then into degrees (corresponding to the Dynamixel wizard angle in degrees)
    
        q_01 = axis_angle_to_quaternion((0,0,1), theta1)
        q_12 = quat_multiply(axis_angle_to_quaternion((1,0,0), math.radians(-90)),
                            axis_angle_to_quaternion((0,0,1), theta2))
        q_23 = quat_multiply(axis_angle_to_quaternion((0,1,0), math.radians(-90)),
                            axis_angle_to_quaternion((0,0,1), theta3))
        q_34_intermediate = quat_multiply(axis_angle_to_quaternion((0.0, 0.0, 1.0), math.radians(180)),
                             axis_angle_to_quaternion((1.0, 0.0, 0.0), math.radians(-90)))
        q_34 = quat_multiply(q_34_intermediate, axis_angle_to_quaternion((0.0, 0.0, 1.0), theta4))
        q_45 = axis_angle_to_quaternion((0.0, 1.0, 0.0), math.radians(180))
        
        q_05_pred = quat_multiply(quat_multiply(quat_multiply(quat_multiply(q_01, q_12), q_23), q_34), q_45)    

        # Print for debugging: predicted end-effector quaternion after IK
        self.get_logger().info(
            f"Predicted end-effector quaternion after IK: w={q_05_pred[0]:.4f}, x={q_05_pred[1]:.4f}, y={q_05_pred[2]:.4f}, z={q_05_pred[3]:.4f}"
        )
        
        # Print for debugging : predicted end-effector rotation after IK
        R_05_pred = quat_to_rot(q_05_pred)
        self.get_logger().info(f"Predicted end-effector rotation R_05 after IK:\n{R_05_pred}")
        # Publish a tf2 Transform for predicted end-effector after IK (optional)
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "gimbal_base"
        t.child_frame_id = "rcm_predicted_from_ik"
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation = quat_to_msg(q_05_pred)
        self.br.sendTransform(t)


        # Convert angles to position counts
        # p1 = int((theta1 * 4096.0 / (2.0 * math.pi)) + self.zero_offset)
        # p2 = int((theta2 * 4096.0 / (2.0 * math.pi)) + self.zero_offset)
        # p3 = int((theta3 * 4096.0 / (2.0 * math.pi)) + self.zero_offset)
        # p4 = int((theta4 * 4096.0 / (2.0 * math.pi)) + self.zero_offset) 

        p1 = angle_rad_to_counts(theta1, self.zero_offset, self.multi_turn)
        p2 = angle_rad_to_counts(theta2, self.zero_offset, self.multi_turn)
        p3 = angle_rad_to_counts(theta3, self.zero_offset, self.multi_turn)
        p4 = angle_rad_to_counts(theta4, self.zero_offset, self.multi_turn)  

        # Write goal positions to each motor
        goal_positions = {
            self.dxl1_id: p1,
            self.dxl2_id: p2,
            self.dxl3_id: p3,
            self.dxl4_id: p4
        }

        for dxl_id, position in goal_positions.items():
            # Enable torque
            self.packetHandler.write1ByteTxRx(
                self.portHandler, dxl_id, self.ADDR_TORQUE_ENABLE,
                self.TORQUE_ENABLE
            )

            # Turn on red LED to indicate torque enabled
            self.packetHandler.write1ByteTxRx(
                self.portHandler, dxl_id, self.ADDR_LED_RED,
                self.LED_ENABLE
            )

            profile_velocity = 50   # ≈ 11.5 rpm for smoother movement

            dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(
                self.portHandler,
                dxl_id,
                self.ADDR_PROFILE_VELOCITY,
                profile_velocity
            )

            dxl_comm_result, dxl_error = (
                self.packetHandler.write4ByteTxRx(
                    self.portHandler, dxl_id, self.ADDR_GOAL_POSITION,
                    position
                )
            )
            # if dxl_comm_result != COMM_SUCCESS or dxl_error != 0:
            #     rospy.logerr("Failed to write goal position for ID %d",
            #                  dxl_id)

            if dxl_comm_result != COMM_SUCCESS:
                self.get_logger().error(
                    f"ID {dxl_id} GOAL_POSITION comm fail: {self.packetHandler.getTxRxResult(dxl_comm_result)}"
                )
            elif dxl_error != 0:
                self.get_logger().error(
                    f"ID {dxl_id} GOAL_POSITION status error: {self.packetHandler.getRxPacketError(dxl_error)}"
                )

                
            time.sleep(1) # Add delay to allow joint by joint movement

        time.sleep(1)
    
    def on_key_press(self, key):
        try:
            if key.char == 'r':
                self.command = "zero"
            if key.char == 'a':
                self.command = "align"
            if key.char == 's' or key.char == '2':  # 's' or Enter key
                self.command = "switch_torque" # using 't' to overlap with "switch_teleop" in dvrk_teleop_gimbal.py
        except AttributeError:
            pass

def main():

    parser = argparse.ArgumentParser(description="Dynamixel Gimbal TF Broadcaster and IK Alignment with dVRK (ROS 2)")
    parser.add_argument('-a', '--arm', choices=['PSM1', 'PSM2', 'PSM3'], default='PSM1')
    args, unknown = parser.parse_known_args()

    rclpy.init(args=unknown)
    node = DynamixelGimbalTF(arm_override=args.arm)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
