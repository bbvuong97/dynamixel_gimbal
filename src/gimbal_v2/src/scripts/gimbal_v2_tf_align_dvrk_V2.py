#!/usr/bin/env python3

import math
import time
import threading

import numpy as np
from pynput import keyboard

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TransformStamped, Quaternion, PoseStamped
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster

from dynamixel_sdk import (
    PortHandler,
    PacketHandler,
    GroupBulkRead,
    COMM_SUCCESS,
)

from std_msgs.msg import Bool
# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def signed_int32(n: int) -> int:
    return n - 2**32 if n > 2**31 - 1 else n


def counts_to_angle_rad(counts: int, zero_offset: int = 2048, multi_turn: bool = False) -> float:
    delta = counts - zero_offset
    if not multi_turn:
        delta = ((delta + 2048) % 4096) - 2048
    return delta * 2.0 * math.pi / 4096.0


def angle_rad_to_counts(theta: float, zero_offset: int = 2048, multi_turn: bool = False) -> int:
    delta = theta * 4096.0 / (2.0 * math.pi)

    if not multi_turn:
        delta = ((delta + 2048.0) % 4096.0) - 2048.0

    counts = int(round(delta + zero_offset))

    if not multi_turn:
        counts = counts % 4096

    return counts


def axis_angle_to_quaternion(axis, angle_rad: float):
    ax, ay, az = axis
    half = angle_rad / 2.0
    s = math.sin(half)
    return (math.cos(half), ax * s, ay * s, az * s)


def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
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
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)]
    ])


def rot_to_quat(R):
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    return (w, x, y, z)


def in_wizard_270_360_from_counts(p):
    p = p % 4096
    return 3072 <= p <= 4095


def in_wizard_180_270_from_counts(p):
    p = p % 4096
    return 2048 <= p < 3072


def in_wizard_90_180_from_counts(p):
    p = p % 4096
    return 1024 <= p < 2048


def in_wizard_270_90_from_counts(p):
    p = p % 4096
    return (0 <= p < 1024) or (3072 <= p <= 4095)


def solve_gimbal_ik(q_04, q_34, eps=1e-9):
    R_04 = quat_to_rot(q_04)
    R_34 = quat_to_rot(q_34)
    R = R_04 @ R_34.T

    s2 = float(np.clip(R[2, 2], -1.0, 1.0))
    c2_abs = math.sqrt(R[0, 2]**2 + R[1, 2]**2)

    if c2_abs < eps:
        theta2 = math.copysign(math.pi / 2, s2)
        theta1 = math.atan2(R[1, 0], R[0, 0])
        theta3 = 0.0
        return theta1, theta2, theta3

    candidates = []
    for c2 in (+c2_abs, -c2_abs):
        theta2 = math.atan2(s2, c2)
        theta1 = math.atan2(R[1, 2], R[0, 2])
        theta3 = math.atan2(R[2, 0], R[2, 1])

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

            q_01 = axis_angle_to_quaternion((0, 0, 1), t1)
            q_12 = quat_multiply(
                axis_angle_to_quaternion((1, 0, 0), -math.pi / 2),
                axis_angle_to_quaternion((0, 0, 1), t2)
            )
            q_23 = quat_multiply(
                axis_angle_to_quaternion((0, 1, 0), -math.pi / 2),
                axis_angle_to_quaternion((0, 0, 1), t3)
            )
            q_03_pred = quat_multiply(quat_multiply(q_01, q_12), q_23)
            R_pred = quat_to_rot(q_03_pred)
            err = np.linalg.norm(R_pred - R)
            candidates.append((err, t1, t2, t3))

    if not candidates:
        for c2 in (+c2_abs, -c2_abs):
            theta2 = math.atan2(s2, c2)
            theta1 = math.atan2(R[1, 2], R[0, 2])
            theta3 = math.atan2(R[2, 0], R[2, 1])

            q_01 = axis_angle_to_quaternion((0, 0, 1), theta1)
            q_12 = quat_multiply(
                axis_angle_to_quaternion((1, 0, 0), -math.pi / 2),
                axis_angle_to_quaternion((0, 0, 1), theta2)
            )
            q_23 = quat_multiply(
                axis_angle_to_quaternion((0, 1, 0), -math.pi / 2),
                axis_angle_to_quaternion((0, 0, 1), theta3)
            )
            q_03_pred = quat_multiply(quat_multiply(q_01, q_12), q_23)
            R_pred = quat_to_rot(q_03_pred)
            err = np.linalg.norm(R_pred - R)
            candidates.append((err, theta1, theta2, theta3))

    candidates.sort(key=lambda t: t[0])
    _, theta1, theta2, theta3 = candidates[0]
    return theta1, theta2, theta3


class DynamixelGimbalTF(Node):
    def __init__(self):
        super().__init__('dynamixel_gimbal_tf')

        # Parameters
        self.declare_parameter('device', '/dev/ttyUSB0')
        self.declare_parameter('baudrate', 57600)
        self.declare_parameter('protocol_version', 2.0)

        self.declare_parameter('dxl1_id', 1)
        self.declare_parameter('dxl2_id', 2)
        self.declare_parameter('dxl3_id', 6)
        self.declare_parameter('dxl4_id', 5)

        self.declare_parameter('zero_offset', 2048)
        self.declare_parameter('multi_turn', False)

        self.declare_parameter('base_frame', 'gimbal_base')
        self.declare_parameter('joint1_frame', 'joint1')
        self.declare_parameter('joint2_frame', 'joint2')
        self.declare_parameter('joint3_frame', 'joint3')
        self.declare_parameter('joint4_frame', 'joint4')
        self.declare_parameter('rcm_frame', 'rcm')

        self.declare_parameter('period', 0.02)
        self.declare_parameter('dvrk_arm', 'PSM1')

        self.devicename = str(self.get_parameter('device').value)
        self.baudrate = int(self.get_parameter('baudrate').value)
        self.protocol_version = float(self.get_parameter('protocol_version').value)

        self.dxl1_id = int(self.get_parameter('dxl1_id').value)
        self.dxl2_id = int(self.get_parameter('dxl2_id').value)
        self.dxl3_id = int(self.get_parameter('dxl3_id').value)
        self.dxl4_id = int(self.get_parameter('dxl4_id').value)

        self.zero_offset = int(self.get_parameter('zero_offset').value)
        self.multi_turn = bool(self.get_parameter('multi_turn').value)

        self.base_frame = str(self.get_parameter('base_frame').value)
        self.joint1_frame = str(self.get_parameter('joint1_frame').value)
        self.joint2_frame = str(self.get_parameter('joint2_frame').value)
        self.joint3_frame = str(self.get_parameter('joint3_frame').value)
        self.joint4_frame = str(self.get_parameter('joint4_frame').value)
        self.rcm_frame = str(self.get_parameter('rcm_frame').value)

        self.period_s = float(self.get_parameter('period').value)
        self.psm_name = str(self.get_parameter('dvrk_arm').value)

        # Thread-safe shared state
        self.state_lock = threading.Lock()
        self.command_lock = threading.Lock()
        self.io_lock = threading.Lock()

        self.psm_q = None
        self.ecm_q = None
        self.psm_js = None

        self.command = None
        self.command_busy = False

        self.R_Cart_GimbalBase = np.array([
            [0, -1, 0],
            [1,  0, 0],
            [0,  0, 1]
        ])

        # Dynamixel constants
        self.ADDR_TORQUE_ENABLE = 64
        self.TORQUE_DISABLE = 0
        self.TORQUE_ENABLE = 1
        self.ADDR_PROFILE_VELOCITY = 112
        self.ADDR_GOAL_POSITION = 116
        self.ADDR_PRESENT_POSITION = 132
        self.LEN_PRESENT_POSITION = 4
        self.ADDR_OPERATING_MODE = 11
        self.POSITION_CONTROL_MODE = 3

        self.ADDR_LED_RED = 65
        self.LEN_LED_RED = 1
        self.LED_ENABLE = 1
        self.LED_DISABLE = 0

        # Dynamixel setup
        self.portHandler = PortHandler(self.devicename)
        self.packetHandler = PacketHandler(self.protocol_version)
        self.groupBulkRead = GroupBulkRead(self.portHandler, self.packetHandler)

        if not self.portHandler.openPort():
            self.get_logger().error(f'Failed to open port: {self.devicename}')
            raise RuntimeError('Port open failed')

        if not self.portHandler.setBaudRate(self.baudrate):
            self.get_logger().error(f'Failed to set baudrate: {self.baudrate}')
            raise RuntimeError('Baudrate set failed')

        for dxl_id in [self.dxl1_id, self.dxl2_id, self.dxl3_id, self.dxl4_id]:
            with self.io_lock:
                # Disable torque first before changing operating mode
                dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(
                    self.portHandler, dxl_id, self.ADDR_TORQUE_ENABLE, self.TORQUE_DISABLE
                )
                if dxl_comm_result != COMM_SUCCESS or dxl_error != 0:
                    self.get_logger().warning(
                        f'Failed to disable torque for ID {dxl_id} '
                        f'(comm={self.packetHandler.getTxRxResult(dxl_comm_result)}, err={dxl_error})'
                    )

                dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(
                    self.portHandler, dxl_id, self.ADDR_OPERATING_MODE, self.POSITION_CONTROL_MODE
                )
                if dxl_comm_result != COMM_SUCCESS or dxl_error != 0:
                    self.get_logger().warning(
                        f'Failed to set operating mode for ID {dxl_id} '
                        f'(comm={self.packetHandler.getTxRxResult(dxl_comm_result)}, err={dxl_error})'
                    )

            ok = self.groupBulkRead.addParam(dxl_id, self.ADDR_PRESENT_POSITION, self.LEN_PRESENT_POSITION)
            if not ok:
                self.get_logger().error(f'groupBulkRead.addParam failed for ID {dxl_id}')
                raise RuntimeError('Bulk read addParam failed')

        # ROS interfaces

        self.teleop_enable_pub = self.create_publisher(Bool, '/dvrk_teleop_gimbal/enable', 1)

        self.br = TransformBroadcaster(self)

        self.psm_cp_topic = f'/{self.psm_name}/measured_cp'
        self.ecm_cp_topic = '/ECM/measured_cp'
        self.dvrk_js_topic = f'/{self.psm_name}/measured_js'

        self.psm_cp_sub = self.create_subscription(
            PoseStamped,
            self.psm_cp_topic,
            self.psm_cp_callback,
            10
        )

        self.ecm_cp_sub = self.create_subscription(
            PoseStamped,
            self.ecm_cp_topic,
            self.ecm_cp_callback,
            10
        )

        self.dvrk_js_sub = self.create_subscription(
            JointState,
            self.dvrk_js_topic,
            self.psm_js_callback,
            10
        )

        self.timer = self.create_timer(self.period_s, self.timer_callback)

        self.listener = keyboard.Listener(on_press=self.on_key_press)
        self.listener.daemon = True
        self.listener.start()

        self.get_logger().info(
            f'DynamixelGimbalTF running. '
            f'device={self.devicename} baud={self.baudrate} '
            f'ids=[{self.dxl1_id},{self.dxl2_id},{self.dxl3_id},{self.dxl4_id}] '
            f'rate={1.0 / self.period_s:.1f} Hz arm={self.psm_name}'
        )

    # ------------------------------------------------------------------
    # Cache-only callbacks
    # ------------------------------------------------------------------

    def psm_cp_callback(self, msg: PoseStamped):
        q = (
            msg.pose.orientation.w,
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z
        )
        with self.state_lock:
            self.psm_q = q

    def ecm_cp_callback(self, msg: PoseStamped):
        q = (
            msg.pose.orientation.w,
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z
        )
        with self.state_lock:
            self.ecm_q = q

    def psm_js_callback(self, msg: JointState):
        with self.state_lock:
            self.psm_js = list(msg.position)

    def on_key_press(self, key):
        try:
            if key.char == 'r':
                cmd = 'zero'
            elif key.char == 'a':
                cmd = 'align'
            elif key.char == 's':
                cmd = 'switch_torque'
            elif key.char == 't':
                cmd = 'start_teleop_mode'
            else:
                return

            with self.command_lock:
                if self.command_busy:
                    self.get_logger().warning('Command already running; ignoring key press')
                    return
                self.command = cmd
        except AttributeError:
            pass

    # ------------------------------------------------------------------
    # Dynamixel helpers
    # ------------------------------------------------------------------

    def _get_pos(self, dxl_id: int):
        with self.io_lock:
            if not self.groupBulkRead.isAvailable(dxl_id, self.ADDR_PRESENT_POSITION, self.LEN_PRESENT_POSITION):
                self.get_logger().warning(f'BulkRead data not available for ID {dxl_id}')
                return None

            raw = self.groupBulkRead.getData(dxl_id, self.ADDR_PRESENT_POSITION, self.LEN_PRESENT_POSITION)

        return signed_int32(raw)

    def _take_pending_command(self):
        with self.command_lock:
            if self.command is None or self.command_busy:
                return None
            cmd = self.command
            self.command = None
            self.command_busy = True
            return cmd

    def _finish_command(self):
        with self.command_lock:
            self.command_busy = False

    def _run_command(self, command):
        try:
            if command == 'zero':
                self.zero_gimbal_position()
            elif command == 'align':
                self.align_gimbal_to_dvrk()
            elif command == 'switch_torque':
                self.switch_torque_state()
            elif command == 'start_teleop_mode':
                self.start_teleop_mode()
        except Exception as exc:
            self.get_logger().error(f'Command "{command}" failed: {exc}')
        finally:
            self._finish_command()

    def start_teleop_mode(self):
        # First disable gimbal torque
        self.switch_torque_state()

        # Then enable teleop
        msg = Bool()
        msg.data = True
        self.teleop_enable_pub.publish(msg)
        self.get_logger().info("Published teleop enable = True")

    # ------------------------------------------------------------------
    # Main periodic update
    # ------------------------------------------------------------------

    def timer_callback(self):
        # Run pending command synchronously so serial access stays single-threaded
        pending_command = self._take_pending_command()
        if pending_command is not None:
            self._run_command(pending_command)
            return

        with self.io_lock:
            dxl_comm_result = self.groupBulkRead.txRxPacket()

        if dxl_comm_result != COMM_SUCCESS:
            self.get_logger().warning(
                f'BulkRead txRxPacket failed: {self.packetHandler.getTxRxResult(dxl_comm_result)}'
            )
            return

        p1 = self._get_pos(self.dxl1_id)
        p2 = self._get_pos(self.dxl2_id)
        p3 = self._get_pos(self.dxl3_id)
        p4 = self._get_pos(self.dxl4_id)
        if p1 is None or p2 is None or p3 is None or p4 is None:
            return

        theta1 = counts_to_angle_rad(p1, zero_offset=self.zero_offset, multi_turn=self.multi_turn)
        theta2 = counts_to_angle_rad(p2, zero_offset=self.zero_offset, multi_turn=self.multi_turn)
        theta3 = counts_to_angle_rad(p3, zero_offset=self.zero_offset, multi_turn=self.multi_turn)
        theta4 = counts_to_angle_rad(p4, zero_offset=self.zero_offset, multi_turn=self.multi_turn)

        q_01 = axis_angle_to_quaternion((0.0, 0.0, 1.0), theta1)

        q_1_intermediate = axis_angle_to_quaternion((1.0, 0.0, 0.0), math.radians(-90))
        q_intermediate_2 = axis_angle_to_quaternion((0.0, 0.0, 1.0), theta2)
        q_12 = quat_multiply(q_1_intermediate, q_intermediate_2)

        q_2_intermediate = axis_angle_to_quaternion((0.0, 1.0, 0.0), math.radians(-90))
        q_intermediate_3 = axis_angle_to_quaternion((0.0, 0.0, 1.0), theta3)
        q_23 = quat_multiply(q_2_intermediate, q_intermediate_3)

        q_3_intermediate = axis_angle_to_quaternion((1.0, 0.0, 0.0), math.radians(90))
        q_intermediate_4 = axis_angle_to_quaternion((0.0, 0.0, 1.0), theta4)
        q_34 = quat_multiply(q_3_intermediate, q_intermediate_4)

        q_02 = quat_multiply(q_01, q_12)
        q_03 = quat_multiply(q_02, q_23)
        q_04 = quat_multiply(q_03, q_34)

        # self.broadcast_transform(self.base_frame, self.joint1_frame, q_01)
        # self.broadcast_transform(self.joint1_frame, self.joint2_frame, q_12)
        # self.broadcast_transform(self.joint2_frame, self.joint3_frame, q_23)
        # self.broadcast_transform(self.joint3_frame, self.joint4_frame, q_34)
        # self.broadcast_transform(self.base_frame, self.rcm_frame, q_04)

        now = self.get_clock().now().to_msg()
        self.send_tf(now, self.base_frame,   self.joint1_frame, 0.0, 0.0, 0.0, q_01)
        self.send_tf(now, self.joint1_frame, self.joint2_frame, 0.0, 0.0, 0.0, q_12)
        self.send_tf(now, self.joint2_frame, self.joint3_frame, 0.0, 0.0, 0.0, q_23)
        self.send_tf(now, self.joint3_frame, self.joint4_frame, 0.0, 0.0, 0.0, q_34)
        self.send_tf(now, self.base_frame,   self.rcm_frame,    0.0, 0.0, 0.0, q_04)

    def send_tf(self, stamp, parent, child, x, y, z, q):
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = parent
        t.child_frame_id = child
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = z
        t.transform.rotation = quat_to_msg(q)
        self.br.sendTransform(t)

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def switch_torque_state(self):
        with self.io_lock:
            for dxl_id in [self.dxl1_id, self.dxl2_id, self.dxl3_id, self.dxl4_id]:
                dxl_torque_state, _, dxl_error = self.packetHandler.read1ByteTxRx(
                    self.portHandler, dxl_id, self.ADDR_TORQUE_ENABLE
                )
                if dxl_error != 0:
                    self.get_logger().error(f'Failed to read torque state for ID {dxl_id}')
                    continue

                if dxl_torque_state == self.TORQUE_ENABLE:
                    self.packetHandler.write1ByteTxRx(
                        self.portHandler, dxl_id, self.ADDR_TORQUE_ENABLE, self.TORQUE_DISABLE
                    )
                    self.packetHandler.write1ByteTxRx(
                        self.portHandler, dxl_id, self.ADDR_LED_RED, self.LED_DISABLE
                    )
                    self.get_logger().info(f'Disabled torque for ID {dxl_id}')
                else:
                    self.packetHandler.write1ByteTxRx(
                        self.portHandler, dxl_id, self.ADDR_TORQUE_ENABLE, self.TORQUE_ENABLE
                    )
                    self.packetHandler.write1ByteTxRx(
                        self.portHandler, dxl_id, self.ADDR_LED_RED, self.LED_ENABLE
                    )
                    self.get_logger().info(f'Enabled torque for ID {dxl_id}')

    def zero_gimbal_position(self):
        zero_positions = {
            self.dxl1_id: self.zero_offset,
            self.dxl2_id: self.zero_offset,
            self.dxl3_id: self.zero_offset,
            self.dxl4_id: self.zero_offset,
        }

        with self.io_lock:
            for dxl_id, position in zero_positions.items():
                self.packetHandler.write1ByteTxRx(
                    self.portHandler, dxl_id, self.ADDR_TORQUE_ENABLE, self.TORQUE_ENABLE
                )
                self.packetHandler.write1ByteTxRx(
                    self.portHandler, dxl_id, self.ADDR_LED_RED, self.LED_ENABLE
                )
                self.packetHandler.write4ByteTxRx(
                    self.portHandler, dxl_id, self.ADDR_PROFILE_VELOCITY, 50
                )

                dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(
                    self.portHandler, dxl_id, self.ADDR_GOAL_POSITION, position
                )

                if dxl_comm_result != COMM_SUCCESS:
                    self.get_logger().error(
                        f'ID {dxl_id} GOAL_POSITION comm fail: '
                        f'{self.packetHandler.getTxRxResult(dxl_comm_result)}'
                    )
                elif dxl_error != 0:
                    self.get_logger().error(
                        f'ID {dxl_id} GOAL_POSITION status error: '
                        f'{self.packetHandler.getRxPacketError(dxl_error)}'
                    )

        time.sleep(1.0)

    def align_gimbal_to_dvrk(self):
        with self.state_lock:
            psm_q = self.psm_q
            ecm_q = self.ecm_q
            psm_js = None if self.psm_js is None else list(self.psm_js)

        if psm_q is None:
            self.get_logger().warning('No desired orientation from dVRK PSM available for alignment.')
            return

        if ecm_q is None:
            self.get_logger().warning('No ECM orientation from dVRK available for alignment.')
            return

        if psm_js is None or len(psm_js) < 4:
            self.get_logger().warning('No dVRK PSM joint states available for theta4.')
            return

        q_45 = axis_angle_to_quaternion((0.0, 1.0, 0.0), math.radians(180))

        R_PSM_ECM = quat_to_rot(psm_q)
        R_ECM_Cart = quat_to_rot(ecm_q)
        R_PSM_Cart = R_ECM_Cart @ R_PSM_ECM

        R_GimbalBase_Cart = self.R_Cart_GimbalBase
        R_PSM_GimbalBase = R_GimbalBase_Cart @ R_PSM_Cart

        q_des = rot_to_quat(R_PSM_GimbalBase)

        now = self.get_clock().now().to_msg()
        self.send_tf(now, 'gimbal_base', 'dvrk_desired', 0.0, 0.0, 0.0, q_des)

        q_45_inv = (q_45[0], -q_45[1], -q_45[2], -q_45[3])
        q_04 = quat_multiply(q_des, q_45_inv)

        theta4 = 2 * math.pi - psm_js[3]

        q_3_intermediate = axis_angle_to_quaternion((0.0, 0.0, 1.0), math.radians(180))
        q_2nd_intermediate = axis_angle_to_quaternion((1.0, 0.0, 0.0), math.radians(-90))
        q_2nd_intermediate_4 = axis_angle_to_quaternion((0.0, 0.0, 1.0), theta4)
        q_3_intermediate = quat_multiply(q_3_intermediate, q_2nd_intermediate)
        q_34 = quat_multiply(q_3_intermediate, q_2nd_intermediate_4)

        theta1, theta2, theta3 = solve_gimbal_ik(q_04, q_34)

        p1 = angle_rad_to_counts(theta1, self.zero_offset, self.multi_turn)
        p2 = angle_rad_to_counts(theta2, self.zero_offset, self.multi_turn)
        p3 = angle_rad_to_counts(theta3, self.zero_offset, self.multi_turn)
        p4 = angle_rad_to_counts(theta4, self.zero_offset, self.multi_turn)

        goal_positions = {
            self.dxl1_id: p1,
            self.dxl2_id: p2,
            self.dxl3_id: p3,
            self.dxl4_id: p4,
        }

        self.get_logger().info(
            f'Computed gimbal angles for alignment: '
            f'θ1={math.degrees(theta1):.2f}°, '
            f'θ2={math.degrees(theta2):.2f}°, '
            f'θ3={math.degrees(theta3):.2f}°, '
            f'θ4={math.degrees(theta4):.2f}°'
        )

        with self.io_lock:
            for dxl_id, position in goal_positions.items():
                self.packetHandler.write1ByteTxRx(
                    self.portHandler, dxl_id, self.ADDR_TORQUE_ENABLE, self.TORQUE_ENABLE
                )
                self.packetHandler.write1ByteTxRx(
                    self.portHandler, dxl_id, self.ADDR_LED_RED, self.LED_ENABLE
                )
                self.packetHandler.write4ByteTxRx(
                    self.portHandler, dxl_id, self.ADDR_PROFILE_VELOCITY, 50
                )

                dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(
                    self.portHandler, dxl_id, self.ADDR_GOAL_POSITION, position
                )

                if dxl_comm_result != COMM_SUCCESS:
                    self.get_logger().error(
                        f'ID {dxl_id} GOAL_POSITION comm fail: '
                        f'{self.packetHandler.getTxRxResult(dxl_comm_result)}'
                    )
                elif dxl_error != 0:
                    self.get_logger().error(
                        f'ID {dxl_id} GOAL_POSITION status error: '
                        f'{self.packetHandler.getRxPacketError(dxl_error)}'
                    )

                # time.sleep(1.0)

        time.sleep(1.0)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def destroy_node(self):
        try:
            if hasattr(self, 'listener') and self.listener is not None:
                self.listener.stop()
        except Exception:
            pass

        try:
            if hasattr(self, 'portHandler') and self.portHandler is not None:
                self.portHandler.closePort()
        except Exception:
            pass

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = None
    try:
        node = DynamixelGimbalTF()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()