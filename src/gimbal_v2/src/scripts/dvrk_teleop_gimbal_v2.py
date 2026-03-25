#!/usr/bin/env python3

import math
import threading

import numpy as np
import PyKDL
from pynput import keyboard

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, HistoryPolicy, ReliabilityPolicy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.duration import Duration
from rclpy.time import Time

from geometry_msgs.msg import TransformStamped, Quaternion, PoseStamped
from std_msgs.msg import Bool

import tf2_ros
from tf2_ros import TransformBroadcaster
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

import crtk
import dvrk


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

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
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
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


# ---------------------------------------------------------------------------
# dVRK teleop node (ROS 2) - ORIENTATION ONLY
# ---------------------------------------------------------------------------

class DVRKTeleopGimbalOrientation(Node):
    def __init__(self, ral):
        super().__init__('dvrk_teleop_gimbal_orientation_node')

        self._ral = ral
        self.arm_name = self._param('dvrk_arm', 'PSM1')
        self.get_logger().info(f'Using dVRK arm: {self.arm_name}')

        self.arm = dvrk.psm(ral, self.arm_name)
        self.arm.enable()
        self.arm.home()

        # ---------------- State ----------------
        self.teleop_active = False
        self.initialized = False

        self.psm_pose = None
        self.psm_ref_pose = None

        self.ecm_q = None
        self.R_gimbal_ref = None

        self.state_lock = threading.Lock()

        self._warned_waiting_psm_pose = False
        self._warned_waiting_ecm = False
        self._warned_waiting_tf = False

        # Known fixed rotation from dVRK Cart frame to gimbal_base frame
        self.R_Cart_GimbalBase = np.array([
            [0, -1, 0],
            [1,  0, 0],
            [0,  0, 1],
        ])

        # ---------------- Callback groups ----------------
        self.timer_group = MutuallyExclusiveCallbackGroup()
        self.subscription_group = MutuallyExclusiveCallbackGroup()

        # ---------------- QoS ----------------
        qos_reliable = bool(self._param('sensor_qos_reliable', True))
        self.sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE if qos_reliable else ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        # ---------------- Timing ----------------
        self.period_s = float(self._param('period', 0.005))
        self.timer = self.create_timer(
            self.period_s,
            self.timer_callback,
            callback_group=self.timer_group,
        )

        # ---------------- Subscriptions ----------------
        self.psm_cp_topic = f'/{self.arm_name}/measured_cp'
        self.psm_cp_sub = self.create_subscription(
            PoseStamped,
            self.psm_cp_topic,
            self.psm_cp_callback,
            self.sensor_qos,
            callback_group=self.subscription_group,
        )

        self.ecm_cp_sub = self.create_subscription(
            PoseStamped,
            '/ECM/measured_cp',
            self.ecm_cp_callback,
            self.sensor_qos,
            callback_group=self.subscription_group,
        )

        enable_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.teleop_enable_sub = self.create_subscription(
            Bool,
            '/dvrk_teleop_gimbal/enable',
            self.teleop_enable_cb,
            enable_qos,
            callback_group=self.subscription_group,
        )

        # ---------------- TF ----------------
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        # ---------------- Keyboard ----------------
        self.keyboard_listener = keyboard.Listener(on_press=self.on_key_press)
        self.keyboard_listener.start()

        self.get_logger().info('Teleop node ready')

    def _param(self, name, default):
        if not self.has_parameter(name):
            self.declare_parameter(name, default)
        return self.get_parameter(name).value

    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------
    def teleop_enable_cb(self, msg: Bool):
        with self.state_lock:
            if msg.data and not self.teleop_active:
                self.teleop_active = True
                self.initialized = False
                self.psm_ref_pose = None
                self.R_gimbal_ref = None
                enabled = True
                disabled = False
            elif (not msg.data) and self.teleop_active:
                self.teleop_active = False
                self.initialized = False
                enabled = False
                disabled = True
            else:
                enabled = False
                disabled = False

        if enabled:
            self.get_logger().info('Teleop ENABLED')
        elif disabled:
            self.get_logger().info('Teleop DISABLED')

    def on_key_press(self, key):
        # Keep keyboard listener alive for parity with the Vive script.
        # Toggle publishing is handled by the companion publisher node below.
        pass

    def psm_cp_callback(self, msg: PoseStamped):
        with self.state_lock:
            self.psm_pose = msg
            self._warned_waiting_psm_pose = False

    def ecm_cp_callback(self, msg: PoseStamped):
        q = msg.pose.orientation
        with self.state_lock:
            self.ecm_q = (q.w, q.x, q.y, q.z)
            self._warned_waiting_ecm = False

    # ------------------------------------------------------------------
    # TF / orientation helpers
    # ------------------------------------------------------------------
    def _lookup_gimbal_q04(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                'gimbal_base',
                'rcm',
                Time(),
                timeout=Duration(seconds=0.0),
            )
            q = transform.transform.rotation
            self._warned_waiting_tf = False
            return (q.w, q.x, q.y, q.z)
        except Exception as exc:
            if not self._warned_waiting_tf:
                self.get_logger().warning(f'Waiting for TF gimbal_base->rcm: {exc}')
                self._warned_waiting_tf = True
            return None

    def gimbal_to_ecm(self, gimbal_q, ecm_q):
        # Rotation from gimbal base to cart
        R_GimbalBase_Cart = self.R_Cart_GimbalBase.T

        # Rotation from cart to ECM
        R_Cart_ECM = quat_to_rot(ecm_q).T

        # Rotation from gimbal base to gimbal end-effector
        R_Gimbal4_GimbalBase = quat_to_rot(gimbal_q)

        # Combined rotation from gimbal RCM to ECM
        R_Gimbal4_ECM = R_Cart_ECM @ R_GimbalBase_Cart @ R_Gimbal4_GimbalBase
        return rot_to_quat(R_Gimbal4_ECM)

    def _publish_debug_tf(self, gimbal_q):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'ECM'
        t.child_frame_id = 'gimbal_rcm_wrt_ecm'
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation = quat_to_msg(gimbal_q)
        self.tf_broadcaster.sendTransform(t)

    # ------------------------------------------------------------------
    # Teleop
    # ------------------------------------------------------------------
    def timer_callback(self):
        with self.state_lock:
            active = self.teleop_active
        if not active:
            return
        self.teleop()

    def teleop(self):
        with self.state_lock:
            initialized = self.initialized
            psm_pose_msg = self.psm_pose
            ecm_q = self.ecm_q

        if psm_pose_msg is None:
            if not self._warned_waiting_psm_pose:
                self.get_logger().warning('Waiting for dVRK measured_cp before latching teleop reference')
                self._warned_waiting_psm_pose = True
            return

        if ecm_q is None:
            if not self._warned_waiting_ecm:
                self.get_logger().warning('Waiting for ECM measured_cp before running teleop')
                self._warned_waiting_ecm = True
            return

        q_04 = self._lookup_gimbal_q04()
        if q_04 is None:
            return

        gimbal_q = self.gimbal_to_ecm(q_04, ecm_q)
        self._publish_debug_tf(gimbal_q)

        if not initialized:
            q = psm_pose_msg.pose.orientation
            p = psm_pose_msg.pose.position
            psm_rot = PyKDL.Rotation.Quaternion(q.x, q.y, q.z, q.w)
            psm_pos = PyKDL.Vector(p.x, p.y, p.z)

            w, x, y, z = gimbal_q
            R_gimbal_ref = PyKDL.Rotation.Quaternion(x, y, z, w)
            psm_ref_pose = PyKDL.Frame(psm_rot, psm_pos)

            with self.state_lock:
                self.psm_ref_pose = psm_ref_pose
                self.R_gimbal_ref = R_gimbal_ref
                self.initialized = True

            self.get_logger().info('Gimbal reference orientation latched')
            return

        with self.state_lock:
            psm_ref_pose = None if self.psm_ref_pose is None else PyKDL.Frame(self.psm_ref_pose.M, self.psm_ref_pose.p)
            R_gimbal_ref = self.R_gimbal_ref

        if psm_ref_pose is None or R_gimbal_ref is None:
            return

        w, x, y, z = gimbal_q
        R_gimbal_curr = PyKDL.Rotation.Quaternion(x, y, z, w)

        R_delta = R_gimbal_ref.Inverse() * R_gimbal_curr
        goal = PyKDL.Frame(psm_ref_pose.M * R_delta, psm_ref_pose.p)
        self.arm.servo_cp(goal)


class TeleopKeyboardPublisher(Node):
    def __init__(self, topic='/dvrk_teleop_gimbal/enable'):
        super().__init__('dvrk_teleop_keyboard_publisher')
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub = self.create_publisher(Bool, topic, qos)
        self.enabled = False

        self.get_logger().info("Keyboard teleop: press 't' to toggle enable")

        self.listener = keyboard.Listener(on_press=self.on_key_press)
        self.listener.start()

    def on_key_press(self, key):
        try:
            if key.char == 't':
                self.enabled = not self.enabled
                msg = Bool()
                msg.data = self.enabled
                self.pub.publish(msg)
                self.get_logger().info(f'Teleop enable = {self.enabled}')
        except AttributeError:
            pass


def main():
    rclpy.init()

    keyboard_node = TeleopKeyboardPublisher()

    ral = crtk.ral('dvrk_teleop_gimbal_crtk')
    teleop = DVRKTeleopGimbalOrientation(ral)

    teleop.get_logger().info('dvrk_teleop_gimbal_orientation node started')

    use_multithread = bool(teleop._param('use_multithread', False))
    executor_threads = int(teleop._param('executor_threads', 2))

    if use_multithread:
        executor = rclpy.executors.MultiThreadedExecutor(
            num_threads=max(1, executor_threads)
        )
    else:
        executor = rclpy.executors.SingleThreadedExecutor()

    executor.add_node(keyboard_node)
    executor.add_node(teleop)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        teleop.keyboard_listener.stop()
        keyboard_node.listener.stop()
        keyboard_node.destroy_node()
        teleop.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()