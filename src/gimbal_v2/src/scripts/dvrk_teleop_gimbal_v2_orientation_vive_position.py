#!/usr/bin/env python3

import math
import threading
import time

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
from sensor_msgs.msg import JointState
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
# dVRK teleop node (ROS 2) - GIMBAL ORIENTATION + VIVE TRANSLATION
# ---------------------------------------------------------------------------

class DVRKTeleopGimbalOrientationVive(Node):
    def __init__(self, ral):
        super().__init__('dvrk_teleop_gimbal_orientation_vive_node')

        self._ral = ral
        self.arm_name = self._param('dvrk_arm', self._param('arm', 'PSM1'))
        self.get_logger().info(f'Using dVRK arm: {self.arm_name}')

        self.arm = dvrk.psm(ral, self.arm_name)
        self.arm.enable()
        self.arm.home()

        # ---------------- State ----------------
        self.teleop_active = False
        self.initialized = False

        self.psm_pose = None
        self.psm_ref_pose = None

        self.jaw_pose = None
        self.jaw_ref_pose = None

        self.ecm_q = None
        self.R_gimbal_ref = None

        self.vive_current_pos = None
        self.vive_ref_pos = None
        self.R_ecm_from_cart_latched = None
        self.R_ecm_from_vive = None

        self.state_lock = threading.Lock()

        self._warned_waiting_psm_pose = False
        self._warned_waiting_ecm = False
        self._warned_waiting_tf = False
        self._warned_waiting_vive = False
        self._warned_waiting_cart_ecm_tf = False
        self._warned_waiting_jaw_pose = False

        # Known fixed rotation from dVRK Cart frame to gimbal_base frame
        self.R_Cart_GimbalBase = np.array([
            [0, -1, 0],
            [1,  0, 0],
            [0,  0, 1],
        ])

        # Vive/SteamVR tracker frame -> Cart frame
        self.R_cart_from_trak = PyKDL.Rotation(
            1.0, 0.0,  0.0,
            0.0, 0.0, -1.0,
            0.0, 1.0,  0.0,
        )

        # ---------------- Callback groups ----------------
        self.timer_group = MutuallyExclusiveCallbackGroup()
        self.subscription_group = MutuallyExclusiveCallbackGroup()

        # ---------------- QoS ----------------
        qos_reliable = bool(self._param('sensor_qos_reliable', True))
        self.sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT if qos_reliable else ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        enable_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)

        # ---------------- Timing / Vive gains ----------------
        self.period_s = float(self._param('period', 0.005))
        self.vive_scale = float(self._param('vive_scale', 0.2))
        self.vive_deadzone = float(self._param('vive_deadzone', 0.002))

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

        self.jaw_cp_topic = f'/{self.arm_name}/jaw/measured_js'
        self.jaw_cp_sub = self.create_subscription(
            JointState,
            self.jaw_cp_topic,
            self.jaw_cp_callback,
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

        self.vive_sub = self.create_subscription(
            PoseStamped,
            '/vive_tracker/pose',
            self.vive_cb,
            self.sensor_qos,
            callback_group=self.subscription_group,
        )

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
                self.vive_ref_pos = None
                self.R_ecm_from_cart_latched = None
                self.R_ecm_from_vive = None
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
        # Keep keyboard listener alive for parity with the orientation-only script.
        # Toggle publishing is handled by the companion publisher node below.
        pass

    def vive_scale_delta_cb(self, delta: float):
        with self.state_lock:
            self.vive_scale = max(0.0, float(self.vive_scale) + float(delta))
            vive_scale = self.vive_scale
        self.get_logger().info(f'Vive scale adjusted to {vive_scale:.2f}')

    def psm_cp_callback(self, msg: PoseStamped):
        with self.state_lock:
            self.psm_pose = msg
            self._warned_waiting_psm_pose = False

    def jaw_cp_callback(self, msg):
        with self.state_lock:
            if msg.position:
                self.jaw_pose = msg.position[0]  # First joint (jaw angle)
                self._warned_waiting_jaw_pose = False

    def ecm_cp_callback(self, msg: PoseStamped):
        q = msg.pose.orientation
        with self.state_lock:
            self.ecm_q = (q.w, q.x, q.y, q.z)
            self._warned_waiting_ecm = False

    def vive_cb(self, msg: PoseStamped):
        vive_pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ], dtype=float)
        with self.state_lock:
            self.vive_current_pos = vive_pos
            self._warned_waiting_vive = False

    # ------------------------------------------------------------------
    # TF / frame helpers
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

    def _lookup_cart_to_ecm_rotation(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                'ECM',
                'Cart',
                Time(),
                timeout=Duration(seconds=0.0),
            )
            q = transform.transform.rotation
            self._warned_waiting_cart_ecm_tf = False
            return PyKDL.Rotation.Quaternion(q.x, q.y, q.z, q.w)
        except Exception as exc:
            if not self._warned_waiting_cart_ecm_tf:
                self.get_logger().warning(f'Waiting for TF Cart->ECM: {exc}')
                self._warned_waiting_cart_ecm_tf = True
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

    def _compute_vive_translation(self, vive_current, vive_ref, vive_scale, vive_deadzone, R_ecm_from_vive):
        if vive_current is None or vive_ref is None or R_ecm_from_vive is None:
            return PyKDL.Vector.Zero()

        disp_vive = (vive_current - vive_ref).copy()
        for i in range(3):
            if abs(disp_vive[i]) < vive_deadzone:
                disp_vive[i] = 0.0
        disp_vive *= vive_scale

        v = PyKDL.Vector(float(disp_vive[0]), float(disp_vive[1]), float(disp_vive[2]))
        return R_ecm_from_vive * v

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
            jaw_pose_msg = self.jaw_pose
            ecm_q = self.ecm_q
            vive_current = None if self.vive_current_pos is None else self.vive_current_pos.copy()
            vive_scale = self.vive_scale
            vive_deadzone = self.vive_deadzone

        if psm_pose_msg is None:
            if not self._warned_waiting_psm_pose:
                self.get_logger().warning('Waiting for dVRK measured_cp before latching teleop reference')
                self._warned_waiting_psm_pose = True
            return

        if jaw_pose_msg is None:
            if not self._warned_waiting_jaw_pose:
                self.get_logger().warning('Waiting for dVRK jaw measured_js before latching teleop reference')
                self._warned_waiting_jaw_pose = True
            return

        if ecm_q is None:
            if not self._warned_waiting_ecm:
                self.get_logger().warning('Waiting for ECM measured_cp before running teleop')
                self._warned_waiting_ecm = True
            return

        if vive_current is None:
            if not self._warned_waiting_vive:
                self.get_logger().warning('Waiting for Vive tracker pose before running teleop')
                self._warned_waiting_vive = True
            return

        q_04 = self._lookup_gimbal_q04()
        if q_04 is None:
            return

        gimbal_q = self.gimbal_to_ecm(q_04, ecm_q)
        self._publish_debug_tf(gimbal_q)

        if not initialized:
            R_ecm_from_cart = self._lookup_cart_to_ecm_rotation()
            if R_ecm_from_cart is None:
                return

            q = psm_pose_msg.pose.orientation
            p = psm_pose_msg.pose.position
            psm_rot = PyKDL.Rotation.Quaternion(q.x, q.y, q.z, q.w)
            psm_pos = PyKDL.Vector(p.x, p.y, p.z)

            w, x, y, z = gimbal_q
            R_gimbal_ref = PyKDL.Rotation.Quaternion(x, y, z, w)
            psm_ref_pose = PyKDL.Frame(psm_rot, psm_pos)
            vive_ref_pos = vive_current.copy()
            R_ecm_from_vive = R_ecm_from_cart * self.R_cart_from_trak

            with self.state_lock:
                self.psm_ref_pose = psm_ref_pose
                self.jaw_ref_pose = jaw_pose_msg
                self.R_gimbal_ref = R_gimbal_ref
                self.vive_ref_pos = vive_ref_pos
                self.R_ecm_from_cart_latched = R_ecm_from_cart
                self.R_ecm_from_vive = R_ecm_from_vive
                self.initialized = True

            self.get_logger().info('Gimbal orientation and Vive translation references latched')
            return

        with self.state_lock:
            psm_ref_pose = None if self.psm_ref_pose is None else PyKDL.Frame(self.psm_ref_pose.M, self.psm_ref_pose.p)
            jaw_ref_pose = None if self.jaw_ref_pose is None else self.jaw_ref_pose
            R_gimbal_ref = self.R_gimbal_ref
            vive_ref = None if self.vive_ref_pos is None else self.vive_ref_pos.copy()
            R_ecm_from_vive = self.R_ecm_from_vive

        if psm_ref_pose is None or R_gimbal_ref is None or vive_ref is None or R_ecm_from_vive is None:
            return

        w, x, y, z = gimbal_q
        R_gimbal_curr = PyKDL.Rotation.Quaternion(x, y, z, w)

        R_delta = R_gimbal_ref.Inverse() * R_gimbal_curr
        R_goal = psm_ref_pose.M * R_delta

        vive_disp = self._compute_vive_translation(
            vive_current,
            vive_ref,
            vive_scale,
            vive_deadzone,
            R_ecm_from_vive,
        )
        p_goal = psm_ref_pose.p + vive_disp

        goal = PyKDL.Frame(R_goal, p_goal)
        self.arm.servo_cp(goal)

class TeleopKeyboardPublisher(Node):
    def __init__(self, teleop_node, topic='/dvrk_teleop_gimbal/enable'):
        super().__init__('dvrk_teleop_keyboard_publisher')
        self.teleop_node = teleop_node

        self.arm_name = str(self._param('dvrk_arm', self._param('arm', 'PSM1')))
        self.jaw_rate_rad_s = float(self._param('jaw_key_rate', 1.0))
        self.jaw_min_rad = float(self._param('jaw_min', math.radians(-20.0)))
        self.jaw_max_rad = float(self._param('jaw_max', math.radians(80.0)))
        self.jaw_loop_dt = float(self._param('jaw_loop_dt', 0.02))
        self.vive_scale_step = float(self._param('vive_scale_step', 0.1))

        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub = self.create_publisher(Bool, topic, qos)
        self.jaw_cmd_pub = self.create_publisher(JointState, f'/{self.arm_name}/jaw/servo_jp', 10)
        self.jaw_measured_sub = self.create_subscription(
            JointState,
            f'/{self.arm_name}/jaw/measured_js',
            self._jaw_measured_cb,
            10,
        )

        self.enabled = False
        self.keys_down = set()
        self.key_lock = threading.Lock()
        self.jaw_inc_down = False
        self.jaw_dec_down = False
        self.jaw_target = None
        self.jaw_measured = None
        self.last_jaw_update_t = time.monotonic()
        self.jaw_timer = self.create_timer(self.jaw_loop_dt, self._jaw_key_step)

        self.get_logger().info("Keyboard teleop: '2' toggles teleop, hold '4' to open jaw, hold '1' to close jaw, Up/Down adjusts Vive scale")

        self.listener = keyboard.Listener(on_press=self.on_key_press, on_release=self.on_key_release)
        self.listener.start()

    def _param(self, name, default):
        if not self.has_parameter(name):
            self.declare_parameter(name, default)
        return self.get_parameter(name).value

    def _jaw_measured_cb(self, msg: JointState):
        if not msg.position:
            return
        measured = float(msg.position[0])
        self.jaw_measured = measured
        if self.jaw_target is None:
            self.jaw_target = measured

    def _publish_jaw_position(self, angle_rad):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.position = [float(angle_rad)]
        self.jaw_cmd_pub.publish(msg)

    def _jaw_key_step(self):
        now = time.monotonic()
        dt = now - self.last_jaw_update_t
        self.last_jaw_update_t = now

        with self.key_lock:
            inc = self.jaw_inc_down
            dec = self.jaw_dec_down

        if not inc and not dec:
            return

        if self.jaw_target is None:
            if self.jaw_measured is None:
                return
            self.jaw_target = self.jaw_measured

        delta = self.jaw_rate_rad_s * dt
        if inc and not dec:
            self.jaw_target += delta
        elif dec and not inc:
            self.jaw_target -= delta

        self.jaw_target = min(self.jaw_max_rad, max(self.jaw_min_rad, self.jaw_target))
        self._publish_jaw_position(self.jaw_target)

    def _matches_key(self, key, target_char):
        # Handle top-row digits and keypad digits across layouts.
        try:
            if key.char == target_char:
                return True
        except AttributeError:
            pass

        vk = getattr(key, 'vk', None)
        if target_char == '4':
            return vk in (52, 65460)
        if target_char == '1':
            return vk in (49, 65457)
        return False

    def on_key_press(self, key):
        with self.key_lock:
            self.keys_down.add(key)
            if self._matches_key(key, '4'):
                self.jaw_inc_down = True
            if self._matches_key(key, '1'):
                self.jaw_dec_down = True

        if key == keyboard.Key.up:
            self.teleop_node.vive_scale_delta_cb(self.vive_scale_step)
            self.get_logger().info(f'Vive scale +{self.vive_scale_step:.2f}')
            return

        if key == keyboard.Key.down:
            self.teleop_node.vive_scale_delta_cb(-self.vive_scale_step)
            self.get_logger().info(f'Vive scale -{self.vive_scale_step:.2f}')
            return

        try:
            if key.char == '2':
                self.enabled = not self.enabled
                msg = Bool()
                msg.data = self.enabled
                self.pub.publish(msg)
                self.get_logger().info(f'Teleop enable = {self.enabled}')
            elif key.char == '4':
                self.get_logger().info('Jaw opening while key is held')
            elif key.char == '1':
                self.get_logger().info('Jaw closing while key is held')
        except AttributeError:
            pass

    def on_key_release(self, key):
        with self.key_lock:
            self.keys_down.discard(key)
            if self._matches_key(key, '4'):
                self.jaw_inc_down = False
            if self._matches_key(key, '1'):
                self.jaw_dec_down = False


def main():
    rclpy.init()

    ral = crtk.ral('dvrk_teleop_gimbal_orientation_vive_crtk')
    teleop = DVRKTeleopGimbalOrientationVive(ral)
    keyboard_node = TeleopKeyboardPublisher(teleop)

    teleop.get_logger().info('dvrk_teleop_gimbal_orientation_vive node started')

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
