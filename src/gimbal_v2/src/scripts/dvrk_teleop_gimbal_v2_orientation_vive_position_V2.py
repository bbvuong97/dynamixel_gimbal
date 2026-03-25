#!/usr/bin/env python3

import math
import threading

import numpy as np
import PyKDL
# from pynput import keyboard

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.executors import SingleThreadedExecutor, MultiThreadedExecutor

from geometry_msgs.msg import Quaternion, PoseStamped, TransformStamped
from std_msgs.msg import Bool

from tf2_ros import Buffer, TransformListener, TransformBroadcaster, TransformException

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
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]
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
# ROS 2 teleop node
# ---------------------------------------------------------------------------

class DVRKTeleopGimbal(Node):
    def __init__(self):
        super().__init__('dvrk_teleop_gimbal')

        # ---------------- Parameters ----------------
        self.arm_name = str(self._param("arm", "PSM1"))
        self.period_s = float(self._param("period", 0.02))
        self.use_multithread = self._param_bool("use_multithread", False)
        self.executor_threads = int(self._param("executor_threads", 2))

        self.vive_scale = float(self._param("vive_scale", 0.2))
        self.vive_deadzone = float(self._param("vive_deadzone", 0.002))

        self.get_logger().info(f'Using dVRK arm: {self.arm_name}')
        self.get_logger().info(f'Loop period: {self.period_s:.6f} s')

        # ---------------- dVRK / CRTK ----------------
        self.ral = crtk.ral(f'{self.get_name()}_crtk')
        self.arm = dvrk.psm(self.ral, self.arm_name)

        self.ral.check_connections()
        # self.ral.spin()

        # self.arm.enable()
        # self.arm.home()

        # ---------------- State ----------------
        self.state_lock = threading.Lock()

        self.teleop_active = False
        # self.toggle_requested = False
        self.requested_enable = None

        self.psm_pose = None
        self.ecm_q = None
        self.vive_current_pos = None

        self.psm_ref_pose = None
        self.R_gimbal_ref = None
        self.vive_ref_pos = None
        self.R_ecm_from_cart_latched = None
        self.R_ecm_from_vive = None

        self._warned_waiting_vive = False
        self._warned_waiting_psm = False
        self._warned_waiting_ecm = False

        self.R_Cart_GimbalBase = np.array([
            [0, -1, 0],
            [1,  0, 0],
            [0,  0, 1]
        ])

        # Vive (steamvr) -> Cart
        self.R_cart_from_trak = PyKDL.Rotation(
            1.0, 0.0,  0.0,
            0.0, 0.0, -1.0,
            0.0, 1.0,  0.0
        )

        # ---------------- ROS pubs/subs/timer ----------------
        # self.enable_pub = self.create_publisher(Bool, '/dvrk_teleop_gimbal/enable', 1)

        self.enable_sub = self.create_subscription(
            Bool,
            '/dvrk_teleop_gimbal/enable',
            self.enable_cb,
            10
        )

        self.psm_cp_sub = self.create_subscription(
            PoseStamped,
            f'/{self.arm_name}/measured_cp',
            self.psm_cp_callback,
            10
        )

        self.ecm_cp_sub = self.create_subscription(
            PoseStamped,
            '/ECM/measured_cp',
            self.ecm_cp_callback,
            10
        )

        self.vive_sub = self.create_subscription(
            PoseStamped,
            '/vive_tracker/pose',
            self.vive_cb,
            10
        )

        self.timer = self.create_timer(self.period_s, self.timer_callback)

        self.arm.enable()
        self.arm.home()

        # ---------------- TF2 ----------------
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        # # ---------------- Keyboard ----------------
        # self.keyboard_listener = keyboard.Listener(on_press=self.on_key_press)
        # self.keyboard_listener.daemon = True
        # self.keyboard_listener.start()

        # self.get_logger().info("Keyboard teleop: 't' toggles enable, Up/Down changes Vive scale")

    # ----------------------------------------------------------------------
    # Parameter helpers
    # ----------------------------------------------------------------------

    def _param(self, name, default):
        if not self.has_parameter(name):
            self.declare_parameter(name, default)
        return self.get_parameter(name).value

    def _param_bool(self, name, default):
        value = self._param(name, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in ("1", "true", "yes", "on"):
                return True
            if normalized in ("0", "false", "no", "off"):
                return False
            return bool(default)
        return bool(value)

    # ----------------------------------------------------------------------
    # ROS callbacks
    # ----------------------------------------------------------------------

    # def enable_cb(self, msg: Bool):
    #     with self.state_lock:
    #         desired_enable = bool(msg.data)
    #         already_active = self.teleop_active

    #         if desired_enable == already_active:
    #             return

    #         # Reuse existing toggle path
    #         self.toggle_requested = True
    def enable_cb(self, msg: Bool):
        self.get_logger().info(f"enable_cb received: {msg.data}")
        with self.state_lock:
            self.requested_enable = bool(msg.data)

    def vive_cb(self, msg: PoseStamped):
        vive_pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ], dtype=float)

        with self.state_lock:
            self.vive_current_pos = vive_pos
            self._warned_waiting_vive = False

    def psm_cp_callback(self, msg: PoseStamped):
        with self.state_lock:
            self.psm_pose = msg
            self._warned_waiting_psm = False

    def ecm_cp_callback(self, msg: PoseStamped):
        ecm_q = (
            msg.pose.orientation.w,
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z
        )
        with self.state_lock:
            self.ecm_q = ecm_q
            self._warned_waiting_ecm = False

    # def on_key_press(self, key):
    #     try:
    #         if key == keyboard.Key.up:
    #             with self.state_lock:
    #                 self.vive_scale += 0.05
    #                 scale = self.vive_scale
    #             self.get_logger().info(f'Vive scale increased: {scale:.2f}')
    #             return

    #         if key == keyboard.Key.down:
    #             with self.state_lock:
    #                 self.vive_scale = max(0.0, self.vive_scale - 0.05)
    #                 scale = self.vive_scale
    #             self.get_logger().info(f'Vive scale decreased: {scale:.2f}')
    #             return

    #         if hasattr(key, 'char') and key.char == 't':
    #             with self.state_lock:
    #                 self.toggle_requested = True
    #             return

    #     except Exception as exc:
    #         self.get_logger().warning(f'Keyboard handler error: {exc}')

    # ----------------------------------------------------------------------
    # TF / frame helpers
    # ----------------------------------------------------------------------

    def _tf_rotation(self, from_frame: str, to_frame: str, timeout_sec: float = 0.05):
        try:
            trans = self.tf_buffer.lookup_transform(
                to_frame,
                from_frame,
                Time(),
                timeout=Duration(seconds=timeout_sec)
            )
            q = trans.transform.rotation
            return PyKDL.Rotation.Quaternion(q.x, q.y, q.z, q.w)
        except Exception as exc:
            self.get_logger().warning(f'TF lookup {from_frame}->{to_frame} failed: {exc}')
            return None

    def gimbal_to_ecm(self, gimbal_q_04):
        with self.state_lock:
            ecm_q = self.ecm_q

        if ecm_q is None:
            return None

        R_GimbalBase_Cart = self.R_Cart_GimbalBase.T
        R_Cart_ECM = quat_to_rot(ecm_q).T
        R_Gimbal4_GimbalBase = quat_to_rot(gimbal_q_04)

        R_Gimbal4_ECM = R_Cart_ECM @ R_GimbalBase_Cart @ R_Gimbal4_GimbalBase
        return rot_to_quat(R_Gimbal4_ECM)

    def _broadcast_gimbal_tf(self, gimbal_q):
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

    # ----------------------------------------------------------------------
    # Toggle / latch logic
    # ----------------------------------------------------------------------

    # def _handle_toggle_request(self, gimbal_q):
    #     with self.state_lock:
    #         self.toggle_requested = False
    #         enabling = not self.teleop_active
    #         psm_pose_msg = self.psm_pose
    #         vive_current = None if self.vive_current_pos is None else self.vive_current_pos.copy()

    #     if enabling:
    #         if psm_pose_msg is None:
    #             self.get_logger().warning("No cached measured_cp yet; cannot start teleop")
    #             return

    #         if vive_current is None:
    #             self.get_logger().warning("No Vive pose yet; cannot start teleop")
    #             return

    #         R_ecm_from_cart = self._tf_rotation("Cart", "ECM", timeout_sec=0.05)
    #         if R_ecm_from_cart is None:
    #             self.get_logger().warning("Could not latch Cart->ECM transform at teleop start")
    #             return

    #         q = psm_pose_msg.pose.orientation
    #         p = psm_pose_msg.pose.position
    #         psm_rot = PyKDL.Rotation.Quaternion(q.x, q.y, q.z, q.w)
    #         psm_pos = PyKDL.Vector(p.x, p.y, p.z)

    #         gw, gx, gy, gz = gimbal_q
    #         R_gimbal_ref = PyKDL.Rotation.Quaternion(gx, gy, gz, gw)

    #         with self.state_lock:
    #             self.psm_ref_pose = PyKDL.Frame(psm_rot, psm_pos)
    #             self.R_gimbal_ref = R_gimbal_ref
    #             self.vive_ref_pos = vive_current
    #             self.R_ecm_from_cart_latched = R_ecm_from_cart
    #             self.R_ecm_from_vive = self.R_ecm_from_cart_latched * self.R_cart_from_trak
    #             self.teleop_active = True

    #         msg = Bool()
    #         msg.data = True
    #         # self.enable_pub.publish(msg)
    #         self.get_logger().info("Teleop enable = True (references latched)")

    #     else:
    #         with self.state_lock:
    #             self.teleop_active = False

    #         msg = Bool()
    #         msg.data = False
    #         # self.enable_pub.publish(msg)
    #         self.get_logger().info("Teleop enable = False")

    def _handle_enable_request(self, gimbal_q, enable_requested: bool):
        self.get_logger().info(f"_handle_enable_request called with enable_requested={enable_requested}")
        with self.state_lock:
            psm_pose_msg = self.psm_pose
            vive_current = None if self.vive_current_pos is None else self.vive_current_pos.copy()
            already_active = self.teleop_active

        if enable_requested == already_active:
            return

        if enable_requested:
            if psm_pose_msg is None:
                self.get_logger().warning("No cached measured_cp yet; cannot start teleop")
                return

            if vive_current is None:
                self.get_logger().warning("No Vive pose yet; cannot start teleop")
                return

            R_ecm_from_cart = self._tf_rotation("Cart", "ECM", timeout_sec=0.05)
            if R_ecm_from_cart is None:
                self.get_logger().warning("Could not latch Cart->ECM transform at teleop start")
                return

            q = psm_pose_msg.pose.orientation
            p = psm_pose_msg.pose.position
            psm_rot = PyKDL.Rotation.Quaternion(q.x, q.y, q.z, q.w)
            psm_pos = PyKDL.Vector(p.x, p.y, p.z)

            gw, gx, gy, gz = gimbal_q
            R_gimbal_ref = PyKDL.Rotation.Quaternion(gx, gy, gz, gw)

            with self.state_lock:
                self.psm_ref_pose = PyKDL.Frame(psm_rot, psm_pos)
                self.R_gimbal_ref = R_gimbal_ref
                self.vive_ref_pos = vive_current
                self.R_ecm_from_cart_latched = R_ecm_from_cart
                self.R_ecm_from_vive = self.R_ecm_from_cart_latched * self.R_cart_from_trak
                self.teleop_active = True

            self.get_logger().info("Teleop enable = True (references latched)")

        else:
            with self.state_lock:
                self.teleop_active = False

            self.get_logger().info("Teleop enable = False")

    # ----------------------------------------------------------------------
    # Main timer
    # ----------------------------------------------------------------------

    def timer_callback(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                'gimbal_base',
                'rcm',
                Time(),
                timeout=Duration(seconds=0.05)
            )
        except TransformException:
            return

        q_04 = (
            transform.transform.rotation.w,
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z
        )

        gimbal_q = self.gimbal_to_ecm(q_04)
        if gimbal_q is None:
            with self.state_lock:
                if not self._warned_waiting_ecm:
                    self.get_logger().warning("Waiting for ECM measured_cp before gimbal orientation can be computed")
                    self._warned_waiting_ecm = True
            return

        self._broadcast_gimbal_tf(gimbal_q)

        # with self.state_lock:
        #     toggle_requested = self.toggle_requested
        #     teleop_active = self.teleop_active

        # if toggle_requested:
        #     self._handle_toggle_request(gimbal_q)
        #     with self.state_lock:
        #         teleop_active = self.teleop_active

        with self.state_lock:
            requested_enable = self.requested_enable
            self.requested_enable = None
            teleop_active = self.teleop_active

        if requested_enable is not None:
            self._handle_enable_request(gimbal_q, requested_enable)
            with self.state_lock:
                teleop_active = self.teleop_active

        if teleop_active:
            self.teleop(gimbal_q)

    def teleop(self, gimbal_q):
        with self.state_lock:
            teleop_active = self.teleop_active
            psm_ref_pose = self.psm_ref_pose
            R_gimbal_ref = self.R_gimbal_ref
            vive_current = None if self.vive_current_pos is None else self.vive_current_pos.copy()
            vive_ref = None if self.vive_ref_pos is None else self.vive_ref_pos.copy()
            vive_scale = self.vive_scale
            vive_deadzone = self.vive_deadzone
            R_ecm_from_vive = self.R_ecm_from_vive

        if not teleop_active:
            return
        if psm_ref_pose is None or R_gimbal_ref is None:
            return

        gw, gx, gy, gz = gimbal_q
        R_gimbal_curr = PyKDL.Rotation.Quaternion(gx, gy, gz, gw)

        R_delta = R_gimbal_ref.Inverse() * R_gimbal_curr
        R_goal = psm_ref_pose.M * R_delta

        vive_disp = self._compute_vive_translation(
            vive_current,
            vive_ref,
            vive_scale,
            vive_deadzone,
            R_ecm_from_vive
        )
        p_goal = psm_ref_pose.p + vive_disp

        goal = PyKDL.Frame(R_goal, p_goal)
        self.get_logger().info("teleop(): sending servo_cp")
        self.arm.servo_cp(goal)

    # ----------------------------------------------------------------------
    # Cleanup
    # ----------------------------------------------------------------------

    # def shutdown(self):
    #     try:
    #         if self.keyboard_listener is not None:
    #             self.keyboard_listener.stop()
    #     except Exception:
    #         pass

    def shutdown(self):
        pass

    def destroy_node(self):
        self.shutdown()

        try:
            if self.ral is not None:
                self.ral.shutdown()
        except Exception:
            pass

        super().destroy_node()


def main():
    rclpy.init()
    node = DVRKTeleopGimbal()

    if node.use_multithread:
        executor = MultiThreadedExecutor(num_threads=max(1, node.executor_threads))
    else:
        executor = SingleThreadedExecutor()

    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()