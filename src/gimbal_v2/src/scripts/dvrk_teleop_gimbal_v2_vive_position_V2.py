#!/usr/bin/env python3

import argparse
import threading
import time

import numpy as np
import PyKDL
from pynput import keyboard

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.duration import Duration
from rclpy.time import Time

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

import crtk
import dvrk


class DVRKTeleopVivePosition(Node):
    def __init__(self, ral):
        super().__init__('dvrk_teleop_vive_position_node')

        self._ral = ral
        self.arm_name = self._param("dvrk_arm", "PSM1")
        self.get_logger().info(f"Using dVRK arm: {self.arm_name}")

        self.arm = dvrk.psm(ral, self.arm_name)
        self.arm.enable()
        self.arm.home()

        # ---------------- State ----------------
        self.teleop_active = False
        self.initialized = False

        self.psm_pose = None
        self.psm_ref_pose = None

        self.vive_current_pos = None
        self.vive_ref_pos = None

        self.vive_scale = float(self._param("vive_scale", 0.2))
        self.vive_deadzone = float(self._param("vive_deadzone", 0.002))

        self._warned_waiting_psm_pose = False
        self._warned_waiting_vive = False
        self._tf_warned_cart_to_ecm = False

        # Same mapping as your previous Vive-only script
        self.R_cart_from_trak = PyKDL.Rotation(
            1.0, 0.0, 0.0,
            0.0, 0.0, -1.0,
            0.0, 1.0, 0.0
        )
        self.R_ecm_from_cart = None
        self.R_ecm_from_vive = None

        # Shared state touched by both ROS callbacks and pynput thread
        self.state_lock = threading.Lock()

        # ---------------- Callback groups ----------------
        self.timer_group = MutuallyExclusiveCallbackGroup()
        self.subscription_group = MutuallyExclusiveCallbackGroup()

        # ---------------- QoS ----------------
        qos_reliable = bool(self._param("sensor_qos_reliable", True))
        self.sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE if qos_reliable else ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.vive_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        # ---------------- Timing ----------------
        self.period_s = float(self._param("period", 0.005))
        self.timer = self.create_timer(
            self.period_s,
            self.timer_callback,
            callback_group=self.timer_group
        )

        # ---------------- Subscriptions ----------------
        self.psm_cp_topic = f"/{self.arm_name}/measured_cp"
        self.psm_cp_sub = self.create_subscription(
            PoseStamped,
            self.psm_cp_topic,
            self.psm_cp_callback,
            self.sensor_qos,
            callback_group=self.subscription_group,
        )

        self.vive_sub = self.create_subscription(
            PoseStamped,
            "/vive_tracker/pose",
            self.vive_cb,
            self.vive_qos,
            callback_group=self.subscription_group,
        )

        # Publish teleop enable exactly like your non-V2 structure
        enable_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.teleop_enable_pub = self.create_publisher(
            Bool,
            "/dvrk_teleop_vive_position/enable",
            enable_qos
        )

        # Optional loopback subscriber
        self.teleop_enable_sub = self.create_subscription(
            Bool,
            "/dvrk_teleop_vive_position/enable",
            self.teleop_enable_cb,
            enable_qos,
            callback_group=self.subscription_group,
        )

        # ---------------- TF ----------------
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.R_ecm_from_cart_latched = None

        # ---------------- Keyboard ----------------
        self.keyboard_listener = keyboard.Listener(on_press=self.on_key_press)
        self.keyboard_listener.start()

        self._key_verbose = bool(self._param("key_verbose", True))
        self._key_repeat_min_period = float(self._param("key_repeat_min_period", 0.03))
        self._last_key_event_t = {}

        self.get_logger().info("Keyboard teleop: press 't' to toggle enable, arrow keys to adjust")

    def _param(self, name, default):
        if not self.has_parameter(name):
            self.declare_parameter(name, default)
        return self.get_parameter(name).value

    # ------------------------------------------------------------------
    # Keyboard handling copied in structure from non-V2
    # ------------------------------------------------------------------
    def on_key_press(self, key):
        now = time.monotonic()

        if key in (keyboard.Key.up, keyboard.Key.down):
            last_t = self._last_key_event_t.get(key, 0.0)
            if now - last_t < self._key_repeat_min_period:
                return
            self._last_key_event_t[key] = now

        try:
            # if key.char == 't':
            #     with self.state_lock:
            #         self.teleop_active = not self.teleop_active
            #         if self.teleop_active:
            #             self.initialized = False
            #     msg = Bool()
            #     msg.data = self.teleop_active
            #     self.teleop_enable_pub.publish(msg)
            #     self.get_logger().info(f"Teleop enable = {self.teleop_active}")

            #     with self.state_lock:
            #         R_ecm_from_cart = self.R_ecm_from_cart_latched

            #     if R_ecm_from_cart is None:
            #         self.get_logger().warning("No latched Cart->ECM transform available")
            #         return

            #     self.R_ecm_from_cart_latched = R_ecm_from_cart
            #     self.R_ecm_from_vive = self.R_ecm_from_cart_latched * self.R_cart_from_trak

            if key.char == 't':
                with self.state_lock:
                    enabling = not self.teleop_active

                if enabling:
                    with self.state_lock:
                        psm_pose_msg = self.psm_pose
                        vive_current = None if self.vive_current_pos is None else self.vive_current_pos.copy()

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

                    with self.state_lock:
                        self.psm_ref_pose = PyKDL.Frame(psm_rot, psm_pos)
                        self.vive_ref_pos = vive_current
                        self.R_ecm_from_cart_latched = R_ecm_from_cart
                        self.R_ecm_from_vive = self.R_ecm_from_cart_latched * self.R_cart_from_trak
                        self.initialized = True
                        self.teleop_active = True

                    # if self.psm_cp_sub is not None:
                    #     self.get_logger().info("Teleop latched; destroying measured_cp subscription")
                    #     self.destroy_subscription(self.psm_cp_sub)
                    #     self.psm_cp_sub = None

                    msg = Bool()
                    msg.data = True
                    self.teleop_enable_pub.publish(msg)
                    self.get_logger().info("Teleop enable = True")

                else:
                    with self.state_lock:
                        self.teleop_active = False
                        self.initialized = False

                    msg = Bool()
                    msg.data = False
                    self.teleop_enable_pub.publish(msg)
                    self.get_logger().info("Teleop enable = False")

        except AttributeError:
            if key == keyboard.Key.up:
                with self.state_lock:
                    self.vive_scale += 0.05
                    current_scale = self.vive_scale
                if self._key_verbose:
                    self.get_logger().info(f"Vive scale increased: {current_scale:.2f}")

            elif key == keyboard.Key.down:
                with self.state_lock:
                    self.vive_scale = max(0.0, self.vive_scale - 0.05)
                    current_scale = self.vive_scale
                if self._key_verbose:
                    self.get_logger().info(f"Vive scale decreased: {current_scale:.2f}")

    def teleop_enable_cb(self, msg: Bool):
        # Keep local state consistent if something external publishes too
        with self.state_lock:
            if msg.data != self.teleop_active:
                self.teleop_active = msg.data
                if self.teleop_active:
                    self.initialized = False
        self.get_logger().info(f"Teleop loopback state = {msg.data}")

    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------
    # def psm_cp_callback(self, msg: PoseStamped):
    #     self.psm_pose = msg
    #     self._warned_waiting_psm_pose = False

    def psm_cp_callback(self, msg: PoseStamped):
        with self.state_lock:
            self.psm_pose = msg

    def vive_cb(self, msg: PoseStamped):
        vive_pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ], dtype=float)

        with self.state_lock:
            self.vive_current_pos = vive_pos
            self._warned_waiting_vive = False

    # ------------------------------------------------------------------
    # TF helpers
    # ------------------------------------------------------------------
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
            if not self._tf_warned_cart_to_ecm:
                self.get_logger().warning(f"TF lookup {from_frame}->{to_frame} failed: {exc}")
                self._tf_warned_cart_to_ecm = True
            return None

    def compute_vive_translation(self):
        with self.state_lock:
            vive_current = None if self.vive_current_pos is None else self.vive_current_pos.copy()
            vive_ref = None if self.vive_ref_pos is None else self.vive_ref_pos.copy()
            vive_scale = self.vive_scale
            vive_deadzone = self.vive_deadzone
            R_ecm_from_vive = self.R_ecm_from_vive

        if vive_current is None or vive_ref is None or R_ecm_from_vive is None:
            return PyKDL.Vector.Zero()

        disp_vive = vive_current - vive_ref

        for i in range(3):
            if abs(disp_vive[i]) < vive_deadzone:
                disp_vive[i] = 0.0

        disp_vive *= vive_scale

        v_vive = PyKDL.Vector(
            float(disp_vive[0]),
            float(disp_vive[1]),
            float(disp_vive[2])
        )

        return R_ecm_from_vive * v_vive

    # ------------------------------------------------------------------
    # Teleop
    # ------------------------------------------------------------------
    # def teleop(self):
    #     if self.psm_pose is None:
    #         if not self._warned_waiting_psm_pose:
    #             self.get_logger().warning("Waiting for dVRK measured_cp before latching teleop reference")
    #             self._warned_waiting_psm_pose = True
    #         return

    #     with self.state_lock:
    #         initialized = self.initialized

    #     if not initialized:
    #         q = self.psm_pose.pose.orientation
    #         p = self.psm_pose.pose.position
    #         psm_rot = PyKDL.Rotation.Quaternion(q.x, q.y, q.z, q.w)
    #         psm_pos = PyKDL.Vector(p.x, p.y, p.z)

    #         R_ecm_from_cart = self._tf_rotation("Cart", "ECM", timeout_sec=0.05)
    #         if R_ecm_from_cart is None:
    #             self.get_logger().warning("Waiting for Cart->ECM transform before latching teleop reference")
    #             return

    #         with self.state_lock:
    #             self.psm_ref_pose = PyKDL.Frame(psm_rot, psm_pos)

    #             if self.vive_current_pos is None:
    #                 need_vive = True
    #             else:
    #                 self.vive_ref_pos = self.vive_current_pos.copy()
    #                 self.R_ecm_from_cart_latched = R_ecm_from_cart
    #                 self.R_ecm_from_vive = self.R_ecm_from_cart_latched * self.R_cart_from_trak
    #                 self.initialized = True
    #                 need_vive = False

    #         if need_vive:
    #             if not self._warned_waiting_vive:
    #                 self.get_logger().warning("Waiting for Vive pose before latching teleop reference")
    #                 self._warned_waiting_vive = True
    #             return

    #         self.get_logger().info("Teleop reference latched")
    #         return

    #     vive_disp = self.compute_vive_translation()

    #     with self.state_lock:
    #         R_goal = self.psm_ref_pose.M
    #         p_goal = self.psm_ref_pose.p + vive_disp

    #     goal = PyKDL.Frame(R_goal, p_goal)
    #     self.arm.servo_cp(goal)

    # def teleop(self):
    #     with self.state_lock:
    #         if not self.initialized or self.psm_ref_pose is None:
    #             return
    #         R_goal = self.psm_ref_pose.M

    #     vive_disp = self.compute_vive_translation()

    #     with self.state_lock:
    #         p_goal = self.psm_ref_pose.p + vive_disp

    #     goal = PyKDL.Frame(R_goal, p_goal)
    #     self.arm.servo_cp(goal)

    def timer_callback(self):
        with self.state_lock:
            active = self.teleop_active
        if not active:
            return
        self.teleop()

    def teleop(self):
        with self.state_lock:
            if not self.initialized or self.psm_ref_pose is None:
                return
            psm_ref_pose = PyKDL.Frame(self.psm_ref_pose.M, self.psm_ref_pose.p)
            vive_current = None if self.vive_current_pos is None else self.vive_current_pos.copy()
            vive_ref = None if self.vive_ref_pos is None else self.vive_ref_pos.copy()
            vive_scale = self.vive_scale
            vive_deadzone = self.vive_deadzone
            R_ecm_from_vive = self.R_ecm_from_vive

        if vive_current is None or vive_ref is None or R_ecm_from_vive is None:
            return

        disp_vive = vive_current - vive_ref
        for i in range(3):
            if abs(disp_vive[i]) < vive_deadzone:
                disp_vive[i] = 0.0
        disp_vive *= vive_scale

        v_vive = PyKDL.Vector(float(disp_vive[0]), float(disp_vive[1]), float(disp_vive[2]))
        vive_disp = R_ecm_from_vive * v_vive

        goal = PyKDL.Frame(psm_ref_pose.M, psm_ref_pose.p + vive_disp)
        self.arm.servo_cp(goal)


# def main():
#     rclpy.init()

#     ral = crtk.ral('dvrk_teleop_vive_position_crtk')
#     teleop = DVRKTeleopVivePosition(ral)

#     teleop.get_logger().info("dvrk_teleop_vive_position node started")

#     executor_threads = int(teleop._param("executor_threads", 2))
#     executor = rclpy.executors.MultiThreadedExecutor(num_threads=max(1, executor_threads))
#     executor.add_node(teleop)

#     try:
#         executor.spin()
#     except KeyboardInterrupt:
#         pass
#     finally:
#         teleop.keyboard_listener.stop()
#         teleop.destroy_node()
#         if rclpy.ok():
#             rclpy.shutdown()

def main():
    rclpy.init()

    ral = crtk.ral('dvrk_teleop_vive_position_crtk')
    teleop = DVRKTeleopVivePosition(ral)

    teleop.get_logger().info("dvrk_teleop_vive_position node started")

    use_multithread = bool(teleop._param("use_multithread", False))
    executor_threads = int(teleop._param("executor_threads", 2))

    if use_multithread:
        executor = rclpy.executors.MultiThreadedExecutor(
            num_threads=max(1, executor_threads)
        )
    else:
        executor = rclpy.executors.SingleThreadedExecutor()

    executor.add_node(teleop)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        teleop.keyboard_listener.stop()
        teleop.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()