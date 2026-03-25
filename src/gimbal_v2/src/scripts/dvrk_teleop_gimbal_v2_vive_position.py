#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, HistoryPolicy, ReliabilityPolicy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.duration import Duration
from rclpy.time import Time
from geometry_msgs.msg import PoseStamped
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

import dvrk
import crtk
import PyKDL
import threading
import time

from std_msgs.msg import Bool
from pynput import keyboard


class DVRKTeleopVivePosition(Node):
    def __init__(self, ral):
        super().__init__('dvrk_teleop_vive_position_node')

        self._ral = ral
        self.get_logger().info(f"Teleop node full name: {self.get_fully_qualified_name()}")

        self.arm_name = self._param("dvrk_arm", "PSM1")
        self.get_logger().info(f"Using dVRK arm: {self.arm_name}")
        self.arm = dvrk.psm(ral, self.arm_name)

        self.arm.enable()
        self.arm.home()

        self.teleop_active = False
        self.initialized = False

        # Use separate callback groups: timer runs independently, subscriptions don't block it
        self.timer_group = MutuallyExclusiveCallbackGroup()
        self.subscription_group = MutuallyExclusiveCallbackGroup()

        qos_reliable = bool(self._param("sensor_qos_reliable", True))
        self.sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE if qos_reliable else ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.period_s = float(self._param("period", 0.005))
        self.timer = self.create_timer(self.period_s, self.timer_callback, callback_group=self.timer_group)
        self.stats_timer = self.create_timer(1.0, self._log_timing_stats, callback_group=self.subscription_group)

        self.psm_cp_topic = f"/{self.arm_name}/measured_cp"
        self.psm_cp_sub = self.create_subscription(
            PoseStamped,
            self.psm_cp_topic,
            self.psm_cp_callback,
            self.sensor_qos,
            callback_group=self.subscription_group,
        )
        self.psm_pose = None
        self.psm_sample_period_s = float(self._param("psm_sample_period", 0.02))
        self._last_psm_store_t = 0.0

        self.vive_sub = self.create_subscription(
            PoseStamped,
            "/vive_tracker/pose",
            self.vive_cb,
            self.sensor_qos,
            callback_group=self.subscription_group,
        )

        # Publish teleop enable state with TRANSIENT_LOCAL QoS (subscribers get last message on connect)
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.teleop_enable_pub = self.create_publisher(Bool, "/dvrk_teleop_gimbal/enable", qos)

        self.vive_current_pos = None
        self.vive_ref_pos = None

        self.vive_scale = float(self._param("vive_scale", 0.2))
        self.vive_deadzone = float(self._param("vive_deadzone", 0.002))

        self._warned_waiting_psm_pose = False
        self._warned_waiting_vive = False
        self._tf_warned_cart_to_ecm = False

        self.R_cart_from_trak = PyKDL.Rotation(
            1.0, 0.0, 0.0,
            0.0, 0.0, -1.0,
            0.0, 1.0, 0.0
        )
        self.R_ecm_from_cart = None
        self.R_ecm_from_vive = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Keyboard listener for all keyboard controls
        self.keyboard_listener = keyboard.Listener(on_press=self.on_key_press)
        self.keyboard_listener.start()

        # Shared state is touched by both ROS callbacks and pynput thread.
        self.state_lock = threading.Lock()

        # Timer diagnostics to track loop jitter and overruns.
        self._last_timer_t = None
        self._timer_calls = 0
        self._timer_overruns = 0
        self._timer_max_dt = 0.0
        self._timer_sum_dt = 0.0
        self._servo_calls = 0
        self._servo_sum_dt = 0.0
        self._servo_max_dt = 0.0
        self._key_verbose = bool(self._param("key_verbose", False))
        self._key_repeat_min_period = float(self._param("key_repeat_min_period", 0.03))
        self._last_key_event_t = {}
        
        self.get_logger().info("Keyboard teleop: press 't' to toggle enable, arrow keys to adjust")

    def _param(self, name, default):
        if not self.has_parameter(name):
            self.declare_parameter(name, default)
        return self.get_parameter(name).value

    def on_key_press(self, key):
        now = time.monotonic()

        # Avoid processing keyboard autorepeat floods faster than desired.
        if key in (keyboard.Key.up, keyboard.Key.down, keyboard.Key.left, keyboard.Key.right):
            last_t = self._last_key_event_t.get(key, 0.0)
            if now - last_t < self._key_repeat_min_period:
                return
            self._last_key_event_t[key] = now

        try:
            if key.char == 't':
                with self.state_lock:
                    self.teleop_active = not self.teleop_active
                    # Re-latch pose reference each time teleop transitions to enabled.
                    if self.teleop_active:
                        self.initialized = False
                msg = Bool()
                msg.data = self.teleop_active
                self.teleop_enable_pub.publish(msg)
                self.get_logger().info(f"Teleop enable = {self.teleop_active}")
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
            elif key == keyboard.Key.left:
                with self.state_lock:
                    if self.initialized and self.psm_ref_pose is not None:
                        self.psm_ref_pose.p[0] -= 0.001
                        x_val = self.psm_ref_pose.p[0]
                    else:
                        x_val = None
                if x_val is not None and self._key_verbose:
                    self.get_logger().info(f"X position decreased: {x_val:.4f}")
            elif key == keyboard.Key.right:
                with self.state_lock:
                    if self.initialized and self.psm_ref_pose is not None:
                        self.psm_ref_pose.p[0] += 0.001
                        x_val = self.psm_ref_pose.p[0]
                    else:
                        x_val = None
                if x_val is not None and self._key_verbose:
                    self.get_logger().info(f"X position increased: {x_val:.4f}")

    def psm_cp_callback(self, msg):
        now = time.monotonic()
        # Throttle high-rate measured_cp callback to reduce Python callback load.
        if now - self._last_psm_store_t < self.psm_sample_period_s:
            return
        self._last_psm_store_t = now

        self.psm_pose = msg.pose
        if self._warned_waiting_psm_pose:
            self.get_logger().info("Received dVRK measured_cp; teleop reference can now be latched")
            self._warned_waiting_psm_pose = False

    def vive_cb(self, msg):
        self.vive_current_pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ])
        if self._warned_waiting_vive:
            self.get_logger().info("Received Vive pose; translation reference can now be latched")
            self._warned_waiting_vive = False

    def _tf_rotation(self, from_frame, to_frame):
        try:
            if not self.tf_buffer.can_transform(to_frame, from_frame, Time(), Duration(seconds=0.0)):
                return PyKDL.Rotation.Identity()

            trans = self.tf_buffer.lookup_transform(
                to_frame, from_frame, Time(), timeout=Duration(seconds=0.0)
            )
            q = trans.transform.rotation
            return PyKDL.Rotation.Quaternion(q.x, q.y, q.z, q.w)
        except Exception as exc:
            if from_frame == "Cart" and to_frame == "ECM":
                if not self._tf_warned_cart_to_ecm:
                    self.get_logger().warning(f"TF lookup {from_frame}->{to_frame} failed: {exc}")
                    self._tf_warned_cart_to_ecm = True
            else:
                self.get_logger().warning(f"TF lookup {from_frame}->{to_frame} failed: {exc}")
            return PyKDL.Rotation.Identity()

    def _ensure_vive_to_ecm_rotation(self):
        if self.R_ecm_from_vive is not None:
            return True

        R_ecm_from_cart = self._tf_rotation("Cart", "ECM")
        if R_ecm_from_cart == PyKDL.Rotation.Identity():
            return False

        self.R_ecm_from_cart = R_ecm_from_cart
        self.R_ecm_from_vive = self.R_ecm_from_cart * self.R_cart_from_trak

        if self._tf_warned_cart_to_ecm:
            self.get_logger().info("TF Cart->ECM available; Vive translation transform initialized")
            self._tf_warned_cart_to_ecm = False

        return True

    def compute_vive_translation(self):
        if self.vive_current_pos is None or self.vive_ref_pos is None:
            return PyKDL.Vector.Zero()

        if not self._ensure_vive_to_ecm_rotation():
            return PyKDL.Vector.Zero()

        disp_vive = self.vive_current_pos - self.vive_ref_pos

        for index in range(3):
            if abs(disp_vive[index]) < self.vive_deadzone:
                disp_vive[index] = 0.0

        disp_vive *= self.vive_scale

        vec = PyKDL.Vector(disp_vive[0], disp_vive[1], disp_vive[2])
        return self.R_ecm_from_vive * vec

    def timer_callback(self):
        now = time.monotonic()
        if self._last_timer_t is not None:
            dt = now - self._last_timer_t
            self._timer_calls += 1
            self._timer_sum_dt += dt
            if dt > self._timer_max_dt:
                self._timer_max_dt = dt
            if dt > 1.5 * self.period_s:
                self._timer_overruns += 1
        self._last_timer_t = now

        with self.state_lock:
            active = self.teleop_active

        if not active:
            return
        self.teleop()

        # # log the setpoint_cp for monitoring using the setpoint_cp function
        # setpoint = self.arm.setpoint_cp()
        # self.get_logger().info(f"Setpoint CP: {setpoint}")


    # def teleop(self):
    #     if self.psm_pose is None:
    #         if not self._warned_waiting_psm_pose:
    #             self.get_logger().warning("Waiting for dVRK measured_cp before latching teleop reference")
    #             self._warned_waiting_psm_pose = True
    #         return

    #     if self.vive_current_pos is None:
    #         if not self._warned_waiting_vive:
    #             self.get_logger().warning("Waiting for Vive pose before latching translation reference")
    #             self._warned_waiting_vive = True
    #         return

    #     if not self.initialized:
    #         q = self.psm_pose.orientation
    #         p = self.psm_pose.position
    #         psm_rot = PyKDL.Rotation.Quaternion(q.x, q.y, q.z, q.w)
    #         psm_pos = PyKDL.Vector(p.x, p.y, p.z)
    #         self.psm_ref_pose = PyKDL.Frame(psm_rot, psm_pos)

    #         self.vive_ref_pos = self.vive_current_pos.copy()

    #         self.initialized = True
    #         self.get_logger().info("Vive translation teleop references latched")
    #         return

    #     vive_disp = self.compute_vive_translation()

    #     R_goal = self.psm_ref_pose.M
    #     p_goal = self.psm_ref_pose.p + vive_disp

    #     goal = PyKDL.Frame(R_goal, p_goal)
    #     self.arm.move_cp(goal)

    #     # log the setpoint_cp for monitoring using the setpoint_cp function
    #     try:
    #         setpoint = self.arm.setpoint_cp()
    #         self.get_logger().info(f"Setpoint CP: {setpoint}")
    #     except TimeoutError:
    #         pass

    def teleop(self):
        if self.psm_pose is None:
            if not self._warned_waiting_psm_pose:
                self.get_logger().warning("Waiting for dVRK measured_cp before latching teleop reference")
                self._warned_waiting_psm_pose = True
            return

        if not self.initialized:
            q = self.psm_pose.orientation
            p = self.psm_pose.position
            psm_rot = PyKDL.Rotation.Quaternion(q.x, q.y, q.z, q.w)
            psm_pos = PyKDL.Vector(p.x, p.y, p.z)
            with self.state_lock:
                self.psm_ref_pose = PyKDL.Frame(psm_rot, psm_pos)
                self.initialized = True
            self.get_logger().info("Teleop reference latched")
            return

        with self.state_lock:
            R_goal = self.psm_ref_pose.M
            p_goal = self.psm_ref_pose.p

        goal = PyKDL.Frame(R_goal, p_goal)
        servo_t0 = time.monotonic()
        self.arm.servo_cp(goal)
        servo_dt = time.monotonic() - servo_t0
        self._servo_calls += 1
        self._servo_sum_dt += servo_dt
        if servo_dt > self._servo_max_dt:
            self._servo_max_dt = servo_dt

    def _log_timing_stats(self):
        if self._timer_calls == 0:
            return

        mean_dt = self._timer_sum_dt / self._timer_calls
        rate = 1.0 / mean_dt if mean_dt > 0.0 else 0.0
        servo_mean_ms = (self._servo_sum_dt / self._servo_calls) * 1000.0 if self._servo_calls > 0 else 0.0
        servo_max_ms = self._servo_max_dt * 1000.0
        self.get_logger().info(
            f"Timer stats: rate={rate:.1f}Hz mean_dt={mean_dt*1000.0:.2f}ms "
            f"max_dt={self._timer_max_dt*1000.0:.2f}ms overruns={self._timer_overruns}/{self._timer_calls} "
            f"servo_mean={servo_mean_ms:.2f}ms servo_max={servo_max_ms:.2f}ms servo_calls={self._servo_calls}"
        )

        self._timer_calls = 0
        self._timer_overruns = 0
        self._timer_max_dt = 0.0
        self._timer_sum_dt = 0.0
        self._servo_calls = 0
        self._servo_sum_dt = 0.0
        self._servo_max_dt = 0.0


def main():
    rclpy.init()

    ral = crtk.ral('dvrk_teleop_gimbal_crtk')
    teleop = DVRKTeleopVivePosition(ral)

    teleop.get_logger().info("dvrk_teleop_gimbal_v2_vive_position node started")

    executor_threads = int(teleop._param("executor_threads", 2))
    executor = rclpy.executors.MultiThreadedExecutor(num_threads=max(1, executor_threads))
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
