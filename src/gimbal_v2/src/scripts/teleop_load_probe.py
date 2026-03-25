#!/usr/bin/env python3
import threading
import time

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped


class TeleopLoadProbe(Node):
    def __init__(self):
        super().__init__("teleop_load_probe")

        self.period_s = float(self._param("period", 0.001))
        self.report_period_s = float(self._param("report_period", 1.0))
        self.enable_keyboard = self._param_bool("enable_keyboard", False)
        self.enable_subscriptions = self._param_bool("enable_subscriptions", False)
        self.enable_dvrk_call = self._param_bool("enable_dvrk_call", False)
        self.active = self._param_bool("active", True)

        self.subscription_topic = str(self._param("subscription_topic", "/PSM1/measured_cp"))
        self.psm_sample_period_s = float(self._param("psm_sample_period", 0.02))
        self.subscription_best_effort = self._param_bool("subscription_best_effort", True)
        self._last_sub_store_t = 0.0

        self.keyboard_x_speed = float(self._param("keyboard_x_speed", 0.02))
        self.keyboard_repeat_min_period = float(self._param("keyboard_repeat_min_period", 0.03))
        self.busy_work_ms = float(self._param("busy_work_ms", 0.0))

        self.timer_group = MutuallyExclusiveCallbackGroup()
        self.subscription_group = MutuallyExclusiveCallbackGroup()

        self.state_lock = threading.Lock()

        self._last_loop_t = None
        self._loop_calls = 0
        self._loop_overruns = 0
        self._loop_sum_dt = 0.0
        self._loop_max_dt = 0.0
        self._loop_min_dt = float("inf")

        self._work_sum_dt = 0.0
        self._work_max_dt = 0.0

        self._sub_callbacks = 0
        self._key_presses = 0

        self._keys_down = set()
        self._last_key_event_t = {}
        self._dummy_x = 0.0

        self._dvrk_arm = None
        self._dvrk_goal = None

        if self.enable_subscriptions:
            sub_qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=ReliabilityPolicy.BEST_EFFORT if self.subscription_best_effort else ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
            )
            self.sub = self.create_subscription(
                PoseStamped,
                self.subscription_topic,
                self._sub_cb,
                sub_qos,
                callback_group=self.subscription_group,
            )

        self._keyboard_listener = None
        if self.enable_keyboard:
            from pynput import keyboard
            self._keyboard = keyboard
            self._keyboard_listener = keyboard.Listener(
                on_press=self._on_key_press,
                on_release=self._on_key_release,
            )
            self._keyboard_listener.start()
        else:
            self._keyboard = None

        if self.enable_dvrk_call:
            self._init_dvrk()

        # Main periodic probe loop is now a ROS timer, not a manual thread.
        self.control_timer = self.create_timer(
            self.period_s,
            self._control_tick,
            callback_group=self.timer_group,
        )

        self.report_timer = self.create_timer(
            self.report_period_s,
            self._report_stats,
            callback_group=self.timer_group,
        )

        self.get_logger().info(
            "Teleop load probe started: "
            f"period={self.period_s:.6f}s "
            f"keyboard={self.enable_keyboard} "
            f"subscriptions={self.enable_subscriptions} "
            f"sub_qos={'BEST_EFFORT' if self.subscription_best_effort else 'RELIABLE'} "
            f"dvrk_call={self.enable_dvrk_call} "
            f"busy_work_ms={self.busy_work_ms:.2f}"
        )

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
            self.get_logger().warning(
                f"Parameter '{name}' has non-boolean value '{value}', using default={default}"
            )
            return bool(default)
        return bool(value)

    def _init_dvrk(self):
        arm_name = str(self._param("dvrk_arm", "PSM1"))
        ral_name = str(self._param("dvrk_ral", "teleop_load_probe_crtk"))
        do_enable = self._param_bool("dvrk_enable", False)
        do_home = self._param_bool("dvrk_home", False)

        import crtk
        import dvrk
        import PyKDL

        ral = crtk.ral(ral_name)
        self._dvrk_arm = dvrk.psm(ral, arm_name)

        if do_enable:
            self._dvrk_arm.enable()
        if do_home:
            self._dvrk_arm.home()

        try:
            self._dvrk_goal = self._dvrk_arm.setpoint_cp()
        except Exception:
            self._dvrk_goal = PyKDL.Frame.Identity()

        self.get_logger().warning(
            "dVRK mode enabled: this probe will call servo_cp at loop rate"
        )

    def _sub_cb(self, msg):
        now = time.monotonic()
        if now - self._last_sub_store_t < self.psm_sample_period_s:
            return

        self._last_sub_store_t = now
        self._sub_callbacks += 1

        with self.state_lock:
            self._dummy_x = msg.pose.position.x

    def _on_key_press(self, key):
        self._key_presses += 1
        now = time.monotonic()

        if self._keyboard is None:
            return

        arrows = (
            self._keyboard.Key.up,
            self._keyboard.Key.down,
            self._keyboard.Key.left,
            self._keyboard.Key.right,
        )
        if key in arrows:
            last_t = self._last_key_event_t.get(key, 0.0)
            if now - last_t < self.keyboard_repeat_min_period:
                return
            self._last_key_event_t[key] = now

        try:
            if key.char == "t":
                with self.state_lock:
                    self.active = not self.active
                self.get_logger().info(f"active={self.active}")
                return
        except AttributeError:
            pass

        with self.state_lock:
            self._keys_down.add(key)

    def _on_key_release(self, key):
        with self.state_lock:
            self._keys_down.discard(key)

    def _apply_keyboard_actions(self, dt):
        if self._keyboard is None:
            return

        with self.state_lock:
            dx = 0.0
            if self._keyboard.Key.left in self._keys_down:
                dx -= self.keyboard_x_speed * dt
            if self._keyboard.Key.right in self._keys_down:
                dx += self.keyboard_x_speed * dt
            self._dummy_x += dx

    def _simulate_busy_work(self):
        if self.busy_work_ms <= 0.0:
            return

        end_t = time.monotonic() + self.busy_work_ms / 1000.0
        while time.monotonic() < end_t:
            pass

    def _probe_dvrk(self):
        if not self.enable_dvrk_call or self._dvrk_arm is None or self._dvrk_goal is None:
            return
        self._dvrk_arm.servo_cp(self._dvrk_goal)

    def _control_tick(self):
        now = time.monotonic()

        if self._last_loop_t is None:
            self._last_loop_t = now
            return

        dt = now - self._last_loop_t
        self._last_loop_t = now

        self._loop_calls += 1
        self._loop_sum_dt += dt
        if dt < self._loop_min_dt:
            self._loop_min_dt = dt
        if dt > self._loop_max_dt:
            self._loop_max_dt = dt
        if dt > 1.5 * self.period_s:
            self._loop_overruns += 1

        with self.state_lock:
            active = self.active

        work_t0 = time.monotonic()
        if active:
            self._apply_keyboard_actions(dt)
            self._simulate_busy_work()
            self._probe_dvrk()
        work_dt = time.monotonic() - work_t0

        self._work_sum_dt += work_dt
        if work_dt > self._work_max_dt:
            self._work_max_dt = work_dt

    def _report_stats(self):
        if self._loop_calls == 0:
            return

        mean_dt = self._loop_sum_dt / self._loop_calls
        rate = 1.0 / mean_dt if mean_dt > 0.0 else 0.0
        min_dt = self._loop_min_dt if self._loop_min_dt != float("inf") else 0.0
        mean_work = self._work_sum_dt / self._loop_calls

        self.get_logger().info(
            f"Loop stats: rate={rate:.1f}Hz "
            f"mean_dt={mean_dt*1000.0:.2f}ms "
            f"min_dt={min_dt*1000.0:.2f}ms "
            f"max_dt={self._loop_max_dt*1000.0:.2f}ms "
            f"overruns={self._loop_overruns}/{self._loop_calls} "
            f"work_mean={mean_work*1000.0:.2f}ms "
            f"work_max={self._work_max_dt*1000.0:.2f}ms "
            f"sub_cb={self._sub_callbacks} key_press={self._key_presses}"
        )

        self._loop_calls = 0
        self._loop_overruns = 0
        self._loop_sum_dt = 0.0
        self._loop_max_dt = 0.0
        self._loop_min_dt = float("inf")
        self._work_sum_dt = 0.0
        self._work_max_dt = 0.0
        self._sub_callbacks = 0
        self._key_presses = 0

    def shutdown(self):
        if self._keyboard_listener is not None:
            self._keyboard_listener.stop()


def main():
    rclpy.init()
    node = TeleopLoadProbe()

    use_multithread = node._param_bool("use_multithread", True)
    executor_threads = int(node._param("executor_threads", 2))

    if use_multithread:
        executor = MultiThreadedExecutor(num_threads=max(1, executor_threads))
    else:
        executor = SingleThreadedExecutor()

    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()