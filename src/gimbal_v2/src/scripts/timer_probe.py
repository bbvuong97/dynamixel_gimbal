#!/usr/bin/env python3
import time

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
from rclpy.node import Node


class TimerProbe(Node):
    def __init__(self):
        super().__init__("timer_probe")

        self.period_s = float(self._param("period", 0.005))
        self.report_period_s = float(self._param("report_period", 1.0))
        self.busy_work_ms = float(self._param("busy_work_ms", 0.0))

        self.use_reentrant = bool(self._param("use_reentrant", False))
        self.use_multithread = bool(self._param("use_multithread", True))
        self.executor_threads = int(self._param("executor_threads", 2))

        if self.use_reentrant:
            self.timer_group = ReentrantCallbackGroup()
            self.report_group = ReentrantCallbackGroup()
        else:
            self.timer_group = MutuallyExclusiveCallbackGroup()
            self.report_group = MutuallyExclusiveCallbackGroup()

        self.timer = self.create_timer(self.period_s, self._tick, callback_group=self.timer_group)
        self.report_timer = self.create_timer(
            self.report_period_s, self._report_stats, callback_group=self.report_group
        )

        self._last_t = None
        self._calls = 0
        self._overruns = 0
        self._sum_dt = 0.0
        self._max_dt = 0.0
        self._min_dt = float("inf")

        self.get_logger().info(
            "Timer probe started: "
            f"period={self.period_s:.6f}s "
            f"report_period={self.report_period_s:.2f}s "
            f"busy_work_ms={self.busy_work_ms:.2f} "
            f"executor={'multi' if self.use_multithread else 'single'} "
            f"threads={self.executor_threads} "
            f"group={'reentrant' if self.use_reentrant else 'mutually_exclusive'}"
        )

    def _param(self, name, default):
        if not self.has_parameter(name):
            self.declare_parameter(name, default)
        return self.get_parameter(name).value

    def _tick(self):
        now = time.monotonic()
        if self._last_t is not None:
            dt = now - self._last_t
            self._calls += 1
            self._sum_dt += dt
            if dt < self._min_dt:
                self._min_dt = dt
            if dt > self._max_dt:
                self._max_dt = dt
            if dt > 1.5 * self.period_s:
                self._overruns += 1
        self._last_t = now

        if self.busy_work_ms > 0.0:
            end_t = now + self.busy_work_ms / 1000.0
            while time.monotonic() < end_t:
                pass

    def _report_stats(self):
        if self._calls == 0:
            return

        mean_dt = self._sum_dt / self._calls
        rate = 1.0 / mean_dt if mean_dt > 0.0 else 0.0
        min_dt = self._min_dt if self._min_dt != float("inf") else 0.0

        self.get_logger().info(
            f"Timer stats: rate={rate:.1f}Hz "
            f"mean_dt={mean_dt * 1000.0:.2f}ms "
            f"min_dt={min_dt * 1000.0:.2f}ms "
            f"max_dt={self._max_dt * 1000.0:.2f}ms "
            f"overruns={self._overruns}/{self._calls}"
        )

        self._calls = 0
        self._overruns = 0
        self._sum_dt = 0.0
        self._max_dt = 0.0
        self._min_dt = float("inf")


def main():
    rclpy.init()
    node = TimerProbe()

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


if __name__ == "__main__":
    main()
