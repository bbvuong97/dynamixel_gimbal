#!/usr/bin/env python3

import socket
import struct
import threading
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped


class ViveTrackerUdpReceiver(Node):
    def __init__(self):
        super().__init__("vive_tracker_udp_receiver")

        self.declare_parameter("udp_ip", "0.0.0.0")
        self.declare_parameter("udp_port", 5005)
        self.declare_parameter("topic_name", "/vive_tracker/pose")
        self.declare_parameter("frame_id", "steamvr_world")
        self.declare_parameter("best_effort", True)

        udp_ip = self.get_parameter("udp_ip").get_parameter_value().string_value
        udp_port = self.get_parameter("udp_port").get_parameter_value().integer_value
        topic_name = self.get_parameter("topic_name").get_parameter_value().string_value
        self.frame_id = self.get_parameter("frame_id").get_parameter_value().string_value
        best_effort = self.get_parameter("best_effort").get_parameter_value().bool_value

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT if best_effort else ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.pub = self.create_publisher(PoseStamped, topic_name, qos)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((udp_ip, udp_port))
        self.sock.settimeout(0.2)
        self._running = True
        self._packet_struct = struct.Struct("fffffff")

        self.get_logger().info(
            f"Listening for Vive Tracker UDP packets on {udp_ip}:{udp_port}"
        )

        self._rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
        self._rx_thread.start()

    def _rx_loop(self):
        while self._running and rclpy.ok():
            try:
                data, _ = self.sock.recvfrom(1024)
            except socket.timeout:
                continue
            except OSError as exc:
                if self._running:
                    self.get_logger().error(f"Socket receive error: {exc}")
                break

            if len(data) != self._packet_struct.size:
                continue

            x, y, z, qw, qx, qy, qz = self._packet_struct.unpack(data)

            msg = PoseStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.frame_id

            msg.pose.position.x = x
            msg.pose.position.y = y
            msg.pose.position.z = z

            msg.pose.orientation.w = qw
            msg.pose.orientation.x = qx
            msg.pose.orientation.y = qy
            msg.pose.orientation.z = qz

            self.pub.publish(msg)

    def destroy_node(self):
        try:
            self._running = False
            self.sock.close()
            if hasattr(self, "_rx_thread") and self._rx_thread.is_alive():
                self._rx_thread.join(timeout=1.0)
        finally:
            super().destroy_node()


def main():
    rclpy.init()
    node = ViveTrackerUdpReceiver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
