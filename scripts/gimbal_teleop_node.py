#!/usr/bin/env python3
"""
Gimbal teleoperation node.

Subscribes to dVRK MTM joint-state messages and maps the wrist
orientation (last three joints: roll, pitch, yaw) to pan / tilt
commands sent to Dynamixel servos.

ROS Parameters
--------------
~port_name          : str  (default '/dev/ttyUSB0')
    Serial port for the U2D2 adapter.
~baudrate           : int  (default 57600)
    Dynamixel bus baudrate.
~pan_servo_id       : int  (default 1)
    Dynamixel ID for the pan (yaw) servo.
~tilt_servo_id      : int  (default 2)
    Dynamixel ID for the tilt (pitch) servo.
~mtm_joint_topic    : str  (default '/dvrk/MTML/state_joint_current')
    Topic publishing the MTM joint states (sensor_msgs/JointState).
~pan_scale          : float (default 1.0)
    Scale factor applied to the MTM yaw angle before commanding pan.
~tilt_scale         : float (default 1.0)
    Scale factor applied to the MTM pitch angle before commanding tilt.
~pan_offset_rad     : float (default 0.0)
    Constant angular offset added to the pan command (rad).
~tilt_offset_rad    : float (default 0.0)
    Constant angular offset added to the tilt command (rad).
~min_angle_rad      : float (default -1.5708)
    Minimum servo angle limit in radians.
~max_angle_rad      : float (default  1.5708)
    Maximum servo angle limit in radians.
~profile_velocity   : int   (default 0)
    Dynamixel profile velocity (0 = max).
~profile_acceleration : int (default 0)
    Dynamixel profile acceleration (0 = max).
~deadband_rad       : float (default 0.01)
    Minimum change in command angle (rad) before a new goal is sent.
"""

import math
import sys

import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

# Import from sibling script.  When installed via catkin the scripts
# directory is on the Python path.
try:
    from dynamixel_interface import DynamixelInterface
except ImportError:
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from dynamixel_interface import DynamixelInterface


class GimbalTeleopNode:
    """ROS node that teleoperates a Dynamixel pan/tilt gimbal from dVRK MTM."""

    def __init__(self):
        rospy.init_node("gimbal_teleop_node")

        # ----------------------------------------------------------------
        # Read ROS parameters
        # ----------------------------------------------------------------
        self._port_name   = rospy.get_param("~port_name",    "/dev/ttyUSB0")
        self._baudrate    = rospy.get_param("~baudrate",      57600)
        self._pan_id      = rospy.get_param("~pan_servo_id",  1)
        self._tilt_id     = rospy.get_param("~tilt_servo_id", 2)
        self._mtm_topic   = rospy.get_param(
            "~mtm_joint_topic", "/dvrk/MTML/state_joint_current"
        )
        self._pan_scale   = rospy.get_param("~pan_scale",        1.0)
        self._tilt_scale  = rospy.get_param("~tilt_scale",       1.0)
        self._pan_offset  = rospy.get_param("~pan_offset_rad",   0.0)
        self._tilt_offset = rospy.get_param("~tilt_offset_rad",  0.0)
        min_angle         = rospy.get_param("~min_angle_rad",  -math.pi / 2)
        max_angle         = rospy.get_param("~max_angle_rad",   math.pi / 2)
        self._prof_vel    = rospy.get_param("~profile_velocity",     0)
        self._prof_acc    = rospy.get_param("~profile_acceleration", 0)
        self._deadband    = rospy.get_param("~deadband_rad",    0.01)

        # ----------------------------------------------------------------
        # State
        # ----------------------------------------------------------------
        self._last_pan  = None
        self._last_tilt = None

        # ----------------------------------------------------------------
        # Dynamixel interface
        # ----------------------------------------------------------------
        self._dxl = DynamixelInterface(
            port_name=self._port_name,
            baudrate=self._baudrate,
            servo_ids=[self._pan_id, self._tilt_id],
            min_angle_rad=min_angle,
            max_angle_rad=max_angle,
        )
        self._dxl.connect()
        rospy.loginfo(
            "GimbalTeleopNode: connected to Dynamixel bus on %s", self._port_name
        )

        # Configure motion profiles (velocity / acceleration).
        for sid in [self._pan_id, self._tilt_id]:
            self._dxl.set_profile_velocity(sid, self._prof_vel)
            self._dxl.set_profile_acceleration(sid, self._prof_acc)

        # ----------------------------------------------------------------
        # Publishers
        # ----------------------------------------------------------------
        self._pub_angles = rospy.Publisher(
            "~gimbal_angles", Float64MultiArray, queue_size=1
        )

        # ----------------------------------------------------------------
        # Subscribers — connect last so the interface is ready first
        # ----------------------------------------------------------------
        rospy.Subscriber(
            self._mtm_topic, JointState, self._mtm_cb, queue_size=1
        )

        rospy.loginfo(
            "GimbalTeleopNode: subscribed to '%s'", self._mtm_topic
        )
        rospy.on_shutdown(self._shutdown)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _mtm_cb(self, msg: JointState) -> None:
        """
        Process incoming MTM joint-state message.

        The dVRK MTM has 7 joints.  The convention used here is:
          - joint[5]  (index 5) → pitch (tilt)
          - joint[6]  (index 6) → yaw   (pan)

        This can be overridden via parameter if a different MTM is used.
        """
        if len(msg.position) < 7:
            rospy.logwarn_throttle(
                5.0,
                "GimbalTeleopNode: expected at least 7 joint positions, "
                "got %d — ignoring message",
                len(msg.position),
            )
            return

        # Map MTM wrist joints to pan / tilt
        raw_tilt = msg.position[5]   # pitch
        raw_pan  = msg.position[6]   # yaw

        pan_cmd  = self._pan_scale  * raw_pan  + self._pan_offset
        tilt_cmd = self._tilt_scale * raw_tilt + self._tilt_offset

        self._send_if_changed(pan_cmd, tilt_cmd)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _send_if_changed(self, pan_rad: float, tilt_rad: float) -> None:
        """Send servo commands only when the angle has changed by more than the deadband."""
        pan_changed  = (
            self._last_pan  is None
            or abs(pan_rad  - self._last_pan)  > self._deadband
        )
        tilt_changed = (
            self._last_tilt is None
            or abs(tilt_rad - self._last_tilt) > self._deadband
        )

        if pan_changed:
            self._dxl.set_goal_position(self._pan_id, pan_rad)
            self._last_pan = pan_rad
            rospy.logdebug("Pan  → %.4f rad", pan_rad)

        if tilt_changed:
            self._dxl.set_goal_position(self._tilt_id, tilt_rad)
            self._last_tilt = tilt_rad
            rospy.logdebug("Tilt → %.4f rad", tilt_rad)

        if pan_changed or tilt_changed:
            msg = Float64MultiArray()
            msg.data = [
                self._last_pan  if self._last_pan  is not None else 0.0,
                self._last_tilt if self._last_tilt is not None else 0.0,
            ]
            self._pub_angles.publish(msg)

    def _shutdown(self) -> None:
        rospy.loginfo("GimbalTeleopNode: shutting down, disabling torque.")
        self._dxl.disconnect()

    # ------------------------------------------------------------------
    # Spin
    # ------------------------------------------------------------------

    def spin(self) -> None:
        """Block until ROS shuts down."""
        rospy.spin()


def main():
    try:
        node = GimbalTeleopNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
    except (IOError, RuntimeError) as exc:
        rospy.logerr("GimbalTeleopNode fatal error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
