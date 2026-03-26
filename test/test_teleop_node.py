#!/usr/bin/env python3
"""
Unit tests for gimbal_teleop_node.py.

The tests mock out:
  * rospy — so no ROS master is needed
  * DynamixelInterface — so no hardware is needed
"""

import math
import sys
import types
import unittest
from unittest.mock import MagicMock, patch, call

# ---------------------------------------------------------------------------
# Stub rospy so we can import and test without a ROS environment
# ---------------------------------------------------------------------------

def _make_mock_rospy():
    rospy = types.ModuleType("rospy")
    rospy._shutdown_hooks = []

    rospy.init_node = MagicMock()
    rospy.get_param = MagicMock(side_effect=lambda key, default=None: default)
    rospy.Subscriber = MagicMock()
    rospy.Publisher = MagicMock(return_value=MagicMock())
    rospy.loginfo = MagicMock()
    rospy.logwarn_throttle = MagicMock()
    rospy.logdebug = MagicMock()
    rospy.logerr = MagicMock()
    rospy.on_shutdown = MagicMock(
        side_effect=lambda fn: rospy._shutdown_hooks.append(fn)
    )
    rospy.spin = MagicMock()
    rospy.ROSInterruptException = Exception

    return rospy


def _make_mock_sensor_msgs():
    sensor_msgs = types.ModuleType("sensor_msgs")
    msg_mod = types.ModuleType("sensor_msgs.msg")

    class JointState:
        def __init__(self):
            self.position = []

    msg_mod.JointState = JointState
    sensor_msgs.msg = msg_mod
    return sensor_msgs, msg_mod


def _make_mock_std_msgs():
    std_msgs = types.ModuleType("std_msgs")
    msg_mod = types.ModuleType("std_msgs.msg")

    class Float64MultiArray:
        def __init__(self):
            self.data = []

    msg_mod.Float64MultiArray = Float64MultiArray
    std_msgs.msg = msg_mod
    return std_msgs, msg_mod


# Register stubs before importing the module under test
_rospy = _make_mock_rospy()
sys.modules["rospy"] = _rospy

_sensor_msgs, _sensor_msgs_msg = _make_mock_sensor_msgs()
sys.modules["sensor_msgs"] = _sensor_msgs
sys.modules["sensor_msgs.msg"] = _sensor_msgs_msg

_std_msgs, _std_msgs_msg = _make_mock_std_msgs()
sys.modules["std_msgs"] = _std_msgs
sys.modules["std_msgs.msg"] = _std_msgs_msg

import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


# ---------------------------------------------------------------------------
# Helper: build a GimbalTeleopNode with a patched DynamixelInterface
# ---------------------------------------------------------------------------

def _make_node(**param_overrides):
    """Return a GimbalTeleopNode with mocked Dynamixel and rospy."""
    defaults = {
        "~port_name":              "/dev/ttyFAKE",
        "~baudrate":               57600,
        "~pan_servo_id":           1,
        "~tilt_servo_id":          2,
        "~mtm_joint_topic":        "/dvrk/MTML/state_joint_current",
        "~pan_scale":              1.0,
        "~tilt_scale":             1.0,
        "~pan_offset_rad":         0.0,
        "~tilt_offset_rad":        0.0,
        "~min_angle_rad":         -math.pi / 2,
        "~max_angle_rad":          math.pi / 2,
        "~profile_velocity":       0,
        "~profile_acceleration":   0,
        "~deadband_rad":           0.01,
    }
    defaults.update(param_overrides)
    _rospy.get_param = MagicMock(side_effect=lambda key, default=None: defaults.get(key, default))

    mock_dxl_instance = MagicMock()
    mock_dxl_instance.connect = MagicMock()
    mock_dxl_instance.disconnect = MagicMock()
    mock_dxl_instance.set_goal_position = MagicMock()
    mock_dxl_instance.set_profile_velocity = MagicMock()
    mock_dxl_instance.set_profile_acceleration = MagicMock()

    # Reload first so module-level imports are fresh, then patch the class
    # inside the module's namespace before instantiating the node.
    import importlib
    import gimbal_teleop_node
    importlib.reload(gimbal_teleop_node)

    mock_dxl_class = MagicMock(return_value=mock_dxl_instance)
    with patch.object(gimbal_teleop_node, "DynamixelInterface", mock_dxl_class):
        node = gimbal_teleop_node.GimbalTeleopNode()

    return node, mock_dxl_instance


class TestGimbalTeleopNodeInit(unittest.TestCase):
    """Verify that the node initialises correctly."""

    def test_connects_on_init(self):
        node, mock_dxl = _make_node()
        mock_dxl.connect.assert_called_once()

    def test_registers_shutdown_hook(self):
        _rospy._shutdown_hooks.clear()
        node, _ = _make_node()
        self.assertTrue(len(_rospy._shutdown_hooks) > 0)

    def test_sets_motion_profiles(self):
        node, mock_dxl = _make_node(
            **{"~profile_velocity": 100, "~profile_acceleration": 50}
        )
        mock_dxl.set_profile_velocity.assert_any_call(1, 100)
        mock_dxl.set_profile_velocity.assert_any_call(2, 100)
        mock_dxl.set_profile_acceleration.assert_any_call(1, 50)
        mock_dxl.set_profile_acceleration.assert_any_call(2, 50)


class TestMTMCallback(unittest.TestCase):
    """Tests for the MTM joint-state callback logic."""

    def _joint_state(self, positions):
        msg = _sensor_msgs_msg.JointState()
        msg.position = positions
        return msg

    def test_first_message_sends_commands(self):
        node, mock_dxl = _make_node()
        positions = [0.0] * 7
        positions[5] = 0.3   # tilt
        positions[6] = 0.5   # pan
        node._mtm_cb(self._joint_state(positions))
        mock_dxl.set_goal_position.assert_any_call(1, 0.5)  # pan
        mock_dxl.set_goal_position.assert_any_call(2, 0.3)  # tilt

    def test_deadband_suppresses_small_changes(self):
        node, mock_dxl = _make_node(**{"~deadband_rad": 0.1})
        positions = [0.0] * 7
        positions[5] = 0.3
        positions[6] = 0.5
        node._mtm_cb(self._joint_state(positions))
        call_count_after_first = mock_dxl.set_goal_position.call_count

        # Change within deadband → no new calls
        positions[5] = 0.305  # change of 0.005 < 0.1 deadband
        positions[6] = 0.505
        node._mtm_cb(self._joint_state(positions))
        self.assertEqual(mock_dxl.set_goal_position.call_count, call_count_after_first)

    def test_deadband_allows_large_changes(self):
        node, mock_dxl = _make_node(**{"~deadband_rad": 0.1})
        positions = [0.0] * 7
        node._mtm_cb(self._joint_state(positions))
        before = mock_dxl.set_goal_position.call_count

        positions[6] = 0.5   # change of 0.5 > 0.1 deadband
        node._mtm_cb(self._joint_state(positions))
        self.assertGreater(mock_dxl.set_goal_position.call_count, before)

    def test_ignores_short_joint_state(self):
        node, mock_dxl = _make_node()
        short = self._joint_state([0.0] * 5)  # only 5 joints
        node._mtm_cb(short)
        mock_dxl.set_goal_position.assert_not_called()

    def test_scale_applied(self):
        node, mock_dxl = _make_node(
            **{"~pan_scale": 2.0, "~tilt_scale": 0.5}
        )
        positions = [0.0] * 7
        positions[5] = 1.0   # raw tilt
        positions[6] = 1.0   # raw pan
        node._mtm_cb(self._joint_state(positions))
        mock_dxl.set_goal_position.assert_any_call(1, 2.0)   # pan scaled
        mock_dxl.set_goal_position.assert_any_call(2, 0.5)   # tilt scaled

    def test_offset_applied(self):
        node, mock_dxl = _make_node(
            **{"~pan_offset_rad": 0.2, "~tilt_offset_rad": -0.1}
        )
        positions = [0.0] * 7
        # Both raw angles are 0.0; expect only offsets
        node._mtm_cb(self._joint_state(positions))
        mock_dxl.set_goal_position.assert_any_call(1, 0.2)
        mock_dxl.set_goal_position.assert_any_call(2, -0.1)


class TestShutdown(unittest.TestCase):
    """Verify teardown."""

    def test_shutdown_disconnects(self):
        node, mock_dxl = _make_node()
        node._shutdown()
        mock_dxl.disconnect.assert_called_once()


if __name__ == "__main__":
    unittest.main()
