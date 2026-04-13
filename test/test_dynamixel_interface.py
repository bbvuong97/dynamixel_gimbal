#!/usr/bin/env python3
"""
Unit tests for dynamixel_interface.py.

These tests do NOT require hardware or the real dynamixel_sdk package.
All SDK calls are replaced by mocks.
"""

import math
import sys
import types
import unittest
from unittest.mock import MagicMock, call, patch

# ---------------------------------------------------------------------------
# Build a minimal mock of dynamixel_sdk so the module can be imported without
# the real SDK being installed.
# ---------------------------------------------------------------------------

def _make_mock_sdk():
    sdk = types.ModuleType("dynamixel_sdk")

    class FakePortHandler:
        def __init__(self, port):
            self._port = port
            self.opened = False
            self.baudrate = None

        def openPort(self):
            self.opened = True
            return True

        def setBaudRate(self, baudrate):
            self.baudrate = baudrate
            return True

        def closePort(self):
            self.opened = False

    class FakePacketHandler:
        def __init__(self, protocol):
            self.protocol = protocol

        # All write/read methods return (value, COMM_SUCCESS=0, error=0) by default.
        def write1ByteTxRx(self, port, sid, addr, val):
            return 0, 0

        def write4ByteTxRx(self, port, sid, addr, val):
            return 0, 0

        def read4ByteTxRx(self, port, sid, addr):
            # Return CENTER_TICK (2048) so rad == 0.0
            return 2048, 0, 0

        def getTxRxResult(self, result):
            return f"result={result}"

        def getRxPacketError(self, error):
            return f"error={error}"

    sdk.PortHandler = FakePortHandler
    sdk.PacketHandler = FakePacketHandler
    return sdk


sys.modules.setdefault("dynamixel_sdk", _make_mock_sdk())

# Now we can safely import our module
import importlib
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from dynamixel_interface import (
    DynamixelInterface,
    rad_to_tick,
    tick_to_rad,
    CENTER_TICK,
    TICKS_PER_REV,
)


class TestConversionHelpers(unittest.TestCase):
    """Tests for rad_to_tick / tick_to_rad round-trip accuracy."""

    def test_center_zero(self):
        self.assertEqual(rad_to_tick(0.0), CENTER_TICK)

    def test_positive_half_pi(self):
        expected = CENTER_TICK + TICKS_PER_REV // 4
        self.assertEqual(rad_to_tick(math.pi / 2), expected)

    def test_negative_half_pi(self):
        expected = CENTER_TICK - TICKS_PER_REV // 4
        self.assertEqual(rad_to_tick(-math.pi / 2), expected)

    def test_round_trip(self):
        # Maximum quantization error is half a tick ≈ π/4096 ≈ 0.00077 rad.
        tolerance = math.pi / TICKS_PER_REV  # one full tick in rad
        for angle in [0.0, 0.5, -0.5, math.pi / 4, -math.pi / 3]:
            tick = rad_to_tick(angle)
            recovered = tick_to_rad(tick)
            self.assertAlmostEqual(
                angle, recovered, delta=tolerance,
                msg=f"Round-trip failed for angle={angle}",
            )


class TestDynamixelInterface(unittest.TestCase):
    """Integration-style tests using the mock SDK."""

    def _make_iface(self, servo_ids=None, **kwargs):
        if servo_ids is None:
            servo_ids = [1, 2]
        return DynamixelInterface(
            port_name="/dev/ttyFAKE",
            baudrate=57600,
            servo_ids=servo_ids,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def test_connect_opens_port(self):
        iface = self._make_iface()
        iface.connect()
        self.assertTrue(iface.is_connected)
        iface.disconnect()

    def test_disconnect_closes_port(self):
        iface = self._make_iface()
        iface.connect()
        iface.disconnect()
        self.assertFalse(iface.is_connected)

    def test_context_manager(self):
        iface = self._make_iface()
        with iface:
            self.assertTrue(iface.is_connected)
        self.assertFalse(iface.is_connected)

    def test_disconnect_without_connect_is_safe(self):
        iface = self._make_iface()
        iface.disconnect()  # should not raise

    # ------------------------------------------------------------------
    # set_goal_position
    # ------------------------------------------------------------------

    def test_set_goal_position_known_servo(self):
        iface = self._make_iface()
        iface.connect()
        # Should not raise for a known servo
        iface.set_goal_position(1, 0.0)
        iface.disconnect()

    def test_set_goal_position_unknown_servo(self):
        iface = self._make_iface(servo_ids=[1, 2])
        iface.connect()
        with self.assertRaises(RuntimeError):
            iface.set_goal_position(99, 0.0)
        iface.disconnect()

    def test_set_goal_position_clamps_to_max(self):
        """Tick sent must be <= max_tick even if angle exceeds max_angle."""
        iface = self._make_iface(
            min_angle_rad=-math.pi / 2,
            max_angle_rad=math.pi / 2,
        )
        iface.connect()
        # Capture the tick value written
        written = []
        orig = iface._packet_handler.write4ByteTxRx

        def capture(port, sid, addr, val):
            written.append(val)
            return 0, 0

        iface._packet_handler.write4ByteTxRx = capture
        iface.set_goal_position(1, math.pi)  # beyond max
        self.assertTrue(all(t <= rad_to_tick(math.pi / 2) for t in written))
        iface.disconnect()

    def test_set_goal_position_clamps_to_min(self):
        iface = self._make_iface(
            min_angle_rad=-math.pi / 2,
            max_angle_rad=math.pi / 2,
        )
        iface.connect()
        written = []

        def capture(port, sid, addr, val):
            written.append(val)
            return 0, 0

        iface._packet_handler.write4ByteTxRx = capture
        iface.set_goal_position(1, -math.pi)  # below min
        self.assertTrue(all(t >= rad_to_tick(-math.pi / 2) for t in written))
        iface.disconnect()

    # ------------------------------------------------------------------
    # get_present_position
    # ------------------------------------------------------------------

    def test_get_present_position_returns_float(self):
        iface = self._make_iface()
        iface.connect()
        pos = iface.get_present_position(1)
        self.assertIsInstance(pos, float)
        self.assertAlmostEqual(pos, 0.0, places=5)
        iface.disconnect()

    # ------------------------------------------------------------------
    # Communication error handling
    # ------------------------------------------------------------------

    def test_comm_error_raises_ioerror(self):
        iface = self._make_iface()
        iface.connect()

        def bad_write(port, sid, addr, val):
            return -1, 0  # COMM_SUCCESS = 0; -1 means error

        iface._packet_handler.write4ByteTxRx = bad_write
        with self.assertRaises(IOError):
            iface.set_goal_position(1, 0.0)
        iface.disconnect()

    def test_packet_error_raises_ioerror(self):
        iface = self._make_iface()
        iface.connect()

        def bad_write(port, sid, addr, val):
            return 0, 1  # non-zero error byte

        iface._packet_handler.write4ByteTxRx = bad_write
        with self.assertRaises(IOError):
            iface.set_goal_position(1, 0.0)
        iface.disconnect()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    def test_servo_ids_property(self):
        iface = self._make_iface(servo_ids=[3, 5, 7])
        self.assertEqual(iface.servo_ids, [3, 5, 7])


if __name__ == "__main__":
    unittest.main()
