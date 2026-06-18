#!/usr/bin/env python3
"""
Dynamixel servo interface for pan/tilt gimbal control.

Provides a thin abstraction over the Dynamixel SDK so the rest of the
package only deals with servo IDs and goal positions in radians.
Supports Protocol 2.0 devices (XM, XL, XD series, …).
"""

import math
import time

# ---------------------------------------------------------------------------
# Dynamixel control-table addresses (Protocol 2.0 / X-series)
# ---------------------------------------------------------------------------
ADDR_TORQUE_ENABLE       = 64
ADDR_OPERATING_MODE      = 11
ADDR_GOAL_POSITION       = 116
ADDR_PRESENT_POSITION    = 132
ADDR_PROFILE_VELOCITY    = 112
ADDR_PROFILE_ACCELERATION = 108
ADDR_MIN_POSITION_LIMIT  = 52
ADDR_MAX_POSITION_LIMIT  = 48

# Byte lengths for the above registers
LEN_GOAL_POSITION        = 4
LEN_PRESENT_POSITION     = 4

# Miscellaneous constants
TORQUE_ENABLE            = 1
TORQUE_DISABLE           = 0
OPERATING_MODE_POSITION  = 3   # extended-position mode not needed for ±π
COMM_SUCCESS             = 0
PROTOCOL_VERSION         = 2.0

# Encoder resolution for X-series (4096 ticks per revolution)
TICKS_PER_REV            = 4096
CENTER_TICK              = 2048  # 0 rad


def rad_to_tick(angle_rad: float) -> int:
    """Convert an angle in radians to a Dynamixel encoder tick value."""
    ticks = int(round(CENTER_TICK + angle_rad * TICKS_PER_REV / (2.0 * math.pi)))
    return ticks


def tick_to_rad(tick: int) -> float:
    """Convert a Dynamixel encoder tick to an angle in radians."""
    return (tick - CENTER_TICK) * 2.0 * math.pi / TICKS_PER_REV


class DynamixelInterface:
    """
    High-level interface to one or more Dynamixel servos.

    Parameters
    ----------
    port_name : str
        Serial port connected to the U2D2 (e.g. ``"/dev/ttyUSB0"``).
    baudrate : int
        Baudrate configured on the servos (default 57600).
    servo_ids : list[int]
        List of Dynamixel IDs that make up the gimbal.
    min_angle_rad : float
        Minimum allowed angle for any servo (radians).
    max_angle_rad : float
        Maximum allowed angle for any servo (radians).

    The interface tries to import ``dynamixel_sdk``.  If the SDK is not
    installed a ``RuntimeError`` is raised so that the rest of the code
    can fail fast with a meaningful message.
    """

    def __init__(
        self,
        port_name: str,
        baudrate: int,
        servo_ids: list,
        min_angle_rad: float = -math.pi / 2,
        max_angle_rad: float = math.pi / 2,
    ):
        try:
            import dynamixel_sdk as dxl  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "dynamixel_sdk is not installed.  "
                "Install it with:  pip install dynamixel-sdk"
            ) from exc

        self._dxl = dxl
        self._port_name = port_name
        self._baudrate = baudrate
        self._servo_ids = list(servo_ids)
        self._min_tick = rad_to_tick(min_angle_rad)
        self._max_tick = rad_to_tick(max_angle_rad)

        self._port_handler = dxl.PortHandler(port_name)
        self._packet_handler = dxl.PacketHandler(PROTOCOL_VERSION)
        self._connected = False

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Open the serial port and enable torque on all servos."""
        if not self._port_handler.openPort():
            raise IOError(f"Failed to open port {self._port_name}")
        if not self._port_handler.setBaudRate(self._baudrate):
            raise IOError(f"Failed to set baudrate {self._baudrate}")
        self._connected = True
        for sid in self._servo_ids:
            self._set_torque(sid, TORQUE_ENABLE)

    def disconnect(self) -> None:
        """Disable torque on all servos and close the serial port."""
        if not self._connected:
            return
        for sid in self._servo_ids:
            self._set_torque(sid, TORQUE_DISABLE)
        self._port_handler.closePort()
        self._connected = False

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.disconnect()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_goal_position(self, servo_id: int, angle_rad: float) -> None:
        """
        Command *servo_id* to move to *angle_rad* (clamped to configured limits).

        Parameters
        ----------
        servo_id : int
            Dynamixel servo ID.
        angle_rad : float
            Desired angle in radians.

        Raises
        ------
        IOError
            If the SDK reports a communication error.
        RuntimeError
            If *servo_id* is not in the list of known servos.
        """
        if servo_id not in self._servo_ids:
            raise RuntimeError(f"Servo ID {servo_id} is not managed by this interface")
        tick = int(
            max(self._min_tick, min(self._max_tick, rad_to_tick(angle_rad)))
        )
        dxl_comm_result, dxl_error = self._packet_handler.write4ByteTxRx(
            self._port_handler, servo_id, ADDR_GOAL_POSITION, tick
        )
        self._check_result(servo_id, dxl_comm_result, dxl_error, "write goal position")

    def get_present_position(self, servo_id: int) -> float:
        """
        Read the current position of *servo_id* in radians.

        Returns
        -------
        float
            Present position in radians.
        """
        tick, dxl_comm_result, dxl_error = self._packet_handler.read4ByteTxRx(
            self._port_handler, servo_id, ADDR_PRESENT_POSITION
        )
        self._check_result(servo_id, dxl_comm_result, dxl_error, "read present position")
        return tick_to_rad(tick)

    def set_profile_velocity(self, servo_id: int, velocity: int) -> None:
        """
        Set the profile velocity for *servo_id* (0 = maximum speed).

        Parameters
        ----------
        velocity : int
            Velocity in Dynamixel raw units (0.229 RPM per unit for X-series).
        """
        dxl_comm_result, dxl_error = self._packet_handler.write4ByteTxRx(
            self._port_handler, servo_id, ADDR_PROFILE_VELOCITY, velocity
        )
        self._check_result(servo_id, dxl_comm_result, dxl_error, "set profile velocity")

    def set_profile_acceleration(self, servo_id: int, acceleration: int) -> None:
        """
        Set the profile acceleration for *servo_id*.

        Parameters
        ----------
        acceleration : int
            Acceleration in Dynamixel raw units (214.577 rev/min² per unit).
        """
        dxl_comm_result, dxl_error = self._packet_handler.write4ByteTxRx(
            self._port_handler, servo_id, ADDR_PROFILE_ACCELERATION, acceleration
        )
        self._check_result(
            servo_id, dxl_comm_result, dxl_error, "set profile acceleration"
        )

    @property
    def servo_ids(self) -> list:
        """Return the list of managed servo IDs."""
        return list(self._servo_ids)

    @property
    def is_connected(self) -> bool:
        """Return ``True`` if the port is open."""
        return self._connected

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _set_torque(self, servo_id: int, enable: int) -> None:
        dxl_comm_result, dxl_error = self._packet_handler.write1ByteTxRx(
            self._port_handler, servo_id, ADDR_TORQUE_ENABLE, enable
        )
        action = "enable" if enable else "disable"
        self._check_result(servo_id, dxl_comm_result, dxl_error, f"{action} torque")

    def _check_result(
        self, servo_id: int, comm_result: int, error: int, context: str
    ) -> None:
        if comm_result != COMM_SUCCESS:
            msg = self._packet_handler.getTxRxResult(comm_result)
            raise IOError(
                f"[Servo {servo_id}] Communication error during '{context}': {msg}"
            )
        if error != 0:
            msg = self._packet_handler.getRxPacketError(error)
            raise IOError(
                f"[Servo {servo_id}] Packet error during '{context}': {msg}"
            )
