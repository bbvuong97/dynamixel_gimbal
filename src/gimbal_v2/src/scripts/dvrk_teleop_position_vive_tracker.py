import rospy
import sys
import time
import crtk
import dvrk
import PyKDL
import numpy as np
from geometry_msgs.msg import PoseStamped
# from trakstar_ros.msg import TrackedPose
from pynput import keyboard
import argparse

import tf2_ros
import tf2_geometry_msgs  # not strictly needed but harmless to have


class ViveTeleop:
    def __init__(self, ral, config_dict):
        self.ral = ral
        self.arm_name = config_dict['arm_name']
        self.scale = config_dict['scale']
        self.expected_interval = config_dict['expected_interval']
        self.control_type = config_dict['control_type']

        self.arm = dvrk.psm(ral, self.arm_name)
        self.running = False  # Main loop control
        self.teleop_active = False  # Toggle teleop on/off
        self.reference_position = None
        self.current_position = None
        self.reference_orientation = None
        self.current_orientation = None

        # --- TF setup ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Give TF a moment to populate (or loop until success if you prefer)
        rospy.sleep(0.5)

        # 1) HTC Vive Tracker steamvr_world -> cart (your known, fixed mapping)
        #    steamvr_world x -> cart y
        #    steamvr_world y -> cart x
        #    steamvr_world z -> cart -z
        R_cart_from_trak = PyKDL.Rotation(
            1.0, 0.0,  0.0,   # first column (cart X)  <= steamvr_world X
            0.0, 0.0,  -1.0,   # second column (cart Y) <= steamvr_world -Z
            0.0, 1.0, 0.0    # third column (cart Z) <= -steamvr_world Y
        )

        # 2) cart -> ECM (camera) from TF
        self.R_ecm_from_cart = self._tf_rotation("Cart", "ECM")

        # 3) trakSTAR base -> ECM (camera)
        self.R_cam_from_trak = self.R_ecm_from_cart * R_cart_from_trak


        # for position control
        self.filtered_disp = np.zeros(3)
        self.pos_deadzone = 0.005   # 1 mm deadzone (meters)
        self.pos_tau = 0.08         # LPF time constant (s) ~80 ms
        self.max_speed = 0.05       # m/s clamp on commanded tip speed

        # # for orientation control
        # self.filtered_rpy = np.zeros(3)
        # self.orientation_alpha = 0.1  # smoothing factor
        # self.orientation_scale = 0.5  # full motion
        # self.orientation_deadzone = np.radians(1.0)


        self.initialized = False

        self.jaw_closed = False  # <--- add this line

        self.jaw_angle = self.arm.jaw.setpoint_js()[0][0]

        # Subscribe to trakSTAR sensor2
        rospy.Subscriber("/vive_tracker/pose", PoseStamped, self.trakstar_callback)

        print(f"Teleop initialized for {self.arm_name}, initial scale: {self.scale}")

    def trakstar_callback(self, msg):
        self.current_position = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])

        # Extract quaternion
        qx = msg.pose.orientation.x
        qy = msg.pose.orientation.y
        qz = msg.pose.orientation.z
        qw = msg.pose.orientation.w

        # Convert quaternion → PyKDL rotation → RPY
        rot = PyKDL.Rotation.Quaternion(qx, qy, qz, qw)
        roll, pitch, yaw = rot.GetRPY()  # radians

        # Store as [yaw, pitch, roll] in radians
        self.current_orientation = np.array([yaw, pitch, roll])

    def _tf_rotation(self, from_frame, to_frame, timeout=1.0):
        """
        Return PyKDL.Rotation such that: v_in_to = R * v_in_from
        i.e., rotation that maps a vector expressed in 'from_frame'
        into 'to_frame'.
        """
        try:
            trans = self.tf_buffer.lookup_transform(
                to_frame, from_frame, rospy.Time(0), rospy.Duration(timeout)
            )
            q = trans.transform.rotation
            return PyKDL.Rotation.Quaternion(q.x, q.y, q.z, q.w)
        except Exception as e:
            rospy.logwarn(f"[trakstar_teleop] TF lookup {from_frame}->{to_frame} failed: {e}")
            return PyKDL.Rotation.Identity()


    def _lpf_alpha(self, dt, tau):
        # first-order low-pass: y += alpha*(x-y); alpha = dt/(tau+dt)
        return dt / (tau + dt) if tau > 0 else 1.0

    def rpy_to_rotation(self, yaw, pitch, roll):  # input in radians
        return PyKDL.Rotation.RPY(roll, pitch, yaw)
    
    def process_orientation_delta(self, R_ref, R_curr):
        R_delta = R_ref.Inverse() * R_curr
        roll, pitch, yaw = R_delta.GetRPY()
        raw_rpy = np.array([roll, pitch, yaw])

        # Dead zone
        for i in range(3):
            if abs(raw_rpy[i]) < self.orientation_deadzone:
                raw_rpy[i] = 0.0

        # Scale
        scaled_rpy = raw_rpy * self.orientation_scale

        # Low-pass filter
        self.filtered_rpy = (
            self.orientation_alpha * scaled_rpy +
            (1 - self.orientation_alpha) * self.filtered_rpy
        )

        return PyKDL.Rotation.RPY(*self.filtered_rpy)


    def home(self):
        self.arm.check_connections()
        if not self.arm.enable(10):
            sys.exit('Failed to enable within 10 seconds')
        if not self.arm.home(10):
            sys.exit('Failed to home within 10 seconds')
        print("PSM homed successfully")

    def set_reference(self):
        if self.current_position is not None and self.current_orientation is not None:
            self.reference_position    = self.current_position.copy()
            self.reference_orientation = self.current_orientation.copy()

            cp = self.arm.setpoint_cp()
            self.initial_psm_pose = PyKDL.Frame(cp.M, cp.p)

            # --- reset filters so first frame = zero motion ---
            self.filtered_disp = np.zeros(3)
            self.filtered_rpy  = np.zeros(3)

            # --- precompute initial sensor-in-ECM for conjugation (stable first frame) ---
            R_ref = self.rpy_to_rotation(*self.reference_orientation)
            self.R_CAM_S0 = self.R_cam_from_trak * R_ref

            # --- enable a short ramp (soft-start) ---
            self._ramp_t0   = time.time()
            self._ramp_T    = 0.20   # 200 ms ramp
            self._ramping   = True

            self.initialized = True
            print("Reference position set:", self.reference_position)
        else:
            print("Waiting for trakSTAR data...")


    def compute_displacement(self):
        if not self.initialized or self.current_position is None:
            return np.zeros(3)

        # trakSTAR positions are in *centimeters* per your code
        disp_trak = (self.current_position - self.reference_position)   # cm -> m

        # deadzone
        for i in range(3):
            if abs(disp_trak[i]) < self.pos_deadzone:
                disp_trak[i] = 0.0

        # low-pass in trak frame
        dt = self.expected_interval if self.expected_interval > 0 else 0.01
        alpha = self._lpf_alpha(dt, self.pos_tau)
        self.filtered_disp = self.filtered_disp + alpha * (disp_trak - self.filtered_disp)

        # scale (still in trak frame)
        disp_scaled_trak = self.filtered_disp * self.scale

        # rotate to ECM/camera frame using the SAME mapping as orientation
        v = PyKDL.Vector(*disp_scaled_trak)
        v_cam = self.R_cam_from_trak * v  # maps trak -> ECM

        return np.array([v_cam[0], v_cam[1], v_cam[2]])
    
    def _ramp_scale(self):
        if getattr(self, "_ramping", False):
            s = (time.time() - self._ramp_t0) / self._ramp_T
            if s >= 1.0:
                self._ramping = False
                return 1.0
            return max(0.0, min(1.0, s))
        return 1.0

    def move_cartesian(self):
        if not self.initialized or not self.teleop_active:
            return

        # ---- Position (unchanged) ----
        s = self._ramp_scale()
        displacement = self.compute_displacement() * s
        goal_pos = self.initial_psm_pose.p + PyKDL.Vector(*displacement)


        # ---- Orientation ----
        if self.reference_orientation is not None and self.current_orientation is not None:
            R_ref  = self.rpy_to_rotation(*self.reference_orientation)
            R_curr = self.rpy_to_rotation(*self.current_orientation)

            R_delta_sensor = self.process_orientation_delta(R_ref, R_curr)  # already filtered in RPY
            # scale the small-angle delta by s
            r,p,y = R_delta_sensor.GetRPY()
            R_delta_sensor_scaled = PyKDL.Rotation.RPY(r*s, p*s, y*s)

            # use the pre-latched initial sensor orientation in ECM
            R_delta_cam = self.R_CAM_S0 * R_delta_sensor_scaled * self.R_CAM_S0.Inverse()
            goal_rot = self.initial_psm_pose.M * R_delta_cam
        else:
            goal_rot = self.initial_psm_pose.M


        # ---- Compose final goal ----
        goal = PyKDL.Frame()
        goal.p = goal_pos
        goal.M = goal_rot

        # ---- Debug ----
        print(f"Reference trakSTAR position: {self.reference_position}")
        print(f"Current trakSTAR position: {self.current_position}")
        print(f"Reference trakSTAR orientation: {self.reference_orientation}")
        print(f"Current trakSTAR orientation: {self.current_orientation}")

        print(f"Displacement: {displacement[0]:.3f}, {displacement[1]:.3f}, {displacement[2]:.3f}")
        print(f"Goal position: {goal.p[0]:.3f}, {goal.p[1]:.3f}, {goal.p[2]:.3f}")
        
        rpy = goal.M.GetRPY()
        print(f"Goal orientation (rad): roll={rpy[0]:.2f}, pitch={rpy[1]:.2f}, yaw={rpy[2]:.2f}")

        # ---- Send command ----
        if self.control_type == 's':
            self.arm.servo_cp(goal)
        elif self.control_type == 'm':
            self.arm.move_cp(goal).wait()


    def move_jaw(self):
        if not hasattr(self, 'jaw_closed'):
            self.jaw_closed = False

        target_angle = 0.0 if self.jaw_closed else 80.0 * np.pi / 180  # radians
        current_angle = self.arm.jaw.setpoint_js()[0][0]
        step = np.deg2rad(2)  # 2 degrees per step

        direction = 1 if target_angle > current_angle else -1
        while abs(target_angle - current_angle) > step:
            current_angle += direction * step
            self.arm.jaw.servo_jp(np.array([current_angle]))
            time.sleep(0.02)
        self.arm.jaw.servo_jp(np.array([target_angle]))

        if self.jaw_closed:
            print(f"[{self.arm_name}] Jaw OPENED")
            self.jaw_closed = False
        else:
            print(f"[{self.arm_name}] Jaw CLOSED")
            self.jaw_closed = True


    def on_press(self, key):
        try:
            if key == keyboard.Key.space:
                print("SPACE pressed")
                if not self.teleop_active:
                    print("Enabling teleoperation... HERE")
                    # delay for 10 ms
                    time.sleep(0.01)  # 10 ms
                    # Start teleop
                    self.set_reference()
                    self.teleop_active = True
                    print("Teleoperation ENABLED")
                else:
                    # Stop teleop
                    self.teleop_active = False
                    print("Teleoperation DISABLED")
            elif key == keyboard.Key.up:
                self.scale += 0.1
                print(f"Scale increased to {self.scale:.2f}")
            elif key == keyboard.Key.down:
                self.scale = max(0.1, self.scale - 0.1)
                print(f"Scale decreased to {self.scale:.2f}")
            elif key == keyboard.Key.esc:
                self.running = False
                return False
            elif key == keyboard.Key.enter:
                # # Toggle jaw state ONLY if teleop is active
                if self.teleop_active:
                    self.move_jaw()
        except AttributeError:
            pass

    def run(self):
        self.home()
        self.running = True
        listener = keyboard.Listener(on_press=self.on_press)
        listener.start()

        print("Press SPACE to toggle teleoperation on/off.")
        print("Use UP/DOWN arrows to adjust scaling dynamically.")
        print("Press ESC to exit.")

        while self.running:
            self.move_cartesian()
            time.sleep(self.expected_interval)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Trakstar teleop for dVRK arms')
    parser.add_argument('-a', '--arm', choices=['PSM1', 'PSM2', 'PSM3', 'ECM'], default='PSM1',
                        help='Arm name to control (PSM1, PSM2, PSM3, ECM)')
    args, unknown = parser.parse_known_args()

    argv = crtk.ral.parse_argv(unknown)
    ral = crtk.ral('trakstar_psm_teleop_toggle')

    config_dict = {
        'arm_name': args.arm,
        'scale': 0.2,
        'expected_interval': 0.001,  # 0.01 for 10 ms
        'control_type': 's'
    }

    teleop = ViveTeleop(ral, config_dict)
    ral.spin_and_execute(teleop.run)

