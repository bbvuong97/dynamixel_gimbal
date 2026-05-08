#!/usr/bin/env python3

"""Simple Tkinter GUI for DVRK teleop + gimbal torque and align control.

- Shows teleop active ON/OFF (subscribes to `/dvrk_teleop_gimbal/enable`).
- Shows torque ON/OFF (subscribes to `/dynamixel_gimbal/torque_state`).
- Buttons:
  - Toggle Teleop -> publishes on `/dvrk_teleop_gimbal/enable` (Bool)
  - Toggle Torque -> publishes on `/dynamixel_gimbal/torque_enable_cmd` (Bool)
  - Run Align -> publishes on `/dynamixel_gimbal/align_cmd` (Bool, True triggers)

Run: `ros2 run gimbal_v2 teleop_gui.py` or `python3 teleop_gui.py`
"""

import threading
import tkinter as tk
from tkinter import ttk
import subprocess
import shutil

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import Bool, Float64, String


class TeleopGUINode(Node):
    def __init__(self):
        super().__init__('dvrk_teleop_gui_node')

        state_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        self.teleop_pub = self.create_publisher(Bool, '/dvrk_teleop_gimbal/enable', state_qos)
        self.torque_cmd_pub = self.create_publisher(Bool, '/dynamixel_gimbal/torque_enable_cmd', 10)
        self.align_pub = self.create_publisher(Bool, '/dynamixel_gimbal/align_cmd', 10)
        self.vive_scale_delta_pub = self.create_publisher(Float64, '/dvrk_teleop_gimbal/vive_scale_delta', 10)
        self.arm_select_pub = self.create_publisher(String, '/dvrk_teleop_gimbal/arm_select', state_qos)

        self.teleop_state = False
        self.torque_state = False
        self.vive_scale = 0.2
        self.vive_scale_step = 0.1
        self.available_arms = ('PSM1', 'PSM2', 'PSM3')
        self.initial_arm = str(self._param('arm', 'PSM1')).strip().upper()
        if self.initial_arm not in self.available_arms:
            self.initial_arm = 'PSM1'
        self.current_arm = self.initial_arm

        # Subscriptions to reflect external changes
        self.create_subscription(Bool, '/dvrk_teleop_gimbal/enable', self._teleop_cb, state_qos)
        self.create_subscription(Bool, '/dynamixel_gimbal/torque_state', self._torque_state_cb, state_qos)
        self.create_subscription(Float64, '/dvrk_teleop_gimbal/vive_scale', self._vive_scale_cb, state_qos)
        self.create_subscription(String, '/dvrk_teleop_gimbal/arm_state', self._arm_state_cb, state_qos)

    def _param(self, name, default):
        if not self.has_parameter(name):
            self.declare_parameter(name, default)
        return self.get_parameter(name).value

    def _teleop_cb(self, msg: Bool):
        self.teleop_state = bool(msg.data)

    def _torque_state_cb(self, msg: Bool):
        self.torque_state = bool(msg.data)

    def _vive_scale_cb(self, msg: Float64):
        self.vive_scale = float(msg.data)

    def _arm_state_cb(self, msg: String):
        if msg.data in self.available_arms:
            self.current_arm = msg.data

    def publish_arm_select(self, arm_name: str):
        if arm_name not in self.available_arms:
            return
        msg = String()
        msg.data = arm_name
        self.arm_select_pub.publish(msg)
        self.current_arm = arm_name

    def publish_teleop(self, enable: bool):
        msg = Bool()
        msg.data = bool(enable)
        self.teleop_pub.publish(msg)
        # also update local copy
        self.teleop_state = bool(enable)

    def publish_torque(self, enable: bool):
        msg = Bool()
        msg.data = bool(enable)
        self.torque_cmd_pub.publish(msg)
        # local copy will be updated if gimbal node publishes state

    def publish_align(self):
        msg = Bool()
        msg.data = True
        self.align_pub.publish(msg)

    def publish_vive_scale_delta(self, delta: float):
        msg = Float64()
        msg.data = float(delta)
        self.vive_scale_delta_pub.publish(msg)


class TeleopGUIApp:
    def __init__(self, node: TeleopGUINode):
        self.node = node
        self.root = tk.Tk()
        self.root.title('dVRK Teleop / Gimbal Control')
        
        # Set up styles for larger fonts
        style = ttk.Style()
        style.configure('Label.TLabel', font=('TkDefaultFont', 12))
        style.configure('TButton', font=('TkDefaultFont', 11), padding=10)
        style.configure('Status.TLabel', font=('TkDefaultFont', 14, 'bold'))
        style.configure('Display.TLabel', font=('TkDefaultFont', 13))
        
        self._build_ui()
        # track last-known states so we can detect external changes
        self._last_teleop_state = self.node.teleop_state
        self._last_torque_state = self.node.torque_state
        self._suppress_next_teleop_beep = False
        # Periodically refresh status from ROS
        self._update_ui()
        self.root.protocol('WM_DELETE_WINDOW', self._on_close)

    def play_beep_on(self):
        """Speak 'ON' for teleop ON (system TTS with bell fallback)."""
        self.speak('ON')

    def play_beep_off(self):
        """Speak 'OFF' once for teleop OFF (system TTS with bell fallback)."""
        self.speak('OFF')

    def speak(self, text: str):
        """Try system TTS: `spd-say`, then `espeak`. Fallback to bell."""
        try:
            if shutil.which('spd-say'):
                subprocess.Popen(['spd-say', text])
            elif shutil.which('espeak'):
                subprocess.Popen(['espeak', text])
            else:
                self.root.bell()
        except Exception:
            # Never crash the GUI for audio errors
            try:
                self.root.bell()
            except Exception:
                pass

    def _build_ui(self):
        frm = ttk.Frame(self.root, padding=10)
        frm.grid()

        # Teleop status
        ttk.Label(frm, text='Teleop:', style='Label.TLabel').grid(column=0, row=0, sticky='w')
        self.teleop_label = ttk.Label(frm, text='OFF', foreground='red', width=8, style='Status.TLabel')
        self.teleop_label.grid(column=1, row=0, sticky='w')
        self.teleop_btn = ttk.Button(frm, text='Toggle Teleop', command=self._on_toggle_teleop, width=16, takefocus=False)
        self.teleop_btn.grid(column=2, row=0, padx=8)

        # Torque status
        ttk.Label(frm, text='Torque:', style='Label.TLabel').grid(column=0, row=1, sticky='w')
        self.torque_label = ttk.Label(frm, text='OFF', foreground='red', width=8, style='Status.TLabel')
        self.torque_label.grid(column=1, row=1, sticky='w')
        self.torque_btn = ttk.Button(frm, text='Toggle Torque', command=self._on_toggle_torque, width=16, takefocus=False)
        self.torque_btn.grid(column=2, row=1, padx=8)

        # Vive scale controls
        ttk.Label(frm, text='Vive Scale:', style='Label.TLabel').grid(column=0, row=2, sticky='w')
        self.vive_scale_var = tk.StringVar(value=f'{self.node.vive_scale:.2f}')
        self.vive_scale_entry = ttk.Entry(frm, textvariable=self.vive_scale_var, width=10, state='readonly', justify='center', font=('TkDefaultFont', 13))
        self.vive_scale_entry.grid(column=1, row=2, sticky='w')
        self.vive_scale_minus_btn = ttk.Button(frm, text='-', width=4, command=self._on_vive_scale_down, takefocus=False)
        self.vive_scale_minus_btn.grid(column=2, row=2, sticky='w', padx=(0, 4))
        self.vive_scale_plus_btn = ttk.Button(frm, text='+', width=4, command=self._on_vive_scale_up, takefocus=False)
        self.vive_scale_plus_btn.grid(column=3, row=2, sticky='w')

        # Active arm selection
        ttk.Label(frm, text='Active Arm:', style='Label.TLabel').grid(column=0, row=3, sticky='w')
        self.arm_var = tk.StringVar(value=self.node.initial_arm)
        self.arm_combo = ttk.Combobox(
            frm,
            textvariable=self.arm_var,
            values=self.node.available_arms,
            state='readonly',
            width=8,
            font=('TkDefaultFont', 13),
        )
        self.arm_combo.grid(column=1, row=3, sticky='w')
        self.arm_combo.bind('<<ComboboxSelected>>', self._on_arm_selected)
        self.arm_status_label = ttk.Label(frm, text=f'Current: {self.node.current_arm}', style='Display.TLabel')
        self.arm_status_label.grid(column=2, row=3, columnspan=2, sticky='w')

        # Align button
        self.align_btn = ttk.Button(frm, text='Run Align', command=self._on_run_align, width=30, takefocus=False)
        self.align_btn.grid(column=0, row=4, columnspan=4, pady=(12,0))

        for child in frm.winfo_children():
            child.grid_configure(pady=4)

    def _on_toggle_teleop(self):
        new_state = not self.node.teleop_state
        if new_state:
            self.play_beep_on()  # Double beep for ON
        else:
            self.play_beep_off()  # Single beep for OFF
        # suppress the next automatic beep triggered by the incoming state
        # update (which reflects this same change via subscription)
        self._suppress_next_teleop_beep = True
        self.node.publish_teleop(new_state)

    def _on_toggle_torque(self):
        new_state = not self.node.torque_state
        # No sound for torque toggles; just publish
        self.node.publish_torque(new_state)

    def _on_run_align(self):
        self.node.publish_align()

    def _on_arm_selected(self, _event):
        self.node.publish_arm_select(self.arm_var.get())

    def _on_vive_scale_up(self):
        self.node.publish_vive_scale_delta(self.node.vive_scale_step)

    def _on_vive_scale_down(self):
        self.node.publish_vive_scale_delta(-self.node.vive_scale_step)

    def _update_ui(self):
        # Detect teleop state changes that may have been caused externally
        if self.node.teleop_state != self._last_teleop_state:
            if self._suppress_next_teleop_beep:
                self._suppress_next_teleop_beep = False
            else:
                if self.node.teleop_state:
                    self.play_beep_on()
                else:
                    self.play_beep_off()
            self._last_teleop_state = self.node.teleop_state

        # Detect torque state changes that may have been caused externally
        if self.node.torque_state != self._last_torque_state:
            # No audio for torque changes; just update last-known state
            self._last_torque_state = self.node.torque_state

        # Update teleop label
        if self.node.teleop_state:
            self.teleop_label.config(text='ON', foreground='green')
        else:
            self.teleop_label.config(text='OFF', foreground='red')

        # Update torque label
        if self.node.torque_state:
            self.torque_label.config(text='ON', foreground='green')
        else:
            self.torque_label.config(text='OFF', foreground='red')

        if self.arm_var.get() != self.node.current_arm:
            self.arm_var.set(self.node.current_arm)
        self.arm_status_label.config(text=f'Current: {self.node.current_arm}')

        self.vive_scale_var.set(f'{self.node.vive_scale:.2f}')

        # schedule next update
        self.root.after(200, self._update_ui)

    def _on_close(self):
        try:
            rclpy.shutdown()
        except Exception:
            pass
        self.root.quit()

    def run(self):
        self.root.mainloop()


def main():
    rclpy.init()
    node = TeleopGUINode()

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    # Spin executor in background thread
    exec_thread = threading.Thread(target=executor.spin, daemon=True)
    exec_thread.start()

    app = TeleopGUIApp(node)
    try:
        app.run()
    finally:
        try:
            executor.shutdown()
        except Exception:
            pass
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
