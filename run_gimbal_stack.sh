#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TRACKER_CMD="cd \"$SCRIPT_DIR/src/gimbal_v2/src/scripts\" && python3 vive_tracker_udp_to_ros2.py"
LAUNCH_CMD="cd \"$SCRIPT_DIR\" && source .venv/bin/activate && source /opt/ros/jazzy/setup.bash && source /home/stanford/ros2_ws/install/setup.bash && source install/setup.bash && ros2 launch gimbal_v2 gimbal_tf_teleop_dvrk_jaw_trigger.launch.py launch_gui:=true"

launch_with_gnome_terminal() {
  gnome-terminal -- bash -lc "$TRACKER_CMD; exec bash" || return 1
  gnome-terminal -- bash -lc "$LAUNCH_CMD; exec bash" || return 1
}

launch_with_xterm() {
  xterm -hold -e bash -lc "$TRACKER_CMD" &
  xterm -hold -e bash -lc "$LAUNCH_CMD" &
}

launch_with_tmux() {
  tmux new-session -d -s gimbal_stack "$TRACKER_CMD"
  tmux new-window -t gimbal_stack:2 "$LAUNCH_CMD"
  echo "Launched in tmux session: gimbal_stack"
  echo "Attach with: tmux attach -t gimbal_stack"
}

if command -v xterm >/dev/null 2>&1; then
  launch_with_xterm
  exit 0
fi

if command -v gnome-terminal >/dev/null 2>&1; then
  if launch_with_gnome_terminal; then
    exit 0
  fi
  echo "Warning: gnome-terminal is installed but failed to start."
fi

if command -v tmux >/dev/null 2>&1; then
  launch_with_tmux
  exit 0
fi

echo "Error: no working terminal launcher found."
echo "Install xterm or tmux, then rerun this script."
exit 1
