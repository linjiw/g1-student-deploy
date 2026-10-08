#!/usr/bin/env bash
# What makes the combined deploy-realistic arm (69/98, 29 collisions mostly under "safe" plans) lose?
# Leave-one-out arms around it, then the rest of the G4 queue. 2 workers (shared box).
set -u
cd "$(dirname "$0")/.."
while pgrep -f "eval_panel.py --nam[e]" > /dev/null; do sleep 20; done
PY=.venv/bin/python
W=${G4_WORKERS:-2}
LIO='{"xy_rw_m": 0.01, "yaw_rw_rad": 0.005, "xy_white_m": 0.002, "yaw_white_rad": 0.001}'
BASE=(--start stand --route sensor --prescan-s 0.3)
run() {
  name=$1; shift
  if [ -f "runs/$name/results.json" ] && [ "$(python3 -c "import json;print(len(json.load(open('runs/$name/results.json'))))")" = 98 ]; then return; fi
  echo "=== $name $*"
  $PY tools/eval_panel.py --name "$name" --workers "$W" "$@" > "runs/$name.log" 2>&1
  grep "^==\|^   " "runs/$name.log"
}
run g4r-mid360-pelvis-density --lidar mid360 --lidar-cfg '{"mount_body": "pelvis", "max_points_per_frame": 5400}'
run g4c-lio               --odom-noise "$LIO"
run g4s-real-nolat        "${BASE[@]}" --lidar mid360 --odom-noise "$LIO"
run g4s-real-noodom       "${BASE[@]}" --lidar mid360 --latency-s 0.32 --anchor predict
run g4s-real-mjray        "${BASE[@]}" --lidar mjray --odom-noise "$LIO" --latency-s 0.32 --anchor predict
run g4s-stand-mid360      --start stand --lidar mid360
run g4d-occl-steep        --lidar mid360 --lidar-cfg '{"occlusion": {"az_deg": 45, "el_lo_deg": 30}}'
run g4d-mount-resid       --lidar mid360 --lidar-cfg '{"mount_error": {"pitch_deg": 0.5, "roll_deg": 0.5, "z_m": 0.01}}'
run g4c-legodom           --odom-noise '{"leg": true, "yaw_drift_rad_s": 0.002}'
run g4-lat050-pred        --prescan-s 0.3 --latency-s 0.50 --anchor predict
run g4e-stand-look10      --start stand --goal-lookahead 1.0
run g4d-occl-front        --lidar mid360 --lidar-cfg '{"occlusion": {"az_deg": 45}}'
run g4d-mount-3deg        --lidar mid360 --lidar-cfg '{"mount_error": {"pitch_deg": 3.0}}'
echo G4S_DONE
