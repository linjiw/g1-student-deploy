#!/usr/bin/env bash
# Gate G4c: the G1's real sensing in MuJoCo (runs after jobs/g4b.sh).
#   mid360      real non-repetitive MID-360 pattern (200k pts/s), per-point timing on the torso mount,
#               per-slice self filter, ideal deskew
#   variants    no deskew, mount calibration error, dropout + range noise, leg odometry, LIO-level drift
#   deploy-real everything the robot will have: mid360 + standing start + lidar route + LIO-level odometry
#               + 0.32 s planner latency with predictive anchoring
set -u
cd "$(dirname "$0")/.."
while pgrep -f "jobs/g4b.s[h]" > /dev/null; do sleep 30; done
PY=.venv/bin/python
LIO='{"xy_rw_m": 0.01, "yaw_rw_rad": 0.005, "xy_white_m": 0.01, "yaw_white_rad": 0.003}'
run() {
  name=$1; shift
  if [ -f "runs/$name/results.json" ] && [ "$(python3 -c "import json;print(len(json.load(open('runs/$name/results.json'))))")" = 98 ]; then return; fi
  echo "=== $name $*"
  $PY tools/eval_panel.py --name "$name" --workers 3 "$@" > "runs/$name.log" 2>&1
  grep "^==\|^   " "runs/$name.log"
}
run g4c-mid360            --lidar mid360
run g4c-mid360-nodeskew   --lidar mid360 --lidar-cfg '{"deskew": "none"}'
run g4c-mid360-pitch2     --lidar mid360 --lidar-cfg '{"mount_error": {"pitch_deg": 2.0, "z_m": 0.02}}'
run g4c-mid360-noisy      --lidar mid360 --lidar-cfg '{"dropout": 0.2, "range_noise_m": 0.04}'
run g4c-legodom           --odom-noise '{"leg": true, "yaw_drift_rad_s": 0.002}'
run g4c-lio               --odom-noise "$LIO"
run g4c-deploy-real       --lidar mid360 --start stand --route sensor --odom-noise "$LIO" --prescan-s 0.3 \
                          --latency-s 0.32 --anchor predict
echo G4C_DONE
