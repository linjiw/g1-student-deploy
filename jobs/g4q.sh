#!/usr/bin/env bash
# Remaining G4 arms in priority order, 2 workers (the box is shared with ~/flow panel v2: Isaac runs need
# RAM headroom; 3 workers of ~2.4 GB each slowed it). Waits for the arm already in flight.
set -u
cd "$(dirname "$0")/.."
while pgrep -f "eval_panel.py --nam[e]" > /dev/null; do sleep 20; done
PY=.venv/bin/python
W=${G4_WORKERS:-2}
LIO='{"xy_rw_m": 0.01, "yaw_rw_rad": 0.005, "xy_white_m": 0.002, "yaw_white_rad": 0.001}'
run() {
  name=$1; shift
  if [ -f "runs/$name/results.json" ] && [ "$(python3 -c "import json;print(len(json.load(open('runs/$name/results.json'))))")" = 98 ]; then return; fi
  echo "=== $name $*"
  $PY tools/eval_panel.py --name "$name" --workers "$W" "$@" > "runs/$name.log" 2>&1
  grep "^==\|^   " "runs/$name.log"
}
run g4c-mid360            --lidar mid360
run g4c-deploy-real       --lidar mid360 --start stand --route sensor --odom-noise "$LIO" --prescan-s 0.3 \
                          --latency-s 0.32 --anchor predict
run g4c-lio               --odom-noise "$LIO"
run g4d-occl-steep        --lidar mid360 --lidar-cfg '{"occlusion": {"az_deg": 45, "el_lo_deg": 30}}'
run g4d-mount-resid       --lidar mid360 --lidar-cfg '{"mount_error": {"pitch_deg": 0.5, "roll_deg": 0.5, "z_m": 0.01}}'
run g4c-legodom           --odom-noise '{"leg": true, "yaw_drift_rad_s": 0.002}'
run g4-lat050-pred        --prescan-s 0.3 --latency-s 0.50 --anchor predict
run g4c-odom-drift        --odom-noise '{"xy_rw_m": 0.04, "yaw_rw_rad": 0.02}'
run g4c-odom-jitter       --odom-noise '{"xy_white_m": 0.01, "yaw_white_rad": 0.005}'
run g4d-occl-front        --lidar mid360 --lidar-cfg '{"occlusion": {"az_deg": 45}}'
run g4d-mount-3deg        --lidar mid360 --lidar-cfg '{"mount_error": {"pitch_deg": 3.0}}'
run g4c-mid360-nodeskew   --lidar mid360 --lidar-cfg '{"deskew": "none"}'
run g4c-mid360-noisy      --lidar mid360 --lidar-cfg '{"dropout": 0.2, "range_noise_m": 0.04}'
echo G4Q_DONE
