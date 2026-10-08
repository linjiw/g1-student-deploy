#!/usr/bin/env bash
# Gate G4d: head-shell occlusion and calibration residuals with the realistic MID-360 (runs after g4c).
set -u
cd "$(dirname "$0")/.."
while pgrep -f "jobs/g4[c].sh" > /dev/null; do sleep 30; done
PY=.venv/bin/python
run() {
  name=$1; shift
  if [ -f "runs/$name/results.json" ] && [ "$(python3 -c "import json;print(len(json.load(open('runs/$name/results.json'))))")" = 98 ]; then return; fi
  echo "=== $name $*"
  $PY tools/eval_panel.py --name "$name" --workers 3 "$@" > "runs/$name.log" 2>&1
  grep "^==\|^   " "runs/$name.log"
}
# shell blocks steep-down rays in front (sensor elevation >= 30 deg on the inverted mount = >= 30 deg below horizon)
run g4d-occl-steep        --lidar mid360 --lidar-cfg '{"occlusion": {"az_deg": 45, "el_lo_deg": 30}}'
# worst case reported on one unit: the whole +-45 deg front sector blind
run g4d-occl-front        --lidar mid360 --lidar-cfg '{"occlusion": {"az_deg": 45}}'
# residual after calibration (0.5 deg, 1 cm) and an uncalibrated head (3 deg, as measured on one unit)
run g4d-mount-resid       --lidar mid360 --lidar-cfg '{"mount_error": {"pitch_deg": 0.5, "roll_deg": 0.5, "z_m": 0.01}}'
run g4d-mount-3deg        --lidar mid360 --lidar-cfg '{"mount_error": {"pitch_deg": 3.0}}'
echo G4D_DONE
