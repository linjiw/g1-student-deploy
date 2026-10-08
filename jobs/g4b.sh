#!/usr/bin/env bash
# Gate G4 (continued): single-effect arms, latency sweep with predictive anchoring, combined deploy arm
# (one plan in flight; two in flight was harmful, see docs/RUNBOOK.md).
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
ODOM='{"xy_rw_m": 0.04, "yaw_rw_rad": 0.02, "xy_white_m": 0.01, "yaw_white_rad": 0.005}'
run() {
  name=$1; shift
  if [ -f "runs/$name/results.json" ] && [ "$(python3 -c "import json;print(len(json.load(open('runs/$name/results.json'))))")" = 98 ]; then return; fi
  echo "=== $name $*"
  $PY tools/eval_panel.py --name "$name" --workers 3 "$@" > "runs/$name.log" 2>&1
  grep "^==\|^   " "runs/$name.log"
}
run g4-mjray              --lidar mjray
run g4-stand              --start stand
run g4-route-sensor       --route sensor
run g4-odom               --odom-noise "$ODOM"
run g4-deploy1            --lidar mjray --start stand --route sensor --odom-noise "$ODOM" --prescan-s 0.3 \
                          --latency-s 0.32 --anchor predict
run g4-lat016-pred        --prescan-s 0.3 --latency-s 0.16 --anchor predict
run g4-lat050-pred        --prescan-s 0.3 --latency-s 0.50 --anchor predict
run g4-lat050-meas        --prescan-s 0.3 --latency-s 0.50
echo G4B_DONE
