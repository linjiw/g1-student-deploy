#!/usr/bin/env bash
# Gate G4: deployment effects on the 98 held-out test-v1 episodes, one at a time, then combined.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
ODOM='{"xy_rw_m": 0.04, "yaw_rw_rad": 0.02, "xy_white_m": 0.01, "yaw_white_rad": 0.005}'
run() { name=$1; shift; [ -f runs/$name/results.json ] && [ "$(python3 -c "import json;print(len(json.load(open('runs/$name/results.json'))))")" = 98 ] && return; echo "=== $name $*"; $PY tools/eval_panel.py --name "$name" --workers 3 "$@" 2>&1 | grep -v Warn | tail -5; }
run g4-prescan            --prescan-s 0.3
run g4-lat032-meas        --prescan-s 0.3 --latency-s 0.32
run g4-lat032-pred        --prescan-s 0.3 --latency-s 0.32 --anchor predict
run g4-lat032-pred-if2    --prescan-s 0.3 --latency-s 0.32 --anchor predict --inflight 2
run g4-mjray              --lidar mjray
run g4-stand              --start stand
run g4-route-sensor       --route sensor
run g4-odom               --odom-noise "$ODOM"
run g4-deploy             --lidar mjray --start stand --route sensor --odom-noise "$ODOM" --prescan-s 0.3 \
                          --latency-s 0.32 --anchor predict --inflight 2
echo G4_DONE
