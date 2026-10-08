#!/usr/bin/env bash
# Fast planner (g1deploy/fastplan.py, --planner-impl fused) in closed loop, 98 held-out episodes.
#  a) equivalence: the eager arms' exact conditions -> expect the eager scores within panel noise
#     (eager: base 0.32 s predictive 79/98; deploy-real 75/98)
#  b) benefit: deploy-real at the fused planner's measured p95 on the loaded deploy PC (0.17 s -> 0.18 s)
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
W=${G4_WORKERS:-2}
run() {
  name=$1; shift
  if [ -f "runs/$name/results.json" ] && [ "$(python3 -c "import json;print(len(json.load(open('runs/$name/results.json'))))")" = 98 ]; then return; fi
  echo "=== $name $*"
  $PY tools/eval_panel.py --name "$name" --workers "$W" --planner-impl fused "$@" > "runs/$name.log" 2>&1
  grep "^==\|^   " "runs/$name.log"
}
REAL=(--prescan-s 0.3 --anchor predict --start stand --route sensor --lidar mid360
      --odom-noise '{"xy_rw_m": 0.01, "yaw_rw_rad": 0.005, "xy_white_m": 0.002, "yaw_white_rad": 0.001}')
run g4w-fused-base-lat032 --prescan-s 0.3 --anchor predict --latency-s 0.32
run g4w-fused-real-lat032 "${REAL[@]}" --latency-s 0.32
run g4w-fused-real-lat018 "${REAL[@]}" --latency-s 0.18
echo G4W_DONE
