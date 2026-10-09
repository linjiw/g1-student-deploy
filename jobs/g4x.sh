#!/usr/bin/env bash
# Denoiser precision (fast planner, --planner-precision): TF32 / bf16 tensor cores for the DiT forward only
# (guidance geometry stays fp32). Velocity error vs fp32: tf32 ~5e-4, bf16 ~6e-3 relative (fp32 path 5e-7).
# Same realistic condition and latency as the fp32 fused arm g4w-fused-real-lat018 (82/98), so the
# difference isolates the arithmetic (the latency curve is flat below 0.32 s).
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
run g4x-tf32-real-lat018 "${REAL[@]}" --latency-s 0.18 --planner-precision tf32
run g4x-bf16-real-lat018 "${REAL[@]}" --latency-s 0.18 --planner-precision bf16
echo G4X_DONE
