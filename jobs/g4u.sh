#!/usr/bin/env bash
# Latency margin on the deploy PC: the planner measured p95 0.357 s / p99 0.382 s / max 0.416 s under full GPU
# contention (RTX 5080 shared). Lock-step arms at those latencies (predictive anchoring, pre-scan, 1 in flight).
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
W=${G4_WORKERS:-2}
run() {
  name=$1; shift
  if [ -f "runs/$name/results.json" ] && [ "$(python3 -c "import json;print(len(json.load(open('runs/$name/results.json'))))")" = 98 ]; then return; fi
  echo "=== $name $*"
  $PY tools/eval_panel.py --name "$name" --workers "$W" "$@" > "runs/$name.log" 2>&1
  grep "^==\|^   " "runs/$name.log"
}
run g4u-lat036-pred       --prescan-s 0.3 --latency-s 0.36 --anchor predict
run g4u-lat040-pred       --prescan-s 0.3 --latency-s 0.40 --anchor predict
run g4u-lat044-pred       --prescan-s 0.3 --latency-s 0.44 --anchor predict
echo G4U_DONE
