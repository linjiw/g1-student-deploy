#!/usr/bin/env bash
# Faster planner configs, each at its own measured p95 latency on the loaded deploy PC (bench, 2026-10-08):
# K16/10 0.389 s (g4u arms) | K8/10 0.328 s | K16/6 0.282 s | K8/6 0.243 s. Runs after g4u.
set -u
cd "$(dirname "$0")/.."
while pgrep -f "jobs/g4[u].sh" > /dev/null; do sleep 30; done
PY=.venv/bin/python
W=${G4_WORKERS:-2}
run() {
  name=$1; shift
  if [ -f "runs/$name/results.json" ] && [ "$(python3 -c "import json;print(len(json.load(open('runs/$name/results.json'))))")" = 98 ]; then return; fi
  echo "=== $name $*"
  $PY tools/eval_panel.py --name "$name" --workers "$W" "$@" > "runs/$name.log" 2>&1
  grep "^==\|^   " "runs/$name.log"
}
COMMON=(--prescan-s 0.3 --anchor predict)
run g4v-k16s6-lat028      "${COMMON[@]}" --ode-steps 6 --latency-s 0.28
run g4v-k8s10-lat032      "${COMMON[@]}" --candidates 8 --latency-s 0.32
run g4v-k8s6-lat024       "${COMMON[@]}" --candidates 8 --ode-steps 6 --latency-s 0.24
echo G4V_DONE
