#!/usr/bin/env bash
# Gate G4e: slower walking for first hardware runs (shorter local goal), from standing (runs after g4q).
set -u
cd "$(dirname "$0")/.."
while pgrep -f "jobs/g4[q].sh" > /dev/null; do sleep 30; done
PY=.venv/bin/python
W=${G4_WORKERS:-2}
run() {
  name=$1; shift
  if [ -f "runs/$name/results.json" ] && [ "$(python3 -c "import json;print(len(json.load(open('runs/$name/results.json'))))")" = 98 ]; then return; fi
  echo "=== $name $*"
  $PY tools/eval_panel.py --name "$name" --workers "$W" "$@" > "runs/$name.log" 2>&1
  grep "^==\|^   " "runs/$name.log"
}
run g4e-stand-look10      --start stand --goal-lookahead 1.0
run g4e-stand-look08      --start stand --goal-lookahead 0.8
echo G4E_DONE
