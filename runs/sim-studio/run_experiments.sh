#!/bin/bash
# Scripted deploy-kit simulations of today's real scenes (2026-10-24), recorded for Studio Replay + video.
cd /home/robotixx/g1-student-deploy
source env.sh
export DISPLAY=:3 G1_PANEL_TOKEN=jBQBmNRHT5ey
L3=$HOME/research-data/g1-deploy-studio/l3
while read -r name layout ahead left; do
  [ -z "$name" ] && continue
  out=runs/sim-studio/exp-$name
  rm -rf "$out"
  mkdir -p "$L3/exp-$name" && ln -sfn "$PWD/$out" "$L3/exp-$name/rehearse"
  echo "=== $name: $layout goal $ahead ahead $left left  $(date +%T)"
  ./g1 sim --layout "$layout" --viewer --panel 8771 --guards deploy --goal-lookahead 1.0 --debug-plans --record "$out" \
     --step-timeout 120 --script "wait:init,key:s,wait:control,sleep:3,goal:$ahead:$left,key:g,wait:arrived,sleep:4" \
     < /dev/null > "$out.out" 2>&1
  python3 -c "import json; t=json.load(open('sim/courses/$layout/task.json')); json.dump(t['scene_boxes_world'], open('$out/boxes.json','w'))"
  grep -A3 '^report' "$out.out" | grep -v Warning | head -3
done <<'JOBS'
door-through lab-door-aligned 2.5 0.0
door-left lab-door-aligned 2.6 0.5
door-right lab-door-aligned 3.0 -0.5
door-stop-before lab-door-aligned 0.9 0.0
spot-1p5 lab-start-spot 1.5 0.0
spot-2p0 lab-start-spot 2.0 0.0
spot-2p0-left lab-start-spot 2.0 0.5
JOBS
echo "=== all done $(date +%T)"
