#!/bin/bash
# Rear-blind comparison (2026-10-24): today's scenes with the node's --rear-blind-deg 180, plus courses needing side/rear view.
cd /home/robotixx/g1-student-deploy
source env.sh
export DISPLAY=:3 G1_PANEL_TOKEN=jBQBmNRHT5ey
L3=$HOME/research-data/g1-deploy-studio/l3
while read -r name layout ahead left extra; do
  [ -z "$name" ] && continue
  out=runs/sim-studio/exp-$name
  rm -rf "$out"
  mkdir -p "$L3/exp-$name" && ln -sfn "$PWD/$out" "$L3/exp-$name/rehearse"
  echo "=== $name: $layout goal $ahead ahead $left left $extra  $(date +%T)"
  ./g1 sim --layout "$layout" --viewer --panel 8771 --guards deploy --goal-lookahead 1.0 --debug-plans --record "$out" $extra \
     --step-timeout 90 --script "wait:init,key:s,wait:control,sleep:3,goal:$ahead:$left,key:g,wait:arrived,sleep:4" \
     < /dev/null > "$out.out" 2>&1
  python3 -c "import json; t=json.load(open('sim/courses/$layout/task.json')); json.dump(t['scene_boxes_world'], open('$out/boxes.json','w'))"
  grep -A2 '^report' "$out.out" | grep -v Warning | head -2
done <<'JOBS'
door-through-nrb lab-door-aligned 2.5 0.0 --rear-blind-deg 180 --rear-blind-range-m 1.0
door-left-nrb lab-door-aligned 2.6 0.5 --rear-blind-deg 180 --rear-blind-range-m 1.0
door-right-nrb lab-door-aligned 3.0 -0.5 --rear-blind-deg 180 --rear-blind-range-m 1.0
door-stop-before-nrb lab-door-aligned 0.9 0.0 --rear-blind-deg 180 --rear-blind-range-m 1.0
spot-1p5-nrb lab-start-spot 1.5 0.0 --rear-blind-deg 180 --rear-blind-range-m 1.0
spot-2p0-nrb lab-start-spot 2.0 0.0 --rear-blind-deg 180 --rear-blind-range-m 1.0
spot-2p0-left-nrb lab-start-spot 2.0 0.5 --rear-blind-deg 180 --rear-blind-range-m 1.0
slalom-nrb slalom 3.4 0.0 --rear-blind-deg 180 --rear-blind-range-m 1.0
pillar-nrb pillar 3.0 0.0 --rear-blind-deg 180 --rear-blind-range-m 1.0
wall-detour-nrb wall-detour 3.2 0.0 --rear-blind-deg 180 --rear-blind-range-m 1.0
rp-rope-behind-full rp-rope-behind 2.0 0.0
rp-rope-behind-rb rp-rope-behind 2.0 0.0 --rear-blind-deg 180
rp-rope-behind-nrb rp-rope-behind 2.0 0.0 --rear-blind-deg 180 --rear-blind-range-m 1.0
rp-door-full rp-door 3.0 0.0
rp-door-rb rp-door 3.0 0.0 --rear-blind-deg 180
rp-door-nrb rp-door 3.0 0.0 --rear-blind-deg 180 --rear-blind-range-m 1.0
rp-door-rope-full rp-door-rope 3.0 0.0
rp-door-rope-rb rp-door-rope 3.0 0.0 --rear-blind-deg 180
rp-door-rope-nrb rp-door-rope 3.0 0.0 --rear-blind-deg 180 --rear-blind-range-m 1.0
JOBS
echo "=== all done $(date +%T)"
