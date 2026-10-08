#!/usr/bin/env bash
# Record the console sessions used in the tutorial videos (real deploy chain in the MuJoCo rehearsal).
set -u
cd "$(dirname "$0")/.."
mkdir -p runs/video
rec() { name=$1; layout=$2; script=$3; ./g1 sim --layout "$layout" --record "runs/video/$name" --script "$script" \
          > "runs/video/$name.out" 2>&1; echo "$name done: $(tail -1 runs/video/$name.out)"; }
rec A_quickstart beams2-100 "wait:init,sleep:1,key:s,wait:control,sleep:2.5,key:g,wait:run,wait:arrived,sleep:2.5,key:t,sleep:4"
rec B_safety empty-100 "wait:init,sleep:1,key:s,wait:control,sleep:2.5,key:g,wait:run,sleep:2.2,key:t,sleep:4,goal:2.5:0,sleep:0.5,key:g,wait:run,wait:arrived,sleep:2.5,key:o,sleep:4"
rec C_detour detour-100 "wait:init,sleep:1,key:s,wait:control,sleep:2.5,key:g,wait:run,wait:arrived,sleep:3"
rec D_beams2_101 beams2-101 "wait:init,sleep:1,key:s,wait:control,sleep:2.5,key:g,wait:run,wait:arrived,sleep:3"
echo RECORD_DONE
