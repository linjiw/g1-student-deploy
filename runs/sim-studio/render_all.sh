#!/bin/bash
cd /home/robotixx/g1-student-deploy; source env.sh; export MUJOCO_GL=egl
render() {
  name=$1; speed=$2; d=runs/sim-studio/exp-$name
  go=$(python3 -c "import json; ev=json.load(open('$d/console_summary.json'))['events']; print(max(0, min(float(e.split()[0]) for e in ev if 'GO (stream on' in e) - 2))")
  .venv/bin/python tools/render_session.py $d --out runs/sim-studio/videos/$name.mp4 --from $go --speed $speed --title "$3" --azimuth 135 --elevation -20 --distance 4.2 > runs/sim-studio/videos/$name.log 2>&1
  echo "$name $(du -h runs/sim-studio/videos/$name.mp4 2>/dev/null | cut -f1)"
}
export -f render
printf '%s\n' "door-through 2 Door twin: goal 2.5 m through the door (2x speed)" "door-left 3 Door twin: goal 2.6 m, 0.5 m left (3x speed)" "door-right 1 Door twin: goal 3.0 m, 0.5 m right" "door-stop-before 1 Door twin: stop 0.9 m before the door" "spot-1p5 1 Start spot: 1.5 m (the real walk 1)" "spot-2p0 1 Start spot: 2.0 m (the real walk 2)" "spot-2p0-left 1 Start spot: 2.0 m, 0.5 m left" | xargs -P 3 -I{} bash -c 'set -- {}; n=$1; s=$2; shift 2; render $n $s "$*"'
