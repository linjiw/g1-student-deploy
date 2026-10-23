#!/usr/bin/env bash
# Robot-day dress rehearsal in simulation: every rung of docs/DEPLOY_KIT.md §3 through the kit, then one summary.
#   H0 doctor · H1/H2 check + calibrate against a standing sim robot · H3 SONIC stand · H4 replay · H5 open floor
#   (leg odometry and truth) · H6 obstacles · navigation mission · ceiling trap · Studio real-robot link
# usage: jobs/dress_rehearsal.sh [OUT_DIR]          (~25 min; nothing touches the robot network: all on loopback)
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
OUT=${1:-runs/dress-$(date +%Y%m%d-%H%M)}
mkdir -p "$OUT"
LOG="$OUT/results.txt"
: > "$LOG"
res() { echo "$(date +%T) $1 exit=$2" | tee -a "$LOG"; }
run() { local name=$1; shift; ./g1 sim --record "$OUT/$name" --operator dress-rehearsal "$@" > "$OUT/$name.out" 2>&1; res "$name" $?; }
WALK="wait:init,sleep:1,key:s,wait:control,sleep:2.5,key:g,wait:run,wait:arrived,sleep:2,key:t,sleep:3"

./g1 doctor > "$OUT/h0-doctor.out" 2>&1; res h0-doctor $?

# H1/H2: the robot stands on SONIC while the read-only tools run
./g1 sim --layout open-3m --record "$OUT/h1h2-stand" --operator dress-rehearsal \
  --script "wait:init,sleep:1,key:s,wait:control,sleep:40,key:t,sleep:2" > "$OUT/h1h2-stand.out" 2>&1 &
CON=$!
for i in $(seq 1 600); do grep -q "transitioning to CONTROL" "$OUT/h1h2-stand/runner.log" 2>/dev/null && break; sleep 1; done
sleep 5
./g1 check --iface lo --seconds 5 > "$OUT/h1-check.out" 2>&1; res h1-check $?
./g1 check --iface lo --seconds 5 --perception --cloud dds --odom leg --png "$OUT/h2-perception.png" > "$OUT/h2-check.out" 2>&1; res h2-check $?
./g1 calibrate --iface lo --seconds 3 --out "$OUT/h2-mount.json" --png "$OUT/h2-coverage.png" > "$OUT/h2-calibrate.out" 2>&1; res h2-calibrate $?
wait $CON; res h1h2-stand $?

# H3: runner + SONIC stand only (no student), then takeover and E-STOP into the virtual gantry
run h3-stand --layout open-3m --script "wait:init,sleep:1,key:s,wait:control,sleep:8,key:t,sleep:3,key:o,sleep:2"

# H4: a recorded empty-floor episode streamed without the planner (lock-step, standing start)
EP=runs/lockstep/course-open-3m-s0
[ -f "$EP/closed_loop.npz" ] || .venv/bin/python sim/lockstep.py --task sim/courses/open-3m/task.json --seed 0 --out "$EP" \
  --start stand --route sensor --lidar mid360 > "$OUT/h4-lockstep.out" 2>&1
run h4-replay --layout open-3m --replay "$EP" --replay-slow 2 --script "wait:init,sleep:1,key:s,wait:control,sleep:2.5,key:g,sleep:9,key:t,sleep:3"

# H5: open floor; first-day odometry (leg) and the truth reference
run h5-open2m-leg --layout open-2m --odom leg --script "$WALK"
run h5-open3m --layout open-3m --script "$WALK"

# H6: ground obstacles
for c in pillar box-low wall-detour slalom; do run "h6-$c" --layout "$c" --script "$WALK"; done

# navigation: multi-goal mission; the 0.8-1.35 m ceiling trap; operator retarget + stand + resume
run nav-mission --layout nav-room --mission course
run trap-table-edge --layout table-edge --script "$WALK"
run nav-console-keys --layout open-3m --goal 3 0 --script \
  "wait:init,sleep:1,key:s,wait:control,sleep:2.5,key:g,wait:run,sleep:1.0,goal:1.5:1.0,wait:arrived,sleep:1.5,goal:0:-2,key:g,wait:run,sleep:1.0,key:h,sleep:3,key:g,wait:arrived,sleep:1.5,key:t,sleep:3"

# Studio's real-robot Navigate workflow against the chain (LIO stand-in)
jobs/rehearse_studio_link.sh "$OUT/studio-link" > "$OUT/studio-link.out" 2>&1; res studio-link $?

.venv/bin/python - "$OUT" <<'EOF'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
rows = []
for d in sorted(p for p in out.iterdir() if (p / "report.json").exists()):
    r = json.loads((d / "report.json").read_text())
    legs = r["node"]["legs"]
    tr = r.get("truth") or {}
    lat = r["node"]["plan_latency_s"] or {}
    rows.append((d.name, " ".join(l.get("end", "?").split(" ")[0] for l in legs) or "-", lat.get("p95"),
                 tr.get("launch_goal_true_dist_m"), tr.get("max_obstacle_contact_n"), (tr.get("odometry_error_m") or {}).get("max"),
                 "; ".join(r["flags"])))
lines = ["| session | walks | plan p95 s | final true dist to launch goal m | max contact N | odom err max m | flags |",
         "|---|---|---|---|---|---|---|"] + [f"| {' | '.join(str(x) for x in row)} |" for row in rows]
(out / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
EOF
echo "dress rehearsal: $OUT/summary.md"
