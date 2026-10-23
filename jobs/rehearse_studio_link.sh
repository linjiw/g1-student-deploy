#!/usr/bin/env bash
# Rehearse Studio's REAL-ROBOT Navigate workflow against the simulated chain (docs/DEPLOY_KIT.md §Navigation):
#   g1 sim --odom lio (runner + node on LIO odometry + LIO stand-in on :5561)  <-  Studio gateway + robot link
# Studio connects, builds its map from the LIO cloud, checks a clicked goal and stages it on the node (stand-only,
# read back); the console then presses g, exactly as the operator would. Passes when the robot arrives at
# Studio's goal.
#
# usage: jobs/rehearse_studio_link.sh [OUT_DIR] [GOAL_X GOAL_Y]   (goal in the odometry frame, default 2.5 1.0)
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
OUT=${1:-runs/verify/studio-link}
GX=${2:-2.5}; GY=${3:-1.0}
STUDIO=${G1_STUDIO:-$HOME/code/g1-deploy-studio}
PORT=${STUDIO_PORT:-8791}
rm -rf "$OUT"; mkdir -p "$OUT"

# 1. the sim chain on LIO odometry; the operator script waits for Studio's goal, then presses g
./g1 sim --layout open-3m --odom lio --record "$OUT/session" --operator studio-link-rehearsal \
  --note "Studio real-robot link rehearsal, goal ($GX, $GY) odom" --step-timeout 120 \
  --script "wait:init,sleep:1,key:s,wait:control,sleep:40,key:g,wait:run,wait:arrived,sleep:2,key:t,sleep:3" \
  > "$OUT/console.out" 2>&1 &
CON=$!

# 2. Studio gateway on a spare port (its own data root, so a running Studio service is not disturbed)
DATA="$PWD/runs/studio-data"; ( cd "$STUDIO" && exec .venv/bin/python -m studio.server --port "$PORT" --data-root "$DATA" ) \
  > "$OUT/studio.out" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT

for i in $(seq 1 600); do grep -q "transitioning to CONTROL" "$OUT/session/runner.log" 2>/dev/null && break; sleep 1; done
sleep 4
.venv/bin/python - "$PORT" "$GX" "$GY" "$OUT" <<'EOF'
import json, sys, time, urllib.request
port, gx, gy, out = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
base = f"http://127.0.0.1:{port}"
def get(path):
    with urllib.request.urlopen(base + path, timeout=10) as r:
        return json.loads(r.read())
def post(path, body, token):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), method="POST", headers={
        "Content-Type": "application/json", "Origin": base, "X-Studio-Token": token})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")
log = open(f"{out}/studio_link.jsonl", "w")
def note(kind, **kw):
    print(kind, json.dumps(kw, default=str)[:400], flush=True)
    log.write(json.dumps({"t": time.time(), "kind": kind, **kw}, default=str) + "\n")
for _ in range(60):
    try:
        wb = get("/api/workbench"); break
    except Exception:
        time.sleep(1)
token = wb["csrf_token"]
note("robot_link_config", status=get("/api/nav/robot"))
code, job = post("/api/nav/connect", {}, token)
note("connect", code=code, job=job)
jid = job.get("id") or job.get("job", {}).get("id")
def latest():
    try:
        return get(f"/api/jobs?id={jid}&light=1").get("latest") or {}
    except Exception as exc:
        note("latest_error", error=str(exc))
        return {}
nav = {}
for _ in range(40):                      # wait for fresh streams, a map and a ready robot
    time.sleep(1)
    nav = (latest().get("navigation") or {})
    if nav.get("can", {}).get("goal"):
        break
note("before_goal", phase=nav.get("phase"), can=nav.get("can"), streams=[(s["id"], s["status"]) for s in nav.get("streams", [])],
     ready=[(r["id"], r["status"]) for r in nav.get("ready", [])], reasons=nav.get("can_reasons"))
code, res = post("/api/nav/command", {"id": jid, "cmd": "goal", "xy": [gx, gy]}, token)
note("goal", code=code, result=res)
for _ in range(15):
    time.sleep(1)
    nav = latest().get("navigation") or {}
    g = nav.get("goal") or {}
    if g.get("verification") in ("verified", "unverified"):
        break
note("readback", goal=nav.get("goal"), warning=nav.get("goal_warning"), ready=[(r["id"], r["status"]) for r in nav.get("ready", [])])
for _ in range(60):                      # the console presses g; wait for the walk to end
    time.sleep(1)
    nav = latest().get("navigation") or {}
    if nav.get("phase") in ("ARRIVED", "STOPPED") and nav.get("walks"):
        break
note("after_walk", phase=nav.get("phase"), goal_dist=nav.get("goal_dist_m"), walks=nav.get("walks"), legs=nav.get("legs"))
code, res = post("/api/nav/command", {"id": jid, "cmd": "end"}, token)
note("end", code=code, result=res)
EOF
wait $CON
echo "console exit $?"
.venv/bin/python - "$OUT" "$GX" "$GY" <<'EOF'
import json, sys, numpy as np
out, gx, gy = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
z = np.load(f"{out}/session/sim.npz")
final = z["root_pos"][-1, :2]
d = float(np.hypot(final[0] - gx, final[1] - gy))
print(f"robot final xy {final.round(2).tolist()}, distance to Studio's goal {d:.2f} m ->", "PASS" if d < 0.6 else "FAIL")
EOF
