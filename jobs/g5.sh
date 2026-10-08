#!/usr/bin/env bash
# Gate G5: real-time rehearsals (DDS sim + realistic MID-360 + C++ SONIC runner + node), two layouts per
# family of the held-out test-v1 panel. Sequential (fixed ports). ~50 s each.
set -u
cd "$(dirname "$0")/.."
source env.sh
NAME=${1:-g5-rt}
shift || true
mkdir -p runs/$NAME
for T in empty-100 empty-101 beam-100 beam-101 beams2-100 beams2-101 corridor_beam-100 corridor_beam-101 \
         detour-100 detour-101 pillars-100 pillars-101 slalom-100 slalom-101; do
  [ -f runs/$NAME/$T/score.json ] && continue
  rm -rf runs/$NAME/$T
  $PYTHON sim/realtime/rehearse.py --task ~/flow/artifacts/panels/test-v1/$T/task.json --out runs/$NAME/$T "$@" \
      > runs/$NAME/$T.out 2>&1
  tail -1 runs/$NAME/$T.out | cut -c1-200
done
$PYTHON - "$NAME" <<'EOF'
import json, sys, glob, numpy as np
rows = [json.load(open(f)) for f in sorted(glob.glob(f"runs/{sys.argv[1]}/*/score.json"))]
ok = sum(r["outcome"] == "goal_completion" for r in rows)
lat = [r["plan_latency_s"]["p50"] for r in rows if r.get("plan_latency_s")]
print(f"== {sys.argv[1]}: {ok}/{len(rows)} goals; outcomes", {o: sum(r['outcome'] == o for r in rows) for o in {r['outcome'] for r in rows}},
      f"plan latency p50 median {np.median(lat):.2f}s")
for r in rows:
    print(f"   {r['task_id']:28s} {r['outcome']:20s} {r.get('elapsed_s', 0):5.2f}s lat {r['plan_latency_s']['p50'] if r.get('plan_latency_s') else float('nan'):.2f}")
EOF
