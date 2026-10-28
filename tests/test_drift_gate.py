"""tools/drift_gate.py: scoring a walk against a task's gates."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import drift_gate as D  # noqa: E402


def test_crossed_through_vs_around():
    gate = [1.8, -0.35, 1.8, 0.35]
    through = np.column_stack([np.linspace(0, 3, 50), np.zeros(50)])
    around = np.column_stack([np.linspace(0, 3, 50), 1.0 * np.sin(np.linspace(0, np.pi, 50))])
    assert D.crossed(through, gate) and not D.crossed(around, gate)


def test_score_and_levels(tmp_path):
    s = tmp_path / "s"
    s.mkdir()
    (s / "attempts.jsonl").write_text(json.dumps({"n": 1, "result": "arrived", "duration_s": 6.0}) + "\n")
    np.savez(s / "sim.npz", root_pos=np.column_stack([np.linspace(0, 3, 50), np.zeros(50), np.full(50, 0.75)]),
             contact_n=np.zeros(50))
    r = D.score(s, {"gates": [[1.8, -0.35, 1.8, 0.35]]})
    assert r["pass"] and r["gates"] and r["contact_n"] == 0.0
    assert D.odom_args("exact", 0) == [] and D.odom_args("leg", 0) == ["--odom", "leg"]
    assert json.loads(D.odom_args("0.9,-4", 0)[3]) == {"scale": 0.9, "rotate_deg": -4.0}
    assert json.loads(D.odom_args("motion:training", 7)[3]) == {"motion": "training", "seed": 7}
    a = D.odom_args("fused+jump", 0)
    assert a[:2] == ["--odom", "fused"] and json.loads(a[3])["fault"]["kind"] == "jump"
