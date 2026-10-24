#!/usr/bin/env python3
"""Build the runner's pre-loaded motion: SONIC's default standing pose, held for 4 s.

The C++ runner needs at least one pre-loaded motion; after `]` it tracks that motion's frame 0
(paused) until streaming is switched on. A stand at the default pose is what the student's first
plan starts from. Written as a LUCID clip and converted with lucid-g1-deploy's own converter
(tools/convert_clip_for_deploy.py), so the CSV format and FK match what the runner was tested with.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from g1deploy import contract as C  # noqa: E402

LUCID = Path(os.environ.get("LUCID_DEPLOY_ROOT", Path.home() / "lucid-sonic/lucid-g1-deploy"))
OUT = ROOT / "sim/realtime/motions"  # the converter writes OUT/<clip name>/


def main():
    import joblib

    n, fps = 120, 30
    clip = {"stand_default": {"dof": np.tile(C.DEFAULT_MJ, (n, 1)), "root_trans_offset": np.tile([0.0, 0.0, 0.793], (n, 1)),
                              "root_rot": np.tile([0.0, 0.0, 0.0, 1.0], (n, 1)), "fps": fps}}
    with tempfile.TemporaryDirectory() as d:
        pkl = Path(d) / "stand_default.pkl"
        joblib.dump(clip, pkl)
        subprocess.run([str(LUCID / ".venv/bin/python"), str(LUCID / "tools/convert_clip_for_deploy.py"), "--clip", str(pkl),
                        "--out", str(OUT)], check=True)
    print(json.dumps({"motion": str(OUT), "files": sorted(p.name for p in (OUT / "stand_default").iterdir())}))


if __name__ == "__main__":
    main()
