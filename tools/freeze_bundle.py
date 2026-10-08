#!/usr/bin/env python3
"""Freeze the evaluated student planner + SONIC tracker into `bundle/`.

Everything the runtime loads comes from here, byte-identical to what produced the
Isaac numbers, with a sha256 manifest:

    bundle/flow/src/motion2scene_flow/**.py     frozen planner source (panel v2 packet snapshot)
    bundle/flow/artifacts/student-s2-kcsc_s0/   checkpoint + receipt
    bundle/flow/artifacts/clean-reference-v2/receipt.json   joint limits (planner.FLOW_ROOT)
    bundle/flow/artifacts/native-reference-v1-final/geometry.json   collision shapes (fk.GEOMETRY)
    bundle/newton/allterrain/...                 MID-360 / D435i mounts (sensor_map.NEWTON)
    bundle/sonic/sonic_release_g1.onnx           fused release g1 encoder + FSQ + decoder (1570 -> 29)
    bundle/config/student.json                   the deployed arm's planner + episode config
    bundle/config/sonic_model_contract.json      joint names + limits used to clip SONIC targets

`bundle/flow` doubles as the planner's FLOW_ROOT and its source root, so `fk.GEOMETRY`
(parents[3] of fk.py) and `FLOW_ROOT / clean_reference` both resolve inside the bundle;
`g1deploy.bundle` repoints the two absolute constants at import time. Large binaries are
hard-linked (read-only checkpoints on this box are already hard links).

usage: freeze_bundle.py [--out bundle] [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

FLOW = Path("/home/linjiw/flow")
SNAPSHOT = FLOW / "artifacts/closed-loop-v2-main/source/src"
STUDENT = FLOW / "artifacts/student-s2-kcsc_s0"
CLEAN_RECEIPT = FLOW / "artifacts/clean-reference-v2/receipt.json"
GEOMETRY = FLOW / "artifacts/native-reference-v1-final/geometry.json"
NEWTON = Path("/home/linjiw/newton-terrain")
NEWTON_FILES = ["allterrain/__init__.py", "allterrain/constants.py", "allterrain/perception/__init__.py",
                "allterrain/perception/raster_map.py", "allterrain/perception/sensor_model.py",
                "allterrain/perception/visibility.py"]
SONIC_EXPORT = Path("/home/linjiw/hoi-retarget/outputs/contact_bfm/sonic_bridge/release_export_v2")
TEST_V4 = FLOW / "artifacts/closed-loop-test-v4/config.json"
V2_MAIN = FLOW / "configs/closed_loop_v2_main.json"
# Joint names/limits the Isaac callback clips SONIC targets to (task["model_contract_path"]).
MODEL_CONTRACT = Path("/home/linjiw/code/lfh_runs/next_stage/native_g1/sonic_model_contract.json")

# Evaluated hashes this bundle must reproduce (receipts in ~/flow and ~/hoi-retarget).
EXPECT = {
    "flow/artifacts/student-s2-kcsc_s0/model.pt": "ad9c5b325bea1e279889da0cbd18cb5ff648ded9401329677c8b00bcd8cff796",
    "sonic/sonic_release_g1.onnx": "325031d750a5199248ad73448f7fa922767e8a78ddc870115651cd19752fb5c6",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def place(src: Path, dst: Path, link: bool = False) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if link:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def student_config() -> dict:
    """The deployed arm: test-v4 `student-guide10m8-grow06` (= panel v2 `student-privroute`), plus
    the lidar-route variant (`student-sensor`) as an alternative route source."""
    v4 = json.loads(TEST_V4.read_text())
    arm = next(a for a in v4["arms"] if a["name"] == "student-guide10m8-grow06")
    planner = dict(v4["planner_defaults"], **arm["planner"])
    planner["run"] = "flow/artifacts/student-s2-kcsc_s0"  # bundle-relative
    v2 = json.loads(V2_MAIN.read_text())
    sensor_arm = next(a for a in v2["arms"] if a["name"] == "student-sensor")
    return {
        "name": "student-guide10m8-grow06",
        "source": {"test_v4": str(TEST_V4), "v2_main": str(V2_MAIN)},
        "isaac_results": {"test-v1 held-out (privileged route)": "83/98 (F38)",
                          "dev-v1": "49/56 (F37)"},
        "planner": planner,
        "episode": {k: v4[k] for k in ("replan_ticks", "goal_lookahead_m", "goal_heading", "steer_lookahead_m")},
        "route_variants": {
            "privileged": {"route_source": "privileged", "route": {}},
            "sensor": {k: sensor_arm[k] for k in ("route_source", "route", "sensor_prescan_s")},
        },
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "bundle"))
    ap.add_argument("--force", action="store_true", help="replace an existing bundle")
    args = ap.parse_args(argv)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        if not args.force:
            print(f"{out} exists; pass --force to rebuild", file=sys.stderr)
            return 2
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    for src in sorted(SNAPSHOT.rglob("*.py")):
        place(src, out / "flow/src" / src.relative_to(SNAPSHOT))
    for src in sorted(STUDENT.iterdir()):
        place(src, out / "flow/artifacts/student-s2-kcsc_s0" / src.name, link=src.suffix == ".pt")
    place(CLEAN_RECEIPT, out / "flow/artifacts/clean-reference-v2/receipt.json")
    place(GEOMETRY, out / "flow/artifacts/native-reference-v1-final/geometry.json")
    for rel in NEWTON_FILES:
        place(NEWTON / rel, out / "newton" / rel)
    for name in ("sonic_release_g1.onnx", "export_receipt.json", "export_parity_vectors.npz"):
        place(SONIC_EXPORT / name, out / "sonic" / name, link=name.endswith(".onnx"))
    (out / "config").mkdir(exist_ok=True)
    place(MODEL_CONTRACT, out / "config/sonic_model_contract.json")
    (out / "config/student.json").write_text(json.dumps(student_config(), indent=2) + "\n")

    files = {}
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name != "MANIFEST.json":
            files[str(p.relative_to(out))] = {"sha256": sha256(p), "bytes": p.stat().st_size}
    bad = {k: (files.get(k, {}).get("sha256"), v) for k, v in EXPECT.items() if files.get(k, {}).get("sha256") != v}
    if bad:
        print(f"hash mismatch against the evaluated artifacts: {bad}", file=sys.stderr)
        return 1
    manifest = {
        "kind": "g1_student_deploy_bundle",
        "created": datetime.now(timezone.utc).isoformat(),
        "sources": {"planner_snapshot": str(SNAPSHOT), "student": str(STUDENT), "sonic_export": str(SONIC_EXPORT),
                    "newton": str(NEWTON)},
        "sonic_checkpoint_sha256": json.loads((SONIC_EXPORT / "export_receipt.json").read_text())["checkpoint_sha256"],
        "files": files,
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
    total = sum(f["bytes"] for f in files.values())
    print(f"bundle {out}: {len(files)} files, {total / 1e6:.1f} MB, evaluated hashes verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
