"""Locate, verify and import the frozen planner bundle (see tools/freeze_bundle.py).

`load_flow()` puts the frozen `motion2scene_flow` on sys.path and repoints the two
absolute paths the vendored source hard-codes (`planner.FLOW_ROOT`,
`sensor_map.NEWTON`) into the bundle, so the source itself stays byte-identical
to what was evaluated in Isaac.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("G1_STUDENT_BUNDLE", Path(__file__).resolve().parents[1] / "bundle"))


class BundleError(RuntimeError):
    pass


def manifest(root: Path = ROOT) -> dict:
    path = root / "MANIFEST.json"
    if not path.is_file():
        raise BundleError(f"no bundle at {root}: run tools/freeze_bundle.py")
    return json.loads(path.read_text())


def verify(root: Path = ROOT, quick: bool = True) -> list[str]:
    """Problems found (empty = OK). quick: sizes for every file, hashes for small files and the
    two weights files; full: every hash."""
    problems = []
    for rel, meta in manifest(root)["files"].items():
        p = root / rel
        if not p.is_file():
            problems.append(f"missing {rel}")
            continue
        if p.stat().st_size != meta["bytes"]:
            problems.append(f"size {rel}")
            continue
        if not quick or p.stat().st_size < 1 << 20 or p.suffix in (".pt", ".onnx"):
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            if h != meta["sha256"]:
                problems.append(f"sha256 {rel}")
    return problems


def student_config(root: Path = ROOT) -> dict:
    cfg = json.loads((root / "config/student.json").read_text())
    cfg["planner"]["run"] = str(root / cfg["planner"]["run"])
    return cfg


def sonic_onnx(root: Path = ROOT) -> Path:
    return root / "sonic/sonic_release_g1.onnx"


_LOADED = None


def load_flow(root: Path = ROOT):
    """Import the frozen planner modules from the bundle. Returns a namespace of modules."""
    global _LOADED
    if _LOADED is not None:
        return _LOADED
    src = str(root / "flow/src")
    existing = sys.modules.get("motion2scene_flow")
    if existing is not None and not str(getattr(existing, "__file__", "")).startswith(src):
        raise BundleError(f"motion2scene_flow already imported from {existing.__file__}, not the bundle")
    if src not in sys.path:
        sys.path.insert(0, src)
    sys.dont_write_bytecode = True  # never write into the frozen tree
    from motion2scene_flow.closed_loop import callback, feed, planner, route, sensor_map
    from motion2scene_flow.wholebody import fk

    planner.FLOW_ROOT = root / "flow"
    sensor_map.NEWTON = str(root / "newton")
    expected_geometry = root / "flow/artifacts/native-reference-v1-final/geometry.json"
    if Path(fk.GEOMETRY).resolve() != expected_geometry.resolve():
        raise BundleError(f"fk.GEOMETRY resolves to {fk.GEOMETRY}, expected {expected_geometry}")

    class Flow:
        pass

    ns = Flow()
    ns.planner, ns.feed, ns.route, ns.sensor_map, ns.callback, ns.fk = planner, feed, route, sensor_map, callback, fk
    ns.root = root
    _LOADED = ns
    return ns
