"""Locate, verify and import the frozen planner bundle (see tools/freeze_bundle.py).

`load_flow()` puts the frozen `motion2scene_flow` on sys.path and repoints the two
absolute paths the vendored source hard-codes (`planner.FLOW_ROOT`,
`sensor_map.NEWTON`) into the bundle, so the source itself stays byte-identical
to what was evaluated in Isaac.

Deployment route/selection corrections are installed in memory by route_compat
and planner_compat. student_config returns effective deployment defaults;
the frozen files and their historical manifest remain unchanged.
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


def needs_rules(cfg: dict) -> bool:
    """A bundle whose student config is not the deployed format (no `route_variants`): its per-replan rules come
    from a deploy rules plugin (`load_rules`), never guessed here."""
    return "route_variants" not in cfg


def student_config(root: Path = ROOT) -> dict:
    cfg = json.loads((root / "config/student.json").read_text())
    for k, v in cfg["planner"].items():  # artifact paths are bundle-relative (the model run and any others)
        if isinstance(v, str) and v.startswith("flow/artifacts/") and (root / v).exists():
            cfg["planner"][k] = str(root / v)
    if needs_rules(cfg):
        return cfg
    # A partially observed wall is a surface, not a complete obstacle extent.
    # Reuse SensorRoute's masked uncertainty growth so short unobserved gaps
    # beside known obstacles are not immediately treated as reliable corridors.
    # This changes only the routing grid; raw support/ceiling/observed masks stay
    # untouched. Return the effective value so node and rehearsal record it.
    cfg["route_variants"]["sensor"].setdefault("route", {}).setdefault("grow_unknown_m", 0.1)
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
    from g1deploy.route_compat import install as install_route_compat
    from g1deploy.planner_compat import install as install_planner_compat

    install_route_compat(route)
    install_planner_compat(planner)

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


def load_rules(cfg: dict, lookahead_m: float | None = None):
    """The bundle's deploy rules plugin, or None for the deployed format.

    A plugin is a Python file (`G1_DEPLOY_RULES`, set by `g1 sim|robot --rules PATH`) with `load(cfg, lookahead_m)`
    returning an object with: `make_route(flow, xy, goal, memory)` (the node's route), `extend_view(view, memory)`
    (extra memory layers its planner reads), `planner_class(base)` (a subclass of the fast planner), `prepare(planner)`
    and `plan_job(planner, flow, job, episode, guards)` (one planner job -> the node's result dict, with
    `rules_state` carried to the next job and `next_due`, ticks to the next replan). Plugins of unpublished bundles
    live outside this public repository. A bundle that needs rules never runs without them."""
    path = os.environ.get("G1_DEPLOY_RULES")
    if not path:
        if needs_rules(cfg):
            raise BundleError(f"the bundle at {ROOT} needs its deploy rules plugin: set G1_DEPLOY_RULES (g1 --rules)")
        return None
    import importlib.util

    spec = importlib.util.spec_from_file_location("g1_deploy_rules", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("g1_deploy_rules", module)
    spec.loader.exec_module(module)
    return module.load(cfg, lookahead_m)

