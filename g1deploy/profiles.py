"""Deploy profiles: one file names everything a run of the student depends on besides the robot itself.

A profile (`profiles/<name>.json`, or `private/profiles/<name>.json` for bundles kept out of this repository, or any
path) sets the console options below. `./g1 sim|robot --profile NAME` reads it; an option given on the command line
wins over the profile, and the profile wins over the robot profile (`robots/<name>.json`, which keeps the robot's own
facts: interface, lidar, body, mount calibration, odometry).

Every session records the profile it ran (name, file, its SHA-256, the options the command line overrode) and a
**deploy identity**: one hash over the effective settings, the bundle's MANIFEST and the rules plugin file. Two
sessions with the same identity ran the same student the same way, whatever flags produced it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEARCH = (ROOT / "profiles", ROOT / "private" / "profiles")

# console option (argparse dest) -> (type, allowed values or None). The defaults are the console's own.
KEYS = {
    "bundle": (str, None),
    "rules": (str, None),
    "guards": (str, ("off", "deploy", "walls")),
    "goal_lookahead": (float, None),
    "rear_blind_deg": (float, None),
    "rear_blind_range_m": (float, None),
    "debug_plans": (bool, None),
    "planner_impl": (str, ("fused", "graph", "eager")),
    "planner_precision": (str, ("fp32", "tf32", "bf16")),
    "candidates": (int, None),
    "ode_steps": (int, None),
    "self_margin": (float, None),
}
DEFAULTS = {"guards": "off", "planner_impl": "fused", "planner_precision": "fp32", "debug_plans": False}
INFO = ("name", "description", "student")  # informational: not settings, not in the identity


class ProfileError(ValueError):
    pass


def find(name_or_path) -> Path:
    """A profile file by name (profiles/, then private/profiles/) or by path."""
    p = Path(name_or_path)
    if p.suffix == ".json" and p.exists():
        return p.resolve()
    for d in SEARCH:
        cand = d / f"{name_or_path}.json"
        if cand.exists():
            return cand.resolve()
    raise ProfileError(f"no deploy profile {name_or_path!r} (looked in {', '.join(str(d) for d in SEARCH)})")


def load(name_or_path) -> tuple[dict, Path]:
    """The profile's settings, checked; paths made absolute (relative ones are relative to this repository)."""
    path = find(name_or_path)
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise ProfileError(f"{path}: not valid JSON ({e})") from e
    unknown = sorted(set(raw) - set(KEYS) - set(INFO))
    if unknown:
        raise ProfileError(f"{path}: unknown keys {unknown} (known: {sorted(KEYS)})")
    out = {}
    for key, value in raw.items():
        if key in INFO:
            continue
        kind, allowed = KEYS[key]
        if value is None:
            out[key] = None
            continue
        if kind is float and isinstance(value, int) and not isinstance(value, bool):
            value = float(value)
        if not isinstance(value, kind) or (kind is int and isinstance(value, bool)):
            raise ProfileError(f"{path}: {key} must be {kind.__name__}, got {value!r}")
        if allowed and value not in allowed:
            raise ProfileError(f"{path}: {key} must be one of {allowed}, got {value!r}")
        if key in ("bundle", "rules") and not Path(value).is_absolute():
            value = str(ROOT / value)
        out[key] = value
    return out, path


def apply(args, name_or_path):
    """Fill every option the command line left unset (None) from the profile. Records `args.profile` (what ran)."""
    if not name_or_path:
        args.profile = None
        return args
    prof, path = load(name_or_path)
    overridden = []
    for key, value in prof.items():
        current = getattr(args, key, None)
        if current is None:
            setattr(args, key, value)
        elif current != value:
            overridden.append(key)
    raw = path.read_bytes()
    args.profile = {"name": json.loads(raw).get("name") or path.stem, "path": str(path),
                    "sha256": hashlib.sha256(raw).hexdigest(), "overridden": overridden}
    return args


def fill_defaults(args):
    """The console's own defaults for what neither the command line nor a profile set."""
    for key, value in DEFAULTS.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    return args


def _file_sha(path):
    p = Path(path) if path else None
    return hashlib.sha256(p.read_bytes()).hexdigest() if p and p.is_file() else None


def identity(args) -> dict:
    """The deploy identity of a run: its effective settings, the bundle MANIFEST's hash and the rules file's hash,
    hashed together. `bundle` None = the default bundle/ directory."""
    bundle = Path(getattr(args, "bundle", None) or ROOT / "bundle").resolve()
    rules = getattr(args, "rules", None)
    settings = {k: getattr(args, k, None) for k in sorted(KEYS) if k not in ("bundle", "rules", "debug_plans")}
    if not settings.get("rear_blind_deg"):  # a blind of 0 deg is no blind, whatever its range
        settings["rear_blind_deg"], settings["rear_blind_range_m"] = 0.0, None
    parts = {"settings": settings, "bundle_manifest_sha256": _file_sha(bundle / "MANIFEST.json"),
             "rules_sha256": _file_sha(rules)}
    digest = hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()
    return dict(parts, sha256=digest, short=digest[:12])
