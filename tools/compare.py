#!/usr/bin/env python3
"""Paired summary of MuJoCo episodes against Isaac, or between two MuJoCo conditions.

usage: compare.py runs/<a> [runs/<b>]     (one dir: MuJoCo vs Isaac; two dirs: a vs b, paired by episode)
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path


def _mcnemar_p(b, c):
    """Exact two-sided McNemar (binomial) p-value for discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * p)


def summarize(rows, a="isaac", b="mujoco", title=""):
    ok = lambda o: o == "goal_completion"  # noqa: E731
    n = len(rows)
    sa, sb = sum(ok(r[a]) for r in rows), sum(ok(r[b]) for r in rows)
    only_a = sum(ok(r[a]) and not ok(r[b]) for r in rows)
    only_b = sum(ok(r[b]) and not ok(r[a]) for r in rows)
    lines = [f"== {title}: {b} {sb}/{n} vs {a} {sa}/{n}  (discordant {a}-only {only_a} : {b}-only {only_b}, "
             f"McNemar p={_mcnemar_p(only_a, only_b):.3f})"]
    for key in (a, b):
        lines.append(f"   {key:8s} outcomes: {dict(Counter(r[key] for r in rows))}")
    fam = defaultdict(lambda: [0, 0, 0])
    for r in rows:
        f = fam[r.get("family")]
        f[0] += 1
        f[1] += ok(r[a])
        f[2] += ok(r[b])
    lines.append("   by family (n, " + a + ", " + b + "): " +
                 ", ".join(f"{k} {v[0]}/{v[1]}/{v[2]}" for k, v in sorted(fam.items(), key=lambda x: str(x[0]))))
    errs = [r for r in rows if r.get("error")]
    if errs:
        lines.append(f"   runtime errors: {len(errs)} e.g. {errs[0]['error']}")
    return "\n".join(lines)


def load(d):
    return {r["name"]: r for r in json.loads((Path(d) / "results.json").read_text())}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) == 1:
        rows = list(load(argv[0]).values())
        print(summarize(rows, title=Path(argv[0]).name))
    else:
        A, B = load(argv[0]), load(argv[1])
        common = sorted(set(A) & set(B))
        rows = [dict(name=k, family=A[k]["family"], a=A[k]["mujoco"], b=B[k]["mujoco"]) for k in common]
        print(summarize(rows, a="a", b="b", title=f"{Path(argv[1]).name} (b) vs {Path(argv[0]).name} (a)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
