#!/usr/bin/env python3
"""Read-only DDS discovery on the robot network: every participant, writer and reader, with host and process.

Joins the domain as a plain participant and only reads the built-in discovery topics (DCPSParticipant,
DCPSPublication, DCPSSubscription). It creates no data writer and sends nothing to the robot.

Use it before taking control: exactly one process should write rt/lowcmd once the SONIC runner runs, and the
lidar topic the node reads (rt/utlidar/cloud_livox_mid360) should have a writer.

usage: dds_discover.py --iface <nic> [--seconds 4] [--json out.json] [--topic SUBSTR]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def props(qos):
    """Cyclone DDS participants announce __Hostname / __ProcessName / __Pid / __NetworkAddresses as properties."""
    out = {}
    try:
        from cyclonedds.qos import Policy

        for pol in qos:
            if isinstance(pol, Policy.Property):
                out[pol.key] = pol.value
            elif isinstance(pol, Policy.Userdata):
                out["userdata"] = bytes(pol.data).decode(errors="replace")
    except Exception:
        pass
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--iface", required=True)
    ap.add_argument("--domain", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--topic", default=None, help="only show endpoints whose topic contains this")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    from unitree_sdk2py.core.channel import ChannelFactory, ChannelFactoryInitialize

    ChannelFactoryInitialize(args.domain, args.iface)  # configures cyclone on this NIC (same as the node)
    from cyclonedds.builtin import BuiltinTopicDcpsParticipant, BuiltinTopicDcpsPublication, BuiltinTopicDcpsSubscription
    from cyclonedds.sub import DataReader

    dp = ChannelFactory()._ChannelFactory__participant  # the SDK's participant: a second one in-process crashes
    readers = {name: DataReader(dp, topic) for name, topic in (("participant", BuiltinTopicDcpsParticipant),
                                                                 ("publication", BuiltinTopicDcpsPublication),
                                                                 ("subscription", BuiltinTopicDcpsSubscription))}
    seen = {"participant": {}, "publication": {}, "subscription": {}}
    t0 = time.monotonic()
    while time.monotonic() - t0 < args.seconds:
        for name, rd in readers.items():
            for s in rd.take(N=500):
                key = str(getattr(s, "key", id(s)))
                seen[name][key] = s
        time.sleep(0.1)
    me = str(dp.guid)
    parts = {}
    for key, s in seen["participant"].items():
        parts[key] = {"key": key, "self": key == me, **props(s.qos)}

    def who(pkey):
        p = parts.get(str(pkey), {})
        host = p.get("__Hostname", "?")
        proc = p.get("__ProcessName", "?")
        pid = p.get("__Pid", "?")
        addr = p.get("__NetworkAddresses", "")
        return f"{host}:{proc}[{pid}] {addr}".strip(), p.get("self", False)

    rows = []
    for kind in ("publication", "subscription"):
        for s in seen[kind].values():
            topic = s.topic_name
            if args.topic and args.topic not in topic:
                continue
            w, mine = who(s.participant_key)
            rows.append({"kind": "writer" if kind == "publication" else "reader", "topic": topic,
                         "type": s.type_name, "participant": w, "self": mine})
    rows = [r for r in rows if not r["self"]]
    rows.sort(key=lambda r: (r["topic"], r["kind"], r["participant"]))
    print(f"participants (excluding this one): {len(parts) - 1}")
    for p in parts.values():
        if not p.get("self"):
            print(f"  {p.get('__Hostname', '?'):<20} {p.get('__ProcessName', '?'):<28} pid {p.get('__Pid', '?'):<8} "
                  f"{p.get('__NetworkAddresses', '')}")
    print(f"\nendpoints: {len(rows)}")
    for r in rows:
        print(f"  {r['kind']:<6} {r['topic']:<44} {r['type']:<44} {r['participant']}")
    if args.json:
        Path(args.json).write_text(json.dumps({"participants": list(parts.values()), "endpoints": rows}, indent=1,
                                              default=str) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
