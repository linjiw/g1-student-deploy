"""DDS settings for the deploy stack's Python participants (node, recorder, check, calibrate, site check).

A MID-360 frame is 0.3-0.9 MB on the wire: one burst of UDP fragments. Linux's default socket receive buffer is
208 KB (net.core.rmem_max = rmem_default = 212992), so whole bursts overflow it: on the deploy PC the kernel counted
millions of "receive buffer errors", reliable readers got frames late (retransmissions) and best-effort readers lost
them (the recorder kept 1 frame of a walled room; a node-style reader got 0 frames in one 6 s test).

`tune_sdk()` asks Cyclone DDS for a large receive buffer in the configuration unitree_sdk2py builds for
ChannelFactoryInitialize. Cyclone takes what the kernel allows, so it also helps nothing until the cap is raised once
per machine (sudo):

    printf 'net.core.rmem_max=33554432\\nnet.core.rmem_default=8388608\\n' | sudo tee /etc/sysctl.d/60-g1-dds.conf
    sudo sysctl --system

`g1 doctor` checks both.
"""

from __future__ import annotations

RECEIVE_BUFFER = "16MB"
INTERNAL = (f'<Internal><SocketReceiveBufferSize min="default" max="{RECEIVE_BUFFER}"/></Internal>')
_done = False


def tune_sdk():
    """Insert the receive-buffer request into unitree_sdk2py's Cyclone configs (idempotent, before any participant)."""
    global _done
    if _done:
        return
    try:
        import unitree_sdk2py.core.channel as ch
        import unitree_sdk2py.core.channel_config as cfg
    except ImportError:  # the SDK is not on the path (unit tests of pure-numpy parts)
        return
    for mod in (ch, cfg):
        for name in ("ChannelConfigHasInterface", "ChannelConfigAutoDetermine"):
            xml = getattr(mod, name, None)
            if isinstance(xml, str) and "SocketReceiveBufferSize" not in xml:
                setattr(mod, name, xml.replace("</General>", "</General>\n" + INTERNAL, 1))
    _done = True


def rmem_max():
    try:
        return int(open("/proc/sys/net/core/rmem_max").read())
    except OSError:
        return None
