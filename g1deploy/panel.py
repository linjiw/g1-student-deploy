"""Operator panel: the console's keys as a web page (phone or laptop), with an always-visible E-STOP, and the robot-day
workspace around them: the ladder's next step, live checks, the planner's map with tap-to-goal, gauges and the attempt
log with the operator's tags (g1deploy/workspace.py, g1deploy/attempts.py).

`g1 robot|sim --panel PORT` serves it from the console process on 127.0.0.1:PORT only. On the deploy PC, Tailscale
runs in userspace mode and hands tailnet connections to 127.0.0.1, so the page is reachable from the operator's own
tailnet devices (http://<tailscale ip>:PORT/?t=TOKEN) and from nowhere else: not the lab Wi-Fi, not the robot LAN.
Every request needs the session's random token.

E-STOP goes straight to the runner from the request thread (Session.estop: the runner's global `o`, damping in any
input mode); every other button is queued for the console's main loop, exactly like a key press.
"""

from __future__ import annotations

import json
import os
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

KEYS = {"s", "g", "h", "t", "c"}


def start_panel(sess, port, host="127.0.0.1", token=None):
    """Serve the panel for `sess` (console.Session) in a daemon thread. Returns (server, token)."""
    token = token or secrets.token_urlsafe(9)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # the console screen is the log
            pass

        def _auth(self):
            q = parse_qs(urlparse(self.path).query)
            if q.get("t", [""])[0] != token:
                self.send_response(403)
                self.end_headers()
                return False
            return True

        def _send(self, code, body, ctype="application/json"):
            data = body.encode() if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if not self._auth():
                return
            path = urlparse(self.path).path
            if path == "/":
                page = PAGE.replace("__TOKEN__", token).replace("__HUB_PORT__", os.environ.get("G1_HUB_PORT", ""))
                self._send(200, page, "text/html; charset=utf-8")
            elif path == "/state":
                self._send(200, json.dumps(sess.panel_state(), default=float))
            elif path == "/map":
                self._send(200, json.dumps(sess.node_map(), default=float))
            else:
                self._send(404, "{}")

        def do_POST(self):
            if not self._auth():
                return
            try:
                n = int(self.headers.get("Content-Length") or 0)
                msg = json.loads(self.rfile.read(n) or b"{}")
            except (ValueError, json.JSONDecodeError):
                self._send(400, '{"ok": false}')
                return
            act = msg.get("act")
            if act == "estop":
                sess.estop("panel E-STOP")  # now, from this thread: never behind the main loop
            elif act in KEYS:
                sess.inbox.put(("key", act, "panel"))
            elif act == "goal":
                sess.inbox.put(("goal", (float(msg["ahead"]), float(msg["left"])), "panel"))
            elif act == "quit":
                sess.inbox.put(("quit", None, "panel"))
            elif act == "arm":
                sess.inbox.put(("arm", None, "panel"))
            elif act == "tag":
                try:
                    ok = sess.attempts.tag(int(msg.get("n", 0)), msg.get("tag"), msg.get("note"))
                except (TypeError, ValueError):
                    ok = False
                self._send(200 if ok else 400, json.dumps({"ok": ok}))
                return
            else:
                self._send(400, '{"ok": false}')
                return
            self._send(200, '{"ok": true}')

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, token


PAGE = (Path(__file__).with_name("panel.html")).read_text()  # the page; __TOKEN__ / __HUB_PORT__ filled per request
