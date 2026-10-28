"""Operator panel: the console's keys as a web page (phone or laptop), with an always-visible E-STOP, and the robot-day
workspace around them: the ladder's next step, live checks, the planner's map with tap-to-goal, gauges and the attempt
log with the operator's tags (g1deploy/workspace.py, g1deploy/attempts.py).

`g1 robot|sim --panel PORT` serves it from the console process on 127.0.0.1:PORT only. On the deploy PC, Tailscale
runs in userspace mode and hands tailnet connections to 127.0.0.1, so the page is reachable from the operator's own
tailnet devices (http://<tailscale ip>:PORT/?t=TOKEN) and from nowhere else: not the lab Wi-Fi, not the robot LAN.
Every request needs the token (the console passes the deploy PC's one panel token, g1deploy/service_token.py).

POST /act {act, id, source, operator, expect?, ...args} (CONTRACT §4.1; the panel page, the lab hub and Studio's gateway
through the hub): E-STOP goes straight to the runner from the request thread (Session.estop: the runner's global `o`,
damping in any input mode) and answers {ok, key_sent, runner_alive, runner, confirmed, t_mono}. Every other act is
queued for the console's main loop, exactly like a key press, and the reply waits up to 1 s for its result:
{ok, id, result: done | refused | queued, reason, session}. An invalid act (unknown, no id, bad arguments) is answered
400. Every act (any source) is findable by its id from the moment the console receives it, before any reply:
GET /act_result?t=&id= -> {known, id, act, result: done | refused | queued | pending, reason, t_mono, t_done, source,
operator, session} (the last 500 acts of the session; console_events.jsonl keeps them all, and an act answered "queued"
has its line there before the reply leaves). It never acts: a relay whose reply was lost asks it instead of retrying.
Every response is strict JSON (no NaN).
"""

from __future__ import annotations

import json
import os
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from g1deploy.workspace import dumps


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
                self._send(200, dumps(sess.panel_state()))
            elif path == "/map":
                self._send(200, dumps(sess.node_map()))
            elif path == "/act_result":  # what became of an act whose reply was lost (CONTRACT v3 D); never acts
                act_id = parse_qs(urlparse(self.path).query).get("id", [""])[0]
                if not act_id:
                    self._send(400, dumps({"ok": False, "known": False, "reason": "id is required"}))
                else:
                    self._send(200, dumps(sess.act_result(act_id)))
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
            if not isinstance(msg, dict):
                self._send(400, '{"ok": false, "reason": "a JSON object"}')
                return
            code, reply = sess.submit(msg)
            self._send(code, dumps(reply))

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, token


PAGE = (Path(__file__).with_name("panel.html")).read_text()  # the page; __TOKEN__ / __HUB_PORT__ filled per request
