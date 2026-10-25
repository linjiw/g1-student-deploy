"""Operator panel: the console's keys as a web page (phone or laptop), with an always-visible E-STOP.

`g1 robot|sim --panel PORT` serves it from the console process on 127.0.0.1:PORT only. On the deploy PC, Tailscale
runs in userspace mode and hands tailnet connections to 127.0.0.1, so the page is reachable from the operator's own
tailnet devices (http://<tailscale ip>:PORT/?t=TOKEN) and from nowhere else: not the lab Wi-Fi, not the robot LAN.
Every request needs the session's random token.

E-STOP goes straight to the runner from the request thread (Session.estop: the runner's global `o`, damping in any
input mode); every other button is queued for the console's main loop, exactly like a key press.
"""

from __future__ import annotations

import json
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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
                self._send(200, PAGE.replace("__TOKEN__", token), "text/html; charset=utf-8")
            elif path == "/state":
                self._send(200, json.dumps(sess.panel_state(), default=float))
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
            else:
                self._send(400, '{"ok": false}')
                return
            self._send(200, '{"ok": true}')

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, token


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>G1 Console</title>
<style>
:root { --bg:#f6f7f9; --fg:#14171c; --muted:#5b6370; --card:#fff; --line:#d9dde3; --ok:#1d7f43; --warn:#a86500;
        --bad:#c0262d; --btn:#e8ebf0; --accent:#1f5fbf; }
@media (prefers-color-scheme: dark) { :root { --bg:#111317; --fg:#e8eaee; --muted:#9aa2ae; --card:#1a1d22;
        --line:#2c3139; --btn:#262a31; --ok:#3fb26b; --warn:#e0a03a; --bad:#ff5a5f; --accent:#6aa0ff; } }
* { box-sizing:border-box } body { margin:0; background:var(--bg); color:var(--fg);
  font:15px/1.35 system-ui, -apple-system, Segoe UI, sans-serif; }
main { max-width:720px; margin:0 auto; padding:0 16px 32px; }
#estop { position:sticky; top:0; z-index:9; display:block; width:100%; margin:0 0 12px; padding:22px 0;
  border:0; border-radius:0 0 14px 14px; background:var(--bad); color:#fff; font-size:30px; font-weight:800;
  letter-spacing:.06em; box-shadow:0 4px 14px rgba(0,0,0,.25); }
#estop:active { filter:brightness(.85) }
.card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:12px 14px; margin:10px 0; }
.row { display:flex; flex-wrap:wrap; gap:6px 18px; } .k { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.04em }
.v { font-weight:600; font-variant-numeric:tabular-nums } .pill { display:inline-block; padding:2px 10px; border-radius:99px; font-weight:700 }
.ok { background:color-mix(in srgb, var(--ok) 18%, transparent); color:var(--ok) }
.warn { background:color-mix(in srgb, var(--warn) 18%, transparent); color:var(--warn) }
.bad { background:color-mix(in srgb, var(--bad) 18%, transparent); color:var(--bad) }
.grid { display:grid; grid-template-columns:repeat(2, minmax(0,1fr)); gap:10px; }
button.act { padding:16px 8px; border-radius:12px; border:1px solid var(--line); background:var(--btn); color:var(--fg);
  font-size:17px; font-weight:700; } button.act small { display:block; font-weight:500; font-size:12px; color:var(--muted) }
button.go { background:var(--accent); color:#fff; border-color:transparent } button.go small { color:#e6eeff }
.goal { display:flex; gap:8px; align-items:end; flex-wrap:wrap } .goal label { font-size:12px; color:var(--muted) }
.goal input { width:90px; padding:10px; font-size:16px; border-radius:8px; border:1px solid var(--line); background:var(--bg); color:var(--fg) }
pre { margin:0; white-space:pre-wrap; font:12px/1.4 ui-monospace, Menlo, monospace; color:var(--muted) }
#conn { font-size:12px; color:var(--muted); text-align:right }
</style></head><body>
<button id="estop" onclick="send({act:'estop'})">E-STOP</button>
<main>
<div id="conn">connecting...</div>
<div class="card"><div class="row">
  <div><div class="k">runner</div><div class="v" id="runner">-</div></div>
  <div><div class="k">driving</div><div class="v" id="driving">-</div></div>
  <div><div class="k">student</div><div class="v" id="phase">-</div></div>
  <div><div class="k">remote</div><div class="v" id="remote">-</div></div>
</div></div>
<div class="card"><div class="row">
  <div><div class="k">goal dist</div><div class="v" id="goal">-</div></div>
  <div><div class="k">plan latency</div><div class="v" id="lat">-</div></div>
  <div><div class="k">pelvis z</div><div class="v" id="z">-</div></div>
  <div><div class="k">tilt</div><div class="v" id="tilt">-</div></div>
  <div><div class="k">lidar age</div><div class="v" id="lidar">-</div></div>
</div><pre id="rec" style="margin-top:8px"></pre></div>
<div class="card" id="armcard" style="display:none"><button class="act go" style="width:100%" onclick="confirm('ARM: start the runner? It releases Unitree\'s controller (the gantry takes the weight) and ramps the joints to the stand pose.') && send({act:'arm'})">Arm: start runner<small>releases Unitree's controller, ramps to the stand pose</small></button></div>
<div class="grid">
  <button class="act" onclick="confirm('Start SONIC control? The robot stands on SONIC.') && send({act:'s'})">Start SONIC<small>s · stand on SONIC</small></button>
  <button class="act go" onclick="confirm('GO: hand control to the student planner?') && send({act:'g'})">GO<small>g · student walks to goal</small></button>
  <button class="act" onclick="send({act:'h'})">Stand<small>h · student settles to a stand</small></button>
  <button class="act" onclick="send({act:'t'})">Takeover<small>t · SONIC's own planner (= D-up)</small></button>
</div>
<div class="card"><button class="act" style="width:100%" onclick="confirm('Clear the map? Only while standing, with nobody and nothing close to the robot. The node maps again from here; GO waits 10 s.') && send({act:'c'})">Clear map<small>c · forget the lidar memory, map again from here (nothing close to the robot)</small></button></div>
<div class="card goal">
  <div><label>ahead (m)</label><br><input id="ga" type="number" step="0.1" value="2.0"></div>
  <div><label>left (m)</label><br><input id="gl" type="number" step="0.1" value="0.0"></div>
  <button class="act" onclick="send({act:'goal', ahead:+ga.value, left:+gl.value})">Set goal<small>relative to the robot now; then GO</small></button>
</div>
<div class="card"><div class="k">events</div><pre id="events"></pre></div>
<div class="card"><button class="act" style="width:100%" onclick="confirm('End the session? This stops the runner: robot in the gantry or sitting.') && send({act:'quit'})">End session<small>q · stops the runner, writes the report</small></button>
<pre id="dir" style="margin-top:8px"></pre></div>
</main>
<script>
const T = "__TOKEN__";
async function send(m) {
  try { const r = await fetch('/act?t=' + T, {method:'POST', body:JSON.stringify(m)});
        if (!r.ok) alert('console refused: ' + r.status); } catch (e) { alert('console unreachable: ' + e); }
}
const $ = id => document.getElementById(id);
function pill(el, text, cls) { el.innerHTML = '<span class="pill ' + cls + '">' + text + '</span>'; }
function f(v, d, u) { return (v === null || v === undefined || Number.isNaN(v)) ? '-' : (+v).toFixed(d) + (u || ''); }
async function tick() {
  try {
    const r = await fetch('/state?t=' + T, {cache:'no-store'}); const s = await r.json();
    $('conn').textContent = 'console t = ' + s.t.toFixed(1) + ' s  ·  ' + s.mode;
    pill($('runner'), s.runner, s.runner === 'E-STOP' ? 'bad' : s.runner === 'CONTROL' ? 'ok' : 'warn');
    $('driving').textContent = s.driving;
    $('armcard').style.display = s.runner === 'not started' ? 'block' : 'none';
    $('phase').textContent = s.phase || '-';
    pill($('remote'), s.remote.connected ? ('ok' + (s.remote.last ? ' · ' + s.remote.last : '')) : 'not seen',
         s.remote.connected ? 'ok' : 'bad');
    $('goal').textContent = f(s.goal_dist_m, 2, ' m'); $('lat').textContent = f(s.latency_s, 2, ' s');
    $('z').textContent = f(s.pelvis_z_m, 2, ' m'); $('tilt').textContent = f(s.tilt_rad, 2, ' rad');
    $('lidar').textContent = f(s.cloud_age_s, 2, ' s');
    $('rec').textContent = s.record; $('events').textContent = s.events.join('\n'); $('dir').textContent = s.dir;
  } catch (e) { $('conn').textContent = 'NO CONNECTION to the console (' + e + ')'; }
}
setInterval(tick, 400); tick();
</script></body></html>
"""
