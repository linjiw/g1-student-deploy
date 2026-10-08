#!/usr/bin/env python3
"""Build the narrated tutorial video: how to use `g1` to rehearse in simulation and deploy on a real G1.

Segments are slides (PIL, 1280x720) and recorded console sessions (tools/render_session.py, captioned from the
console's own event timeline). Narration: Kokoro-82M offline TTS (`--tts-python`, a venv with kokoro, soundfile,
misaki[en], en_core_web_sm); without it the video is silent with captions. Output: one mp4 + per-clip mp4s.

usage: make_tutorial.py --sessions runs/video --out runs/video/g1_deploy_tutorial.mp4 [--tts-python PY]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
W, H = 1280, 720
SANS, SANS_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

SLIDES = [
    dict(key="title", title="Deploying the lidar student on a Unitree G1",
         body=["The g1 tool: one command for simulation and the real robot", "",
               "student planner (flow matching, MID-360 lidar) + GR00T SONIC tracker",
               "real deploy chain: DDS, C++ SONIC runner (TensorRT), ZMQ stream"],
         say="This video shows how to run the lidar student on a Unitree G1 with the g1 tool. The same command drives "
             "a MuJoCo rehearsal and the real robot, through the real deployment chain."),
    dict(key="arch", title="What runs",
         body=["robot (or MuJoCo G1)   <->  rt/lowcmd, rt/lowstate, rt/utlidar/cloud_livox_mid360  (DDS)",
               "SONIC C++ runner      : tracks references at 50 Hz, keyboard + wireless remote",
               "student node          : lidar memory -> planner (0.3 s) -> 1 s plans -> ZMQ stream",
               "operator console ./g1 : status + single keys; every session is logged"],
         code=True,
         say="Three processes run. The SONIC runner tracks references at fifty hertz and listens to the keyboard and "
             "the wireless remote. The student node reads the robot's own lidar topic, builds a height and ceiling "
             "memory, plans one second ahead, and streams the plan to the runner. The console shows everything and "
             "takes single keys."),
    dict(key="install", title="Install and check (deploy PC)",
         body=["./g1 test          # 22 tests: bundle hashes, SONIC parity, stream, perception, safety",
               "./g1 bench         # planner latency here: p95 must be <= 0.32 s",
               "./g1 sim --layout beams2-100      # rehearse with the keyboard"],
         code=True,
         say="On the deploy PC, run g1 test, then g1 bench to check that the planner is fast enough on this machine. "
             "Then rehearse in simulation."),
    dict(key="keys", title="Keys",
         body=["s   start SONIC: the robot stands",
               "g   GO: the student pre-scans, plans and walks to the goal",
               "h   student stand        n   new goal (ahead left, metres)",
               "t   takeover: SONIC's own planner, standing   (remote D-up)",
               "o   E-STOP: damping, the robot drops          (remote Select)",
               "q   quit"],
         code=True,
         say="The keys: s starts SONIC, g hands control to the student, h makes the student stand, n sets a new goal, "
             "t takes over with SONIC's own planner, and o is the emergency stop."),
    dict(key="clipA", clip="A_quickstart", title="Simulation: two beams",
         say="Here is a full run. Press s and SONIC stands the robot. Press g: the student scans with its lidar, plans, "
             "and walks under two beams to the goal, then settles to a stand. Press t and SONIC's own planner takes over."),
    dict(key="safety", title="Overrides",
         body=["t / remote D-up : SONIC planner stands the robot   (not within ~1 m of a beam)",
               "o / remote Select: damping                        (gantry or spotter only)",
               "automatic        : tilt > 0.75 rad -> E-STOP;  lidar lost 1 s -> stand",
               "                   stand is deferred under ceilings"],
         code=True,
         say="Two overrides are always available: takeover, which stands the robot with SONIC's planner, and the "
             "emergency stop. The node also stops by itself when the pelvis tilts past zero point seven five radians, "
             "or stands when the lidar goes quiet."),
    dict(key="clipB", clip="B_safety", title="Takeover, new goal, E-STOP",
         say="On open floor: the student walks, the operator takes over and the robot stands. A new goal, go again, "
             "the student walks to it. Finally the emergency stop: damping, and the robot drops. On the real robot this "
             "is why it hangs in a gantry."),
    dict(key="clipC", clip="C_detour", title="Lidar-only detour",
         say="The student uses only its lidar memory, here routing around a wall."),
    dict(key="clipD", clip="D_beams2_101", title="Automatic stop: tilt guard",
         say="A harder layout. Under the second beam the student ducks so deep that the pelvis tilts past the guard. "
             "The node asks for a stop and the console sends damping by itself, without the operator. On the robot, "
             "the gantry catches it. This is why overhead obstacles come last on the ladder."),
    dict(key="robot", title="On the robot",
         body=["once:  network 192.168.123.x wired;  Unitree lidar service on;  --body dex3|rubber",
               "       ./g1 calibrate --iface eth0 --out mount.json --png coverage.png",
               "check: ./g1 check --iface eth0 --perception",
               "run:   ./g1 robot --iface eth0 --goal 3 0 --odom leg --mount-calib mount.json",
               "       yes -> s (or remote Start) -> slacken gantry -> g -> ... t -> o -> q"],
         code=True,
         say="On the robot: connect the PC by cable, keep Unitree's lidar service on, calibrate the head lidar on open "
             "floor, and run the read-only check. Then g1 robot with the network interface and a goal a few metres "
             "ahead. The keys are the same; the remote's Start, D-up and Select work too."),
    dict(key="ladder", title="Bring-up ladder",
         body=["H0 desk: test, bench, rehearsal       H1 network check      H2 perception standing",
               "H3 SONIC stand in the gantry          H4 stream replay (no planner)",
               "H5 student on open floor, 2-3 m       H6 ground obstacles",
               "H7 overhead obstacles: only after the deep-duck envelope is verified",
               "",
               "docs: OPERATOR_GUIDE.md  HARDWARE.md  G1_PERCEPTION.md  READINESS.md"],
         code=True,
         say="Climb the bring-up ladder one step at a time and stop at the first surprise. No overhead obstacles until "
             "the deep duck is verified on the robot. The guides in the docs folder have every command."),
]


def slide_image(s):
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (W, H), (16, 20, 28))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 8], fill=(90, 160, 255))
    d.text((70, 70), s["title"], font=ImageFont.truetype(SANS_B, 44), fill=(255, 255, 255))
    f = ImageFont.truetype(MONO if s.get("code") else SANS, 24 if s.get("code") else 30)
    y = 180
    for line in s.get("body", []):
        d.text((80, y), line, font=f, fill=(215, 225, 240))
        y += 46 if s.get("code") else 52
    d.text((70, H - 50), "g1-student-deploy  ·  G1 lidar student + GR00T SONIC", font=ImageFont.truetype(SANS, 18),
           fill=(110, 125, 150))
    return img


def session_events(sdir):
    """(console time, text) of every event line ever shown on the console screen."""
    seen, out = set(), []
    for line in (Path(sdir) / "console_frames.jsonl").read_text().splitlines():
        for row in json.loads(line)["text"].splitlines():
            m = re.match(r"^\s+(\d+\.\d)\s{2}(.*)$", row)
            if m and (m.group(1), m.group(2)) not in seen:
                seen.add((m.group(1), m.group(2)))
                out.append((float(m.group(1)), m.group(2)))
    return sorted(out)


CAPTION_RULES = [
    ("operator: start SONIC", "s: SONIC takes control, the robot stands"),
    ("operator: remote Start", "remote Start: SONIC takes control"),
    ("operator: GO", "g: the student pre-scans the lidar and plans"),
    ("node: run", "the student walks: a new 1 s plan every ~0.3 s"),
    ("node: arrived", "goal reached: the student settles to a stand"),
    ("operator: TAKEOVER", "t: takeover, SONIC's own planner stands the robot"),
    ("operator: new goal", "n: a new goal relative to the robot"),
    ("operator: E-STOP", "o: E-STOP, damping (the robot drops: gantry!)"),
    ("AUTO E-STOP", "the node's tilt guard stopped the robot automatically"),
    ("node: stand_deferred", "stand deferred: a ceiling is overhead"),
]


def captions_for(sdir, t_from):
    caps = []
    for t, text in session_events(sdir):
        for key, cap in CAPTION_RULES:
            if text.startswith(key) and t >= t_from:
                caps.append({"t": t, "dur": 2.6, "text": cap})
    return caps


def tts(texts, py, workdir):
    """[text] -> [wav path or None]."""
    if not py:
        return [None] * len(texts)
    spec = Path(workdir) / "tts.json"
    spec.write_text(json.dumps(texts))
    code = f"""
import json, numpy as np, soundfile as sf
from kokoro import KPipeline
p = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')
texts = json.load(open({str(spec)!r}))
for i, t in enumerate(texts):
    a = np.concatenate([x.numpy() if hasattr(x, 'numpy') else x for _, _, x in p(t, voice='af_heart', speed=1.1)])
    sf.write({str(workdir)!r} + f'/tts_{{i:02d}}.wav', a.astype('float32'), 24000)
print('ok')
"""
    subprocess.run([py, "-c", code], check=True, env=dict(os.environ, HF_HUB_OFFLINE="1"))
    return [str(Path(workdir) / f"tts_{i:02d}.wav") for i in range(len(texts))]


def dur(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                         capture_output=True, text=True).stdout
    return float(out.strip())


def mux(video, audio, out, min_dur=None):
    """video (+ optional narration, padded with silence to the video) -> out mp4 with AAC audio."""
    args = ["ffmpeg", "-y", "-loglevel", "error", "-i", video]
    if audio:
        args += ["-i", audio, "-filter_complex", "[1:a]apad[a]", "-map", "0:v", "-map", "[a]", "-shortest"]
    else:
        args += ["-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono", "-map", "0:v", "-map", "1:a", "-shortest"]
    subprocess.run(args + ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-ar", "48000", "-ac", "2", out],
                   check=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sessions", default=str(ROOT / "runs/video"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--tts-python", default=None)
    args = ap.parse_args(argv)
    sessions = Path(args.sessions)
    work = Path(tempfile.mkdtemp(prefix="g1tut_", dir=sessions))
    slides = [s for s in SLIDES if "clip" not in s or (sessions / s["clip"] / "sim.npz").exists()]
    wavs = tts([s["say"] for s in slides], args.tts_python, work)
    parts = []
    for k, (s, wav) in enumerate(zip(slides, wavs)):
        seg = work / f"seg_{k:02d}.mp4"
        if "clip" in s:
            sdir = sessions / s["clip"]
            ev = session_events(sdir)
            t_start = next((t for t, e in ev if e.startswith(("operator: start SONIC", "operator: remote Start"))), 0) - 1.0
            t_end = ev[-1][0] + 1.0
            caps = work / f"caps_{k:02d}.json"
            caps.write_text(json.dumps(captions_for(sdir, t_start)))
            raw = work / f"clip_{k:02d}.mp4"
            subprocess.run([sys.executable, str(ROOT / "tools/render_session.py"), str(sdir), "--out", str(raw),
                            "--captions", str(caps), "--from", str(t_start), "--to", str(t_end),
                            "--title", s["title"], "--azimuth", "60", "--distance", "4.0"], check=True,
                           env=dict(os.environ, MUJOCO_GL="egl"))
            mux(str(raw), wav, str(seg))
            (sessions / f"{s['clip']}.mp4").write_bytes(seg.read_bytes())
        else:
            png = work / f"slide_{k:02d}.png"
            slide_image(s).save(png)
            length = (dur(wav) + 0.8) if wav else 6.0
            still = work / f"still_{k:02d}.mp4"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", str(png), "-t", f"{length:.2f}",
                            "-r", "25", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(still)], check=True)
            mux(str(still), wav, str(seg))
        parts.append(seg)
        print(f"segment {k}: {s['key']} {dur(str(seg)):.1f} s", flush=True)
    lst = work / "list.txt"
    lst.write_text("".join(f"file '{p.resolve()}'\n" for p in parts))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
                    args.out], check=True)
    print(f"wrote {args.out}: {dur(args.out):.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
