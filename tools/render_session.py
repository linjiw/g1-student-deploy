#!/usr/bin/env python3
"""Render a recorded `g1 sim` session as a video: operator console (left) + the robot in MuJoCo (right).

Inputs (all written by `g1 sim --record DIR`): console_frames.jsonl (screen text + wall time), sim.npz (50 Hz qpos +
wall time + the task boxes). The robot is re-rendered offline from the logged states (no real-time cost), with
EGL. Optional captions: JSON list of {"t": s, "dur": s, "text": "..."} in console time (seconds after start).

usage: MUJOCO_GL=egl render_session.py DIR --out video.mp4 [--captions caps.json] [--from S --to S] [--speed 1]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LUCID = Path(os.environ.get("LUCID_DEPLOY_ROOT", Path.home() / "lucid-sonic/lucid-g1-deploy"))
SCENE = LUCID / "sim/models/g1/scene_43dof.xml"
W, H, CON_W = 1280, 720, 600
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
SANS_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def robot_renderer(boxes, width, height, azimuth=90.0, elevation=-12.0, distance=3.8):
    import mujoco

    from sim.scene import augment_spec

    spec = mujoco.MjSpec.from_file(str(SCENE))
    model = augment_spec(spec, boxes).compile()
    model.vis.global_.offwidth, model.vis.global_.offheight = max(width, 640), max(height, 480)
    data = mujoco.MjData(model)
    rend = mujoco.Renderer(model, height, width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

    def draw(qpos, look):
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        cam.lookat[:] = [look[0], look[1], 0.75]
        rend.update_scene(data, cam)
        return rend.render()

    return draw, data, pelvis


def text_panel(text, width, height, font, title_font):
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (width, height), (18, 20, 26))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, width, 34], fill=(32, 37, 48))
    d.text((12, 8), "operator console  (./g1 sim)", font=title_font, fill=(150, 200, 255))
    y = 44
    for i, line in enumerate(text.splitlines()):
        color = (230, 230, 230)
        if line.startswith(" ["):
            color = (150, 200, 255)
        elif "E-STOP" in line or "estop" in line:
            color = (255, 110, 100)
        elif "operator:" in line:
            color = (255, 210, 120)
        elif "node:" in line:
            color = (150, 220, 160)
        elif line.startswith(" ["):
            color = (150, 200, 255)
        d.text((10, y), line[:84], font=font, fill=color)
        y += 17
    return img


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("session")
    ap.add_argument("--out", required=True)
    ap.add_argument("--captions", default=None)
    ap.add_argument("--from", dest="t_from", type=float, default=None, help="console time (s) to start at")
    ap.add_argument("--to", dest="t_to", type=float, default=None)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--title", default=None, help="text banner on the robot view")
    ap.add_argument("--azimuth", type=float, default=90.0)
    ap.add_argument("--elevation", type=float, default=-12.0)
    ap.add_argument("--distance", type=float, default=3.8)
    args = ap.parse_args(argv)
    from PIL import Image, ImageDraw, ImageFont

    d = Path(args.session)
    sess = json.loads((d / "session.json").read_text()) if (d / "session.json").exists() else {}
    frames = [json.loads(l) for l in (d / "console_frames.jsonl").read_text().splitlines() if l.strip()]
    fw = np.array([f["wall"] for f in frames])
    z = np.load(d / "sim.npz")
    q_wall = z["wall"][::2][: len(z["qpos"])]
    qpos = z["qpos"][: len(q_wall)]
    boxes = json.loads(str(z["boxes"]))
    t0 = sess.get("t0", fw[0])
    caps = json.loads(Path(args.captions).read_text()) if args.captions else []
    start = t0 + (args.t_from if args.t_from is not None else 0.0)
    end = t0 + (args.t_to if args.t_to is not None else fw[-1] - t0)
    draw, data, pelvis = robot_renderer(boxes, W - CON_W, H, args.azimuth, args.elevation, args.distance)
    font = ImageFont.truetype(MONO, 12)
    title = ImageFont.truetype(SANS_B, 14)
    cap_font = ImageFont.truetype(SANS_B, 20)
    n = int((end - start) / args.speed * args.fps)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(args.fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                           args.out], stdin=subprocess.PIPE)
    look = None
    for k in range(n):
        tw = start + k * args.speed / args.fps
        i = int(np.clip(np.searchsorted(fw, tw) - 1, 0, len(frames) - 1))
        canvas = Image.new("RGB", (W, H))
        canvas.paste(text_panel(frames[i]["text"], CON_W, H, font, title), (0, 0))
        if len(q_wall) and tw >= q_wall[0]:
            j = int(np.clip(np.searchsorted(q_wall, tw), 0, len(qpos) - 1))
            target = qpos[j][:2]
            look = target if look is None else look + 0.15 * (target - look)  # smoothed follow
            img = Image.fromarray(draw(qpos[j], look))
        else:
            img = Image.new("RGB", (W - CON_W, H), (40, 44, 52))
            ImageDraw.Draw(img).text((40, H // 2), "simulator starting ...", font=title, fill=(200, 200, 200))
        dd = ImageDraw.Draw(img)
        if args.title:
            dd.rectangle([0, 0, W - CON_W, 30], fill=(0, 0, 0))
            dd.text((10, 6), args.title, font=title, fill=(255, 255, 255))
        tc = tw - t0
        for c in caps:
            if c["t"] <= tc < c["t"] + c["dur"]:
                lines = c["text"].split("\n")
                hgt = 14 + 26 * len(lines)
                dd.rectangle([0, H - hgt, W - CON_W, H], fill=(0, 0, 0))
                for li, line in enumerate(lines):
                    dd.text((14, H - hgt + 8 + 26 * li), line, font=cap_font, fill=(255, 235, 140))
        canvas.paste(img, (CON_W, 0))
        ff.stdin.write(canvas.tobytes())
    ff.stdin.close()
    ff.wait()
    print(f"wrote {args.out}: {n} frames, {n / args.fps:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
