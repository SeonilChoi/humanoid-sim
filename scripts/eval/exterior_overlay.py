"""Overlay the leg-to-shell clearance on a cutaway exterior play video.

play.py --exterior --exterior_cut writes videos/play/exterior_frames.npz next to the clip (per policy step = video
frame: env 0 leg link poses in the pelvis frame). This computes the clearance per frame
(humanoid_sim.utils.exterior.ShellClearance) and draws: the closest link and its clearance, left / right minimum,
a red border + CONTACT when any link is below 0, and a clearance timeline at the bottom.

  python scripts/eval/exterior_overlay.py <clip.mp4> <exterior_frames.npz> <out.mp4>
"""

import argparse

import cv2
import imageio_ffmpeg
import numpy as np

from humanoid_sim.utils.exterior import ShellClearance

WARN_MM = 10.0


def _text(img, s, org, scale, color, thick=2):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick + 6, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("frames")
    ap.add_argument("out")
    ap.add_argument("--frame_shift", type=int, default=1, help="frame i is rendered with the pose of row i - shift")
    args = ap.parse_args()

    d = np.load(args.frames)
    sc = ShellClearance()
    clear, _ = sc(d["R_rel"], d["t_rel"])  # (T, L) m
    mm = clear * 1e3
    links = sc.links
    side_min = {s: mm[:, [j for j, n in enumerate(links) if n.startswith(s)]].min(1) for s in ("left", "right")}
    worst = mm.min(1)
    T = len(worst)
    print(f"{T} rows: min clearance {worst.min():.1f} mm, frames < 0: {np.mean(worst < 0):.2%}, "
          f"< {WARN_MM:.0f} mm: {np.mean(worst < WARN_MM):.2%}")
    for j, n in enumerate(links):
        print(f"  {n:20s} min {mm[:, j].min():7.1f} mm  at t {d['t'][np.argmin(mm[:, j])]:.2f} s")

    reader = imageio_ffmpeg.read_frames(args.video)
    meta = next(reader)
    w, h = meta["size"]
    writer = imageio_ffmpeg.write_frames(args.out, (w, h), fps=meta["fps"], codec="libx264", quality=8,
                                         macro_block_size=1)
    writer.send(None)
    # timeline geometry (bottom strip): y = clearance clipped to [-10, 60] mm
    x0, x1, yb, ht = 40, w - 40, h - 30, 150
    lo, hi = -10.0, 60.0

    def ty(v):
        return int(yb - (np.clip(v, lo, hi) - lo) / (hi - lo) * ht)

    xs = np.linspace(x0, x1, T).astype(int)
    pts = np.stack([xs, [ty(v) for v in worst]], 1).astype(np.int32)
    i = -1
    for i, raw in enumerate(reader):
        frame = np.frombuffer(raw, np.uint8).reshape(h, w, 3).copy()
        k = min(max(0, i - args.frame_shift), T - 1)
        c = worst[k]
        j = int(np.argmin(mm[k]))
        col = (220, 30, 30) if c < 0 else (230, 140, 0) if c < WARN_MM else (20, 140, 40)
        if c < 0:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (220, 30, 30), 16)
            _text(frame, "CONTACT", (w - 330, 80), 1.8, (220, 30, 30), 4)
        _text(frame, f"t {d['t'][k]:5.2f} s   g {np.round(d['g'][k], 3).tolist()}", (40, 60), 1.0, (0, 0, 0))
        _text(frame, f"min clearance {c:6.1f} mm  ({links[j]})", (40, 110), 1.2, col, 3)
        _text(frame, f"left {side_min['left'][k]:6.1f} mm   right {side_min['right'][k]:6.1f} mm", (40, 155), 1.0,
              (0, 0, 0))
        # timeline
        cv2.rectangle(frame, (x0, yb - ht), (x1, yb), (255, 255, 255), -1)
        for v, cc in ((0.0, (220, 30, 30)), (WARN_MM, (230, 140, 0))):
            cv2.line(frame, (x0, ty(v)), (x1, ty(v)), cc, 1, cv2.LINE_AA)
        cv2.polylines(frame, [pts], False, (60, 60, 60), 2, cv2.LINE_AA)
        cv2.line(frame, (xs[k], yb - ht), (xs[k], yb), col, 3)
        _text(frame, f"min clearance (mm, {lo:.0f}..{hi:.0f})", (x0 + 8, yb - ht + 26), 0.7, (60, 60, 60))
        writer.send(np.ascontiguousarray(frame))
    writer.close()
    print(f"{i + 1} frames, {T} rows -> {args.out}")


if __name__ == "__main__":
    main()
