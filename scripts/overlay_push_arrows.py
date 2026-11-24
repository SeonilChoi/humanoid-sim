"""Draw the play.py scheduled pushes on a recorded Olaf play video as arrows.

play.py writes videos/play/push_frames.csv next to the clip: per policy step (= video frame) env 0's pelvis, the
camera eye / lookat and the active push (dvx, dvy, shown for 1 s after each kick). The arrow is the horizontal
velocity kick, drawn in 3D at the pelvis height with its tip at the pelvis (where the push lands), projected with
the viewport pinhole camera (Kit default perspective camera: 18.147 mm focal length, 20.955 mm horizontal aperture).

  python scripts/overlay_push_arrows.py <clip.mp4> <push_frames.csv> <out.mp4>
"""

import argparse
import csv

import cv2
import imageio_ffmpeg
import numpy as np

FOCAL_MM, H_APERTURE_MM = 18.147, 20.955
ARROW_M_PER_MPS = 1.2   # arrow length per m/s of the kick
TIP_GAP_M = 0.06        # stop just short of the pelvis


def project(pts, eye, look, w, h):
    fwd = look - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    f = w * FOCAL_MM / H_APERTURE_MM
    rel = pts - eye
    z = rel @ fwd
    u = w / 2 + f * (rel @ right) / z
    v = h / 2 - f * (rel @ up) / z
    return np.stack((u, v), -1)


def _inset(frame, vx, vy, eye, look, w, yaw=None):
    """Top view in the upper-right corner, oriented like the camera (up = away from the camera, right = screen right):
    a push toward / away from the camera is foreshortened in the 3D arrow but reads clearly here."""
    fwd = (look - eye)[:2]
    fwd /= np.linalg.norm(fwd)
    right = np.array([fwd[1], -fwd[0]])
    r, cx, cy = 110, w - 170, 170
    cv2.circle(frame, (cx, cy), r, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), r, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 14, (90, 90, 90), -1, cv2.LINE_AA)   # the robot
    if yaw is not None:   # robot facing direction (green)
        h = np.array([np.cos(yaw), np.sin(yaw)])
        hx, hy = float(h @ right), float(h @ fwd)
        cv2.arrowedLine(frame, (cx, cy), (int(cx + 0.55 * r * hx), int(cy - 0.55 * r * hy)), (40, 170, 40), 5,
                        cv2.LINE_AA, tipLength=0.35)
    v = np.array([vx, vy]) / np.hypot(vx, vy)
    sx, sy = float(v @ right), float(v @ fwd)
    tail = (int(cx - 0.85 * r * sx), int(cy + 0.85 * r * sy))
    tip = (int(cx - 18 * sx), int(cy + 18 * sy))
    cv2.arrowedLine(frame, tail, tip, (235, 30, 30), 8, cv2.LINE_AA, tipLength=0.3)
    txt = "top view  red: push  green: robot front"
    (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.putText(frame, txt, (w - tw - 20, cy + r + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "camera", (cx - 52, cy + r - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (90, 90, 90), 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("csv")
    ap.add_argument("out")
    ap.add_argument("--frame_shift", type=int, default=1, help="frame i is rendered with the camera of row i - shift")
    args = ap.parse_args()
    rows = [list(map(float, r.values())) for r in csv.DictReader(open(args.csv))]
    reader = imageio_ffmpeg.read_frames(args.video)   # cv2's bundled ffmpeg cannot decode the recorder's clips
    meta = next(reader)
    w, h = meta["size"]
    writer = imageio_ffmpeg.write_frames(args.out, (w, h), fps=meta["fps"], codec="libx264", quality=8,
                                         macro_block_size=1)
    writer.send(None)
    i = 0
    for raw in reader:
        frame = np.frombuffer(raw, np.uint8).reshape(h, w, 3).copy()   # RGB
        if i < len(rows):
            r = rows[i]
            c = rows[max(0, i - args.frame_shift)]
            vx, vy = r[10], r[11]
            if vx or vy:
                mag = float(np.hypot(vx, vy))
                d = np.array([vx, vy, 0.0]) / mag
                pelvis = np.array(r[1:4])
                tip = pelvis - TIP_GAP_M * d
                tail = tip - ARROW_M_PER_MPS * mag * d
                (tu, tv), (hu, hv) = project(np.stack((tail, tip)), np.array(c[4:7]), np.array(c[7:10]), w, h)
                a, b = (int(tu), int(tv)), (int(hu), int(hv))
                cv2.arrowedLine(frame, a, b, (0, 0, 0), 14, cv2.LINE_AA, tipLength=0.25)
                cv2.arrowedLine(frame, a, b, (235, 30, 30), 9, cv2.LINE_AA, tipLength=0.25)
                label = f"push dvx {vx:+.2f} dvy {vy:+.2f} m/s"
                cv2.putText(frame, label, (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 8, cv2.LINE_AA)
                cv2.putText(frame, label, (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (235, 30, 30), 3, cv2.LINE_AA)
            if vx or vy:
                _inset(frame, vx, vy, np.array(c[4:7]), np.array(c[7:10]), w, r[12] if len(r) > 12 else None)
        writer.send(np.ascontiguousarray(frame))
        i += 1
    writer.close()
    print(f"{i} frames, {len(rows)} log rows -> {args.out}")


if __name__ == "__main__":
    main()
