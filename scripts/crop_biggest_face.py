#!/usr/bin/env python3
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import torch

print("[boot] starting", __file__)
from facenet_pytorch import MTCNN

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def load_image(p: Path):
    return Image.open(p).convert("RGB")


def get_biggest_face(pil_img, detector, prob_thresh=0.80):
    """Return bounding box (x1,y1,x2,y2) of the biggest face in an image."""
    boxes, probs = detector.detect(pil_img)
    if boxes is None or len(boxes) == 0:
        return None
    valid = [
        (i, b)
        for i, (b, pr) in enumerate(zip(boxes, probs))
        if pr is None or pr >= prob_thresh
    ]
    if not valid:
        return None
    # pick largest by area
    _, box = max(valid, key=lambda x: (x[1][2] - x[1][0]) * (x[1][3] - x[1][1]))
    x1, y1, x2, y2 = map(int, box)
    return (x1, y1, x2, y2)


def process_video_dir(frame_dir: Path, out_root: Path, detector, overwrite=False):
    """Process one frame_<video> folder, save only the biggest face crop."""
    stem = frame_dir.name.replace("frame_", "")
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"{stem}_crop.png"

    if out_path.exists() and not overwrite:
        print(f"[skip] {stem} already cropped")
        return 0

    frames = sorted([p for p in frame_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
    best_area = 0
    best_crop = None

    for fp in frames:
        try:
            img = load_image(fp)
        except Exception:
            continue
        box = get_biggest_face(img, detector)
        if not box:
            continue
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)
        if area > best_area:
            best_area = area
            best_crop = img.crop((x1, y1, x2, y2))

    if best_crop:
        best_crop.save(out_path, format="PNG", compress_level=3)
        print(f"[ok] {stem}: saved biggest crop ({best_area}px) -> {out_path}")
        return 1
    else:
        print(f"[warn] {stem}: no face detected")
        return 0


def main():
    ap = argparse.ArgumentParser(description="Crop only the single biggest face per video.")
    ap.add_argument("--in_root", type=Path, default=Path("data/frames_raw"),
                    help="Root containing frame_<video> dirs")
    ap.add_argument("--out_root", type=Path, default=Path("data/frames_face_biggest"),
                    help="Where to save *_crop.png")
    ap.add_argument("--device", type=str, default="cpu", help="cpu or mps/cuda")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(1)  # avoids slow OpenMP init on macOS
    print("[init] creating facenet-pytorch MTCNN...")
    detector = MTCNN(keep_all=True, device=args.device)
    print("[init] detector ready")

    args.in_root = args.in_root.resolve()
    args.out_root = args.out_root.resolve()
    args.out_root.mkdir(parents=True, exist_ok=True)

    print("[start] cwd =", Path.cwd())
    print("[start] in_root =", args.in_root)
    print("[start] out_root =", args.out_root)

    if not args.in_root.exists():
        print("[error] in_root does not exist:", args.in_root)
        raise SystemExit(1)

    # recursive search for frame_* folders
    frame_dirs = [p for p in args.in_root.rglob("frame_*") if p.is_dir()]

    if not frame_dirs:
        print("[warn] found 0 frame_* folders under:", args.in_root)
        return

    print("[debug] found", len(frame_dirs), "frame_* folders")
    for d in sorted(frame_dirs)[:5]:
        print("   ", d)

    total = 0
    for fd in sorted(frame_dirs):
        total += process_video_dir(fd, args.out_root, detector, overwrite=args.overwrite)

    print(f"[done] total crops saved: {total}")


if __name__ == "__main__":
    main()
