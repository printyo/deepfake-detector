import argparse, csv, sys, time
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import torch, torch.nn.functional as F
import open_clip

LABEL_MAP = {"FAKE": 1, "REAL": 0}

def encode_image_to_vec(pil_img, model, preprocess, device):
    x = preprocess(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.encode_image(x)          # (1, D)
        feat = F.normalize(feat, dim=-1).squeeze(0).cpu().numpy()  # (D,)
    return feat

def main():
    ap = argparse.ArgumentParser(description="Read meta.csv (with filename .mp4), load *_crop.png from --image_root, encode to CLIP, save as *_vector.npy")
    ap.add_argument("--csv", required=True, help="Path to meta.csv (expects a 'filename' column with .mp4)")
    ap.add_argument("--image_root", required=True, help="Folder containing the flat *_crop.png images (e.g., data/frames_biggest)")
    ap.add_argument("--out_root", required=True, help="Output folder for *_vector.npy")
    ap.add_argument("--label_col", default="label", help="CSV column for FAKE/REAL (default: label)")
    ap.add_argument("--filename_col", default="filename", help="CSV column for video filename (default: filename)")
    ap.add_argument("--model", default="ViT-B-16", help="CLIP vision backbone")
    ap.add_argument("--pretrained", default="openai", help="CLIP weights tag")
    args = ap.parse_args()

    csv_path = Path(args.csv).resolve()
    image_root = Path(args.image_root).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # Load CSV
    df = pd.read_csv(csv_path)
    if args.filename_col not in df.columns:
        print(f"[error] CSV must contain '{args.filename_col}' column. Found: {list(df.columns)}")
        sys.exit(1)

    # Device & CLIP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(min(torch.get_num_threads(), 4))
    print("[info] device:", device)
    print(f"[info] loading CLIP {args.model} ({args.pretrained})")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained, device=device
    )
    model.eval()

    # Index CSV
    idx_path = out_root / "index.csv"
    idx_f = open(idx_path, "w", newline="", encoding="utf-8")
    w = csv.writer(idx_f)
    w.writerow(["name", "label", "src_img", "dst_vec", "dim", "model"])

    total = len(df)
    done = 0
    t0 = time.time()

    for _, row in df.iterrows():
        vidname = str(row[args.filename_col]).strip()          # e.g., "abcxyz.mp4"
        stem = Path(vidname).stem                               # -> "abcxyz"
        img_name = f"{stem}_crop.png"                           # input image name
        vec_name = f"{stem}_vector.npy"                         # output vector name

        src = image_root / img_name
        if not src.exists():
            print(f"[warn] missing image: {src}")
            continue

        label_val = -1
        if args.label_col in df.columns:
            lbl = str(row[args.label_col]).strip().upper()
            if lbl in LABEL_MAP:
                label_val = LABEL_MAP[lbl]

        try:
            pil = Image.open(src).convert("RGB")
            vec = encode_image_to_vec(pil, model, preprocess, device)
        except Exception as e:
            print(f"[warn] FAILED on {src}: {e}")
            continue

        dst = out_root / vec_name
        np.save(dst, vec)

        w.writerow([stem, label_val, str(src), str(dst), vec.shape[0], args.model])
        done += 1
        if done % 200 == 0:
            print(f"[info] encoded {done}/{total} in {time.time()-t0:.1f}s")

    idx_f.close()
    print(f"[done] wrote {done} vectors to {out_root}")
    print(f"[done] index: {idx_path}")

if __name__ == "__main__":
    main()
