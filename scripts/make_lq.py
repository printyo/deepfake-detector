import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
from tqdm import tqdm

TIERS = {
    "LQ1-480ppppppp": {"scale": 640, "bv": "800k", "ba": "64k", "fps": 15},  # 480p-ish
    "LQ2-360p": {"scale": 480, "bv": "400k", "ba": "64k", "fps": 15},  # 360p-ish
    "LQ3-240p": {"scale": 426, "bv": "200k", "ba": "48k", "fps": 15},  # 240p-ish
}

def out_path(base_out, split, tier, part, filename):
    return Path(base_out) / split / tier / part / filename

def encode(ffmpeg, in_path: Path, cfg: dict, out_file: Path):
    """Return True if encoded OK, False if failed (existing files handled outside)."""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-y", "-loglevel", "error", "-i", str(in_path),
        "-vf", f"scale={cfg['scale']}:-2,fps={cfg['fps']}",
        "-b:v", cfg["bv"], "-c:v", "libx264", "-preset", "veryfast",
        "-b:a", cfg["ba"], "-c:a", "aac",
        str(out_file),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        return True
    except Exception:
        return False

def process_csv(csv_path: str, base_out: str, root_dir: Path, ffmpeg="ffmpeg", workers=8):
    df = pd.read_csv(csv_path)

    # Expect columns from the indexer/splitter
    required = {"rel_path", "part", "filename"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"{csv_path} missing columns: {missing}. Rebuild index/splits first.")

    # Split label
    if "split" in df.columns and df["split"].nunique() == 1:
        split = df["split"].iloc[0]
    else:
        split = "train" if "train" in str(csv_path).lower() else "test"

    # Keep only rows whose files exist on disk
    df = df[df["rel_path"].apply(lambda p: (root_dir / p).exists())].copy()

    # Build job list & count already existing outputs to show accurate progress
    jobs = []               # (in_path, out_file, cfg)
    skipped_existing = 0
    for _, row in df.iterrows():
        in_path = root_dir / row["rel_path"]
        for tier, cfg in TIERS.items():
            out_file = out_path(base_out, split, tier, row["part"], row["filename"])
            if out_file.exists():
                skipped_existing += 1
                continue
            jobs.append((in_path, out_file, cfg))

    total = len(jobs)
    if total == 0:
        print(f"[{split}] Nothing to do. Already up-to-date. (skipped existing: {skipped_existing})")
        return

    done_ok = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=workers) as ex, tqdm(total=total, desc=f"Compress {split}", unit="file") as pbar:
        futures = [ex.submit(encode, ffmpeg, in_p, cfg, out_p) for (in_p, out_p, cfg) in jobs]
        for fut in as_completed(futures):
            ok = fut.result()
            if ok: 
                done_ok += 1
            else:
                failed += 1
            pbar.update(1)

    print(f"[{split}] done: {done_ok} | failed: {failed} | skipped_existing: {skipped_existing} | total_requested: {total + skipped_existing}")

def main(train_csv, test_csv, base_out, root_dir, ffmpeg="ffmpeg", workers=8):
    root_dir = Path(root_dir).resolve()
    if train_csv:
        process_csv(train_csv, base_out, root_dir, ffmpeg, workers)
    if test_csv:
        process_csv(test_csv, base_out, root_dir, ffmpeg, workers)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default="data/splits/train.csv")
    ap.add_argument("--test_csv",  default="data/splits/test.csv")
    ap.add_argument("--out", required=True, help="e.g. data/compressed")
    ap.add_argument("--root_dir", default="data/dataset_original", help="Folder that contains all dfdc_train_part_*")
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    main(args.train_csv, args.test_csv, args.out, args.root_dir, args.ffmpeg, args.workers)
