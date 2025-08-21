import csv, json
from pathlib import Path
from build_index_dfdc import index_folder

def main(root_dir, out_csv):
    root_dir = Path(root_dir).resolve()
    part_dirs = sorted([p for p in root_dir.iterdir() if p.is_dir() and p.name.startswith("dfdc_train_part_")])

    all_rows = []
    for part_dir in part_dirs:
        print("Indexing", part_dir.name)
        rows = index_folder(part_dir, root_dir)
        all_rows.extend(rows)

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Wrote {len(all_rows)} rows from {len(part_dirs)} folders into {out_csv}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", default="data/dataset_original", help="Folder containing all dfdc_train_part_* dirs")
    ap.add_argument("--out", default="data/meta_all.csv")
    args = ap.parse_args()
    main(args.root_dir, args.out)

#python scripts/build_index_all.py --root_dir data/dataset_original --out data/meta_all.csv
