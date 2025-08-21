import json, csv
from pathlib import Path

def load_one_metadata(meta_path: Path):
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)

def index_folder(part_dir: Path, root_dir: Path):
    part_dir = part_dir.resolve()
    root_dir = root_dir.resolve()

    meta_path = part_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata.json in {part_dir}")

    meta = load_one_metadata(meta_path)
    rows = []
    for fname, info in meta.items():
        file_path = (part_dir / fname).resolve()

        # make sure file_path is under root_dir
        try:
            rel_path = file_path.relative_to(root_dir)
        except ValueError:
            # Helpful message so users set --root_dir correctly
            raise ValueError(
                f"{str(file_path)!r} is not under root_dir {str(root_dir)!r}.\n"
                f"Tip: set --root_dir to the folder that CONTAINS {part_dir.name}.\n"
                f"Example: if part_dir={part_dir}, use --root_dir={part_dir.parent}"
            )

        label = info.get("label")
        original = info.get("original", "")
        group_id = original if label == "FAKE" else fname

        rows.append({
            "part": part_dir.name,
            "filename": fname,
            "rel_path": str(rel_path).replace("\\", "/"),  # portable
            "label": label,
            "original": original,
            "group_id": group_id
        })
    return rows

def main(part_dir, out_csv, root_dir):
    part_dir = Path(part_dir)
    root_dir = Path(root_dir)

    rows = index_folder(part_dir, root_dir)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_csv}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--part_dir", required=True, help="e.g. data/dataset_original/dfdc_train_part_00")
    ap.add_argument("--root_dir", default="data/raw", help="Folder that CONTAINS the dfdc_train_part_* dirs")
    ap.add_argument("--out", default="data/meta.csv")
    args = ap.parse_args()
    main(args.part_dir, args.out, args.root_dir)
