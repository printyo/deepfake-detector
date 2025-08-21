# This script splits the dataset into train and test sets based on group_id,
# ensuring that the split is stratified based on the presence of FAKE labels.

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

def make_group_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("group_id")["label"].agg(lambda x: (x == "FAKE").mean()).reset_index()
    g["bucket"] = (g["label"] > 0).astype(int)
    return g

def apply_split(df: pd.DataFrame, g_train: pd.DataFrame, g_test: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["split"] = "train"
    out.loc[out["group_id"].isin(g_test["group_id"]), "split"] = "test"
    return out

def check_no_overlap(df: pd.DataFrame):
    by = df.groupby("group_id")["split"].nunique()
    bad = by[by > 1]
    if len(bad) > 0:
        examples = ", ".join(list(bad.index[:5]))
        raise RuntimeError(f"Leakage detected: some group_id appear in multiple splits (e.g., {examples})")

def print_counts(df: pd.DataFrame):
    for s in ["train", "test"]:
        d = df[df["split"] == s]
        print(f"{s:>5} {len(d):6d} | REAL: {(d.label=='REAL').sum():6d}  FAKE: {(d.label=='FAKE').sum():6d}  groups: {d['group_id'].nunique():6d}")

def main(meta_csv: str, out_dir: str, train_ratio: float = 0.8, seed: int = 42):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(meta_csv)
    # Expect columns like: part, filename, rel_path, label, original, group_id
    required_cols = {"group_id", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {meta_csv}: {missing}")

    # Build group table for stratification
    groups = make_group_table(df)

    # Train/test split at group level
    g_train, g_test = train_test_split(
        groups,
        test_size=(1.0 - train_ratio),
        random_state=seed,
        stratify=groups["bucket"]
    )

    # Apply split back to the full row-level dataframe
    df_split = apply_split(df, g_train, g_test)

    # Safety check
    check_no_overlap(df_split)

    # Save
    df_split[df_split["split"] == "train"].to_csv(out_dir / "train.csv", index=False)
    df_split[df_split["split"] == "test"].to_csv(out_dir / "test.csv", index=False)

    # Report
    print_counts(df_split)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="data/meta.csv")
    ap.add_argument("--out", default="data/splits")
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args.meta, args.out, args.train_ratio, args.seed)
