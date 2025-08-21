# Deepfake Detection Project

Detect deepfakes in short social media videos and study how **video quality** (HQ vs LQ1/LQ2/LQ3) affects detection.

---

## Quickstart

1. **Get Python** 3.13.7 (same version as the team).
2. **Clone or copy** the project folder `deepfake-project/` to your machine or external SSD.
3. Open a terminal in the project root and create a virtual env:
    ```bash
    python -m venv .venv
    # Windows PowerShell
    .venv\Scripts\activate
    # macOS/Linux
    # source .venv/bin/activate
    ```
4. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
5. Put the **dataset** (DFDC parts with `metadata.json` + mp4) under:
    ```
    data/dataset_original/
    └── dfdc_train_part_00/
        ├── metadata.json
        └── *.mp4
    └── dfdc_train_part_01/
        ├── metadata.json
        └── *.mp4
        .
        .
        .
    ```
    (You can symlink/mount if the raw data lives elsewhere—keep the relative path the same.)
6. Verify **ffmpeg** is available:
    ```bash
    ffmpeg -version
    ```
    - If Windows says “not recognized”, install ffmpeg and add `C:\ffmpeg\bin` to PATH, or edit scripts to pass the full path to `ffmpeg.exe`.
7. Build index and make **train/test** splits for one part (pilot):
    ```bash
    python scripts/build_index_dfdc.py --part_dir "data/dataset_original/dfdc_train_part_00" --out data/meta.csv
    python scripts/split_train_test.py --meta data/meta.csv --out data/splits --train_ratio 0.8
    ```
8. Make **low-quality** copies (LQ1/LQ2/LQ3) for the pilot split:

    ```bash
    python scripts/make_lq.py --train_csv data/splits/train.csv --test_csv data/splits/test.csv --out data/dataset_compressed --workers 12

    ```

That’s the first milestone done. Next steps: face crops → features → train.

---

## Requirements

-   Python **3.13.7**
-   Git
-   ffmpeg (system tool)
-   Disk space: DFDC raw (~550 GB) + processed outputs (varies)

> **GPU note:** On Windows + AMD (e.g., RX 6650 XT), PyTorch will run on **CPU**. For GPU training, use Linux (ROCm) or cloud GPUs (Colab, RunPod, Lambda, etc.).

Check PyTorch GPU availability:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

---

## Project Structure

```
deepfake-project/
│
├── data/
│   ├── dataset_original/     # Original DFDC parts
│   ├── splits/               # train.csv, test.csv, val.csv
│   ├── dataset_compressed/   # HQ, LQ1, LQ2, LQ3 (by split/part)
│   ├── frames_face/          # Cropped face images (by split/tier/part/video_id)
│
├── scripts/                  # Python scripts
├── models/                   # Checkpoints
├── reports/                  # Metrics, logs, figures
└── README.md
```

---

## Common Commands

### Create / repair venv

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate
pip install -r requirements.txt
```

Update the lockfile:

```bash
pip freeze > requirements.txt
```

### Build index (pilot: one DFDC part)

```bash
python scripts/build_index_dfdc.py --part_dir "data/dataset_original/dfdc_train_part_00" --out data/meta.csv
```

### Train/test split (group-safe, no leakage)

```bash
python scripts/split_train_test.py --meta data/meta.csv --out data/splits --train_ratio 0.8
```

### Make low-quality tiers (LQ1/LQ2/LQ3)

```bash
python scripts/make_lq.py --train_csv data/splits/train.csv --test_csv data/splits/test.csv --out data/dataset_compressed
```

---

## Tips

-   Keep everything **relative** to the repo root to avoid absolute path issues across PCs.
-   For Windows + ffmpeg:
    -   Install from https://www.gyan.dev/ffmpeg/builds/
    -   Add `C:\ffmpeg\bin` to PATH, or pass `--ffmpeg "C:\ffmpeg\bin\ffmpeg.exe"` to scripts.
-   Preprocess on CPU overnight if needed, then **train on a GPU service** to save time.
-   Commit only **code/configs**. Ignore large data:
    ```
    data/
    models/
    *.mp4
    *.npz
    *.jpg
    ```

---

## Milestones

1. **Pilot** on one DFDC part: index → split → compress → crop → features → baseline.
2. Scale to **all parts**: global split by `group_id`, repeat pipeline.
3. Compare HQ vs LQ metrics; fine-tune for LQ or do mixed-quality training.

---

## Troubleshooting

-   **WinError 2** during compression → ffmpeg not found. Check PATH or pass the full path via `--ffmpeg`.
-   **Slow processing** → reduce fps to 10; limit frames per video in scripts (e.g., `--max_frames 64` for crops/features).
-   **GPU not used** on Windows + AMD → expected; use Linux ROCm or cloud GPUs for training.

---
