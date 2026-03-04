# Hilti-Trimble Fine-Tuning Guide

Fine-tuning MixVPR on the **Hilti-Trimble SLAM Challenge 2026** indoor dataset.
This page describes everything you need to go from raw ROS2 bags to fine-tuned
weights and a Recall@K evaluation on a held-out run.

---

## Table of Contents
1. [Data Layout](#1-data-layout)
2. [Step 0 — Extract Images](#2-step-0--extract-images)
3. [Step 1 — Align Frames to GT Poses](#3-step-1--align-frames-to-gt-poses)
4. [Preprocessing Choices](#4-preprocessing-choices)
5. [Place Labelling Scheme](#5-place-labelling-scheme)
6. [Dual-Camera (cam0 + cam1) Strategy](#6-dual-camera-cam0--cam1-strategy)
7. [Fine-Tuning: Smoke Run](#7-fine-tuning-smoke-run)
8. [Fine-Tuning: Real Leave-One-Out](#8-fine-tuning-real-leave-one-out)
9. [Evaluation: Recall@K on Held-Out Run](#9-evaluation-recallk-on-held-out-run)
10. [Leave-One-Out Automation](#10-leave-one-out-automation)
11. [Frequently Asked Questions](#11-frequently-asked-questions)

---

## 1. Data Layout

```
hilti-trimble-slam-challenge-2026/
├── groundtruth/
│   ├── floor_1_2025-05-05_run_1.txt        # TUM format: ts tx ty tz qx qy qz qw
│   ├── floor_2_2025-05-05_run_1.txt
│   ├── floor_2_2025-10-28_run_1.txt
│   ├── floor_2_2025-10-28_run_2.txt
│   └── floor_UG1_2025-10-16_run_1.txt
└── data/
    └── <floor>/<date>/<run>/rosbag/        # ROS2 .mcap bag files
```

After extraction and alignment, each run gains a per-camera directory:

```
challenge_tools_ros/vpr/
└── data/
    └── <floor>_<date>_<run>/
        ├── cam0/
        │   ├── images/          # extracted JPEGs named by ns timestamp
        │   └── aligned.csv      # produced by align_frames_to_gt.py
        └── cam1/                # optional second camera
            ├── images/
            └── aligned.csv
```

**aligned.csv columns** (produced by `align_frames_to_gt.py`):

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | float | Image time in seconds |
| `tx` | float | GT pose X translation (metres) |
| `ty` | float | GT pose Y translation (metres) |
| `tz` | float | GT pose Z translation (metres) |
| `qx,qy,qz,qw` | float | GT pose quaternion |
| `image_path` | str | Absolute OR CSV-relative path to JPEG |

---

## 2. Step 0 — Extract Images

Use the challenge tools to extract frames from the ROS2 bag at **10 Hz**
(recommended — see [Why 10 Hz?](#why-not-30-hz)).

```bash
# cam0
python challenge_tools_ros/vpr/src/extract_images_cam0.py \
    --bag  hilti-trimble-slam-challenge-2026/data/floor_1/2025-05-05/run_1/rosbag \
    --out  challenge_tools_ros/vpr/data/floor_1_2025-05-05_run_1/cam0/images \
    --hz   10

# cam1 (if available — same script, different topic)
python challenge_tools_ros/vpr/src/extract_images_cam1.py \
    --bag  hilti-trimble-slam-challenge-2026/data/floor_1/2025-05-05/run_1/rosbag \
    --out  challenge_tools_ros/vpr/data/floor_1_2025-05-05_run_1/cam1/images \
    --hz   10
```

Repeat for all 5 runs.  Extracted images are named `<nanosecond_timestamp>.jpg`.

---

## 3. Step 1 — Align Frames to GT Poses

Interpolates GT TUM poses to image timestamps:

```bash
python challenge_tools_ros/vpr/src/align_frames_to_gt.py \
    --images  challenge_tools_ros/vpr/data/floor_1_2025-05-05_run_1/cam0/images \
    --gt      hilti-trimble-slam-challenge-2026/groundtruth/floor_1_2025-05-05_run_1.txt \
    --out     challenge_tools_ros/vpr/data/floor_1_2025-05-05_run_1/cam0/aligned.csv
```

Do the same for cam1, and for the other 4 runs.

---

## 4. Preprocessing Choices

All preprocessing is defined in `dataloaders/HiltiDataset.py` as two constants:

```python
HILTI_TRAIN_TRANSFORM   # used during fine-tuning
HILTI_EVAL_TRANSFORM    # used during recall evaluation — must match training!
```

| Step | Choice | Justification |
|------|--------|---------------|
| **Rotation 180°** | Applied in transform (not at extraction) | Keeps raw files intact; fix is **automatically applied** in every code path that imports these transforms (training, smoke, eval). |
| **Resize** | `(320, 320)` bicubic | Required by the pretrained MixVPR backbone (stride 16, input 320→20×20 feature maps). |
| **Normalization** | ImageNet mean/std | Pretrained ResNet50 backbone expects ImageNet statistics. |
| **Fisheye distortion** | Kept raw (no undistortion) | Undistortion requires calibration data and adds complexity. The model can adapt via fine-tuning. If calibration is available, add `cv2.fisheye.undistortImage` before PIL conversion. |
| **Color augmentation** (train only) | `ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)` | Indoor lighting varies significantly; mild jitter improves robustness without destroying structural features. |

> **Important:** `src/demo.py`'s `load_image()` does NOT rotate and is for non-Hilti data.
> Always use `HILTI_EVAL_TRANSFORM` for Hilti recall evaluation, not `load_image()`.
> `src/run_hilti_recall.py` uses `BaseDataset` → if using Hilti images, pass
> `--use_hilti_transform` (or patch `load_image()` to call `HILTI_EVAL_TRANSFORM`).

---

## 5. Place Labelling Scheme

**Definition:**

```
place_label = (run_id, floor(tx / grid_m), floor(ty / grid_m))
```

Implemented as a single integer:
```
place_id = run_id * (STRIDE²) + bin_x * STRIDE + bin_y
```

**Default: `grid_m = 1.0` metre.**

| Parameter | Value | Why |
|-----------|-------|-----|
| `grid_m` | 1.0 m | At 10 Hz the robot moves ~0.2–0.5 m/frame; a 1 m cell contains ~3–5 frames — enough for `img_per_place=4`. |
| Axis | XY only | Indoor robots operate on floors; Z varies little. Binning Z would over-fragment. |
| Minimum images/place | = `img_per_place` (4) | Places with fewer images cannot fill a training sample; they are silently dropped and the drop% is logged. |

**Adjust `grid_m` if:**
- Drop% is too high (> 30%) → increase `grid_m` (e.g. 1.5 m)
- Place descriptors are too similar within a place → decrease `grid_m` (e.g. 0.5 m)

---

## 6. Dual-Camera (cam0 + cam1) Strategy

**Strategy: Option 1 — same place label, extra images.**

Both cameras are mounted rigidly on the same rig; they capture the same
location at the same time from slightly different directions.  We assign both
sets of images to the **same place label** derived from the shared GT pose.
This increases intra-place visual diversity (different viewing angles) that
helps the metric-learning loss form more informative pairs.

**To enable cam1**, supply both aligned.csv files as separate entries in
`--train_csvs`:

```bash
python src/train_hilti.py \
    --train_csvs \
        run1/cam0/aligned.csv \
        run1/cam1/aligned.csv \   # same run, cam1 → same place labels via shared pose
        run2/cam0/aligned.csv \
        ...
```

> **Caveat:** `run_id` in `HiltiDataset` is assigned by CSV file index, so
> cam0 and cam1 of the same run get *different* `run_id`s, and hence *different*
> place integer IDs — they become separate, adjacent grid cells in label space.
> If you want cam0 and cam1 to share the exact same place ID, pre-merge their
> aligned.csv files (concatenate rows) before passing to `--train_csvs`.

**Graceful fallback:** If only cam0 is available, simply omit `cam1/aligned.csv`.

---

## 7. Fine-Tuning: Smoke Run

A smoke run validates the whole pipeline end-to-end in < 5 minutes before
committing to overnight training.

```bash
python src/train_hilti.py \
    --train_csvs challenge_tools_ros/vpr/data/floor_1_2025-05-05_run_1/cam0/aligned.csv \
    --ckpt       LOGS/resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt \
    --output_dir LOGS/smoke_test \
    --smoke
```

Smoke mode automatically sets:
- 1 CSV (only first entry from `--train_csvs`)
- `cap_places = 50`
- `batch_size = 8`, `img_per_place = 4` → 32 images/batch
- `max_epochs = 1`
- `num_workers = 0` (simpler debugging)

**Expected smoke output:**
```
[HiltiDataset] places total=XXX  kept=YYY  dropped=Z.Z%  images=NNNN
[HiltiDataModule] train: 50 places, ~200 images, grid=1.0 m, batch=8×4
[train_hilti] Training complete in ~X.X min.
Best checkpoint : LOGS/smoke_test/mixvpr_hilti_epoch=00_...ckpt
```

**After smoke training — verify inference:**
```bash
python src/run_hilti_recall.py \
    --eval_root <path_to_eval_root_with_db_query_gt> \
    --ckpt      LOGS/smoke_test/last.ckpt \
    --device    mps \
    --no_cache
```

---

## 8. Fine-Tuning: Real Leave-One-Out

Train on 4 runs, evaluate on the 5th.  Example: hold out `floor_1_run_1`.

```bash
python src/train_hilti.py \
    --train_csvs \
        challenge_tools_ros/vpr/data/floor_2_2025-05-05_run_1/cam0/aligned.csv \
        challenge_tools_ros/vpr/data/floor_2_2025-10-28_run_1/cam0/aligned.csv \
        challenge_tools_ros/vpr/data/floor_2_2025-10-28_run_2/cam0/aligned.csv \
        challenge_tools_ros/vpr/data/floor_UG1_2025-10-16_run_1/cam0/aligned.csv \
    --held_out_csv \
        challenge_tools_ros/vpr/data/floor_1_2025-05-05_run_1/cam0/aligned.csv \
    --ckpt       LOGS/resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt \
    --output_dir LOGS/hilti_finetune/fold_floor1_run1 \
    --max_epochs 30 \
    --lr         2e-4 \
    --batch_size 16
```

Recommended hyperparameters for M2 Mac overnight:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `max_epochs` | 30 | ~4–8 h depending on run count |
| `lr` | 2e-4 | AdamW; safe for fine-tuning |
| `weight_decay` | 1e-3 | Same as pretrained repo |
| `layers_to_freeze` | 2 | Freeze BN + ResNet stages 1–2 |
| `batch_size` | 16 | 16 places × 4 images = 64 images/batch; fits M2 16 GB |
| `precision` | 32 | fp16 can NaN on MPS with MultiSimilarityLoss |
| `grid_m` | 1.0 | Start here; adjust if >30% places dropped |

---

## 9. Evaluation: Recall@K on Held-Out Run

1. **Build eval set** using the challenge tools:
   ```bash
   python challenge_tools_ros/vpr/src/build_mixvpr_evalset.py \
       --run_dir challenge_tools_ros/vpr/data/floor_1_2025-05-05_run_1/cam0 \
       --out     eval/floor_1_2025-05-05_run_1_cam0 \
       --pos_dist_m 3.0
   ```
   Expected outputs: `eval/.../db/`, `eval/.../query/`, `eval/.../gt_positives.npy`

   **Sanity check:** If `queries_with_zero_positives` is high (> 20%), try
   increasing `--pos_dist_m` (e.g. 5.0) or revisiting extraction Hz.

2. **Run recall script** with the fine-tuned checkpoint:
   ```bash
   python src/run_hilti_recall.py \
       --eval_root eval/floor_1_2025-05-05_run_1_cam0 \
       --ckpt      LOGS/hilti_finetune/fold_floor1_run1/last.ckpt \
       --device    mps \
       --no_cache
   ```

> **Transform alignment:** `run_hilti_recall.py` uses `src/demo.py`'s
> `load_image()` which does NOT apply the 180° rotation.  For Hilti inference,
> patch `load_image()` or subclass `BaseDataset` to use `HILTI_EVAL_TRANSFORM`:
>
> ```python
> # In src/run_hilti_recall.py, replace BaseDataset with:
> from dataloaders.HiltiDataset import HILTI_EVAL_TRANSFORM
>
> class HiltiBaseDataset(BaseDataset):
>     def __getitem__(self, index):
>         pil = Image.open(self.img_path_list[index]).convert("RGB")
>         return HILTI_EVAL_TRANSFORM(pil), index
> ```

---

## 10. Leave-One-Out Automation

Use `src/run_loo_hilti.py` to iterate over all 5 folds automatically:

```bash
python src/run_loo_hilti.py \
    --csv_dir  challenge_tools_ros/vpr/data \
    --eval_dir eval \
    --ckpt     LOGS/resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt \
    --output   LOGS/loo_results.json \
    --max_epochs 30
```

Results are written to `LOGS/loo_results.json` with per-fold Recall@1/5/10.

---

## 11. Frequently Asked Questions

### Why not 30 Hz?
At 30 Hz consecutive frames are < 0.2 m apart.  The MultiSimilarityLoss needs
visually **distinct** positives (same place, different viewpoint).  Near-duplicate
frames inflate the positive set without adding information and make the loss
trivially small early in training.  **10 Hz** gives ~3–5 frames per 1 m grid
cell — enough diversity for `img_per_place=4`.

### Why two separate transform constants?
`HILTI_TRAIN_TRANSFORM` includes `ColorJitter` for data augmentation.
`HILTI_EVAL_TRANSFORM` is deterministic (no augmentation).  Using augmentation
at eval time would degrade retrieval score.

### Why AdamW instead of SGD (original repo uses SGD)?
The original SGD with `lr=0.05` is tuned for full-scale training on ~100k
places.  Fine-tuning on Hilti's ~2–5k places requires a small, stable LR.
AdamW's adaptive per-parameter scaling converges reliably at `lr=2e-4` without
per-run LR search.

### Why layers_to_freeze=2?
Stages 0–2 extract low-level features (Gabor / edge / colour) that transfer
well from street-view pretraining.  Stages 3–4 learn mid/high-level semantic
patterns that need updating for indoor fisheye texture.  Freezing more would
leave the model too close to the pretrained GSVCities domain.

### The training loss is not decreasing — what to try?
1. Check `cap_places` is off (smoke only).
2. Lower `lr` to `5e-5`.
3. Increase `min_img_per_place` to ensure diverse positives.
4. Increase `grid_m` to 1.5 m if most places have exactly 4 images (no diversity).
5. Verify the 180° rotation is consistent between training and eval images.


# Recall without fine-tuning (5m):
```
python src/run_hilti_recall_fixed.py \
    --eval_root "/Volumes/T9/DevSpace/Github/hilti-trimble-slam-challenge-2026/challenge_tools_ros/vpr/data/floor_1_2025-05-05_run_1/eval/mixvpr_evalset" \
    --cam 0 \
    --ckpt "./LOGS/resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt" \
    --device mps \
    --batch 16 \
    --hilti \
    --label "pretrained"
[1/6] Loading checkpoint from ./LOGS/resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt ...
[2/6] Building datasets (cam=0, transform=hilti) ...
      db=812 images  query=348 images
[3/6] Inferring feature dimension ...
[W NNPACK.cpp:64] Could not initialize NNPACK! Reason: Unsupported hardware.
      feat_dim=4096
[4/6] Extracting DB descriptors (batch=16) ...
Extracting db features: 100%|███████████████████████████████████████| 51/51 [00:42<00:00,  1.21it/s]
      db_desc shape: (812, 4096)
[5/6] Extracting query descriptors (batch=16) ...
Extracting query features: 100%|████████████████████████████████████| 22/22 [00:26<00:00,  1.20s/it]
      q_desc shape: (348, 4096)
[6/6] Loading ground truth from /Volumes/T9/DevSpace/Github/hilti-trimble-slam-challenge-2026/challenge_tools_ros/vpr/data/floor_1_2025-05-05_run_1/eval/mixvpr_evalset/gt_positives.npy ...
db_desc=(812, 4096) q_desc=(348, 4096) gt_queries=348

⚠️  WARNING: 294 queries have empty GT positives: [38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347]
    These queries will be excluded from recall calculation to avoid deflating metrics.
    Filtering: 54 queries with GT / 348 total queries

Normalizing descriptors for cosine similarity...
      Normalized shapes: db=(812, 4096), query=(54, 4096)

Calculating recalls...

+----------------------------------+
| Performances on pretrained_cam0  |
+----------+-------+-------+-------+
|    K     |   1   |   5   |   10  |
+----------+-------+-------+-------+
| Recall@K | 64.81 | 68.52 | 70.37 |
+----------+-------+-------+-------+
recalls: {1: 0.6481481481481481, 5: 0.6851851851851852, 10: 0.7037037037037037}

[DEBUG] Recalls on unnormalized descriptors (for comparison):

+----------------------------------------------+
| Performances on pretrained_cam0_unnormalized |
+--------------+---------+---------+-----------+
|      K       |    1    |    5    |     10    |
+--------------+---------+---------+-----------+
|   Recall@K   |  64.81  |  68.52  |   70.37   |
+--------------+---------+---------+-----------+
```

# Recall without fine-tuning (1m):
````

````

