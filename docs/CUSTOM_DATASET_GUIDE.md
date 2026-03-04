# Custom Dataset Guide for MixVPR

This guide explains everything you need to provide in order to evaluate the pretrained MixVPR model on your own image dataset using `test_custom_dataset.py`.

---

## 1. Core Concepts

Visual Place Recognition (VPR) is a **retrieval task**, not a classification task.

| Concept | Meaning |
|---------|---------|
| **Database images** | Your "map" — the reference set of places you want to recognise. Sometimes called *gallery* or *reference* images. |
| **Query images** | Images taken at test time whose location you want to determine by matching them to the database. |
| **Descriptor** | A fixed-length vector (4096-dim for this checkpoint) produced by MixVPR from a single image. |
| **Recall@K** | The fraction of queries for which at least one of the top-K retrieved database images is a true positive (a correct match). This is the standard VPR evaluation metric. |
| **Ground truth** | For each query image, the list of database image indices that are considered correct matches. |

---

## 2. Folder Structure

The only hard requirement is **two separate directories**: one for database images and one for query images.

```
my_dataset/
├── database/          ← reference / map images
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
└── queries/           ← probe images to localise
    ├── q_001.jpg
    ├── q_002.jpg
    └── ...
```

**Rules:**
- Images in each folder are sorted **alphabetically** by filename when loaded. The order determines integer indices (0, 1, 2, …) used for ground truth.
- Each folder must contain only image files. Subdirectories are ignored.
- Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`
- Any resolution is fine — images are automatically resized to 320×320.

---

## 3. Ground Truth

Ground truth tells the script which database images are correct matches for each query. There are three modes.

---

### Mode A — GPS Coordinates (recommended for real-world datasets)

You supply two CSV files, one for database images and one for queries.

**CSV format:**

```csv
lat,lon
37.7749,-122.4194
37.7751,-122.4188
...
```

- Must have columns named `lat` and `lon` (case-insensitive).
- **Row order must match the alphabetical image order in the folder.**  
  Row 0 corresponds to the first image alphabetically, row 1 to the second, etc.
- Extra columns (e.g. `filename`, `timestamp`, `altitude`) are simply ignored.

**What the script does with GPS:**  
For each query, it finds all database images whose GPS coordinate is within `--dist-thr` metres (default 25 m). Those are the positives. This uses the haversine formula, so coordinates must be decimal degrees (WGS-84), not UTM.

**Choosing a distance threshold:**  
| Scene type | Typical threshold |
|------------|-------------------|
| Dense urban (narrow streets) | 10–15 m |
| Suburban / campus | 25 m |
| Rural / open environment | 50–100 m |

**Example command:**
```bash
python test_custom_dataset.py \
    --db-dir  ./my_dataset/database \
    --q-dir   ./my_dataset/queries \
    --db-csv  ./my_dataset/database.csv \
    --q-csv   ./my_dataset/queries.csv \
    --gt-mode gps \
    --dist-thr 25 \
    --dataset-name "MyCity-val"
```

---

### Mode B — Pre-built NumPy Ground Truth

Use this if you already have ground truth in a different form (e.g. place IDs, overlap scores) and want to convert it yourself.

**Required format:**

```python
import numpy as np

# gt is a 1-D numpy array of length = num_query_images
# Each element is an array of integer database indices (the positive matches)
gt = np.empty(num_queries, dtype=object)
gt[0] = np.array([4, 17, 23])   # query 0 matches database images 4, 17, and 23
gt[1] = np.array([2])            # query 1 only matches database image 2
gt[2] = np.array([])             # query 2 has no positives in the database

np.save('./my_dataset/ground_truth.npy', gt)
```

**Important:** index values refer to the **alphabetical position** of images inside `--db-dir`, starting at 0.

**Example command:**
```bash
python test_custom_dataset.py \
    --db-dir  ./my_dataset/database \
    --q-dir   ./my_dataset/queries \
    --gt-mode npy \
    --gt-npy  ./my_dataset/ground_truth.npy \
    --dataset-name "MyCity-val"
```

---

### Mode C — No Ground Truth

If you have no labels yet, the script still runs and prints the top-1 nearest database image for every query. Useful for sanity-checking that your pipeline works and the model retrieves sensible results.

```bash
python test_custom_dataset.py \
    --db-dir ./my_dataset/database \
    --q-dir  ./my_dataset/queries \
    --gt-mode none
```

---

## 4. Image Guidelines

| Property | Recommendation |
|----------|---------------|
| **Content** | Street-level / ground-level outdoor photos. The pretrained model was trained on Google Street View data (GSV-Cities). Performance degrades on aerial, indoor, or very different domains. |
| **Perspective** | Forward-facing is ideal. Panoramas should be cropped to perspective views first. |
| **Resolution** | Any resolution ≥ 320×320 px is fine. Lower resolution images will be upscaled. |
| **Channels** | RGB. Greyscale images are automatically converted to RGB. |
| **Overlap** | Database and query images should depict the **same physical locations** (possibly from different times, viewpoints, or conditions). If no database image is within `dist-thr` of a query, that query contributes 0 to Recall@K. |

---

## 5. Matching Database and CSV Rows

The most common mistake is a **mismatch between image order and CSV row order.**

The script sorts images alphabetically by filename. Your CSV rows must follow the same order.

**Quick way to generate a correctly-ordered CSV from image filenames:**

```python
import pandas as pd
from pathlib import Path

# Replace with your actual coordinates
image_dir = Path('./my_dataset/database')
images = sorted(image_dir.glob('*.jpg'))

# You must fill in the actual lat/lon for each image
coords = [
    # (lat, lon)
    (37.7749, -122.4194),  # img_001.jpg
    (37.7751, -122.4188),  # img_002.jpg
    # ...
]

df = pd.DataFrame({'filename': [p.name for p in images],
                   'lat': [c[0] for c in coords],
                   'lon': [c[1] for c in coords]})
df.to_csv('./my_dataset/database.csv', index=False)
```

---

## 6. Full Argument Reference

```
python test_custom_dataset.py --help

  --ckpt              Path to the .ckpt weights file
                      (default: ./LOGS/resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt)
  --db-dir            Directory of database/reference images  [required]
  --q-dir             Directory of query images               [required]
  --gt-mode           gps | npy | none                        (default: none)

  GPS mode:
    --db-csv          CSV with lat,lon for each database image
    --q-csv           CSV with lat,lon for each query image
    --dist-thr        Positive match radius in metres         (default: 25)

  NPY mode:
    --gt-npy          Path to .npy ground truth file

  Inference options:
    --image-size H W  Resize resolution                       (default: 320 320)
    --batch-size      Batch size                              (default: 32)
    --num-workers     DataLoader workers                      (default: 4)
    --device          cuda or cpu                             (auto-detected)
    --k-values        Recall@K values to report              (default: 1 5 10 15 20 50 100)
    --dataset-name    Label shown in results table            (default: CustomDataset)
    --save-descriptors PATH   Save descriptor matrix as .npy (optional)
```

---

## 7. Interpreting Results

```
+--------------------------------------------------------------------+
|              Performances on MyCity-val                            |
+----------+-------+-------+-------+-------+-------+-------+--------+
| K        |   1   |   5   |  10   |  15   |  20   |  50   |  100   |
+----------+-------+-------+-------+-------+-------+-------+--------+
| Recall@K | 72.30 | 89.10 | 92.40 | 93.80 | 94.50 | 96.20 | 97.10  |
+----------+-------+-------+-------+-------+-------+-------+--------+
```

- **Recall@1** — how often the single best match is correct. The strictest metric.  
- **Recall@5** — how often a correct match appears in the top 5. Standard for VPR benchmarks.  
- Values are percentages (0–100).
- Published MixVPR results on Pittsburgh30k-val: R@1 ≈ 94.9%, R@5 ≈ 98.4%.

**If Recall@K is unexpectedly low**, consider:
1. **Domain mismatch** — your images look very different from GSV training data.
2. **Threshold too tight** — try increasing `--dist-thr`.
3. **Image ordering bug** — verify CSV rows align with alphabetical image order.
4. **Image quality** — blurry, night, or heavily occluded images hurt retrieval.

---

## 8. Indoor Datasets from Video Footage

This section covers the full workflow for using video recordings of an indoor environment — no GPS available.

---

### 8.1 Core Strategy

With video you have two options depending on how many recordings you have:

| Scenario | Database | Queries |
|----------|----------|---------|
| **Two separate traversals** of the same space (recommended) | All frames from traversal 1 | All frames from traversal 2 |
| **Single traversal** | Every Nth frame | Every Mth frame (different stride, interleaved) |

The two-traversal setup is strongly preferred because database and queries come from genuinely different viewpoints/times, which makes the benchmark realistic. If you only have one video, the interleaved approach is an acceptable substitute.

---

### 8.2 Extracting Frames from Video

Use `ffmpeg` (recommended — fast, no Python needed) or the provided Python snippet.

**Option A — ffmpeg (1 frame per second):**
```bash
# Database video → 1 fps, zero-padded filenames so alphabetical = temporal order
ffmpeg -i traversal1.mp4 -vf fps=1 ./my_dataset/database/frame_%06d.jpg

# Query video → same fps
ffmpeg -i traversal2.mp4 -vf fps=1 ./my_dataset/queries/frame_%06d.jpg
```

Change `fps=1` to e.g. `fps=2` or `fps=0.5` to extract more or fewer frames. For a slowly moving camera in a building, 1–2 fps is usually sufficient.

**Option B — Python / OpenCV (every Nth frame):**
```python
import cv2
from pathlib import Path

def extract_frames(video_path: str, out_dir: str, every_n_frames: int = 15):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_idx, saved = 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % every_n_frames == 0:
            out_path = f"{out_dir}/frame_{saved:06d}.jpg"
            cv2.imwrite(out_path, frame)
            saved += 1
        frame_idx += 1
    cap.release()
    print(f"Saved {saved} frames to {out_dir}")

extract_frames('traversal1.mp4', './my_dataset/database', every_n_frames=15)
extract_frames('traversal2.mp4', './my_dataset/queries',  every_n_frames=15)
```

**Frame extraction tips:**
- Name frames with zero-padded numbers (`frame_000001.jpg`) so alphabetical order equals temporal order — this matters for ground truth.
- Skip the first and last few seconds of video if they contain the camera being picked up/put down.
- Avoid extracting at very high fps — consecutive frames are near-identical and bloat the database without adding information. 1 fps or every 10–30 frames at typical recording fps is a good starting point.
- If the video has severe motion blur (fast panning), increase `every_n_frames` to only keep sharper frames, or add a blur-detection filter (see 8.4).

---

### 8.3 Creating Ground Truth Without GPS

Since there is no GPS, you have two practical approaches.

#### Approach 1 — Frame-Index Proximity (automatic, fast)

The key insight: if two videos traverse the same route at roughly the same speed, **nearby frame indices correspond to nearby physical locations**.

```python
import numpy as np

# Number of database and query frames you extracted
num_db = 500    # len of database/ folder
num_q  = 480    # len of queries/ folder

# How many frames on either side counts as a "match"
# e.g. if 1 fps and you move ~0.5 m/s, a window of ±10 frames ≈ ±10 m radius
FRAME_WINDOW = 10

gt = np.empty(num_q, dtype=object)
for q_idx in range(num_q):
    # Map query index to corresponding database index
    # If the two videos have different lengths, scale proportionally
    db_center = int(round(q_idx * num_db / num_q))
    low  = max(0, db_center - FRAME_WINDOW)
    high = min(num_db - 1, db_center + FRAME_WINDOW)
    gt[q_idx] = np.arange(low, high + 1)

np.save('./my_dataset/ground_truth.npy', gt)
print(f"GT saved. Each query has ~{2*FRAME_WINDOW+1} positives on average.")
```

Then run:
```bash
python test_custom_dataset.py \
    --db-dir  ./my_dataset/database \
    --q-dir   ./my_dataset/queries \
    --gt-mode npy \
    --gt-npy  ./my_dataset/ground_truth.npy \
    --dataset-name "IndoorDataset"
```

**When this works well:** both videos traverse the same path in the same direction at similar speed.  
**When it breaks:** if the camera speed varies a lot between traversals, or the query video follows a different route order, you need Approach 2.

#### Approach 2 — Manual / Semi-Manual Labelling (ground truth quality)

For a reliable benchmark, manually annotate a subset of query frames with their correct database matches. This is the most accurate but most time-consuming.

Practical workflow:
1. Extract frames as above.
2. Open both frame folders side-by-side in a file manager or use a simple viewer script.
3. For each query frame, record the indices of visually matching database frames.
4. Build the `.npy` file using the same format as Approach 1.

For a first sanity check, ~50–100 manually labelled query frames are enough to get a meaningful Recall@K number.

#### Approach 3 — Descriptor Similarity Bootstrap (no labels at all)

If you just want descriptors and nearest-neighbour rankings without any evaluation metric, use `--gt-mode none`. The script will print the top-1 database match for every query frame, and you can visually inspect whether the matches are correct.

---

### 8.4 Filtering Blurry Frames (optional but recommended)

Indoor video often contains motion blur during fast panning. Blurry frames harm both database quality and query matching.

```python
import cv2
import numpy as np
from pathlib import Path

def is_sharp(img_path: str, threshold: float = 100.0) -> bool:
    """Return True if the image is sharp enough (Laplacian variance > threshold)."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    return cv2.Laplacian(img, cv2.CV_64F).var() > threshold

def filter_blurry(folder: str, threshold: float = 100.0):
    paths = sorted(Path(folder).glob('*.jpg'))
    removed = 0
    for p in paths:
        if not is_sharp(str(p), threshold):
            p.unlink()   # delete the blurry frame
            removed += 1
    print(f"Removed {removed}/{len(paths)} blurry frames from '{folder}'")

filter_blurry('./my_dataset/database', threshold=100.0)
filter_blurry('./my_dataset/queries',  threshold=100.0)
```

Run this **before** building ground truth, since deleting frames changes the frame count and index mapping.

---

### 8.5 Domain Mismatch Warning

The pretrained MixVPR checkpoint was trained exclusively on **outdoor Google Street View** images. Indoor environments look fundamentally different (lighting, textures, scale, lack of sky). As a result:

- Absolute Recall@K numbers will be lower than the published Pittsburgh benchmarks — this is expected and not a bug.
- The model can still produce useful descriptors and rankings for indoor scenes; it just won't be at peak performance.
- If you need strong indoor performance, consider fine-tuning the model on your indoor data using the GSV-Cities training pipeline with your own dataset substituted in.

---

### 8.6 Recommended File Layout for an Indoor Video Dataset

```
my_dataset/
├── database/
│   ├── frame_000000.jpg
│   ├── frame_000001.jpg
│   └── ...                     ← frames from traversal 1
├── queries/
│   ├── frame_000000.jpg
│   ├── frame_000001.jpg
│   └── ...                     ← frames from traversal 2
└── ground_truth.npy             ← built by the frame-index script above
```

Quick-start command once everything is in place:
```bash
python test_custom_dataset.py \
    --db-dir  ./my_dataset/database \
    --q-dir   ./my_dataset/queries \
    --gt-mode npy \
    --gt-npy  ./my_dataset/ground_truth.npy \
    --dataset-name "MyIndoorSpace" \
    --k-values 1 5 10 20
```
