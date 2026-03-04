# Hilti Dataset Recall Evaluation - Sanity Check Report

## Executive Summary

Your MixVPR recall evaluation script has **3 critical issues** that affect the accuracy of your recall metrics. The data setup is correct, but the recall calculation has problems with empty GT entries and uses the wrong distance metric for VPR.

---

## Data Setup Verification ✅

### Your Dataset Structure
```
floor_1_2025-05-05_run_1_cam0/
├── db/images/          175 images (10008508529000.jpg ... etc)
├── query/images/        76 images (10096096005000.jpg ... etc)
└── gt_positives.npy     76 entries (one per query)
```

### Data Verification Results
- ✅ **DB Images**: 175 images, sorted by nanosecond timestamp
- ✅ **Query Images**: 76 images, sorted by nanosecond timestamp  
- ✅ **GT Format**: Array of 76 entries, each containing DB indices
- ✅ **GT Index Range**: 0-174 (matches 175 DB images)
- ✅ **Image Sorting**: Correctly sorted by integer filename

### Sample GT Entries
```python
Query 0: [168, 169, 170, 171, 172, 173, 174]  # 7 positive matches
Query 1: [169, 170, 171, 172, 173, 174]       # 6 positive matches
Query 2: [170, 171, 172, 173, 174]            # 5 positive matches
...
Query 73: []  # ⚠️ Empty!
Query 74: []  # ⚠️ Empty!
Query 75: []  # ⚠️ Empty!
```

---

## Critical Issues Found 🚨

### Issue #1: Empty GT Entries Deflate Recall Scores

**Problem**: Queries 73, 74, 75 have empty ground truth arrays `[]`.

**Impact**: 
- These queries are counted in the recall denominator (76 total queries)
- But they can NEVER contribute to correct matches
- This artificially reduces recall by ~4% (3/76)

**Why this happens**:
```python
# In utils/validation.py line 29-34
for q_idx, pred in enumerate(predictions):
    for i, n in enumerate(k_values):
        if np.any(np.in1d(pred[:n], gt[q_idx])):  # Always False for empty GT!
            correct_at_k[i:] += 1
            break

# correct_at_k divided by len(predictions) = 76
# But only 73 queries have valid GT!
```

**Fix**: Filter out queries with empty GT before recall calculation.

---

### Issue #2: Wrong Distance Metric for VPR

**Problem**: Uses `faiss.IndexFlatL2` (L2 distance) on **unnormalized** descriptors.

**Why it's wrong**:
- VPR systems use **cosine similarity** to compare place descriptors
- Cosine similarity measures direction, not magnitude
- L2 distance on raw vectors can be dominated by magnitude differences
- This leads to suboptimal retrieval performance

**What you need**:
```python
# Normalize descriptors
db_norm = db / np.linalg.norm(db, axis=1, keepdims=True)
q_norm = q / np.linalg.norm(q, axis=1, keepdims=True)

# Use inner product index (equivalent to cosine similarity on normalized vectors)
faiss_index = faiss.IndexFlatIP(embed_size)  # IP = Inner Product
```

---

### Issue #3: No Descriptor Normalization

**Problem**: Descriptors extracted from MixVPR are not normalized before FAISS search.

**Impact**: 
- Raw descriptor magnitudes vary
- L2 distance comparisons are not meaningful
- Recall scores are lower than they should be

**Fix**: Add L2 normalization after descriptor extraction.

---

## Proposed Fixes

I've created two fixed versions:

### 1. Fixed Script: `run_hilti_recall_fixed.py`

Key improvements:
- ✅ Detects and reports empty GT entries
- ✅ Filters out queries with empty GT
- ✅ Validates GT indices don't exceed DB size
- ✅ Normalizes descriptors for cosine similarity
- ✅ Compares normalized vs unnormalized results

### 2. Fixed Validation: `utils/validation_fixed.py`

Key improvements:
- ✅ Supports both L2 and cosine similarity
- ✅ Validates descriptor normalization
- ✅ Better error messages and warnings
- ✅ Default use of cosine similarity for VPR

---

## How to Use the Fixed Version

```bash
# Using the fixed script
python src/run_hilti_recall_fixed.py \
    --eval_root "/Volumes/T9/DevSpace/Github/hilti-trimble-slam-challenge-2026/challenge_tools_ros/vpr/eval/floor_1_2025-05-05_run_1_cam0" \
    --ckpt "./LOGS/resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt" \
    --device mps \
    --batch 16
```

Expected output:
```
⚠️  WARNING: 3 queries have empty GT positives: [73, 74, 75]
    These queries will be excluded from recall calculation to avoid deflating metrics.
    Filtering: 73 queries with GT / 76 total queries

Normalizing descriptors for cosine similarity...
      Normalized shapes: db=(175, 4096), query=(73, 4096)

Calculating recalls...
+------------------------------------------+
|  Performances on floor_1_2025-05-05_...  |
+----------+------+------+------+
| K        | 1    | 5    | 10   |
+----------+------+------+------+
| Recall@K | XX.X | XX.X | XX.X |
+----------+------+------+------+
```

---

## Expected Recall Improvements

After applying fixes, you should see:
1. **~4% higher recall** from excluding empty GT queries
2. **Better absolute recall values** from using cosine similarity
3. **More stable results** from normalized descriptors

---

## Additional Recommendations

### 1. Consider Using Multiple Ground Truth Distances

Your GT appears to be spatial-based (nearby DB images are positives). Consider:
- Using different distance thresholds (e.g., 5m, 10m, 25m)
- Reporting recall at multiple thresholds
- Analyzing performance vs. distance tolerance

### 2. Analyze the Empty GT Queries

Investigate why queries 73-75 have no positives:
- Are they outside the mapped area?
- Are they in a different environment?
- Should they be excluded from the test set?

### 3. Visualize Top-K Retrievals

Add visualization to inspect failures:
```python
from demo import visualize
top_k_matches = calculate_top_k(q_desc_norm, db_desc_norm, top_k=10)
visualize(top_k_matches, q_ds, db_ds, num_samples=20, out_file='matches.png')
```

### 4. Cache Separately for Normalized/Unnormalized

Update cache keys to distinguish between normalized and unnormalized descriptors:
```python
cache_path = f'./LOGS/global_descriptors_{dataset_tag}_{split}_{"norm" if normalized else "raw"}.npy'
```

---

## Summary

✅ **Data setup is correct** - your GT file format and image organization are proper

🚨 **Script has 3 critical bugs** - empty GT handling, wrong distance metric, missing normalization

✅ **Fixes provided** - use `run_hilti_recall_fixed.py` for accurate evaluation

📈 **Expected result** - higher and more accurate recall scores after fixes

---

## Quick Comparison

| Metric | Original Script | Fixed Script |
|--------|----------------|--------------|
| Query count | 76 | 73 (filtered) |
| Distance metric | L2 (wrong) | Cosine (correct) |
| Normalization | ❌ No | ✅ Yes |
| Empty GT handling | ❌ Counted | ✅ Filtered |
| Recall accuracy | ⚠️ Deflated | ✅ Accurate |

