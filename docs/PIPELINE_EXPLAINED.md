# MixVPR End-to-End Pipeline Visualization

## Visual Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     MixVPR TESTING PIPELINE                             │
└─────────────────────────────────────────────────────────────────────────┘

STEP 1: LOAD MODEL
═════════════════════════════════════════════════════════════════════════

    Load Pretrained Checkpoint
            ↓
    Initialize VPRModel (PyTorch Lightning Module)
            ↓
    ┌─ Backbone: ResNet50 (copy with layer 4 cropped)
    ├─ Aggregator: MixVPR (all-MLP feature mixer)
    └─ Device: GPU or CPU
            ↓
    Ready for inference


STEP 2: LOAD DATASET
═════════════════════════════════════════════════════════════════════════

    Pittsburgh30k-val Dataset
            ↓
    ┌─ Reference images (database): 6,605 images
    ├─ Query images: 266 images
    └─ Ground truth: Which queries match which references
            ↓
    For each image:
    ├─ Load image (JPG)
    ├─ Resize to 320×320
    ├─ Normalize (ImageNet mean/std)
    └─ Convert to tensor
            ↓
    Create DataLoader (batch_size=32)


STEP 3: EXTRACT DESCRIPTORS (Main Computation)
═════════════════════════════════════════════════════════════════════════

    For each batch of images:
    
    Image Batch [32, 3, 320, 320]
            ↓
            ▼─────────────────────────────────────────────────┐
            │                                                 │
            │          BACKBONE: ResNet50                     │
            │      Extracts multi-scale features              │
            │                                                 │
            │  [32, 1024, 20, 20]  ← Feature maps             │
            │      (32 images, 1024 channels, 20×20 spatial)  │
            │                                                 │
            └─────────────────────────────────────────────────┘
            ↓
            
            ▼─────────────────────────────────────────────────┐
            │                                                 │
            │          AGGREGATOR: MixVPR                     │
            │   Fuses spatial and channel information          │
            │                                                 │
            │  Flatten: [B, 1024, 400]                        │
            │    (400 = 20×20 spatial positions)              │
            │          ↓                                      │
            │    Mix 4 layers (feature mixing)                │
            │    Each layer: LayerNorm + MLP + Residual        │
            │          ↓                                      │
            │    Channel projection: 1024 → 1024              │
            │          ↓                                      │
            │    Spatial projection: 400 → 4 rows             │
            │          ↓                                      │
            │    L2 normalize: [B, 4096]                      │
            │                                                 │
            │  [32, 4096]  ← Global descriptors               │
            │     (32 images, each described by 4096 numbers) │
            │                                                 │
            └─────────────────────────────────────────────────┘
            ↓

    Collect all descriptors:
    ┌─ Reference descriptors: [6605, 4096]  (database)
    └─ Query descriptors: [266, 4096]       (queries)


STEP 4: SIMILARITY SEARCH (Using FAISS)
═════════════════════════════════════════════════════════════════════════

    For each query descriptor [1, 4096]:
    
        Query [4096]
            ↓
        Compute distances to ALL references
        using FAISS (exact L2 distance)
            ↓
        Get top-10 nearest neighbors by distance
            ↓
    
        Top-10 matches ranked by similarity
        ┌─ Rank 1: Reference image 456 (distance: 0.234)
        ├─ Rank 2: Reference image 789 (distance: 0.456)
        ├─ Rank 3: Reference image 234 (distance: 0.567)
        ├─ ...
        └─ Rank 10: Reference image 567 (distance: 1.234)


STEP 5: EVALUATE AGAINST GROUND TRUTH
═════════════════════════════════════════════════════════════════════════

    For each query, check: is correct reference in top-K?

    Example Query #1:
    ├─ Ground truth: Reference 456 is correct match
    ├─ Top-1 prediction: Reference 456 ✓
    └─ → MATCH! (contributes to R@1, R@5, R@10, etc.)

    Example Query #2:
    ├─ Ground truth: Reference 789 is correct match
    ├─ Top-1 prediction: Reference 456 ✗
    ├─ Top-5 predictions: [456, 234, 567, 789, 345] ✓
    └─ → MATCH in Top-5! (contributes to R@5, R@10, etc.)


STEP 6: CALCULATE RECALL@K
═════════════════════════════════════════════════════════════════════════

    Recall@K = (# matching queries) / (total queries)

    Results:
    ┌─ R@1  = 91.73% : 244 out of 266 queries match in top-1
    ├─ R@5  = 95.49% : 254 out of 266 queries match in top-5
    ├─ R@10 = 96.24% : 256 out of 266 queries match in top-10
    └─ ...

```

---

## Data Flow Through MixVPR Aggregator

```
INPUT: Feature maps from ResNet50
[32, 1024, 20, 20]

       ↓ Flatten spatial dimensions
       ↓
[32, 1024, 400]  (400 = 20×20)

       ↓ Apply feature mixer (4 layers)
       ↓
Each layer:
├─ LayerNorm: normalize over 400 spatial dims
├─ MLP: 400 → 400 (mix information across space)
└─ Residual: x = x + mlp(x)

After mixing:
[32, 1024, 400]

       ↓ Permute: [32, 400, 1024]
       ↓ Channel projection: 1024 → 1024
       ↓ Permute: [32, 1024, 400]
       ↓ Row projection: 400 → 4
       ↓
[32, 1024, 4]

       ↓ L2 normalize and flatten: 1024 × 4 = 4096
       ↓
[32, 4096]  ← Final descriptor!

```

---

## Why This Works

1. **Feature Mixing**: Spatial positions share information through MLPs
   - Position (i,j) learns what other positions are important
   - Creates global context without attention

2. **Channel Projection**: Reduces redundancy while preserving discriminative info
   - 1024×400 = 409,600 values → 4096 values
   - 100× compression without significant info loss

3. **Row Projection**: Further spatial aggregation
   - 400 spatial positions → 4 abstract regions
   - Creates spatially-aware descriptor

4. **L2 Normalization**: Makes descriptor suitable for cosine similarity
   - All descriptors normalized to unit sphere
   - Easier for nearest neighbor search

---

## Code Correspondence

**main.py: VPRModel**
- Defines training logic
- Orchestrates backbone + aggregator
- We use mostly for validation in our test

**models/backbones/resnet.py: ResNet**
- Extracts feature maps
- Returns [B, 1024, 20, 20] when cropped at layer 4

**models/aggregators/mixvpr.py: MixVPR**
- The core innovation
- Mixes features across space
- Returns [B, 4096] descriptors

**utils/validation.py: get_validation_recalls()**
- Uses FAISS to search
- Computes Recall@K
- Prints results

---

## Key Numbers to Remember

| Property | Value |
|----------|-------|
| Input image size | 320 × 320 × 3 |
| ResNet output | 1024 × 20 × 20 |
| Feature space | 1024 channels × 400 spatial |
| MixVPR output | 4096 dimensions |
| Pittsburgh30k | 6,605 reference + 266 query |
| Expected R@1 | ~91-92% |
| Inference time (GPU) | ~30 seconds |
| Inference time (CPU) | ~10 minutes |

---

## What test_pretrained.py Does

1. **Load** → Initialize model + load checkpoint
2. **Process** → Forward all images through model
3. **Extract** → Collect 4096-dimensional descriptors
4. **Compare** → Use FAISS to find top-10 similar references for each query
5. **Evaluate** → Check how many queries have correct match in top-K

---

Congratulations! You now understand the complete end-to-end pipeline. 🎉
