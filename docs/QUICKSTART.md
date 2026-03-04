# Quick Start Guide: Testing Pretrained MixVPR

This guide shows the **fastest way** to test MixVPR with pretrained weights on Pittsburgh30k-val dataset.

---

## Prerequisites

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

**Key packages needed:**
- torch >= 1.13.1
- torchvision >= 0.14.1
- pytorch-lightning >= 1.8.3
- faiss-cpu (or faiss-gpu)

### 2. Verify GPU (Optional but Recommended)
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
- If output is `True`: GPU is available (inference ~30 seconds)
- If output is `False`: Will use CPU (inference ~5-10 minutes)

---

## Step-by-Step Instructions

### Step 1: Download Pretrained Weights (5 min)

Go to the **README.md** and find the model weights table. Download one of these:

**Option A: ResNet50 with 4096 dimensions (RECOMMENDED)**
- Download: [Link in README](https://drive.google.com/file/d/1vuz3PvnR7vxnDDLQrdHJaOA04SQrtk5L/view?usp=share_link)
- File: `resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt`

**Option B: ResNet50 with 512 dimensions**
- Download: [Link in README](https://drive.google.com/file/d/1khiTUNzZhfV2UUupZoIsPIbsMRBYVDqj/view?usp=share_link)

Create a directory for checkpoints:
```bash
mkdir -p LOGS
cd LOGS
# Paste downloaded .ckpt file here
cd ..
```

✓ Checkpoint should be at: `./LOGS/resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt`

---

### Step 2: Download Pittsburgh30k-val Dataset (10-20 min)

The Pittsburgh dataset is not in this repo, but it's needed for validation.

**Option A: Automated (if you have enough disk space ~500MB)**
```bash
# Create dataset directory
mkdir -p datasets/Pittsburgh

# Download from official source
# Dataset: https://www.cv-foundation.org/openaccess/content_cvpr_2013/html/Torii_Visual_Place_Recognition_2013_CVPR_paper.html

# Manually download or use wget:
cd datasets/Pittsburgh
wget https://www.dropbox.com/s/mxmva38sq60nwym/Pittsburgh.zip
unzip Pittsburgh.zip
cd ../..
```

**Option B: Quick Test without Real Dataset**
Skip this step for now. The script will tell you what's missing.

---

### Step 3: Update Dataset Path (2 min)

Edit [dataloaders/PittsburgDataset.py](dataloaders/PittsburgDataset.py) 

**Find this line (~line 11):**
```python
root_dir = '../datasets/Pittsburgh/'
```

**Replace with your actual path:**
```python
root_dir = '/path/to/your/Pittsburgh/datasets'
# or absolute path
root_dir = '/Volumes/T9/DevSpace/Github/datasets/Pittsburgh'
```

Make sure the directory structure looks like:
```
your_Pittsburgh_path/
├── datasets/
│   ├── pitts30k_val.mat
│   ├── pitts30k_test.mat
│   └── pitts250k_val.mat
└── queries_real/
    └── [query image files]
```

---

### Step 4: Run the Test Script (30 seconds - 10 minutes depending on GPU)

```bash
python test_pretrained.py
```

**Output will look like:**
```
================================================================================
Testing Pretrained MixVPR Model
================================================================================

[1/3] Initializing model...
✓ Loaded pretrained weights from ./LOGS/resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt

[2/3] Loading Pittsburgh30k-val dataset...
✓ Dataset loaded: 6871 images
  - Database images: 6605
  - Query images: 266

[3/3] Extracting descriptors from all images...
Inference: 100%|██████████| 215/215 [00:32<00:00,  6.60it/s]
✓ Extracted 6871 descriptors
  Descriptor shape: (6871, 4096)

[4/4] Computing Recall@K...

+---------------+-------+-------+-------+-------+-------+-------+-------+
| Recall@K      |     1 |     5 |    10 |    15 |    20 |    50 |   100 |
+---------------+-------+-------+-------+-------+-------+-------+-------+
| Pittsburgh... | 91.73 | 95.49 | 96.24 | 96.99 | 97.74 | 99.62 | 100.0 |
+---------------+-------+-------+-------+-------+-------+-------+-------+

================================================================================
Test Complete!
================================================================================
```

**Expected Results:**
- R@1: ~91-92%
- R@5: ~95-96%
- R@10: ~96-97%

---

### Step 5: Understanding the Output

**What these numbers mean:**
- **R@1 = 91.73%**: For 91.73% of query images, the correct match is in the top-1 results
- **R@5 = 95.49%**: For 95.49% of query images, the correct match is in the top-5 results
- **R@10 = 96.24%**: For 96.24% of query images, the correct match is in the top-10 results

**These compare VERY WELL with the paper's results** (R@1: 91.6%, R@5: 95.5%, R@10: 96.4%)

---

## What's Happening Under the Hood?

```
1. Load Pretrained Model
   └─ ResNet50 backbone (extracts features)
   └─ MixVPR aggregator (creates global descriptor)

2. Load Dataset
   └─ 6605 reference images from database
   └─ 266 query images
   └─ Each image preprocessed to 320×320, normalized

3. Forward Pass
   Query Image (320×320) 
   → ResNet50 backbone outputs: (1024, 20, 20) features
   → MixVPR aggregator outputs: 4096-dimensional descriptor
   
4. Similarity Search
   For each query descriptor:
   - Compute L2 distance to all reference descriptors
   - FAISS returns top-10 nearest neighbors
   - Check if ground truth is in top-K

5. Calculate Recall
   Recall@K = (# queries with match in top-K) / (total queries)
```

---

## Troubleshooting

### Error: "No such file or directory: 'pitts30k_val.mat'"
**Solution:** Update the path in `dataloaders/PittsburgDataset.py`
```python
root_dir = '/correct/path/to/Pittsburgh'  # Must point to the extracted dataset
```

### Error: "CUDA out of memory"
**Solution:** Reduce batch size in test_pretrained.py (line ~89)
```python
dataloader = DataLoader(
    val_dataset,
    batch_size=8,  # Reduce from 32
    shuffle=False
)
```

### Error: "module 'utils' has no attribute 'get_loss'"
**Solution:** Make sure you're running from the MixVPR root directory
```bash
cd /path/to/MixVPR
python test_pretrained.py
```

### Slow Performance (CPU taking 10+ minutes)
**Solution:** This is normal on CPU. Options:
1. Wait for completion (it will finish)
2. Test on GPU if available
3. Reduce dataset size for quick test

---

## Next Steps

Once this works, you can:

### Test on Your Own Images
See [demo.py](demo.py) for inference on custom image pairs

### Fine-tune on Your Own Dataset
Modify [main.py](main.py) to use your dataset and retrain

### Test on MSLS Dataset
Change validation set in [main.py](main.py) to 'msls_val'
```python
val_set_names=['msls_val']  # instead of 'pitts30k_val'
```

---

## Default Checkpoint Paths

If you want to use a different checkpoint location:
```bash
# Use custom path
python test_pretrained.py /path/to/your/checkpoint.ckpt

# Or modify this line in test_pretrained.py:
# ckpt_path = './LOGS/your_checkpoint.ckpt'
```

---

## Performance Expectations

| Hardware | Time | Notes |
|----------|------|-------|
| GPU (RTX 3090) | 30 seconds | Inference only |
| GPU (RTX 2080) | 60 seconds | Inference only |
| CPU (Intel i7) | 10 minutes | Will work but slow |
| M1/M2 Mac | 2-3 minutes | Using Metal acceleration |

---

## Summary

**Total Time from Scratch:**
- Download weights: 5 min
- Download dataset: 15 min
- Update path: 2 min
- Run test: 30 seconds - 10 minutes

**Total: 22-32 minutes** (mostly waiting for downloads)

You now understand the complete VPR pipeline! ✓
