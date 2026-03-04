# Testing Checklist - Print & Follow

## ☐ Pre-Flight Check (5 minutes)

```
☐ 1. Check you have at least 2GB free disk space
☐ 2. Verify Python 3.7+ installed: python --version
☐ 3. Verify GPU (optional): python -c "import torch; print(torch.cuda.is_available())"
☐ 4. Terminal in MixVPR directory: pwd should end with /MixVPR
```

---

## ☐ Installation (2 minutes)

```
☐ 5. Install requirements: pip install -r requirements.txt
☐ 6. Verify installation: python -c "import torch; import faiss; print('OK')"
```

---

## ☐ Download Weights (5 minutes)

```
☐ 7. Create directory: mkdir -p LOGS

☐ 8. Download pretrained weights from README 
     Link: https://drive.google.com/file/d/1vuz3PvnR7vxnDDLQrdHJaOA04SQrtk5L/view?usp=share_link
     
     File should be named: resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt
     
☐ 9. Save file to: ./LOGS/resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt

☐ 10. Verify download:
      ls -lh LOGS/resnet50_MixVPR_4096*
      (Should show ~103MB file)
```

---

## ☐ Download Dataset (Optional but Recommended - 10 min)

```
☐ 11. Create dataset directory: mkdir -p datasets/Pittsburgh

☐ 12. Download Pittsburgh30k from official source:
      https://www.cv-foundation.org/openaccess/content_cvpr_2013/html/Torii_Visual_Place_Recognition_2013_CVPR_paper.html
      
      OR use wget if available:
      cd datasets/Pittsburgh
      wget [download-link-from-paper]
      unzip Pittsburgh.zip
      cd ../..

☐ 13. Verify dataset structure:
      ls -la ../datasets/Pittsburgh/
      Should show: datasets/, queries_real/ directories
```

---

## ☐ Update Code Paths (1 minute)

```
☐ 14. Open: dataloaders/PittsburgDataset.py

☐ 15. Find line ~11:
      root_dir = '../datasets/Pittsburgh/'
      
☐ 16. Replace with your actual path:
      root_dir = '/absolute/path/to/Pittsburgh'
      
      Example (macOS):
      root_dir = '/Volumes/T9/DevSpace/Github/MixVPR/datasets/Pittsburgh'
      
      Example (Linux):
      root_dir = '/home/user/datasets/Pittsburgh'

☐ 17. Save file (Ctrl+S)
```

---

## ☐ Run Tests (1-10 minutes depending on hardware)

```
Option A: Test Everything (RECOMMENDED)
─────────────────────────────────────────
☐ 18. Run: python test_pretrained.py
☐ 19. Wait for completion (GPU: 30 sec, CPU: 10 min)
☐ 20. Check for "Test Complete!" message
☐ 21. Verify Recall values appear (should be 90%+ for R@1)

Option B: Test Components (Debug mode)
──────────────────────────────────────
☐ 18a. Run: python inspect_step1.py  (Check model loads)
☐ 18b. Run: python inspect_step2.py  (Check dataset loads)
☐ 18c. Run: python inspect_step3.py  (Check feature extraction)
☐ 18d. Run: python inspect_step4.py  (Check similarity search)
☐ 18e. If all pass, run: python test_pretrained.py
```

---

## ☐ Verify Results

```
☐ 22. Expected output format:
      ┌─────────────┬────┬────┬────┬────┬────┬────┐
      │ Recall@K    │ @1 │ @5 │@10 │@15 │@20 │@50 │
      ├─────────────┼────┼────┼────┼────┼────┼────┤
      │ Pittsburgh..│ 92 │ 96 │ 96 │ 97 │ 98 │ 99 │
      └─────────────┴────┴────┴────┴────┴────┴────┘

☐ 23. Success criteria:
      ✓ R@1 should be 91-92%
      ✓ R@5 should be 95-96%
      ✓ R@10 should be 96-97%
      
      (These match paper's reported values)

☐ 24. If different, check:
      ☐ Wrong dataset path?
      ☐ Different checkpoint file?
      ☐ Corrupted download?
```

---

## ☐ Next Steps

```
☐ 25. Read PIPELINE_EXPLAINED.md to understand what happened

☐ 26. (Optional) Read all documentation:
      - QUICKSTART.md (detailed guide)
      - COMMANDS.md (command reference)
      - STEP_BY_STEP_INSPECTION.md (debugging)

☐ 27. (Optional) Test on your own images using demo.py

☐ 28. (Optional) Fine-tune on your dataset (see main.py)
```

---

## 🚨 Troubleshooting Quick Fixes

```
❌ "Checkpoint not found"
   ✓ Create LOGS directory: mkdir -p LOGS
   ✓ Check file exists: ls LOGS/resnet50*
   ✓ File should be 103MB, not empty

❌ "Dataset not found"
   ✓ Create directory: mkdir -p datasets/Pittsburgh
   ✓ Update path in dataloaders/PittsburgDataset.py
   ✓ Check structure: ls datasets/Pittsburgh/datasets/

❌ "CUDA out of memory"
   ✓ Edit test_pretrained.py line 89
   ✓ Change: batch_size=32 → batch_size=8
   ✓ Re-run: python test_pretrained.py

❌ "Very slow (10+ minutes)"
   ✓ You're on CPU - this is normal!
   ✓ Will finish eventually
   ✓ GPU is 10-20x faster if available

❌ "Import errors"
   ✓ Install requirements: pip install -r requirements.txt
   ✓ Check Python version: python --version
```

---

## ⏱️ Time Tracking

```
Phase 1: Setup
  Pre-flight:        5 min
  Installation:      2 min
  Subtotal:          7 min

Phase 2: Download
  Weights:           5 min
  Dataset:          10 min  (skip if no dataset)
  Subtotal:         15 min

Phase 3: Configuration
  Update paths:      1 min
  Subtotal:          1 min

Phase 4: Testing
  Run test:        0.5-10 min (depends on GPU)
  Subtotal:          1-10 min

═════════════════════════════════════════
TOTAL:           24-33 minutes
(Mostly download time, not your computer's fault)
```

---

## 📋 Success Checklist

```
At the end, you should have:

✓ Pretrained weights loaded successfully
✓ VPR model initialized with MixVPR aggregator
✓ Pittsburgh30k validation dataset loaded
✓ Descriptors extracted from 6,871 images
✓ Recall@K metrics computed and printed
✓ Results matching paper's reported values (~91% R@1)

This means you understand:
✓ How to load pretrained VPR models
✓ How MixVPR architecture works
✓ How place recognition evaluation works
✓ How to extract and compare image descriptors
✓ How the complete pipeline flows end-to-end
```

---

## 🎉 You're Done!

You have successfully:
1. Set up the MixVPR codebase
2. Loaded pretrained weights
3. Tested on a full dataset
4. Verified the implementation works

You understand the complete pipeline and can now:
- Test on your own images
- Fine-tune on your own data
- Modify architectures
- Build your own VPR system

---

**Print this page and check off items as you go!**
If you get stuck, find the error in the "Troubleshooting" section above.
