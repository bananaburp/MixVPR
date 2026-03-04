# Quick Reference Card

## 🚀 Super Quick (Under 1 minute to get started)

```bash
# 1. Download pretrained weights
#    From: https://drive.google.com/file/d/1vuz3PvnR7vxnDDLQrdHJaOA04SQrtk5L/view?usp=share_link
#    Save to: ./LOGS/resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt

mkdir -p LOGS
# [paste downloaded file]

# 2. Update dataset path in dataloaders/PittsburgDataset.py line 11
# root_dir = '/your/absolute/path/to/Pittsburgh'

# 3. Run test
python test_pretrained.py

# ✓ Done! Should show Recall values in ~30 sec (GPU) or ~10 min (CPU)
```

---

## 📖 Where to Start?

| Goal | Read | Time |
|------|------|------|
| See it work NOW | [COMMANDS.md](COMMANDS.md) | 5 min |
| Guided walkthrough | [QUICKSTART.md](QUICKSTART.md) | 20 min |
| Deep understanding | [PIPELINE_EXPLAINED.md](PIPELINE_EXPLAINED.md) | 10 min |
| Debug issues | [STEP_BY_STEP_INSPECTION.md](STEP_BY_STEP_INSPECTION.md) | 30 min |
| Print & follow | [CHECKLIST.md](CHECKLIST.md) | 25 min |

---

## 🔑 Key Concepts

**Input → Processing → Output**

```
Image (320×320)
    ↓
[ResNet50] extracts features
    ↓
Features (1024, 20, 20)
    ↓
[MixVPR] mixes features with MLPs
    ↓
Descriptor (4096 dimensions)
    ↓
[FAISS] finds nearest neighbors
    ↓
Recall@K score computed
```

---

## 📊 Expected Results

```
Pittsburgh30k-val Benchmark:
Recall@1  = 91-92%  ✓
Recall@5  = 95-96%  ✓
Recall@10 = 96-97%  ✓
```

If you see these numbers → **You did it correctly!**

---

## ⚠️ Most Common Issues & Fixes

| Error | Fix |
|-------|-----|
| "Checkpoint not found" | Download it to `./LOGS/` |
| "Dataset not found" | Update path in `PittsburgDataset.py` line 11 |
| "CUDA out of memory" | Change `batch_size=32` to `batch_size=8` |
| "Very slow (10+ min)" | You're on CPU - normal! Or try GPU |
| "Import errors" | Run `pip install -r requirements.txt` |

---

## 📁 Essential Files

```
test_pretrained.py          ← Run this (main test script)
main.py                     ← VPR model definition
models/aggregators/mixvpr.py ← MixVPR architecture
utils/validation.py         ← Evaluation code
```

---

## 🎯 4 Steps to Victory

```
1. Download weights (5 min)
   └─ From README link to ./LOGS/

2. Download dataset (15 min)
   └─ Pittsburgh30k dataset structure

3. Update one path (1 min)
   └─ dataloaders/PittsburgDataset.py line 11

4. Run: python test_pretrained.py (30 sec - 10 min)
   └─ See Recall@K values printed
```

**Total: 22-31 minutes (mostly download time)**

---

## 💡 What You're Learning

After this completes, you understand:
✓ How to load pretrained VPR models
✓ How MixVPR aggregator works
✓ How place recognition is evaluated
✓ How to extract global image descriptors
✓ Complete VPR pipeline end-to-end

---

## 📞 Need Help?

1. **Quick answer**: Check the guide's FAQ section
2. **Stuck on setup**: Use [CHECKLIST.md](CHECKLIST.md)
3. **Understanding code**: Use [PIPELINE_EXPLAINED.md](PIPELINE_EXPLAINED.md)
4. **Debugging**: Run `python inspect_step1.py` through `step4.py`

---

## 🌟 Key Files to Read

```
INDEX.md                     ← You are here
QUICKSTART.md                ← Official guide
PIPELINE_EXPLAINED.md        ← How it works
test_pretrained.py           ← Test script
```

---

## ⏱️ Time Estimates

| Task | GPU | CPU |
|------|-----|-----|
| Download & setup | 20 min | 20 min |
| Full test run | 30 sec | 10 min |
| **Total** | **~21 min** | **~30 min** |

Get started now! → Pick a guide from above →→

---

**Print this page and keep it handy while testing!**
