# Ultra-Quick Command Reference

Copy-paste these commands in order. 

## 1. Install Dependencies
```bash
pip install -r requirements.txt
```

## 2. Create Checkpoint Directory
```bash
mkdir -p LOGS
```

## 3. Download Pretrained Weights

**Download from:** https://drive.google.com/file/d/1vuz3PvnR7vxnDDLQrdHJaOA04SQrtk5L/view?usp=share_link

**Save to:** `./LOGS/resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt`

## 4. Download Pittsburgh Dataset (Optional but Recommended)

```bash
mkdir -p datasets/Pittsburgh
cd datasets/Pittsburgh
# Manually download from: https://www.cv-foundation.org/openaccess/content_cvpr_2013/html/Torii_Visual_Place_Recognition_2013_CVPR_paper.html
# Or use wget if available
cd ../..
```

## 5. Update Dataset Path

Edit `dataloaders/PittsburgDataset.py` line 11:
```python
root_dir = '/absolute/path/to/Pittsburgh'
```

## 6. Run Test
```bash
python test_pretrained.py
```

---

## Expected Output

```
Test Complete!
========================================================================
| Recall@K      |     1 |     5 |    10 |    15 |    20 |    50 |   100 |
+---------------+-------+-------+-------+-------+-------+-------+-------+
| Pittsburgh... | 91.73 | 95.49 | 96.24 | 96.99 | 97.74 | 99.62 | 100.0 |
+-----------────+-------+-------+-------+-------+-------+-------+-------+
```

---

## That's it!

You now have a working VPR system. The descriptor vector is 4096-dimensional and created by the MixVPR aggregator.

See `QUICKSTART.md` for detailed instructions and troubleshooting.
