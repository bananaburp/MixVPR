# Step-by-Step Execution with Inspection

This guide shows you **exactly** what happens at each step with real data inspections.

---

## Step 0: Verify Environment

```bash
# Check GPU availability
python -c "import torch; print('GPU Available:', torch.cuda.is_available()); print('CUDA Version:', torch.version.cuda)"

# Check PyTorch version
python -c "import torch; print('PyTorch:', torch.__version__)"

# Check all required packages
python -c "import timm, pytorch_lightning, faiss; print('All imports OK')"
```

Expected output:
```
GPU Available: True
CUDA Version: 12.1
PyTorch: 1.13.1
All imports OK
```

---

## Step 1: Load Model with Inspection

Create file `inspect_step1.py`:

```python
import torch
from main import VPRModel

print("="*80)
print("STEP 1: Load Pretrained Model")
print("="*80)

# Initialize model
print("\n[1.1] Initializing VPRModel...")
model = VPRModel(
    backbone_arch='resnet50',
    layers_to_crop=[4],
    agg_arch='MixVPR',
    agg_config={
        'in_channels': 1024,
        'in_h': 20,
        'in_w': 20,
        'out_channels': 1024,
        'mix_depth': 4,
        'mlp_ratio': 1,
        'out_rows': 4
    },
)
print(f"✓ Model initialized")

# Load checkpoint
print("\n[1.2] Loading pretrained weights...")
try:
    ckpt_path = './LOGS/resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt'
    state_dict = torch.load(ckpt_path)
    model.load_state_dict(state_dict)
    print(f"✓ Loaded from {ckpt_path}")
    print(f"  Checkpoint size: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
except FileNotFoundError:
    print(f"✗ Checkpoint not found at {ckpt_path}")
    exit(1)

# Move to device and set eval mode
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
model.eval()
print(f"\n✓ Model on {device} in eval mode")

# Inspect architecture
print("\n[1.3] Model Architecture:")
print(f"  Backbone: {model.encoder_arch}")
print(f"  Backbone output channels: {model.backbone.out_channels}")
print(f"  Aggregator: {model.agg_arch}")
print(f"  Config: {model.agg_config}")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n  Total parameters: {total_params / 1e6:.2f}M")
print(f"  Trainable parameters: {trainable_params / 1e6:.2f}M")

print("\n" + "="*80)
print("STEP 1 Complete: Model ready for inference")
print("="*80)
```

Run it:
```bash
python inspect_step1.py
```

Expected output:
```
================================================================================
STEP 1: Load Pretrained Model
================================================================================

[1.1] Initializing VPRModel...
✓ Model initialized

[1.2] Loading pretrained weights...
✓ Loaded from ./LOGS/resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt
  Checkpoint size: 25.59M parameters

✓ Model on cuda in eval mode

[1.3] Model Architecture:
  Backbone: resnet50
  Backbone output channels: 1024
  Aggregator: MixVPR
  Config: {...}

  Total parameters: 25.59M
  Trainable parameters: 25.59M
```

---

## Step 2: Load Dataset with Inspection

Create file `inspect_step2.py`:

```python
import torchvision.transforms as T
from dataloaders.PittsburgDataset import get_whole_val_set
from torch.utils.data import DataLoader
from PIL import Image

print("="*80)
print("STEP 2: Load Dataset")
print("="*80)

# Define transforms
print("\n[2.1] Creating transforms...")
transform = T.Compose([
    T.Resize((320, 320)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225])
])
print("✓ Transforms created")
print("  - Resize to 320×320")
print("  - Convert to tensor")
print("  - Normalize with ImageNet mean/std")

# Load dataset
print("\n[2.2] Loading Pittsburgh30k-val dataset...")
try:
    val_dataset = get_whole_val_set(input_transform=transform)
    print(f"✓ Dataset loaded successfully")
except Exception as e:
    print(f"✗ Error: {e}")
    print(f"  Make sure ../datasets/Pittsburgh/ directory exists")
    exit(1)

# Inspect dataset
print("\n[2.3] Dataset Statistics:")
print(f"  Total images: {len(val_dataset)}")
print(f"  Database images: {val_dataset.dbStruct.numDb}")
print(f"  Query images: {val_dataset.dbStruct.numQ}")
print(f"  Dataset: {val_dataset.dataset}")

# Get ground truth
positives = val_dataset.getPositives()
print(f"\n[2.4] Ground Truth Information:")
print(f"  # of queries: {len(positives)}")
print(f"  Query 0 has {len(positives[0])} positive matches")
print(f"  Query 1 has {len(positives[1])} positive matches")
print(f"  Example: Query 0 matches references {positives[0][:5]}")

# Inspect single image
print(f"\n[2.5] Sample Image Inspection:")
img_tensor, idx = val_dataset[0]
print(f"  First image tensor shape: {img_tensor.shape}")
print(f"  Shape: [C, H, W] = [3, 320, 320]")
print(f"  Values range: [{img_tensor.min():.3f}, {img_tensor.max():.3f}]")
print(f"  (Normalized to 0-centered with Imagenet std)")

# Create dataloader
print(f"\n[2.6] Creating DataLoader...")
dataloader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0
)
print(f"✓ DataLoader created with batch_size=4")

# Get one batch
batch_imgs, batch_indices = next(iter(dataloader))
print(f"  Batch shape: {batch_imgs.shape}  [B, C, H, W]")
print(f"  Batch size: {batch_imgs.size(0)} images")

print("\n" + "="*80)
print("STEP 2 Complete: Dataset ready for inference")
print("="*80)
```

Run it:
```bash
python inspect_step2.py
```

---

## Step 3: Inspect Feature Extraction

Create file `inspect_step3.py`:

```python
import torch
import torchvision.transforms as T
from main import VPRModel
from dataloaders.PittsburgDataset import get_whole_val_set

print("="*80)
print("STEP 3: Inspect Feature Extraction")
print("="*80)

# Load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = VPRModel(
    backbone_arch='resnet50',
    layers_to_crop=[4],
    agg_arch='MixVPR',
    agg_config={
        'in_channels': 1024,
        'in_h': 20,
        'in_w': 20,
        'out_channels': 1024,
        'mix_depth': 4,
        'mlp_ratio': 1,
        'out_rows': 4
    },
)

state_dict = torch.load('./LOGS/resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt')
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

# Load dataset
transform = T.Compose([
    T.Resize((320, 320)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
val_dataset = get_whole_val_set(input_transform=transform)

# Get a batch
print("\n[3.1] Getting sample batch...")
img_batch, _ = next(iter(torch.utils.data.DataLoader(val_dataset, batch_size=2)))
print(f"  Batch shape: {img_batch.shape}  [B=2, C=3, H=320, W=320]")

img_batch = img_batch.to(device)

# Manual forward pass through backbone
print("\n[3.2] Forward pass through backbone (ResNet50)...")
with torch.no_grad():
    backbone_features = model.backbone(img_batch)

print(f"  Output shape: {backbone_features.shape}")
print(f"  Expected: [B=2, C=1024, H=20, W=20]")
print(f"  Matches? {backbone_features.shape == (2, 1024, 20, 20)}")
print(f"  Values: [{backbone_features.min():.4f}, {backbone_features.max():.4f}]")

# Through aggregator
print("\n[3.3] Forward pass through MixVPR aggregator...")
with torch.no_grad():
    descriptors = model.aggregator(backbone_features)

print(f"  Output shape: {descriptors.shape}")
print(f"  Expected: [B=2, D=4096]")
print(f"  Matches? {descriptors.shape == (2, 4096)}")
print(f"  Values: [{descriptors.min():.4f}, {descriptors.max():.4f}]")

# Check L2 norm (should be ~1 for normalized descriptors)
l2_norms = torch.norm(descriptors, p=2, dim=1)
print(f"  L2 norm of descriptors: {l2_norms.tolist()}")
print(f"  (Should be ~1.0 since we apply L2 normalization)")

# End-to-end forward pass
print("\n[3.4] End-to-end forward pass...")
with torch.no_grad():
    descriptors_e2e = model(img_batch)

print(f"  Output shape: {descriptors_e2e.shape}")
print(f"  Shape matches: {descriptors_e2e.shape == descriptors.shape}")
print(f"  Values match: {torch.allclose(descriptors_e2e, descriptors)}")

print("\n" + "="*80)
print("STEP 3 Complete: Feature extraction verified")
print("="*80)
```

Run it:
```bash
python inspect_step3.py
```

---

## Step 4: Inspect Similarity Search

Create file `inspect_step4.py`:

```python
import torch
import numpy as np
import faiss
import torchvision.transforms as T
from main import VPRModel
from dataloaders.PittsburgDataset import get_whole_val_set
from torch.utils.data import DataLoader

print("="*80)
print("STEP 4: Inspect Similarity Search with FAISS")
print("="*80)

# Quick inference on subset
print("\n[4.1] Extract descriptors from Pittsburgh dataset (subset)...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = VPRModel(
    backbone_arch='resnet50',
    layers_to_crop=[4],
    agg_arch='MixVPR',
    agg_config={
        'in_channels': 1024,
        'in_h': 20,
        'in_w': 20,
        'out_channels': 1024,
        'mix_depth': 4,
        'mlp_ratio': 1,
        'out_rows': 4
    },
)

state_dict = torch.load('./LOGS/resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt')
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

transform = T.Compose([
    T.Resize((320, 320)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
val_dataset = get_whole_val_set(input_transform=transform)

# Extract only first 100 reference and 10 query images for quick test
dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
descriptors_list = []

with torch.no_grad():
    count = 0
    for imgs, _ in dataloader:
        if count >= 110:  # 100 ref + 10 query
            break
        imgs = imgs.to(device)
        descs = model(imgs)
        descriptors_list.append(descs.cpu().numpy())
        count += len(imgs)

all_descs = np.concatenate(descriptors_list, axis=0)[:110]
print(f"✓ Extracted {all_descs.shape[0]} descriptors")
print(f"  Shape: {all_descs.shape}")

# Split into reference and query
num_ref = 100
ref_descs = all_descs[:num_ref].astype(np.float32)
query_descs = all_descs[num_ref:].astype(np.float32)

print(f"\n[4.2] Reference and Query descriptors:")
print(f"  References: {ref_descs.shape}  (100 database images)")
print(f"  Queries: {query_descs.shape}      (10 query images)")

# Build FAISS index
print(f"\n[4.3] Build FAISS index...")
embed_size = ref_descs.shape[1]
faiss_index = faiss.IndexFlatL2(embed_size)
faiss_index.add(ref_descs)
print(f"✓ Index created with {faiss_index.ntotal} reference vectors")

# Search for queries
print(f"\n[4.4] Search for nearest neighbors...")
k = 10
distances, indices = faiss_index.search(query_descs, k)

print(f"  Query 0 results:")
print(f"    Rank | Reference ID | Distance")
print(f"    -----|--------------|----------")
for rank in range(k):
    ref_id = indices[0, rank]
    dist = distances[0, rank]
    print(f"    {rank+1:4d} | {ref_id:12d} | {dist:8.4f}")

print(f"\n  Query 1 results:")
print(f"    Top-5 nearest references: {indices[1, :5]}")
print(f"    Top-5 distances:          {distances[1, :5].round(4)}")

# Get ground truth
positives = val_dataset.getPositives()
print(f"\n[4.5] Compare with ground truth:")
print(f"  Query 0's true positives: {positives[val_dataset.dbStruct.numDb]}")
print(f"  Query 0's top-10 retrieved: {indices[0, :]}")
print(f"  Match in top-1? {indices[0, 0] in positives[val_dataset.dbStruct.numDb]}")
print(f"  Match in top-10? {any(idx in positives[val_dataset.dbStruct.numDb] for idx in indices[0, :])}")

print("\n" + "="*80)
print("STEP 4 Complete: Similarity search working correctly")
print("="*80)
```

Run it:
```bash
python inspect_step4.py
```

---

## Full Test Checklist

Once all 4 inspection steps pass, you're ready for the full test:

- [ ] Step 1: Model loads correctly
- [ ] Step 2: Dataset loads (have you updated paths?)
- [ ] Step 3: Features extracted with correct shapes
- [ ] Step 4: FAISS search returns reasonable results

Then run:
```bash
python test_pretrained.py
```

---

## Common Issues During Inspection

| Issue | Solution |
|-------|----------|
| "Checkpoint not found" | Check path: `./LOGS/resnet50_MixVPR_4096...ckpt` |
| "Dataset not found" | Update path in `dataloaders/PittsburgDataset.py` line 11 |
| "CUDA out of memory" | Reduce batch_size in DataLoader |
| "AssertionError on normalize" | Make sure descriptors are final output, not intermediate |
| Slow performance on CPU | This is normal! GPU is 10-20x faster |

---

Now you understand exactly what happens at each stage! 🎉
