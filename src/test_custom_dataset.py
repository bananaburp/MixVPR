"""
Test pretrained MixVPR on a custom dataset.

Supports three ground-truth modes:
  --gt-mode gps     CSV files with lat/lon columns; positives auto-computed by distance threshold
  --gt-mode npy     Pre-built .npy file of shape (num_queries,) with arrays of matching db indices
  --gt-mode none    No ground truth; script still extracts descriptors and shows top-1 matches

Usage examples
--------------
# GPS mode (most common for custom data):
python test_custom_dataset.py \
    --db-dir   ./my_dataset/database \
    --q-dir    ./my_dataset/queries \
    --db-csv   ./my_dataset/database.csv \
    --q-csv    ./my_dataset/queries.csv \
    --gt-mode  gps \
    --dist-thr 25

# Pre-built ground truth mode:
python test_custom_dataset.py \
    --db-dir   ./my_dataset/database \
    --q-dir    ./my_dataset/queries \
    --gt-mode  npy \
    --gt-npy   ./my_dataset/ground_truth.npy

# No ground truth (just extract + rank):
python test_custom_dataset.py \
    --db-dir   ./my_dataset/database \
    --q-dir    ./my_dataset/queries \
    --gt-mode  none
"""

import argparse
import numpy as np
import torch
import torchvision.transforms as T
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from main import VPRModel
from utils.validation import get_validation_recalls


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}


def _collect_images(folder: Path) -> list[Path]:
    """Return sorted list of image paths in folder (non-recursive)."""
    imgs = sorted([
        p for p in folder.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    ])
    if not imgs:
        raise FileNotFoundError(
            f"No images found in '{folder}'. "
            f"Supported extensions: {SUPPORTED_EXTENSIONS}"
        )
    return imgs


class CustomDataset(Dataset):
    """
    Loads database images followed by query images from two separate directories.
    __getitem__ returns (image_tensor, global_index).
    """

    def __init__(self, db_dir: str, q_dir: str, transform=None):
        self.db_paths = _collect_images(Path(db_dir))
        self.q_paths  = _collect_images(Path(q_dir))
        self.all_paths = self.db_paths + self.q_paths
        self.num_references = len(self.db_paths)
        self.transform = transform

        print(f"  Database images : {len(self.db_paths)}")
        print(f"  Query images    : {len(self.q_paths)}")

    def __len__(self):
        return len(self.all_paths)

    def __getitem__(self, index):
        img = Image.open(self.all_paths[index]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, index


# ---------------------------------------------------------------------------
# Ground truth helpers
# ---------------------------------------------------------------------------

def _gt_from_gps(db_csv: str, q_csv: str, dist_thr: float) -> np.ndarray:
    """
    Build ground truth from GPS coordinates.

    Expects CSVs with at least two columns named 'lat' and 'lon'
    (column order does not matter; extra columns are ignored).

    Returns
    -------
    gt : np.ndarray of shape (num_queries,)
        Each element is an int array of database indices that are within
        dist_thr metres of the query.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for GPS mode: pip install pandas")

    from sklearn.neighbors import BallTree

    db_df = pd.read_csv(db_csv)
    q_df  = pd.read_csv(q_csv)

    for name, df in [('db_csv', db_df), ('q_csv', q_df)]:
        missing = {'lat', 'lon'} - set(df.columns.str.lower())
        if missing:
            raise ValueError(
                f"'{name}' is missing columns: {missing}. "
                "Column names are case-insensitive but must include 'lat' and 'lon'."
            )
    db_df.columns = db_df.columns.str.lower()
    q_df.columns  = q_df.columns.str.lower()

    # BallTree works in radians for haversine
    db_coords = np.radians(db_df[['lat', 'lon']].values)
    q_coords  = np.radians(q_df[['lat', 'lon']].values)

    EARTH_RADIUS_M = 6_371_000
    tree = BallTree(db_coords, metric='haversine')
    indices = tree.query_radius(q_coords, r=dist_thr / EARTH_RADIUS_M)

    num_with_pos = sum(1 for idx in indices if len(idx) > 0)
    print(f"  GPS positives   : {num_with_pos}/{len(q_df)} queries have ≥1 match "
          f"within {dist_thr} m")
    if num_with_pos == 0:
        print("  WARNING: No positives found. Check your lat/lon values "
              "and consider increasing --dist-thr.")

    return indices  # shape (num_queries,), dtype=object


def _gt_from_npy(npy_path: str) -> np.ndarray:
    """Load pre-built ground truth .npy file."""
    gt = np.load(npy_path, allow_pickle=True)
    print(f"  Loaded GT       : {len(gt)} query entries from '{npy_path}'")
    return gt


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def extract_descriptors(model, dataloader, device) -> np.ndarray:
    all_desc = []
    model.eval()
    with torch.no_grad():
        for imgs, _ in tqdm(dataloader, desc="Extracting descriptors", ncols=80):
            imgs = imgs.to(device)
            desc = model(imgs)
            all_desc.append(desc.cpu())
    return torch.cat(all_desc, dim=0).numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate pretrained MixVPR on a custom dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--ckpt',      default='./LOGS/resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt',
                   help='Path to pretrained checkpoint (.ckpt)')
    p.add_argument('--db-dir',    required=True,
                   help='Directory containing database (reference/map) images')
    p.add_argument('--q-dir',     required=True,
                   help='Directory containing query images')
    p.add_argument('--gt-mode',   choices=['gps', 'npy', 'none'], default='none',
                   help='How to supply ground truth (default: none)')

    # GPS mode
    p.add_argument('--db-csv',    default=None,
                   help='[gps mode] CSV with lat,lon for each database image (same order as --db-dir)')
    p.add_argument('--q-csv',     default=None,
                   help='[gps mode] CSV with lat,lon for each query image (same order as --q-dir)')
    p.add_argument('--dist-thr',  type=float, default=25.0,
                   help='[gps mode] Distance threshold in metres for a positive match (default: 25)')

    # NPY mode
    p.add_argument('--gt-npy',    default=None,
                   help='[npy mode] Path to .npy ground truth file')

    # Model / inference
    p.add_argument('--image-size', type=int, nargs=2, default=[320, 320],
                   metavar=('H', 'W'),
                   help='Resize images to H x W before inference (default: 320 320)')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--device',    default=None,
                   help='cuda or cpu (auto-detected if omitted)')
    p.add_argument('--k-values',  type=int, nargs='+', default=[1, 5, 10, 15, 20, 50, 100],
                   help='Recall@K values to compute (default: 1 5 10 15 20 50 100)')
    p.add_argument('--dataset-name', default='CustomDataset',
                   help='Display name used in results table')

    # Save descriptors
    p.add_argument('--save-descriptors', default=None, metavar='PATH',
                   help='If specified, save extracted descriptors as a .npy file')

    return p.parse_args()


def main():
    args = parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice : {device}")

    # ------------------------------------------------------------------
    # 1. Model
    # ------------------------------------------------------------------
    print("\n[1/4] Loading model...")
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
            'out_rows': 4,
        },
    )

    try:
        state_dict = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(state_dict)
        print(f"  Loaded weights  : {args.ckpt}")
    except Exception as e:
        print(f"  ERROR loading checkpoint: {e}")
        return

    model = model.to(device)

    # ------------------------------------------------------------------
    # 2. Dataset
    # ------------------------------------------------------------------
    print("\n[2/4] Loading dataset...")

    transform = T.Compose([
        T.Resize(tuple(args.image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    dataset = CustomDataset(args.db_dir, args.q_dir, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == 'cuda'),
    )

    # ------------------------------------------------------------------
    # 3. Ground truth
    # ------------------------------------------------------------------
    print("\n[3/4] Preparing ground truth...")
    gt = None

    if args.gt_mode == 'gps':
        if not args.db_csv or not args.q_csv:
            print("  ERROR: --db-csv and --q-csv are required for --gt-mode gps")
            return
        gt = _gt_from_gps(args.db_csv, args.q_csv, args.dist_thr)

    elif args.gt_mode == 'npy':
        if not args.gt_npy:
            print("  ERROR: --gt-npy is required for --gt-mode npy")
            return
        gt = _gt_from_npy(args.gt_npy)

    else:
        print("  No ground truth provided — skipping Recall@K, will show top-1 matches only.")

    # ------------------------------------------------------------------
    # 4. Descriptors
    # ------------------------------------------------------------------
    print("\n[4/4] Extracting descriptors...")
    all_desc = extract_descriptors(model, dataloader, device)

    print(f"  Descriptor matrix : {all_desc.shape}  "
          f"(dtype={all_desc.dtype})")

    if args.save_descriptors:
        np.save(args.save_descriptors, all_desc)
        print(f"  Saved to          : {args.save_descriptors}")

    r_list = all_desc[:dataset.num_references]
    q_list = all_desc[dataset.num_references:]

    # ------------------------------------------------------------------
    # 5. Evaluate
    # ------------------------------------------------------------------
    print()
    if gt is not None:
        # Validate GT length matches number of queries
        if len(gt) != len(q_list):
            print(f"ERROR: Ground truth has {len(gt)} entries but there are "
                  f"{len(q_list)} query images. They must match.")
            return

        results = get_validation_recalls(
            r_list=r_list,
            q_list=q_list,
            k_values=args.k_values,
            gt=gt,
            print_results=True,
            dataset_name=args.dataset_name,
        )
        return results

    else:
        # No GT: just find and print the top-1 nearest database image for each query
        import faiss
        idx = faiss.IndexFlatL2(r_list.shape[1])
        idx.add(r_list)
        _, predictions = idx.search(q_list, 1)

        db_paths = dataset.db_paths
        q_paths  = dataset.q_paths

        print(f"Top-1 nearest-neighbour results ({len(q_paths)} queries):\n")
        print(f"  {'Query':<45}  {'Top-1 Match':}")
        print("  " + "-" * 90)
        for i, pred in enumerate(predictions):
            q_name  = q_paths[i].name
            db_name = db_paths[pred[0]].name
            print(f"  {q_name:<45}  {db_name}")


if __name__ == '__main__':
    main()
