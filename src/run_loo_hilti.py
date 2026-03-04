"""
run_loo_hilti.py — Leave-One-Out fine-tuning + evaluation driver.

For each fold (one csv held out as test), this script:
  1. Trains MixVPR on the other (N-1) CSVs using train_hilti.py logic.
  2. Evaluates Recall@1/5/10 on the held-out run using run_hilti_recall.py.
  3. Writes per-fold and aggregate results to a JSON file.

PRE-REQUISITES
--------------
• aligned.csv exists for every run (cam0 at minimum).
• Eval sets (db/, query/, gt_positives.npy) are pre-built for each run using:
    python challenge_tools_ros/vpr/src/build_mixvpr_evalset.py
  Pass the eval root dir for each run via --eval_roots (same order as --csvs).

USAGE
-----
python src/run_loo_hilti.py \
    --csvs \
        data/floor_1_run_1/cam0/aligned.csv \
        data/floor_2_run_1/cam0/aligned.csv \
        data/floor_2_run_2/cam0/aligned.csv \
        data/floor_2_run_3/cam0/aligned.csv \
        data/floor_UG1_run_1/cam0/aligned.csv \
    --eval_roots \
        eval/floor_1_run_1_cam0 \
        eval/floor_2_run_1_cam0 \
        eval/floor_2_run_2_cam0 \
        eval/floor_2_run_3_cam0 \
        eval/floor_UG1_run_1_cam0 \
    --ckpt  LOGS/resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt \
    --output_dir LOGS/loo_results \
    --max_epochs 30

Results are written to LOGS/loo_results/loo_summary.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import subprocess
import time
from pathlib import Path

# ── make repo root importable ─────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import torch

from main import VPRModel
from dataloaders.HiltiDataset import HILTI_EVAL_TRANSFORM
from utils.validation import get_validation_recalls


# ─────────────────────────────────────────────────────────────────────────────
# Inline recall evaluation (avoids subprocess overhead)
# ─────────────────────────────────────────────────────────────────────────────

def _eval_recall(ckpt_path: str, eval_root: str, device: str, batch: int = 16) -> dict:
    """Run Recall@K evaluation inline. Returns {1: R@1, 5: R@5, 10: R@10}."""
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    import glob

    class _ImgDS(Dataset):
        def __init__(self, folder):
            imgs = glob.glob(os.path.join(folder, "**", "*.jpg"), recursive=True)
            self.paths = sorted(imgs, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            assert self.paths, f"No images in {folder}"
        def __len__(self): return len(self.paths)
        def __getitem__(self, i):
            img = Image.open(self.paths[i]).convert("RGB")
            return HILTI_EVAL_TRANSFORM(img), i

    model = VPRModel(
        backbone_arch="resnet50", pretrained=False, layers_to_crop=[4],
        agg_arch="MixVPR",
        agg_config={"in_channels":1024,"in_h":20,"in_w":20,
                    "out_channels":1024,"mix_depth":4,"mlp_ratio":1,"out_rows":4},
    )
    state = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    dev = torch.device(device if device != "mps" or torch.backends.mps.is_available() else "cpu")
    model.to(dev)

    db_ds = _ImgDS(os.path.join(eval_root, "db"))
    q_ds  = _ImgDS(os.path.join(eval_root, "query"))

    def extract(ds):
        dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)
        out = []
        with torch.no_grad():
            for imgs, _ in dl:
                out.append(model(imgs.to(dev)).cpu().numpy())
        return np.vstack(out).astype(np.float32)

    db_desc = extract(db_ds)
    q_desc  = extract(q_ds)
    gt      = np.load(os.path.join(eval_root, "gt_positives.npy"), allow_pickle=True)

    if len(gt) < len(q_desc):
        q_desc = q_desc[:len(gt)]

    recalls = get_validation_recalls(
        r_list=db_desc, q_list=q_desc,
        k_values=[1, 5, 10],
        gt=gt, print_results=True,
        dataset_name=os.path.basename(eval_root),
    )
    return recalls


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(
        description="Leave-One-Out fine-tuning + recall evaluation for Hilti.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--csvs", nargs="+", required=True,
                    help="One aligned.csv per run (order must match --eval_roots).")
    ap.add_argument("--eval_roots", nargs="+", required=True,
                    help="Pre-built eval dirs (db/ query/ gt_positives.npy) per run.")
    ap.add_argument("--ckpt",       required=True, help="Pretrained checkpoint.")
    ap.add_argument("--output_dir", default="LOGS/loo_results")
    ap.add_argument("--max_epochs", type=int,   default=30)
    ap.add_argument("--lr",         type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int,   default=16)
    ap.add_argument("--img_per_place", type=int, default=4)
    ap.add_argument("--grid_m",     type=float, default=1.0)
    ap.add_argument("--num_workers",type=int,   default=4)
    ap.add_argument("--device",     default="mps")
    ap.add_argument("--folds",      nargs="+",  type=int, default=None,
                    help="Run only these fold indices (0-based). Default: all folds.")
    return ap.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    assert len(args.csvs) == len(args.eval_roots), (
        "--csvs and --eval_roots must have the same number of entries."
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n      = len(args.csvs)
    folds  = args.folds if args.folds is not None else list(range(n))
    results = {}

    for fold_idx in folds:
        held_out_csv  = args.csvs[fold_idx]
        held_out_eval = args.eval_roots[fold_idx]
        train_csvs    = [c for i, c in enumerate(args.csvs) if i != fold_idx]
        fold_name     = Path(held_out_csv).parent.parent.name     # run dir name
        fold_out      = output_dir / f"fold_{fold_idx:02d}_{fold_name}"
        fold_out.mkdir(parents=True, exist_ok=True)

        print("\n" + "="*70)
        print(f"FOLD {fold_idx+1}/{n}  held-out: {held_out_csv}")
        print(f"  train runs: {[Path(c).parent.parent.name for c in train_csvs]}")
        print("="*70 + "\n")

        # ── Train ─────────────────────────────────────────────────────────
        t0 = time.time()
        train_script = str(_REPO_ROOT / "src" / "train_hilti.py")
        train_cmd = [
            sys.executable, train_script,
            "--train_csvs", *train_csvs,
            "--held_out_csv", held_out_csv,
            "--ckpt", args.ckpt,
            "--output_dir", str(fold_out),
            "--max_epochs", str(args.max_epochs),
            "--lr", str(args.lr),
            "--batch_size", str(args.batch_size),
            "--img_per_place", str(args.img_per_place),
            "--grid_m", str(args.grid_m),
            "--num_workers", str(args.num_workers),
        ]
        print(f"[LOO] Running: {' '.join(train_cmd)}\n")
        ret = subprocess.run(train_cmd, check=True)
        train_elapsed = time.time() - t0

        # Load run_meta.json to find checkpoint
        meta_path = fold_out / "run_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            best_ckpt = meta.get("best_checkpoint") or meta.get("last_checkpoint")
        else:
            # Fallback: find last.ckpt
            candidates = sorted(fold_out.glob("last.ckpt"))
            best_ckpt = str(candidates[-1]) if candidates else None

        if not best_ckpt or not Path(best_ckpt).exists():
            print(f"[LOO] WARNING: no checkpoint found for fold {fold_idx}; skipping eval.")
            results[fold_name] = {"error": "no_checkpoint"}
            continue

        # ── Eval ──────────────────────────────────────────────────────────
        print(f"\n[LOO] Evaluating on {held_out_eval} with {best_ckpt} …")
        try:
            recalls = _eval_recall(
                ckpt_path=best_ckpt,
                eval_root=held_out_eval,
                device=args.device,
            )
        except Exception as e:
            print(f"[LOO] Eval error: {e}")
            recalls = {"error": str(e)}

        fold_result = {
            "fold_idx":       fold_idx,
            "held_out_csv":   held_out_csv,
            "train_csvs":     train_csvs,
            "best_checkpoint":best_ckpt,
            "recalls":        {str(k): float(v) for k, v in recalls.items()
                               if not isinstance(v, str)},
            "train_elapsed_sec": train_elapsed,
        }
        results[fold_name] = fold_result

        print(f"\n[LOO] Fold {fold_idx} Recall@1 = "
              f"{recalls.get(1, 0)*100:.2f}%  "
              f"@5 = {recalls.get(5, 0)*100:.2f}%  "
              f"@10 = {recalls.get(10, 0)*100:.2f}%")

        # Save intermediate results
        summary_path = output_dir / "loo_summary.json"
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[LOO] Partial results saved to {summary_path}")

    # ── Aggregate ─────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("LOO COMPLETE — Aggregate Results")
    print("="*70)
    r1s, r5s, r10s = [], [], []
    for name, res in results.items():
        rec = res.get("recalls", {})
        r1  = rec.get("1",  rec.get(1,  None))
        r5  = rec.get("5",  rec.get(5,  None))
        r10 = rec.get("10", rec.get(10, None))
        if r1 is not None:
            r1s.append(r1); r5s.append(r5); r10s.append(r10)
            print(f"  {name}: R@1={r1*100:.2f}%  R@5={r5*100:.2f}%  R@10={r10*100:.2f}%")

    if r1s:
        agg = {
            "mean_R@1":  float(np.mean(r1s)),
            "mean_R@5":  float(np.mean(r5s)),
            "mean_R@10": float(np.mean(r10s)),
        }
        results["aggregate"] = agg
        print(f"\n  MEAN: R@1={agg['mean_R@1']*100:.2f}%  "
              f"R@5={agg['mean_R@5']*100:.2f}%  "
              f"R@10={agg['mean_R@10']*100:.2f}%")

    summary_path = output_dir / "loo_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[LOO] Final results saved to {summary_path}")


if __name__ == "__main__":
    main()
