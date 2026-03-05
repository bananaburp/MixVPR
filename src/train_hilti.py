"""
train_hilti.py — Fine-tuning entrypoint for MixVPR on Hilti-Trimble data.

WHAT: Fine-tunes the pretrained resnet50+MixVPR model on Hilti indoor runs
        using a leave-one-out scheme (train on N-1 runs, eval recall on held-out).

WHY ADAMW over SGD (original repo uses SGD)?
    Fine-tuning on a small domain-shifted dataset benefits from AdamW's adaptive
    per-parameter LR. SGD requires careful LR tuning that varies with batch size.
    AdamW with lr=2e-4 is a well-known safe choice for metric-learning fine-tuning
    on ~thousands of images.

WHY layers_to_freeze=2 (same as repo default)?
    We freeze BN + first 2 ResNet stages (low-level Gabor/edge features transfer
    well). Stages 3 & 4 + the MixVPR aggregator are fine-tuned to adapt to
    fisheye indoor texture statistics. Freezing more would be too conservative
    given the large domain gap (outdoor street → indoor fisheye).

WHY precision=32 on MPS?
    PyTorch MPS fp16 can produce NaN gradients with pytorch-metric-learning's
    MultiSimilarityLoss (involves DotProductSimilarity with large embedding
    norms). Use precision=32 unless you confirm stability.

USAGE
=====
# Smoke run (1 run, 50 places, 1 epoch)
python src/train_hilti.py \
    --train_csvs path/to/floor_1_run_1/cam0/aligned.csv \
    --ckpt LOGS/resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt \
    --smoke

# Real leave-one-out: train on 4 runs, held-out = floor_2_run_1

python3 src/train_hilti.py \
    --train_csvs \
        /Volumes/T9/DevSpace/Github/hilti-trimble-slam-challenge-2026/challenge_tools_ros/vpr/data/floor_2_2025-05-05_run_1/eval/aligned_frames_cam0.csv \
        /Volumes/T9/DevSpace/Github/hilti-trimble-slam-challenge-2026/challenge_tools_ros/vpr/data/floor_2_2025-05-05_run_1/eval/aligned_frames_cam1.csv \
        /Volumes/T9/DevSpace/Github/hilti-trimble-slam-challenge-2026/challenge_tools_ros/vpr/data/floor_2_2025-10-28_run_1/eval/aligned_frames_cam0.csv \
        /Volumes/T9/DevSpace/Github/hilti-trimble-slam-challenge-2026/challenge_tools_ros/vpr/data/floor_2_2025-10-28_run_1/eval/aligned_frames_cam1.csv \
        /Volumes/T9/DevSpace/Github/hilti-trimble-slam-challenge-2026/challenge_tools_ros/vpr/data/floor_2_2025-10-28_run_2/eval/aligned_frames_cam0.csv \
        /Volumes/T9/DevSpace/Github/hilti-trimble-slam-challenge-2026/challenge_tools_ros/vpr/data/floor_2_2025-10-28_run_2/eval/aligned_frames_cam1.csv \
        /Volumes/T9/DevSpace/Github/hilti-trimble-slam-challenge-2026/challenge_tools_ros/vpr/data/floor_UG1_2025-10-16_run_1/eval/aligned_frames_cam0.csv \
        /Volumes/T9/DevSpace/Github/hilti-trimble-slam-challenge-2026/challenge_tools_ros/vpr/data/floor_UG1_2025-10-16_run_1/eval/aligned_frames_cam1.csv \
    --ckpt "LOGS/resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt" \
    --output_dir LOGS/hilti_finetune/fold_floor1 \
    --max_epochs 2 \
    --warmup_steps 50 \
    --lr 2e-4 \
    --batch_size 16 \
    --num_workers 4
    
NEPTUNE:

python3 src/train_hilti.py \
    --train_csvs \
        hilti_data/vpr/data/floor_2_2025-05-05_run_1/eval/aligned_frames_cam0.csv \
        hilti_data/vpr/data/floor_2_2025-05-05_run_1/eval/aligned_frames_cam1.csv \
        hilti_data/vpr/data/floor_2_2025-10-28_run_1/eval/aligned_frames_cam0.csv \
        hilti_data/vpr/data/floor_2_2025-10-28_run_1/eval/aligned_frames_cam1.csv \
        hilti_data/vpr/data/floor_2_2025-10-28_run_2/eval/aligned_frames_cam0.csv \
        hilti_data/vpr/data/floor_2_2025-10-28_run_2/eval/aligned_frames_cam1.csv \
        hilti_data/vpr/data/floor_UG1_2025-10-16_run_1/eval/aligned_frames_cam0.csv \
        hilti_data/vpr/data/floor_UG1_2025-10-16_run_1/eval/aligned_frames_cam1.csv \
    --ckpt "LOGS/resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt" \
    --output_dir LOGS/hilti_finetune/fold_floor1_v2 \
    --max_epochs 50 \
    --warmup_steps 50 \
    --lr 2e-4 \
    --batch_size 8 \
    --num_workers 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ── make repo root importable regardless of where script is invoked ──────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
# ─────────────────────────────────────────────────────────────────────────────

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, LearningRateMonitor
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe on macOS without a display
import matplotlib.pyplot as plt
import csv

from main import VPRModel
from dataloaders.HiltiDataModule import HiltiDataModule
from dataloaders.HiltiDataset import HILTI_EVAL_TRANSFORM


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint save logger callback
# ─────────────────────────────────────────────────────────────────────────────

class CheckpointLogger(Callback):
    """Logs clear messages about checkpoint saves to help track training progress.
    
    Logs:
      - When a checkpoint is saved (with epoch, metric value, and path)
      - When an epoch completes but no checkpoint is saved (metric didn't improve)
      - Summary of top-k checkpoints at any time
    """
    
    def __init__(self):
        super().__init__()
        self.last_logged_epoch = -1
        self.epoch_metric_value = None
        self.checkpoint_saved_this_epoch = False
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Capture the metric value at epoch end."""
        self.epoch_metric_value = trainer.callback_metrics.get("epoch_loss")
        self.checkpoint_saved_this_epoch = False
        self.last_logged_epoch = trainer.current_epoch
    
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """Log when a checkpoint is being saved."""
        epoch = trainer.current_epoch
        metric_val = self.epoch_metric_value
        
        # Find the ModelCheckpoint callback to get the filename
        ckpt_callback = None
        for cb in trainer.callbacks:
            if isinstance(cb, ModelCheckpoint):
                ckpt_callback = cb
                break
        
        if ckpt_callback and metric_val is not None:
            # Get the actual filename that will be saved
            filename = ckpt_callback.format_checkpoint_name(
                {"epoch": epoch, "epoch_loss": float(metric_val)}
            )
            filepath = os.path.join(ckpt_callback.dirpath, filename + ".ckpt")
            
            print(f"\n{'='*70}")
            print(f"✓ CHECKPOINT SAVED — Epoch {epoch}")
            print(f"  Metric (epoch_loss): {metric_val:.6f}")
            print(f"  Path: {filepath}")
            print(f"  Reason: Loss improved (top-{ckpt_callback.save_top_k} models)")
            print(f"{'='*70}\n", flush=True)
            
            self.checkpoint_saved_this_epoch = True
    
    def on_validation_end(self, trainer, pl_module):
        """After validation/epoch ends, log if no checkpoint was saved."""
        # We check at validation_end because on_save_checkpoint fires before this
        # This ensures we can report "no save" after we know for sure
        epoch = trainer.current_epoch
        
        # Only report once per epoch
        if epoch == self.last_logged_epoch and not self.checkpoint_saved_this_epoch:
            metric_val = self.epoch_metric_value
            if metric_val is not None:
                # Find best metric to compare
                ckpt_callback = None
                for cb in trainer.callbacks:
                    if isinstance(cb, ModelCheckpoint):
                        ckpt_callback = cb
                        break
                
                best_so_far = "N/A"
                if ckpt_callback and hasattr(ckpt_callback, 'best_model_score'):
                    if ckpt_callback.best_model_score is not None:
                        best_so_far = f"{float(ckpt_callback.best_model_score):.6f}"
                
                print(f"\n{'='*70}")
                print(f"○ NO CHECKPOINT SAVED — Epoch {epoch}")
                print(f"  Metric (epoch_loss): {metric_val:.6f}")
                print(f"  Best so far: {best_so_far}")
                print(f"  Reason: Loss did not improve")
                print(f"{'='*70}\n", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Loss curve plotter callback
# ─────────────────────────────────────────────────────────────────────────────

class LossCurvePlotter(Callback):
    """Saves a loss + batch-accuracy PNG after every epoch so you can
    inspect training progress without waiting for it to finish.

    Files written:
      <output_dir>/loss_curve.png              — updated every epoch (overwritten)
      <output_dir>/training_steps.csv          — raw step-by-step losses and batch acc
      <output_dir>/training_epochs.csv         — epoch-level aggregated metrics

    WHY collect in on_train_batch_end instead of on_train_epoch_end?
      HiltiVPRModel.on_train_epoch_end fires BEFORE callbacks and clears
      _epoch_losses, so by the time this callback runs per-epoch values are
      gone from both the model and callback_metrics (PL flushes logged metrics
      after all on_train_epoch_end hooks).  Collecting step values here is
      the only reliable approach.
    """

    def __init__(self, output_dir: str):
        super().__init__()
        self.output_dir  = Path(output_dir)
        self.epoch_losses: list[float] = []
        self.epoch_baccs:  list[float] = []
        self.epochs:       list[int]   = []
        self._step_losses: list[float] = []
        self._step_baccs:  list[float] = []
        self._global_step = 0
        
        # Initialize CSV files
        self.steps_csv_path  = self.output_dir / "training_steps.csv"
        self.epochs_csv_path = self.output_dir / "training_epochs.csv"
        
        # Create step-level CSV with header
        with open(self.steps_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["global_step", "epoch", "loss", "batch_acc"])
        
        # Create epoch-level CSV with header
        with open(self.epochs_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "mean_loss", "mean_batch_acc"])

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # outputs is the dict returned by training_step: {"loss": tensor}
        loss_val = outputs.get("loss") if isinstance(outputs, dict) else None
        if loss_val is not None:
            loss_float = float(loss_val)
            self._step_losses.append(loss_float)
        else:
            loss_float = float("nan")
            
        bacc = trainer.callback_metrics.get("b_acc")
        if bacc is not None:
            bacc_float = float(bacc)
            self._step_baccs.append(bacc_float)
        else:
            bacc_float = float("nan")
        
        # Write step-level data to CSV
        with open(self.steps_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self._global_step,
                trainer.current_epoch,
                loss_float,
                bacc_float
            ])
        self._global_step += 1

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if not self._step_losses:
            return
        mean_loss = sum(self._step_losses) / len(self._step_losses)
        mean_bacc = (sum(self._step_baccs) / len(self._step_baccs)
                     if self._step_baccs else float("nan"))
        
        # Write epoch-level data to CSV
        with open(self.epochs_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, mean_loss, mean_bacc])
        
        self._step_losses.clear()
        self._step_baccs.clear()

        self.epoch_losses.append(mean_loss)
        self.epoch_baccs.append(mean_bacc)
        self.epochs.append(epoch)
        self._save_plot()

    def _save_plot(self):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle("MixVPR Hilti Fine-Tuning", fontsize=13)

        # ── Loss curve ────────────────────────────────────────────────
        axes[0].plot(self.epochs, self.epoch_losses,
                     marker="o", linewidth=2, color="steelblue", label="epoch loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Mean Loss")
        axes[0].set_title("Training Loss")
        axes[0].grid(True, alpha=0.3)
        if len(self.epoch_losses) > 1:
            axes[0].annotate(
                f"{self.epoch_losses[-1]:.4f}",
                xy=(self.epochs[-1], self.epoch_losses[-1]),
                xytext=(5, 5), textcoords="offset points", fontsize=9,
            )

        # ── Batch accuracy curve ──────────────────────────────────────
        # b_acc = fraction of non-trivial pairs; rising → model learning
        valid = [(e, b) for e, b in zip(self.epochs, self.epoch_baccs)
                 if not (b != b)]  # filter NaN
        if valid:
            es, bs = zip(*valid)
            axes[1].plot(es, bs, marker="s", linewidth=2,
                         color="darkorange", label="batch acc (non-trivial pairs)")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Batch Accuracy")
        axes[1].set_title("Batch Accuracy (↑ = harder mining = learning)")
        axes[1].set_ylim(-0.05, 1.05)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        out = self.output_dir / "loss_curve.png"
        fig.savefig(str(out), dpi=120)
        plt.close(fig)
        print(f"[LossCurvePlotter] Updated {out}  "
              f"(epoch {self.epochs[-1]}, loss={self.epoch_losses[-1]:.4f})",
              flush=True)
        print(f"[LossCurvePlotter] CSV logs: {self.steps_csv_path}, {self.epochs_csv_path}",
              flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Subclass to make VPRModel compatible with both old and new PL API
# and to skip internal recall validation (we use external eval script).
# ─────────────────────────────────────────────────────────────────────────────

class HiltiVPRModel(VPRModel):
    """VPRModel subclass adapted for Hilti fine-tuning.

    Changes vs parent:
    1. optimizer_step: uses the new PL 1.9+ / 2.x API signature.
    2. training_epoch_end: logs per-epoch mean loss for checkpoint monitoring.
    3. validation_epoch_end: no-op (validation is external via recall script).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._epoch_losses: list[float] = []

    def training_step(self, batch, batch_idx):
        result = super().training_step(batch, batch_idx)
        self._epoch_losses.append(result["loss"].item())
        return result

    # PL >= 1.9 renamed training_epoch_end → on_train_epoch_end (no args).
    # We override both for maximum compatibility.
    def on_train_epoch_end(self):
        if self._epoch_losses:
            mean_loss = sum(self._epoch_losses) / len(self._epoch_losses)
            self.log("epoch_loss", mean_loss, prog_bar=True, logger=True, on_epoch=True)
            self._epoch_losses.clear()
        self.batch_acc = []   # reset parent's batch_acc accumulator

    def on_validation_epoch_end(self):
        """New PL API — same: skip inline recall computation."""
        pass

    def loss_function(self, descriptors, labels):
        """Robust loss_function for fine-tuning on domain-shifted data.

        Strategy:
          1. Try the miner-based MultiSimilarityLoss.
          2. On any PML error (AssertionError / 0 pairs), fall back to a
             hand-rolled supervised InfoNCE that is device-agnostic and
             always produces gradient signal even from collapsed embeddings.
        """
        # ── Attempt 1: miner-based PML loss ─────────────────────────────────
        try:
            if self.miner is not None:
                miner_outputs = self.miner(descriptors, labels)
                n_mined = len(miner_outputs[0]) + len(miner_outputs[2])
                if n_mined == 0:
                    raise ValueError("miner found 0 pairs")
                loss = self.loss_fn(descriptors, labels, miner_outputs)
                nb_samples = descriptors.shape[0]
                nb_mined   = len(set(miner_outputs[0].detach().cpu().numpy()))
                batch_acc  = 1.0 - (nb_mined / nb_samples)
            else:
                loss      = self.loss_fn(descriptors, labels)
                batch_acc = 0.0
            self.batch_acc.append(batch_acc)
            self.log("b_acc", sum(self.batch_acc) / len(self.batch_acc),
                     prog_bar=True, logger=True)
            return loss
        except (AssertionError, ValueError, RuntimeError):
            pass  # fall through to supervised InfoNCE fallback

        # ── Attempt 2: supervised InfoNCE (no PML, device-agnostic) ─────────
        # descriptors are L2-normalised: dot product == cosine similarity
        temperature = 0.07
        sim = torch.mm(descriptors, descriptors.T) / temperature        # (N, N)
        lbl = labels.unsqueeze(1)
        pos_mask = (lbl == lbl.T).float()
        pos_mask.fill_diagonal_(0.0)
        neg_mask = (lbl != lbl.T).float()
        if neg_mask.sum() == 0:                                          # degenerate
            self.batch_acc.append(0.0)
            return descriptors.sum() * 0.0                               # zero grad
        exp_sim   = torch.exp(sim - sim.max(dim=1, keepdim=True).values) # stable
        pos_sum   = (exp_sim * pos_mask).sum(dim=1)
        total_sum = (exp_sim * (pos_mask + neg_mask)).sum(dim=1).clamp(min=1e-9)
        row_loss  = -torch.log((pos_sum / total_sum).clamp(min=1e-9))
        has_pos   = pos_mask.sum(dim=1) > 0
        loss      = row_loss[has_pos].mean() if has_pos.any() else descriptors.sum() * 0.0
        if loss.isnan() or loss.isinf():
            loss  = descriptors.sum() * 0.0
        self.batch_acc.append(0.0)
        self.log("b_acc", 0.0, prog_bar=True, logger=True)
        return loss

    def optimizer_step(self, *args, **kwargs):
        """Warmup LR scheduling, compatible with PL 1.x and 2.x."""
        # PL 2.x calls: (epoch, batch_idx, optimizer, optimizer_closure)
        # PL 1.x calls: (epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, ...)
        # We extract optimizer and closure from args/kwargs regardless of version.
        optimizer       = kwargs.get("optimizer",       args[2] if len(args) > 2 else None)
        optimizer_closure = kwargs.get("optimizer_closure",
                                       args[4] if len(args) > 4   # old API idx 4
                                       else args[3] if len(args) > 3  # new API idx 3
                                       else None)
        if optimizer is None:
            # Fall back to parent signature if we can't parse
            super().optimizer_step(*args, **kwargs)
            return

        # Warmup: linearly scale LR for the first `warmpup_steps` global steps
        if self.trainer.global_step < self.warmpup_steps:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.warmpup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.lr

        if optimizer_closure is not None:
            optimizer.step(closure=optimizer_closure)
        else:
            optimizer.step()


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(
        description="Fine-tune MixVPR on Hilti-Trimble indoor data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── Data ──
    ap.add_argument("--train_csvs", nargs="+", required=True,
                    help="Paths to aligned.csv for training runs (1+ files).")
    ap.add_argument("--held_out_csv", default=None,
                    help="aligned.csv for held-out run (for eval only; not trained on).")
    ap.add_argument("--grid_m", type=float, default=1.0,
                    help="XY grid cell size for place labels (metres).")
    ap.add_argument("--img_per_place", type=int, default=4,
                    help="Images sampled per place per step.")
    ap.add_argument("--min_img_per_place", type=int, default=None,
                    help="Min images for a place to be included (default = img_per_place).")

    # ── Model ──
    ap.add_argument("--ckpt", required=True,
                    help="Pretrained checkpoint to initialise from "
                         "(resnet50_MixVPR_4096_....ckpt).")
    ap.add_argument("--layers_to_freeze", type=int, default=2,
                    help="ResNet stages to freeze (0=none, 2=freeze stages 1+2).")

    # ── Optimiser ──
    ap.add_argument("--lr",           type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--warmup_steps", type=int,   default=50,
                    help="LR warmup steps. Keep low (50) for small Hilti dataset; "
                         "200 (old default) starves a full epoch when <300 iterations total.")

    # ── Training ──
    ap.add_argument("--batch_size",  type=int,   default=16,
                    help="Places per batch.")
    ap.add_argument("--max_epochs",  type=int,   default=30)
    ap.add_argument("--num_workers", type=int,   default=4)
    ap.add_argument("--precision",   type=int,   default=32,
                    help="32 (safe) or 16. fp16 can NaN on MPS metric learning.")
    ap.add_argument("--output_dir",  type=str,   default="LOGS/hilti_finetune",
                    help="Directory for checkpoints and logs.")

    # ── Smoke test ──
    ap.add_argument("--smoke", action="store_true",
                    help="Smoke mode: 1 run, 50 places, 8 batch, 1 epoch.")

    return ap.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Smoke overrides ──────────────────────────────────────────────────────
    if args.smoke:
        print("\n" + "="*60)
        print("SMOKE MODE — 1 CSV, 50 places, batch=8, 1 epoch")
        print("="*60 + "\n")
        args.train_csvs      = args.train_csvs[:1]   # only first CSV
        args.batch_size      = 8
        args.max_epochs      = 1
        cap_places           = 50
        args.num_workers     = 0          # simpler debugging in smoke mode
    else:
        cap_places = None

    min_ipp = args.min_img_per_place if args.min_img_per_place else args.img_per_place

    # ── Detect accelerator (done early so faiss_gpu can be set correctly) ───
    _torch_mps = torch.backends.mps.is_available() and torch.backends.mps.is_built()

    if _torch_mps:
        try:
            import lightning_fabric.accelerators.mps as _fab_mps
            _fab_mps.MPSAccelerator.is_available = staticmethod(lambda: True)
            print("[train_hilti] Patched PL MPSAccelerator.is_available → True "
                  "(platform.processor() Rosetta workaround).")
        except Exception as _patch_err:
            print(f"[train_hilti] Warning: could not patch PL MPS check: {_patch_err}")
        accelerator, devices = "mps", 1
        faiss_gpu = False
        print("[train_hilti] Using MPS (Apple Silicon) accelerator.")
    elif torch.cuda.is_available():
        accelerator, devices = "gpu", [0]
        faiss_gpu = True
        print("[train_hilti] Using CUDA GPU accelerator.")
    else:
        accelerator, devices = "cpu", 1
        faiss_gpu = False
        print("[train_hilti] WARNING: No GPU/MPS found — training on CPU (very slow).")

    # ── DataModule ───────────────────────────────────────────────────────────
    datamodule = HiltiDataModule(
        train_csv_files=args.train_csvs,
        val_csv_files=None,               # no inline recall validation
        batch_size=args.batch_size,
        img_per_place=args.img_per_place,
        min_img_per_place=min_ipp,
        grid_m=args.grid_m,
        num_workers=args.num_workers,
        cap_places=cap_places,
    )

    # ── Model ────────────────────────────────────────────────────────────────
    print(f"\n[train_hilti] Initialising HiltiVPRModel …")
    model = HiltiVPRModel(
        backbone_arch="resnet50",
        pretrained=False,               # weights come from fine-tune ckpt below
        layers_to_freeze=args.layers_to_freeze,
        layers_to_crop=[4],             # crop last ResNet stage (required by pretrained config)

        agg_arch="MixVPR",
        agg_config={
            "in_channels": 1024,
            "in_h":        20,
            "in_w":        20,
            "out_channels": 1024,
            "mix_depth":   4,
            "mlp_ratio":   1,
            "out_rows":    4,
        },

        # Fine-tuning optimiser
        optimizer     = "adamw",
        lr            = args.lr,
        weight_decay  = args.weight_decay,
        warmpup_steps = args.warmup_steps,
        milestones    = [10, 20, 30],   # LR decay epochs (may not all be reached)
        lr_mult       = 0.3,

        loss_name   = "MultiSimilarityLoss",
        miner_name  = "MultiSimilarityMiner",
        miner_margin = 0.1,
        faiss_gpu    = faiss_gpu,
    )

    # Load pretrained weights
    print(f"[train_hilti] Loading pretrained weights from {args.ckpt} …")
    state = torch.load(args.ckpt, map_location="cpu")
    # The .ckpt may be a raw state_dict or a Lightning checkpoint
    if "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [warn] Missing keys ({len(missing)}): {missing[:5]} …")
    if unexpected:
        print(f"  [warn] Unexpected keys ({len(unexpected)}): {unexpected[:5]} …")
    print("  [ok] Pretrained weights loaded.\n")

    # ── Callbacks ────────────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir),
        filename="epoch{epoch:03d}_loss{epoch_loss:.6f}",  # More digits for better tracking
        monitor="epoch_loss",
        mode="min",
        save_top_k=3,
        save_weights_only=True,
        save_last=True,  # Always save last.ckpt for progress tracking
        verbose=True,    # Enable PL's built-in checkpoint logging
    )
    ckpt_logger  = CheckpointLogger()
    lr_monitor   = LearningRateMonitor(logging_interval="step")
    loss_plotter = LossCurvePlotter(output_dir=str(output_dir))

    # ── Trainer ──────────────────────────────────────────────────────────────
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        default_root_dir=str(output_dir),
        max_epochs=args.max_epochs,
        precision=args.precision,            # 32 safe for MPS; 16 usable on CUDA
        callbacks=[checkpoint_cb, ckpt_logger, lr_monitor, loss_plotter],
        log_every_n_steps=10,
        num_sanity_val_steps=0,             # no sanity val (no inline val set)
        check_val_every_n_epoch=9999,       # effectively disable inline validation
        reload_dataloaders_every_n_epochs=1, # re-shuffle place sampling each epoch
        enable_progress_bar=True,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    t0 = time.time()
    print(f"[train_hilti] Starting training: {args.max_epochs} epochs …\n")
    trainer.fit(model=model, datamodule=datamodule)
    elapsed = time.time() - t0
    print(f"\n[train_hilti] Training complete in {elapsed/60:.1f} min.")

    # ── Save run metadata ────────────────────────────────────────────────────
    best_ckpt = checkpoint_cb.best_model_path
    last_ckpt = checkpoint_cb.last_model_path
    
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Best checkpoint : {best_ckpt}")
    print(f"Last checkpoint : {last_ckpt}")
    print(f"Elapsed time    : {elapsed/60:.1f} minutes")
    
    # List all saved checkpoints
    saved_ckpts = sorted(output_dir.glob("epoch*.ckpt"))
    if saved_ckpts:
        print(f"\nAll saved checkpoints ({len(saved_ckpts)}):")
        for ckpt in saved_ckpts:
            size_mb = ckpt.stat().st_size / (1024*1024)
            print(f"  - {ckpt.name} ({size_mb:.1f} MB)")
    print(f"{'='*70}\n")

    meta = {
        "train_csvs":      args.train_csvs,
        "held_out_csv":    args.held_out_csv,
        "grid_m":          args.grid_m,
        "img_per_place":   args.img_per_place,
        "batch_size":      args.batch_size,
        "max_epochs":      args.max_epochs,
        "lr":              args.lr,
        "best_checkpoint": best_ckpt,
        "elapsed_sec":     elapsed,
        "smoke":           args.smoke,
    }
    meta_path = output_dir / "run_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[train_hilti] Run metadata saved to {meta_path}")
    print(f"[train_hilti] Loss curve        : {output_dir / 'loss_curve.png'}")
    print(f"[train_hilti] Step-level CSV    : {output_dir / 'training_steps.csv'}")
    print(f"[train_hilti] Epoch-level CSV   : {output_dir / 'training_epochs.csv'}")

    # ── Post-smoke instructions ───────────────────────────────────────────────
    if args.smoke:
        print("\n" + "="*60)
        print("SMOKE DONE — next step: run recall on tiny eval set:")
        print(
            "  python src/run_hilti_recall.py \\\n"
            "    --eval_root <your_eval_root_with_db_query_gt> \\\n"
            f"    --ckpt {best_ckpt} \\\n"
            "    --device mps --no_cache"
        )
        print("="*60)

    return best_ckpt


if __name__ == "__main__":
    pl.seed_everything(42, workers=True)
    main()
