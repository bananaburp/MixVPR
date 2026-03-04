"""
HiltiDataModule.py — PyTorch Lightning DataModule for Hilti-Trimble fine-tuning.

WHY no val_dataloader by default?
  The Hilti validation workflow is handled externally by src/run_hilti_recall.py
  (which builds a db/query eval set and computes Recall@K with FAISS).  Adding
  an internal validation step would require duplicating that FAISS logic inside
  LightningModule and would introduce confusion.  We deliberately skip internal
  val and rely on the external recall script for the held-out run.

  If you want lightweight in-training validation, pass val_csv_files; the
  DataModule will expose a simple flat DataLoader over those images so the
  trainer framework does not crash.  However, recall metrics are NOT computed
  inline — only train loss is tracked.
"""

from __future__ import annotations

from typing import List, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from dataloaders.HiltiDataset import (
    HiltiDataset,
    HILTI_TRAIN_TRANSFORM,
    HILTI_EVAL_TRANSFORM,
)


class HiltiDataModule(pl.LightningDataModule):
    """LightningDataModule wrapping HiltiDataset.

    Parameters
    ----------
    train_csv_files : list[str]
        Paths to aligned.csv for the *training* runs (e.g. 4 runs in
        leave-one-out).
    val_csv_files : list[str], optional
        Paths to aligned.csv for a val run.  If None the module has no
        validation loader (monitoring is train loss only).
    batch_size : int
        Number of *places* per batch (B in B × N × C × H × W).  Default 16.
    img_per_place : int
        Images sampled per place (N).  Default 4.
    min_img_per_place : int
        Minimum images a place must have to be non-trivially included.
        Defaults to img_per_place.
    grid_m : float
        XY grid cell size in metres for place labelling.  Default 1.0.
    num_workers : int
        DataLoader worker processes.  Keep low on macOS / MPS.  Default 4.
    cap_places : int, optional
        Smoke-test: cap total training places to this number.
    """

    def __init__(
        self,
        train_csv_files: List[str],
        val_csv_files: Optional[List[str]] = None,
        batch_size: int = 16,
        img_per_place: int = 4,
        min_img_per_place: Optional[int] = None,
        grid_m: float = 1.0,
        num_workers: int = 4,
        cap_places: Optional[int] = None,
    ):
        super().__init__()
        self.train_csv_files  = train_csv_files
        self.val_csv_files    = val_csv_files
        self.batch_size       = batch_size
        self.img_per_place    = img_per_place
        self.min_img_per_place = min_img_per_place if min_img_per_place is not None \
                                  else img_per_place
        self.grid_m       = grid_m
        self.num_workers  = num_workers
        self.cap_places   = cap_places

        # Populated in setup()
        self.train_dataset: Optional[HiltiDataset] = None
        self.val_dataset:   Optional[HiltiDataset] = None
        # Keep val_datasets / val_set_names consistent with VPRModel expectations
        self.val_datasets: list = []
        self.val_set_names: list = []

    # ------------------------------------------------------------------
    # LightningDataModule interface
    # ------------------------------------------------------------------

    def setup(self, stage: Optional[str] = None):
        if stage in ("fit", None):
            print("\n[HiltiDataModule] Building training dataset …")
            self.train_dataset = HiltiDataset(
                csv_files=self.train_csv_files,
                img_per_place=self.img_per_place,
                min_img_per_place=self.min_img_per_place,
                grid_m=self.grid_m,
                transform=HILTI_TRAIN_TRANSFORM,
                cap_places=self.cap_places,
                random_sample=True,
            )
            print(
                f"[HiltiDataModule] train: {len(self.train_dataset)} places, "
                f"{self.train_dataset.total_images} images, "
                f"grid={self.grid_m} m, "
                f"batch={self.batch_size}×{self.img_per_place}"
            )
            print(
                f"[HiltiDataModule] ~{len(self.train_dataset) // max(self.batch_size, 1)} "
                f"iterations per epoch\n"
            )

            if self.val_csv_files:
                print("[HiltiDataModule] Building validation dataset …")
                self.val_dataset = HiltiDataset(
                    csv_files=self.val_csv_files,
                    img_per_place=self.img_per_place,
                    min_img_per_place=self.min_img_per_place,
                    grid_m=self.grid_m,
                    transform=HILTI_EVAL_TRANSFORM,
                    random_sample=False,
                )
                # Expose for VPRModel.validation_epoch_end compatibility
                self.val_datasets   = [self.val_dataset]
                self.val_set_names  = ["hilti_val"]

    def train_dataloader(self) -> DataLoader:
        """Reload the dataset every epoch to re-shuffle place sampling."""
        self.setup("fit")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,   # pin_memory causes issues on some MPS setups
            drop_last=True,     # avoid partial batches that confuse the loss
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        """
        Returns a flat DataLoader over (images, labels) place-batches from
        val_csv_files.  The trainer will call validation_step, but we do NOT
        compute Recall@K here — that is done externally via run_hilti_recall.py.
        If val_csv_files was not provided this returns an empty list so the
        trainer skips validation silently.
        """
        if self.val_dataset is None:
            return []
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=max(self.num_workers // 2, 1),
            pin_memory=False,
            drop_last=False,
        )
