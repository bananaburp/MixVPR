"""
HiltiDataset.py — MixVPR-compatible Dataset for the Hilti-Trimble SLAM
Challenge 2026 dataset.

WHY a separate file instead of editing GSVCitiesDataset?
    GSVCities uses a city-based CSV structure entirely different from Hilti.
    Adding Hilti support there would tangle two unrelated domains.  We add a
    clean, self-contained class that follows the exact interface required by
    VPRModel.training_step (batch = (places, labels) with shape
    (BS, N, C, H, W) and (BS*N,)).

──────────────────────────────────────────────────────────────────────────────
EXPECTED aligned.csv SCHEMA (produced by align_frames_to_gt.py)
──────────────────────────────────────────────────────────────────────────────
    timestamp  : float  — image capture time (seconds, ROS nanoseconds / 1e9)
    tx, ty, tz : float  — metric XYZ translation from GT TUM pose
    qx,qy,qz,qw: float  — quaternion rotation (not used for place labels here)
    image_path  : str   — absolute path to the extracted JPEG
                OR a path relative to the directory that contains aligned.csv.

Optionally a separate cam1 CSV (same columns) can be provided; its images are
added to the same place labels as cam0, increasing intra-place diversity.

──────────────────────────────────────────────────────────────────────────────
PLACE LABEL STRATEGY
──────────────────────────────────────────────────────────────────────────────
    label = (run_id, floor(tx / grid_m), floor(ty / grid_m))

    WHY XY only? The robot operates on floors; z varies little.  Two frames in
    the same (binned_x, binned_y) cell are visually close enough to be
    positives in metric learning.

    WHY grid_m = 1.0 m by default?
    At ~10 Hz the robot moves ~0.2–0.5 m/frame.  A 1 m cell captures ~3–5
    frames — ideal for img_per_place = 4.  Larger cells make places less
    distinctive (false positives in retrieval loss).

──────────────────────────────────────────────────────────────────────────────
WHY ROTATE 180° IN THE TRANSFORM?
──────────────────────────────────────────────────────────────────────────────
    The Hilti rig mounts both cameras upside down.  Correcting in the
    torchvision transform (rather than during extraction) keeps raw images
    intact on disk and guarantees the same fix is applied in EVERY code path —
    training, smoke test, and recall evaluation — as long as all paths import
    HILTI_TRAIN_TRANSFORM / HILTI_EVAL_TRANSFORM from this module.

──────────────────────────────────────────────────────────────────────────────
WHY 5–10 Hz EXTRACTION INSTEAD OF 30 Hz?
──────────────────────────────────────────────────────────────────────────────
    At 30 Hz consecutive frames are < 0.2 m apart; the MultiSimilarityLoss
    needs visually DISTINCT positives (same place, different viewpoint) not
    near-duplicate frames.  10 Hz yields ~3–5 frames per 1 m grid cell, which
    is enough for img_per_place=4 without flooding with duplicates.

──────────────────────────────────────────────────────────────────────────────
DUAL CAMERA STRATEGY (Option 1 — recommended)
──────────────────────────────────────────────────────────────────────────────
    cam0 and cam1 are mounted on the same rig and capture the same location at
    the same time (just different viewing directions).  We treat both as
    additional images belonging to the SAME place label.  This naturally
    increases intra-place variety without requiring any special loss change.
    If only cam0 is available the dataset degrades gracefully (cam1_csv=None).
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

# ──────────────────────────────────────────────────────────────────────────────
# Canonical transforms — import these in eval scripts too so train/eval match.
# ──────────────────────────────────────────────────────────────────────────────

# Rotate 180° deterministically.
# Named callable (not a lambda) so it survives pickle by multiprocessing workers
# when num_workers > 0. T.Lambda(lambda ...) is not picklable on macOS (spawn).
class _Rotate180:
    def __call__(self, img):
        return img.rotate(180)

_ROTATE_180 = _Rotate180()

# ImageNet statistics (same as demo.py / GSVCitiesDataloader.py)
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

HILTI_TRAIN_TRANSFORM = T.Compose([
    _ROTATE_180,                                        # fix upside-down rig mount
    T.Resize((320, 320), interpolation=T.InterpolationMode.BICUBIC),
    T.ColorJitter(brightness=0.3, contrast=0.3,         # mild augmentation for
                  saturation=0.2, hue=0.05),            # indoor lighting variation
    T.ToTensor(),
    T.Normalize(mean=_MEAN, std=_STD),
])

HILTI_EVAL_TRANSFORM = T.Compose([
    _ROTATE_180,                                        # must match training!
    T.Resize((320, 320), interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=_MEAN, std=_STD),
])


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class HiltiDataset(Dataset):
    """Metric-learning dataset for one or more Hilti runs.

    Parameters
    ----------
    csv_files : list[str | Path]
        Paths to aligned.csv files (one per run).  Every CSV must have columns:
        ``timestamp, tx, ty, tz, qx, qy, qz, qw, image_path``.
    img_per_place : int
        Number of images sampled per place per training step (= N in the
        (BS, N, C, H, W) batch).  Default 4.
    min_img_per_place : int
        Minimum images a place must have to be included.  Typically equal to
        img_per_place.
    grid_m : float
        XY grid cell side length in metres.  Places are defined as
        (run_id, floor(tx/grid_m), floor(ty/grid_m)).  Default 1.0 m.
    transform : callable, optional
        Image transform applied to each PIL image.  Defaults to
        HILTI_TRAIN_TRANSFORM (180° rotation + resize 320 + normalize).
    cap_places : int, optional
        If set, randomly sub-sample this many places.  Useful for smoke tests.
    random_sample : bool
        If True (default), randomly pick img_per_place images per place at
        each __getitem__ call.  If False, always take the first N images
        (deterministic; useful for debugging).
    """

    def __init__(
        self,
        csv_files: List[str],
        img_per_place: int = 4,
        min_img_per_place: int = 4,
        grid_m: float = 1.0,
        transform=None,
        cap_places: Optional[int] = None,
        random_sample: bool = True,
    ):
        super().__init__()
        assert img_per_place <= min_img_per_place, (
            f"img_per_place ({img_per_place}) must be <= "
            f"min_img_per_place ({min_img_per_place})"
        )
        self.img_per_place  = img_per_place
        self.min_img_per_place = min_img_per_place
        self.grid_m         = grid_m
        self.transform      = transform if transform is not None else HILTI_TRAIN_TRANSFORM
        self.random_sample  = random_sample

        df = self._load_and_label(csv_files, grid_m)
        df, stats = self._filter_places(df, min_img_per_place)
        self._print_stats(stats)

        if cap_places is not None and cap_places < len(df["place_id"].unique()):
            rng = np.random.default_rng(0)
            kept = rng.choice(df["place_id"].unique(), cap_places, replace=False)
            df   = df[df["place_id"].isin(kept)]
            print(f"[HiltiDataset] smoke-cap: kept {cap_places} places out of "
                  f"{stats['total_places_before_filter']} (after filter).")

        # Build a mapping place_id -> list[image_path]
        self.place_to_imgs: dict[int, list[str]] = (
            df.groupby("place_id")["image_path"]
              .apply(list)
              .to_dict()
        )
        self.place_ids = list(self.place_to_imgs.keys())
        self.total_images = len(df)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _remap_container_path(img_path: str, csv_path: Path) -> str:
        """Re-root a Docker/container absolute path to the local filesystem.

        Walks upward through csv_path's ancestors looking for a directory name
        that also appears in img_path's components.  When found, it replaces
        the foreign prefix with the local ancestor root.

        Example
        -------
        csv_path  = /local/hilti-repo/data/run1/cam0.csv
        img_path  = /ros2_ws/src/hilti-repo/data/run1/frames/img.jpg
        anchor    = "hilti-repo"
        result    = /local/hilti-repo/data/run1/frames/img.jpg
        """
        p = Path(img_path)
        if p.exists():
            return img_path
        img_parts = p.parts
        for i, part in enumerate(csv_path.parts):
            if part in img_parts:
                j = img_parts.index(part)
                local_root = Path(*csv_path.parts[:i]) if i > 0 else Path("/")
                suffix = Path(*img_parts[j:])
                candidate = local_root / suffix
                if candidate.exists():
                    return str(candidate)
        return img_path  # unchanged — will be filtered out by exists_mask below

    @staticmethod
    def _load_and_label(csv_files: List[str], grid_m: float) -> pd.DataFrame:
        """Load all CSVs, assign run_id, compute integer place labels."""
        dfs = []
        for run_id, path in enumerate(csv_files):
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"aligned.csv not found: {path}")
            df = pd.read_csv(path)

            # Accept 'img_path' as an alias for 'image_path' (Hilti CSV format)
            if "image_path" not in df.columns and "img_path" in df.columns:
                df = df.rename(columns={"img_path": "image_path"})

            required = {"tx", "ty", "image_path"}
            missing = required - set(df.columns)
            if missing:
                raise ValueError(
                    f"{path} is missing columns: {missing}. "
                    "Expected: timestamp, tx, ty, tz, qx, qy, qz, qw, image_path "
                    "(also accepts img_path as alias for image_path)"
                )
            df["run_id"] = run_id

            # Resolve relative image paths.
            # Paths in the CSV may be relative to CWD (repo root) or to the
            # CSV's own directory.  Try CWD-relative first (most common),
            # fall back to csv_dir-relative.
            csv_dir = path.parent
            def _resolve_path(p: str) -> str:
                if Path(p).is_absolute():
                    return str(p)
                cwd_rel = Path(p)          # relative to CWD
                if cwd_rel.exists():
                    return str(cwd_rel)
                return str(csv_dir / p)    # fall back to CSV-directory-relative
            df["image_path"] = df["image_path"].apply(_resolve_path)

            # Remap container/Docker paths that don't exist locally
            first_path = df["image_path"].iloc[0] if len(df) > 0 else ""
            if first_path and not Path(first_path).exists():
                df["image_path"] = df["image_path"].apply(
                    lambda p: HiltiDataset._remap_container_path(p, path)
                )

            # Drop rows whose image file does not exist
            exists_mask = df["image_path"].apply(lambda p: Path(p).exists())
            n_missing   = (~exists_mask).sum()
            if n_missing > 0:
                warnings.warn(
                    f"{path.name}: {n_missing} image paths not found on disk "
                    "— rows dropped."
                )
            df = df[exists_mask].reset_index(drop=True)

            if len(df) == 0:
                warnings.warn(
                    f"{path.name}: no valid image paths remain after filtering "
                    "— skipping this CSV entirely."
                )
                continue

            # Grid-bin in XY to get a unique integer per cell per run.
            # We encode as: run_id * LARGE_PRIME + (binned_x * STRIDE + binned_y)
            # using a large-enough stride so labels stay unique across runs.
            STRIDE     = 100_000        # max ~100 km range per axis in 1 m cells
            BIG        = STRIDE * STRIDE
            bin_x = np.floor(df["tx"].values / grid_m).astype(np.int64)
            bin_y = np.floor(df["ty"].values / grid_m).astype(np.int64)
            # shift to non-negative (poses can be negative)
            bin_x -= bin_x.min()
            bin_y -= bin_y.min()
            df["place_id"] = (
                run_id * BIG + bin_x * STRIDE + bin_y
            ).astype(np.int64)

            dfs.append(df)

        if not dfs:
            raise RuntimeError(
                "No valid images found in any of the provided CSV files. "
                "Check that image paths in the CSVs resolve to existing files."
            )
        return pd.concat(dfs, ignore_index=True)

    @staticmethod
    def _filter_places(
        df: pd.DataFrame, min_img: int
    ) -> Tuple[pd.DataFrame, dict]:
        counts = df.groupby("place_id")["place_id"].transform("size")
        total  = df["place_id"].nunique()
        kept   = df[counts >= min_img]
        kept_n = kept["place_id"].nunique()
        dropped_pct = 100.0 * (total - kept_n) / max(total, 1)
        stats = {
            "total_places_before_filter":  total,
            "places_after_filter": kept_n,
            "dropped_percent": dropped_pct,
            "total_images": len(kept),
        }
        return kept.reset_index(drop=True), stats

    @staticmethod
    def _print_stats(stats: dict) -> None:
        print(
            f"[HiltiDataset] places total={stats['total_places_before_filter']}  "
            f"kept={stats['places_after_filter']}  "
            f"dropped={stats['dropped_percent']:.1f}%  "
            f"images={stats['total_images']}"
        )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Returns number of valid places (not images)."""
        return len(self.place_ids)

    def __getitem__(self, index: int):
        """Return (images_tensor, labels_tensor).

        Returns
        -------
        images : Tensor[img_per_place, C, H, W]
        labels : Tensor[img_per_place]   — all equal to place_id
        """
        place_id  = self.place_ids[index]
        img_paths = self.place_to_imgs[place_id]

        if self.random_sample:
            chosen = np.random.choice(img_paths, self.img_per_place, replace=False)
        else:
            chosen = img_paths[: self.img_per_place]

        imgs = []
        for p in chosen:
            pil = Image.open(p).convert("RGB")
            img = self.transform(pil)
            imgs.append(img)

        images_t = torch.stack(imgs)                              # (N, C, H, W)
        labels_t = torch.full(
            (self.img_per_place,), fill_value=int(place_id % 2**31), dtype=torch.long
        )                                                         # (N,)
        return images_t, labels_t
