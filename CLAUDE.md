# Project: MixVPR — Visual Place Recognition (WACV 2023)

All-MLP feature aggregation for large-scale Visual Place Recognition, fine-tuned on Hilti-Trimble indoor fisheye data.

## Model Routing

| Task type                                      | Model  |
|------------------------------------------------|--------|
| Typo, rename, single-file edit, quick question | haiku  |
| Feature work, tests, refactor, docs            | sonnet |
| Architecture, multi-file debug, multi-agent    | opus   |

## Stack

- Language: Python 3.12
- Framework: PyTorch Lightning (`pytorch_lightning`)
- Key deps: `torch`, `faiss`, `pytorch-metric-learning`, `numpy`, `matplotlib`, `tqdm`
- Venv: `lib/python3.12/` (local venv in repo root — activate before running)

## Commands

```bash
# Fine-tune on Hilti data
python src/train_hilti.py --train_csvs <csv_paths...> --ckpt LOGS/resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt

# Smoke test (1 epoch, 50 places)
python src/train_hilti.py --train_csvs <csv> --ckpt LOGS/... --smoke

# Evaluate recall on Hilti eval set
python src/run_hilti_recall_fixed.py --eval_root <mixvpr_evalset_path> --cam 0 --ckpt <ckpt> --device mps --hilti

# Test pretrained model on standard benchmarks
python src/test_pretrained.py
```

## Directory Structure

- `main.py` — `VPRModel` (PyTorch Lightning module, backbone + aggregator)
- `src/` — training/eval scripts (`train_hilti.py`, `run_hilti_recall_fixed.py`, `demo.py`, `test_pretrained.py`)
- `models/backbones/` — ResNet50, EfficientNet, Swin
- `models/aggregators/` — MixVPR, ConvAP, CosPlace, GeM
- `dataloaders/` — `HiltiDataset.py`, `HiltiDataModule.py`, GSVCities, Mapillary, Pittsburgh
- `utils/` — `losses.py`, `validation.py` (FAISS recall computation)
- `LOGS/` — pretrained checkpoint + fine-tune output dirs
- `datasets/` — numpy index files for MSLS-val

## Conventions

- Named exports only (no default exports)
- Commits: Conventional Commits (`feat:`, `fix:`, `chore:`, `docs:`)

## Ignore

- `docs/` — never read, reference, or modify anything in this folder

## Claude Behavior

- Prefer editing existing files over creating new ones
- Do not add comments or docstrings to code you did not change
- Do not over-engineer — build only what is asked
- Never expose secrets, API keys, or credentials in code or logs

## Gotchas

- **MPS precision**: Use `--precision 32` on Apple Silicon — fp16 causes NaN gradients with `MultiSimilarityLoss`
- **MPS DataLoader**: `pin_memory=False`, `num_workers=0` required to avoid CUDA-pin and fork/FAISS conflicts on macOS
- **OMP deadlock**: `OMP_NUM_THREADS=1` must be set before importing FAISS on macOS
- **Hilti transform**: Eval images need `HILTI_EVAL_TRANSFORM` (rotate 180° + resize 320×320 + normalize) — pass `--hilti` flag
- **Input size**: All models expect 320×320 images
- **Pretrained ckpt**: `LOGS/resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt` — raw state dict (no `state_dict` wrapper), output dim=4096
- **Fine-tune ckpt**: Lightning checkpoints wrap weights under `state_dict` key — both formats handled in load code
- **Hilti data path**: External dataset at `hilti-trimble-slam-challenge-2026/challenge_tools_ros/vpr/data/`
- **MPS PL patch**: `lightning_fabric.accelerators.mps.MPSAccelerator.is_available` patched at runtime for Rosetta compatibility
