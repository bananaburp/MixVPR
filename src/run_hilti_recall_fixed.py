import os, sys, argparse
# Must be set before faiss is imported to prevent OpenMP thread-pool deadlock on macOS.
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
from torch.utils import data as torch_data
from PIL import Image
from tqdm import tqdm

from main import VPRModel
from demo import BaseDataset, InferencePipeline
from utils.validation import get_validation_recalls
from dataloaders.HiltiDataset import HILTI_EVAL_TRANSFORM

def _dbg(msg: str):
    """Print a debug line and immediately flush stdout."""
    print(f"[DBG] {msg}", flush=True)


'''
python src/run_hilti_recall_fixed.py \
    --eval_root "/Volumes/T9/DevSpace/Github/hilti-trimble-slam-challenge-2026/challenge_tools_ros/vpr/data/floor_1_2025-05-05_run_1/eval/mixvpr_evalset" \
    --cam 0 \
    --ckpt ".LOGS/hilti_finetune/fold_floor1/last.ckpt" \
    --device mps \
    --batch 16 \
    --hilti \
    --label "pretrained"\
    --no_cache


python src/run_hilti_recall_fixed.py \
    --eval_root "/Volumes/T9/DevSpace/Github/hilti-trimble-slam-challenge-2026/challenge_tools_ros/vpr/data/floor_1_2025-05-05_run_1/eval/mixvpr_evalset" \
    --cam 0 \
    --ckpt "/Volumes/T9/DevSpace/Github/MixVPR/LOGS/smoke_test/last-v1.ckpt" \
    --device mps \
    --batch 16 \
    --hilti \
    --label "pretrained"\
    --no_cache
'''


class CameraSpecificInferencePipeline(InferencePipeline):
    """InferencePipeline that includes camera info in cache keys to avoid collisions."""
    def __init__(self, model, dataset, feature_dim, camera_id, batch_size=4, num_workers=0, device='cuda', no_cache=False):
        super().__init__(model, dataset, feature_dim, batch_size, num_workers, device, no_cache)
        self.camera_id = camera_id
        # Re-create the DataLoader with pin_memory=False.
        # pin_memory=True (hardcoded in the base class) stalls on macOS MPS because
        # PyTorch tries to allocate CUDA-pinned memory which does not exist on MPS.
        self.dataloader = torch_data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,          # 0 = no forked workers → no fork/FAISS conflict
            pin_memory=False,       # False = no CUDA-pin attempt on MPS
            drop_last=False,
        )
    
    def run(self, split: str = 'db') -> np.ndarray:
        # Generate camera-specific cache path
        dataset_tag = os.path.basename(os.path.dirname(self.dataset.img_path.rstrip('/')))
        cache_path = f'./LOGS/global_descriptors_{dataset_tag}_cam{self.camera_id}_{split}.npy'
        
        if os.path.exists(cache_path) and not self.no_cache:
            print(f"Skipping {split} features extraction, loading from cache ({cache_path})", flush=True)
            return np.load(cache_path)
        
        # Otherwise compute descriptors
        _dbg(f"{split}: moving model to {self.device}")
        self.model.to(self.device)
        _dbg(f"{split}: starting DataLoader loop over {len(self.dataset)} images")
        with torch.no_grad():
            global_descriptors = np.zeros((len(self.dataset), self.feature_dim))
            for i, batch in enumerate(tqdm(self.dataloader, ncols=100, desc=f'Extracting {split} features')):
                if i == 0:
                    _dbg(f"{split}: first batch received, sending to {self.device}")
                imgs, indices = batch
                imgs = imgs.to(self.device)
                if i == 0:
                    _dbg(f"{split}: first batch on device, running model forward pass")
                descriptors = self.model(imgs)
                if i == 0:
                    _dbg(f"{split}: first batch forward pass done, shape={descriptors.shape}")
                descriptors = descriptors.detach().cpu().numpy()
                global_descriptors[np.array(indices), :] = descriptors
        
        _dbg(f"{split}: extraction complete, saving to {cache_path}")
        np.save(cache_path, global_descriptors)
        return global_descriptors


class HiltiBaseDataset(BaseDataset):
    """BaseDataset variant that applies HILTI_EVAL_TRANSFORM (rotate 180° + resize + normalize).
    Use this for any run extracted from the Hilti rig so that eval matches fine-tuning training.
    """
    def __getitem__(self, index):
        pil = Image.open(self.img_path_list[index]).convert("RGB")
        return HILTI_EVAL_TRANSFORM(pil), index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_root", required=True,
                    help="Path to mixvpr_evalset folder (e.g., floor_1_2025-05-05_run_1/eval/mixvpr_evalset)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cam", type=int, required=True, choices=[0, 1],
                    help="Camera ID (0 or 1) - determines which db_camN and query_camN folders to use")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--no_cache", action="store_true", help="Ignore cached descriptors and recompute from scratch")
    ap.add_argument("--hilti", action="store_true",
                    help=   "Apply Hilti-specific eval transform (rotate 180° + resize 320 + normalize). "
                            "Required when evaluating fine-tuned Hilti model.")
    ap.add_argument("--label", default=None,
                    help="Human-readable label printed in results (e.g. 'pretrained' or 'finetuned').")
    args = ap.parse_args()

    model = VPRModel(
        backbone_arch="resnet50",
        layers_to_crop=[4],
        agg_arch="MixVPR",
        agg_config={"in_channels":1024,"in_h":20,"in_w":20,"out_channels":1024,"mix_depth":4,"mlp_ratio":1,"out_rows":4},
    )
    print(f"[1/6] Loading checkpoint from {args.ckpt} ...")
    state = torch.load(args.ckpt, map_location="cpu")
    # Lightning checkpoints wrap weights under "state_dict"; raw exports don't.
    if "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()

    db_path = os.path.join(args.eval_root, f"db_cam{args.cam}")
    q_path  = os.path.join(args.eval_root, f"query_cam{args.cam}")
    gt_path = os.path.join(args.eval_root, f"gt_positives.npy")

    # Verify paths exist
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"DB path not found: {db_path}")
    if not os.path.exists(q_path):
        raise FileNotFoundError(f"Query path not found: {q_path}")
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"GT path not found: {gt_path}")
    
    print(f"[2/6] Building datasets (cam={args.cam}, transform={'hilti' if args.hilti else 'default'}) ...")
    DS = HiltiBaseDataset if args.hilti else BaseDataset
    db_ds = DS(db_path)
    q_ds  = DS(q_path)
    print(f"      db={len(db_ds)} images  query={len(q_ds)} images")

    # infer feature dim by running one image
    print(f"[3/6] Inferring feature dimension ...")
    with torch.no_grad():
        x, _ = db_ds[0]
        y = model(x.unsqueeze(0))
        feat_dim = y.shape[1]
    print(f"      feat_dim={feat_dim}")

    db_pipe = CameraSpecificInferencePipeline(model, db_ds, feature_dim=feat_dim, camera_id=args.cam, batch_size=args.batch, num_workers=0, device=args.device, no_cache=args.no_cache)
    q_pipe  = CameraSpecificInferencePipeline(model, q_ds,  feature_dim=feat_dim, camera_id=args.cam, batch_size=args.batch, num_workers=0, device=args.device, no_cache=args.no_cache)

    print(f"[4/6] Extracting DB descriptors (batch={args.batch}) ...", flush=True)
    db_desc = db_pipe.run(split="db").astype(np.float32)
    print(f"      db_desc shape: {db_desc.shape}", flush=True)

    print(f"[5/6] Extracting query descriptors (batch={args.batch}) ...", flush=True)
    q_desc  = q_pipe.run(split="query").astype(np.float32)
    print(f"      q_desc shape: {q_desc.shape}", flush=True)

    # Release pipeline objects (frees any lingering DataLoader resources)
    del db_pipe, q_pipe

    print(f"[6/6] Loading ground truth from {gt_path} ...", flush=True)
    gt = np.load(gt_path, allow_pickle=True)

    print(f"db_desc={db_desc.shape} q_desc={q_desc.shape} gt_queries={len(gt)}")

    # gt may cover only a subset of queries (e.g. 76 of 152);
    # trim q_desc to match so recall indexing doesn't go out of bounds
    if len(gt) < len(q_desc):
        print(f"      gt covers {len(gt)} queries; trimming q_desc from {len(q_desc)} → {len(gt)}")
        q_desc = q_desc[:len(gt)]

    # FIX 1: Check for and report empty GT entries
    empty_gt_indices = [i for i, g in enumerate(gt) if len(g) == 0]
    if empty_gt_indices:
        print(f"\n⚠️  WARNING: {len(empty_gt_indices)} queries have empty GT positives: {empty_gt_indices}")
        print(f"    These queries will be excluded from recall calculation to avoid deflating metrics.")
    
    # FIX 2: Validate GT indices don't exceed DB size
    max_gt_idx = max([max(g) if len(g) > 0 else -1 for g in gt])
    if max_gt_idx >= len(db_desc):
        raise ValueError(f"GT contains DB index {max_gt_idx} but DB only has {len(db_desc)} images!")
    
    # FIX 3: Filter out queries with empty GT
    valid_indices = [i for i, g in enumerate(gt) if len(g) > 0]
    if len(valid_indices) < len(gt):
        print(f"    Filtering: {len(valid_indices)} queries with GT / {len(gt)} total queries")
        q_desc_filtered = q_desc[valid_indices]
        gt_filtered = gt[valid_indices]
    else:
        q_desc_filtered = q_desc
        gt_filtered = gt
    
    # FIX 4: Normalize descriptors for cosine similarity
    print("\nNormalizing descriptors for cosine similarity...")
    db_desc_norm = db_desc / np.linalg.norm(db_desc, axis=1, keepdims=True)
    q_desc_norm = q_desc_filtered / np.linalg.norm(q_desc_filtered, axis=1, keepdims=True)
    print(f"      Normalized shapes: db={db_desc_norm.shape}, query={q_desc_norm.shape}")

    # Generate run label with camera info
    if args.label:
        run_label = f"{args.label}_cam{args.cam}"
    else:
        # Extract run name from eval_root (e.g., floor_1_2025-05-05_run_1 from path)
        eval_parent = os.path.dirname(os.path.dirname(args.eval_root))  # go up from mixvpr_evalset/eval
        run_name = os.path.basename(eval_parent) if os.path.exists(eval_parent) else "unknown"
        run_label = f"{run_name}_cam{args.cam}"
    
    _dbg("building FAISS index")
    print("\nCalculating recalls...", flush=True)
    recalls = get_validation_recalls(
        r_list=db_desc_norm,
        q_list=q_desc_norm,
        k_values=[1,5,10],
        gt=gt_filtered,
        print_results=True,
        faiss_gpu=False,
        dataset_name=run_label
    )
    print("recalls:", recalls, flush=True)
    _dbg("normalized recall done")
    
    # Also compute on unnormalized for comparison
    _dbg("building second FAISS index (unnormalized)")
    print("\n[DEBUG] Recalls on unnormalized descriptors (for comparison):", flush=True)
    recalls_unnorm = get_validation_recalls(
        r_list=db_desc,
        q_list=q_desc_filtered,
        k_values=[1,5,10],
        gt=gt_filtered,
        print_results=True,
        faiss_gpu=False,
        dataset_name=f"{run_label}_unnormalized"
    )

if __name__ == "__main__":
    main()
