import os, sys, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch

from main import VPRModel
from demo import BaseDataset, InferencePipeline
from utils.validation import get_validation_recalls

# --------------------------------------------------------------
# python src/run_hilti_recall.py \
#     --eval_root "/Volumes/T9/DevSpace/Github/hilti-trimble-slam-challenge-2026/challenge_tools_ros/vpr/eval/floor_1_2025-05-05_run_1_cam0" \
#     --ckpt "./LOGS/resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt" \
#     --device mps \
#     --batch 16
# --------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_root", required=True)   # folder containing db/ and query/ and gt_positives.npy
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--no_cache", action="store_true", help="Ignore cached descriptors and recompute from scratch")
    args = ap.parse_args()

    model = VPRModel(
        backbone_arch="resnet50",
        layers_to_crop=[4],
        agg_arch="MixVPR",
        agg_config={"in_channels":1024,"in_h":20,"in_w":20,"out_channels":1024,"mix_depth":4,"mlp_ratio":1,"out_rows":4},
    )
    print(f"[1/6] Loading checkpoint from {args.ckpt} ...")
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    # device = torch.device(args.device)
    # model = model.to(device)
    # print(f"device: {device} model loaded from {args.ckpt}")

    db_path = os.path.join(args.eval_root, "db")
    q_path  = os.path.join(args.eval_root, "query")
    gt_path = os.path.join(args.eval_root, "gt_positives.npy")

    print(f"[2/6] Building datasets ...")
    db_ds = BaseDataset(db_path)
    q_ds  = BaseDataset(q_path)
    print(f"      db={len(db_ds)} images  query={len(q_ds)} images")

    # infer feature dim by running one image
    print(f"[3/6] Inferring feature dimension ...")
    with torch.no_grad():
        x, _ = db_ds[0]
        y = model(x.unsqueeze(0))
        feat_dim = y.shape[1]
    print(f"      feat_dim={feat_dim}")

    db_pipe = InferencePipeline(model, db_ds, feature_dim=feat_dim, batch_size=args.batch, num_workers=2, device=args.device, no_cache=args.no_cache)
    q_pipe  = InferencePipeline(model, q_ds,  feature_dim=feat_dim, batch_size=args.batch, num_workers=2, device=args.device, no_cache=args.no_cache)

    print(f"[4/6] Extracting DB descriptors (batch={args.batch}) ...")
    db_desc = db_pipe.run(split="db").astype(np.float32)
    print(f"      db_desc shape: {db_desc.shape}")

    print(f"[5/6] Extracting query descriptors (batch={args.batch}) ...")
    q_desc  = q_pipe.run(split="query").astype(np.float32)
    print(f"      q_desc shape: {q_desc.shape}")

    print(f"[6/6] Loading ground truth from {gt_path} ...")
    gt = np.load(gt_path, allow_pickle=True)

    print(f"db_desc={db_desc.shape} q_desc={q_desc.shape} gt_queries={len(gt)}")

    # gt may cover only a subset of queries (e.g. 76 of 152);
    # trim q_desc to match so recall indexing doesn't go out of bounds
    if len(gt) < len(q_desc):
        print(f"      gt covers {len(gt)} queries; trimming q_desc from {len(q_desc)} → {len(gt)}")
        q_desc = q_desc[:len(gt)]

    print("\nCalculating recalls...")
    recalls = get_validation_recalls(
        r_list=db_desc,
        q_list=q_desc,
        k_values=[1,5,10],
        gt=gt,
        print_results=True,
        faiss_gpu=False,
        dataset_name=os.path.basename(args.eval_root)
    )
    print("recalls:", recalls)

if __name__ == "__main__":
    main()