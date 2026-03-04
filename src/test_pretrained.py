"""
Quick test script for pretrained MixVPR model
Tests on Pittsburgh30k-val dataset
"""

import torch
import numpy as np
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm

from main import VPRModel
from dataloaders.PittsburgDataset import get_whole_val_set
from utils.validation import get_validation_recalls


def test_pretrained_model(ckpt_path: str, device='cuda'):
    """
    Test pretrained MixVPR on Pittsburgh30k validation dataset
    
    Args:
        ckpt_path: Path to the pretrained checkpoint file
        device: 'cuda' or 'cpu'
    """
    
    print("=" * 80)
    print("Testing Pretrained MixVPR Model")
    print("=" * 80)
    
    # ===== STEP 1: Initialize Model =====
    print("\n[1/3] Initializing model...")
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
    
    # Load pretrained weights
    try:
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"✓ Loaded pretrained weights from {ckpt_path}")
    except Exception as e:
        print(f"✗ Error loading checkpoint: {e}")
        print(f"  Make sure {ckpt_path} exists!")
        return
    
    model = model.to(device)
    model.eval()
    
    # ===== STEP 2: Load Validation Dataset =====
    print("\n[2/3] Loading Pittsburgh30k-val dataset...")
    
    transform = T.Compose([
        T.Resize((320, 320)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225])
    ])
    
    try:
        val_dataset = get_whole_val_set(input_transform=transform)
        print(f"✓ Dataset loaded: {len(val_dataset)} images")
        print(f"  - Database images: {val_dataset.dbStruct.numDb}")
        print(f"  - Query images: {val_dataset.dbStruct.numQ}")
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        print(f"  Make sure Pittsburgh dataset is in '../datasets/Pittsburgh/'")
        return
    
    # Create dataloader
    dataloader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # ===== STEP 3: Extract Descriptors =====
    print("\n[3/3] Extracting descriptors from all images...")
    
    all_descriptors = []
    
    with torch.no_grad():
        for batch_images, _ in tqdm(dataloader, desc="Inference", ncols=80):
            batch_images = batch_images.to(device)
            descriptors = model(batch_images)
            all_descriptors.append(descriptors.cpu())
    
    all_descriptors = torch.cat(all_descriptors, dim=0).numpy()
    print(f"✓ Extracted {all_descriptors.shape[0]} descriptors")
    print(f"  Descriptor shape: {all_descriptors.shape}")
    
    # ===== STEP 4: Calculate Recall@K =====
    print("\n[4/4] Computing Recall@K...")
    
    # Split references and queries
    num_references = val_dataset.dbStruct.numDb
    r_list = all_descriptors[:num_references].astype(np.float32)
    q_list = all_descriptors[num_references:].astype(np.float32)
    
    # Get positives (ground truth matches)
    positives = val_dataset.getPositives()
    
    # Calculate Recall@K
    results = get_validation_recalls(
        r_list=r_list,
        q_list=q_list,
        k_values=[1, 5, 10, 15, 20, 50, 100],
        gt=positives,
        print_results=True,
        dataset_name='Pittsburgh30k-val'
    )
    
    print("\n" + "=" * 80)
    print("Test Complete!")
    print("=" * 80)
    
    return results


if __name__ == '__main__':
    import sys
    
    # Default checkpoint path
    ckpt_path = './LOGS/resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt'
    
    # Allow custom checkpoint path as argument
    if len(sys.argv) > 1:
        ckpt_path = sys.argv[1]
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    test_pretrained_model(ckpt_path, device=device)
