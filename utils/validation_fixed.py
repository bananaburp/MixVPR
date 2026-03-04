import numpy as np
import faiss
import faiss.contrib.torch_utils
from prettytable import PrettyTable


def get_validation_recalls(r_list, q_list, k_values, gt, print_results=True, faiss_gpu=False, dataset_name='dataset without name ?', use_cosine=True):
    """
    Calculate recall@K metrics for VPR.
    
    Args:
        r_list: Database descriptors (N_db, dim) - should be L2-normalized if use_cosine=True
        q_list: Query descriptors (N_q, dim) - should be L2-normalized if use_cosine=True
        k_values: List of K values for recall@K (e.g., [1, 5, 10])
        gt: Ground truth array where gt[i] contains DB indices that are positives for query i
        print_results: Whether to print results table
        faiss_gpu: Whether to use GPU for FAISS
        dataset_name: Name of dataset for display
        use_cosine: If True, use cosine similarity (IndexFlatIP), else L2 distance (IndexFlatL2)
    
    Returns:
        Dictionary mapping K values to recall@K scores
    """
    
    embed_size = r_list.shape[1]
    
    # Validate inputs
    if use_cosine:
        # Check if descriptors are normalized (helpful warning)
        db_norms = np.linalg.norm(r_list, axis=1)
        q_norms = np.linalg.norm(q_list, axis=1)
        if not (np.allclose(db_norms, 1.0, atol=1e-3) and np.allclose(q_norms, 1.0, atol=1e-3)):
            print("⚠️  WARNING: use_cosine=True but descriptors don't appear to be L2-normalized!")
            print(f"    DB norms: min={db_norms.min():.4f}, max={db_norms.max():.4f}, mean={db_norms.mean():.4f}")
            print(f"    Query norms: min={q_norms.min():.4f}, max={q_norms.max():.4f}, mean={q_norms.mean():.4f}")
    
    # Build FAISS index
    if faiss_gpu:
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.useFloat16 = True
        flat_config.device = 0
        if use_cosine:
            # Inner product for cosine similarity (requires normalized vectors)
            faiss_index = faiss.GpuIndexFlatIP(res, embed_size, flat_config)
        else:
            # L2 distance
            faiss_index = faiss.GpuIndexFlatL2(res, embed_size, flat_config)
    else:
        if use_cosine:
            # Inner product for cosine similarity (requires normalized vectors)
            faiss_index = faiss.IndexFlatIP(embed_size)
        else:
            # L2 distance
            faiss_index = faiss.IndexFlatL2(embed_size)
    
    # Add database descriptors
    faiss_index.add(r_list)

    # Search for queries in the index
    # For IP (cosine), higher is better; for L2, lower is better
    # FAISS always returns indices sorted by "best match first"
    _, predictions = faiss_index.search(q_list, max(k_values))
    
    # Calculate recall@k
    correct_at_k = np.zeros(len(k_values))
    for q_idx, pred in enumerate(predictions):
        for i, n in enumerate(k_values):
            # Check if any of the top-N predictions are in the ground truth
            if np.any(np.in1d(pred[:n], gt[q_idx])):
                # If found at position K, also found at all K' > K (cumulative)
                correct_at_k[i:] += 1
                break
    
    # Normalize by number of queries
    correct_at_k = correct_at_k / len(predictions)
    d = {k:v for (k,v) in zip(k_values, correct_at_k)}

    if print_results:
        print() # print a new line
        table = PrettyTable()
        table.field_names = ['K']+[str(k) for k in k_values]
        table.add_row(['Recall@K']+ [f'{100*v:.2f}' for v in correct_at_k])
        similarity_type = "Cosine" if use_cosine else "L2"
        print(table.get_string(title=f"Performances on {dataset_name} ({similarity_type})"))
    
    return d
