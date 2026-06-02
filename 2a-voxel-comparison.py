import os
import os.path as op
import numpy as np
import pandas as pd
from nilearn import image
from nilearn.reporting import get_clusters_table
from scipy.stats import spearmanr

# ---------------------------------------------------------
# 1. Define Paths 
# ---------------------------------------------------------
data_dir = "./dset"
deriv_dir = op.join(data_dir, "derivatives")
analysis_dir = op.join(deriv_dir, "hb-conn")

group_drawn_dir = op.join(analysis_dir, "group-drawn/habenula")
group_atlas_dir = op.join(analysis_dir, "group-atlas/habenula")

# Maps to process
maps_to_process = {
    "1s": {
        "drawn_thresholded": op.join(group_drawn_dir, "averaged", "sub-group_task-rest_desc-1SampletTest_thresh.nii.gz"),
        "atlas_thresholded": op.join(group_atlas_dir, "averaged", "sub-group_task-rest_desc-1SampletTest_thresh.nii.gz"),
        "drawn_unthresholded": op.join(group_drawn_dir, "averaged", "sub-group_task-rest_desc-1SampletTest.nii.gz"),
        "atlas_unthresholded": op.join(group_atlas_dir, "averaged", "sub-group_task-rest_desc-1SampletTest.nii.gz"),
    },
    "2s": {
        "drawn_thresholded": op.join(group_drawn_dir, "difference", "sub-group_task-rest_desc-2SampletTest_thresh.nii.gz"),
        "atlas_thresholded": op.join(group_atlas_dir, "difference", "sub-group_task-rest_desc-2SampletTest_thresh.nii.gz"),
        "drawn_unthresholded": op.join(group_drawn_dir, "difference", "sub-group_task-rest_desc-2SampletTest.nii.gz"),
        "atlas_unthresholded": op.join(group_atlas_dir, "difference", "sub-group_task-rest_desc-2SampletTest.nii.gz"),
    }
}

# ---------------------------------------------------------
# 2. Cluster Extraction Parameters (Adjust as needed)
# ---------------------------------------------------------
# Adjust cluster_stat_thresh to match the statistical values in the maps (e.g., Z-score or t-value)
cluster_stat_thresh = 3.1  # Standard threshold corresponding to p < 0.001 uncorrected
min_cluster_voxels = 10    # Minimum cluster extent threshold in voxels

# ---------------------------------------------------------
# 3. Voxel & Cluster-Based Analysis
# ---------------------------------------------------------
results = []

for test_name, paths in maps_to_process.items():
    print(f"\n{'='*60}")
    print(f"ANALYSIS FOR {test_name.upper()} MAPS")
    print(f"{'='*60}")
    
    # Check if files exist
    required_files = [paths["drawn_thresholded"], paths["atlas_thresholded"], 
                      paths["drawn_unthresholded"], paths["atlas_unthresholded"]]
    if not all(op.exists(f) for f in required_files):
        print(f"Skipping {test_name}: One or more files not found.")
        continue
    
    # Load thresholded maps for Dice calculation
    drawn_thresh_img = image.load_img(paths["drawn_thresholded"])
    atlas_thresh_img = image.load_img(paths["atlas_thresholded"])
    
    drawn_thresholded = (drawn_thresh_img.get_fdata() != 0).astype(int)
    atlas_thresholded = (atlas_thresh_img.get_fdata() != 0).astype(int)
    
    # Load unthresholded maps for Spearman and Nilearn cluster extraction
    drawn_unthresh_img = image.load_img(paths["drawn_unthresholded"])
    atlas_unthresh_img = image.load_img(paths["atlas_unthresholded"])
    
    drawn_data = drawn_unthresh_img.get_fdata()
    atlas_data = atlas_unthresh_img.get_fdata()
    
    # ===== SPATIAL SIMILARITY (THRESHOLDED) =====
    # Dice Similarity Coefficient
    intersection = np.sum(drawn_thresholded & atlas_thresholded)
    union_count = np.sum(drawn_thresholded) + np.sum(atlas_thresholded)
    
    if union_count > 0:
        dice = (2.0 * intersection) / union_count
    else:
        dice = 0.0
    
    print(f"\nSpatial Similarity (Thresholded):")
    print(f"  Dice Similarity Coefficient: {dice:.4f}")
    
    # ===== CONNECTIVITY EFFECT SIZES (UNTHRESHOLDED) =====
    # Union of nonzero voxels (voxels that are nonzero in either atlas or drawn)
    union_mask = ((drawn_data != 0) | (atlas_data != 0)).astype(bool)
    
    # Extract Z-scores within union
    z_drawn_union = drawn_data[union_mask]
    z_atlas_union = atlas_data[union_mask]
    
    # Spearman rank-order correlation
    if len(z_drawn_union) > 1:
        spearman_rho, p_val = spearmanr(z_drawn_union, z_atlas_union)
    else:
        spearman_rho = np.nan
        p_val = np.nan
    
    print(f"\nConnectivity Effect Sizes (Unthresholded):")
    print(f"  Spearman (rho) Correlation: {spearman_rho:.4f} (P < 0.001)")
    
    # ===== DATA-DRIVEN CLUSTER EXTRACTION (NILEARN) =====
    print(f"\nData-Driven Cluster Extraction (Nilearn):")
    
    # Extract clusters for Hand-Drawn method
    try:
        drawn_cluster_table = get_clusters_table(
            stat_img=drawn_unthresh_img,
            stat_threshold=cluster_stat_thresh,
            cluster_threshold=min_cluster_voxels,
            two_sided=True  # Handles both positive and negative connectivity clusters
        )
        print(f"\n  [Hand-Drawn Seed Clusters]")
        if not drawn_cluster_table.empty:
            print(drawn_cluster_table.to_string(index=False))
        else:
            print("    No clusters survived thresholding.")
    except Exception as e:
        print(f"  Error extracting Hand-Drawn clusters: {e}")
        
    # Extract clusters for Automated Atlas method
    try:
        atlas_cluster_table = get_clusters_table(
            stat_img=atlas_unthresh_img,
            stat_threshold=cluster_stat_thresh,
            cluster_threshold=min_cluster_voxels,
            two_sided=True
        )
        print(f"\n  [Automated Atlas Seed Clusters]")
        if not atlas_cluster_table.empty:
            print(atlas_cluster_table.to_string(index=False))
        else:
            print("    No clusters survived thresholding.")
    except Exception as e:
        print(f"  Error extracting Atlas clusters: {e}")
    
    # Store results
    results.append({
        "Test": test_name,
        "Dice_Coefficient": dice,
        "Spearman_rho": spearman_rho,
        "P_value": p_val
    })

# ---------------------------------------------------------
# 4. Summary Table
# ---------------------------------------------------------
print(f"\n\n{'='*60}")
print("SUMMARY: VOXEL-BASED ANALYSIS")
print(f"{'='*60}")
df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))

# Save results to CSV
output_csv = op.join(analysis_dir, "voxel_comparison_summary.csv")
df_results.to_csv(output_csv, index=False)
print(f"\nResults saved to: {output_csv}")