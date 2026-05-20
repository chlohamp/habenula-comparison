import os
import os.path as op
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import image
from scipy.stats import pearsonr, spearmanr

# ---------------------------------------------------------
# 1. Define Paths 
# ---------------------------------------------------------
data_dir = "./dset"
deriv_dir = op.join(data_dir, "derivatives")
analysis_dir = op.join(deriv_dir, "hb-conn")

group_drawn_dir = op.join(analysis_dir, "group-drawn/habenula")
group_atlas_dir = op.join(analysis_dir, "group-atlas/habenula")

# HCPex Atlas path
atlas_hcpex_filename = op.join(data_dir, "HCPex_2mm", "HCPex_2mm.nii")
hcpex_labels_filename = op.join(data_dir, "HCPex_2mm", "HCPex_2mm.csv")

# Note: We use the UNTHRESHOLDED maps here to accurately capture the 
# full variance of connectivity values (Z-scores) within each region.
maps_to_process = {
    "1s": {
        "drawn": op.join(group_drawn_dir, "averaged", "sub-group_task-rest_desc-1SampletTest.nii.gz"),
        "atlas": op.join(group_atlas_dir, "averaged", "sub-group_task-rest_desc-1SampletTest.nii.gz"),
        "output": op.join(analysis_dir, "hcpex_connectivity_comparison_1s.csv")
    },
    "2s": {
        "drawn": op.join(group_drawn_dir, "difference", "sub-group_task-rest_desc-2SampletTest.nii.gz"),
        "atlas": op.join(group_atlas_dir, "difference", "sub-group_task-rest_desc-2SampletTest.nii.gz"),
        "output": op.join(analysis_dir, "hcpex_connectivity_comparison_2s.csv")
    }
}

# ---------------------------------------------------------
# 2. Load HCPex Atlas and Labels
# ---------------------------------------------------------
print(f"Loading base HCPex atlas from {atlas_hcpex_filename}...")
hcpex_img = image.load_img(atlas_hcpex_filename)

# Load HCPex labels
print(f"Loading HCPex labels from {hcpex_labels_filename}...")
hcpex_labels_df = pd.read_csv(hcpex_labels_filename)
# Create a mapping from Index to Full Label
label_mapping = dict(zip(hcpex_labels_df['Index'], hcpex_labels_df['Full Label']))

for test_name, paths in maps_to_process.items():
    print(f"\n=========================================")
    print(f"Comparing Connectivity for {test_name.upper()} Maps...")
    print(f"=========================================")
    
    if not op.exists(paths["drawn"]) or not op.exists(paths["atlas"]):
        print(f"Skipping {test_name}: Unthresholded files not found.")
        continue
        
    drawn_img = image.load_img(paths["drawn"])
    atlas_img = image.load_img(paths["atlas"])

    # Resample atlas to target statistical grid via nearest-neighbor
    hcpex_resampled = image.resample_to_img(hcpex_img, drawn_img, interpolation='nearest')
    
    # Extract raw Z-score data
    drawn_data = drawn_img.get_fdata()
    atlas_data = atlas_img.get_fdata()
    hcpex_data = np.rint(hcpex_resampled.get_fdata()).astype(int)

    # Get unique regions
    regions = np.unique(hcpex_data)
    regions = regions[regions > 0]

    conn_results = []

    for region_id in regions:
        region_mask = (hcpex_data == region_id)
        voxel_count = np.sum(region_mask)
        
        if voxel_count == 0:
            continue
            
        # Isolate the Z-scores within this specific HCPex region
        z_drawn = drawn_data[region_mask]
        z_atlas = atlas_data[region_mask]
        
        # 1. Calculate Mean Connectivity (Z-score) for the region
        mean_z_drawn = np.nanmean(z_drawn)
        mean_z_atlas = np.nanmean(z_atlas)
        
        # Calculate the absolute magnitude of the difference between the means
        mean_difference = abs(mean_z_drawn - mean_z_atlas)
        
        # 2. Calculate Pattern Similarity (Correlation of voxels within the region)
        # We need >1 voxel and non-zero standard deviation to run correlations
        if len(z_drawn) > 1 and np.std(z_drawn) > 0 and np.std(z_atlas) > 0:
            pearson_r, _ = pearsonr(z_drawn, z_atlas)
            spearman_rho, _ = spearmanr(z_drawn, z_atlas)
        else:
            pearson_r = np.nan
            spearman_rho = np.nan
            
        # Append data row
        conn_results.append({
            "HCPex_Region_ID": region_id,
            "Full_Label": label_mapping.get(region_id, "Unknown"),
            "Voxel_Count": voxel_count,
            "Mean_Z_Drawn": mean_z_drawn,
            "Mean_Z_Atlas": mean_z_atlas,
            "Absolute_Z_Difference": mean_difference,
            "Pearson_r": pearson_r,
            "Spearman_rho": spearman_rho
        })

    # Save to CSV
    df_conn = pd.DataFrame(conn_results)
    
    # Optional: Sort by the regions with the largest difference in connectivity
    df_conn = df_conn.sort_values(by="Absolute_Z_Difference", ascending=False)
    
    df_conn.to_csv(paths["output"], index=False)
    print(f"Saved regional connectivity comparison to: {paths['output']}")
    print("Top 3 regions with largest connectivity differences:")
    print(df_conn.head(3))