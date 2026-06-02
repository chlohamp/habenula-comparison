import os
import os.path as op
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import image
from scipy.stats import pearsonr, spearmanr
from scipy.ndimage import center_of_mass

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

maps_to_process = {
    "1s": {
        "drawn_unthresholded": op.join(group_drawn_dir, "averaged", "sub-group_task-rest_desc-1SampletTest.nii.gz"),
        "atlas_unthresholded": op.join(group_atlas_dir, "averaged", "sub-group_task-rest_desc-1SampletTest.nii.gz"),
        "drawn_thresholded": op.join(group_drawn_dir, "averaged", "sub-group_task-rest_desc-1SampletTest_thresh.nii.gz"),
        "atlas_thresholded": op.join(group_atlas_dir, "averaged", "sub-group_task-rest_desc-1SampletTest_thresh.nii.gz"),
        "output": op.join(analysis_dir, "hcpex_connectivity_comparison_1s.csv")
    },
    "2s": {
        "drawn_unthresholded": op.join(group_drawn_dir, "difference", "sub-group_task-rest_desc-2SampletTest.nii.gz"),
        "atlas_unthresholded": op.join(group_atlas_dir, "difference", "sub-group_task-rest_desc-2SampletTest.nii.gz"),
        "drawn_thresholded": op.join(group_drawn_dir, "difference", "sub-group_task-rest_desc-2SampletTest_thresh.nii.gz"),
        "atlas_thresholded": op.join(group_atlas_dir, "difference", "sub-group_task-rest_desc-2SampletTest_thresh.nii.gz"),
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
    
    required = [paths["drawn_unthresholded"], paths["atlas_unthresholded"],
                paths["drawn_thresholded"], paths["atlas_thresholded"]]
    if not all(op.exists(f) for f in required):
        print(f"Skipping {test_name}: One or more files not found.")
        continue

    drawn_img = image.load_img(paths["drawn_unthresholded"])
    atlas_img = image.load_img(paths["atlas_unthresholded"])
    drawn_thresh_img = image.load_img(paths["drawn_thresholded"])
    atlas_thresh_img = image.load_img(paths["atlas_thresholded"])

    # Resample HCPex atlas to statistical map grid via nearest-neighbor
    hcpex_resampled = image.resample_to_img(hcpex_img, drawn_img, interpolation='nearest')

    # Extract Z-score data (unthresholded) and binary masks (thresholded)
    drawn_data = drawn_img.get_fdata()
    atlas_data = atlas_img.get_fdata()
    drawn_thresh_data = (drawn_thresh_img.get_fdata() != 0).astype(int)
    atlas_thresh_data = (atlas_thresh_img.get_fdata() != 0).astype(int)
    hcpex_data = np.rint(hcpex_resampled.get_fdata()).astype(int)

    # Get affine for voxel to mm conversion
    affine = drawn_img.affine

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

        # Isolate thresholded binary masks within this region
        drawn_thresholded = drawn_thresh_data[region_mask]
        atlas_thresholded = atlas_thresh_data[region_mask]

        # Get 3D coordinates of voxels in the region
        region_coords = np.where(region_mask)

        # ===== SPATIAL SIMILARITY (THRESHOLDED) =====
        # Suprathreshold voxel counts
        drawn_active_voxels = int(np.sum(drawn_thresholded))
        atlas_active_voxels = int(np.sum(atlas_thresholded))

        # Dice Similarity Coefficient
        intersection = np.sum(drawn_thresholded & atlas_thresholded)
        union = drawn_active_voxels + atlas_active_voxels

        if union > 0:
            dice = (2.0 * intersection) / union
        else:
            dice = 0.0

        # ===== SPATIAL CENTERING: CENTER OF MASS DISTANCE =====
        # Calculate CoM in voxel space using full 3D coordinates
        drawn_coords = np.array(region_coords)[:, drawn_thresholded.astype(bool)]
        atlas_coords = np.array(region_coords)[:, atlas_thresholded.astype(bool)]
        
        if drawn_coords.shape[1] > 0:
            drawn_com_voxel = drawn_coords.mean(axis=1)
        else:
            drawn_com_voxel = None
            
        if atlas_coords.shape[1] > 0:
            atlas_com_voxel = atlas_coords.mean(axis=1)
        else:
            atlas_com_voxel = None
        
        # Convert to mm using affine
        if drawn_com_voxel is not None and atlas_com_voxel is not None:
            # Convert from voxel to mm coordinates
            drawn_com_mm = affine[:3, :3] @ drawn_com_voxel + affine[:3, 3]
            atlas_com_mm = affine[:3, :3] @ atlas_com_voxel + affine[:3, 3]
            
            com_distance = np.linalg.norm(drawn_com_mm - atlas_com_mm)
        else:
            com_distance = np.nan
        
        # ===== CONNECTIVITY EFFECT SIZES (UNTHRESHOLDED) =====
        # 1. Calculate Mean Connectivity (Z-score) for the region
        mean_z_drawn = np.nanmean(z_drawn)
        mean_z_atlas = np.nanmean(z_atlas)
        
        # Calculate percent difference relative to atlas
        if mean_z_atlas != 0:
            percent_difference = (abs(mean_z_drawn - mean_z_atlas) / abs(mean_z_atlas)) * 100
        else:
            percent_difference = 0.0
        
        # 2. Calculate Spearman rank-order correlation
        if len(z_drawn) > 1 and np.std(z_drawn) > 0 and np.std(z_atlas) > 0:
            spearman_rho, _ = spearmanr(z_drawn, z_atlas)
        else:
            spearman_rho = np.nan
            
        # Append data row
        conn_results.append({
            "HCPex_Region_ID": region_id,
            "Full_Label": label_mapping.get(region_id, "Unknown"),
            "Voxel_Count": voxel_count,
            "Drawn_Active_Voxels": drawn_active_voxels,
            "Atlas_Active_Voxels": atlas_active_voxels,
            "Dice_Coefficient": dice,
            "CoM_Distance_mm": com_distance,
            "Mean_Z_Atlas": mean_z_atlas,
            "Mean_Z_Drawn": mean_z_drawn,
            "Percent_Difference": percent_difference,
            "Spearman_rho": spearman_rho
        })

    # Save to CSV
    df_conn = pd.DataFrame(conn_results)
    
    # Sort by Mean_Z_Atlas (descending: strongest connectivity first)
    df_conn = df_conn.sort_values(by="Mean_Z_Atlas", ascending=False)
    
    df_conn.to_csv(paths["output"], index=False)
    print(f"Saved regional connectivity comparison to: {paths['output']}")
    print("\nTop 5 regions (sorted by Mean Z-score of Atlas ROI):")
    print(df_conn[["Full_Label", "Voxel_Count", "Drawn_Active_Voxels", "Atlas_Active_Voxels",
                   "Dice_Coefficient", "CoM_Distance_mm", "Mean_Z_Atlas", "Mean_Z_Drawn", 
                   "Spearman_rho"]].head(5).to_string(index=False))