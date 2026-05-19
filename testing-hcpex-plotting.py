import os
import os.path as op
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import image
from scipy.stats import pearsonr, spearmanr

# ---------------------------------------------------------
# 1. Define Paths (based on your existing scripts)
# ---------------------------------------------------------
data_dir = "./dset"
deriv_dir = op.join(data_dir, "derivatives")
analysis_dir = op.join(deriv_dir, "hb-conn")

group_drawn_dir = op.join(analysis_dir, "group-drawn/habenula")
group_atlas_dir = op.join(analysis_dir, "group-atlas/habenula")

# HCPex Atlas path
atlas_hcpex_filename = op.join(data_dir, "HCPex_2mm", "HCPex_2mm.nii")

# Define both map types in a dictionary to process them dynamically
maps_to_process = {
    "1s": {
        "drawn": op.join(group_drawn_dir, "averaged", "sub-group_task-rest_desc-1SampletTest_thresh.nii.gz"),
        "atlas": op.join(group_atlas_dir, "averaged", "sub-group_task-rest_desc-1SampletTest_thresh.nii.gz"),
        "output": op.join(analysis_dir, "hcpex_region_breakdown_1s.csv")
    },
    "2s": {
        "drawn": op.join(group_drawn_dir, "difference", "sub-group_task-rest_desc-2SampletTest_thresh.nii.gz"),
        "atlas": op.join(group_atlas_dir, "difference", "sub-group_task-rest_desc-2SampletTest_thresh.nii.gz"),
        "output": op.join(analysis_dir, "hcpex_region_breakdown_2s.csv")
    }
}

# ---------------------------------------------------------
# 2. Load HCPex Atlas Base Image
# ---------------------------------------------------------
print(f"Loading base HCPex atlas from {atlas_hcpex_filename}...")
hcpex_img = image.load_img(atlas_hcpex_filename)

# ---------------------------------------------------------
# 3. Loop and Generate Regional Breakdown Tables
# ---------------------------------------------------------
for test_name, paths in maps_to_process.items():
    print(f"\n=========================================")
    print(f"Processing {test_name.upper()} Contrast Maps...")
    print(f"=========================================")
    
    # Check if the files exist before running
    if not op.exists(paths["drawn"]) or not op.exists(paths["atlas"]):
        print(f"Skipping {test_name} because one or both thresholded NIfTI files do not exist.")
        continue
        
    drawn_img = image.load_img(paths["drawn"])
    atlas_img = image.load_img(paths["atlas"])

    # Nearest-neighbor resampling ensures the HCPex integer region IDs are preserved
    print(f"Resampling HCPex atlas to match the voxel grid of {test_name} maps...")
    hcpex_resampled = image.resample_to_img(hcpex_img, drawn_img, interpolation='nearest')

    # Extract arrays
    drawn_data = drawn_img.get_fdata()
    atlas_data = atlas_img.get_fdata()
    hcpex_data = np.rint(hcpex_resampled.get_fdata()).astype(int)

    # Binarize thresholded maps for Dice calculations (np.abs handles two-sided effects)
    drawn_bin = (np.abs(drawn_data) > 0).astype(int)
    atlas_bin = (np.abs(atlas_data) > 0).astype(int)

    # Extract all active unique region IDs (ignoring 0/background)
    regions = np.unique(hcpex_data)
    regions = regions[regions > 0]

    results = []

    for region_id in regions:
        # Create mask for the current region
        region_mask = (hcpex_data == region_id)
        voxel_count = np.sum(region_mask)
        
        if voxel_count == 0:
            continue
            
        # Isolate voxels within the current region boundary
        drawn_region = drawn_data[region_mask]
        atlas_region = atlas_data[region_mask]
        
        drawn_region_bin = drawn_bin[region_mask]
        atlas_region_bin = atlas_bin[region_mask]
        
        # -- Calculate Dice Coefficient --
        intersection = np.logical_and(drawn_region_bin, atlas_region_bin).sum()
        volume_sum = drawn_region_bin.sum() + atlas_region_bin.sum()
        dice = (2.0 * intersection / volume_sum) if volume_sum > 0 else np.nan
        
        # -- Calculate Voxel Intensity Correlations --
        # Requires more than 1 voxel and standard deviation > 0 to prevent divide-by-zero errors
        if len(drawn_region) > 1 and np.std(drawn_region) > 0 and np.std(atlas_region) > 0:
            pearson_r, _ = pearsonr(drawn_region, atlas_region)
            spearman_rho, _ = spearmanr(drawn_region, atlas_region)
        else:
            pearson_r = np.nan
            spearman_rho = np.nan
            
        # Append data row
        results.append({
            "HCPex_Region_ID": region_id,
            "Total_Region_Voxels": voxel_count,
            "Drawn_Active_Voxels": drawn_region_bin.sum(),
            "Atlas_Active_Voxels": atlas_region_bin.sum(),
            "Dice_Coefficient": dice,
            "Pearson_r": pearson_r,
            "Spearman_rho": spearman_rho
        })

    # Save results to its respective CSV file
    df = pd.DataFrame(results)
    df.to_csv(paths["output"], index=False)
    print(f"Saved regional breakdown to: {paths['output']}")
    print(df.head(3))