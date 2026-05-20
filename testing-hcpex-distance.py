import os
import os.path as op
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import image
from scipy.spatial.distance import directed_hausdorff

# ---------------------------------------------------------
# 1. Define Paths (aligned with your existing pipeline)
# ---------------------------------------------------------
data_dir = "./dset"
deriv_dir = op.join(data_dir, "derivatives")
analysis_dir = op.join(deriv_dir, "hb-conn")

group_drawn_dir = op.join(analysis_dir, "group-drawn/habenula")
group_atlas_dir = op.join(analysis_dir, "group-atlas/habenula")

# HCPex Atlas path
atlas_hcpex_filename = op.join(data_dir, "HCPex_2mm", "HCPex_2mm.nii")
hcpex_labels_filename = op.join(data_dir, "HCPex_2mm", "HCPex_2mm.csv")

# Map paths dictionary
maps_to_process = {
    "1s": {
        "drawn": op.join(group_drawn_dir, "averaged", "sub-group_task-rest_desc-1SampletTest_thresh.nii.gz"),
        "atlas": op.join(group_atlas_dir, "averaged", "sub-group_task-rest_desc-1SampletTest_thresh.nii.gz"),
        "output": op.join(analysis_dir, "hcpex_distance_metrics_1s.csv")
    },
    "2s": {
        "drawn": op.join(group_drawn_dir, "difference", "sub-group_task-rest_desc-2SampletTest_thresh.nii.gz"),
        "atlas": op.join(group_atlas_dir, "difference", "sub-group_task-rest_desc-2SampletTest_thresh.nii.gz"),
        "output": op.join(analysis_dir, "hcpex_distance_metrics_2s.csv")
    }
}

# ---------------------------------------------------------
# 4. Helper function to convert voxel indices to MNI coordinates
# ---------------------------------------------------------
def voxel_to_mni(voxel_coords, affine):
    """Transforms an (N, 3) array of voxel coordinates to (N, 3) MNI mm coordinates."""
    if len(voxel_coords) == 0:
        return np.array([])
    # Append a column of ones for the affine matrix multiplication
    homogeneous_coords = np.hstack([voxel_coords, np.ones((len(voxel_coords), 1))])
    # Multiply by the transpose of the affine matrix and take the first 3 columns
    return (homogeneous_coords @ affine.T)[:, :3]

# ---------------------------------------------------------
# 3. Load HCPex Atlas and Labels
# ---------------------------------------------------------
print(f"Loading HCPex atlas...")
hcpex_img = image.load_img(atlas_hcpex_filename)

# Load HCPex labels
print(f"Loading HCPex labels from {hcpex_labels_filename}...")
hcpex_labels_df = pd.read_csv(hcpex_labels_filename)
# Create a mapping from Index to Full Label
label_mapping = dict(zip(hcpex_labels_df['Index'], hcpex_labels_df['Full Label']))

for test_name, paths in maps_to_process.items():
    print(f"\n=========================================")
    print(f"Calculating Distance Metrics for {test_name.upper()}...")
    print(f"=========================================")
    
    if not op.exists(paths["drawn"]) or not op.exists(paths["atlas"]):
        print(f"Skipping {test_name}: Files not found.")
        continue
        
    drawn_img = image.load_img(paths["drawn"])
    atlas_img = image.load_img(paths["atlas"])
    affine = drawn_img.affine  # Both maps are in standard 2mm MNI space

    # Resample HCPex atlas to the exact target functional image grid
    hcpex_resampled = image.resample_to_img(hcpex_img, drawn_img, interpolation='nearest')
    hcpex_data = np.rint(hcpex_resampled.get_fdata()).astype(int)

    # Binarize thresholded maps to locate active voxels
    drawn_bin = (np.abs(drawn_img.get_fdata()) > 0)
    atlas_bin = (np.abs(atlas_img.get_fdata()) > 0)

    # Grab all valid HCPex regions (ignoring 0/background)
    regions = np.unique(hcpex_data)
    regions = regions[regions > 0]

    distance_results = []

    for region_id in regions:
        region_mask = (hcpex_data == region_id)
        
        # Locate indices where voxels are both inside the HCPex region AND active
        voxels_drawn = np.argwhere(region_mask & drawn_bin)
        voxels_atlas = np.argwhere(region_mask & atlas_bin)
        
        count_drawn = len(voxels_drawn)
        count_atlas = len(voxels_atlas)
        
        # Default metrics to NaN if comparisons are impossible
        com_distance_mm = np.nan
        hausdorff_distance_mm = np.nan
        drawn_com_mni = [np.nan, np.nan, np.nan]
        atlas_com_mni = [np.nan, np.nan, np.nan]

        # Distance metrics require that BOTH maps have at least one active voxel in the region
        if count_drawn > 0 and count_atlas > 0:
            # 1. Transform voxel coordinates into actual MNI millimeter space
            mni_drawn = voxel_to_mni(voxels_drawn, affine)
            mni_atlas = voxel_to_mni(voxels_atlas, affine)
            
            # 2. Compute Center of Mass (CoM) in MNI space
            drawn_com_mni = mni_drawn.mean(axis=0)
            atlas_com_mni = mni_atlas.mean(axis=0)
            
            # 3. Calculate Euclidean Distance between the two Centers of Mass
            com_distance_mm = np.linalg.norm(drawn_com_mni - atlas_com_mni)
            
            # 4. Calculate Bidirectional Hausdorff Distance in mm
            d_drawn_to_atlas = directed_hausdorff(mni_drawn, mni_atlas)[0]
            d_atlas_to_drawn = directed_hausdorff(mni_atlas, mni_drawn)[0]
            hausdorff_distance_mm = max(d_drawn_to_atlas, d_atlas_to_drawn)

        # Log entry for this specific anatomical region
        distance_results.append({
            "HCPex_Region_ID": region_id,
            "Full_Label": label_mapping.get(region_id, "Unknown"),
            "Drawn_Active_Count": count_drawn,
            "Atlas_Active_Count": count_atlas,
            "Drawn_CoM_X": drawn_com_mni[0],
            "Drawn_CoM_Y": drawn_com_mni[1],
            "Drawn_CoM_Z": drawn_com_mni[2],
            "Atlas_CoM_X": atlas_com_mni[0],
            "Atlas_CoM_Y": atlas_com_mni[1],
            "Atlas_CoM_Z": atlas_com_mni[2],
            "CoM_Distance_mm": com_distance_mm,
            "Hausdorff_Distance_mm": hausdorff_distance_mm
        })

    # Convert results matrix to a DataFrame and write out to disk
    df_dist = pd.DataFrame(distance_results)
    df_dist.to_csv(paths["output"], index=False)
    print(f"Saved distance metric breakdown to: {paths['output']}")
    print(df_dist.dropna(subset=['CoM_Distance_mm']).head(3))