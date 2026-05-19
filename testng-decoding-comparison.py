import os
import os.path as op
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# ---------------------------------------------------------
# 1. Setup Correct Data Paths
# ---------------------------------------------------------
data_dir = "./dset"
deriv_dir = op.join(data_dir, "derivatives")
analysis_dir = op.join(deriv_dir, "hb-conn")

# This matches the folder where your modified plot-connectivity loop saves CSV outputs
decoding_results_dir = op.join(analysis_dir, "decoding_results")

# Dictionary holding the paths for 1-Sample and 2-Sample decoding tables
contrasts_to_compare = {
    "1s": {
        "name": "1-Sample t-Test (Group Mean Connectivity)",
        "drawn": op.join(decoding_results_dir, "1s_drawn_neuroquery_terms.csv"),
        "atlas": op.join(decoding_results_dir, "1s_atlas_neuroquery_terms.csv"),
        "output": op.join(analysis_dir, "decoding_landscape_comparison_1s.csv")
    },
    "2s": {
        "name": "2-Sample t-Test (Group Difference Map)",
        "drawn": op.join(decoding_results_dir, "2s_drawn_neuroquery_terms.csv"),
        "atlas": op.join(decoding_results_dir, "2s_atlas_neuroquery_terms.csv"),
        "output": op.join(analysis_dir, "decoding_landscape_comparison_2s.csv")
    }
}

# ---------------------------------------------------------
# 2. Process and Compare Decoding Profiles
# ---------------------------------------------------------
for test_key, paths in contrasts_to_compare.items():
    print(f"\n==================================================")
    print(f"RUNNING DECODING SIMILARITY FOR: {paths['name']}")
    print(f"==================================================")
    
    # Verification check to make sure you ran the plot script first
    if not op.exists(paths["drawn"]) or not op.exists(paths["atlas"]):
        print(f"Skipping {test_key}: Decoding files not found in {decoding_results_dir}.")
        print(f"Make sure to execute your modified plot-connectivity code first to generate them.")
        continue
        
    # Load dataframes
    df_drawn = pd.read_csv(paths["drawn"])
    df_atlas = pd.read_csv(paths["atlas"])
    
    # Identify the column containing the correlation metric (usually named 'r' or 'Correlation')
    # This automatically detects NiMare or NiMQ standard column schemas
    drawn_metric_col = 'r' if 'r' in df_drawn.columns else 'Correlation'
    atlas_metric_col = 'r' if 'r' in df_atlas.columns else 'Correlation'
    
    # Rename columns to avoid name collisions during merge
    df_drawn = df_drawn[['Term', drawn_metric_col]].rename(columns={drawn_metric_col: 'Drawn_r'})
    df_atlas = df_atlas[['Term', atlas_metric_col]].rename(columns={atlas_metric_col: 'Atlas_r'})
    
    # Inner merge on 'Term' to ensure word-for-word alignment across semantic profiles
    merged_decoding = pd.merge(df_drawn, df_atlas, on="Term")
    
    # 3. Calculate Global Semantic Profiles Similarity Metrics
    # Metric A: Pearson r (Evaluates if overall loading magnitudes align)
    profile_pearson, p_val_p = pearsonr(merged_decoding['Drawn_r'], merged_decoding['Atlas_r'])
    
    # Metric B: Spearman rho (Evaluates if the precise rank order of terms is preserved)
    profile_spearman, p_val_s = spearmanr(merged_decoding['Drawn_r'], merged_decoding['Atlas_r'])
    
    # 4. Save and Display Results
    # Create an index tracking the absolute difference between term weights
    merged_decoding['Absolute_Discrepancy'] = (merged_decoding['Drawn_r'] - merged_decoding['Atlas_r']).abs()
    merged_decoding = merged_decoding.sort_values(by="Drawn_r", ascending=False)
    
    merged_decoding.to_csv(paths["output"], index=False)
    
    print(f"Saved complete comparative array to: {paths['output']}")
    print(f" -> Profile Semantic Pearson r  : {profile_pearson:.4f} (p = {p_val_p:.3e})")
    print(f" -> Profile Rank Spearman rho : {profile_spearman:.4f} (p = {p_val_s:.3e})")
    
    print("\nTop 5 Cognitive Term Mappings (Ranked by Drawn Strength):")
    print(merged_decoding[['Term', 'Drawn_r', 'Atlas_r', 'Absolute_Discrepancy']].head(5).to_string(index=False))