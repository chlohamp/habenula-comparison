import os
import os.path as op
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# ---------------------------------------------------------
# 1. Setup Data Paths
# ---------------------------------------------------------
analysis_dir = "./dset/derivatives/hb-conn"
decoding_dir = "./decoding"

# Assume you ran Gradec/Nimare on both maps and saved the output tables
# If you haven't saved them yet, do so using: df.to_csv('1s_drawn_decoding.csv')
drawn_decoding_fn = op.join(decoding_dir, "1s_drawn_neuroquery_terms.csv")
atlas_decoding_fn = op.join(decoding_dir, "1s_atlas_neuroquery_terms.csv")

# ---------------------------------------------------------
# 2. Load and Align Decoding Profiles
# ---------------------------------------------------------
# Let's assume files have columns: 'Term' and 'Correlation' (or 'Weight')
df_drawn = pd.read_csv(drawn_decoding_fn)
df_atlas = pd.read_csv(atlas_decoding_fn)

# Rename columns to avoid collision during merge
df_drawn = df_drawn.rename(columns={'Correlation': 'Drawn_r'})
df_atlas = df_atlas.rename(columns={'Correlation': 'Atlas_r'})

# Merge tables on the semantic Term to ensure perfect alignment
merged_decoding = pd.merge(df_drawn, df_atlas, on="Term")

# ---------------------------------------------------------
# 3. Calculate Meta-Similarity Statistics
# ---------------------------------------------------------
# Metric 1: Pearson correlation across the whole language landscape
profile_pearson, p_val_p = pearsonr(merged_decoding['Drawn_r'], merged_decoding['Atlas_r'])

# Metric 2: Spearman correlation (Checks if the absolute rank order of top terms matches)
profile_spearman, p_val_s = spearmanr(merged_decoding['Drawn_r'], merged_decoding['Atlas_r'])

# ---------------------------------------------------------
# 4. Isolate Top Functional Hits for Visual Verification
# ---------------------------------------------------------
# Sort by Drawn strength to inspect the top cognitive drivers
top_hits = merged_decoding.sort_values(by="Drawn_r", ascending=False).head(10)

print("=== FUNCTIONAL DECODING LANDSCAPE SIMILARITY ===")
print(f"Profile Semantic Pearson r  : {profile_pearson:.4f} (p = {p_val_p:.3e})")
print(f"Profile Rank Spearman rho : {profile_spearman:.4f} (p = {p_val_s:.3e})")
print("\nTop 10 Overlapping Cognitive Terms:")
print(top_hits[['Term', 'Drawn_r', 'Atlas_r']].to_string(index=False))

# Save comparison data
merged_decoding.to_csv(op.join(analysis_dir, "decoding_landscape_comparison.csv"), index=False)