import os
import os.path as op
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Setup Paths
# ---------------------------------------------------------
data_dir = "./dset"
analysis_dir = op.join(data_dir, "derivatives", "hb-conn")
decoding_results_dir = op.join(analysis_dir, "decoding_results")

contrasts = ["1s", "2s"]
sns.set_theme(style="whitegrid")

# =========================================================
# PART A: HCPex Regional Connectivity Alignment Metrics
# =========================================================
for test in contrasts:
    print(f"Processing separate HCPex Regional plots for {test.upper()}...")
    
    overlap_csv = op.join(analysis_dir, f"hcpex_region_breakdown_{test}.csv")
    distance_csv = op.join(analysis_dir, f"hcpex_distance_metrics_{test}.csv")
    conn_csv = op.join(analysis_dir, f"hcpex_connectivity_comparison_{test}.csv")
    
    if op.exists(overlap_csv) and op.exists(distance_csv) and op.exists(conn_csv):
        df_overlap = pd.read_csv(overlap_csv)
        df_dist = pd.read_csv(distance_csv)
        df_conn = pd.read_csv(conn_csv).rename(columns={
            'Spearman_rho': 'Conn_Spearman_rho', 
            'Pearson_r': 'Conn_Pearson_r'
        })
        
        # Merge matrices and drop background regions with no data
        df = pd.merge(df_overlap, df_dist, on="HCPex_Region_ID")
        df = pd.merge(df, df_conn, on="HCPex_Region_ID").dropna(
            subset=['Dice_Coefficient', 'CoM_Distance_mm', 'Conn_Spearman_rho']
        )
        
        # Calculate Consensus Ranks
        df['Rank_Dice'] = df['Dice_Coefficient'].rank(ascending=False)
        df['Rank_Spearman'] = df['Conn_Spearman_rho'].rank(ascending=False)
        df['Rank_Dist'] = df['CoM_Distance_mm'].rank(ascending=True)
        df['Rank_Z'] = df['Absolute_Z_Difference'].rank(ascending=True)
        df['Mean_Z'] = df['Mean_Z_Drawn'].rank(ascending=True)
        
        df['Region_Label'] = df['Full_Label']
        
        # Define metric parameters: (Title, Subtitle, Color, Ascending_Sort_Order)
        # For Dice and Spearman: Higher is Better -> Sort Descending (ascending=False)
        # For Distance, Z-diff, and Ranks: Lower is Better -> Sort Ascending (ascending=True)
        regional_metrics = {
            'Dice_Coefficient': ("Dice Overlap Coefficient", "Higher is Better", "steelblue", False),
            'CoM_Distance_mm': ("Center of Mass Distance (mm)", "Lower is Better", "steelblue", True),
            'Hausdorff_Distance_mm': ("Hausdorff Distance (mm)", "Lower is Better", "steelblue", True),
            'Mean_Z_Drawn': ("Mean Z-Score (Drawn ROI)", "Higher is Better", "darkgreen", False),
            'Mean_Z_Atlas': ("Mean Z-Score (Atlas ROI)", "Higher is Better", "darkblue", False),
            'Absolute_Z_Difference': ("Absolute Delta Z-Score", "Lower is Better", "steelblue", True),
            'Conn_Spearman_rho': ("Unthresholded Spearman rho", "Higher is Better", "steelblue", False)
        }
        
        for metric, (title, subtitle, color_name, asc_order) in regional_metrics.items():
            # Dynamically sort the dataframe for THIS specific metric loop
            plot_df = df.sort_values(by=metric, ascending=asc_order)
            
            # Dynamically scale figure height based on the number of regions found
            fig_height = max(6, len(plot_df) * 0.22)
            fig, ax = plt.subplots(figsize=(6, fig_height))
            
            sns.stripplot(
                data=plot_df, x=metric, y="Region_Label", size=8, orient="h", 
                jitter=False, color=color_name, linewidth=1, edgecolor="w", ax=ax
            )
            
            ax.set_title(f"{title}\n({subtitle})", fontsize=11, fontweight='bold', pad=12)
            ax.xaxis.grid(False)
            ax.yaxis.grid(True, color='lightgray', linestyle='-')
            ax.set_xlabel("")
            ax.set_ylabel("")
            sns.despine(left=True, bottom=True, ax=ax)
            
            output_png = op.join(analysis_dir, f"plot_regional_{metric}_{test}.png")
            plt.savefig(output_png, bbox_inches="tight", dpi=300)
            plt.close()

# =========================================================
# PART A.5: Combined Mean Z-Score Plots (Atlas ROI Order)
# =========================================================
for test in contrasts:
    print(f"Processing overlaid Mean Z-Score plot for {test.upper()}...")
    
    overlap_csv = op.join(analysis_dir, f"hcpex_region_breakdown_{test}.csv")
    distance_csv = op.join(analysis_dir, f"hcpex_distance_metrics_{test}.csv")
    conn_csv = op.join(analysis_dir, f"hcpex_connectivity_comparison_{test}.csv")
    
    if op.exists(overlap_csv) and op.exists(distance_csv) and op.exists(conn_csv):
        df_overlap = pd.read_csv(overlap_csv)
        df_dist = pd.read_csv(distance_csv)
        df_conn = pd.read_csv(conn_csv).rename(columns={
            'Spearman_rho': 'Conn_Spearman_rho', 
            'Pearson_r': 'Conn_Pearson_r'
        })
        
        # Merge matrices and drop background regions with no data
        df = pd.merge(df_overlap, df_dist, on="HCPex_Region_ID")
        df = pd.merge(df, df_conn, on="HCPex_Region_ID").dropna(
            subset=['Dice_Coefficient', 'CoM_Distance_mm', 'Conn_Spearman_rho']
        )
        
        df['Region_Label'] = df['Full_Label']
        
        # Sort by Atlas Mean Z-Score (descending: higher is better at top)
        df_sorted = df.sort_values(by='Mean_Z_Atlas', ascending=False)
        
        # Capture the sorted order of regions to preserve it during melting
        sorted_region_order = df_sorted['Region_Label'].tolist()
        
        # Pivot to long format for overlay comparison
        melted_z = pd.melt(
            df_sorted,
            id_vars=['Region_Label'],
            value_vars=['Mean_Z_Atlas', 'Mean_Z_Drawn'],
            var_name='ROI_Type',
            value_name='Mean_Z_Score'
        )
        
        # Clean legend labels
        melted_z['ROI_Type'] = melted_z['ROI_Type'].map({
            'Mean_Z_Atlas': 'Atlas ROI',
            'Mean_Z_Drawn': 'Drawn ROI'
        })
        
        # Dynamically scale figure height
        fig_height = max(6, len(df_sorted) * 0.22)
        fig, ax = plt.subplots(figsize=(7, fig_height))
        
        # Draw connecting lines between Atlas and Drawn ROI points for each region
        for i, region in enumerate(sorted_region_order):
            region_data = melted_z[melted_z['Region_Label'] == region]
            if len(region_data) == 2:
                atlas_z = region_data[region_data['ROI_Type'] == 'Atlas ROI']['Mean_Z_Score'].values[0]
                drawn_z = region_data[region_data['ROI_Type'] == 'Drawn ROI']['Mean_Z_Score'].values[0]
                # Use red line if Drawn ROI has larger mean Z than Atlas ROI, otherwise black
                line_color = "#9A9B9A" if drawn_z > atlas_z else 'black'
                ax.plot([atlas_z, drawn_z], [i, i], color=line_color, linewidth=2.5, zorder=1)
        
        # Overlay both ROI types on same plot
        sns.stripplot(
            data=melted_z,
            x='Mean_Z_Score',
            y='Region_Label',
            hue='ROI_Type',
            order=sorted_region_order,  # Preserve atlas ROI order
            size=8,
            orient="h",
            jitter=False,
            palette={'Atlas ROI': '#E93524', 'Drawn ROI': '#789A2D'},
            linewidth=0.8,
            edgecolor="w",
            alpha=0.85,
            ax=ax
        )
        
        # Title and Formatting
        ax.set_title(f"Mean Z-Score Comparison by Region ({test.upper()})\nSorted by Atlas ROI Mean Z-Score",
                     fontsize=11, fontweight='bold', pad=14)
        
        ax.xaxis.grid(False)
        ax.yaxis.grid(True, color='lightgray', linestyle='-')
        ax.set_xlabel("Mean Z-Score", fontsize=10, labelpad=8)
        ax.set_ylabel("")
        
        # Style and place the legend
        ax.legend(title="ROI Definition Method", loc="lower right", frameon=True, facecolor="w")
        
        sns.despine(left=True, bottom=True, ax=ax)
        
        output_png = op.join(analysis_dir, f"plot_mean_z_overlaid_comparison_{test}.png")
        plt.savefig(output_png, bbox_inches="tight", dpi=300)
        plt.close()

print("Overlaid Mean Z-Score comparison figures generated successfully!")

# =========================================================
# PART B: NeuroQuery Functional Decoding Alignment Metrics
# =========================================================
for test in contrasts:
    print(f"Processing separate Decoding plots for {test.upper()}...")
    
    drawn_path = op.join(decoding_results_dir, f"{test}_drawn_neuroquery_terms.csv")
    atlas_path = op.join(decoding_results_dir, f"{test}_atlas_neuroquery_terms.csv")
    
    if op.exists(drawn_path) and op.exists(atlas_path):
        df_drawn = pd.read_csv(drawn_path)
        df_atlas = pd.read_csv(atlas_path)
        
        d_col = 'r' if 'r' in df_drawn.columns else 'Correlation'
        a_col = 'r' if 'r' in df_atlas.columns else 'Correlation'
        
        df_drawn = df_drawn[['Term', d_col]].rename(columns={d_col: 'Drawn_r'})
        df_atlas = df_atlas[['Term', a_col]].rename(columns={a_col: 'Atlas_r'})
        
        merged_dec = pd.merge(df_drawn, df_atlas, on="Term")
        merged_dec['Absolute_Discrepancy'] = (merged_dec['Drawn_r'] - merged_dec['Atlas_r']).abs()
        
        # For raw correlations: Highest values mean strongest association -> Sort Descending (False)
        # For discrepancy: Lower difference is better -> Sort Ascending (True)
        decoding_metrics = {
            'Drawn_r': ("Drawn ROI Correlation (r)", "Functional Decoding Vector Strength", "teal", False),
            'Atlas_r': ("Atlas ROI Correlation (r)", "Functional Decoding Vector Strength", "teal", False),
            'Absolute_Discrepancy': ("Absolute Discrepancy (|Delta r|)", "Lower is Better", "crimson", True)
        }
        
        for metric, (title, subtitle, color_name, asc_order) in decoding_metrics.items():
            # Dynamically sort the dataframe for THIS specific decoding loop
            plot_dec = merged_dec.sort_values(by=metric, ascending=asc_order)
            
            # Dynamically scale figure height based on the total number of terms decoded
            fig_height = max(6, len(plot_dec) * 0.22)
            fig, ax = plt.subplots(figsize=(6, fig_height))
            
            sns.stripplot(
                data=plot_dec, x=metric, y="Term", size=8, orient="h", 
                jitter=False, color=color_name, linewidth=1, edgecolor="w", ax=ax
            )
            
            ax.set_title(f"{title}\n({subtitle})", fontsize=11, fontweight='bold', pad=12)
            ax.xaxis.grid(False)
            ax.yaxis.grid(True, color='lightgray', linestyle='-')
            ax.set_xlabel("")
            ax.set_ylabel("")
            sns.despine(left=True, bottom=True, ax=ax)
            
            output_png = op.join(analysis_dir, f"plot_decoding_{metric}_{test}.png")
            plt.savefig(output_png, bbox_inches="tight", dpi=300)
            plt.close()

print("\nAll customized standalone metric figures generated and saved successfully!")

# =========================================================
# PART B: NeuroQuery Functional Decoding Alignment Metrics
# =========================================================
for test in contrasts:
    print(f"Processing overlaid Decoding plot for {test.upper()}...")
    
    drawn_path = op.join(decoding_results_dir, f"{test}_drawn_neuroquery_terms.csv")
    atlas_path = op.join(decoding_results_dir, f"{test}_atlas_neuroquery_terms.csv")
    
    if op.exists(drawn_path) and op.exists(atlas_path):
        df_drawn = pd.read_csv(drawn_path)
        df_atlas = pd.read_csv(atlas_path)
        
        d_col = 'r' if 'r' in df_drawn.columns else 'Correlation'
        a_col = 'r' if 'r' in df_atlas.columns else 'Correlation'
        
        df_drawn = df_drawn[['Term', d_col]].rename(columns={d_col: 'Drawn_r'})
        df_atlas = df_atlas[['Term', a_col]].rename(columns={a_col: 'Atlas_r'})
        
        # Merge data and calculate absolute discrepancy
        merged_dec = pd.merge(df_drawn, df_atlas, on="Term")
        merged_dec['Absolute_Discrepancy'] = (merged_dec['Drawn_r'] - merged_dec['Atlas_r']).abs()
        
        # SORTING: Lowest absolute discrepancy (most similar) goes to the top
        merged_dec = merged_dec.sort_values(by='Absolute_Discrepancy', ascending=True)
        
        # Capture the sorted order of terms to preserve it during melting
        sorted_term_order = merged_dec['Term'].tolist()
        
        # Pivot the dataframe to long format so Seaborn can group by "Method"
        melted_dec = pd.melt(
            merged_dec, 
            id_vars=['Term', 'Absolute_Discrepancy'], 
            value_vars=['Drawn_r', 'Atlas_r'],
            var_name='Method', 
            value_name='Correlation'
        )
        
        # Give the legend tracks clean publication-ready labels
        melted_dec['Method'] = melted_dec['Method'].map({'Drawn_r': 'Drawn ROI', 'Atlas_r': 'Atlas ROI'})
        
        # Dynamically scale figure height based on total terms
        fig_height = max(6, len(merged_dec) * 0.22)
        fig, ax = plt.subplots(figsize=(7, fig_height))
        
        # Plot both sets of points on the same axis using 'hue'
        # Adjust palette colors here if you prefer different aesthetics
        sns.stripplot(
            data=melted_dec, 
            x='Correlation', 
            y='Term', 
            hue='Method',
            order=sorted_term_order,  # Enforces the "most similar at top" layout
            size=8, 
            orient="h", 
            jitter=False, 
            palette={'Drawn ROI': '#008080', 'Atlas ROI': '#FF8C00'}, # Teal vs Dark Orange
            linewidth=0.8, 
            edgecolor="w", 
            alpha=0.85,
            ax=ax
        )
        
        # Title and Formatting
        ax.set_title(f"Meta-Analytic Functional Decoding Convergence ({test.upper()})\nSorted by Absolute Discrepancy (Most Similar at Top)", 
                     fontsize=11, fontweight='bold', pad=14)
        
        ax.xaxis.grid(False)
        ax.yaxis.grid(True, color='lightgray', linestyle='-')
        ax.set_xlabel("Correlation Coefficient (r)", fontsize=10, labelpad=8)
        ax.set_ylabel("")
        
        # Style and place the legend neatly out of the way
        ax.legend(title="ROI Definition Method", loc="lower right", frameon=True, facecolor="w")
        
        sns.despine(left=True, bottom=True, ax=ax)
        
        output_png = op.join(analysis_dir, f"plot_decoding_overlaid_comparison_{test}.png")
        plt.savefig(output_png, bbox_inches="tight", dpi=300)
        plt.close()

print("\nOverlaid decoding comparison figures generated successfully!")