"""
AMF Weighted Population Estimation
===================================

Based on OECD/Agasisti et al. (2018) methodology
Using W_FSTUWT sampling weights for population inference

Author: Jung-Ah Lee
Date: 2025-11
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import rankdata

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def normalize_ses_to_percentile_with_outliers(escs, iqr_k=1.5):
    """
    Remove SES outliers using IQR method (consistent with amf_engine.py)
    Returns: clean_idx, ses_percentile
    """
    q1 = escs.quantile(0.25)
    q3 = escs.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - iqr_k * iqr
    upper = q3 + iqr_k * iqr
    mask = (escs >= lower) & (escs <= upper)
    escs_clean = escs[mask].copy()
    
    # Calculate percentile using scipy rankdata for consistency
    ranks = rankdata(escs_clean, method="average")
    ses_percentile = (ranks - 1) / (len(escs_clean) - 1)
    
    return escs_clean.index, pd.Series(ses_percentile, index=escs_clean.index)

def load_pisa_data(filepath):
    """Load PISA 2022 Korea data (consistent with amf_engine.py)"""
    print("Loading PISA 2022 Korea dataset...")
    df = pd.read_csv(filepath)
    
    # Use PV1MATH only (consistent with run_amf_all_results.py)
    print(f"Total observations: {len(df)}")
    
    # Apply IQR-based outlier removal to ESCS
    escs = df['ESCS']
    idx_clean, ses_p = normalize_ses_to_percentile_with_outliers(escs)
    
    print(f"After SES outlier removal: {len(idx_clean)} observations")
    
    # Extract clean data
    df_clean = df.loc[idx_clean].copy()
    df_clean['SES_percentile_orig'] = ses_p
    df_clean['MATH'] = df_clean['PV1MATH']  # Use PV1MATH (consistent with paper)
    
    # Check for missing W_FSTUWT
    print(f"Missing W_FSTUWT: {df_clean['W_FSTUWT'].isna().sum()}")
    
    return df_clean

def calculate_ses_percentile(df):
    """
    SES percentile is already calculated in load_pisa_data()
    This function just renames the column for consistency
    """
    df['SES_percentile'] = df['SES_percentile_orig']
    return df

def apply_amf_correction(df, alpha):
    """
    Apply AMF correction formula:
    M_i* = M_i + alpha(0.5 - S_i)
    """
    df[f'MATH_corrected_a{alpha}'] = df['MATH'] + alpha * (0.5 - df['SES_percentile'])
    return df

def identify_conditional_admits(df, alpha, top_pct=0.10):
    """
    Identify conditional admits under AMF
    
    Parameters:
    -----------
    df : DataFrame
    alpha : correction strength
    top_pct : top percentage for threshold (default: 0.10 = top 10%)
    
    Returns:
    --------
    DataFrame with conditional admits
    """
    # Step 1: Calculate N based on percentage
    N = int(len(df) * top_pct)
    
    # Step 2: Sort by raw score
    df_sorted = df.sort_values('MATH', ascending=False).reset_index(drop=True)
    
    # Step 3: Regular admits (top N)
    regular_admits = df_sorted.iloc[:N].copy()
    threshold = regular_admits['MATH'].min()
    
    print(f"\n{'='*60}")
    print(f"Alpha = {alpha}")
    print(f"{'='*60}")
    print(f"Regular admits (N={N}): Top {top_pct*100:.0f}% by raw MATH score")
    print(f"Threshold (T): {threshold:.2f} (raw score of {N}th admit)")
    
    # Step 3: Non-regular admits
    non_regular = df_sorted.iloc[N:].copy()
    
    # Step 4: Apply correction to non-regular
    non_regular = apply_amf_correction(non_regular, alpha)
    
    # Step 5: Find conditional admits
    col_corrected = f'MATH_corrected_a{alpha}'
    conditional = non_regular[non_regular[col_corrected] > threshold].copy()
    conditional['gap'] = conditional[col_corrected] - threshold
    
    # SAMPLE COUNT (original method)
    n_conditional_sample = len(conditional)
    pct_sample = (n_conditional_sample / len(df)) * 100
    
    # WEIGHTED POPULATION ESTIMATE (new method)
    weight_sum_conditional = conditional['W_FSTUWT'].sum()
    weight_sum_total = df['W_FSTUWT'].sum()
    n_conditional_weighted = weight_sum_conditional
    pct_weighted = (weight_sum_conditional / weight_sum_total) * 100
    
    print(f"\n--- Sample-based (OLD) ---")
    print(f"Conditional admits: {n_conditional_sample} students")
    print(f"Percentage: {pct_sample:.2f}%")
    
    print(f"\n--- Population-weighted (NEW) ---")
    print(f"Conditional admits: {n_conditional_weighted:.0f} students (weighted)")
    print(f"Percentage: {pct_weighted:.2f}%")
    
    print(f"\n--- Gap Analysis ---")
    if len(conditional) > 0:
        print(f"Gap range: {conditional['gap'].min():.2f} to {conditional['gap'].max():.2f}")
        print(f"Mean gap: {conditional['gap'].mean():.2f}")
        print(f"Median gap: {conditional['gap'].median():.2f}")
    
    # SES distribution
    print(f"\n--- SES Distribution of Conditional Admits ---")
    if len(conditional) > 0:
        q1_count = (conditional['SES_percentile'] <= 0.25).sum()
        q2_count = ((conditional['SES_percentile'] > 0.25) & (conditional['SES_percentile'] <= 0.5)).sum()
        q3_count = ((conditional['SES_percentile'] > 0.5) & (conditional['SES_percentile'] <= 0.75)).sum()
        q4_count = (conditional['SES_percentile'] > 0.75).sum()
        
        print(f"Q1 (0-25%):   {q1_count} ({q1_count/len(conditional)*100:.1f}%)")
        print(f"Q2 (25-50%):  {q2_count} ({q2_count/len(conditional)*100:.1f}%)")
        print(f"Q3 (50-75%):  {q3_count} ({q3_count/len(conditional)*100:.1f}%)")
        print(f"Q4 (75-100%): {q4_count} ({q4_count/len(conditional)*100:.1f}%)")
        
        bottom_50_pct = ((q1_count + q2_count) / len(conditional)) * 100
        print(f"\nBottom 50% SES: {bottom_50_pct:.1f}%")
    
    return {
        'threshold': threshold,
        'conditional_sample': n_conditional_sample,
        'conditional_weighted': n_conditional_weighted,
        'pct_sample': pct_sample,
        'pct_weighted': pct_weighted,
        'conditional_df': conditional
    }

def run_full_analysis(filepath):
    """Run complete weighted analysis"""
    
    # Load data
    df = load_pisa_data(filepath)
    
    # Calculate SES percentiles
    df = calculate_ses_percentile(df)
    
    # Test different alpha values
    alphas = [5, 10, 15]
    results = {}
    
    for alpha in alphas:
        results[alpha] = identify_conditional_admits(df, alpha, top_pct=0.10)
    
    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY: Sample vs Population-Weighted Estimates")
    print(f"{'='*70}")
    print(f"{'Alpha':<10} {'Sample N':<15} {'Sample %':<15} {'Weighted N':<15} {'Weighted %':<15}")
    print(f"{'-'*70}")
    
    for alpha in alphas:
        r = results[alpha]
        print(f"{alpha:<10} {r['conditional_sample']:<15} {r['pct_sample']:<15.2f} "
              f"{r['conditional_weighted']:<15.0f} {r['pct_weighted']:<15.2f}")
    
    print(f"{'='*70}")
    print(f"Total sample size: {len(df):,}")
    print(f"Total weighted population: {df['W_FSTUWT'].sum():,.0f}")
    print(f"{'='*70}")
    
    return results, df

def create_visualization(results, df):
    """Create comparison visualization"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Sample vs Weighted comparison
    alphas = [5, 10, 15]
    sample_counts = [results[a]['conditional_sample'] for a in alphas]
    weighted_counts = [results[a]['conditional_weighted'] for a in alphas]
    
    x = np.arange(len(alphas))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, sample_counts, width, label='Sample Count', alpha=0.8)
    axes[0, 0].bar(x + width/2, weighted_counts, width, label='Weighted (Population)', alpha=0.8)
    axes[0, 0].set_xlabel('Alpha')
    axes[0, 0].set_ylabel('Number of Conditional Admits')
    axes[0, 0].set_title('Sample vs Population-Weighted Estimates')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(alphas)
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Plot 2: Percentage comparison
    sample_pcts = [results[a]['pct_sample'] for a in alphas]
    weighted_pcts = [results[a]['pct_weighted'] for a in alphas]
    
    axes[0, 1].bar(x - width/2, sample_pcts, width, label='Sample %', alpha=0.8)
    axes[0, 1].bar(x + width/2, weighted_pcts, width, label='Weighted %', alpha=0.8)
    axes[0, 1].set_xlabel('Alpha')
    axes[0, 1].set_ylabel('Percentage of Cohort')
    axes[0, 1].set_title('Percentage: Sample vs Population-Weighted')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(alphas)
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Plot 3: SES distribution (alpha=15)
    conditional_15 = results[15]['conditional_df']
    if len(conditional_15) > 0:
        axes[1, 0].hist(conditional_15['SES_percentile'], bins=20, 
                       alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(0.5, color='red', linestyle='--', 
                          linewidth=2, label='Median SES')
        axes[1, 0].set_xlabel('SES Percentile')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('SES Distribution of Conditional Admits (alpha=15)')
        axes[1, 0].legend()
        axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Plot 4: Gap distribution (alpha=15)
    if len(conditional_15) > 0:
        axes[1, 1].hist(conditional_15['gap'], bins=15, 
                       alpha=0.7, edgecolor='black', color='green')
        axes[1, 1].axvline(conditional_15['gap'].mean(), color='red', 
                          linestyle='--', linewidth=2, label=f"Mean: {conditional_15['gap'].mean():.2f}")
        axes[1, 1].set_xlabel('Gap Above Threshold')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Merit Gap Distribution (alpha=15)')
        axes[1, 1].legend()
        axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = 'amf_weighted_results.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved: {output_path}")
    
    return fig

if __name__ == "__main__":
    
    # File path (same directory)
    filepath = "pisa2022_korea.csv"
    
    print("""
    ================================================================
      AMF Weighted Population Estimation
      Based on OECD PISA 2022 Korea (N=6,377)
      Using W_FSTUWT sampling weights
    ================================================================
    """)
    
    try:
        results, df = run_full_analysis(filepath)
        fig = create_visualization(results, df)
        
        print("\nAnalysis complete!")
        print("Results saved to: amf_weighted_results.png")
        
    except FileNotFoundError:
        print("\nError: PISA data file not found!")
        print("Please ensure pisa2022_korea.csv is in the same directory")
        print("\nExpected columns: ESCS, PV1MATH-PV10MATH, W_FSTUWT")

