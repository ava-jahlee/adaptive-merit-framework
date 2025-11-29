"""
AMF Complete Results Generator

Imports amf_engine.py to generate all simulation results.
Includes PISA, Synthetic, Robustness, and DBN analyses.
"""
import numpy as np
import pandas as pd
from amf_engine import (
    normalize_ses_to_percentile,
    run_amf_once,
    run_pisa_amf,
    run_synthetic_amf,
    generate_synthetic_population,
    summarize_amf_result,
    ses_to_quartile,
    robustness_ses_noise,
    robustness_score_variance,
    robustness_threshold,
    robustness_alpha_grid,
    run_dbn_scenarios,
    compute_mobility_metrics,
)

# ============================================================================
# [1] PISA-based AMF Simulation
# ============================================================================
print("=" * 80)
print("[1] PISA-based AMF Simulation")
print("=" * 80)

PISA_CSV_PATH = "pisa2022_korea.csv"
SCORE_COL = "PV1MATH"
SES_COL = "ESCS"

# 1-1. Run PISA AMF
pisa_results = run_pisa_amf(
    csv_path=PISA_CSV_PATH,
    score_col=SCORE_COL,
    ses_col=SES_COL,
    alpha_list=(5, 10, 15)
)

# 1-2. Summary statistics
summary_rows = []
for alpha, res in pisa_results.items():
    summ = summarize_amf_result(res)
    summ["alpha"] = alpha
    summary_rows.append(summ)

pisa_summary_df = pd.DataFrame(summary_rows)
# Reorder columns
pisa_summary_df = pisa_summary_df[[
    "alpha", "n_total", "n_baseline", "n_amf", "n_additional",
    "mean_ses_additional", "mean_gap_additional",
    "q1_share", "q2_share", "q3_share", "q4_share",
    "threshold"
]]
pisa_summary_df.to_csv("pisa_amf_summary.csv", index=False)
print("[OK] pisa_amf_summary.csv")

# 1-3. Conditional admits detail (per alpha)
for alpha, res in pisa_results.items():
    additional_idx = res["additional_idx"]
    if len(additional_idx) == 0:
        # Empty DataFrame
        detail_df = pd.DataFrame(columns=[
            "student_id", "raw_score", "ses_percentile", "quartile",
            "correction", "corrected_score", "threshold_gap"
        ])
    else:
        scores = res["scores"].loc[additional_idx]
        ses = res["ses"].loc[additional_idx]
        correction = res["correction"].loc[additional_idx]
        corrected = res["corrected"].loc[additional_idx]
        threshold = res["threshold"]
        
        quartiles = ses_to_quartile(ses)
        threshold_gap = corrected - threshold
        
        detail_df = pd.DataFrame({
            "student_id": additional_idx,
            "raw_score": scores.values,
            "ses_percentile": ses.values,
            "quartile": quartiles.values,
            "correction": correction.values,
            "corrected_score": corrected.values,
            "threshold_gap": threshold_gap.values,
        })
    
    filename = f"pisa_additional_alpha{alpha}.csv"
    detail_df.to_csv(filename, index=False)
    print(f"[OK] {filename}")

# ============================================================================
# [2] Synthetic AMF Simulation (MC 100 runs)
# ============================================================================
print("\n" + "=" * 80)
print("[2] Synthetic AMF Simulation (Monte Carlo 100 runs)")
print("=" * 80)

# 2-1. Monte Carlo summary
all_runs, summary = run_synthetic_amf(
    n=1000,
    alpha_list=(5, 10, 15),
    n_runs=100
)

synthetic_mc_rows = []
for alpha, stats in summary.items():
    synthetic_mc_rows.append({
        "alpha": alpha,
        "n_additional_mean": stats["n_additional_mean"],
        "n_additional_sd": stats["n_additional_sd"],
    })

synthetic_mc_df = pd.DataFrame(synthetic_mc_rows)
synthetic_mc_df.to_csv("synthetic_mc_summary.csv", index=False)
print("[OK] synthetic_mc_summary.csv")

# 2-2. Synthetic single run detail
np.random.seed(42)
scores_syn, ses_syn = generate_synthetic_population(
    n=1000,
    mean_score=500,
    sd_score=100,
    ses_mean=0.0,
    ses_sd=1.0,
    corr=0.38
)

for alpha in [5, 10, 15]:
    res = run_amf_once(scores_syn, ses_syn, alpha)
    additional_idx = res["additional_idx"]
    
    if len(additional_idx) == 0:
        detail_df = pd.DataFrame(columns=[
            "student_id", "raw_score", "ses_percentile", "quartile",
            "correction", "corrected_score", "threshold_gap"
        ])
    else:
        scores = res["scores"].loc[additional_idx]
        ses = res["ses"].loc[additional_idx]
        correction = res["correction"].loc[additional_idx]
        corrected = res["corrected"].loc[additional_idx]
        threshold = res["threshold"]
        
        quartiles = ses_to_quartile(ses)
        threshold_gap = corrected - threshold
        
        detail_df = pd.DataFrame({
            "student_id": additional_idx,
            "raw_score": scores.values,
            "ses_percentile": ses.values,
            "quartile": quartiles.values,
            "correction": correction.values,
            "corrected_score": corrected.values,
            "threshold_gap": threshold_gap.values,
        })
    
    filename = f"synthetic_additional_alpha{alpha}.csv"
    detail_df.to_csv(filename, index=False)
    print(f"[OK] {filename}")

# ============================================================================
# [3] Robustness Tests (PISA-based)
# ============================================================================
print("\n" + "=" * 80)
print("[3] Robustness Tests (PISA-based, alpha=10)")
print("=" * 80)

# Prepare PISA data (using alpha=10 result)
pisa_res_alpha10 = pisa_results[10]
pisa_scores = pisa_res_alpha10["scores"]
pisa_ses = pisa_res_alpha10["ses"]

# 3-1. SES noise robustness
df_ses_noise = robustness_ses_noise(
    scores=pisa_scores,
    ses_percentile=pisa_ses,
    alpha=10,
    noise_levels=(0.05, 0.10),
    n_runs=50,
    random_seed=1234
)
df_ses_noise.to_csv("robustness_ses_noise.csv", index=False)
print("[OK] robustness_ses_noise.csv")

# 3-2. Score variance robustness
df_score_var = robustness_score_variance(
    scores=pisa_scores,
    ses_percentile=pisa_ses,
    alpha=10,
    var_scales=(0.8, 1.0, 1.2)
)
df_score_var.to_csv("robustness_score_variance.csv", index=False)
print("[OK] robustness_score_variance.csv")

# 3-3. Threshold robustness
df_threshold = robustness_threshold(
    scores=pisa_scores,
    ses_percentile=pisa_ses,
    alpha=10,
    top_pcts=(0.05, 0.10, 0.15)
)
df_threshold.to_csv("robustness_threshold.csv", index=False)
print("[OK] robustness_threshold.csv")

# 3-4. Alpha grid robustness
df_alpha_grid = robustness_alpha_grid(
    scores=pisa_scores,
    ses_percentile=pisa_ses,
    alphas=(2, 4, 6, 8, 10, 12, 15, 20)
)
df_alpha_grid.to_csv("robustness_alpha_grid.csv", index=False)
print("[OK] robustness_alpha_grid.csv")

# ============================================================================
# [4] DBN / Long-run Simulation
# ============================================================================
print("\n" + "=" * 80)
print("[4] DBN / Long-run Simulation")
print("=" * 80)

# 4-1. Compute admission probability per quartile from PISA
# Baseline (using alpha=10)
res_base = pisa_results[10]
threshold_raw = res_base["threshold"]
scores_pisa = res_base["scores"]
ses_pisa = res_base["ses"]

# Compute quartiles
quartiles_pisa = ses_to_quartile(ses_pisa)

# Baseline admits
baseline_admits = scores_pisa >= threshold_raw

# AMF admits
corrected_scores = res_base["corrected"]
amf_admits = corrected_scores >= threshold_raw

# Compute probability per quartile
p_admit_base = np.zeros(4)
p_admit_amf = np.zeros(4)

for q in [1, 2, 3, 4]:
    mask_q = quartiles_pisa == q
    n_q = mask_q.sum()
    if n_q > 0:
        p_admit_base[q-1] = baseline_admits[mask_q].sum() / n_q
        p_admit_amf[q-1] = amf_admits[mask_q].sum() / n_q

# Save to CSV
dbn_prob_rows = []
for q in [1, 2, 3, 4]:
    dbn_prob_rows.append({
        "scenario": "baseline",
        "quartile": q,
        "p_admit": p_admit_base[q-1]
    })
    dbn_prob_rows.append({
        "scenario": "amf",
        "quartile": q,
        "p_admit": p_admit_amf[q-1]
    })

dbn_prob_df = pd.DataFrame(dbn_prob_rows)
dbn_prob_df.to_csv("dbn_admission_probs.csv", index=False)
print("[OK] dbn_admission_probs.csv")

# 4-2. Set mobility matrices
M_admit = np.array([
    [0.50, 0.30, 0.15, 0.05],  # Parent Q1 -> Child Q1~Q4 (if admitted)
    [0.20, 0.40, 0.30, 0.10],  # Q2
    [0.10, 0.25, 0.40, 0.25],  # Q3
    [0.05, 0.15, 0.30, 0.50],  # Q4
])

M_not = np.array([
    [0.70, 0.20, 0.08, 0.02],  # Parent Q1 -> Child Q1~Q4 (if not admitted)
    [0.30, 0.40, 0.20, 0.10],  # Q2
    [0.15, 0.30, 0.35, 0.20],  # Q3
    [0.10, 0.20, 0.30, 0.40],  # Q4
])

# Initial distribution (unequal state)
v0 = np.array([0.35, 0.30, 0.20, 0.15])

# 4-3. Run DBN simulation
traj_df = run_dbn_scenarios(
    p_admit_base=p_admit_base,
    p_admit_amf=p_admit_amf,
    M_admit=M_admit,
    M_not=M_not,
    v0=v0,
    T=30
)
traj_df.to_csv("dbn_trajectory_quartiles.csv", index=False)
print("[OK] dbn_trajectory_quartiles.csv")

# Compute mobility metrics
metrics_df = compute_mobility_metrics(traj_df)
metrics_df.to_csv("dbn_mobility_metrics.csv", index=False)
print("[OK] dbn_mobility_metrics.csv")

# ============================================================================
# [5] Output Summary
# ============================================================================
print("\n" + "=" * 80)
print("Generated files:")
print("=" * 80)

generated_files = [
    "pisa_amf_summary.csv",
    "pisa_additional_alpha5.csv",
    "pisa_additional_alpha10.csv",
    "pisa_additional_alpha15.csv",
    "synthetic_mc_summary.csv",
    "synthetic_additional_alpha5.csv",
    "synthetic_additional_alpha10.csv",
    "synthetic_additional_alpha15.csv",
    "robustness_ses_noise.csv",
    "robustness_score_variance.csv",
    "robustness_threshold.csv",
    "robustness_alpha_grid.csv",
    "dbn_admission_probs.csv",
    "dbn_trajectory_quartiles.csv",
    "dbn_mobility_metrics.csv",
]

for f in generated_files:
    print(f"- {f}")

print("\n[OK] All results generated successfully!")
