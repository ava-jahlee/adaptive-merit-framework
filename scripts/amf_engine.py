import numpy as np
import pandas as pd
from scipy.stats import rankdata

RNG = np.random.default_rng(42)  # Global seed for reproducibility

# 1. SES outlier removal + percentile conversion
def normalize_ses_to_percentile(escs: pd.Series, iqr_k=1.5):
    q1 = escs.quantile(0.25)
    q3 = escs.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - iqr_k * iqr
    upper = q3 + iqr_k * iqr
    mask = (escs >= lower) & (escs <= upper)
    escs_clean = escs[mask].copy()
    
    # percentile rank (0~1)
    ranks = rankdata(escs_clean, method="average")
    ses_percentile = (ranks - 1) / (len(escs_clean) - 1)
    return escs_clean.index, pd.Series(ses_percentile, index=escs_clean.index)

# 2. AMF core engine
def run_amf_once(scores: pd.Series, ses_percentile: pd.Series, alpha: float, top_pct=0.10):
    # Use only common indices
    idx = scores.index.intersection(ses_percentile.index)
    scores = scores.loc[idx]
    ses = ses_percentile.loc[idx]
    
    # baseline threshold (top 10% raw)
    threshold = np.percentile(scores, 100 * (1 - top_pct))  # 0.9 -> top 10%
    
    # correction
    correction = alpha * (0.5 - ses)
    correction[correction < 0] = 0.0
    corrected = scores + correction
    
    # baseline & AMF admits
    baseline_admits = scores >= threshold
    amf_admits = corrected >= threshold
    
    additional_mask = (~baseline_admits) & amf_admits
    
    result = {
        "threshold": threshold,
        "n_total": len(scores),
        "n_baseline": baseline_admits.sum(),
        "n_amf": amf_admits.sum(),
        "n_additional": additional_mask.sum(),
        "additional_idx": scores.index[additional_mask].tolist(),
        "scores": scores,
        "ses": ses,
        "corrected": corrected,
        "correction": correction
    }
    return result

def run_pisa_amf(csv_path: str, score_col="PV1MATH", ses_col="ESCS", alpha_list=(5, 10, 15)):
    df = pd.read_csv(csv_path)
    
    # Clean SES
    escs = df[ses_col]
    idx_clean, ses_p = normalize_ses_to_percentile(escs)
    
    # Scores
    scores = df.loc[idx_clean, score_col]
    
    results = {}
    for alpha in alpha_list:
        res = run_amf_once(scores, ses_p, alpha)
        results[alpha] = res
    return results

def generate_synthetic_population(
    n=1000, 
    mean_score=500, 
    sd_score=100, 
    ses_mean=0.0, 
    ses_sd=1.0,
    corr=0.38
):
    # Build covariance matrix for joint SES & score sampling
    cov = np.array([
        [ses_sd**2, corr * ses_sd * sd_score],
        [corr * ses_sd * sd_score, sd_score**2]
    ])
    mean = np.array([ses_mean, mean_score])
    
    samples = RNG.multivariate_normal(mean, cov, size=n)
    ses_raw = samples[:, 0]
    scores = samples[:, 1]
    
    # SES -> percentile
    ses_series = pd.Series(ses_raw, index=np.arange(n))
    idx_clean, ses_p = normalize_ses_to_percentile(ses_series)
    
    scores = pd.Series(scores, index=np.arange(n))
    scores = scores.loc[idx_clean]
    
    return scores, ses_p

def run_synthetic_amf(
    n=1000, 
    alpha_list=(5, 10, 15), 
    n_runs=100
):
    all_runs = {alpha: [] for alpha in alpha_list}
    
    for run in range(n_runs):
        # Fixed seed: different per run, reproducible overall
        local_rng = np.random.default_rng(42 + run)
        
        # SES & score sampling (independent per run)
        cov = np.array([
            [1.0, 0.38 * 1.0 * 100],
            [0.38 * 1.0 * 100, 100**2]
        ])
        mean = np.array([0.0, 500.0])
        samples = local_rng.multivariate_normal(mean, cov, size=n)
        ses_raw = pd.Series(samples[:,0], index=np.arange(n))
        scores = pd.Series(samples[:,1], index=np.arange(n))
        
        idx_clean, ses_p = normalize_ses_to_percentile(ses_raw)
        scores = scores.loc[idx_clean]
        ses_p = ses_p  # Already based on idx_clean
        
        for alpha in alpha_list:
            res = run_amf_once(scores, ses_p, alpha)
            all_runs[alpha].append(res)
    
    # Build summary statistics
    summary = {}
    for alpha, runs in all_runs.items():
        n_add = np.array([r["n_additional"] for r in runs])
        summary[alpha] = {
            "n_additional_mean": n_add.mean(),
            "n_additional_sd": n_add.std(ddof=1),
        }
    return all_runs, summary

# SES percentile -> quartile (1~4)
def ses_to_quartile(ses_percentile: pd.Series) -> pd.Series:
    return pd.cut(
        ses_percentile,
        bins=[0.0, 0.25, 0.50, 0.75, 1.0000001],
        labels=[1, 2, 3, 4],
        include_lowest=True
    ).astype(int)

# Extract key statistics from run_amf_once result
def summarize_amf_result(res: dict) -> dict:
    scores = res["scores"]
    ses = res["ses"]
    corrected = res["corrected"]
    correction = res["correction"]
    threshold = res["threshold"]

    additional_idx = res["additional_idx"]
    add_scores = scores.loc[additional_idx]
    add_ses = ses.loc[additional_idx]
    add_corr = correction.loc[additional_idx]
    add_corr_score = corrected.loc[additional_idx]
    
    if len(additional_idx) > 0:
        gaps = add_corr_score - threshold
        quartiles = ses_to_quartile(add_ses)
        q_counts = quartiles.value_counts(normalize=True).reindex([1,2,3,4], fill_value=0.0)
        add_mean_ses = add_ses.mean()
        add_mean_gap = gaps.mean()
    else:
        q_counts = pd.Series([0,0,0,0], index=[1,2,3,4])
        add_mean_ses = np.nan
        add_mean_gap = np.nan

    return {
        "n_total": res["n_total"],
        "n_baseline": res["n_baseline"],
        "n_amf": res["n_amf"],
        "n_additional": res["n_additional"],
        "mean_ses_additional": add_mean_ses,
        "mean_gap_additional": add_mean_gap,
        "q1_share": q_counts.loc[1],
        "q2_share": q_counts.loc[2],
        "q3_share": q_counts.loc[3],
        "q4_share": q_counts.loc[4],
        "threshold": threshold,
    }

def robustness_ses_noise(scores: pd.Series,
                         ses_percentile: pd.Series,
                         alpha: float,
                         noise_levels=(0.05, 0.10),
                         n_runs=50,
                         top_pct=0.10,
                         random_seed=1234):
    """
    SES measurement error robustness.
    Both correction AND quartile classification use noisy SES consistently.
    """
    rng = np.random.default_rng(random_seed)
    rows = []

    # 0) Baseline: true SES for correction AND evaluation
    base_res = run_amf_once(scores, ses_percentile, alpha, top_pct=top_pct)
    base_summary = summarize_amf_result(base_res)
    base_summary["scenario"] = "baseline"
    base_summary["noise_level"] = 0.0
    base_summary["run"] = 0
    rows.append(base_summary)

    for nl in noise_levels:
        for r in range(n_runs):
            # 1) Generate noisy SES for correction
            noise = rng.normal(loc=0.0, scale=nl, size=len(ses_percentile))
            ses_noisy = np.clip(ses_percentile + noise, 0.0, 1.0)
            ses_noisy = pd.Series(ses_noisy, index=ses_percentile.index)

            # 2) Apply AMF correction using noisy SES
            res = run_amf_once(scores, ses_noisy, alpha, top_pct=top_pct)

            # 3) Summarize using noisy SES for both correction and quartiling
            summ = summarize_amf_result(res)
            summ["scenario"] = "ses_noise"
            summ["noise_level"] = nl
            summ["run"] = r + 1
            rows.append(summ)

    df = pd.DataFrame(rows)
    return df

def robustness_score_variance(scores: pd.Series,
                              ses_percentile: pd.Series,
                              alpha: float,
                              var_scales=(0.8, 1.0, 1.2),
                              top_pct=0.10):
    rows = []
    mean_score = scores.mean()
    centered = scores - mean_score
    for vs in var_scales:
        scaled_scores = centered * np.sqrt(vs) + mean_score
        res = run_amf_once(scaled_scores, ses_percentile, alpha, top_pct=top_pct)
        summ = summarize_amf_result(res)
        summ["scenario"] = "score_variance"
        summ["var_scale"] = vs
        rows.append(summ)
    
    return pd.DataFrame(rows)

def robustness_threshold(scores: pd.Series,
                         ses_percentile: pd.Series,
                         alpha: float,
                         top_pcts=(0.05, 0.10, 0.15)):
    rows = []
    for tp in top_pcts:
        res = run_amf_once(scores, ses_percentile, alpha, top_pct=tp)
        summ = summarize_amf_result(res)
        summ["scenario"] = "threshold"
        summ["top_pct"] = tp
        rows.append(summ)
    return pd.DataFrame(rows)

def robustness_alpha_grid(scores: pd.Series,
                          ses_percentile: pd.Series,
                          alphas=(2, 4, 6, 8, 10, 12, 15, 20),
                          top_pct=0.10):
    rows = []
    for a in alphas:
        res = run_amf_once(scores, ses_percentile, a, top_pct=top_pct)
        summ = summarize_amf_result(res)
        summ["scenario"] = "alpha_grid"
        summ["alpha"] = a
        rows.append(summ)
    return pd.DataFrame(rows)

def save_robustness_results(base_name: str,
                            ses_noise_df=None,
                            var_df=None,
                            th_df=None,
                            alpha_df=None):
    if ses_noise_df is not None:
        ses_noise_df.to_csv(f"{base_name}_ses_noise.csv", index=False)
    if var_df is not None:
        var_df.to_csv(f"{base_name}_score_variance.csv", index=False)
    if th_df is not None:
        th_df.to_csv(f"{base_name}_threshold.csv", index=False)
    if alpha_df is not None:
        alpha_df.to_csv(f"{base_name}_alpha_grid.csv", index=False)

def simulate_dbn_generations(
    p_admit: np.ndarray,
    M_admit: np.ndarray,
    M_not: np.ndarray,
    v0: np.ndarray = None,
    T: int = 30
):
    """
    p_admit: shape (4,), admission probability per SES quartile (Q1~Q4)
    M_admit: shape (4,4), parent quartile -> child quartile (if admitted)
    M_not:   shape (4,4), parent quartile -> child quartile (if not admitted)
    v0:      shape (4,), initial SES distribution (sum=1). None = uniform.
    T:       number of generations
    """
    p_admit = np.asarray(p_admit, dtype=float)
    M_admit = np.asarray(M_admit, dtype=float)
    M_not = np.asarray(M_not, dtype=float)

    if v0 is None:
        v = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)
    else:
        v = np.asarray(v0, dtype=float)
        v = v / v.sum()

    traj = [v.copy()]  # t=0
    for t in range(T):
        # Admit / not-admit ratio per parent quartile
        admitted = v * p_admit       # shape (4,)
        not_admitted = v * (1 - p_admit)

        # Next generation distribution: admitted * M_admit + not_admitted * M_not
        v_next = admitted @ M_admit + not_admitted @ M_not
        v_next = v_next / v_next.sum()
        traj.append(v_next)

        v = v_next

    traj = np.vstack(traj)  # shape (T+1, 4)
    df = pd.DataFrame(
        traj,
        columns=["Q1", "Q2", "Q3", "Q4"]
    )
    df["generation"] = np.arange(T+1)
    return df

def run_dbn_scenarios(
    p_admit_base: np.ndarray,
    p_admit_amf: np.ndarray,
    M_admit: np.ndarray,
    M_not: np.ndarray,
    v0: np.ndarray = None,
    T: int = 30
):
    """
    p_admit_base: baseline admission probability (Q1~Q4)
    p_admit_amf:  admission probability after AMF (Q1~Q4)
    """
    base_traj = simulate_dbn_generations(
        p_admit_base, M_admit, M_not, v0=v0, T=T
    )
    base_traj["scenario"] = "baseline"
    amf_traj = simulate_dbn_generations(
        p_admit_amf, M_admit, M_not, v0=v0, T=T
    )
    amf_traj["scenario"] = "amf"
    df = pd.concat([base_traj, amf_traj], ignore_index=True)
    return df

def compute_mobility_metrics(traj_df: pd.DataFrame) -> pd.DataFrame:
    df = traj_df.copy()
    df["top2_share"] = df["Q3"] + df["Q4"]
    df["bottom_share"] = df["Q1"]
    df["top_bottom_ratio"] = df["top2_share"] / (df["bottom_share"] + 1e-9)
    return df
