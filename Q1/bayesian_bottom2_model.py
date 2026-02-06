"""
DWTS S28+ 后两名规则专用模型(v3 - 增强版)

适用于第 28 季及以后的赛季：后两名由评委和观众评分综合决定，
然后评委进行现场投票以决定淘汰人选。

关键要点：
1. 在后两名中，淘汰由评委投票决定（而非最低分获胜）
2. 我们仅要求被淘汰的选手位于后两名中(n_lower <= 1)
3. n_lower=0(最低分)和 n_lower=1(第二低分)均有效

v3 版本模型改进：
- 增强型 MCMC:4 条链,3000 次抽样以获得更好的收敛性
- 更强的信息先验：集中度=5.0(评委-观众相关性)
- 更宽松的约束：尺度=20,阈值=1.3,惩罚=20
- 目标：更均衡的 n_lower 分布(并非全部为 0)
DWTS S28+ Last Two-Player Rule-Specific Model (v3 - Enhanced Version)
Applicable to Season 28 and beyond: The last two spots are determined by a combination of judge and audience scores,
and then the judges vote live to decide who gets eliminated.

Key Points:

1. In the last two spots, elimination is decided by judge vote (not the lowest score wins).
2. We only require the eliminated contestant to be in the last two spots (n_lower <= 1).
3. n_lower=0 (lowest score) and n_lower=1 (second lowest score) are both valid.

v3 Model Improvements:
- Enhanced MCMC: 4 chains, 3000 samplings for better convergence
- Stronger prior information: Concentration = 5.0 (judge-audience correlation)
- More lenient constraints: Scale = 20, Threshold = 1.3, Penalty = 20
- Goal: A more balanced n_lower distribution (not all n_lower values are 0)
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import pytensor.tensor as pt
from scipy.stats import rankdata
import warnings
warnings.filterwarnings('ignore')

# MCMC parameters (enhanced for better convergence)
N_SAMPLES = 3000               # Increased from 2000
N_TUNE = 1500                  # Increased from 1000
N_CHAINS = 4                   # Increased from 2 for better R-hat
RANDOM_SEED = 42
BOTTOM2_START_SEASON = 28

# Model parameters for soft constraint (v3 - more relaxed)
SOFT_CONSTRAINT_SCALE = 20.0   # Even softer boundary (was 30)
PRIOR_CONCENTRATION = 5.0      # Stronger informative prior (was 3)
PENALTY_THRESHOLD = 1.3        # More tolerance (was 1.2)
PENALTY_STRENGTH = 20.0        # Gentler penalty (was 30)


def load_and_preprocess_data(filepath: str) -> pd.DataFrame:
    """Load and preprocess DWTS data."""
    df = pd.read_csv(filepath, encoding='utf-8-sig')

    numeric_cols = ['season', 'week', 'total_judge_score', 'judge_percent',
                    'judge_rank', 'contestants_this_week', 'eliminated_week']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['season', 'week', 'total_judge_score', 'celebrity_name'])
    df = df[df['total_judge_score'] > 0].copy()
    return df


def compute_judge_points_rank(judge_scores: np.ndarray) -> np.ndarray:
    """Compute judge points based on ranking (highest gets N points)."""
    n = len(judge_scores)
    ranks = rankdata(-judge_scores, method='average')
    return n - ranks + 1


def identify_eliminated_contestant(week_data: pd.DataFrame, week: int):
    """Identify who was eliminated in this week."""
    eliminated = week_data[week_data['eliminated_week'] == week]
    if len(eliminated) == 1:
        return eliminated['celebrity_name'].values[0]
    elif len(eliminated) > 1:
        return list(eliminated['celebrity_name'].values)
    return None


def build_model_bottom2(week_data: pd.DataFrame, eliminated_name: str):
    """Build Bayesian model for Bottom 2 rule with informative prior.

    S28+ rule: Bottom 2 determined by combined scores, then judges vote.

    Key modeling choices:
    1. INFORMATIVE PRIOR: Fan votes assumed to correlate with judge scores.
       Higher judge score -> higher expected fan share.
    2. SOFT CONSTRAINT: Eliminated must be in Bottom 2 (n_lower <= 1).
       Uses gentle penalty that allows some flexibility.
    3. NO PREFERENCE WITHIN B2: Both n_lower=0 and n_lower=1 are valid.
       The model doesn't force eliminated to be absolute lowest.
    """
    contestants = week_data['celebrity_name'].values
    judge_scores = week_data['total_judge_score'].values.astype(float)
    n = len(contestants)

    eliminated_idx = np.where(contestants == eliminated_name)[0]
    if len(eliminated_idx) == 0:
        return None, None
    eliminated_idx = int(eliminated_idx[0])

    judge_points = compute_judge_points_rank(judge_scores)
    judge_points_normalized = judge_points / judge_points.sum()

    survivor_mask = np.ones(n, dtype=bool)
    survivor_mask[eliminated_idx] = False
    survivor_indices = np.where(survivor_mask)[0].tolist()

    with pm.Model() as model:
        # === INFORMATIVE PRIOR ===
        # Assume fan votes correlate with judge performance
        # Use judge score percentile to set Dirichlet alpha
        # Higher judge score -> higher alpha -> higher expected fan share
        judge_percentile = (rankdata(judge_scores) - 1) / max(n - 1, 1)  # 0 to 1
        alpha = 0.5 + PRIOR_CONCENTRATION * judge_percentile  # Range: [0.5, 3.5]

        fan_shares = pm.Dirichlet('fan_shares', a=alpha)
        total_score = pt.constant(judge_points_normalized) + fan_shares
        eliminated_score = total_score[eliminated_idx]

        # === SOFT BOTTOM 2 CONSTRAINT ===
        # Count survivors with lower total score than eliminated
        # Lower SOFT_CONSTRAINT_SCALE makes the sigmoid boundary less sharp
        count_lower = pt.constant(0.0)
        for surv_idx in survivor_indices:
            diff = eliminated_score - total_score[surv_idx]
            prob_lower = pm.math.sigmoid(diff * SOFT_CONSTRAINT_SCALE)
            count_lower = count_lower + prob_lower

        # Soft constraint: penalize only when clearly outside Bottom 2
        # PENALTY_THRESHOLD > 1 allows some tolerance (n_lower can be ~1)
        # This enables both n_lower=0 (lowest) and n_lower=1 (2nd lowest)
        excess = count_lower - PENALTY_THRESHOLD
        penalty = pm.math.switch(excess > 0, -PENALTY_STRENGTH * excess ** 2, 0.0)
        pm.Potential('bottom2_soft_constraint', penalty)

    return model, contestants


def fit_week_model(week_data: pd.DataFrame, season: int, week: int,
                   eliminated_name: str, verbose: bool = True):
    """Fit the Bayesian model for a (season, week) pair."""
    if verbose:
        print(f"  S{season}W{week}: N={len(week_data)}, Eliminated={eliminated_name}")

    try:
        model, contestants = build_model_bottom2(week_data, eliminated_name)

        if model is None:
            return None, None

        with model:
            trace = pm.sample(
                draws=N_SAMPLES, tune=N_TUNE, chains=N_CHAINS,
                random_seed=RANDOM_SEED, progressbar=verbose,
                return_inferencedata=True, step=pm.Slice()
            )

        return trace, contestants

    except Exception as e:
        if verbose:
            print(f"    ERROR: {str(e)[:80]}")
        return None, None


def extract_results(trace, contestants: np.ndarray) -> pd.DataFrame:
    """Extract fan share estimates and HDI from trace."""
    fan_shares_samples = trace.posterior['fan_shares'].values
    fan_shares_samples = fan_shares_samples.reshape(-1, len(contestants))

    results = []
    for i, name in enumerate(contestants):
        samples = fan_shares_samples[:, i]
        mean = np.mean(samples)
        hdi = az.hdi(samples, hdi_prob=0.95)
        results.append({
            'celebrity_name': name,
            'fan_share_mean': mean,
            'fan_share_hdi_lower': hdi[0],
            'fan_share_hdi_upper': hdi[1],
            'fan_share_std': np.std(samples)
        })
    return pd.DataFrame(results)


def compute_diagnostics(trace, contestants: np.ndarray) -> dict:
    """Compute MCMC diagnostics: R-hat and ESS."""
    summary = az.summary(trace, var_names=['fan_shares'])
    r_hat_values = summary['r_hat'].values
    ess_bulk = summary['ess_bulk'].values
    ess_tail = summary['ess_tail'].values

    return {
        'r_hat_max': np.max(r_hat_values),
        'r_hat_mean': np.mean(r_hat_values),
        'ess_bulk_min': np.min(ess_bulk),
        'ess_tail_min': np.min(ess_tail),
        'converged': np.max(r_hat_values) < 1.05
    }


def validate_bottom2_constraint(week_results: pd.DataFrame, eliminated_name: str) -> dict:
    """Validate that eliminated contestant is in Bottom 2."""
    elim_row = week_results[week_results['celebrity_name'] == eliminated_name]
    if len(elim_row) == 0:
        return {'is_consistent': False, 'n_lower': -1, 'eliminated_fan_share': np.nan}

    elim_score = elim_row['fan_share_mean'].values[0]
    survivors = week_results[week_results['celebrity_name'] != eliminated_name]

    # Count survivors with lower fan share (proxy for lower total score)
    n_lower = (survivors['fan_share_mean'] < elim_score).sum()

    return {
        'is_consistent': n_lower <= 1,  # In Bottom 2
        'n_lower': int(n_lower),
        'eliminated_fan_share': elim_score,
        'consistency_margin': 1 - n_lower  # 1 if lowest, 0 if exactly 1 lower
    }


def run_bottom2_analysis(data_path: str, verbose: bool = True) -> tuple:
    """Run S28+ Bottom-2 analysis with soft constraints."""
    print("DWTS S28+ Bottom-2 Rule Model (v3 - Enhanced)")
    print(f"\nMCMC: Draws={N_SAMPLES}, Tune={N_TUNE}, Chains={N_CHAINS}, Sampler=Slice")
    print(f"Prior: Informative (concentration={PRIOR_CONCENTRATION})")
    print(f"Constraint: Soft (scale={SOFT_CONSTRAINT_SCALE}, threshold={PENALTY_THRESHOLD}, penalty={PENALTY_STRENGTH})")
    print(f"Goal: Balanced n_lower distribution reflecting judge vote randomness\n")

    df = load_and_preprocess_data(data_path)
    df = df[df['season'] >= BOTTOM2_START_SEASON]

    season_weeks = df.groupby(['season', 'week']).size().reset_index(name='n_contestants')
    print(f"Total (season, week) combinations: {len(season_weeks)}\n")

    all_results = []
    all_diagnostics = []
    all_validations = []
    n_success = n_skipped = 0

    for _, row in season_weeks.iterrows():
        season, week = int(row['season']), int(row['week'])
        week_data = df[(df['season'] == season) & (df['week'] == week)].copy()

        if len(week_data) < 2:
            n_skipped += 1
            continue

        eliminated = identify_eliminated_contestant(week_data, week)
        if eliminated is None:
            n_skipped += 1
            continue
        if isinstance(eliminated, list):
            eliminated = eliminated[0]

        trace, contestants = fit_week_model(week_data, season, week, eliminated, verbose)

        if trace is None:
            continue

        week_results = extract_results(trace, contestants)
        week_results['season'] = season
        week_results['week'] = week
        week_results['eliminated'] = (week_results['celebrity_name'] == eliminated)

        diag = compute_diagnostics(trace, contestants)
        diag['season'] = season
        diag['week'] = week
        diag['n_contestants'] = len(contestants)

        validation = validate_bottom2_constraint(week_results, eliminated)
        validation['season'] = season
        validation['week'] = week

        all_results.append(week_results)
        all_diagnostics.append(diag)
        all_validations.append(validation)
        n_success += 1

        if verbose:
            status = "Converged" if diag['converged'] else "NOT converged"
            b2_status = "In B2" if validation['is_consistent'] else "NOT in B2"
            print(f"    R-hat={diag['r_hat_max']:.3f}, ESS_bulk={diag['ess_bulk_min']:.0f} [{status}]")
            print(f"    n_lower={validation['n_lower']}, [{b2_status}]\n")

    print(f"\nComplete: Success={n_success}, Skipped={n_skipped}")

    if all_results:
        results_df = pd.concat(all_results, ignore_index=True)
        cols = ['season', 'week', 'celebrity_name', 'fan_share_mean',
                'fan_share_hdi_lower', 'fan_share_hdi_upper', 'fan_share_std', 'eliminated']
        results_df = results_df[cols]
        diagnostics_df = pd.DataFrame(all_diagnostics)
        validation_df = pd.DataFrame(all_validations)

        # Summary stats
        print(f"\n--- Validation Summary ---")
        n_consistent = validation_df['is_consistent'].sum()
        print(f"Bottom 2 consistency: {n_consistent}/{len(validation_df)} = {n_consistent/len(validation_df)*100:.1f}%")

        n_lower_dist = validation_df['n_lower'].value_counts().sort_index()
        print(f"\nn_lower distribution:")
        for n, count in n_lower_dist.items():
            print(f"  {n}: {count} weeks")

        return results_df, diagnostics_df, validation_df

    return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


if __name__ == "__main__":
    import os

    # 绝对路径配置
    DATA_DIR = r"d:\2026mcmC\DataProcessed"
    Q1_DIR = r"d:\2026mcmC\Q1"
    Q2_DIR = r"d:\2026mcmC\Q2"

    DATA_PATH = os.path.join(DATA_DIR, "DWTS_Processed_Long.csv")
    OUTPUT_PATH = os.path.join(Q1_DIR, "fan_vote_bottom2.csv")
    DIAGNOSTICS_PATH = os.path.join(Q2_DIR, "bottom2_mcmc_diagnostics.csv")
    VALIDATION_PATH = os.path.join(Q2_DIR, "bottom2_validation.csv")

    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Data file not found at {DATA_PATH}")
        exit(1)

    results, diagnostics, validation = run_bottom2_analysis(DATA_PATH, verbose=True)

    if len(results) > 0:
        results.to_csv(OUTPUT_PATH, index=False)
        print(f"\nResults saved to: {OUTPUT_PATH}")

        diagnostics.to_csv(DIAGNOSTICS_PATH, index=False)
        print(f"Diagnostics saved to: {DIAGNOSTICS_PATH}")

        validation.to_csv(VALIDATION_PATH, index=False)
        print(f"Validation saved to: {VALIDATION_PATH}")

        print(f"\nConvergence: {diagnostics['converged'].sum()}/{len(diagnostics)} weeks")
        print(f"Max R-hat: {diagnostics['r_hat_max'].max():.4f}")
        print(f"Min ESS (bulk): {diagnostics['ess_bulk_min'].min():.0f}")