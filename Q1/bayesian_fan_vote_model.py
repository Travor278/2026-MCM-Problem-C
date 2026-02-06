"""
《与星共舞》粉丝投票份额贝叶斯 MCMC 模型
使用 PyMC 推断潜在粉丝投票份额，并采用不同的淘汰规则：
- 第 1-2 季：基于排名的积分系统
- 第 3 季至(BOTTOM2_START_SEASON-1):基于百分比的系统
- 赛季 >= BOTTOM2_START_SEASON:后两名规则
Dancing with the Stars Fan Voting Share Bayesian MCMC Model
Using PyMC to infer potential fan voting share, with different elimination rules applied:
- Seasons 1-2: Ranking-based points system
- Seasons 3 through (BOTTOM2_START_SEASON-1): Percentage-based system
- Seasons >= BOTTOM2_START_SEASON: Last two place rule
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import pytensor.tensor as pt
from scipy.stats import rankdata
import warnings
warnings.filterwarnings('ignore')

# Sensitivity analysis parameters
BOTTOM2_START_SEASON = 28  # Can be changed to 27 or 29
PERCENT_START_SEASON = 3

# MCMC parameters
N_SAMPLES = 2000
N_TUNE = 1000
N_CHAINS = 2
RANDOM_SEED = 42
CONSTRAINT_SCALE = 100.0

# Bottom-2 soft constraint parameters (S28+) - v3 settings
B2_SOFT_SCALE = 20.0          # Softer sigmoid boundary (was 30)
B2_PRIOR_CONCENTRATION = 5.0   # Stronger informative prior (was 3)
B2_PENALTY_THRESHOLD = 1.3     # More tolerance (was 1.2)
B2_PENALTY_STRENGTH = 20.0     # Gentler penalty (was 30)


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


def normalize_judge_percent_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize judge_percent to sum to 1 within each (season, week)."""
    df = df.copy()
    group_sums = df.groupby(['season', 'week'])['judge_percent'].transform('sum')
    df['judge_percent_normalized'] = df['judge_percent'] / group_sums
    df.loc[group_sums == 0, 'judge_percent_normalized'] = 1.0 / df.groupby(['season', 'week'])['celebrity_name'].transform('count')
    return df


def identify_eliminated_contestant(week_data: pd.DataFrame, week: int):
    """Identify who was eliminated in this week."""
    eliminated = week_data[week_data['eliminated_week'] == week]
    if len(eliminated) == 1:
        return eliminated['celebrity_name'].values[0]
    elif len(eliminated) > 1:
        return list(eliminated['celebrity_name'].values)
    return None


def compute_judge_points_rank(judge_scores: np.ndarray) -> np.ndarray:
    """Compute judge points based on ranking (highest gets N points)."""
    n = len(judge_scores)
    ranks = rankdata(-judge_scores, method='average')
    return n - ranks + 1


def build_model_rank_based_s1s2(week_data: pd.DataFrame, eliminated_name: str):
    """Build Bayesian model for Seasons 1-2 (rank-based points system)."""
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
        alpha = np.ones(n)
        fan_shares = pm.Dirichlet('fan_shares', a=alpha)
        total_score = pt.constant(judge_points_normalized) + fan_shares
        eliminated_score = total_score[eliminated_idx]

        # Constraint: eliminated has lowest total score
        for i, surv_idx in enumerate(survivor_indices):
            diff = total_score[surv_idx] - eliminated_score
            constraint = pm.math.log(pm.math.sigmoid(diff * CONSTRAINT_SCALE))
            pm.Potential(f'rank_constraint_{i}', constraint)

        # Tie-breaker: if scores close, lower fan share eliminated
        for i, surv_idx in enumerate(survivor_indices):
            score_diff = pt.abs(total_score[surv_idx] - eliminated_score)
            is_tie = pm.math.sigmoid(-score_diff * 1000 + 5)
            fan_diff = fan_shares[surv_idx] - fan_shares[eliminated_idx]
            tie_constraint = is_tie * pm.math.log(pm.math.sigmoid(fan_diff * CONSTRAINT_SCALE))
            pm.Potential(f'tie_breaker_{i}', tie_constraint)

    return model, contestants


def build_model_percentage_based(week_data: pd.DataFrame, eliminated_name: str):
    """Build Bayesian model for percentage-based elimination."""
    contestants = week_data['celebrity_name'].values
    judge_percent = week_data['judge_percent_normalized'].values.astype(float)
    n = len(contestants)

    eliminated_idx = np.where(contestants == eliminated_name)[0]
    if len(eliminated_idx) == 0:
        return None, None
    eliminated_idx = int(eliminated_idx[0])

    survivor_mask = np.ones(n, dtype=bool)
    survivor_mask[eliminated_idx] = False
    survivor_indices = np.where(survivor_mask)[0].tolist()

    with pm.Model() as model:
        alpha = np.ones(n)
        fan_shares = pm.Dirichlet('fan_shares', a=alpha)
        total_score = pt.constant(judge_percent) + fan_shares
        eliminated_score = total_score[eliminated_idx]

        for i, surv_idx in enumerate(survivor_indices):
            diff = total_score[surv_idx] - eliminated_score
            constraint = pm.math.log(pm.math.sigmoid(diff * CONSTRAINT_SCALE))
            pm.Potential(f'elim_constraint_{i}', constraint)

    return model, contestants


def build_model_bottom2(week_data: pd.DataFrame, eliminated_name: str):
    """Build Bayesian model for Bottom 2 rule with informative prior.

    S28+ rule: Bottom 2 determined by combined scores, then judges vote.

    Key modeling choices:
    1. INFORMATIVE PRIOR: Fan votes assumed to correlate with judge scores.
    2. SOFT CONSTRAINT: Eliminated must be in Bottom 2 (n_lower <= 1).
    3. NO PREFERENCE WITHIN B2: Both n_lower=0 and n_lower=1 are valid.
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
        judge_percentile = (rankdata(judge_scores) - 1) / max(n - 1, 1)
        alpha = 0.5 + B2_PRIOR_CONCENTRATION * judge_percentile

        fan_shares = pm.Dirichlet('fan_shares', a=alpha)
        total_score = pt.constant(judge_points_normalized) + fan_shares
        eliminated_score = total_score[eliminated_idx]

        # === SOFT BOTTOM 2 CONSTRAINT ===
        count_lower = pt.constant(0.0)
        for surv_idx in survivor_indices:
            diff = eliminated_score - total_score[surv_idx]
            prob_lower = pm.math.sigmoid(diff * B2_SOFT_SCALE)
            count_lower = count_lower + prob_lower

        # Soft constraint: penalize only when clearly outside Bottom 2
        excess = count_lower - B2_PENALTY_THRESHOLD
        penalty = pm.math.switch(excess > 0, -B2_PENALTY_STRENGTH * excess ** 2, 0.0)
        pm.Potential('bottom2_soft_constraint', penalty)

    return model, contestants


def get_rule_type(season: int) -> str:
    """Determine elimination rule for given season."""
    if season < PERCENT_START_SEASON:
        return 'rank_s1s2'
    elif season < BOTTOM2_START_SEASON:
        return 'percentage'
    return 'bottom2'


def fit_week_model(week_data: pd.DataFrame, season: int, week: int,
                   eliminated_name: str, verbose: bool = False):
    """Fit Bayesian model for a given (season, week)."""
    rule_type = get_rule_type(season)

    if verbose:
        name_display = str(eliminated_name)[:15] if eliminated_name else "Unknown"
        print(f"  S{season}W{week}: {rule_type}, N={len(week_data)}, Elim={name_display}...")

    try:
        if rule_type == 'rank_s1s2':
            model, contestants = build_model_rank_based_s1s2(week_data, eliminated_name)
        elif rule_type == 'percentage':
            model, contestants = build_model_percentage_based(week_data, eliminated_name)
        else:
            model, contestants = build_model_bottom2(week_data, eliminated_name)

        if model is None:
            return None, None, rule_type

        with model:
            if rule_type == 'percentage':
                # Percentage-based uses NUTS (gradient-based, faster)
                trace = pm.sample(draws=N_SAMPLES, tune=N_TUNE, chains=N_CHAINS,
                                  random_seed=RANDOM_SEED, progressbar=False,
                                  return_inferencedata=True)
            else:
                # rank_s1s2 and bottom2 use Slice (handles soft constraints better)
                trace = pm.sample(draws=N_SAMPLES, tune=N_TUNE, chains=N_CHAINS,
                                  random_seed=RANDOM_SEED, progressbar=False,
                                  return_inferencedata=True, step=pm.Slice())

        return trace, contestants, rule_type

    except Exception as e:
        if verbose:
            print(f"    ERROR: {str(e)[:80]}")
        return None, None, rule_type


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


def run_full_analysis(data_path: str, verbose: bool = True,
                      max_seasons: int = None) -> pd.DataFrame:
    """Run full Bayesian analysis for all (season, week) combinations."""
    print("DWTS Fan Vote Share Bayesian MCMC Model")
    print(f"\nBOTTOM2_START_SEASON = {BOTTOM2_START_SEASON}")
    print(f"  S1-2: Rank-based | S3-{BOTTOM2_START_SEASON-1}: Percentage | S{BOTTOM2_START_SEASON}+: Bottom 2\n")

    df = load_and_preprocess_data(data_path)
    df = normalize_judge_percent_df(df)
    season_weeks = df.groupby(['season', 'week']).size().reset_index(name='n_contestants')

    if max_seasons:
        season_weeks = season_weeks[season_weeks['season'] <= max_seasons]

    print(f"Total (season, week) combinations: {len(season_weeks)}\n")

    all_results = []
    n_success = n_skipped = n_failed = 0

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

        trace, contestants, rule_type = fit_week_model(week_data, season, week, eliminated, verbose)

        if trace is None:
            n_failed += 1
            continue

        week_results = extract_results(trace, contestants)
        week_results['season'] = season
        week_results['week'] = week
        week_results['rule_type'] = rule_type
        week_results['eliminated'] = (week_results['celebrity_name'] == eliminated)
        all_results.append(week_results)
        n_success += 1

        if verbose and n_success % 20 == 0:
            print(f"  Progress: {n_success} weeks completed")

    print(f"\nComplete: Success={n_success}, Skipped={n_skipped}, Failed={n_failed}")

    if all_results:
        final = pd.concat(all_results, ignore_index=True)
        cols = ['season', 'week', 'celebrity_name', 'fan_share_mean',
                'fan_share_hdi_lower', 'fan_share_hdi_upper', 'fan_share_std',
                'rule_type', 'eliminated']
        return final[cols]
    return pd.DataFrame()


def validate_elimination_consistency(results_df: pd.DataFrame,
                                     original_df: pd.DataFrame) -> pd.DataFrame:
    """Validate model correctly predicts eliminations."""
    validation_results = []

    for (season, week), group in results_df.groupby(['season', 'week']):
        rule_type = group['rule_type'].iloc[0]
        eliminated_row = group[group['eliminated']]
        if len(eliminated_row) == 0:
            continue
        eliminated_row = eliminated_row.iloc[0]

        week_original = original_df[
            (original_df['season'] == season) & (original_df['week'] == week)
        ].copy()

        if len(week_original) == 0:
            continue

        # Calculate judge score component based on rule type
        if rule_type == 'percentage':
            # Use percentage directly
            week_original['judge_component'] = week_original['judge_percent_normalized']
        else:
            # Use rank-based points for rank_s1s2 and bottom2
            judge_scores = week_original['total_judge_score'].values.astype(float)
            judge_points = compute_judge_points_rank(judge_scores)
            week_original['judge_component'] = judge_points / judge_points.sum()

        merged = group.merge(
            week_original[['celebrity_name', 'judge_component', 'total_judge_score']],
            on='celebrity_name', how='left'
        )
        merged['total_score'] = merged['judge_component'].fillna(0) + merged['fan_share_mean']

        elim_mask = merged['eliminated']
        if elim_mask.sum() == 0:
            continue

        eliminated_total = merged.loc[elim_mask, 'total_score'].values[0]
        survivor_totals = merged.loc[~elim_mask, 'total_score'].values

        if len(survivor_totals) == 0:
            continue

        if rule_type in ['rank_s1s2', 'percentage']:
            is_consistent = eliminated_total <= survivor_totals.min() + 1e-6
            margin = survivor_totals.min() - eliminated_total
        else:
            n_lower = np.sum(survivor_totals < eliminated_total - 1e-6)
            is_consistent = n_lower <= 1
            margin = 1 - n_lower

        validation_results.append({
            'season': season, 'week': week, 'rule_type': rule_type,
            'n_contestants': len(group), 'eliminated_name': eliminated_row['celebrity_name'],
            'eliminated_fan_share': eliminated_row['fan_share_mean'],
            'eliminated_total_score': eliminated_total,
            'min_survivor_total': survivor_totals.min(),
            'is_consistent': is_consistent, 'consistency_margin': margin
        })

    return pd.DataFrame(validation_results)


if __name__ == "__main__":
    import os

    # 绝对路径配置
    DATA_DIR = r"d:\2026mcmC\DataProcessed"
    Q1_DIR = r"d:\2026mcmC\Q1"

    DATA_PATH = os.path.join(DATA_DIR, "DWTS_Processed_Long.csv")
    OUTPUT_PATH = os.path.join(Q1_DIR, "fan_vote_estimates.csv")
    VALIDATION_PATH = os.path.join(Q1_DIR, "elimination_validation.csv")

    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Data file not found at {DATA_PATH}")
        exit(1)

    results = run_full_analysis(DATA_PATH, verbose=True)

    if len(results) > 0:
        results.to_csv(OUTPUT_PATH, index=False)
        print(f"\nResults saved to: {OUTPUT_PATH}")

        original_df = load_and_preprocess_data(DATA_PATH)
        original_df = normalize_judge_percent_df(original_df)

        validation = validate_elimination_consistency(results, original_df)
        validation.to_csv(VALIDATION_PATH, index=False)
        print(f"Validation saved to: {VALIDATION_PATH}")

        consistency_rate = validation['is_consistent'].mean() * 100
        print(f"\nConsistency Rate: {consistency_rate:.1f}% ({len(validation)} weeks)")

        for rule, grp in validation.groupby('rule_type'):
            rate = grp['is_consistent'].mean() * 100
            print(f"  {rule}: {rate:.1f}% ({len(grp)} weeks)")
