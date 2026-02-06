"""
比较第一赛季和第二赛季的排名评分与百分比评分

使用增强后的第一赛季和第二赛季数据（排名评分），并拟合百分比模型
以比较哪种方法能带来更高的球迷影响力。
Compare Rank-Based vs Percentage-Based Scoring for Seasons 1-2

Uses the enhanced S1-S2 data (rank-based) and fits percentage-based models
to compare which method gives higher fan influence.
"""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from scipy.stats import rankdata
import warnings
warnings.filterwarnings('ignore')

# MCMC parameters (matching enhanced S1-S2 settings)
N_SAMPLES = 5000
N_TUNE = 2000
N_CHAINS = 4
RANDOM_SEED = 42
CONSTRAINT_SCALE = 100.0


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, encoding='utf-8-sig')
    numeric_cols = ['season', 'week', 'total_judge_score', 'judge_percent',
                    'judge_rank', 'contestants_this_week', 'eliminated_week']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['season', 'week', 'total_judge_score', 'celebrity_name'])
    df = df[df['total_judge_score'] > 0].copy()
    return df


def build_model_percentage(week_data: pd.DataFrame, eliminated_idx: int):
    """Build model using PERCENTAGE-based scoring (like S3-27)."""
    judge_scores = week_data['total_judge_score'].values.astype(float)
    n = len(judge_scores)

    # Direct percentage (no rank transformation)
    judge_normalized = judge_scores / judge_scores.sum()

    survivor_indices = [i for i in range(n) if i != eliminated_idx]

    with pm.Model() as model:
        alpha = np.ones(n)
        fan_shares = pm.Dirichlet('fan_shares', a=alpha)
        total_score = pt.constant(judge_normalized) + fan_shares
        eliminated_score = total_score[eliminated_idx]

        for i, surv_idx in enumerate(survivor_indices):
            diff = total_score[surv_idx] - eliminated_score
            constraint = pm.math.log(pm.math.sigmoid(diff * CONSTRAINT_SCALE))
            pm.Potential(f'elim_constraint_{i}', constraint)

    return model


def fit_model(model):
    """Fit model and return trace."""
    try:
        with model:
            trace = pm.sample(
                draws=N_SAMPLES, tune=N_TUNE, chains=N_CHAINS,
                random_seed=RANDOM_SEED, progressbar=True,
                return_inferencedata=True
            )
        return trace
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def run_s1s2_comparison():
    """Compare both methods for S1-S2."""
    print("Rank-Based vs Percentage-Based Comparison for Seasons 1-2")

    # Load original data and enhanced results
    df = load_data(r"d:\2026mcmC\DataProcessed\DWTS_Processed_Long.csv")
    df = df[df['season'].isin([1, 2])]

    enhanced_df = pd.read_csv(r"d:\2026mcmC\Q1\fan_vote_s1s2_enhanced.csv")

    results = []

    for season in [1, 2]:
        season_data = df[df['season'] == season]
        weeks = sorted(season_data['week'].unique())

        print(f"\nSeason {season}: {len(weeks)} weeks")

        for week in weeks:
            week_data = season_data[season_data['week'] == week].copy()

            if len(week_data) < 2:
                continue

            # Find eliminated contestant
            eliminated = week_data[week_data['eliminated_week'] == week]
            if len(eliminated) != 1:
                continue

            eliminated_name = eliminated['celebrity_name'].values[0]
            contestants = week_data['celebrity_name'].values
            eliminated_idx = np.where(contestants == eliminated_name)[0]
            if len(eliminated_idx) == 0:
                continue
            eliminated_idx = int(eliminated_idx[0])

            # Get RANK-based fan share from enhanced results
            enhanced_week = enhanced_df[
                (enhanced_df['season'] == season) &
                (enhanced_df['week'] == week) &
                (enhanced_df['eliminated'] == True)
            ]
            if len(enhanced_week) == 0:
                continue
            rank_fan_share = enhanced_week['fan_share_mean'].values[0]

            # Fit PERCENTAGE-based model
            print(f"  S{season}W{week}: Fitting percentage model for {eliminated_name[:15]}...")
            model_pct = build_model_percentage(week_data, eliminated_idx)
            trace_pct = fit_model(model_pct)

            if trace_pct is None:
                continue

            # Extract eliminated fan share
            pct_samples = trace_pct.posterior['fan_shares'].values.reshape(-1, len(contestants))
            pct_fan_share = np.mean(pct_samples[:, eliminated_idx])

            diff = rank_fan_share - pct_fan_share

            results.append({
                'season': season,
                'week': week,
                'eliminated': eliminated_name,
                'rank_fan_share': rank_fan_share,
                'pct_fan_share': pct_fan_share,
                'diff_rank_minus_pct': diff,
                'more_fan_favorable': 'rank' if diff > 0 else 'percentage'
            })

            print(f"    Rank={rank_fan_share:.4f}, Pct={pct_fan_share:.4f}, Diff={diff:+.4f}")

    return pd.DataFrame(results)


def analyze_and_save(results_df: pd.DataFrame):
    """Analyze results and append to method_comparison files."""
    print("Season 1-2 Comparison Summary")

    # Overall stats
    rank_wins = (results_df['more_fan_favorable'] == 'rank').sum()
    total = len(results_df)
    avg_diff = results_df['diff_rank_minus_pct'].mean()

    print(f"\nOverall: Rank wins {rank_wins}/{total} ({rank_wins/total*100:.1f}%)")
    print(f"Average difference (Rank - Pct): {avg_diff:+.4f}")

    # By season summary
    season_summary = results_df.groupby('season').agg({
        'rank_fan_share': 'mean',
        'pct_fan_share': 'mean',
        'diff_rank_minus_pct': 'mean',
        'more_fan_favorable': lambda x: (x == 'rank').sum()
    }).rename(columns={'more_fan_favorable': 'rank_wins'})

    season_summary['n_weeks'] = results_df.groupby('season').size()
    season_summary['rank_win_pct'] = season_summary['rank_wins'] / season_summary['n_weeks'] * 100

    print("\n--- By Season ---")
    print(season_summary.to_string())

    # Save detailed results
    results_df.to_csv(r"d:\2026mcmC\Q2\method_comparison_s1s2.csv", index=False)
    print(f"\nDetailed results saved to: method_comparison_s1s2.csv")

    # Append to existing method_comparison_by_season.csv
    existing = pd.read_csv(r"d:\2026mcmC\Q2\method_comparison_by_season.csv")

    # Remove S1, S2 if already present
    existing = existing[~existing['season'].isin([1, 2])]

    # Add new S1, S2 rows
    new_rows = season_summary.reset_index()
    new_rows.columns = ['season', 'rank_fan_share', 'pct_fan_share',
                        'diff_rank_minus_pct', 'rank_wins', 'n_weeks', 'rank_win_pct']

    combined = pd.concat([new_rows, existing], ignore_index=True)
    combined = combined.sort_values('season').reset_index(drop=True)

    combined.to_csv(r"d:\2026mcmC\Q2\method_comparison_by_season.csv", index=False)
    print(f"Updated: method_comparison_by_season.csv (added S1-S2)")

    return season_summary


if __name__ == "__main__":
    print(f"MCMC: Draws={N_SAMPLES}, Tune={N_TUNE}, Chains={N_CHAINS}")

    results = run_s1s2_comparison()

    if len(results) > 0:
        season_summary = analyze_and_save(results)
