"""
比较排名制和百分比制评分方法

本脚本将这两种方法应用于所有赛季，并比较：
1. 哪种方法能让被淘汰的选手获得更高的粉丝投票份额？
2. 哪种方法更“受粉丝欢迎”（被淘汰选手的粉丝份额越高，粉丝的影响力越大）？

关键信息：如果被淘汰的选手在方法 X 下拥有更高的粉丝份额，
则意味着方法 X 更重视粉丝投票（即使粉丝投票，也无法挽救他们）。
Compare Rank-Based vs Percentage-Based Scoring Methods

This script applies BOTH methods to ALL seasons and compares:
1. Which method gives higher fan vote share to eliminated contestants
2. Which method is more "fan-favorable" (higher eliminated fan share = more fan influence)

Key insight: If eliminated contestant has HIGHER fan share under method X,
it means method X gives MORE weight to fan votes (fans couldn't save them despite voting).
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import pytensor.tensor as pt
from scipy.stats import rankdata
import warnings
warnings.filterwarnings('ignore')

# MCMC parameters (lighter for comparison speed)
N_SAMPLES = 1500
N_TUNE = 800
N_CHAINS = 2
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


def compute_judge_points_rank(judge_scores: np.ndarray) -> np.ndarray:
    """Rank-based: highest score gets N points."""
    n = len(judge_scores)
    ranks = rankdata(-judge_scores, method='average')
    return n - ranks + 1


def build_model_rank(week_data: pd.DataFrame, eliminated_idx: int):
    """Build model using RANK-based scoring."""
    judge_scores = week_data['total_judge_score'].values.astype(float)
    n = len(judge_scores)

    judge_points = compute_judge_points_rank(judge_scores)
    judge_normalized = judge_points / judge_points.sum()

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


def build_model_percentage(week_data: pd.DataFrame, eliminated_idx: int):
    """Build model using PERCENTAGE-based scoring."""
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
    """Fit model and return fan share samples."""
    try:
        with model:
            trace = pm.sample(
                draws=N_SAMPLES, tune=N_TUNE, chains=N_CHAINS,
                random_seed=RANDOM_SEED, progressbar=False,
                return_inferencedata=True
            )
        return trace
    except:
        return None


def run_comparison(data_path: str, seasons_to_compare: list = None):
    """Compare both methods across specified seasons."""
    print("Rank-Based vs Percentage-Based Scoring Comparison")

    df = load_data(data_path)

    if seasons_to_compare is None:
        seasons_to_compare = sorted(df['season'].unique())

    results = []

    for season in seasons_to_compare:
        season_data = df[df['season'] == season]
        weeks = sorted(season_data['week'].unique())

        season_rank_shares = []
        season_pct_shares = []

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

            # Fit RANK-based model
            model_rank = build_model_rank(week_data, eliminated_idx)
            trace_rank = fit_model(model_rank)

            # Fit PERCENTAGE-based model
            model_pct = build_model_percentage(week_data, eliminated_idx)
            trace_pct = fit_model(model_pct)

            if trace_rank is None or trace_pct is None:
                continue

            # Extract eliminated fan shares
            rank_samples = trace_rank.posterior['fan_shares'].values.reshape(-1, len(contestants))
            pct_samples = trace_pct.posterior['fan_shares'].values.reshape(-1, len(contestants))

            rank_elim_share = np.mean(rank_samples[:, eliminated_idx])
            pct_elim_share = np.mean(pct_samples[:, eliminated_idx])

            season_rank_shares.append(rank_elim_share)
            season_pct_shares.append(pct_elim_share)

            results.append({
                'season': season,
                'week': week,
                'eliminated': eliminated_name,
                'rank_fan_share': rank_elim_share,
                'pct_fan_share': pct_elim_share,
                'diff_rank_minus_pct': rank_elim_share - pct_elim_share,
                'more_fan_favorable': 'rank' if rank_elim_share > pct_elim_share else 'percentage'
            })

            print(f"  W{week}: Rank={rank_elim_share:.4f}, Pct={pct_elim_share:.4f}, "
                  f"Diff={rank_elim_share - pct_elim_share:+.4f}")

        if season_rank_shares:
            avg_rank = np.mean(season_rank_shares)
            avg_pct = np.mean(season_pct_shares)
            print(f"  Season {season} avg: Rank={avg_rank:.4f}, Pct={avg_pct:.4f}, "
                  f"Diff={avg_rank - avg_pct:+.4f}")

    return pd.DataFrame(results)


def analyze_results(results_df: pd.DataFrame):
    """Analyze and summarize comparison results."""
    print("Analysis Summary")

    # Overall comparison
    rank_wins = (results_df['more_fan_favorable'] == 'rank').sum()
    pct_wins = (results_df['more_fan_favorable'] == 'percentage').sum()
    total = len(results_df)

    print(f"\nOverall: Rank method more fan-favorable in {rank_wins}/{total} weeks ({rank_wins/total*100:.1f}%)")
    print(f"         Percentage method more fan-favorable in {pct_wins}/{total} weeks ({pct_wins/total*100:.1f}%)")

    avg_diff = results_df['diff_rank_minus_pct'].mean()
    print(f"\nAverage difference (Rank - Percentage): {avg_diff:+.4f}")

    if avg_diff > 0:
        print("=> RANK method gives higher fan share to eliminated contestants")
        print("=> RANK method is MORE FAN-FAVORABLE (fans have more influence)")
    else:
        print("=> PERCENTAGE method gives higher fan share to eliminated contestants")
        print("=> PERCENTAGE method is MORE FAN-FAVORABLE (fans have more influence)")

    # By season
    print("\n--- By Season ---")
    season_summary = results_df.groupby('season').agg({
        'rank_fan_share': 'mean',
        'pct_fan_share': 'mean',
        'diff_rank_minus_pct': 'mean',
        'more_fan_favorable': lambda x: (x == 'rank').sum()
    }).rename(columns={'more_fan_favorable': 'rank_wins'})

    season_summary['n_weeks'] = results_df.groupby('season').size()
    season_summary['rank_win_pct'] = season_summary['rank_wins'] / season_summary['n_weeks'] * 100

    print(season_summary.to_string())

    return season_summary


if __name__ == "__main__":
    import os

    # 绝对路径配置
    DATA_DIR = r"d:\2026mcmC\DataProcessed"
    Q2_DIR = r"d:\2026mcmC\Q2"

    DATA_PATH = os.path.join(DATA_DIR, "DWTS_Processed_Long.csv")
    OUTPUT_PATH = os.path.join(Q2_DIR, "method_comparison.csv")

    # Compare a sample of seasons (adjust as needed)
    # Using seasons 3-10 and 20-27 as representative samples
    seasons = list(range(3, 11)) + list(range(20, 28))

    print(f"Comparing seasons: {seasons}")
    print(f"MCMC: Draws={N_SAMPLES}, Tune={N_TUNE}, Chains={N_CHAINS}")

    results = run_comparison(DATA_PATH, seasons)

    if len(results) > 0:
        results.to_csv(OUTPUT_PATH, index=False)
        print(f"\nResults saved to: {OUTPUT_PATH}")

        season_summary = analyze_results(results)
        season_summary.to_csv(os.path.join(Q2_DIR, "method_comparison_by_season.csv"))
        print("\nSeason summary saved to: method_comparison_by_season.csv")