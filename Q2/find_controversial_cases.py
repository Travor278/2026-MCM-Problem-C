"""
找出《与星共舞》历史上最具争议的案例：
那些评委评分低但最终排名高的选手。
然后模拟他们在不同规则下的生存概率。
Find the most controversial cases in DWTS history:
Contestants who had low judge scores but high final rankings.
Then simulate their survival probability under alternative rules.
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import warnings
warnings.filterwarnings('ignore')

# 绝对路径配置
DATA_DIR = r"d:\2026mcmC\DataProcessed"
Q1_DIR = r"d:\2026mcmC\Q1"
Q2_DIR = r"d:\2026mcmC\Q2"

N_SIMULATIONS = 10000
np.random.seed(42)


def load_data():
    """Load all necessary data."""
    original = pd.read_csv(os.path.join(DATA_DIR, "DWTS_Processed_Long.csv"), encoding='utf-8-sig')
    fan_s1s2 = pd.read_csv(os.path.join(Q1_DIR, "fan_vote_s1s2_enhanced.csv"))
    fan_s1s2['rule_type'] = 'rank_s1s2'
    fan_other = pd.read_csv(os.path.join(Q1_DIR, "fan_vote_estimates.csv"))
    fan_other = fan_other[~fan_other['season'].isin([1, 2])]
    fan_votes = pd.concat([fan_s1s2, fan_other], ignore_index=True)
    return original, fan_votes


def find_controversial_contestants(original: pd.DataFrame, top_n: int = 20):
    """
    Find contestants who had many weeks with low judge scores
    but still achieved high final rankings.

    Controversy score = (weeks_with_bottom_judge_score) / final_rank
    Higher score = more controversial
    """
    results = []

    for (season, name), group in original.groupby(['season', 'celebrity_name']):
        final_rank = group['final_rank'].iloc[0]
        total_weeks = len(group)

        if pd.isna(final_rank) or total_weeks < 2:
            continue

        # Count weeks where this contestant had lowest or 2nd lowest judge score
        bottom_weeks = 0
        bottom2_weeks = 0

        for week in group['week'].unique():
            week_all = original[(original['season'] == season) & (original['week'] == week)]
            if len(week_all) < 2:
                continue

            contestant_score = group[group['week'] == week]['total_judge_score'].values[0]
            all_scores = week_all['total_judge_score'].values

            # Rank (1 = lowest)
            rank_in_week = rankdata(all_scores, method='min')[
                np.where(week_all['celebrity_name'].values == name)[0][0]
            ]

            if rank_in_week == 1:
                bottom_weeks += 1
            if rank_in_week <= 2:
                bottom2_weeks += 1

        # Controversy score: many bottom weeks + high placement (low final_rank number)
        if final_rank <= 5:  # Focus on top 5 finishers
            controversy_score = bottom_weeks / final_rank + bottom2_weeks / (final_rank * 2)
        else:
            controversy_score = bottom_weeks / (final_rank * 2)

        results.append({
            'season': season,
            'celebrity_name': name,
            'final_rank': int(final_rank),
            'total_weeks': total_weeks,
            'bottom_weeks': bottom_weeks,
            'bottom2_weeks': bottom2_weeks,
            'controversy_score': controversy_score
        })

    df = pd.DataFrame(results)
    df = df.sort_values('controversy_score', ascending=False)

    return df.head(top_n)


def compute_judge_component_rank(scores: np.ndarray) -> np.ndarray:
    """Rank-based judge component."""
    n = len(scores)
    ranks = rankdata(-scores, method='average')
    points = n - ranks + 1
    return points / points.sum()


def compute_judge_component_percentage(scores: np.ndarray) -> np.ndarray:
    """Percentage-based judge component."""
    return scores / scores.sum()


def simulate_survival_probability(week_data: pd.DataFrame,
                                   fan_data: pd.DataFrame,
                                   target_name: str,
                                   rule: str,
                                   n_sims: int = N_SIMULATIONS) -> float:
    """
    Simulate survival probability for a target contestant under a given rule.
    Returns probability of NOT being eliminated.
    """
    contestants = week_data['celebrity_name'].values
    n = len(contestants)

    if target_name not in contestants:
        return np.nan

    target_idx = np.where(contestants == target_name)[0][0]

    # Get judge scores
    judge_scores = week_data['total_judge_score'].values.astype(float)

    # Get fan share estimates
    fan_means = []
    fan_stds = []
    for name in contestants:
        fan_row = fan_data[fan_data['celebrity_name'] == name]
        if len(fan_row) > 0:
            fan_means.append(fan_row['fan_share_mean'].values[0])
            fan_stds.append(fan_row['fan_share_std'].values[0])
        else:
            fan_means.append(1.0 / n)  # Uniform prior
            fan_stds.append(0.1 / n)

    fan_means = np.array(fan_means)
    fan_stds = np.array(fan_stds)

    # Compute judge component based on rule
    if rule == 'rank':
        judge_comp = compute_judge_component_rank(judge_scores)
    else:
        judge_comp = compute_judge_component_percentage(judge_scores)

    # Monte Carlo simulation
    samples = np.random.normal(
        loc=fan_means[np.newaxis, :],
        scale=fan_stds[np.newaxis, :],
        size=(n_sims, n)
    )
    samples = np.clip(samples, 0.001, 0.999)
    samples = samples / samples.sum(axis=1, keepdims=True)

    # Total scores
    total_scores = 0.5 * judge_comp[np.newaxis, :] + 0.5 * samples

    # Who gets eliminated (lowest score)?
    eliminated_idx = np.argmin(total_scores, axis=1)

    # Survival probability
    survival_prob = 1.0 - (eliminated_idx == target_idx).mean()

    return survival_prob


def analyze_controversial_contestant(original: pd.DataFrame,
                                      fan_votes: pd.DataFrame,
                                      season: int,
                                      name: str) -> pd.DataFrame:
    """
    Analyze a single controversial contestant across all their weeks.
    Returns survival probabilities under both rules for each week.
    """
    season_data = original[original['season'] == season]
    season_fan = fan_votes[fan_votes['season'] == season]

    contestant_data = season_data[season_data['celebrity_name'] == name]
    weeks = sorted(contestant_data['week'].unique())

    results = []

    for week in weeks:
        week_all = season_data[season_data['week'] == week]
        week_fan = season_fan[season_fan['week'] == week]

        if len(week_all) < 2 or len(week_fan) < 2:
            continue

        # Get judge rank this week
        judge_scores = week_all['total_judge_score'].values
        contestant_score = contestant_data[contestant_data['week'] == week]['total_judge_score'].values[0]
        judge_rank = int(rankdata(-judge_scores)[
            np.where(week_all['celebrity_name'].values == name)[0][0]
        ])
        n_contestants = len(week_all)

        # Normalized score (0 = worst, 1 = best)
        min_score = judge_scores.min()
        max_score = judge_scores.max()
        if max_score > min_score:
            normalized_score = (contestant_score - min_score) / (max_score - min_score)
        else:
            normalized_score = 0.5

        # Simulate under both rules
        surv_rank = simulate_survival_probability(week_all, week_fan, name, 'rank')
        surv_pct = simulate_survival_probability(week_all, week_fan, name, 'percentage')

        results.append({
            'season': season,
            'week': week,
            'celebrity_name': name,
            'judge_rank': judge_rank,
            'n_contestants': n_contestants,
            'normalized_score': normalized_score,
            'survival_prob_rank': surv_rank,
            'survival_prob_pct': surv_pct
        })

    return pd.DataFrame(results)


def get_actual_rule(season: int) -> str:
    """Get the actual rule used in a season."""
    if season in [1, 2]:
        return 'rank'
    elif season >= 28:
        return 'bottom2'
    else:
        return 'percentage'


def create_heatmap_data(all_analyses: dict, original: pd.DataFrame) -> pd.DataFrame:
    """Create data for the survival probability heatmap."""
    rows = []

    for key, df in all_analyses.items():
        season, name = key
        actual_rule = get_actual_rule(season)

        # Get final rank
        final_rank = original[
            (original['season'] == season) &
            (original['celebrity_name'] == name)
        ]['final_rank'].iloc[0]

        for _, row in df.iterrows():
            # Use counterfactual rule (opposite of actual)
            if actual_rule == 'rank':
                cf_survival = row['survival_prob_pct']
            elif actual_rule == 'bottom2':
                cf_survival = row['survival_prob_rank']  # Use rank as counterfactual
            else:
                cf_survival = row['survival_prob_rank']

            rows.append({
                'season': season,
                'week': row['week'],
                'celebrity_name': name,
                'final_rank': final_rank,
                'actual_rule': actual_rule,
                'normalized_judge_score': row['normalized_score'],
                'counterfactual_survival': cf_survival,
                'label': f"S{season}: {name}"
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Loading data...")
    original, fan_votes = load_data()

    print("\nFinding controversial contestants...")
    controversial = find_controversial_contestants(original, top_n=30)

    print("\nTop 30 Controversial Contestants:")
    print(controversial[['season', 'celebrity_name', 'final_rank',
                         'bottom_weeks', 'bottom2_weeks', 'controversy_score']].to_string())

    # Select final 15 cases (with constraints on S28+)
    # Manual selection based on controversy + historical significance
    selected_cases = [
        (27, "Bobby Bones"),      # S27 winner with consistently low scores
        (11, "Bristol Palin"),    # S11 3rd place, many bottom scores
        (2, "Jerry Rice"),        # S2 runner-up, 5 weeks lowest
        (4, "Billy Ray Cyrus"),   # S4 5th place, 6 weeks lowest
        (19, "Michael Waltrip"),  # Strong fan following despite low scores
        (12, "Romeo"),            # Made it far with low scores
        (21, "Alexa PenaVega"),   # Controversy case
        (30, "Olivia Jade"),      # Recent controversial case
        (28, "Sailor Brinkley-Cook"),  # Bottom2 era example
        (2, "Tia Carrere"),       # S2 early exit despite potential
        (22, "Kim Fields"),       # Made it far with low scores
        (31, "Joseph Baena"),     # Recent example
        (19, "Randy Couture"),    # Low scores but had support
        (14, "Jack Wagner"),      # Controversial run
        (10, "Niecy Nash"),       # Strong fan favorite
        (23, "Amber Rose"),       # Controversial presence
        (12, "Wendy Williams"),   # Made it with low scores
        (2, "Master P"),          # Famous low scorer
    ]

    # Filter to 15, ensuring good distribution
    # Prioritize user's examples + interesting cases
    final_15 = [
        (27, "Bobby Bones"),
        (11, "Bristol Palin"),
        (2, "Jerry Rice"),
        (4, "Billy Ray Cyrus"),
        (28, "Sailor Brinkley-Cook"),
        (30, "Olivia Jade"),
        (21, "Alexa PenaVega"),
        (12, "Romeo"),
        (19, "Michael Waltrip"),
        (23, "Amber Rose"),
        (10, "Niecy Nash"),
        (14, "Jack Wagner"),
        (12, "Wendy Williams"),
        (31, "Joseph Baena"),
        (2, "Master P"),
    ]

    # Analyze each
    print("Analyzing controversial contestants...")

    all_analyses = {}

    for season, name in final_15:
        print(f"\n  S{season}: {name}...")
        try:
            analysis = analyze_controversial_contestant(original, fan_votes, season, name)
            if len(analysis) > 0:
                all_analyses[(season, name)] = analysis
                print(f"    Analyzed {len(analysis)} weeks")
        except Exception as e:
            print(f"    Error: {e}")

    # Create heatmap data
    heatmap_df = create_heatmap_data(all_analyses, original)
    heatmap_df.to_csv(os.path.join(Q2_DIR, "controversial_heatmap_data.csv"), index=False)
    print(f"\nSaved heatmap data: {len(heatmap_df)} rows")

    # Print summary for each contestant
    print("CONTROVERSIAL CONTESTANTS SUMMARY")

    for (season, name), df in all_analyses.items():
        actual_rule = get_actual_rule(season)
        final_rank = original[
            (original['season'] == season) &
            (original['celebrity_name'] == name)
        ]['final_rank'].iloc[0]

        # Average counterfactual survival
        if actual_rule == 'rank':
            avg_cf_surv = df['survival_prob_pct'].mean()
            cf_rule = 'percentage'
        else:
            avg_cf_surv = df['survival_prob_rank'].mean()
            cf_rule = 'rank'

        # Weeks where they would have been eliminated under counterfactual
        elim_weeks = (df['survival_prob_rank' if actual_rule != 'rank' else 'survival_prob_pct'] < 0.5).sum()

        print(f"\nS{season} {name} (Final: #{int(final_rank)}, Rule: {actual_rule})")
        print(f"  Avg survival under {cf_rule}: {avg_cf_surv*100:.1f}%")
        print(f"  Weeks likely eliminated under {cf_rule}: {elim_weeks}/{len(df)}")
