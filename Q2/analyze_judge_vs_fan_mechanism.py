"""
分析15名争议选手在两种极端机制下的存活概率：
1. 纯评委打分机制 (Judge-Only): 100%评委分数，0%粉丝投票
2. 纯粉丝投票机制 (Fan-Only): 0%评委分数，100%粉丝投票

若两者结果差异显著，则说明结合机制的合理性。

Analyze 15 controversial contestants under two extreme mechanisms:
1. Judge-Only: 100% judge scores, 0% fan votes
2. Fan-Only: 0% judge scores, 100% fan votes

If results differ significantly, it demonstrates the rationality of the combined mechanism.
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle, Patch
import warnings
import os
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


def simulate_survival_judge_only(week_data: pd.DataFrame,
                                  target_name: str,
                                  n_sims: int = N_SIMULATIONS) -> float:
    """
    纯评委打分机制：只看评委分数，分数最低者淘汰。
    Judge-Only: Eliminate the contestant with lowest judge score.
    """
    contestants = week_data['celebrity_name'].values
    n = len(contestants)

    if target_name not in contestants:
        return np.nan

    target_idx = np.where(contestants == target_name)[0][0]
    judge_scores = week_data['total_judge_score'].values.astype(float)

    # 最低分者淘汰（确定性，无需模拟）
    min_score = judge_scores.min()
    min_count = (judge_scores == min_score).sum()

    if judge_scores[target_idx] == min_score:
        # 如果有并列最低，平均分配淘汰概率
        survival_prob = 1.0 - (1.0 / min_count)
    else:
        survival_prob = 1.0

    return survival_prob


def simulate_survival_fan_only(week_data: pd.DataFrame,
                                fan_data: pd.DataFrame,
                                target_name: str,
                                n_sims: int = N_SIMULATIONS) -> float:
    """
    纯粉丝投票机制：只看粉丝投票份额，份额最低者淘汰。
    Fan-Only: Eliminate the contestant with lowest fan vote share.
    Uses Monte Carlo simulation due to uncertainty in fan vote estimates.
    """
    contestants = week_data['celebrity_name'].values
    n = len(contestants)

    if target_name not in contestants:
        return np.nan

    target_idx = np.where(contestants == target_name)[0][0]

    # Get fan share estimates
    fan_means = []
    fan_stds = []
    for name in contestants:
        fan_row = fan_data[fan_data['celebrity_name'] == name]
        if len(fan_row) > 0:
            fan_means.append(fan_row['fan_share_mean'].values[0])
            fan_stds.append(fan_row['fan_share_std'].values[0])
        else:
            fan_means.append(1.0 / n)
            fan_stds.append(0.1 / n)

    fan_means = np.array(fan_means)
    fan_stds = np.array(fan_stds)

    # Monte Carlo simulation
    samples = np.random.normal(
        loc=fan_means[np.newaxis, :],
        scale=fan_stds[np.newaxis, :],
        size=(n_sims, n)
    )
    samples = np.clip(samples, 0.001, 0.999)
    samples = samples / samples.sum(axis=1, keepdims=True)

    # 最低粉丝份额者淘汰
    eliminated_idx = np.argmin(samples, axis=1)
    survival_prob = 1.0 - (eliminated_idx == target_idx).mean()

    return survival_prob


def analyze_contestant_both_mechanisms(original: pd.DataFrame,
                                        fan_votes: pd.DataFrame,
                                        season: int,
                                        name: str) -> dict:
    """
    Analyze a contestant's average survival probability under both mechanisms.
    """
    season_data = original[original['season'] == season]
    season_fan = fan_votes[fan_votes['season'] == season]

    contestant_data = season_data[season_data['celebrity_name'] == name]
    weeks = sorted(contestant_data['week'].unique())

    judge_survivals = []
    fan_survivals = []

    for week in weeks:
        week_all = season_data[season_data['week'] == week]
        week_fan = season_fan[season_fan['week'] == week]

        if len(week_all) < 2:
            continue

        # Judge-only survival
        surv_judge = simulate_survival_judge_only(week_all, name)

        # Fan-only survival
        if len(week_fan) >= 2:
            surv_fan = simulate_survival_fan_only(week_all, week_fan, name)
        else:
            surv_fan = np.nan

        if not np.isnan(surv_judge):
            judge_survivals.append(surv_judge)
        if not np.isnan(surv_fan):
            fan_survivals.append(surv_fan)

    # Get final rank
    final_rank = contestant_data['final_rank'].iloc[0] if len(contestant_data) > 0 else np.nan

    return {
        'season': season,
        'name': name,
        'final_rank': int(final_rank) if not pd.isna(final_rank) else None,
        'avg_survival_judge_only': np.mean(judge_survivals) if judge_survivals else np.nan,
        'avg_survival_fan_only': np.mean(fan_survivals) if fan_survivals else np.nan,
        'weeks_analyzed': len(judge_survivals),
        'judge_survivals': judge_survivals,
        'fan_survivals': fan_survivals
    }


def create_comparison_visualization(results: list):
    """
    创建可视化对比图：展示评委机制 vs 粉丝机制下的存活概率差异。
    """
    # Prepare data
    df = pd.DataFrame(results)
    df = df.dropna(subset=['avg_survival_judge_only', 'avg_survival_fan_only'])
    df['label'] = df.apply(lambda x: f"S{x['season']}: {x['name']}", axis=1)
    df['difference'] = df['avg_survival_fan_only'] - df['avg_survival_judge_only']

    # Sort by difference (fan advantage)
    df = df.sort_values('difference', ascending=True)
    y_positions = np.arange(len(df))

    # Calculate correlation
    corr = df['avg_survival_judge_only'].corr(df['avg_survival_fan_only'])

    # Combined Figure: Scatter (Correlation) + Diverging Bar Chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))

    # --- Left Panel: Scatter Plot (Correlation) ---
    # Plot diagonal line (equal survival)
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Equal Survival')

    # Fill regions
    ax1.fill_between([0, 1], [0, 1], [1, 1], alpha=0.15, color='#E94F37')
    ax1.fill_between([0, 1], [0, 0], [0, 1], alpha=0.15, color='#2E86AB')

    # Scatter plot
    scatter = ax1.scatter(df['avg_survival_judge_only'], df['avg_survival_fan_only'],
                          s=180, c=df['final_rank'], cmap='viridis_r',
                          edgecolors='white', linewidths=2, zorder=3)

    # Add labels for each point (with custom offsets for overlapping labels)
    for _, row in df.iterrows():
        season = row['season']
        # Custom offsets for specific seasons to avoid overlap
        if season in [12, 14]:  # S12, S14 - move to left
            offset = (-12, 0)
            ha = 'right'
        elif season == 28:  # S28 - move down
            offset = (6, -15)
            ha = 'left'
        elif season == 31:  # S31 - move up
            offset = (6, 10)
            ha = 'left'
        elif season == 21:  # S21 - move down-left
            offset = (-5, -15)
            ha = 'right'
        else:
            offset = (6, 6)
            ha = 'left'
        ax1.annotate(f"S{season}",
                     (row['avg_survival_judge_only'], row['avg_survival_fan_only']),
                     xytext=offset, textcoords='offset points', fontsize=9, fontweight='bold', ha=ha)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax1, shrink=0.7, pad=0.02)
    cbar.set_label('Final Rank (1=Winner)', fontsize=10)

    # Correlation annotation
    ax1.text(0.05, 0.95, f'Correlation: r = {corr:.3f}',
             transform=ax1.transAxes, fontsize=12, fontweight='bold',
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Region labels
    ax1.text(0.25, 0.85, 'Fan-Only\nBetter', transform=ax1.transAxes,
             fontsize=11, color='#E94F37', fontweight='bold', ha='center')
    ax1.text(0.75, 0.15, 'Judge-Only\nBetter', transform=ax1.transAxes,
             fontsize=11, color='#2E86AB', fontweight='bold', ha='center')

    ax1.set_xlabel('Survival Probability (Judge-Only)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Survival Probability (Fan-Only)', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Correlation Between Mechanisms', fontsize=13, fontweight='bold')
    ax1.set_xlim(-0.02, 1.05)
    ax1.set_ylim(-0.02, 1.05)
    ax1.set_aspect('equal')
    ax1.grid(alpha=0.3)

    # --- Right Panel: Diverging Bar Chart ---
    colors = ['#E94F37' if d > 0 else '#2E86AB' for d in df['difference']]
    bars = ax2.barh(y_positions, df['difference'], color=colors, height=0.7, edgecolor='white', linewidth=0.5)

    # Add value labels
    for i, (bar, diff) in enumerate(zip(bars, df['difference'])):
        if diff > 0:
            ax2.text(diff + 0.02, i, f'+{diff*100:.0f}%', va='center', fontsize=9, fontweight='bold')
        else:
            ax2.text(diff - 0.02, i, f'{diff*100:.0f}%', va='center', ha='right', fontsize=9, fontweight='bold')

    ax2.set_yticks(y_positions)
    ax2.set_yticklabels(df['label'], fontsize=10)
    ax2.axvline(x=0, color='black', linewidth=1.5)
    ax2.set_xlabel('Survival Probability Difference\n(Fan-Only − Judge-Only)', fontsize=12, fontweight='bold')
    ax2.set_title('(b) Individual Contestant Differences', fontsize=13, fontweight='bold', pad=20)

    # Add legend annotations (below title, not overlapping)
    ax2.text(-0.35, len(df)-0.3, '← Judge-Only favors', fontsize=10, color='#2E86AB', fontweight='bold')
    ax2.text(0.55, len(df)-0.3, 'Fan-Only favors →', fontsize=10, color='#E94F37', fontweight='bold')

    ax2.grid(axis='x', alpha=0.3)
    ax2.set_xlim(-0.4, 1.1)

    # Main title
    fig.suptitle('Judge-Only vs Fan-Only Mechanism: Survival Probability Comparison\n(15 Most Controversial Contestants)',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(Q2_DIR, "judge_vs_fan_combined.png"), dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: judge_vs_fan_combined.png")

    # Also save individual figures for flexibility
    # Figure: Dumbbell Chart
    fig_dumb, ax_dumb = plt.subplots(figsize=(12, 10))
    for i, (_, row) in enumerate(df.iterrows()):
        ax_dumb.plot([row['avg_survival_judge_only'], row['avg_survival_fan_only']],
                [i, i], color='#888888', linewidth=1.5, zorder=1)
    ax_dumb.scatter(df['avg_survival_judge_only'], y_positions,
               s=120, c='#2E86AB', label='Judge-Only', zorder=2, edgecolors='white', linewidths=1)
    ax_dumb.scatter(df['avg_survival_fan_only'], y_positions,
               s=120, c='#E94F37', label='Fan-Only', zorder=2, edgecolors='white', linewidths=1)
    ax_dumb.set_yticks(y_positions)
    ax_dumb.set_yticklabels(df['label'], fontsize=10)
    ax_dumb.set_xlabel('Average Survival Probability', fontsize=12, fontweight='bold')
    ax_dumb.set_ylabel('Controversial Contestants', fontsize=12, fontweight='bold')
    ax_dumb.set_title('Survival Probability: Judge-Only vs Fan-Only Mechanism',
                 fontsize=14, fontweight='bold')
    ax_dumb.axvline(x=0.5, color='#CCCCCC', linestyle='--', linewidth=1, alpha=0.7)
    ax_dumb.legend(loc='lower right', fontsize=11)
    ax_dumb.grid(axis='x', alpha=0.3)
    ax_dumb.set_xlim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(Q2_DIR, "judge_vs_fan_dumbbell.png"), dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: judge_vs_fan_dumbbell.png")

    return df


if __name__ == "__main__":
    print("Loading data...")
    original, fan_votes = load_data()

    # 15 controversial contestants (same as before)
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

    print("Analyzing 15 Controversial Contestants under Judge-Only vs Fan-Only")

    results = []

    for season, name in final_15:
        print(f"\n  Analyzing S{season}: {name}...")
        try:
            result = analyze_contestant_both_mechanisms(original, fan_votes, season, name)
            results.append(result)

            print(f"    Final Rank: #{result['final_rank']}")
            print(f"    Judge-Only Survival: {result['avg_survival_judge_only']*100:.1f}%")
            print(f"    Fan-Only Survival: {result['avg_survival_fan_only']*100:.1f}%")
            diff = result['avg_survival_fan_only'] - result['avg_survival_judge_only']
            print(f"    Difference (Fan - Judge): {diff*100:+.1f}%")
        except Exception as e:
            print(f"    Error: {e}")

    # Create visualizations
    print("Creating visualizations...")

    df_results = create_comparison_visualization(results)

    # Save results to CSV
    df_results.to_csv(os.path.join(Q2_DIR, "judge_vs_fan_comparison.csv"), index=False)
    print("\nSaved: judge_vs_fan_comparison.csv")

    # Summary statistics
    print("SUMMARY")

    avg_judge = df_results['avg_survival_judge_only'].mean()
    avg_fan = df_results['avg_survival_fan_only'].mean()

    print(f"\nAverage survival probability across 15 contestants:")
    print(f"  Judge-Only: {avg_judge*100:.1f}%")
    print(f"  Fan-Only: {avg_fan*100:.1f}%")
    print(f"  Difference: {(avg_fan - avg_judge)*100:+.1f}%")

    # Count who benefits from each mechanism
    fan_better = (df_results['difference'] > 0).sum()
    judge_better = (df_results['difference'] < 0).sum()

    print(f"\nContestants who benefit more from:")
    print(f"  Fan-Only mechanism: {fan_better}/15")
    print(f"  Judge-Only mechanism: {judge_better}/15")

    # Correlation analysis
    corr = df_results['avg_survival_judge_only'].corr(df_results['avg_survival_fan_only'])
    print(f"\nCorrelation between Judge-Only and Fan-Only survival: {corr:.3f}")

    if corr < 0.5:
        print("  → Low correlation indicates the two mechanisms produce DIFFERENT outcomes")
        print("  → This supports the use of a COMBINED mechanism")
    else:
        print("  → High correlation suggests mechanisms often agree")
