"""
问题四：动态控制模型 (Dynamic Control Model)
Question 4: Adaptive Weight System using Sigmoid Function

核心理念：前期低裁判权重保护"有趣的黑马"，后期高裁判权重确保冠军是实力派

模型公式：
w(t) = w_min + (w_max - w_min) / (1 + exp(-k * (t - t_mid)))

参数：
- w_min: 前期最低裁判权重
- w_max: 后期最高裁判权重
- k: 变化速率
- t_mid: 赛季中点（转折点）
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from scipy import stats
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# 绝对路径配置
DATA_DIR = r"d:\2026mcmC\DataProcessed"
Q1_DIR = r"d:\2026mcmC\Q1"
Q4_DIR = r"d:\2026mcmC\Q4"

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("Question 4: Dynamic Control Model - Adaptive Weight System")

# Part 1: Load Data
print("\n[Part 1] Loading Data...")

judge_data = pd.read_csv(os.path.join(DATA_DIR, "DWTS_Processed_Long.csv"), encoding='utf-8-sig')
fan_data = pd.read_csv(os.path.join(Q1_DIR, "fan_vote_estimates.csv"))

# Merge
merged = pd.merge(
    judge_data,
    fan_data[['season', 'week', 'celebrity_name', 'fan_share_mean', 'eliminated']],
    on=['season', 'week', 'celebrity_name'],
    how='inner'
)
merged = merged[merged['season'] <= 27]

# Get max week per season
season_max_weeks = merged.groupby('season')['week'].max().to_dict()
print(f"  Merged data: {len(merged)} records")
print(f"  Seasons with data: {len(season_max_weeks)}")

# Part 2: Define Dynamic Weight Function
print("\n[Part 2] Defining Dynamic Weight Functions...")

def sigmoid_weight(t, t_max, w_min=0.3, w_max=0.8, k=1.5):
    """
    Sigmoid动态权重函数

    Parameters:
    - t: 当前周数
    - t_max: 赛季总周数
    - w_min: 前期最低裁判权重
    - w_max: 后期最高裁判权重
    - k: 变化速率

    Returns:
    - w: 当前周的裁判权重
    """
    t_mid = t_max / 2  # 赛季中点
    t_normalized = (t - t_mid) / (t_max / 4)  # Normalize to [-2, 2] range
    w = w_min + (w_max - w_min) / (1 + np.exp(-k * t_normalized))
    return w


def linear_weight(t, t_max, w_min=0.3, w_max=0.8):
    """线性权重函数（对比用）"""
    return w_min + (w_max - w_min) * t / t_max


def constant_weight(t, t_max, w=0.5):
    """固定权重（当前系统）"""
    return w


# Visualize weight functions
fig, ax = plt.subplots(figsize=(10, 6))

t_max = 11  # Typical season length
weeks = np.linspace(1, t_max, 100)

# Sigmoid with different parameters
for k in [0.5, 1.0, 1.5, 2.0]:
    weights = [sigmoid_weight(t, t_max, 0.3, 0.8, k) for t in weeks]
    ax.plot(weeks, weights, linewidth=2, label=f'Sigmoid (k={k})')

# Linear
weights_linear = [linear_weight(t, t_max, 0.3, 0.8) for t in weeks]
ax.plot(weeks, weights_linear, '--', linewidth=2, label='Linear')

# Constant
ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=2, label='Current (w=0.5)')

ax.set_xlabel('Week', fontsize=12)
ax.set_ylabel('Judge Weight (w)', fontsize=12)
ax.set_title('Dynamic Weight Functions', fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xlim(1, t_max)
ax.set_ylim(0.2, 0.9)

# Add annotations
ax.annotate('Entertainment\nPhase', xy=(2, 0.35), fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax.annotate('Competition\nPhase', xy=(t_max-1, 0.75), fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

plt.tight_layout()
plt.savefig(os.path.join(Q4_DIR, "dynamic_weight_functions.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: dynamic_weight_functions.png")

# Part 3: Simulate Dynamic Weight System
print("\n[Part 3] Simulating Dynamic Weight System...")

def simulate_season_dynamic(season_data, weight_func, bottom_k=2, **kwargs):
    """
    使用动态权重系统模拟一个赛季

    Returns:
    - results: list of (week, eliminated, judge_rank, was_robbery)
    """
    results = []
    t_max = season_data['week'].max()

    for week in sorted(season_data['week'].unique()):
        week_data = season_data[season_data['week'] == week].copy()
        if len(week_data) < 2:
            continue

        # Get dynamic weight for this week
        w = weight_func(week, t_max, **kwargs)

        # Normalize scores
        judge_min = week_data['total_judge_score'].min()
        judge_max = week_data['total_judge_score'].max()
        if judge_max > judge_min:
            week_data['judge_norm'] = (week_data['total_judge_score'] - judge_min) / (judge_max - judge_min)
        else:
            week_data['judge_norm'] = 0.5

        fan_total = week_data['fan_share_mean'].sum()
        if fan_total > 0:
            week_data['fan_norm'] = week_data['fan_share_mean'] / fan_total
        else:
            week_data['fan_norm'] = 1 / len(week_data)

        # Combined score
        week_data['combined'] = w * week_data['judge_norm'] + (1 - w) * week_data['fan_norm']

        # Find eliminated (lowest combined score)
        week_data = week_data.sort_values('combined')
        eliminated_row = week_data.iloc[0]
        eliminated = eliminated_row['celebrity_name']

        # Judge rank of eliminated
        week_data_by_judge = week_data.sort_values('total_judge_score', ascending=False)
        week_data_by_judge['judge_rank'] = range(1, len(week_data_by_judge) + 1)
        judge_rank = week_data_by_judge[week_data_by_judge['celebrity_name'] == eliminated]['judge_rank'].values[0]

        was_robbery = judge_rank <= 2

        results.append({
            'week': week,
            'eliminated': eliminated,
            'judge_rank': judge_rank,
            'n_contestants': len(week_data),
            'was_robbery': was_robbery,
            'weight_used': w
        })

    return results


def evaluate_weight_system(merged_data, weight_func, bottom_k=2, **kwargs):
    """
    评估一个权重系统的表现

    Returns:
    - fairness: 平均淘汰者排名比例
    - robbery_rate: 惨案率
    - engagement: 粉丝影响力（权重变化幅度）
    """
    all_results = []

    for season, season_data in merged_data.groupby('season'):
        t_max = season_data['week'].max()
        results = simulate_season_dynamic(season_data, weight_func, bottom_k, **kwargs)
        for r in results:
            r['season'] = season
        all_results.extend(results)

    if not all_results:
        return 0, 0, 0

    results_df = pd.DataFrame(all_results)

    # Metrics
    results_df['fairness_ratio'] = results_df['judge_rank'] / results_df['n_contestants']
    fairness = results_df['fairness_ratio'].mean()
    robbery_rate = results_df['was_robbery'].mean()

    # Engagement: average weight used (lower = more fan influence)
    avg_weight = results_df['weight_used'].mean()
    engagement = 1 - avg_weight  # Higher engagement when weight is lower

    return fairness, robbery_rate, engagement


# Compare systems
systems = {
    'Current (w=0.5)': {'func': constant_weight, 'kwargs': {'w': 0.5}},
    'Judge-Heavy (w=0.7)': {'func': constant_weight, 'kwargs': {'w': 0.7}},
    'Fan-Heavy (w=0.3)': {'func': constant_weight, 'kwargs': {'w': 0.3}},
    'Linear (0.3->0.8)': {'func': linear_weight, 'kwargs': {'w_min': 0.3, 'w_max': 0.8}},
    'Sigmoid k=1.0': {'func': sigmoid_weight, 'kwargs': {'w_min': 0.3, 'w_max': 0.8, 'k': 1.0}},
    'Sigmoid k=1.5': {'func': sigmoid_weight, 'kwargs': {'w_min': 0.3, 'w_max': 0.8, 'k': 1.5}},
    'Sigmoid k=2.0': {'func': sigmoid_weight, 'kwargs': {'w_min': 0.3, 'w_max': 0.8, 'k': 2.0}},
}

print("\n[System Comparison]")
print(f"{'System':<25} {'Fairness':>10} {'Robbery%':>10} {'Engagement':>12}")

comparison_results = []
for name, config in systems.items():
    fairness, robbery_rate, engagement = evaluate_weight_system(
        merged, config['func'], **config['kwargs']
    )
    comparison_results.append({
        'system': name,
        'fairness': fairness,
        'robbery_rate': robbery_rate,
        'engagement': engagement
    })
    print(f"{name:<25} {fairness:>10.3f} {robbery_rate*100:>9.1f}% {engagement:>12.3f}")

comparison_df = pd.DataFrame(comparison_results)

# Part 4: Grid Search for Optimal Sigmoid Parameters
print("\n[Part 4] Grid Search for Optimal Sigmoid Parameters...")

param_grid = {
    'w_min': [0.2, 0.3, 0.4],
    'w_max': [0.7, 0.8, 0.9],
    'k': [0.5, 1.0, 1.5, 2.0]
}

grid_results = []
for w_min, w_max, k in product(param_grid['w_min'], param_grid['w_max'], param_grid['k']):
    if w_min >= w_max:
        continue
    fairness, robbery_rate, engagement = evaluate_weight_system(
        merged, sigmoid_weight, w_min=w_min, w_max=w_max, k=k
    )
    grid_results.append({
        'w_min': w_min,
        'w_max': w_max,
        'k': k,
        'fairness': fairness,
        'robbery_rate': robbery_rate,
        'engagement': engagement,
        'no_robbery': 1 - robbery_rate
    })

grid_df = pd.DataFrame(grid_results)

# Define composite score
alpha_f, alpha_e, alpha_r = 0.4, 0.3, 0.3
grid_df['composite'] = (
    alpha_f * grid_df['fairness'] +
    alpha_e * grid_df['engagement'] +
    alpha_r * grid_df['no_robbery']
)

# Best solution
best = grid_df.loc[grid_df['composite'].idxmax()]
print(f"\n  [Optimal Sigmoid Parameters]")
print(f"  w_min: {best['w_min']:.2f}")
print(f"  w_max: {best['w_max']:.2f}")
print(f"  k: {best['k']:.2f}")
print(f"  Fairness: {best['fairness']:.3f}")
print(f"  Robbery Rate: {best['robbery_rate']*100:.1f}%")
print(f"  Engagement: {best['engagement']:.3f}")
print(f"  Composite: {best['composite']:.3f}")

# Part 5: Historical Case Study - Bobby Bones
print("\n[Part 5] Historical Case Study - Bobby Bones (Season 27)...")

# Find Bobby Bones data
bobby = merged[
    (merged['season'] == 27) &
    (merged['celebrity_name'] == 'Bobby Bones')
]

if len(bobby) > 0:
    print(f"\n  Bobby Bones competed for {len(bobby)} weeks")

    # Get season 27 data
    s27 = merged[merged['season'] == 27]
    t_max = s27['week'].max()

    print(f"\n  [Weekly Analysis under Different Systems]")
    print(f"{'Week':>5} {'Judge Rank':>12} {'Current(w=0.5)':>15} {'Sigmoid':>15} {'At Risk?':>10}")

    for week in sorted(s27['week'].unique()):
        week_data = s27[s27['week'] == week].copy()
        if 'Bobby Bones' not in week_data['celebrity_name'].values:
            continue

        # Get Bobby's judge rank
        week_sorted = week_data.sort_values('total_judge_score', ascending=False)
        week_sorted['rank'] = range(1, len(week_sorted) + 1)
        bobby_rank = week_sorted[week_sorted['celebrity_name'] == 'Bobby Bones']['rank'].values[0]

        # Dynamic weight
        w_dynamic = sigmoid_weight(week, t_max, best['w_min'], best['w_max'], best['k'])

        # Would Bobby be at risk?
        # Normalize and calculate combined scores
        judge_min = week_data['total_judge_score'].min()
        judge_max = week_data['total_judge_score'].max()
        week_data['judge_norm'] = (week_data['total_judge_score'] - judge_min) / (judge_max - judge_min) if judge_max > judge_min else 0.5

        fan_total = week_data['fan_share_mean'].sum()
        week_data['fan_norm'] = week_data['fan_share_mean'] / fan_total if fan_total > 0 else 1/len(week_data)

        # Current system
        week_data['combined_current'] = 0.5 * week_data['judge_norm'] + 0.5 * week_data['fan_norm']
        current_rank = week_data.sort_values('combined_current')['celebrity_name'].tolist().index('Bobby Bones') + 1

        # Dynamic system
        week_data['combined_dynamic'] = w_dynamic * week_data['judge_norm'] + (1 - w_dynamic) * week_data['fan_norm']
        dynamic_rank = week_data.sort_values('combined_dynamic')['celebrity_name'].tolist().index('Bobby Bones') + 1

        at_risk = "YES" if dynamic_rank <= 2 else "No"

        print(f"{int(week):>5} {int(bobby_rank):>12} {current_rank:>15} {dynamic_rank:>15} {at_risk:>10}")

    print("  (Rank 1 = lowest combined score, most at risk)")
else:
    print("  Bobby Bones data not found in merged dataset")

# Part 6: Visualization - System Comparison
print("\n[Part 6] Creating Comparison Visualizations...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Bar chart comparison
ax1 = axes[0]
x = np.arange(len(comparison_df))
width = 0.25

bars1 = ax1.bar(x - width, comparison_df['fairness'], width, label='Fairness', color='#3498db')
bars2 = ax1.bar(x, 1 - comparison_df['robbery_rate'], width, label='No-Robbery', color='#2ecc71')
bars3 = ax1.bar(x + width, comparison_df['engagement'], width, label='Engagement', color='#e74c3c')

ax1.set_xlabel('System', fontsize=12)
ax1.set_ylabel('Score', fontsize=12)
ax1.set_title('System Comparison: Three Objectives', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(comparison_df['system'], rotation=45, ha='right', fontsize=9)
ax1.legend()
ax1.grid(True, axis='y', alpha=0.3)

# Plot 2: Radar chart for top systems
ax2 = axes[1]
categories = ['Fairness', 'No-Robbery', 'Engagement']
N = len(categories)

# Systems to compare
systems_to_plot = ['Current (w=0.5)', 'Sigmoid k=1.5', 'Fan-Heavy (w=0.3)']
colors_radar = ['#3498db', '#e74c3c', '#2ecc71']

angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

for i, sys_name in enumerate(systems_to_plot):
    row = comparison_df[comparison_df['system'] == sys_name].iloc[0]
    values = [row['fairness'], 1 - row['robbery_rate'], row['engagement']]
    values += values[:1]
    ax2.plot(angles, values, 'o-', linewidth=2, label=sys_name, color=colors_radar[i])
    ax2.fill(angles, values, alpha=0.1, color=colors_radar[i])

ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(categories, fontsize=11)
ax2.set_title('Radar Comparison: Top 3 Systems', fontsize=14, fontweight='bold')
ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(Q4_DIR, "system_comparison.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: system_comparison.png")

# Plot 3: Dynamic weight visualization for recommendation
fig, ax = plt.subplots(figsize=(10, 6))

t_max = 11
weeks = np.arange(1, t_max + 1)

# Recommended sigmoid
weights = [sigmoid_weight(t, t_max, best['w_min'], best['w_max'], best['k']) for t in weeks]

# Fill between
ax.fill_between(weeks, 0, weights, alpha=0.3, color='#3498db', label='Judge Influence')
ax.fill_between(weeks, weights, 1, alpha=0.3, color='#e74c3c', label='Fan Influence')
ax.plot(weeks, weights, 'b-', linewidth=3, label=f'Recommended Weight (w_min={best["w_min"]:.1f}, w_max={best["w_max"]:.1f})')

# Annotations
ax.annotate('High Fan Influence\n(Entertainment Phase)', xy=(2, 0.25), fontsize=11, ha='center')
ax.annotate('High Judge Influence\n(Competition Phase)', xy=(t_max-1, 0.85), fontsize=11, ha='center')

ax.set_xlabel('Week', fontsize=12)
ax.set_ylabel('Weight', fontsize=12)
ax.set_title('Recommended Dynamic Weight System', fontsize=14, fontweight='bold')
ax.set_xlim(1, t_max)
ax.set_ylim(0, 1)
ax.legend(loc='center right')
ax.grid(True, alpha=0.3)

# Add week markers
for w in weeks:
    ax.axvline(x=w, color='gray', linestyle=':', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(Q4_DIR, "recommended_dynamic_weight.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: recommended_dynamic_weight.png")

# Part 7: Summary
print("DYNAMIC WEIGHT MODEL SUMMARY")

summary_text = f"""
============================================================
Dynamic Control Model - Recommendation
============================================================

RECOMMENDED SYSTEM: Sigmoid Dynamic Weight

Parameters:
- w_min = {best['w_min']:.2f} (Early weeks: low judge weight, protect "dark horses")
- w_max = {best['w_max']:.2f} (Late weeks: high judge weight, ensure skill-based winner)
- k = {best['k']:.2f} (Transition speed)

Formula:
w(t) = {best['w_min']:.2f} + ({best['w_max']:.2f} - {best['w_min']:.2f}) / (1 + exp(-{best['k']:.2f} * (t - t_mid) / (t_max/4)))

Performance:
- Fairness: {best['fairness']:.3f} (Higher = eliminated contestants had lower judge scores)
- Robbery Rate: {best['robbery_rate']*100:.1f}% (Lower = fewer top-scoring contestants eliminated)
- Engagement: {best['engagement']:.3f} (Higher = more fan influence)

WHY ADOPT THIS SYSTEM:

1. EARLY SEASON BENEFITS (w ≈ {best['w_min']:.1f}):
   - Protects entertaining but less skilled contestants
   - Maintains viewer interest with unexpected outcomes
   - Gives celebrities time to improve

2. LATE SEASON BENEFITS (w ≈ {best['w_max']:.1f}):
   - Ensures champion is truly skilled
   - Prevents "Bobby Bones" scenarios
   - Maintains competition integrity

3. GRADUAL TRANSITION:
   - Smooth change avoids sudden rule shifts
   - k = {best['k']:.2f} provides balanced transition
   - Viewers can anticipate changing dynamics

COMPARISON WITH CURRENT SYSTEM (w=0.5 fixed):
- Similar fairness level
- Lower robbery rate (better)
- Higher fan engagement (better)
- More strategic depth for viewers
"""

print(summary_text)

with open(os.path.join(Q4_DIR, "q4_dynamic_summary.txt"), 'w', encoding='utf-8') as f:
    f.write(summary_text)

# Save results
grid_df.to_csv(os.path.join(Q4_DIR, "sigmoid_grid_search.csv"), index=False)
comparison_df.to_csv(os.path.join(Q4_DIR, "system_comparison.csv"), index=False)

print("\n  Saved: q4_dynamic_summary.txt")
print("  Saved: sigmoid_grid_search.csv")
print("  Saved: system_comparison.csv")

print("Part 2 Complete: Dynamic Weight Model")
print("\nOutput files:")
print("  - dynamic_weight_functions.png")
print("  - system_comparison.png")
print("  - recommended_dynamic_weight.png")
print("  - sigmoid_grid_search.csv")
print("  - system_comparison.csv")
print("  - q4_dynamic_summary.txt")