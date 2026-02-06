"""
问题四：多目标优化 + 帕累托前沿分析 + 膝点分析
Question 4: Multi-Objective Optimization with Pareto Frontier & Knee Point

目标函数：
1. 公平性 (Fairness): 最大化淘汰结果与裁判评分排名的一致性
2. 参与度 (Engagement): 最大化粉丝投票对结果的边际影响
3. 避免惨案 (No Robberies): 最小化裁判高分选手被淘汰的概率

决策变量：
- w: 裁判权重 (0-1)

注意: k (bottom k) 参数已移除，因为粉丝投票数据是根据实际淘汰结果反推的，
改变 k 值不会产生有意义的差异。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 绝对路径配置
DATA_DIR = r"d:\2026mcmC\DataProcessed"
Q1_DIR = r"d:\2026mcmC\Q1"
Q4_DIR = r"d:\2026mcmC\Q4"

print("Question 4: Multi-Objective Optimization - Knee Point Analysis")

# Part 1: Load and Prepare Data
print("\n[Part 1] Loading Data...")

# Load judge scores
judge_data = pd.read_csv(os.path.join(DATA_DIR, "DWTS_Processed_Long.csv"), encoding='utf-8-sig')

# Load fan vote estimates
fan_data = pd.read_csv(os.path.join(Q1_DIR, "fan_vote_estimates.csv"))

print(f"  Judge data: {len(judge_data)} records, Seasons: {judge_data['season'].min()}-{judge_data['season'].max()}")
print(f"  Fan vote data: {len(fan_data)} records")

# Merge judge and fan data
merged = pd.merge(
    judge_data,
    fan_data[['season', 'week', 'celebrity_name', 'fan_share_mean', 'eliminated']],
    on=['season', 'week', 'celebrity_name'],
    how='inner'
)

# Filter to seasons 1-27 (where we have both judge and fan data)
merged = merged[merged['season'] <= 27]
print(f"  Merged data: {len(merged)} records")

# Part 2: Define Objective Functions
print("\n[Part 2] Defining Objective Functions...")

def simulate_elimination(week_data, w):
    """
    模拟一周的淘汰决策

    Parameters:
    - week_data: DataFrame with judge scores and fan shares for one week
    - w: judge weight (0-1), fan weight = 1-w

    Returns:
    - eliminated: name of eliminated contestant
    - judge_rank_of_eliminated: where the eliminated contestant ranked in judge scores
    """
    if len(week_data) < 2:
        return None, None

    week_data = week_data.copy()

    # Normalize judge scores within week (min-max to [0,1])
    judge_min = week_data['total_judge_score'].min()
    judge_max = week_data['total_judge_score'].max()
    if judge_max > judge_min:
        week_data['judge_norm'] = (week_data['total_judge_score'] - judge_min) / (judge_max - judge_min)
    else:
        week_data['judge_norm'] = 0.5

    # Normalize fan shares within week (proportion, sum to 1)
    fan_total = week_data['fan_share_mean'].sum()
    if fan_total > 0:
        week_data['fan_norm'] = week_data['fan_share_mean'] / fan_total
    else:
        week_data['fan_norm'] = 1 / len(week_data)

    # Combined score: weighted sum
    week_data['combined_score'] = w * week_data['judge_norm'] + (1 - w) * week_data['fan_norm']

    # Eliminated is the one with lowest combined score
    week_data = week_data.sort_values('combined_score')
    eliminated_name = week_data.iloc[0]['celebrity_name']

    # What was the judge ranking of the eliminated contestant?
    week_data_sorted = week_data.sort_values('total_judge_score', ascending=False)
    week_data_sorted['judge_rank'] = range(1, len(week_data_sorted) + 1)
    judge_rank = week_data_sorted[week_data_sorted['celebrity_name'] == eliminated_name]['judge_rank'].values[0]

    return eliminated_name, judge_rank


def calculate_objectives(merged_data, w):
    """
    计算三个目标函数值

    Returns:
    - fairness: 淘汰者裁判排名与选手数比值的平均（越高越公平，淘汰的是真正差的）
    - engagement: 粉丝投票对结果的边际影响（越高参与度越高）
    - no_robbery: 避免裁判高分选手被淘汰（越高越好）
    """
    fairness_scores = []
    robbery_count = 0
    total_eliminations = 0
    fan_decisive_count = 0

    for (season, week), week_data in merged_data.groupby(['season', 'week']):
        if len(week_data) < 3:  # Need at least 3 contestants
            continue

        # Simulate elimination with given weight
        eliminated, judge_rank = simulate_elimination(week_data, w)
        if eliminated is None:
            continue

        n_contestants = len(week_data)
        total_eliminations += 1

        # Fairness: eliminated should have low judge score
        # judge_rank / n_contestants: higher = eliminated was ranked lower by judges = good
        fairness_ratio = judge_rank / n_contestants
        fairness_scores.append(fairness_ratio)

        # Robbery: if eliminated was top 2 in judge scores (high rank = top performer)
        if judge_rank <= 2:
            robbery_count += 1

        # Fan decisive: check if fan vote changed the outcome vs judge-only
        eliminated_judge_only, _ = simulate_elimination(week_data, 1.0)
        if eliminated != eliminated_judge_only:
            fan_decisive_count += 1

    if total_eliminations == 0:
        return 0, 0, 0

    # Objective 1: Fairness (average of fairness ratios, higher = better)
    fairness = np.mean(fairness_scores) if fairness_scores else 0

    # Objective 2: Engagement (proportion of times fan vote was decisive)
    engagement = fan_decisive_count / total_eliminations

    # Objective 3: No Robbery (1 - robbery rate)
    no_robbery = 1 - (robbery_count / total_eliminations)

    return fairness, engagement, no_robbery


# Part 3: Compute Objectives for Different Weights
print("\n[Part 3] Computing Objectives for Different Weights...")

# Define weight grid (fine granularity)
w_values = np.arange(0.0, 1.01, 0.05)

results = []
for w in w_values:
    fairness, engagement, no_robbery = calculate_objectives(merged, w)
    results.append({
        'w': round(w, 2),
        'fairness': fairness,
        'engagement': engagement,
        'no_robbery': no_robbery
    })

results_df = pd.DataFrame(results)

# Print key values
print("\n  Weight (w) | Fairness | Engagement | No-Robbery")
print("  " + "-" * 50)
for _, row in results_df[results_df['w'].isin([0.0, 0.25, 0.5, 0.75, 1.0])].iterrows():
    print(f"     {row['w']:.2f}    |  {row['fairness']:.3f}   |   {row['engagement']:.3f}    |   {row['no_robbery']:.3f}")

# Part 4: Knee Point Analysis (膝点分析)
print("\n[Part 4] Knee Point Analysis...")

def find_knee_kneedle(x, y):
    """
    Kneedle Algorithm: 找到帕累托前沿上到端点连线距离最大的点
    这代表"性价比最高"的点
    """
    # Normalize to [0, 1]
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-10)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-10)

    # Line from first point to last point
    x0, y0 = x_norm[0], y_norm[0]
    x1, y1 = x_norm[-1], y_norm[-1]

    # Distance from each point to the line
    # Line: (y1-y0)x - (x1-x0)y + (x1-x0)y0 - (y1-y0)x0 = 0
    a = y1 - y0
    b = -(x1 - x0)
    c = (x1 - x0) * y0 - (y1 - y0) * x0

    distances = (a * x_norm + b * y_norm + c) / np.sqrt(a**2 + b**2 + 1e-10)

    knee_idx = np.argmax(distances)
    return knee_idx, distances


def find_knee_marginal(w_values, fairness, engagement):
    """
    边际成本分析: 找到边际成本开始显著下降的点
    边际成本 = dFairness / |dEngagement|
    """
    marginal_costs = []
    for i in range(1, len(w_values)):
        dF = fairness[i] - fairness[i-1]
        dE = engagement[i] - engagement[i-1]
        if abs(dE) > 0.001:
            mc = -dF / dE  # negative because E decreases as w increases
        else:
            mc = 0
        marginal_costs.append(mc)

    marginal_costs = np.array(marginal_costs)

    # Find where marginal cost drops below 50% of peak
    mc_max = np.max(marginal_costs)
    threshold = mc_max * 0.5

    knee_idx = 1
    for i, mc in enumerate(marginal_costs):
        if i > 0 and mc < threshold and marginal_costs[i-1] >= threshold:
            knee_idx = i + 1
            break

    if knee_idx == 1:
        knee_idx = np.argmax(marginal_costs) + 1

    return knee_idx, marginal_costs


# Prepare data for knee analysis (sort by engagement descending = Pareto frontier order)
pareto_data = results_df.sort_values('engagement', ascending=False).reset_index(drop=True)
x_engagement = pareto_data['engagement'].values
y_fairness = pareto_data['fairness'].values

# Method 1: Kneedle Algorithm
knee_idx_kneedle, distances = find_knee_kneedle(x_engagement, y_fairness)
knee_point_kneedle = pareto_data.iloc[knee_idx_kneedle]

# Method 2: Marginal Cost Analysis
sorted_by_w = results_df.sort_values('w').reset_index(drop=True)
knee_idx_marginal, marginal_costs = find_knee_marginal(
    sorted_by_w['w'].values,
    sorted_by_w['fairness'].values,
    sorted_by_w['engagement'].values
)
knee_point_marginal = sorted_by_w.iloc[knee_idx_marginal]

print("\n  [Method 1: Kneedle Algorithm - Best Value Point]")
print(f"    Recommended w = {knee_point_kneedle['w']:.2f}")
print(f"    Interpretation: {knee_point_kneedle['w']*100:.0f}% Judge + {(1-knee_point_kneedle['w'])*100:.0f}% Fan")
print(f"    Fairness:   {knee_point_kneedle['fairness']:.3f}")
print(f"    Engagement: {knee_point_kneedle['engagement']:.3f}")
print(f"    No-Robbery: {knee_point_kneedle['no_robbery']:.3f}")

print("\n  [Method 2: Marginal Cost - Diminishing Returns Point]")
print(f"    Recommended w = {knee_point_marginal['w']:.2f}")
print(f"    Interpretation: {knee_point_marginal['w']*100:.0f}% Judge + {(1-knee_point_marginal['w'])*100:.0f}% Fan")
print(f"    Fairness:   {knee_point_marginal['fairness']:.3f}")
print(f"    Engagement: {knee_point_marginal['engagement']:.3f}")
print(f"    No-Robbery: {knee_point_marginal['no_robbery']:.3f}")

# Use Kneedle as the recommended solution
best_solution = knee_point_kneedle

# Part 5: Visualization
print("\n[Part 5] Creating Visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# --- Plot 1: Pareto Frontier with Knee Point ---
ax1 = axes[0, 0]
ax1.plot(x_engagement, y_fairness, 'b-o', linewidth=2, markersize=8, label='Pareto Frontier', zorder=2)

# Mark knee points
ax1.scatter(knee_point_kneedle['engagement'], knee_point_kneedle['fairness'],
            s=250, c='red', marker='*', zorder=5, edgecolors='black', linewidths=1,
            label=f'Kneedle: w={knee_point_kneedle["w"]:.2f}')
ax1.scatter(knee_point_marginal['engagement'], knee_point_marginal['fairness'],
            s=200, c='orange', marker='D', zorder=5, edgecolors='black', linewidths=1,
            label=f'Marginal: w={knee_point_marginal["w"]:.2f}')

# Draw baseline (line from first to last point)
ax1.plot([x_engagement[0], x_engagement[-1]], [y_fairness[0], y_fairness[-1]],
         'k--', alpha=0.5, linewidth=1.5, label='Baseline')

# Annotate regions
ax1.annotate('High Engagement\nLow Fairness', xy=(0.5, 0.80), fontsize=9, ha='center', color='gray')
ax1.annotate('High Fairness\nLow Engagement', xy=(0.05, 0.97), fontsize=9, ha='center', color='gray')

ax1.set_xlabel('Engagement (Fan Vote Impact)', fontsize=12)
ax1.set_ylabel('Fairness (Eliminates Low Scorers)', fontsize=12)
ax1.set_title('Pareto Frontier: Fairness vs Engagement\nwith Knee Point Analysis', fontsize=14, fontweight='bold')
ax1.legend(loc='lower left', fontsize=9)
ax1.grid(True, alpha=0.3)

# --- Plot 2: Marginal Cost Analysis ---
ax2 = axes[0, 1]
w_plot = sorted_by_w['w'].values[1:]  # Marginal cost starts from w[1]
bars = ax2.bar(w_plot, marginal_costs, width=0.04, alpha=0.7, color='steelblue', edgecolor='black')

# Color bars by value
for i, (bar, mc) in enumerate(zip(bars, marginal_costs)):
    if mc > 0.4:
        bar.set_color('#2ecc71')  # Green: high value
    elif mc < 0.25:
        bar.set_color('#e74c3c')  # Red: low value
    else:
        bar.set_color('#f39c12')  # Orange: medium

ax2.axvline(x=knee_point_marginal['w'], color='red', linestyle='--', linewidth=2,
            label=f'Knee Point: w={knee_point_marginal["w"]:.2f}')
ax2.axhline(y=0.4, color='green', linestyle=':', alpha=0.7, label='High Value Threshold')
ax2.axhline(y=0.25, color='red', linestyle=':', alpha=0.7, label='Low Value Threshold')

ax2.set_xlabel('Judge Weight (w)', fontsize=12)
ax2.set_ylabel('Marginal Cost\n(Fairness Gain per Engagement Loss)', fontsize=12)
ax2.set_title('Marginal Cost Analysis\n(Higher = Better Value)', fontsize=14, fontweight='bold')
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(True, alpha=0.3, axis='y')

# --- Plot 3: Trade-off Curves ---
ax3 = axes[1, 0]
w_sorted = sorted_by_w['w'].values
ax3.plot(w_sorted, sorted_by_w['fairness'].values, 'b-o', linewidth=2, markersize=6, label='Fairness')
ax3.plot(w_sorted, sorted_by_w['engagement'].values, 'r-s', linewidth=2, markersize=6, label='Engagement')
ax3.plot(w_sorted, sorted_by_w['no_robbery'].values, 'g-^', linewidth=2, markersize=6, label='No-Robbery')

# Mark recommended point
ax3.axvline(x=best_solution['w'], color='purple', linestyle='--', linewidth=2, alpha=0.8,
            label=f'Recommended: w={best_solution["w"]:.2f}')

# Fill area between fairness and engagement
ax3.fill_between(w_sorted, sorted_by_w['fairness'].values, sorted_by_w['engagement'].values,
                  alpha=0.15, color='purple')

ax3.set_xlabel('Judge Weight (w)', fontsize=12)
ax3.set_ylabel('Objective Value (0-1)', fontsize=12)
ax3.set_title('Trade-off: Three Objectives vs Judge Weight', fontsize=14, fontweight='bold')
ax3.legend(loc='center right', fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(-0.02, 1.02)
ax3.set_ylim(0, 1.05)

# --- Plot 4: Distance to Baseline (Kneedle metric) ---
ax4 = axes[1, 1]

# Need to map distances back to w values
pareto_w = pareto_data['w'].values
ax4.plot(pareto_w, distances, 'b-o', linewidth=2, markersize=6, label='Distance to Baseline')
ax4.scatter(knee_point_kneedle['w'], distances[knee_idx_kneedle],
            s=200, c='red', marker='*', zorder=5, label=f'Max Distance: w={knee_point_kneedle["w"]:.2f}')

ax4.axvline(x=knee_point_kneedle['w'], color='red', linestyle='--', linewidth=2, alpha=0.7)

ax4.set_xlabel('Judge Weight (w)', fontsize=12)
ax4.set_ylabel('Distance to Baseline', fontsize=12)
ax4.set_title('Kneedle Algorithm:\nDistance to Baseline vs Weight', fontsize=14, fontweight='bold')
ax4.legend(loc='upper right', fontsize=9)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("os.path.join(Q4_DIR, "knee_point_analysis.png")", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: knee_point_analysis.png")

# --- Additional: Simple Trade-off Plot ---
fig2, ax = plt.subplots(figsize=(10, 6))
ax.plot(w_sorted, sorted_by_w['fairness'].values, 'b-o', linewidth=2.5, markersize=8, label='Fairness')
ax.plot(w_sorted, sorted_by_w['engagement'].values, 'r-s', linewidth=2.5, markersize=8, label='Engagement')
ax.axvline(x=best_solution['w'], color='green', linestyle='--', linewidth=3, alpha=0.8,
            label=f'Optimal: w={best_solution["w"]:.2f}')
ax.fill_between(w_sorted, sorted_by_w['fairness'].values, sorted_by_w['engagement'].values,
                 alpha=0.2, color='gray')
ax.set_xlabel('Judge Weight (w)', fontsize=14)
ax.set_ylabel('Objective Value', fontsize=14)
ax.set_title('The Core Trade-off: Fairness vs Fan Engagement', fontsize=16, fontweight='bold')
ax.legend(loc='center right', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig("os.path.join(Q4_DIR, "tradeoff_curve.png")", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: tradeoff_curve.png")

# Part 6: Results Summary
print("OPTIMIZATION RESULTS SUMMARY")

# Current system
current = results_df[results_df['w'] == 0.5].iloc[0]

print("\n  +---------------------------------------------------------------+")
print("  |                    COMPARISON TABLE                           |")
print("  +---------------------------------------------------------------+")
print("  | System              |   w   | Fairness | Engagement | NoRob  |")
print("  +---------------------------------------------------------------+")
print(f"  | Current (w=0.50)    | 0.50  |  {current['fairness']:.3f}   |   {current['engagement']:.3f}    | {current['no_robbery']:.3f}  |")
print(f"  | Kneedle Optimal     | {best_solution['w']:.2f}  |  {best_solution['fairness']:.3f}   |   {best_solution['engagement']:.3f}    | {best_solution['no_robbery']:.3f}  |")
print(f"  | Marginal Optimal    | {knee_point_marginal['w']:.2f}  |  {knee_point_marginal['fairness']:.3f}   |   {knee_point_marginal['engagement']:.3f}    | {knee_point_marginal['no_robbery']:.3f}  |")
print(f"  | Pure Judge (w=1.0)  | 1.00  |  {results_df[results_df['w']==1.0].iloc[0]['fairness']:.3f}   |   {results_df[results_df['w']==1.0].iloc[0]['engagement']:.3f}    | {results_df[results_df['w']==1.0].iloc[0]['no_robbery']:.3f}  |")
print(f"  | Pure Fan (w=0.0)    | 0.00  |  {results_df[results_df['w']==0.0].iloc[0]['fairness']:.3f}   |   {results_df[results_df['w']==0.0].iloc[0]['engagement']:.3f}    | {results_df[results_df['w']==0.0].iloc[0]['no_robbery']:.3f}  |")
print("  +---------------------------------------------------------------+")

print(f"\n  [RECOMMENDED SOLUTION]")
print(f"  Judge Weight (w): {best_solution['w']:.2f}")
print(f"  Formula: Combined Score = {best_solution['w']:.0%} × Judge Score + {1-best_solution['w']:.0%} × Fan Vote")
print(f"")
print(f"  Expected Performance:")
print(f"    - Fairness:   {best_solution['fairness']:.1%} of eliminations are low-scoring dancers")
print(f"    - Engagement: {best_solution['engagement']:.1%} of outcomes are influenced by fan votes")
print(f"    - No-Robbery: {best_solution['no_robbery']:.1%} of top scorers are protected")

print(f"\n  [IMPROVEMENT vs Current System (w=0.5)]")
print(f"    Fairness:   {(best_solution['fairness'] - current['fairness'])*100:+.1f} percentage points")
print(f"    Engagement: {(best_solution['engagement'] - current['engagement'])*100:+.1f} percentage points")
print(f"    No-Robbery: {(best_solution['no_robbery'] - current['no_robbery'])*100:+.1f} percentage points")

# Save results
results_df.to_csv("os.path.join(Q4_DIR, "pareto_results.csv")", index=False)
print("\n  Saved: pareto_results.csv")

print("Analysis Complete!")
print("\nOutput files:")
print("  - knee_point_analysis.png (4-panel analysis)")
print("  - tradeoff_curve.png (simple trade-off visualization)")
print("  - pareto_results.csv (all computed values)")
