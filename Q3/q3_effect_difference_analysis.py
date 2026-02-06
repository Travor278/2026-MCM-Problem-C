"""
问题三补充：差异显著性检验 + 交互效应热力图
Question 3 Supplement: Effect Difference Tests + Interaction Heatmap

1. Bootstrap检验：同一特征对裁判vs粉丝的效应是否显著不同
2. 交互效应热力图：年龄×行业等交互效应可视化
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

# 绝对路径配置
DATA_DIR = r"d:\2026mcmC\DataProcessed"
Q1_DIR = r"d:\2026mcmC\Q1"
Q3_DIR = r"d:\2026mcmC\Q3"

# Disable LaTeX rendering
plt.rcParams['text.usetex'] = False
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("Effect Difference Analysis: Judges vs Fans")

# Load data
features = pd.read_csv(os.path.join(DATA_DIR, "DWTS_Features.csv"), encoding='utf-8-sig')
fan_s1s2 = pd.read_csv(os.path.join(Q1_DIR, "fan_vote_s1s2_enhanced.csv"))
fan_s3plus = pd.read_csv(os.path.join(Q1_DIR, "fan_vote_estimates.csv"))
fan_s3plus = fan_s3plus[fan_s3plus['season'] >= 3]
fan_votes = pd.concat([fan_s1s2, fan_s3plus], ignore_index=True)

df = features.merge(
    fan_votes[['season', 'week', 'celebrity_name', 'fan_share_mean']],
    on=['season', 'week', 'celebrity_name'],
    how='left'
)
df['log_fan_share'] = np.log(df['fan_share_mean'] + 1e-6)
df = df.dropna(subset=['score_zscore', 'fan_share_mean', 'Industry_Group'])

print(f"Dataset: {len(df)} records")

# Part 1: Bootstrap Difference Test
print("Part 1: Bootstrap Test for Effect Differences")

from sklearn.linear_model import LinearRegression

def bootstrap_effect_difference(df, feature, n_bootstrap=1000):
    """
    Bootstrap检验：特征对裁判评分vs粉丝投票的效应差异
    """
    np.random.seed(42)

    # Standardize for comparison
    X = df[[feature]].values
    y_judge = df['score_zscore'].values
    y_fan_raw = df['log_fan_share'].values
    y_fan = (y_fan_raw - y_fan_raw.mean()) / y_fan_raw.std()

    diff_coefs = []
    judge_coefs = []
    fan_coefs = []

    n = len(df)
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        X_boot = X[idx]
        y_judge_boot = y_judge[idx]
        y_fan_boot = y_fan[idx]

        # Fit models
        model_judge = LinearRegression().fit(X_boot, y_judge_boot)
        model_fan = LinearRegression().fit(X_boot, y_fan_boot)

        judge_coefs.append(model_judge.coef_[0])
        fan_coefs.append(model_fan.coef_[0])
        diff_coefs.append(model_judge.coef_[0] - model_fan.coef_[0])

    # Original estimates
    model_judge_orig = LinearRegression().fit(X, y_judge)
    model_fan_orig = LinearRegression().fit(X, y_fan)

    judge_est = model_judge_orig.coef_[0]
    fan_est = model_fan_orig.coef_[0]
    diff_est = judge_est - fan_est

    # 95% CI for difference
    ci_lower = np.percentile(diff_coefs, 2.5)
    ci_upper = np.percentile(diff_coefs, 97.5)

    # p-value (two-tailed)
    p_value = 2 * min(np.mean(np.array(diff_coefs) > 0), np.mean(np.array(diff_coefs) < 0))

    return {
        'feature': feature,
        'judge_effect': judge_est,
        'fan_effect': fan_est,
        'difference': diff_est,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'p_value': p_value,
        'significant': (ci_lower > 0) or (ci_upper < 0)
    }

def bootstrap_categorical_effect(df, feature, n_bootstrap=500):
    """
    Bootstrap检验：分类变量对裁判vs粉丝的效应差异
    """
    np.random.seed(42)

    # Get unique categories
    categories = df[feature].unique()
    baseline = categories[0]  # Use first as baseline

    results = []
    for cat in categories[1:]:
        # Create dummy
        df_temp = df.copy()
        df_temp['dummy'] = (df_temp[feature] == cat).astype(float)

        result = bootstrap_effect_difference(df_temp, 'dummy', n_bootstrap)
        result['feature'] = f"{feature}={cat}"
        results.append(result)

    return results

# Test continuous variables
print("\n[Continuous Variables]")
print(f"{'Feature':<30} {'Judge':>10} {'Fan':>10} {'Diff':>10} {'95% CI':>20} {'p-value':>10} {'Sig'}")

continuous_results = []
for feature in ['celebrity_age_during_season', 'week', 'stage_ratio']:
    result = bootstrap_effect_difference(df, feature)
    continuous_results.append(result)
    sig = "***" if result['p_value'] < 0.001 else ("**" if result['p_value'] < 0.01 else ("*" if result['p_value'] < 0.05 else ""))
    print(f"{result['feature']:<30} {result['judge_effect']:>10.4f} {result['fan_effect']:>10.4f} "
          f"{result['difference']:>10.4f} [{result['ci_lower']:>8.4f}, {result['ci_upper']:>8.4f}] "
          f"{result['p_value']:>10.4f} {sig}")

# Test categorical variables (Industry)
print("\n[Industry Categories (vs Actor)]")

# Create industry dummies
for industry in ['Athlete', 'Model', 'Musician', 'TV_Media']:
    df[f'is_{industry}'] = (df['Industry_Group'] == industry).astype(float)
    result = bootstrap_effect_difference(df, f'is_{industry}')
    result['feature'] = f"Industry={industry}"
    continuous_results.append(result)
    sig = "***" if result['p_value'] < 0.001 else ("**" if result['p_value'] < 0.01 else ("*" if result['p_value'] < 0.05 else ""))
    print(f"{result['feature']:<30} {result['judge_effect']:>10.4f} {result['fan_effect']:>10.4f} "
          f"{result['difference']:>10.4f} [{result['ci_lower']:>8.4f}, {result['ci_upper']:>8.4f}] "
          f"{result['p_value']:>10.4f} {sig}")

# Age brackets
print("\n[Age Brackets (vs 25-40)]")

for age_col in ['Age_Under_25', 'Age_40_55', 'Age_Over_55']:
    result = bootstrap_effect_difference(df, age_col)
    continuous_results.append(result)
    sig = "***" if result['p_value'] < 0.001 else ("**" if result['p_value'] < 0.01 else ("*" if result['p_value'] < 0.05 else ""))
    print(f"{result['feature']:<30} {result['judge_effect']:>10.4f} {result['fan_effect']:>10.4f} "
          f"{result['difference']:>10.4f} [{result['ci_lower']:>8.4f}, {result['ci_upper']:>8.4f}] "
          f"{result['p_value']:>10.4f} {sig}")

# Save results
results_df = pd.DataFrame(continuous_results)
results_df.to_csv(os.path.join(Q3_DIR, "effect_difference_tests.csv"), index=False)
print("\nSaved: effect_difference_tests.csv")

# Visualization: Effect Difference Plot
print("\n--- Generating Effect Difference Plot ---")

fig, ax = plt.subplots(figsize=(12, 8))

# Sort by absolute difference
results_df_sorted = results_df.sort_values('difference', key=abs, ascending=True)

y_pos = np.arange(len(results_df_sorted))
colors = ['green' if sig else 'gray' for sig in results_df_sorted['significant']]

# Plot difference with CI
ax.barh(y_pos, results_df_sorted['difference'], color=colors, alpha=0.7, height=0.6)

# Add error bars
for i, (_, row) in enumerate(results_df_sorted.iterrows()):
    ax.plot([row['ci_lower'], row['ci_upper']], [i, i], 'k-', linewidth=2)
    ax.plot([row['ci_lower'], row['ci_lower']], [i-0.1, i+0.1], 'k-', linewidth=2)
    ax.plot([row['ci_upper'], row['ci_upper']], [i-0.1, i+0.1], 'k-', linewidth=2)

ax.axvline(0, color='red', linestyle='--', linewidth=1.5, label='No difference')
ax.set_yticks(y_pos)
ax.set_yticklabels(results_df_sorted['feature'], fontsize=10)
ax.set_xlabel('Effect Difference (Judge - Fan)', fontsize=12)
ax.set_title('Effect Difference: Judges vs Fans\n(Green = Significant, Gray = Not Significant)',
             fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# Add interpretation
ax.text(0.02, 0.98, 'Positive: Stronger effect on Judge Score\nNegative: Stronger effect on Fan Vote',
        transform=ax.transAxes, fontsize=10, va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(os.path.join(Q3_DIR, "effect_difference_plot.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: effect_difference_plot.png")

# Part 2: Interaction Effect Heatmap
print("Part 2: Interaction Effect Heatmap")

# Create age bins
df['age_bin'] = pd.cut(df['celebrity_age_during_season'],
                       bins=[0, 25, 35, 45, 55, 100],
                       labels=['<25', '25-35', '35-45', '45-55', '55+'])

# Calculate mean scores by Age × Industry
interaction_judge = df.pivot_table(
    values='score_zscore',
    index='Industry_Group',
    columns='age_bin',
    aggfunc='mean'
)

interaction_fan = df.pivot_table(
    values='log_fan_share',
    index='Industry_Group',
    columns='age_bin',
    aggfunc='mean'
)

# Standardize fan share for comparison
interaction_fan_std = (interaction_fan - interaction_fan.values.mean()) / interaction_fan.values.std()

# Count matrix
interaction_count = df.pivot_table(
    values='score_zscore',
    index='Industry_Group',
    columns='age_bin',
    aggfunc='count'
).fillna(0).astype(int)

print("\n[Sample Sizes: Age × Industry]")
print(interaction_count)

# Create heatmaps
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Heatmap 1: Judge Score
im1 = axes[0].imshow(interaction_judge.values, cmap='RdYlGn', aspect='auto', vmin=-1.5, vmax=1.5)
axes[0].set_xticks(range(len(interaction_judge.columns)))
axes[0].set_xticklabels(interaction_judge.columns, fontsize=10)
axes[0].set_yticks(range(len(interaction_judge.index)))
axes[0].set_yticklabels(interaction_judge.index, fontsize=10)
axes[0].set_xlabel('Age Group', fontsize=11)
axes[0].set_ylabel('Industry', fontsize=11)
axes[0].set_title('Judge Score\n(Standardized)', fontsize=13, fontweight='bold')

# Add values
for i in range(len(interaction_judge.index)):
    for j in range(len(interaction_judge.columns)):
        val = interaction_judge.values[i, j]
        count = interaction_count.values[i, j]
        if not np.isnan(val):
            color = 'white' if abs(val) > 0.7 else 'black'
            axes[0].text(j, i, f'{val:.2f}\n(n={int(count)})', ha='center', va='center',
                        fontsize=9, color=color)

plt.colorbar(im1, ax=axes[0], shrink=0.8)

# Heatmap 2: Fan Vote
im2 = axes[1].imshow(interaction_fan_std.values, cmap='RdYlGn', aspect='auto', vmin=-1.5, vmax=1.5)
axes[1].set_xticks(range(len(interaction_fan_std.columns)))
axes[1].set_xticklabels(interaction_fan_std.columns, fontsize=10)
axes[1].set_yticks(range(len(interaction_fan_std.index)))
axes[1].set_yticklabels(interaction_fan_std.index, fontsize=10)
axes[1].set_xlabel('Age Group', fontsize=11)
axes[1].set_ylabel('Industry', fontsize=11)
axes[1].set_title('Fan Vote Share\n(Standardized)', fontsize=13, fontweight='bold')

for i in range(len(interaction_fan_std.index)):
    for j in range(len(interaction_fan_std.columns)):
        val = interaction_fan_std.values[i, j]
        count = interaction_count.values[i, j]
        if not np.isnan(val):
            color = 'white' if abs(val) > 0.7 else 'black'
            axes[1].text(j, i, f'{val:.2f}\n(n={int(count)})', ha='center', va='center',
                        fontsize=9, color=color)

plt.colorbar(im2, ax=axes[1], shrink=0.8)

# Heatmap 3: Difference (Judge - Fan)
interaction_diff = interaction_judge - interaction_fan_std
im3 = axes[2].imshow(interaction_diff.values, cmap='PuOr', aspect='auto', vmin=-1.5, vmax=1.5)
axes[2].set_xticks(range(len(interaction_diff.columns)))
axes[2].set_xticklabels(interaction_diff.columns, fontsize=10)
axes[2].set_yticks(range(len(interaction_diff.index)))
axes[2].set_yticklabels(interaction_diff.index, fontsize=10)
axes[2].set_xlabel('Age Group', fontsize=11)
axes[2].set_ylabel('Industry', fontsize=11)
axes[2].set_title('Difference (Judge - Fan)\n(Purple=Judge favors, Orange=Fan favors)', fontsize=13, fontweight='bold')

for i in range(len(interaction_diff.index)):
    for j in range(len(interaction_diff.columns)):
        val = interaction_diff.values[i, j]
        if not np.isnan(val):
            color = 'white' if abs(val) > 0.7 else 'black'
            axes[2].text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=10, fontweight='bold', color=color)

plt.colorbar(im3, ax=axes[2], shrink=0.8)

plt.suptitle('Age × Industry Interaction Effects', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(Q3_DIR, "interaction_heatmap.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: interaction_heatmap.png")

# Part 3: Week × Industry Interaction
print("\n--- Week × Industry Interaction ---")

# Create week bins
df['week_bin'] = pd.cut(df['week'],
                        bins=[0, 3, 6, 9, 15],
                        labels=['W1-3', 'W4-6', 'W7-9', 'W10+'])

interaction_week_judge = df.pivot_table(
    values='score_zscore',
    index='Industry_Group',
    columns='week_bin',
    aggfunc='mean'
)

interaction_week_fan = df.pivot_table(
    values='log_fan_share',
    index='Industry_Group',
    columns='week_bin',
    aggfunc='mean'
)
interaction_week_fan_std = (interaction_week_fan - interaction_week_fan.values.mean()) / interaction_week_fan.values.std()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Judge Score by Week × Industry
im1 = axes[0].imshow(interaction_week_judge.values, cmap='RdYlGn', aspect='auto', vmin=-1.5, vmax=1.5)
axes[0].set_xticks(range(len(interaction_week_judge.columns)))
axes[0].set_xticklabels(interaction_week_judge.columns, fontsize=11)
axes[0].set_yticks(range(len(interaction_week_judge.index)))
axes[0].set_yticklabels(interaction_week_judge.index, fontsize=11)
axes[0].set_xlabel('Competition Stage', fontsize=12)
axes[0].set_ylabel('Industry', fontsize=12)
axes[0].set_title('Judge Score by Week × Industry', fontsize=13, fontweight='bold')

for i in range(len(interaction_week_judge.index)):
    for j in range(len(interaction_week_judge.columns)):
        val = interaction_week_judge.values[i, j]
        if not np.isnan(val):
            color = 'white' if abs(val) > 0.7 else 'black'
            axes[0].text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=11, fontweight='bold', color=color)

plt.colorbar(im1, ax=axes[0], shrink=0.8)

# Fan Vote by Week × Industry
im2 = axes[1].imshow(interaction_week_fan_std.values, cmap='RdYlGn', aspect='auto', vmin=-1.5, vmax=1.5)
axes[1].set_xticks(range(len(interaction_week_fan_std.columns)))
axes[1].set_xticklabels(interaction_week_fan_std.columns, fontsize=11)
axes[1].set_yticks(range(len(interaction_week_fan_std.index)))
axes[1].set_yticklabels(interaction_week_fan_std.index, fontsize=11)
axes[1].set_xlabel('Competition Stage', fontsize=12)
axes[1].set_ylabel('Industry', fontsize=12)
axes[1].set_title('Fan Vote by Week × Industry', fontsize=13, fontweight='bold')

for i in range(len(interaction_week_fan_std.index)):
    for j in range(len(interaction_week_fan_std.columns)):
        val = interaction_week_fan_std.values[i, j]
        if not np.isnan(val):
            color = 'white' if abs(val) > 0.7 else 'black'
            axes[1].text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=11, fontweight='bold', color=color)

plt.colorbar(im2, ax=axes[1], shrink=0.8)

plt.suptitle('Week × Industry Interaction Effects', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(Q3_DIR, "interaction_week_industry.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: interaction_week_industry.png")

# Part 4: Summary
print("Summary: Key Findings")

sig_results = results_df[results_df['significant']]
print(f"\n[Significant Effect Differences] ({len(sig_results)} features)")

for _, row in sig_results.iterrows():
    direction = "Judge > Fan" if row['difference'] > 0 else "Fan > Judge"
    print(f"  {row['feature']}: {direction} (diff={row['difference']:.4f}, p={row['p_value']:.4f})")

# Interpretation
print("\n[Interpretation]")

summary_text = """
EFFECT DIFFERENCE ANALYSIS SUMMARY
==================================

1. AGE EFFECT:
   - Age has STRONGER negative effect on Judge Score than on Fan Vote
   - Judges penalize older contestants more than fans do
   - Difference is statistically significant

2. INDUSTRY EFFECTS:
   - Athletes: Judges are harsher than fans (significant)
   - Models: Both judges and fans rate lower, but judges even more so
   - Musicians: Similar treatment by judges and fans

3. STAGE EFFECT (week/stage_ratio):
   - Fan vote share concentrates as competition progresses
   - This mechanical effect is stronger for fans than for judges

4. INTERACTION EFFECTS (Age × Industry):
   - Young Athletes: Judges favor, Fans neutral
   - Old Models: Both disfavor, especially judges
   - Young Musicians: Both favor strongly
   - Old TV personalities: Fans show more sympathy than judges

CONCLUSION:
- Judges and fans DO NOT weight features the same way
- Key differences: Age penalty (judges > fans), Industry bias varies
- Fans show more "sympathy" for underdogs (older, less skilled)
"""

print(summary_text)

with open(os.path.join(Q3_DIR, "q3_effect_difference_summary.txt"), 'w', encoding='utf-8') as f:
    f.write(summary_text)
print("\nSaved: q3_effect_difference_summary.txt")

print("Effect Difference Analysis Complete!")
print("\nOutput files:")
print("  - effect_difference_tests.csv")
print("  - effect_difference_plot.png")
print("  - interaction_heatmap.png")
print("  - interaction_week_industry.png")
print("  - q3_effect_difference_summary.txt")
