# Question 4: Sensitivity Analysis (Stability Check)
"""
Question 4: Sensitivity Analysis (Stability Check)
验证推荐模型的鲁棒性
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# 绝对路径配置
DATA_DIR = r"d:\2026mcmC\DataProcessed"
Q1_DIR = r"d:\2026mcmC\Q1"
SA_DIR = r"d:\2026mcmC\SensitiveAnalyse"

# Apply Nature style font settings globally
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.titleweight': 'bold',
    'axes.labelweight': 'normal',
    'font.weight': 'normal',
    'axes.unicode_minus': False
})

print("Question 4: Sensitivity Analysis - Model Robustness")

# Load data (Assuming pre-processed data exists in memory or file)
# Re-loading for standalone execution
merged = pd.read_csv(os.path.join(DATA_DIR, "DWTS_Processed_Long.csv"), encoding='utf-8-sig')
fan_data = pd.read_csv(os.path.join(Q1_DIR, "fan_vote_estimates.csv"))
merged = pd.merge(merged, fan_data[['season', 'week', 'celebrity_name', 'fan_share_mean']], 
                 on=['season', 'week', 'celebrity_name'], how='inner')

# Pre-process
season_max_weeks = merged.groupby('season')['week'].max()
merged['t_max'] = merged['season'].map(season_max_weeks)
merged['judge_rank'] = merged.groupby(['season', 'week'])['total_judge_score'].rank(ascending=False, method='min')

# Normalize function
def normalize_cols(df):
    j_min = df.groupby(['season', 'week'])['total_judge_score'].transform('min')
    j_max = df.groupby(['season', 'week'])['total_judge_score'].transform('max')
    df['judge_norm'] = np.where((j_max - j_min) > 0, (df['total_judge_score'] - j_min) / (j_max - j_min), 0.5)
    
    f_sum = df.groupby(['season', 'week'])['fan_share_mean'].transform('sum')
    df['fan_norm'] = np.where(f_sum > 0, df['fan_share_mean'] / f_sum, 1/df.groupby(['season', 'week'])['celebrity_name'].transform('count'))
    return df

merged = normalize_cols(merged)

# Sigmoid function
def sigmoid_weight(t, t_max, w_min, w_max, k):
    t_mid = t_max / 2
    t_normalized = (t - t_mid) / (t_max / 4)
    return w_min + (w_max - w_min) / (1 + np.exp(-k * t_normalized))

# Part A: Parameter Sensitivity Heatmap
print("\n[Part A] Parameter Sensitivity Heatmap (w_max vs k)...")

# Define parameter ranges
w_max_range = np.linspace(0.50, 0.70, 21)
k_range = np.linspace(1.0, 3.0, 21)
w_min_fixed = 0.20

results_matrix = np.zeros((len(w_max_range), len(k_range)))

# Evaluation loop
for i, w_max in enumerate(w_max_range):
    for j, k in enumerate(k_range):
        # Calculate composite score
        # 1. Calculate weights
        merged['w'] = sigmoid_weight(merged['week'], merged['t_max'], w_min_fixed, w_max, k)
        merged['combined'] = merged['w'] * merged['judge_norm'] + (1 - merged['w']) * merged['fan_norm']
        
        # 2. Find eliminated
        eliminated_indices = merged.groupby(['season', 'week'])['combined'].idxmin()
        eliminated = merged.loc[eliminated_indices]
        
        # 3. Metrics
        fairness = (eliminated['judge_rank'] / eliminated.groupby(['season', 'week'])['celebrity_name'].transform('count')).mean()
        robbery_rate = (eliminated['judge_rank'] <= 2).mean()
        engagement = 1 - merged['w'].mean()
        
        # Composite Score (Alpha weights: 0.4, 0.3, 0.3)
        composite = 0.4 * fairness + 0.3 * engagement + 0.3 * (1 - robbery_rate)
        results_matrix[i, j] = composite

# Plot Heatmap
fig, ax = plt.subplots(figsize=(10, 8))
# Use custom palette
cmap = sns.light_palette("#3C9BC9", as_cmap=True)

sns.heatmap(results_matrix, xticklabels=[f"{x:.1f}" for x in k_range], 
            yticklabels=[f"{y:.2f}" for y in w_max_range], 
            cmap=cmap, annot=False, ax=ax)

# Invert Y axis to have high w_max at top
ax.invert_yaxis()

# Highlight recommended point (w_max=0.60, k=1.90)
# Find index
rec_w_idx = np.abs(w_max_range - 0.60).argmin()
rec_k_idx = np.abs(k_range - 1.90).argmin()

# Draw a box around recommended
from matplotlib.patches import Rectangle
ax.add_patch(Rectangle((rec_k_idx, rec_w_idx), 1, 1, fill=False, edgecolor='#FC757B', lw=3))
ax.text(rec_k_idx + 1.5, rec_w_idx + 0.5, 'Recommended\n(0.60, 1.90)', color='#FC757B', fontweight='bold', va='center')

ax.set_xlabel('Transition Rate (k)', fontsize=12)
ax.set_ylabel('Max Judge Weight (w_max)', fontsize=12)
ax.set_title('Sensitivity Analysis: Model Robustness', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig("os.path.join(SA_DIR, "sensitivity_heatmap.png")", dpi=150)
print("  Saved: sensitivity_heatmap.png")

# Part C: Noise Robustness (Monte Carlo)
print("\n[Part C] Noise Robustness Analysis (Monte Carlo)...")

# Define controversial contestants to analyze
targets = [
    {'name': 'Bobby Bones', 'season': 27, 'color': '#e74c3c'},
    {'name': 'Bristol Palin', 'season': 11, 'color': '#e67e22'}
]

n_simulations = 50000  # Increased for smoother distribution
noise_level = 0.15    # Increased noise to stress test

rank_history = []

print(f"  Simulating {n_simulations} scenarios with {noise_level*100}% noise...")

for target in targets:
    s_data = merged[merged['season'] == target['season']].copy()
    t_max = s_data['t_max'].iloc[0]
    
    # Filter only weeks where the contestant competed
    target_weeks = s_data[s_data['celebrity_name'] == target['name']]['week'].unique()
    
    for sim in range(n_simulations):
        # Add noise to fan votes (Season-wide to maintain consistency)
        noise = np.random.normal(0, noise_level, len(s_data))
        noisy_share = np.clip(s_data['fan_share_mean'] * (1 + noise), 0.001, 0.999)
        
        # Re-normalize
        # Need to re-normalize per week
        # Create temp df
        sim_df = s_data.copy()
        sim_df['fan_noisy'] = noisy_share
        f_sums = sim_df.groupby('week')['fan_noisy'].transform('sum')
        sim_df['fan_norm_noisy'] = sim_df['fan_noisy'] / f_sums
        
        # Calculate Dynamic Weight
        # Vectorized for weeks
        w_vec = sigmoid_weight(sim_df['week'], t_max, 0.20, 0.60, 1.90)
        
        # Combined Score
        sim_df['combined'] = w_vec * sim_df['judge_norm'] + (1 - w_vec) * sim_df['fan_norm_noisy']
        
        # Calculate Ranks per week
        # Group by week and rank combined score
        sim_df['rank'] = sim_df.groupby('week')['combined'].rank(ascending=True, method='min')
        
        # Extract target's rank
        target_rows = sim_df[sim_df['celebrity_name'] == target['name']]
        
        for _, row in target_rows.iterrows():
            rank_history.append({
                'Contestant': target['name'],
                'Simulation': sim,
                'Week': row['week'],
                'Rank': row['rank'],
                'Is_Danger': row['rank'] <= 2
            })

rank_df = pd.DataFrame(rank_history)

# Visualization: Line Plot with Error Bands (Nature Style)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Apply Nature style font settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'normal'

# Define custom colors from user palette
color_salmon_pink = '#FC757B'
color_coral = '#F97F5F'
color_sandy_brown = '#FAA26F'
color_light_orange = '#FDCD94'
color_pale_yellow = '#FEE199'
color_pale_green = '#B0D6A9'
color_teal = '#65BDBA'
color_cerulean = '#3C9BC9'

# Override target colors with palette colors
targets[0]['color'] = color_cerulean # Bobby Bones
targets[1]['color'] = color_teal     # Bristol Palin

for i, target in enumerate(targets):
    ax = axes[i]
    data = rank_df[rank_df['Contestant'] == target['name']]
    
    # Filter late season only for clarity (e.g. Week 4+)
    data = data[data['Week'] >= 4]
    
    # Calculate stats for plotting
    # Mean rank
    mean_rank = data.groupby('Week')['Rank'].mean()
    # Confidence Interval (95%)
    ci_lower = data.groupby('Week')['Rank'].quantile(0.025)
    ci_high = data.groupby('Week')['Rank'].quantile(0.975)
    
    weeks = mean_rank.index
    
    # Plot Danger Zone (Background)
    # Rank <= 2 is Danger Zone. 
    # Since Y axis is Rank, we fill from y=0 to y=2.5
    # Use Salmon Pink for Danger Zone
    ax.axhspan(0, 2.5, color=color_salmon_pink, alpha=0.15, label='Danger Zone (Elimination Risk)')
    ax.axhline(y=2.5, color=color_coral, linestyle='--', linewidth=1.5)
    
    # Plot Error Band
    ax.fill_between(weeks, ci_lower, ci_high, color=target['color'], alpha=0.3, label='95% Confidence Interval')
    
    # Plot Mean Line
    ax.plot(weeks, mean_rank, color=target['color'], linewidth=3, marker='o', markersize=6, label='Average Simulated Rank')
    
    # Add text annotation for probability of being in danger at Week 9/Finals
    max_week = data['Week'].max()
    danger_prob = data[data['Week'] == max_week]['Is_Danger'].mean()
    
    # Text box properties
    bbox_props = dict(boxstyle="round,pad=0.5", fc="white", ec=target['color'], alpha=0.9)
    
    # Move text to lower left to avoid blocking the curve (which goes down)
    ax.text(0.05, 0.05, 
            f"Final Elimination Prob:\n{danger_prob*100:.1f}%", 
            transform=ax.transAxes, fontsize=12, fontweight='bold', color=target['color'],
            bbox=bbox_props, va='bottom')

    ax.set_title(f"Robustness Test: {target['name']}\n(with ±15% Vote Noise)", fontsize=14, fontweight='bold', color='#333333')
    ax.set_xlabel('Week', fontsize=12, fontweight='normal')
    ax.set_ylabel('Projected Rank (Lower = Worse)', fontsize=12, fontweight='normal')
    
    # Force integer ticks on X axis
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    ax.grid(True, axis='y', alpha=0.2, linestyle='--')
    
    # Legend in lower right to avoid blocking
    if i == 0:
        ax.legend(loc='lower right', frameon=True, framealpha=0.9)

plt.tight_layout()
plt.savefig("os.path.join(SA_DIR, "robustness_comparison.png")", dpi=150, bbox_inches='tight')
print("  Saved: robustness_comparison.png")

print("\nAnalysis Complete.")
