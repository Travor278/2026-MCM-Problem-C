"""
可视化 S28+ 的底部 2 验证结果
Visualize Bottom-2 validation results for S28+
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
validation = pd.read_csv(r'd:\2026mcmC\Q2\bottom2_validation.csv')
results = pd.read_csv(r'd:\2026mcmC\Q1\fan_vote_bottom2.csv')
diagnostics = pd.read_csv(r'd:\2026mcmC\Q2\bottom2_mcmc_diagnostics.csv')

print("Bottom-2 Rule Validation Summary (S28+)")

# Summary statistics
print(f"\nTotal weeks analyzed: {len(validation)}")
print(f"All in Bottom 2: {validation['is_consistent'].sum()}/{len(validation)} = {validation['is_consistent'].mean()*100:.1f}%")

# n_lower distribution
print(f"\nn_lower distribution:")
for n in sorted(validation['n_lower'].unique()):
    count = (validation['n_lower'] == n).sum()
    print(f"  n_lower={n}: {count} weeks ({count/len(validation)*100:.1f}%)")

print(f"\nEliminated fan share: mean={validation['eliminated_fan_share'].mean():.4f}, "
      f"std={validation['eliminated_fan_share'].std():.4f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. n_lower distribution (bar chart)
n_lower_counts = validation['n_lower'].value_counts().sort_index()
colors = ['green' if n <= 1 else 'red' for n in n_lower_counts.index]
axes[0, 0].bar(n_lower_counts.index, n_lower_counts.values, color=colors, alpha=0.7, edgecolor='black')
axes[0, 0].set_xlabel('n_lower (survivors with lower score)', fontsize=11)
axes[0, 0].set_ylabel('Frequency', fontsize=11)
axes[0, 0].set_title('Bottom-2 Constraint: n_lower Distribution', fontsize=12)
axes[0, 0].set_xticks(range(max(n_lower_counts.index) + 1))
axes[0, 0].axvline(x=1.5, color='red', linestyle='--', label='Bottom-2 threshold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3, axis='y')

# 2. Eliminated fan share distribution
axes[0, 1].hist(validation['eliminated_fan_share'], bins=20, color='steelblue',
                alpha=0.7, edgecolor='black')
axes[0, 1].set_xlabel('Eliminated Contestant Fan Share', fontsize=11)
axes[0, 1].set_ylabel('Frequency', fontsize=11)
axes[0, 1].set_title('Fan Share of Eliminated Contestants (S28+)', fontsize=12)
axes[0, 1].axvline(x=validation['eliminated_fan_share'].mean(), color='red',
                   linestyle='--', label=f'Mean={validation["eliminated_fan_share"].mean():.3f}')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Fan share by season
season_stats = validation.groupby('season')['eliminated_fan_share'].agg(['mean', 'std'])
axes[1, 0].bar(season_stats.index, season_stats['mean'], yerr=season_stats['std'],
               color='coral', alpha=0.7, edgecolor='black', capsize=3)
axes[1, 0].set_xlabel('Season', fontsize=11)
axes[1, 0].set_ylabel('Mean Eliminated Fan Share', fontsize=11)
axes[1, 0].set_title('Eliminated Fan Share by Season', fontsize=12)
axes[1, 0].grid(True, alpha=0.3, axis='y')

# 4. Fan share over time (week progression within season)
for season in validation['season'].unique():
    season_data = validation[validation['season'] == season].sort_values('week')
    axes[1, 1].plot(season_data['week'], season_data['eliminated_fan_share'],
                    marker='o', label=f'S{season}', alpha=0.7)
axes[1, 1].set_xlabel('Week', fontsize=11)
axes[1, 1].set_ylabel('Eliminated Fan Share', fontsize=11)
axes[1, 1].set_title('Eliminated Fan Share Progression by Week', fontsize=12)
axes[1, 1].legend(loc='upper left', fontsize=8, ncol=2)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(r'd:\2026mcmC\Q2\bottom2_validation_analysis.png', dpi=200, bbox_inches='tight')
plt.show()
print("\nSaved: bottom2_validation_analysis.png")

# Additional analysis: Compare with original validation
print("Comparison with Original Model")

try:
    orig_validation = pd.read_csv(r'd:\2026mcmC\Q1\elimination_validation.csv')
    orig_b2 = orig_validation[orig_validation['rule_type'] == 'bottom2']

    print(f"\nOriginal model (S28+):")
    print(f"  n_lower=0: {(orig_b2['consistency_margin'] == 1).sum()} weeks")
    print(f"  n_lower=1: {(orig_b2['consistency_margin'] == 0).sum()} weeks")

    print(f"\nNew model (S28+):")
    print(f"  n_lower=0: {(validation['n_lower'] == 0).sum()} weeks")
    print(f"  n_lower=1: {(validation['n_lower'] == 1).sum()} weeks")

except FileNotFoundError:
    print("Original validation file not found for comparison")