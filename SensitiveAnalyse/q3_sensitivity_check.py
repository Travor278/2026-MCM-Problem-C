"""
问题三补充：敏感性分析 (Sensitivity Analysis)
Temporal Sensitivity Check: Early Seasons vs Late Seasons

验证核心结论在不同时间段的稳健性：
1. 年龄效应 (Age Effect) 是否始终为负？
2. 模特劣势 (Model Penalty) 是否始终显著？
3. 职业舞伴效应 (Partner ICC) 是否随时间变化？
"""

import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# 绝对路径配置
DATA_DIR = r"d:\2026mcmC\DataProcessed"
SA_DIR = r"d:\2026mcmC\SensitiveAnalyse"

# Academic style
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

print("Sensitivity Analysis: Temporal Robustness Check")

# Load data
features = pd.read_csv(os.path.join(DATA_DIR, "DWTS_Features.csv"))
features = features.dropna(subset=['score_zscore', 'Industry_Group', 'celebrity_age_during_season'])

# Split data: Early (S1-S15) vs Late (S16-S31)
# Season 15 was All-Stars, a good dividing line
early_era = features[features['season'] <= 15].copy()
late_era = features[features['season'] > 15].copy()

print(f"Early Era (S1-S15): {len(early_era)} records")
print(f"Late Era (S16-S31): {len(late_era)} records")

# Define LMM formula (Using C() for categorical to match previous analysis)
formula = "score_zscore ~ celebrity_age_during_season + C(Industry_Group) + week"

results = []

def run_lmm(data, label):
    # Fit LMM
    model = smf.mixedlm(formula, data, groups=data["ballroom_partner"])
    fit = model.fit()
    
    # Extract coefficients
    age_coef = fit.params['celebrity_age_during_season']
    age_p = fit.pvalues['celebrity_age_during_season']
    age_ci = fit.conf_int().loc['celebrity_age_during_season']
    
    # Extract Model coefficient (Name might be C(Industry_Group)[T.Model])
    model_key = 'C(Industry_Group)[T.Model]'
    if model_key in fit.params:
        model_coef = fit.params[model_key]
        model_p = fit.pvalues[model_key]
        model_ci = fit.conf_int().loc[model_key]
    else:
        model_coef = np.nan
        model_p = np.nan
        model_ci = [np.nan, np.nan]
    
    # Calculate ICC
    var_re = fit.cov_re.iloc[0, 0]
    var_resid = fit.scale
    icc = var_re / (var_re + var_resid)
    
    return {
        'Era': label,
        'Age_Coef': age_coef,
        'Age_CI_Low': age_ci[0],
        'Age_CI_High': age_ci[1],
        'Age_P': age_p,
        'Model_Coef': model_coef,
        'Model_CI_Low': model_ci[0],
        'Model_CI_High': model_ci[1],
        'Model_P': model_p,
        'ICC': icc
    }

# Run models
results.append(run_lmm(features, 'Full Dataset'))
results.append(run_lmm(early_era, 'Early (S1-S15)'))
results.append(run_lmm(late_era, 'Late (S16-S31)'))

res_df = pd.DataFrame(results)
print("\n[Sensitivity Results]")
print(res_df[['Era', 'Age_Coef', 'Age_P', 'Model_Coef', 'Model_P', 'ICC']].round(4))

# Save results text
with open("c:/Users/34886/OneDrive/桌面/2026mcmC/q3_sensitivity_summary.txt", 'w') as f:
    f.write(res_df.to_string())

# Visualization: Enhanced Panel Plot (Publication Ready)
print("\n--- Generating Enhanced Sensitivity Plot ---")

import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

# Data preparation
eras = res_df['Era']
x_pos = np.arange(len(eras))
colors = ['#65BDBA', '#3C9BC9', '#FC757B'] # Teal (Full), Blue (Early), Red (Late)
markers = ['D', 'o', 's'] # Diamond, Circle, Square

# Helper function to plot with style (Dumbbell for Panel 1 & 2)
def plot_dumbbell(ax, data, ci_low, ci_high, title, ylabel):
    # Dumbbell: Line connecting Early (idx 1) and Late (idx 2)
    # We ignore Full (idx 0) for the line, or keep it as reference?
    # Usually Dumbbell compares two states. Let's focus on Early vs Late.
    
    # 1. Plot Early -> Late Arrow
    # Coordinates
    early_val = data[1]
    late_val = data[2]
    
    # Draw line
    ax.plot([1, 2], [early_val, late_val], color='gray', linewidth=2, zorder=1, linestyle='--')
    
    # Add arrow head (optional, tricky with simple plot, stick to line)
    
    # 2. Plot Points & Error Bars
    for i in range(len(data)): # Plot Full, Early, Late
        # Error bar
        ax.plot([i, i], [ci_low[i], ci_high[i]], color=colors[i], linewidth=2, alpha=0.6)
        # Cap
        ax.plot([i-0.1, i+0.1], [ci_low[i], ci_low[i]], color=colors[i], linewidth=2, alpha=0.6)
        ax.plot([i-0.1, i+0.1], [ci_high[i], ci_high[i]], color=colors[i], linewidth=2, alpha=0.6)
        
        # Point
        ax.scatter(i, data[i], color=colors[i], s=150, marker=markers[i], zorder=5, edgecolors='white', linewidth=1.5)
        
        # Text Label
        offset = (max(data) - min(data)) * 0.15 if (max(data) - min(data)) > 0 else 0.01
        text_y = ci_high[i] + offset if data[i] < 0 else ci_high[i] + offset
        ax.text(i, text_y, f"{data[i]:.3f}", ha='center', va='bottom', fontsize=11, fontweight='bold', color=colors[i])

    # Add annotation for change
    change = late_val - early_val
    mid_x = 1.5
    mid_y = (early_val + late_val) / 2
    ax.text(mid_x, mid_y + 0.01, f"Δ = {change:+.3f}", ha='center', va='bottom', fontsize=10, color='gray', style='italic')

    # Styling
    ax.set_xticks(x_pos)
    ax.set_xticklabels(eras, fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    sns.despine(ax=ax)

# Helper function for Bar (ICC) - kept same
def plot_bar(ax, data, title, ylabel):
    for i in range(len(data)):
        ax.bar(i, data[i], color=colors[i], alpha=0.7, width=0.5, edgecolor='black')
        ax.text(i, data[i] + 0.005, f"{data[i]:.3f}", ha='center', va='bottom', fontsize=11, fontweight='bold', color=colors[i])
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(eras, fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_ylim(0, 0.25)
    sns.despine(ax=ax)

# 1. Age Effect (Dumbbell)
plot_dumbbell(axes[0], res_df['Age_Coef'], res_df['Age_CI_Low'], res_df['Age_CI_High'], 
           'Age Effect (Coefficient)', 'Impact on Z-Score')

# 2. Model Effect (Dumbbell)
plot_dumbbell(axes[1], res_df['Model_Coef'], res_df['Model_CI_Low'], res_df['Model_CI_High'], 
           'Model Industry Penalty', 'Impact on Z-Score')

# 3. ICC (Bar)
plot_bar(axes[2], res_df['ICC'], 'Partner Influence (ICC)', 'Variance Explained')

plt.tight_layout()
plt.savefig("os.path.join(SA_DIR, "sensitivity_check_enhanced.png")", dpi=300, bbox_inches='tight')
plt.close()
print("Saved: sensitivity_check_enhanced.png")

print("Sensitivity Analysis Complete!")
