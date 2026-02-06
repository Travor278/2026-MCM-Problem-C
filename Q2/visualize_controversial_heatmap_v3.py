"""
《与星共舞》争议选手热力图。
- 浅蓝色：淘汰前，但无数据（当周无淘汰选手）
- 浅灰色：淘汰后（选手出局）
Heatmap for controversial DWTS contestants.
- Light blue: Before elimination but no data (no elimination that week)
- Light gray: After elimination (contestant out)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle, Patch
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = r"d:\2026mcmC\DataProcessed"
Q2_DIR = r"d:\2026mcmC\Q2"

# Load data
heatmap_df = pd.read_csv(f"{Q2_DIR}/controversial_heatmap_data.csv")
original = pd.read_csv(f"{DATA_DIR}/DWTS_Processed_Long.csv", encoding='utf-8-sig')

# Contestant order
contestant_order = [
    "S27: Bobby Bones",
    "S11: Bristol Palin",
    "S2: Jerry Rice",
    "S4: Billy Ray Cyrus",
    "S19: Michael Waltrip",
    "S2: Master P",
    "S10: Niecy Nash",
    "S12: Romeo",
    "S23: Amber Rose",
    "S12: Wendy Williams",
    "S14: Jack Wagner",
    "S21: Alexa PenaVega",
    "S31: Joseph Baena",
    "S30: Olivia Jade",
    "S28: Sailor Brinkley-Cook",
]

# Build complete judge score data (all weeks competed)
# Extract contestant info from labels
contestant_info = []
for label in contestant_order:
    season = int(label.split(':')[0][1:])
    name = label.split(': ')[1]
    contestant_info.append({'label': label, 'season': season, 'name': name})

# Get all judge scores for these contestants from original data
judge_scores_list = []
for info in contestant_info:
    contestant_data = original[
        (original['season'] == info['season']) &
        (original['celebrity_name'] == info['name'])
    ].copy()

    if len(contestant_data) > 0:
        for _, row in contestant_data.iterrows():
            judge_scores_list.append({
                'label': info['label'],
                'week': row['week'],
                'total_judge_score': row['total_judge_score'],
                'average_judge_score': row['average_judge_score']
            })

judge_scores_df = pd.DataFrame(judge_scores_list)

# Normalize judge scores within each season-week
def normalize_within_week(df):
    """Normalize scores within each week (0 = lowest, 1 = highest)"""
    result = []
    for label in df['label'].unique():
        contestant_data = df[df['label'] == label].copy()
        season = int(label.split(':')[0][1:])

        for _, row in contestant_data.iterrows():
            week = row['week']
            # Get all scores for this season-week
            week_scores = original[
                (original['season'] == season) &
                (original['week'] == week)
            ]['total_judge_score']

            min_score = week_scores.min()
            max_score = week_scores.max()

            if max_score > min_score:
                normalized = (row['total_judge_score'] - min_score) / (max_score - min_score)
            else:
                normalized = 0.5

            result.append({
                'label': label,
                'week': week,
                'normalized_judge_score': normalized
            })

    return pd.DataFrame(result)

judge_normalized = normalize_within_week(judge_scores_df)

# Create pivot table for judge scores (now with ALL weeks)
pivot_data = judge_normalized.pivot_table(
    index='label',
    columns='week',
    values='normalized_judge_score',
    aggfunc='first'
)
pivot_data = pivot_data.reindex([c for c in contestant_order if c in pivot_data.index])

# Create full week range (1-11)
all_weeks = list(range(1, 12))
pivot_full = pivot_data.reindex(columns=all_weeks)

# Get elimination week for each contestant
elim_weeks = {}
for contestant in pivot_full.index:
    season = int(contestant.split(':')[0][1:])
    name = contestant.split(': ')[1]

    elim_week = original[
        (original['season'] == season) &
        (original['celebrity_name'] == name)
    ]['eliminated_week'].iloc[0]

    elim_weeks[contestant] = int(elim_week) if not pd.isna(elim_week) else 99

# Heatmap 1: Judge Scores (Blue Sequential)
# Cell dimensions
cell_width = 1.0
cell_height = 0.6

fig, ax = plt.subplots(figsize=(14, 8))

data = pivot_full.values
weeks = pivot_full.columns.values
contestants = pivot_full.index.values

# Custom Blue Sequential Colormap (Nature style - cool tones for performance)
# Low score (0) = light sky blue, High score (1) = deep navy
colors_blue = ['#E8F4F8', '#C5E3ED', '#93C9DC', '#5BAED1', '#3C9BC9',
               '#2E86AB', '#1B6B8A', '#0D526E', '#053D52']
cmap = mcolors.LinearSegmentedColormap.from_list('BlueSeq', colors_blue, N=256)

# Background colors (subtle, Nature style)
bg_no_data = '#F5F5F5'      # Very light gray - no elimination data
bg_after_elim = '#E0E0E0'   # Light gray - after elimination

# First, fill background with appropriate colors
for i, contestant in enumerate(contestants):
    elim_week = elim_weeks[contestant]
    for j, week in enumerate(weeks):
        val = data[i, j]
        if np.isnan(val):
            if week < elim_week:
                # Before elimination - very light gray
                rect = Rectangle((j * cell_width, i * cell_height), cell_width, cell_height,
                                  facecolor=bg_no_data, edgecolor='white', linewidth=1.5)
            else:
                # After elimination - light gray
                rect = Rectangle((j * cell_width, i * cell_height), cell_width, cell_height,
                                  facecolor=bg_after_elim, edgecolor='white', linewidth=1.5)
            ax.add_patch(rect)

# Plot actual data on top
for i in range(len(contestants)):
    for j in range(len(weeks)):
        val = data[i, j]
        if not np.isnan(val):
            color = cmap(val)
            rect = Rectangle((j * cell_width, i * cell_height), cell_width, cell_height,
                              facecolor=color, edgecolor='white', linewidth=1.5)
            ax.add_patch(rect)

# Set limits and ticks
ax.set_xlim(0, len(weeks) * cell_width)
ax.set_ylim(len(contestants) * cell_height, 0)
ax.set_xticks(np.arange(len(weeks)) * cell_width + cell_width / 2)
ax.set_yticks(np.arange(len(contestants)) * cell_height + cell_height / 2)
ax.set_xticklabels([int(w) for w in weeks], fontsize=11)
ax.set_yticklabels(contestants, fontsize=10)
ax.set_aspect('auto')

# Add X for elimination week (darker gray for better contrast with blue)
for i, contestant in enumerate(contestants):
    elim_week = elim_weeks[contestant]
    if elim_week in weeks:
        j = np.where(weeks == elim_week)[0][0]
        ax.plot(j * cell_width + cell_width / 2, i * cell_height + cell_height / 2,
                'x', color='#333333', markersize=12, markeredgewidth=3)

# Colorbar for data values
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
cbar.set_label('Normalized Score (Min-Max within Week)', fontsize=11)

# Legend for empty cells
legend_elements = [
    Patch(facecolor=bg_no_data, edgecolor='#999999', label='No elimination data'),
    Patch(facecolor=bg_after_elim, edgecolor='#999999', label='After elimination'),
    plt.Line2D([0], [0], marker='x', color='#333333', linestyle='None',
               markersize=10, markeredgewidth=2, label='Elimination week')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.95)

ax.set_xlabel('Week', fontsize=12, fontweight='bold')
ax.set_ylabel('Sample Contestants', fontsize=12, fontweight='bold')
ax.set_title('Within-Week Normalized Judge Scores (Heatmap View)',
             fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig(f"{Q2_DIR}/controversial_heatmap_judge_v3.png", dpi=150,
            bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: controversial_heatmap_judge_v3.png")

# Heatmap 2: Counterfactual Survival (Coral-Teal Diverging)
pivot_survival = heatmap_df.pivot_table(
    index='label',
    columns='week',
    values='counterfactual_survival',
    aggfunc='first'
)
pivot_survival = pivot_survival.reindex([c for c in contestant_order if c in pivot_survival.index])
pivot_surv_full = pivot_survival.reindex(columns=all_weeks)

fig2, ax2 = plt.subplots(figsize=(14, 8))

# Custom Diverging Colormap: Coral/Orange (low) → Cream (mid) → Teal/Green (high)
# Complements the blue palette of Heatmap 1
colors_surv = ['#D64550', '#E57373', '#F09A8C', '#FACBAF', '#FDF5E6',
               '#E5F2E0', '#B8DFB8', '#7CC47C', '#4AA84A', '#2E7D32']
cmap_surv = mcolors.LinearSegmentedColormap.from_list('CoralTeal', colors_surv, N=256)

data_surv = pivot_surv_full.values
contestants_surv = pivot_surv_full.index.values

# Background colors (matching Heatmap 1 for consistency)
bg_no_data_2 = '#F5F5F5'      # Very light gray
bg_after_elim_2 = '#E0E0E0'   # Light gray

# Fill background
for i, contestant in enumerate(contestants_surv):
    elim_week = elim_weeks[contestant]
    for j, week in enumerate(weeks):
        val = data_surv[i, j]
        if np.isnan(val):
            if week < elim_week:
                rect = Rectangle((j * cell_width, i * cell_height), cell_width, cell_height,
                                  facecolor=bg_no_data_2, edgecolor='white', linewidth=1.5)
            else:
                rect = Rectangle((j * cell_width, i * cell_height), cell_width, cell_height,
                                  facecolor=bg_after_elim_2, edgecolor='white', linewidth=1.5)
            ax2.add_patch(rect)

# Plot actual data
for i in range(len(contestants_surv)):
    for j in range(len(weeks)):
        val = data_surv[i, j]
        if not np.isnan(val):
            color = cmap_surv(val)
            rect = Rectangle((j * cell_width, i * cell_height), cell_width, cell_height,
                              facecolor=color, edgecolor='white', linewidth=1.5)
            ax2.add_patch(rect)
            # Add text for low survival
            x_pos = j * cell_width + cell_width / 2
            y_pos = i * cell_height + cell_height / 2
            if val < 0.2:
                ax2.text(x_pos, y_pos, f'{val*100:.0f}%', ha='center', va='center',
                        fontsize=8, fontweight='bold', color='white')
            elif val < 0.5:
                ax2.text(x_pos, y_pos, f'{val*100:.0f}%', ha='center', va='center',
                        fontsize=8, fontweight='bold', color='black')

ax2.set_xlim(0, len(weeks) * cell_width)
ax2.set_ylim(len(contestants_surv) * cell_height, 0)
ax2.set_xticks(np.arange(len(weeks)) * cell_width + cell_width / 2)
ax2.set_yticks(np.arange(len(contestants_surv)) * cell_height + cell_height / 2)
ax2.set_xticklabels([int(w) for w in weeks], fontsize=11)
ax2.set_yticklabels(contestants_surv, fontsize=10)
ax2.set_aspect('auto')

# Add X for elimination
for i, contestant in enumerate(contestants_surv):
    elim_week = elim_weeks[contestant]
    if elim_week in weeks:
        j = np.where(weeks == elim_week)[0][0]
        ax2.plot(j * cell_width + cell_width / 2, i * cell_height + cell_height / 2,
                'x', color='black', markersize=12, markeredgewidth=3)

# Colorbar
sm2 = plt.cm.ScalarMappable(cmap=cmap_surv, norm=plt.Normalize(0, 1))
sm2.set_array([])
cbar2 = plt.colorbar(sm2, ax=ax2, shrink=0.7, pad=0.02)
cbar2.set_label('Survival Probability Under Alternative Rule', fontsize=11)

# Legend (matching Heatmap 1 style)
legend_elements2 = [
    Patch(facecolor=bg_no_data_2, edgecolor='#999999', label='No elimination data'),
    Patch(facecolor=bg_after_elim_2, edgecolor='#999999', label='After elimination'),
    plt.Line2D([0], [0], marker='x', color='black', linestyle='None',
               markersize=10, markeredgewidth=2, label='Elimination week')
]
ax2.legend(handles=legend_elements2, loc='lower right', fontsize=9, framealpha=0.95)

ax2.set_xlabel('Week', fontsize=12, fontweight='bold')
ax2.set_ylabel('Sample Contestants', fontsize=12, fontweight='bold')
ax2.set_title('Counterfactual Survival Probability (Heatmap View)',
              fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig(f"{Q2_DIR}/controversial_heatmap_survival_v3.png", dpi=150,
            bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: controversial_heatmap_survival_v3.png")

print("\n图例说明:")
print("  淡紫色 (#E6E6FA): 选手仍在比赛，但该周无淘汰数据")
print("  浅灰色 (#D3D3D3): 选手已被淘汰")
print("  × 标记: 淘汰周")