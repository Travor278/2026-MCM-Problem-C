"""
DWTS Fan Vote Share - Visualization and Analysis
《与星共舞》粉丝投票份额可视化与分析

1. Consistency analysis (一致性分析)
2. Uncertainty analysis (不确定性分析)

Style: Nature journal, Times New Roman font
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.ticker as ticker

# NATURE STYLE CONFIGURATION
def setup_nature_style():
    """Configure matplotlib for Nature journal style."""
    plt.rcParams.update({
        # Font settings
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 8,

        # Figure settings
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',

        # Axes settings
        'axes.linewidth': 0.8,
        'axes.edgecolor': '#333333',
        'axes.labelcolor': '#333333',
        'axes.spines.top': False,
        'axes.spines.right': False,

        # Grid
        'axes.grid': False,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,

        # Ticks
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.color': '#333333',
        'ytick.color': '#333333',

        # Lines
        'lines.linewidth': 1.5,

        # Legend
        'legend.frameon': False,
        'legend.loc': 'best',

        # Unicode minus
        'axes.unicode_minus': False,
    })

# Color palette from user reference
COLORS = {
    'pink': '#FC757B',         # Coral pink
    'orange_red': '#F97F5F',   # Orange-red
    'orange': '#FAA26F',       # Orange
    'peach': '#FDCD94',        # Light peach
    'yellow': '#FEE199',       # Light yellow
    'green': '#B0D6A9',        # Light green
    'teal': '#65BDBA',         # Teal
    'blue': '#3C9BC9',         # Blue
    'neutral': '#6C757D',      # Gray
}

ERA_COLORS = {
    'rank_s1s2': '#3C9BC9',    # Blue - Era 1 (S1-S2)
    'percentage': '#FAA26F',   # Orange - Era 2 (S3-S27)
    'bottom2': '#FC757B',      # Coral pink - Era 3 (S28+)
}

ERA_LABELS = {
    'rank_s1s2': 'Era 1: Rank-Based (S1–S2)',
    'percentage': 'Era 2: Percentage (S3–S27)',
    'bottom2': 'Era 3: Bottom-Two (S28+)',
}

# DATA LOADING
def load_data():
    """Load all required datasets."""
    all_data = pd.read_csv(r'd:\2026mcmC\Q1\fan_vote_estimates.csv')
    validation = pd.read_csv(r'd:\2026mcmC\Q1\elimination_validation.csv')
    diagnostics = pd.read_csv(r'd:\2026mcmC\Q2\s1s2_mcmc_diagnostics.csv')
    return all_data, validation, diagnostics


# CONSISTENCY ANALYSIS VISUALIZATION
def analyze_consistency(validation, save_path=r'd:\2026mcmC\Q2\consistency_analysis.png'):
    """
    Create Nature-style consistency analysis figure.

    Layout: 2x2 grid
    (a) Consistency margin distribution by Era (violin + strip plot)
    (b) Summary statistics table/bar
    (c) Margin vs Season timeline
    (d) Fan share of eliminated contestants
    """
    setup_nature_style()

    fig = plt.figure(figsize=(7.5, 6.5))  # Nature single column ~89mm, double ~183mm

    # Create grid with custom spacing
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3,
                          left=0.10, right=0.95, top=0.92, bottom=0.08)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # Main title (bold)
    fig.suptitle('Model Self-Consistency Analysis', fontsize=12, fontweight='bold', y=0.98)

    # (a) Consistency Margin Distribution by Era - Violin + Box
    eras = ['rank_s1s2', 'percentage', 'bottom2']
    positions = [1, 2, 3]

    # Lighter colors for violin (to make boxplot more visible)
    VIOLIN_COLORS = {
        'rank_s1s2': '#65BDBA',    # Teal (lighter than blue)
        'percentage': '#FEE199',   # Light yellow
        'bottom2': '#FDCD94',      # Light peach
    }

    for i, era in enumerate(eras):
        era_data = validation[validation['rule_type'] == era]['consistency_margin'].values

        if len(era_data) > 0:
            # Violin plot (lighter color, lower alpha)
            parts = ax_a.violinplot([era_data], positions=[positions[i]],
                                    showmeans=False, showmedians=False, showextrema=False)
            for pc in parts['bodies']:
                pc.set_facecolor(VIOLIN_COLORS[era])
                pc.set_edgecolor(ERA_COLORS[era])
                pc.set_linewidth(0.8)
                pc.set_alpha(0.5)

            # Box plot overlay (darker, more prominent)
            bp = ax_a.boxplot([era_data], positions=[positions[i]], widths=0.18,
                             patch_artist=True, showfliers=False, zorder=4)
            bp['boxes'][0].set_facecolor(ERA_COLORS[era])
            bp['boxes'][0].set_edgecolor('#222222')
            bp['boxes'][0].set_linewidth(1.2)
            bp['boxes'][0].set_alpha(0.9)
            bp['medians'][0].set_color('white')
            bp['medians'][0].set_linewidth(2)
            for whisker in bp['whiskers']:
                whisker.set_color('#222222')
                whisker.set_linewidth(1.2)
            for cap in bp['caps']:
                cap.set_color('#222222')
                cap.set_linewidth(1.2)

            # Add individual points (jittered, smaller)
            jitter = np.random.uniform(-0.12, 0.12, len(era_data))
            ax_a.scatter(positions[i] + jitter, era_data,
                        c=ERA_COLORS[era], s=12, alpha=0.5, edgecolors='white',
                        linewidths=0.3, zorder=3)

    ax_a.set_xticks(positions)
    ax_a.set_xticklabels(['Era 1\n(S1–S2)', 'Era 2\n(S3–S27)', 'Era 3\n(S28+)'], fontsize=8)
    ax_a.set_ylabel('Consistency Margin')
    ax_a.set_title('(a) Margin Distribution by Era', fontsize=10, fontweight='normal', loc='left')
    ax_a.set_xlim(0.4, 3.6)
    ax_a.set_ylim(-0.05, validation['consistency_margin'].max() * 1.1)
    ax_a.axhline(y=0, color='#999999', linewidth=0.5, linestyle='--', zorder=1)

    # (b) Summary Statistics - Horizontal Bar Chart
    stats_data = []
    for era in eras:
        era_df = validation[validation['rule_type'] == era]
        n_total = len(era_df)
        n_consistent = era_df['is_consistent'].sum()
        mean_margin = era_df['consistency_margin'].mean()
        stats_data.append({
            'era': era,
            'n': n_total,
            'consistent': n_consistent,
            'rate': n_consistent / n_total * 100,
            'mean_margin': mean_margin
        })

    stats_df = pd.DataFrame(stats_data)
    y_pos = np.arange(len(eras))
    max_n = stats_df['n'].max()

    bars = ax_b.barh(y_pos, stats_df['n'], color=[ERA_COLORS[e] for e in eras],
                     edgecolor='#333333', linewidth=0.8, height=0.55)

    # Add text annotations (adjusted for visibility)
    for i, (idx, row) in enumerate(stats_df.iterrows()):
        # Right side: sample size and actual rate
        rate_str = f"{row['rate']:.1f}%" if row['rate'] < 100 else "100%"
        ax_b.text(row['n'] + 3, i, f"n={int(row['n'])}, {rate_str}",
                 va='center', ha='left', fontsize=8, color='#333333')
        # Inside bar: mean margin (only if bar is wide enough)
        if row['n'] > max_n * 0.15:
            ax_b.text(row['n']/2, i, f"μ={row['mean_margin']:.3f}",
                     va='center', ha='center', fontsize=7, color='white', fontweight='bold')
        else:
            # For short bars, put μ value outside on the right
            ax_b.text(row['n'] + 3, i + 0.22, f"μ={row['mean_margin']:.3f}",
                     va='center', ha='left', fontsize=7, color='#666666')

    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels([ERA_LABELS[e].split(':')[0] for e in eras], fontsize=9)
    ax_b.set_xlabel('Number of Weeks')
    ax_b.set_title('(b) Sample Size and Consistency Rate', fontsize=10, fontweight='normal', loc='left')
    ax_b.set_xlim(0, max_n * 1.45)
    ax_b.invert_yaxis()

    # (c) Margin Timeline by Season
    season_stats = validation.groupby('season').agg({
        'consistency_margin': ['mean', 'std', 'count'],
        'rule_type': 'first'
    }).reset_index()
    season_stats.columns = ['season', 'mean', 'std', 'n', 'rule_type']

    for era in eras:
        era_seasons = season_stats[season_stats['rule_type'] == era]
        ax_c.errorbar(era_seasons['season'], era_seasons['mean'],
                     yerr=era_seasons['std'], fmt='o-', color=ERA_COLORS[era],
                     markersize=4, capsize=2, capthick=0.8, linewidth=1,
                     label=ERA_LABELS[era].split(':')[0], elinewidth=0.8)

    # Add era boundary lines
    ax_c.axvline(x=2.5, color='#CCCCCC', linewidth=1, linestyle=':', zorder=1)
    ax_c.axvline(x=27.5, color='#CCCCCC', linewidth=1, linestyle=':', zorder=1)

    ax_c.set_xlabel('Season')
    ax_c.set_ylabel('Mean Margin ± SD')
    ax_c.set_title('(c) Consistency Margin by Season', fontsize=10, fontweight='normal', loc='left')
    ax_c.legend(fontsize=7, loc='upper left')
    ax_c.set_xlim(0, 34)

    # (d) Fan Share of Eliminated Contestants - Density/Histogram
    # Color intensity: Era 1 (darkest) > Era 3 (medium) > Era 2 (lightest)
    bins = np.linspace(0, 0.4, 20)

    # Custom colors and alpha for each era (Era1=dark, Era3=medium, Era2=light)
    hist_styles = {
        'rank_s1s2': {'color': '#3C9BC9', 'alpha': 0.85, 'edgecolor': '#2A7A9E', 'zorder': 5},  # Blue (darkest)
        'bottom2': {'color': '#FC757B', 'alpha': 0.70, 'edgecolor': '#D45A60', 'zorder': 4},    # Coral pink (medium)
        'percentage': {'color': '#FEE199', 'alpha': 0.60, 'edgecolor': '#E5C87A', 'zorder': 3}, # Light yellow (lightest)
    }

    # Plot in order: Era2 (back), Era3 (middle), Era1 (front)
    plot_order = ['percentage', 'bottom2', 'rank_s1s2']
    for era in plot_order:
        era_data = validation[validation['rule_type'] == era]['eliminated_fan_share']
        style = hist_styles[era]
        ax_d.hist(era_data, bins=bins,
                 alpha=style['alpha'],
                 color=style['color'],
                 edgecolor=style['edgecolor'],
                 linewidth=1.0,
                 zorder=style['zorder'],
                 label=f"{ERA_LABELS[era].split(':')[0]} (μ={era_data.mean():.3f})")

    ax_d.set_xlabel('Fan Vote Share of Eliminated Contestant')
    ax_d.set_ylabel('Frequency')
    ax_d.set_title('(d) Eliminated Contestants\' Fan Share', fontsize=10, fontweight='normal', loc='left')
    ax_d.legend(fontsize=7, loc='upper right')
    ax_d.set_xlim(0, 0.4)

    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"Saved: {save_path}")

    # Print statistics
    print("CONSISTENCY ANALYSIS SUMMARY")
    total = len(validation)
    consistent = validation['is_consistent'].sum()
    print(f"\nOverall Consistency: {consistent}/{total} = {consistent/total*100:.1f}%")
    print("\nBy Era:")
    for era in eras:
        era_df = validation[validation['rule_type'] == era]
        n, c = len(era_df), era_df['is_consistent'].sum()
        margin_mean = era_df['consistency_margin'].mean()
        margin_std = era_df['consistency_margin'].std()
        rate = c / n * 100
        print(f"  {ERA_LABELS[era]}: {c}/{n} ({rate:.1f}%), margin = {margin_mean:.4f} ± {margin_std:.4f}")


# UNCERTAINTY ANALYSIS VISUALIZATION
def analyze_uncertainty(all_data, diagnostics, save_path=r'd:\2026mcmC\Q2\uncertainty_analysis.png'):
    """Create Nature-style uncertainty analysis figure."""
    setup_nature_style()

    data = all_data.copy()
    data['hdi_width'] = data['fan_share_hdi_upper'] - data['fan_share_hdi_lower']

    fig = plt.figure(figsize=(7.5, 6.5))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3,
                          left=0.10, right=0.95, top=0.92, bottom=0.08)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    fig.suptitle('Estimation Uncertainty Analysis', fontsize=12, fontweight='bold', y=0.98)

    # (a) HDI Width by Season
    season_hdi = data.groupby(['season', 'rule_type'])['hdi_width'].mean().reset_index()

    for rule in ERA_COLORS.keys():
        rule_data = season_hdi[season_hdi['rule_type'] == rule]
        ax_a.bar(rule_data['season'], rule_data['hdi_width'],
                color=ERA_COLORS[rule], edgecolor='#333333', linewidth=0.5, width=0.8)

    ax_a.axvline(x=2.5, color='#999999', linewidth=1, linestyle=':', zorder=1)
    ax_a.axvline(x=27.5, color='#999999', linewidth=1, linestyle=':', zorder=1)
    ax_a.set_xlabel('Season')
    ax_a.set_ylabel('Mean HDI Width')
    ax_a.set_title('(a) Estimation Uncertainty by Season', fontsize=10, fontweight='normal', loc='left')

    # Legend
    legend_elements = [Patch(facecolor=ERA_COLORS[e], edgecolor='#333333', label=ERA_LABELS[e].split(':')[0])
                      for e in ERA_COLORS.keys()]
    ax_a.legend(handles=legend_elements, fontsize=7, loc='upper right')

    # (b) Uncertainty vs Number of Contestants
    week_data = data.groupby(['season', 'week']).agg({
        'celebrity_name': 'count', 'hdi_width': 'mean', 'rule_type': 'first'
    }).reset_index()
    week_data.columns = ['season', 'week', 'n_contestants', 'avg_hdi_width', 'rule_type']

    for rule in ERA_COLORS.keys():
        rule_weeks = week_data[week_data['rule_type'] == rule]
        ax_b.scatter(rule_weeks['n_contestants'], rule_weeks['avg_hdi_width'],
                    c=ERA_COLORS[rule], s=20, alpha=0.6, edgecolors='none')

    # Trend line (all data)
    z = np.polyfit(week_data['n_contestants'], week_data['avg_hdi_width'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(week_data['n_contestants'].min(), week_data['n_contestants'].max(), 100)
    ax_b.plot(x_line, p(x_line), '-', color='#333333', linewidth=2, alpha=0.8)

    # Correlation and equation in upper right with box
    corr = week_data['n_contestants'].corr(week_data['avg_hdi_width'])
    eq_text = f'$y = {z[0]:.3f}x + {z[1]:.2f}$\n$r = {corr:.3f}$'
    ax_b.text(0.97, 0.97, eq_text, transform=ax_b.transAxes,
             fontsize=8, ha='right', va='top',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='#CCCCCC', alpha=0.9))

    ax_b.set_xlabel('Number of Contestants')
    ax_b.set_ylabel('Mean HDI Width')
    ax_b.set_title('(b) Uncertainty vs. Contestant Count', fontsize=10, fontweight='normal', loc='left')

    # (c) Eliminated vs Survived Uncertainty
    eliminated_hdi = data[data['eliminated'] == True]['hdi_width']
    survived_hdi = data[data['eliminated'] == False]['hdi_width']

    bp = ax_c.boxplot([eliminated_hdi, survived_hdi],
                      positions=[1, 2], widths=0.5, patch_artist=True, showfliers=False)

    colors_box = [COLORS['pink'], COLORS['blue']]  # Eliminated=pink, Survived=blue
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_edgecolor('#333333')
        patch.set_alpha(0.8)
    for median in bp['medians']:
        median.set_color('white')
        median.set_linewidth(1.5)

    ax_c.set_xticks([1, 2])
    ax_c.set_xticklabels(['Eliminated', 'Survived'])
    ax_c.set_ylabel('HDI Width')
    ax_c.set_title('(c) Uncertainty by Elimination Status', fontsize=10, fontweight='normal', loc='left')

    # Add mean values
    ax_c.text(1, eliminated_hdi.median() + 0.01, f'μ={eliminated_hdi.mean():.4f}',
             ha='center', fontsize=7)
    ax_c.text(2, survived_hdi.median() + 0.01, f'μ={survived_hdi.mean():.4f}',
             ha='center', fontsize=7)

    # (d) MCMC Convergence (S1-S2)
    x = range(len(diagnostics))
    labels = [f"S{int(row['season'])}W{int(row['week'])}" for _, row in diagnostics.iterrows()]

    ax_d2 = ax_d.twinx()

    # ESS bars
    bars = ax_d.bar(x, diagnostics['ess_bulk_min'], color=COLORS['teal'],
                   alpha=0.75, edgecolor='#333333', linewidth=0.5, label='ESS (bulk)')
    ax_d.axhline(y=400, color=COLORS['orange_red'], linewidth=1, linestyle='--', label='ESS threshold')

    # R-hat line
    ax_d2.plot(x, diagnostics['r_hat_max'], 'o-', color=COLORS['pink'],
              markersize=4, linewidth=1.5, label='R-hat')
    ax_d2.axhline(y=1.05, color=COLORS['pink'], linewidth=1, linestyle=':', alpha=0.5)

    ax_d.set_xticks(x)
    ax_d.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax_d.set_ylabel('Effective Sample Size', color=COLORS['teal'])
    ax_d2.set_ylabel('R-hat', color=COLORS['pink'])
    ax_d.set_title('(d) MCMC Convergence (S1–S2)', fontsize=10, fontweight='normal', loc='left')

    ax_d.tick_params(axis='y', labelcolor=COLORS['teal'])
    ax_d2.tick_params(axis='y', labelcolor=COLORS['pink'])
    ax_d2.set_ylim(0.99, 1.08)

    # Combined legend
    lines1, labels1 = ax_d.get_legend_handles_labels()
    lines2, labels2 = ax_d2.get_legend_handles_labels()
    ax_d.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='upper right')

    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"Saved: {save_path}")

    # Print statistics
    print("UNCERTAINTY ANALYSIS SUMMARY")
    print(f"\nHDI Width: mean = {data['hdi_width'].mean():.4f}, std = {data['hdi_width'].std():.4f}")
    print(f"Correlation (contestants vs uncertainty): r = {corr:.4f}")
    print(f"\nMCMC Convergence (S1-S2):")
    print(f"  Max R-hat: {diagnostics['r_hat_max'].max():.4f}")
    print(f"  Min ESS: {diagnostics['ess_bulk_min'].min():.0f}")


# MAIN
if __name__ == "__main__":
    print("DWTS Fan Vote Share - Visualization (Nature Style)")

    all_data, validation, diagnostics = load_data()

    analyze_consistency(validation)
    analyze_uncertainty(all_data, diagnostics)

    print("All figures generated successfully.")
