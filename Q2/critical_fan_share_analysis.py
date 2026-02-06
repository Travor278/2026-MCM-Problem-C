"""
临界粉丝份额分析 (Critical Fan Share Analysis)

对于每个争议选手，计算在不同规则下需要多少粉丝支持才能避免被淘汰。
这种方法避免了蒙特卡洛模拟的循环论证问题。

For each controversial contestant, calculate the minimum fan share needed
to avoid elimination under different rules. This avoids the circular
reasoning problem of Monte Carlo simulation.
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata
import os
import warnings
warnings.filterwarnings('ignore')

# 绝对路径配置
DATA_DIR = r"d:\2026mcmC\DataProcessed"
Q2_DIR = r"d:\2026mcmC\Q2"

# Load data
original = pd.read_csv(os.path.join(DATA_DIR, "DWTS_Processed_Long.csv"), encoding='utf-8-sig')

# 15 controversial contestants
controversial_contestants = [
    {"season": 27, "name": "Bobby Bones"},
    {"season": 11, "name": "Bristol Palin"},
    {"season": 2, "name": "Jerry Rice"},
    {"season": 4, "name": "Billy Ray Cyrus"},
    {"season": 19, "name": "Michael Waltrip"},
    {"season": 2, "name": "Master P"},
    {"season": 10, "name": "Niecy Nash"},
    {"season": 12, "name": "Romeo"},
    {"season": 23, "name": "Amber Rose"},
    {"season": 12, "name": "Wendy Williams"},
    {"season": 14, "name": "Jack Wagner"},
    {"season": 21, "name": "Alexa PenaVega"},
    {"season": 31, "name": "Joseph Baena"},
    {"season": 30, "name": "Olivia Jade"},
    {"season": 28, "name": "Sailor Brinkley-Cook"},
]


def compute_judge_contribution_rank(scores):
    """
    排名法下的裁判贡献。
    分数 → 排名 → 点数 → 归一化
    Rank-based judge contribution (S1-S2 style).
    """
    n = len(scores)
    # 高分 → 低排名数字 → 高点数
    ranks = rankdata(-scores, method='average')
    points = n - ranks + 1
    return points / points.sum()


def compute_judge_contribution_pct(scores):
    """
    百分比法下的裁判贡献。
    分数直接归一化。
    Percentage-based judge contribution (S3-S27 style).
    """
    return scores / scores.sum()


def compute_critical_fan_share(target_idx, judge_contributions, n_contestants):
    """
    计算目标选手避免被淘汰所需的最小粉丝份额。

    假设：其他选手的粉丝份额均匀分布。

    总分公式: T_i = 0.5 * J_i + 0.5 * f_i

    目标选手不被淘汰的条件：存在至少一个 j 使得 T_target > T_j

    Args:
        target_idx: 目标选手的索引
        judge_contributions: 所有选手的裁判贡献 (已归一化)
        n_contestants: 选手数量

    Returns:
        min_fan_share: 避免被淘汰所需的最小粉丝份额
        would_survive_uniform: 如果粉丝均匀分布(1/n)，是否存活
    """
    J_target = judge_contributions[target_idx]
    n = n_contestants

    # 找出裁判贡献最低的非目标选手
    other_J = [judge_contributions[i] for i in range(n) if i != target_idx]
    min_other_J = min(other_J)
    min_other_idx = [i for i in range(n) if i != target_idx and judge_contributions[i] == min_other_J][0]

    # 临界条件：T_target > T_min_other
    # 0.5 * J_target + 0.5 * f_target > 0.5 * J_min_other + 0.5 * f_min_other
    #
    # 假设其他选手粉丝份额均匀：f_j = (1 - f_target) / (n-1) for j ≠ target
    #
    # J_target + f_target > J_min_other + (1 - f_target) / (n-1)
    # f_target - (1 - f_target) / (n-1) > J_min_other - J_target
    # f_target * (1 + 1/(n-1)) - 1/(n-1) > J_min_other - J_target
    # f_target * n/(n-1) > J_min_other - J_target + 1/(n-1)
    # f_target > (J_min_other - J_target + 1/(n-1)) * (n-1) / n
    # f_target > (J_min_other - J_target) * (n-1)/n + 1/n

    delta_J = min_other_J - J_target  # 如果为正，说明目标选手裁判贡献更低
    critical_fan_share = delta_J * (n-1) / n + 1/n

    # 确保在有效范围内
    critical_fan_share = max(0, min(1, critical_fan_share))

    # 如果粉丝均匀分布，是否存活？
    uniform_share = 1 / n
    if J_target > min_other_J:
        # 裁判贡献不是最低，一定存活
        would_survive_uniform = True
    elif J_target == min_other_J:
        # 并列最低，可能被淘汰
        would_survive_uniform = False  # 假设并列时被淘汰
    else:
        # 裁判贡献最低
        would_survive_uniform = False

    return critical_fan_share, would_survive_uniform


def analyze_contestant_week(season, week, target_name):
    """
    分析特定选手在特定周的临界粉丝份额。
    """
    # 获取该周所有选手数据
    week_data = original[(original['season'] == season) & (original['week'] == week)].copy()

    if len(week_data) == 0:
        return None

    # 检查目标选手是否在该周
    if target_name not in week_data['celebrity_name'].values:
        return None

    contestants = week_data['celebrity_name'].values
    scores = week_data['total_judge_score'].values
    n = len(contestants)

    # 找到目标选手索引
    target_idx = np.where(contestants == target_name)[0][0]
    target_score = scores[target_idx]

    # 计算两种规则下的裁判贡献
    J_rank = compute_judge_contribution_rank(scores)
    J_pct = compute_judge_contribution_pct(scores)

    # 计算临界粉丝份额
    critical_rank, survive_uniform_rank = compute_critical_fan_share(target_idx, J_rank, n)
    critical_pct, survive_uniform_pct = compute_critical_fan_share(target_idx, J_pct, n)

    # 计算裁判分数在该周的排名
    score_rank = rankdata(-scores, method='average')[target_idx]

    return {
        'season': season,
        'week': week,
        'contestant': target_name,
        'n_contestants': n,
        'judge_score': target_score,
        'score_rank': int(score_rank),
        'is_lowest_score': score_rank == n,
        'J_rank': J_rank[target_idx],
        'J_pct': J_pct[target_idx],
        'critical_fan_share_rank': critical_rank,
        'critical_fan_share_pct': critical_pct,
        'survive_uniform_rank': survive_uniform_rank,
        'survive_uniform_pct': survive_uniform_pct,
        'protection_gap': critical_pct - critical_rank,  # 正数 = 百分比法更严格
    }


def main():
    print("临界粉丝份额分析 (Critical Fan Share Analysis)")
    print("\n分析15位争议选手在不同规则下需要的最低粉丝支持...")
    print("Analyzing minimum fan support needed for 15 controversial contestants...\n")

    all_results = []

    for contestant in controversial_contestants:
        season = contestant['season']
        name = contestant['name']

        # 获取该选手参赛的所有周
        contestant_data = original[
            (original['season'] == season) &
            (original['celebrity_name'] == name)
        ]

        if len(contestant_data) == 0:
            print(f"Warning: No data found for {name} (S{season})")
            continue

        weeks = contestant_data['week'].unique()

        for week in sorted(weeks):
            result = analyze_contestant_week(season, week, name)
            if result:
                all_results.append(result)

    # 转换为DataFrame
    results_df = pd.DataFrame(all_results)

    # 保存详细结果
    results_df.to_csv(os.path.join(Q2_DIR, "critical_fan_share_detailed.csv"), index=False)
    print(f"Saved detailed results: critical_fan_share_detailed.csv")
    print(f"Total records: {len(results_df)}\n")

    # 汇总分析

    print("关键发现 (Key Findings)")

    # 1. 按选手汇总
    print("\n【按选手汇总】Average by Contestant:")

    summary_by_contestant = results_df.groupby('contestant').agg({
        'season': 'first',
        'critical_fan_share_rank': 'mean',
        'critical_fan_share_pct': 'mean',
        'protection_gap': 'mean',
        'is_lowest_score': 'sum',
        'week': 'count'
    }).rename(columns={'week': 'weeks_competed', 'is_lowest_score': 'weeks_lowest_score'})

    summary_by_contestant = summary_by_contestant.sort_values('protection_gap', ascending=False)

    print(f"{'选手':<25} {'赛季':>4} {'排名法临界':>10} {'百分比法临界':>12} {'保护差距':>10} {'最低分周数':>10}")

    for name, row in summary_by_contestant.iterrows():
        print(f"{name:<25} S{int(row['season']):>2} "
              f"{row['critical_fan_share_rank']*100:>9.1f}% "
              f"{row['critical_fan_share_pct']*100:>11.1f}% "
              f"{row['protection_gap']*100:>+9.1f}% "
              f"{int(row['weeks_lowest_score']):>5}/{int(row['weeks_competed'])}")

    # 2. 最危险的周次
    print("\n\n【最危险周次】Most Dangerous Weeks (百分比法下需要最多粉丝支持):")

    dangerous_weeks = results_df.nlargest(10, 'critical_fan_share_pct')

    print(f"{'选手':<25} {'周':>4} {'裁判分排名':>10} {'排名法需要':>12} {'百分比法需要':>14} {'差距':>10}")

    for _, row in dangerous_weeks.iterrows():
        print(f"{row['contestant']:<25} W{int(row['week']):>2} "
              f"{int(row['score_rank']):>5}/{int(row['n_contestants'])} "
              f"{row['critical_fan_share_rank']*100:>11.1f}% "
              f"{row['critical_fan_share_pct']*100:>13.1f}% "
              f"{row['protection_gap']*100:>+9.1f}%")

    # 3. 规则差异最大的案例
    print("\n\n【规则差异最大】Largest Rule Differences (百分比法比排名法更严格):")

    biggest_gaps = results_df.nlargest(10, 'protection_gap')

    for _, row in biggest_gaps.iterrows():
        print(f"S{int(row['season'])} W{int(row['week'])} {row['contestant']}: "
              f"排名法需{row['critical_fan_share_rank']*100:.1f}%, "
              f"百分比法需{row['critical_fan_share_pct']*100:.1f}%, "
              f"差距 {row['protection_gap']*100:+.1f}%")

    # 4. Jerry Rice 详细分析
    print("\n\n【案例分析: Jerry Rice (S2)】")

    jerry_data = results_df[results_df['contestant'] == 'Jerry Rice']

    for _, row in jerry_data.iterrows():
        status = "[LOWEST]" if row['is_lowest_score'] else ""
        print(f"Week {int(row['week'])}: Rank {int(row['score_rank'])}/{int(row['n_contestants'])} {status}")
        print(f"    Rank-based: need >={row['critical_fan_share_rank']*100:.1f}% fan share")
        print(f"    Percentage: need >={row['critical_fan_share_pct']*100:.1f}% fan share")
        print(f"    Gap: {row['protection_gap']*100:+.1f}% (positive = pct rule stricter)")

        # Check if would survive under uniform distribution
        uniform = 100 / row['n_contestants']
        if row['critical_fan_share_pct'] > uniform / 100:
            print(f"    -> Under pct rule, if fans uniform ({uniform:.1f}%), would be eliminated!")
        print()

    # 5. Bobby Bones detailed analysis
    print("\n[Case Study: Bobby Bones (S27)]")

    bobby_data = results_df[results_df['contestant'] == 'Bobby Bones']

    for _, row in bobby_data.iterrows():
        status = "[LOWEST]" if row['is_lowest_score'] else ""
        print(f"Week {int(row['week'])}: Rank {int(row['score_rank'])}/{int(row['n_contestants'])} {status}")
        print(f"    Rank-based: need >={row['critical_fan_share_rank']*100:.1f}% fan share")
        print(f"    Percentage: need >={row['critical_fan_share_pct']*100:.1f}% fan share")
        print(f"    Gap: {row['protection_gap']*100:+.1f}%")
        print()

    # 6. 总结统计
    print("总结统计 (Summary Statistics)")

    # 计算平均保护差距
    avg_gap = results_df['protection_gap'].mean()
    print(f"\n平均保护差距: {avg_gap*100:+.2f}%")
    print("  (正数表示百分比法对低分选手更严格)")

    # 在最低分周的差距
    lowest_score_weeks = results_df[results_df['is_lowest_score']]
    if len(lowest_score_weeks) > 0:
        avg_gap_lowest = lowest_score_weeks['protection_gap'].mean()
        print(f"\n最低分周的平均保护差距: {avg_gap_lowest*100:+.2f}%")
        print(f"  这些周共有 {len(lowest_score_weeks)} 周")

    # 均匀分布下会被淘汰的周数
    uniform_shares = 1 / results_df['n_contestants']
    would_die_rank = (results_df['critical_fan_share_rank'] > uniform_shares).sum()
    would_die_pct = (results_df['critical_fan_share_pct'] > uniform_shares).sum()

    print(f"\n若粉丝均匀分布:")
    print(f"  排名法下会被淘汰的周数: {would_die_rank}/{len(results_df)}")
    print(f"  百分比法下会被淘汰的周数: {would_die_pct}/{len(results_df)}")

    print("结论 (Conclusion)")

    if avg_gap > 0:
        print("\n百分比法对低分选手更严格（需要更多粉丝支持才能存活）")
        print("排名法通过压缩分差，降低了低分选手的淘汰风险")
    else:
        print("\n排名法对低分选手更严格")

    print("\n这验证了配对比较分析的结论：")
    print("  - 排名法压缩分差 → 粉丝权重相对增大 → 低分选手更容易被救")
    print("  - 百分比法保留分差 → 评委权重更大 → 低分选手更难存活")


if __name__ == "__main__":
    main()
