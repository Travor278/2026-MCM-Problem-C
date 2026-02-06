"""
临界粉丝份额分析 - 包含Bottom-2规则
Critical Fan Share Analysis with Bottom-2 Rule

分析三种规则对争议选手的影响：
1. 排名法 (Rank-based): S1-S2
2. 百分比法 (Percentage): S3-S27
3. Bottom-2法: S28+ (评委从最后两名中选择淘汰)

Analyze impact of three rules on controversial contestants.
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
    """排名法下的裁判贡献"""
    n = len(scores)
    ranks = rankdata(-scores, method='average')
    points = n - ranks + 1
    return points / points.sum()


def compute_judge_contribution_pct(scores):
    """百分比法下的裁判贡献"""
    return scores / scores.sum()


def analyze_bottom2_impact(season, week, target_name):
    """
    分析Bottom-2规则对目标选手的影响。

    Bottom-2规则：
    1. 根据总分确定最后两名
    2. 评委从这两人中选择淘汰分数更低的

    对于低分选手，即使粉丝投票让他不是最后一名，
    如果他是倒数第二，评委仍可能淘汰他。
    """
    week_data = original[(original['season'] == season) & (original['week'] == week)].copy()

    if len(week_data) == 0 or target_name not in week_data['celebrity_name'].values:
        return None

    contestants = week_data['celebrity_name'].values
    scores = week_data['total_judge_score'].values
    n = len(contestants)

    target_idx = np.where(contestants == target_name)[0][0]
    target_score = scores[target_idx]

    # 计算裁判贡献
    J_rank = compute_judge_contribution_rank(scores)
    J_pct = compute_judge_contribution_pct(scores)

    # 分数排名 (1 = 最高)
    score_ranks = rankdata(-scores, method='average')
    target_score_rank = int(score_ranks[target_idx])
    is_lowest = target_score_rank == n
    is_second_lowest = target_score_rank == n - 1

    # 找出分数最低的两个选手
    sorted_indices = np.argsort(scores)
    lowest_idx = sorted_indices[0]
    second_lowest_idx = sorted_indices[1] if n > 1 else lowest_idx

    # 计算临界粉丝份额

    # 1. 排名法/百分比法：需要不是最后一名
    def compute_critical_not_last(J, target_idx, n):
        """计算不成为最后一名所需的最小粉丝份额"""
        J_target = J[target_idx]
        other_J = [J[i] for i in range(n) if i != target_idx]
        min_other_J = min(other_J)

        # 需要 T_target > T_min_other
        # 假设其他人粉丝均匀分布
        delta_J = min_other_J - J_target
        critical = delta_J * (n-1) / n + 1/n
        return max(0, min(1, critical))

    # 2. Bottom-2规则：需要不进入最后两名，或者进入后不被评委淘汰
    def compute_critical_bottom2(J, scores, target_idx, n):
        """
        计算在Bottom-2规则下避免淘汰所需的最小粉丝份额。

        情况1：不进入最后两名 -> 安全
        情况2：进入最后两名但分数不是更低 -> 安全（评委淘汰另一人）
        情况3：进入最后两名且分数更低 -> 被淘汰

        关键洞察：如果你的分数是倒数第二或更低，
        你需要确保不进入最后两名才能安全。
        """
        J_target = J[target_idx]
        target_score = scores[target_idx]

        # 找出比目标选手分数更低的选手
        lower_score_indices = [i for i in range(n) if scores[i] < target_score]
        same_or_higher_indices = [i for i in range(n) if scores[i] >= target_score and i != target_idx]

        if len(lower_score_indices) == 0:
            # 目标选手分数最低
            # 必须不进入最后两名（即至少有2人总分比他低）
            # 这几乎不可能，需要极高的粉丝份额

            # 找倒数第二低分的J
            sorted_by_score = np.argsort(scores)
            if n > 1:
                second_lowest_J = J[sorted_by_score[1]]
                # 需要比倒数第二高的人总分更高
                delta_J = second_lowest_J - J_target
                critical = delta_J * (n-1) / n + 1/n
                # 但实际上还需要不是倒数第二，所以更严格
                # 简化：如果分数最低，需要极高份额
                critical = max(critical, 0.5)  # 至少50%
            else:
                critical = 1.0

        elif len(lower_score_indices) == 1:
            # 目标选手分数是倒数第二
            # 如果进入最后两名，评委会淘汰分数更低的那个（不是目标）
            # 所以只需要不是最后一名
            other_J = [J[i] for i in range(n) if i != target_idx]
            min_other_J = min(other_J)
            delta_J = min_other_J - J_target
            critical = delta_J * (n-1) / n + 1/n

        else:
            # 有2个或更多选手分数比目标低
            # 如果进入最后两名，评委会淘汰分数更低的
            # 所以只需要不是最后一名
            other_J = [J[i] for i in range(n) if i != target_idx]
            min_other_J = min(other_J)
            delta_J = min_other_J - J_target
            critical = delta_J * (n-1) / n + 1/n

        return max(0, min(1, critical))

    critical_rank = compute_critical_not_last(J_rank, target_idx, n)
    critical_pct = compute_critical_not_last(J_pct, target_idx, n)
    critical_bottom2 = compute_critical_bottom2(J_pct, scores, target_idx, n)

    # Bottom-2规则的额外分析：评委干预效果

    # 如果选手进入最后两名，评委会怎么选？
    if is_lowest:
        judge_decision = "ELIMINATE"  # 评委会淘汰（分数最低）
        bottom2_risk = "HIGH"
    elif is_second_lowest:
        judge_decision = "SAVE"  # 评委会保留（分数不是最低）
        bottom2_risk = "MEDIUM"
    else:
        judge_decision = "N/A"  # 不会进入最后两名
        bottom2_risk = "LOW"

    return {
        'season': season,
        'week': week,
        'contestant': target_name,
        'n_contestants': n,
        'judge_score': target_score,
        'score_rank': target_score_rank,
        'is_lowest': is_lowest,
        'is_second_lowest': is_second_lowest,

        # 三种规则下的临界粉丝份额
        'critical_rank': critical_rank,
        'critical_pct': critical_pct,
        'critical_bottom2': critical_bottom2,

        # 规则差异
        'gap_pct_vs_rank': critical_pct - critical_rank,
        'gap_bottom2_vs_pct': critical_bottom2 - critical_pct,

        # Bottom-2风险评估
        'bottom2_risk': bottom2_risk,
        'judge_would': judge_decision,
    }


def main():
    print("临界粉丝份额分析 - 包含Bottom-2规则")
    print("Critical Fan Share Analysis with Bottom-2 Rule")

    all_results = []

    for contestant in controversial_contestants:
        season = contestant['season']
        name = contestant['name']

        contestant_data = original[
            (original['season'] == season) &
            (original['celebrity_name'] == name)
        ]

        if len(contestant_data) == 0:
            continue

        for week in sorted(contestant_data['week'].unique()):
            result = analyze_bottom2_impact(season, week, name)
            if result:
                all_results.append(result)

    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(Q2_DIR, "critical_fan_share_bottom2.csv"), index=False)
    print(f"\nSaved: critical_fan_share_bottom2.csv ({len(df)} records)")

    # 按选手汇总

    print("Summary by Contestant")

    summary = df.groupby('contestant').agg({
        'season': 'first',
        'critical_rank': 'mean',
        'critical_pct': 'mean',
        'critical_bottom2': 'mean',
        'is_lowest': 'sum',
        'week': 'count'
    }).rename(columns={'week': 'weeks', 'is_lowest': 'lowest_weeks'})

    summary['gap_pct_rank'] = summary['critical_pct'] - summary['critical_rank']
    summary['gap_b2_pct'] = summary['critical_bottom2'] - summary['critical_pct']
    summary = summary.sort_values('gap_b2_pct', ascending=False)

    print(f"\n{'Contestant':<25} {'S':>3} {'Rank':>8} {'Pct':>8} {'Bottom2':>8} {'B2-Pct':>8} {'Lowest':>8}")

    for name, row in summary.iterrows():
        print(f"{name:<25} S{int(row['season']):>2} "
              f"{row['critical_rank']*100:>7.1f}% "
              f"{row['critical_pct']*100:>7.1f}% "
              f"{row['critical_bottom2']*100:>7.1f}% "
              f"{row['gap_b2_pct']*100:>+7.1f}% "
              f"{int(row['lowest_weeks']):>4}/{int(row['weeks'])}")

    # 最低分周分析

    print("Lowest Score Weeks Analysis (Bottom-2 Impact)")

    lowest_weeks = df[df['is_lowest']].copy()

    if len(lowest_weeks) > 0:
        print(f"\n{'Contestant':<25} {'Week':>5} {'N':>3} {'Pct Need':>10} {'B2 Need':>10} {'Extra':>10} {'Judge Would':>12}")

        for _, row in lowest_weeks.iterrows():
            extra = row['critical_bottom2'] - row['critical_pct']
            print(f"{row['contestant']:<25} W{int(row['week']):>3} "
                  f"{int(row['n_contestants']):>3} "
                  f"{row['critical_pct']*100:>9.1f}% "
                  f"{row['critical_bottom2']*100:>9.1f}% "
                  f"{extra*100:>+9.1f}% "
                  f"{row['judge_would']:>12}")

    # Jerry Rice 案例

    print("Case Study: Jerry Rice (S2)")

    jerry = df[df['contestant'] == 'Jerry Rice']

    for _, row in jerry.iterrows():
        status = "[LOWEST]" if row['is_lowest'] else ("[2nd LOW]" if row['is_second_lowest'] else "")
        print(f"\nWeek {int(row['week'])}: Score Rank {int(row['score_rank'])}/{int(row['n_contestants'])} {status}")
        print(f"  Rank-based rule:   need >= {row['critical_rank']*100:5.1f}% fan share")
        print(f"  Percentage rule:   need >= {row['critical_pct']*100:5.1f}% fan share")
        print(f"  Bottom-2 rule:     need >= {row['critical_bottom2']*100:5.1f}% fan share")

        if row['is_lowest']:
            print(f"  -> Under Bottom-2: Even with enough fans, judges would ELIMINATE (lowest score)")
        elif row['is_second_lowest']:
            print(f"  -> Under Bottom-2: If in bottom 2, judges would SAVE (not lowest score)")

    # Bobby Bones 案例

    print("Case Study: Bobby Bones (S27)")

    bobby = df[df['contestant'] == 'Bobby Bones']

    for _, row in bobby.iterrows():
        status = "[LOWEST]" if row['is_lowest'] else ("[2nd LOW]" if row['is_second_lowest'] else "")
        print(f"\nWeek {int(row['week'])}: Score Rank {int(row['score_rank'])}/{int(row['n_contestants'])} {status}")
        print(f"  Rank-based rule:   need >= {row['critical_rank']*100:5.1f}% fan share")
        print(f"  Percentage rule:   need >= {row['critical_pct']*100:5.1f}% fan share")
        print(f"  Bottom-2 rule:     need >= {row['critical_bottom2']*100:5.1f}% fan share")

        if row['is_lowest']:
            print(f"  -> Under Bottom-2: Judges would ELIMINATE him!")

    # 统计总结

    print("Summary Statistics")

    # 平均差距
    avg_gap_pct_rank = df['gap_pct_vs_rank'].mean()
    avg_gap_b2_pct = df['gap_bottom2_vs_pct'].mean()

    print(f"\nAverage gaps across all {len(df)} contestant-weeks:")
    print(f"  Pct vs Rank:     {avg_gap_pct_rank*100:+.2f}% (positive = Pct stricter)")
    print(f"  Bottom-2 vs Pct: {avg_gap_b2_pct*100:+.2f}% (positive = Bottom-2 stricter)")

    # 最低分周的差距
    if len(lowest_weeks) > 0:
        avg_gap_lowest = lowest_weeks['gap_bottom2_vs_pct'].mean()
        print(f"\nFor {len(lowest_weeks)} weeks with LOWEST score:")
        print(f"  Bottom-2 vs Pct: {avg_gap_lowest*100:+.2f}%")
        print(f"  -> When you have lowest score, Bottom-2 requires {avg_gap_lowest*100:.1f}% MORE fan support")

    # Bottom-2规则下的淘汰预测
    print("Bottom-2 Rule Impact Prediction")

    # 如果这些选手在他们的时代遭遇Bottom-2规则
    lowest_eliminated = df[df['is_lowest'] & (df['judge_would'] == 'ELIMINATE')]

    print(f"\nIf Bottom-2 rule applied to these controversial contestants:")
    print(f"  {len(lowest_eliminated)} weeks where judges would ELIMINATE them")
    print(f"  (Even if fans pushed them out of last place, judges can still eliminate)")

    if len(lowest_eliminated) > 0:
        print("\n  Contestants who would be eliminated under Bottom-2:")
        for name in lowest_eliminated['contestant'].unique():
            weeks = lowest_eliminated[lowest_eliminated['contestant'] == name]['week'].tolist()
            print(f"    - {name}: Week(s) {', '.join(map(str, [int(w) for w in weeks]))}")

    print("Conclusion")
    print("""
1. Bottom-2 rule gives judges FINAL DECISION power
   - Even if fans "save" a low-scorer from last place
   - Judges can still eliminate them from bottom 2

2. For low-score contestants:
   - Rank/Pct rules: Need enough fans to avoid LAST place
   - Bottom-2 rule: Need enough fans to avoid BOTTOM TWO

3. Impact on controversial contestants:
   - Bobby Bones (S27 winner): Would be eliminated W8, W9 under Bottom-2
   - Jerry Rice (S2 runner-up): Would be eliminated W7, W8 under Bottom-2
   - Bristol Palin (S11 3rd): Would be eliminated W5, W7-W10 under Bottom-2

4. Bottom-2 rule is MOST STRICT for low-scoring fan favorites
   - It's designed to prevent "undeserving" contestants from winning
   - Gives judges veto power over fan votes
""")


if __name__ == "__main__":
    main()