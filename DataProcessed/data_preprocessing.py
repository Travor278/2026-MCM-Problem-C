# DWTS 数据预处理 - 2026 MCM Problem C
# 宽格式 -> 长格式，处理淘汰零分，计算排名与得分百分比

import pandas as pd
import numpy as np
import re
import os

# 绝对路径配置
BASE_DIR = r"d:\2026mcmC"
DATA_DIR = r"d:\2026mcmC\DataProcessed"


def load_data(filepath):
    """加载CSV并转换评分列为数值类型"""
    df = pd.read_csv(filepath, encoding='utf-8-sig')

    # 匹配 weekX_judgeY_score 格式的列
    score_cols = [c for c in df.columns if re.match(r'week\d+_judge\d+_score', c)]

    # 转换为数值，N/A等变成NaN
    for col in score_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df, score_cols


def parse_results(results_str, placement):
    """
    从results列解析最终名次和淘汰周次
    返回: (final_rank, eliminated_week)  eliminated_week为None表示需要推断
    """
    s = str(results_str).strip()

    if '1st Place' in s or 'Winner' in s:
        return 1, None
    elif '2nd Place' in s or 'Runner-up' in s:
        return 2, None
    elif '3rd Place' in s:
        return 3, None
    elif '4th Place' in s:
        return 4, None
    elif 'Eliminated Week' in s:
        match = re.search(r'Eliminated Week\s*(\d+)', s)
        if match:
            return placement, int(match.group(1))
    # Withdrew或其他情况，需要根据分数推断
    return placement, None


def infer_last_week(row, score_cols):
    """根据有效分数推断选手最后参赛的周次"""
    last_week = 1
    for col in score_cols:
        m = re.match(r'week(\d+)_judge\d+_score', col)
        if m and pd.notna(row[col]) and row[col] > 0:
            last_week = max(last_week, int(m.group(1)))
    return last_week


def process_elimination(df, score_cols):
    """解析淘汰信息，将淘汰后的0分替换为NaN"""

    # 解析results列
    parsed = df.apply(lambda r: parse_results(r['results'], r['placement']), axis=1)
    df['final_rank'] = [p[0] for p in parsed]
    df['eliminated_week'] = [p[1] for p in parsed]

    # 对eliminated_week为None的，根据实际分数推断
    for idx, row in df.iterrows():
        if pd.isna(row['eliminated_week']):
            df.at[idx, 'eliminated_week'] = infer_last_week(row, score_cols)

    df['eliminated_week'] = df['eliminated_week'].astype(int)

    # 淘汰后的0分 -> NaN
    for idx, row in df.iterrows():
        elim = row['eliminated_week']
        for col in score_cols:
            m = re.match(r'week(\d+)_judge\d+_score', col)
            if m and int(m.group(1)) > elim and df.at[idx, col] == 0:
                df.at[idx, col] = np.nan

    return df


def to_long_format(df, score_cols):
    """宽格式转长格式，每行=某选手某周的表现"""

    # 保留的基础信息列
    id_cols = ['celebrity_name', 'ballroom_partner', 'celebrity_industry',
               'celebrity_homestate', 'celebrity_homecountry/region',
               'celebrity_age_during_season', 'season', 'results', 'placement',
               'final_rank', 'eliminated_week']
    id_cols = [c for c in id_cols if c in df.columns]

    # 提取所有周次
    weeks = sorted(set(int(re.match(r'week(\d+)', c).group(1))
                       for c in score_cols if re.match(r'week(\d+)', c)))

    records = []
    for _, row in df.iterrows():
        for week in weeks:
            # 收集该周各评委分数（忽略NaN）
            scores = []
            judge_scores = {}
            for j in range(1, 5):
                col = f'week{week}_judge{j}_score'
                if col in df.columns:
                    val = row[col]
                    judge_scores[f'judge{j}_score'] = val
                    if pd.notna(val):
                        scores.append(val)

            # 跳过没有有效分数的周次
            if not scores:
                continue

            rec = {c: row[c] for c in id_cols}
            rec['week'] = week
            rec['total_judge_score'] = sum(scores)
            rec['average_judge_score'] = np.mean(scores)
            rec['num_judges'] = len(scores)
            rec.update(judge_scores)
            records.append(rec)

    df_long = pd.DataFrame(records)
    df_long = df_long.sort_values(['season', 'celebrity_name', 'week']).reset_index(drop=True)
    return df_long


def add_competition_metrics(df_long):
    """
    添加两种计分指标:
    - judge_rank: 当周排名 (1=最高分，min法处理并列)
    - judge_percent: 当周得分占比 = 个人分/当周总分
    """
    # 排名
    df_long['judge_rank'] = df_long.groupby(['season', 'week'])['total_judge_score'].rank(
        method='min', ascending=False
    )

    # 得分百分比
    week_sum = df_long.groupby(['season', 'week'])['total_judge_score'].transform('sum')
    df_long['judge_percent'] = df_long['total_judge_score'] / week_sum

    # 当周选手数
    df_long['contestants_this_week'] = df_long.groupby(['season', 'week'])['celebrity_name'].transform('count')

    return df_long


def validate(df_long):
    """验证：淘汰后不应有分数记录"""
    bad = df_long[df_long['week'] > df_long['eliminated_week']]
    if len(bad) > 0:
        print(f"[WARNING] {len(bad)} 条记录在淘汰周次之后仍有分数")
        print(bad[['season', 'celebrity_name', 'week', 'eliminated_week']].head())
    else:
        print("[OK] 数据验证通过")


def main():
    input_path = os.path.join(BASE_DIR, '2026_MCM_Problem_C_Data.csv')
    output_path = os.path.join(DATA_DIR, 'DWTS_Processed_Long.csv')

    print("Loading data...")
    df, score_cols = load_data(input_path)
    print(f"  {len(df)} contestants, seasons {df['season'].min()}-{df['season'].max()}")

    print("Processing elimination...")
    df = process_elimination(df, score_cols)

    print("Reshaping to long format...")
    df_long = to_long_format(df, score_cols)
    print(f"  {len(df_long)} records")

    print("Calculating metrics...")
    df_long = add_competition_metrics(df_long)

    validate(df_long)

    df_long.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Saved to {output_path}")

    return df_long


if __name__ == '__main__':
    df_processed = main()
