"""
《与星共舞》第一季至第二季专用贝叶斯模型

针对第一季和第二季的排名系统进行了增强采样。
采用切片采样器，增加抽样次数，以获得更好的收敛性。
DWTS S1-S2 Specialized Bayesian Model

Enhanced sampling for Seasons 1-2 rank-based system.
Uses Slice sampler with increased draws for better convergence.

排名法的建模简化说明：
题目原意：
    根据粉丝票数给选手排名，第1名得N分，第2名得N-1分...
    评委也按分数排名赋分
    两个排名积分相加，总分最低的淘汰

本代码的简化：
    评委分：正常按排名赋分（compute_judge_points_rank函数）
    粉丝分：直接用Dirichlet分布采样的"粉丝份额" f_i，没有做排名转换

为什么要简化？
  排名操作是离散的、不可微的。在贝叶斯MCMC采样中，我们需要对参数求梯度
  （或至少保持连续性），而 rank() 这种操作会破坏这一点。
  
  举例：f = [0.3, 0.31, 0.39] 排名后变成 [3, 2, 1]
        如果f稍微变成 [0.3, 0.32, 0.38]，排名还是 [3, 2, 1]
        但如果变成 [0.3, 0.35, 0.35]，排名就跳变成 [3, 1.5, 1.5]
  这种跳变采样器无法处理，导致采样失败或不收敛。

为什么简化是合理的？（单调性假设）
  排名赋分的本质是保序变换：份额越大 → 排名越好 → 积分越高
  即：如果 f_i > f_j，则 rank(f_i) < rank(f_j)，则 points(f_i) > points(f_j)
  
  所以粉丝份额的相对大小关系 ≡ 排名积分的相对大小关系
  
  在约束条件"被淘汰者总分最低"下：
    用份额直接加：judge_points_normalized + fan_shares
    用排名积分加：judge_rank_points + fan_rank_points
  两者推断出的粉丝份额分布在排序关系上是等价的。

数学表述：
  设 f_i 为选手i的粉丝份额（Dirichlet采样），j_i 为评委排名积分（已知）
  原始规则：总分 = rank_points(f_i) + j_i，最低者淘汰
  简化规则：总分 = f_i + j_i（归一化后），最低者淘汰
  
  由于 rank_points() 是单调变换，两种规则下"谁最低"的判断一致。

这是贝叶斯推断中常用的近似方法——用连续的份额代理离散的排名，
同时保持排序关系不变，使得MCMC采样可行。
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import pytensor.tensor as pt
from scipy.stats import rankdata
import warnings
warnings.filterwarnings('ignore')

# Enhanced MCMC parameters for S1-S2
N_SAMPLES = 5000
N_TUNE = 2000
N_CHAINS = 4
RANDOM_SEED = 42
CONSTRAINT_SCALE = 100.0


# C:      pd.DataFrame func(char* filepath) { ... }
# Python: def func(filepath: str) -> pd.DataFrame:
#
# def = 定义函数的关键字
# filepath: str = 参数filepath是字符串类型（冒号后面是类型提示）
# -> pd.DataFrame = 返回值是DataFrame（箭头后面是返回类型）
# 注意：Python的类型提示只是给IDE看的，不写也能跑

def load_and_preprocess_data(filepath: str) -> pd.DataFrame:
    # 读CSV，utf-8-sig是为了处理Excel导出时可能带的BOM头
    df = pd.read_csv(filepath, encoding='utf-8-sig')

    # 这些列应该是数字，但CSV读进来可能是字符串
    numeric_cols = ['season', 'week', 'total_judge_score', 'judge_percent',
                    'judge_rank', 'contestants_this_week', 'eliminated_week']
    
    # 强制转数字，转不了的变NaN（不会报错崩掉）
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 扔掉缺关键信息的行，后面建模没法用
    df = df.dropna(subset=['season', 'week', 'total_judge_score', 'celebrity_name'])
    
    # 过滤掉评委分0分的数据
    df = df[df['total_judge_score'] > 0].copy()
    
    return df


def normalize_judge_percent_df(df: pd.DataFrame) -> pd.DataFrame:
    # 复制，避免修改原数据
    df = df.copy()
    
    # 算每周所有选手的judge_percent总和
    group_sums = df.groupby(['season', 'week'])['judge_percent'].transform('sum')
    
    # 归一化
    df['judge_percent_normalized'] = df['judge_percent'] / group_sums
    
    # 如果某周总和是0（数据有问题），就平均分配
    df.loc[group_sums == 0, 'judge_percent_normalized'] = 1.0 / df.groupby(['season', 'week'])['celebrity_name'].transform('count')
    
    return df


# 找出这周被淘汰的选手
def identify_eliminated_contestant(week_data: pd.DataFrame, week: int):
    # 筛选出eliminated_week等于当前周的行
    eliminated = week_data[week_data['eliminated_week'] == week]
    if len(eliminated) == 1:
        # 正常情况：只有一个人被淘汰
        return eliminated['celebrity_name'].values[0]
    elif len(eliminated) > 1:
        # 特殊情况：双淘汰，返回列表
        return list(eliminated['celebrity_name'].values)
    # 没找到被淘汰的人，返回None
    return None


# 根据评委分数算排名分
# 分数最高的得N分，第二高得N-1分...以此类推
def compute_judge_points_rank(judge_scores: np.ndarray) -> np.ndarray:
    n = len(judge_scores)
    # rankdata(-scores)：分数取负是为了让高分排在前面
    # method='average'：同分的取平均排名
    ranks = rankdata(-judge_scores, method='average')
    # n - ranks + 1：把排名转成积分（第1名得n分，第2名得n-1分...）
    return n - ranks + 1


# 思路：已知谁被淘汰了，反推每个选手的粉丝投票份额
def build_model_rank_based(week_data: pd.DataFrame, eliminated_name: str):
    contestants = week_data['celebrity_name'].values
    judge_scores = week_data['total_judge_score'].values.astype(float)
    n = len(contestants)

    # 找被淘汰选手的索引
    eliminated_idx = np.where(contestants == eliminated_name)[0]
    if len(eliminated_idx) == 0:
        return None, None
    eliminated_idx = int(eliminated_idx[0])

    # 算评委积分并归一化（让总和=1，方便和粉丝份额相加）
    judge_points = compute_judge_points_rank(judge_scores)
    judge_points_normalized = judge_points / judge_points.sum()

    # 标记没被淘汰的选手
    survivor_mask = np.ones(n, dtype=bool)
    survivor_mask[eliminated_idx] = False
    survivor_indices = np.where(survivor_mask)[0].tolist()

    # 开始建PyMC模型
    with pm.Model() as model:
        # Dirichlet先验：粉丝投票份额，alpha全1表示无信息先验（均匀）
        alpha = np.ones(n)
        fan_shares = pm.Dirichlet('fan_shares', a=alpha)
        
        # 总分 = 评委分（归一化） + 粉丝份额
        total_score = pt.constant(judge_points_normalized) + fan_shares
        eliminated_score = total_score[eliminated_idx]

        # 约束1：被淘汰的人总分必须是最低的
        # 用sigmoid软约束，diff>0时constraint接近0，diff<0时惩罚很大
        for i, surv_idx in enumerate(survivor_indices):
            diff = total_score[surv_idx] - eliminated_score
            constraint = pm.math.log(pm.math.sigmoid(diff * CONSTRAINT_SCALE))
            pm.Potential(f'rank_constraint_{i}', constraint)

        # 约束2：平分时的tiebreaker，粉丝票少的被淘汰
        # is_tie：分差小于阈值时接近1，否则接近0
        for i, surv_idx in enumerate(survivor_indices):
            score_diff = pt.abs(total_score[surv_idx] - eliminated_score)
            is_tie = pm.math.sigmoid(-score_diff * 1000 + 5)
            fan_diff = fan_shares[surv_idx] - fan_shares[eliminated_idx]
            tie_constraint = is_tie * pm.math.log(pm.math.sigmoid(fan_diff * CONSTRAINT_SCALE))
            pm.Potential(f'tie_breaker_{i}', tie_constraint)

    return model, contestants


# 对某一周的数据跑MCMC采样
def fit_week_model(week_data: pd.DataFrame, season: int, week: int,
                   eliminated_name: str, verbose: bool = True):
    if verbose:
        print(f"  S{season}W{week}: N={len(week_data)}, Eliminated={eliminated_name}")

    try:
        model, contestants = build_model_rank_based(week_data, eliminated_name)

        if model is None:
            return None, None

        with model:
            # 用Slice采样器，比默认的NUTS更稳定
            trace = pm.sample(
                draws=N_SAMPLES,      # 采样次数
                tune=N_TUNE,          # 预热次数（调参用的，不计入结果）
                chains=N_CHAINS,      # 跑几条链（多条链可以检验收敛性）
                random_seed=RANDOM_SEED,
                progressbar=verbose,
                return_inferencedata=True,
                step=pm.Slice()
            )

        return trace, contestants

    except Exception as e:
        if verbose:
            print(f"    ERROR: {str(e)[:80]}")
        return None, None


# 从采样结果里提取粉丝份额的估计值
def extract_results(trace, contestants: np.ndarray) -> pd.DataFrame:
    # 拿出所有采样的fan_shares
    fan_shares_samples = trace.posterior['fan_shares'].values
    # reshape成 (采样数, 选手数)
    fan_shares_samples = fan_shares_samples.reshape(-1, len(contestants))

    results = []
    for i, name in enumerate(contestants):
        samples = fan_shares_samples[:, i]
        mean = np.mean(samples)
        # HDI: 95%最高密度区间，比普通置信区间更适合非对称分布
        hdi = az.hdi(samples, hdi_prob=0.95)
        results.append({
            'celebrity_name': name,
            'fan_share_mean': mean,
            'fan_share_hdi_lower': hdi[0],
            'fan_share_hdi_upper': hdi[1],
            'fan_share_std': np.std(samples)
        })
    return pd.DataFrame(results)


# 计算MCMC诊断指标，看采样有没有收敛
def compute_diagnostics(trace, contestants: np.ndarray) -> dict:
    summary = az.summary(trace, var_names=['fan_shares'])
    r_hat_values = summary['r_hat'].values      # R-hat：多链一致性，<1.05算收敛
    ess_bulk = summary['ess_bulk'].values       # ESS：有效样本量，越大越好
    ess_tail = summary['ess_tail'].values

    return {
        'r_hat_max': np.max(r_hat_values),
        'r_hat_mean': np.mean(r_hat_values),
        'ess_bulk_min': np.min(ess_bulk),
        'ess_tail_min': np.min(ess_tail),
        'converged': np.max(r_hat_values) < 1.05  # 判断是否收敛
    }


# 主流程：跑S1-S2所有周的分析
def run_s1s2_analysis(data_path: str, verbose: bool = True) -> tuple:
    print("DWTS S1-S2 Rank-Based Model (Enhanced Sampling)")
    print(f"\nMCMC: Draws={N_SAMPLES}, Tune={N_TUNE}, Chains={N_CHAINS}, Sampler=Slice\n")

    # 加载数据，只保留第1、2季
    df = load_and_preprocess_data(data_path)
    df = normalize_judge_percent_df(df)
    df = df[df['season'].isin([1, 2])]

    # 统计有多少个(赛季,周)组合
    season_weeks = df.groupby(['season', 'week']).size().reset_index(name='n_contestants')
    print(f"Total (season, week) combinations: {len(season_weeks)}\n")

    all_results = []
    all_diagnostics = []
    n_success = n_skipped = 0

    # 遍历每一周
    for _, row in season_weeks.iterrows():
        season, week = int(row['season']), int(row['week'])
        week_data = df[(df['season'] == season) & (df['week'] == week)].copy()

        # 少于2人没法比较，跳过
        if len(week_data) < 2:
            n_skipped += 1
            continue

        eliminated = identify_eliminated_contestant(week_data, week)
        if eliminated is None:
            n_skipped += 1
            continue
        # 双淘汰只取第一个（简化处理）
        if isinstance(eliminated, list):
            eliminated = eliminated[0]

        # 跑模型
        trace, contestants = fit_week_model(week_data, season, week, eliminated, verbose)

        if trace is None:
            continue

        # 提取结果
        week_results = extract_results(trace, contestants)
        week_results['season'] = season
        week_results['week'] = week
        week_results['eliminated'] = (week_results['celebrity_name'] == eliminated)

        # 记录诊断信息
        diag = compute_diagnostics(trace, contestants)
        diag['season'] = season
        diag['week'] = week
        diag['n_contestants'] = len(contestants)

        all_results.append(week_results)
        all_diagnostics.append(diag)
        n_success += 1

        if verbose:
            status = "Converged" if diag['converged'] else "NOT converged"
            print(f"    R-hat={diag['r_hat_max']:.3f}, ESS_bulk={diag['ess_bulk_min']:.0f} [{status}]\n")

    print(f"Complete: Success={n_success}, Skipped={n_skipped}")

    # 合并所有周的结果
    if all_results:
        results_df = pd.concat(all_results, ignore_index=True)
        cols = ['season', 'week', 'celebrity_name', 'fan_share_mean',
                'fan_share_hdi_lower', 'fan_share_hdi_upper', 'fan_share_std', 'eliminated']
        results_df = results_df[cols]
        diagnostics_df = pd.DataFrame(all_diagnostics)
        return results_df, diagnostics_df
    return pd.DataFrame(), pd.DataFrame()

# main入口
if __name__ == "__main__":
    import os

    # 绝对路径配置
    DATA_DIR = r"d:\2026mcmC\DataProcessed"
    Q1_DIR = r"d:\2026mcmC\Q1"

    DATA_PATH = os.path.join(DATA_DIR, "DWTS_Processed_Long.csv")
    OUTPUT_PATH = os.path.join(Q1_DIR, "fan_vote_s1s2_enhanced.csv")
    DIAGNOSTICS_PATH = os.path.join(Q1_DIR, "s1s2_mcmc_diagnostics.csv")

    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Data file not found at {DATA_PATH}")
        exit(1)

    results, diagnostics = run_s1s2_analysis(DATA_PATH, verbose=True)

    if len(results) > 0:
        results.to_csv(OUTPUT_PATH, index=False)
        print(f"\nResults saved to: {OUTPUT_PATH}")

        diagnostics.to_csv(DIAGNOSTICS_PATH, index=False)
        print(f"Diagnostics saved to: {DIAGNOSTICS_PATH}")

        print(f"\nConvergence: {diagnostics['converged'].sum()}/{len(diagnostics)} weeks")
        print(f"Max R-hat: {diagnostics['r_hat_max'].max():.4f}")
        print(f"Min ESS (bulk): {diagnostics['ess_bulk_min'].min():.0f}")
