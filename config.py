# -*- coding: utf-8 -*-
"""
DWTS 项目路径配置文件 / DWTS Project Path Configuration

使用方法 / Usage:
    from config import DATA_DIR, Q1_DIR, ...
"""

import os

# 基础路径 / Base Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 子目录路径 / Subdirectory Paths
DATA_DIR = os.path.join(BASE_DIR, "DataProcessed")
Q1_DIR = os.path.join(BASE_DIR, "Q1")
Q2_DIR = os.path.join(BASE_DIR, "Q2")
Q3_DIR = os.path.join(BASE_DIR, "Q3")
Q4_DIR = os.path.join(BASE_DIR, "Q4")
SA_DIR = os.path.join(BASE_DIR, "SensitiveAnalyse")
PAPER_DIR = os.path.join(BASE_DIR, "Paper")

# 常用数据文件路径 / Common Data File Paths
RAW_DATA_PATH = os.path.join(BASE_DIR, "2026_MCM_Problem_C_Data.csv")
PROCESSED_LONG_PATH = os.path.join(DATA_DIR, "DWTS_Processed_Long.csv")
FEATURES_PATH = os.path.join(DATA_DIR, "DWTS_Features.csv")

# Q1 输出文件
FAN_VOTE_ESTIMATES_PATH = os.path.join(Q1_DIR, "fan_vote_estimates.csv")
FAN_VOTE_S1S2_PATH = os.path.join(Q1_DIR, "fan_vote_s1s2_enhanced.csv")
FAN_VOTE_BOTTOM2_PATH = os.path.join(Q1_DIR, "fan_vote_bottom2.csv")
ELIMINATION_VALIDATION_PATH = os.path.join(Q1_DIR, "elimination_validation.csv")

# Q2 输出文件
CONTROVERSIAL_CASES_PATH = os.path.join(Q2_DIR, "controversial_cases.csv")
JUDGE_FAN_COMPARISON_PATH = os.path.join(Q2_DIR, "judge_vs_fan_comparison.csv")

# Q3 输出文件
PRO_PARTNER_EFFECTS_PATH = os.path.join(Q3_DIR, "pro_partner_effects.csv")
PRO_PARTNER_SURVIVAL_PATH = os.path.join(Q3_DIR, "pro_partner_survival.csv")

# Q4 输出文件
SIGMOID_GRID_SEARCH_PATH = os.path.join(Q4_DIR, "sigmoid_grid_search.csv")
PARETO_RESULTS_PATH = os.path.join(Q4_DIR, "pareto_results.csv")


def ensure_dir(path: str) -> str:
    """确保目录存在，不存在则创建"""
    os.makedirs(path, exist_ok=True)
    return path


def get_output_path(subdir: str, filename: str) -> str:
    """获取输出文件路径并确保目录存在"""
    dir_path = os.path.join(BASE_DIR, subdir)
    ensure_dir(dir_path)
    return os.path.join(dir_path, filename)


def check_paths():
    """检查关键路径是否存在"""
    paths_to_check = {
        "BASE_DIR": BASE_DIR,
        "DATA_DIR": DATA_DIR,
        "RAW_DATA": RAW_DATA_PATH,
        "PROCESSED_DATA": PROCESSED_LONG_PATH,
    }

    print("Path Check:")
    all_ok = True
    for name, path in paths_to_check.items():
        exists = os.path.exists(path)
        status = "OK" if exists else "NOT FOUND"
        print(f"  {name}: {path} -> {status}")
        if not exists:
            all_ok = False
    return all_ok


if __name__ == "__main__":
    check_paths()
    print(f"\nBASE_DIR = {BASE_DIR}")
