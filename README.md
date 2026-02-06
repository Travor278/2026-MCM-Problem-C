# 2026 MCM Problem C: Data With The Stars

**[English Version](README_EN.md)** | 中文版

> **队伍编号**: 2622622
> **比赛成员**: ysk、Aurora、Travor
> **比赛时间**: 2026.01.30——2026.02.03

---

## 项目概述

本项目是2026年美国大学生数学建模竞赛(MCM) Problem C的完整解决方案。我们分析了《与星共舞》(Dancing with the Stars, DWTS)节目34季的数据，研究评委评分与粉丝投票的结合机制，并设计了一套优化的投票系统。

### 核心贡献

1. **贝叶斯粉丝投票估计模型** - 利用MCMC采样从淘汰结果反推粉丝投票份额
2. **争议案例量化分析** - 识别并模拟15名争议选手在不同规则下的存活概率
3. **多模型特征分析** - LMM + XGBoost/SHAP + Cox生存分析三角验证
4. **动态权重优化系统** - Sigmoid函数 + Bottom-2混合机制

![团队工作流程](Paper/Ourwork.jpg)

---

## 问题背景

### 节目规则演变

DWTS节目中，明星与职业舞者搭档每周表演，评委打分(1-10分)与粉丝投票结合决定淘汰人选。历史上使用过三种计分方式：

| 时期 | 赛季 | 计分方式 | 特点 |
|------|------|----------|------|
| Era 1 | S1-S2 | 排名法 | 评委分和粉丝票分别排名后相加 |
| Era 2 | S3-S27 | 百分比法 | 评委分百分比 + 粉丝票百分比 |
| Era 3 | S28+ | Bottom-2法 | 综合得分最低两人由评委投票决定淘汰 |

### 经典争议案例

- **Bobby Bones (S27)**: 评委分数始终垫底，却赢得冠军
- **Jerry Rice (S2)**: 5周评委最低分，仍获得亚军
- **Bristol Palin (S11)**: 12次评委最低分，最终获得季军
- **Billy Ray Cyrus (S4)**: 6周评委最低分，排名第5

---

## 目录结构

```
2026mcmC/
├── 2026_MCM_Problem_C_Data.csv    # 原始数据 (34季)
│
├── DataProcessed/                  # 数据预处理模块
│   ├── data_preprocessing.py       # 宽格式→长格式转换
│   ├── DWTS_Processed_Long.csv     # 处理后长格式数据
│   └── DWTS_Features.csv           # 特征工程数据
│
├── Q1/                             # 问题1: 贝叶斯粉丝投票估计
│   ├── bayesian_fan_vote_model.py  # 主模型控制器 (S3-S27)
│   ├── bayesian_s1s2_rank_model.py # S1-S2排名法专用模型
│   ├── bayesian_bottom2_model.py   # S28+ Bottom-2专用模型
│   ├── fan_vote_estimates.csv      # S3-S27粉丝份额估计
│   ├── fan_vote_s1s2_enhanced.csv  # S1-S2粉丝份额估计
│   ├── fan_vote_bottom2.csv        # S28+粉丝份额估计
│   └── elimination_validation.csv  # 一致性验证结果
│
├── Q2/                             # 问题2: 计分规则比较与争议分析
│   ├── find_controversial_cases.py # 争议选手识别算法
│   ├── analyze_judge_vs_fan_mechanism.py  # 评委vs粉丝机制对比
│   ├── compare_scoring_methods.py  # 排名法vs百分比法比较
│   ├── critical_fan_share_analysis.py     # 临界粉丝份额分析
│   ├── critical_fan_share_bottom2.py      # 含Bottom-2的临界分析
│   ├── visualize_controversial_heatmap_v3.py  # 反事实热力图
│   └── judge_vs_fan_comparison.csv # 机制对比结果
│
├── Q3/                             # 问题3: 名人特征与舞伴影响
│   ├── q3_lmm_xgboost_analysis.py  # LMM + XGBoost + SHAP
│   ├── q3_cox_survival_analysis.py # Cox比例风险模型
│   ├── q3_effect_difference_analysis.py  # Bootstrap效应差异检验
│   ├── q3_sensitivity_check.py     # 时间稳健性检验
│   ├── pro_partner_effects.csv     # 舞伴随机效应
│   └── pro_partner_survival.csv    # 舞伴生存统计
│
├── Q4/                             # 问题4: 最优投票系统设计
│   ├── q4_dynamic_weight_model.py  # Sigmoid动态权重模型
│   ├── q4_pareto_optimization.py   # 多目标Pareto优化
│   ├── q4_historical_validation.py # 历史案例验证
│   ├── q4_sensitivity_analysis.py  # 参数敏感性分析
│   └── sigmoid_grid_search.csv     # 网格搜索结果
│
├── SensitiveAnalyse/               # 敏感性分析汇总
│
├── Paper/                          # 论文LaTeX文件
│   ├── math_model_part2.tex        # 数学模型公式推导
│   ├── dwts_algorithm_pseudocode.tex  # 27个算法伪代码
│   ├── memo.tex                    # 1页制片人备忘录
│   └── ai_use_report.tex           # AI使用报告
│
└── README.md                       # 本文件
```

---

## 核心方法论

### Q1: 贝叶斯粉丝投票份额估计

#### 数学建模

设第$s$季第$w$周有$N$名选手，定义：
- $J_i$: 选手$i$的评委总分 (已知)
- $\theta_i$: 选手$i$的粉丝投票份额 (待估计)
- $E$: 被淘汰选手

粉丝份额满足**单纯形约束**：
$$\sum_{i=1}^{N} \theta_i = 1, \quad \theta_i \geq 0$$

#### 先验分布

- **Era 1-2 (无信息先验)**: $\boldsymbol{\theta} \sim \text{Dirichlet}(\mathbf{1}_N)$
- **Era 3 (信息先验)**: $\alpha_i = 0.5 + 5.0 \cdot \frac{r^J(i) - 1}{N - 1}$

#### 似然函数 (软约束)

**Era 1-2**: 被淘汰者综合得分最低
$$\log \mathcal{L} = \sum_{k \neq E} \log \sigma(\lambda \cdot (S_k - S_E)), \quad \lambda = 100$$

**Era 3 (Bottom-2)**: 被淘汰者在最后两名中
$$\log \mathcal{L} = -20 \cdot \max(0, \widehat{N_{\text{lower}}} - 1.3)^2$$

#### MCMC采样参数

| 参数 | 标准 (S3-S27) | 增强 (S1-S2) | Bottom-2 (S28+) |
|------|---------------|--------------|-----------------|
| 采样次数 | 2000 | 5000 | 3000 |
| 预热次数 | 1000 | 2000 | 1500 |
| 链数 | 2 | 4 | 4 |
| 采样器 | NUTS | Slice | Slice |

#### 验证指标

- **一致性率**: 模型预测与实际淘汰结果一致的比例 (>95%)
- **收敛诊断**: $\hat{R} < 1.05$, ESS > 400

#### S1-S2 排名法的建模技巧

**原始规则**：粉丝票数排名 → 第1名得N分，第2名得N-1分... → 与评委排名积分相加 → 最低者淘汰

**建模难点**：排名操作 `rank()` 是离散的、不可微的，MCMC 采样器需要连续梯度

**简化策略**：用连续的粉丝份额 $f_i$ 直接代替离散的排名积分

**为什么合理？（单调性保证）**

排名赋分本质是**保序变换**：
$$f_i > f_j \Rightarrow \text{rank}(f_i) < \text{rank}(f_j) \Rightarrow \text{points}(f_i) > \text{points}(f_j)$$

因此，在"谁的总分最低"这一判断上：
- 原始规则：$\text{Total}_i = \text{rank\_points}(f_i) + j_i$
- 简化规则：$\text{Total}_i = f_i + j_i$（归一化后）

两者的**相对大小关系完全一致**，推断出的粉丝份额分布等价。

**示例**：
```
f = [0.30, 0.31, 0.39]  →  排名 [3, 2, 1]  →  积分 [1, 2, 3]
f = [0.30, 0.32, 0.38]  →  排名 [3, 2, 1]  →  积分 [1, 2, 3]  (微小变化，排名不变)
f = [0.30, 0.35, 0.35]  →  排名 [3, 1.5, 1.5]  →  积分跳变！(采样器无法处理)
```

这是贝叶斯推断中常用的**连续近似**方法——保持排序关系不变，使 MCMC 采样可行。

---

### Q2: 计分规则比较与争议案例分析

#### 争议选手识别算法

$$\text{Controversy Score} = \frac{W_{\text{bottom}}}{R_{\text{final}}} + \frac{W_{\text{bottom2}}}{2 \cdot R_{\text{final}}}$$

其中 $W_{\text{bottom}}$ 为评委最低分周数，$R_{\text{final}}$ 为最终排名。

#### 反事实生存模拟

蒙特卡洛模拟 ($M = 10000$次)：
1. 从后验分布采样粉丝份额 $\tilde{f}_i \sim \mathcal{N}(\hat{f}_i, \sigma_i^2)$
2. 计算综合得分 $T_i = 0.5 \cdot J_i + 0.5 \cdot \tilde{f}_i$
3. 统计目标选手存活次数

#### 核心发现

| 对比维度 | 排名法 | 百分比法 | 差异 |
|----------|--------|----------|------|
| 被淘汰者平均粉丝份额 | 8.2% | 11.5% | +3.3% |
| 粉丝影响力 | 较低 | 较高 | 百分比法更利于粉丝 |
| 评委-粉丝机制相关性 | - | - | r = -0.320 |

---

### Q3: 名人特征与舞伴影响分析

#### 线性混合效应模型 (LMM)

$$y_{ig} = \mathbf{x}_i^\top \boldsymbol{\beta} + u_g + \varepsilon_{ig}$$

- 固定效应: 年龄、行业、周次、赛季进度
- 随机效应: 舞伴 $u_g \sim \mathcal{N}(0, \sigma_u^2)$

**组内相关系数 (ICC)**:
$$\rho = \frac{\sigma_u^2}{\sigma_u^2 + \sigma^2}$$

| 目标变量 | ICC | 含义 |
|----------|-----|------|
| 评委分数 | 18.5% | 舞伴解释18.5%的评委分方差 |
| 粉丝份额 | 7.0% | 舞伴对粉丝影响较小 |

#### XGBoost + SHAP特征重要性

| 排名 | 评委分数驱动因素 | 粉丝投票驱动因素 |
|------|------------------|------------------|
| 1 | 周次 (Week) | 行业 (Industry) |
| 2 | 赛季进度 | 周次 |
| 3 | 年龄 | 年龄 |
| 4 | 行业 | 赛季进度 |

#### Cox比例风险模型

$$h(t|\mathbf{x}) = h_0(t) \exp(\boldsymbol{\beta}^\top \mathbf{x})$$

| 因素 | 风险比 (HR) | 95% CI | 解释 |
|------|-------------|--------|------|
| 年龄 (+10岁) | 1.15 | [1.08, 1.23] | 年龄越大，淘汰风险越高 |
| 运动员 vs 演员 | 0.82 | [0.71, 0.95] | 运动员淘汰风险较低 |
| 模特 vs 演员 | 1.34 | [1.12, 1.60] | 模特淘汰风险较高 |

---

### Q4: 最优投票系统设计

#### Sigmoid动态权重函数

$$w(t) = w_{\min} + \frac{w_{\max} - w_{\min}}{1 + \exp(-k \cdot t_{\text{norm}})}$$

其中 $t_{\text{norm}} = \frac{t - t_{\text{mid}}}{t_{\max}/4}$

**最优参数 (网格搜索)**:
- $w_{\min} = 0.20$ (前期评委权重)
- $w_{\max} = 0.60$ (后期评委权重)
- $k = 1.90$ (变化速率)

#### 多目标Pareto优化

三个目标函数：
1. **公平性 (Fairness)**: 淘汰者评委排名越低越好
2. **参与度 (Engagement)**: 粉丝投票对结果的边际影响
3. **避免惨案 (No-Robbery)**: 避免高分选手被淘汰

**膝点分析**: Kneedle算法确定最优权重 $w^* = 0.55$

#### 推荐系统: 动态权重 + Week-7 Bottom-2

```
Early (Week 1-4):  w = 20-30%  → 保护"黑马"
Mid   (Week 5-6):  w = 35-45%  → 渐进过渡
Late  (Week 7+):   w = 50-60% + Bottom-2  → 确保冠军实力
```

#### 给制片人的备忘录

基于以上分析，我们为 DWTS 节目制片人撰写了一份建议备忘录：

![制片人备忘录](Paper/memo.png)

---

## 运行环境

### Python依赖

```bash
# 核心依赖
pip install numpy pandas scipy matplotlib seaborn

# 贝叶斯模型
pip install pymc arviz pytensor

# 统计建模
pip install statsmodels

# 机器学习
pip install xgboost shap scikit-learn

# 生存分析
pip install lifelines
```

### 完整安装

```bash
cd d:\2026mcmC
pip install -r requirements.txt
```

### 运行顺序

```bash
# 1. 数据预处理
python DataProcessed/data_preprocessing.py

# 2. Q1: 贝叶斯粉丝投票估计 (耗时较长)
python Q1/bayesian_s1s2_rank_model.py      # S1-S2
python Q1/bayesian_fan_vote_model.py       # S3-S27
python Q1/bayesian_bottom2_model.py        # S28+

# 3. Q2: 争议分析
python Q2/find_controversial_cases.py
python Q2/analyze_judge_vs_fan_mechanism.py

# 4. Q3: 特征分析
python Q3/q3_lmm_xgboost_analysis.py
python Q3/q3_cox_survival_analysis.py

# 5. Q4: 系统优化
python Q4/q4_pareto_optimization.py
python Q4/q4_dynamic_weight_model.py
```

---

## 主要结论

### 1. 计分规则比较

| 规则 | 优点 | 缺点 | 推荐场景 |
|------|------|------|----------|
| 排名法 | 简单直观 | 放大小差距 | 早期赛季 |
| 百分比法 | 粉丝影响力大 | 可能产生争议 | 中期赛季 |
| Bottom-2 | 平衡性最好 | 增加复杂度 | 后期赛季 |

### 2. 推荐方案效果

| 指标 | 现行系统 | 推荐系统 | 改进 |
|------|----------|----------|------|
| 惨案率 | 42% | 8% | -34% |
| 粉丝参与度 | 基准 | +22% | 显著提升 |
| 公平性 | 基准 | 96% | 大幅改善 |

### 3. 给制片人的建议

1. **保留百分比计分法** - 维持投票悬念和粉丝参与度
2. **采用动态评委权重** - 前期低(30%)保护黑马，后期高(60%)确保冠军
3. **从第7周启用Bottom-2** - 创造戏剧性同时防止极端结果
4. **考虑投票积分制** - 多次投票同一选手需消耗更多积分

---

## 算法清单

本项目包含**27个核心算法**，详见 `Paper/dwts_algorithm_pseudocode.tex`:

| 编号 | 算法名称 | 所属问题 |
|------|----------|----------|
| 1 | 数据预处理 | 基础 |
| 2-5 | 贝叶斯MCMC模型 | Q1 |
| 6-7 | 一致性/不确定性分析 | Q1 |
| 8-12 | 计分规则比较与热力图 | Q2 |
| 13-18 | LMM/XGBoost/Cox分析 | Q3 |
| 19-24 | Pareto优化与动态权重 | Q4 |
| 25-27 | 敏感性分析 | 验证 |

---

## 文件说明

### 输入数据

| 文件 | 描述 | 记录数 |
|------|------|--------|
| `2026_MCM_Problem_C_Data.csv` | 原始宽格式数据 | 424名选手 |

### 中间数据

| 文件 | 描述 |
|------|------|
| `DWTS_Processed_Long.csv` | 长格式 (每行=选手×周) |
| `DWTS_Features.csv` | 特征工程后数据 |
| `fan_vote_*.csv` | 各时期粉丝份额估计 |

### 输出结果

| 文件 | 描述 |
|------|------|
| `*_comparison.csv` | 规则比较结果 |
| `pro_partner_effects.csv` | 舞伴效应 |
| `sigmoid_grid_search.csv` | 参数优化结果 |
| `*.png` | 可视化图片 |

---

## 反思与改进

### 本次比赛的经验教训

#### 1. 文档管理
- **问题**: 论文各部分分散在多个 LaTeX 文件中，版本同步困难
- **改进**:
  - 使用 **Endnote** 管理参考文献
  - 使用 **腾讯文档** 进行初稿协作编辑
  - 自建 **Overleaf 社区版**（虚拟机部署），实现 LaTeX 实时协作
  - 使用 **GitHub 私有仓库** 进行版本控制，支持分支管理和代码审查

#### 2. 代码规范
- **问题**: 初期代码使用相对路径，文件整理后路径失效
- **改进**:
  - 所有 Python 文件统一使用 **绝对路径** 配置
  - 建议项目初期就建立 `config.py` 统一管理路径

#### 3. 使用 Jupyter Notebook 改善代码框架
- **问题**: 纯 Python 脚本调试周期长，数据探索和可视化需要反复运行整个程序
- **改进**:
  - **分模块开发**: 将数据加载、预处理、建模、可视化拆分为独立的 Notebook Cell，便于逐步调试
  - **交互式探索**: 使用 Jupyter 的即时输出特性，快速查看中间变量和数据分布
  - **可视化调优**: 在 Notebook 中实时调整图表参数（颜色、字体、布局），无需重跑整个脚本
  - **Markdown 文档**: 在代码旁添加说明文字和公式推导，形成自解释的分析报告
  - **版本管理**: 配合 `nbstripout` 工具清理输出，避免 git diff 混乱
  - **环境隔离**: 建议使用虚拟环境 + `ipykernel` 注册内核，确保依赖一致性

**推荐工作流**:
```
1. 新功能开发 → Jupyter Notebook 快速原型
2. 功能稳定后 → 提取核心逻辑到 .py 模块
3. 最终集成 → 主脚本调用模块，Notebook 保留探索记录
```

#### 4. 模型验证
- **问题**: MCMC 收敛诊断未在早期充分关注，导致部分结果需重跑
- **改进**:
  - 建立标准化的收敛检查流程 ($\hat{R} < 1.05$, ESS > 400)
  - 使用 ArviZ 自动生成诊断报告

#### 5. 时间分配
- **问题**: 前期数据清洗耗时过多，后期优化模块时间紧张
- **改进**:
  - 提前熟悉数据格式，准备通用预处理模板
  - 为每个问题预留充足的敏感性分析时间

#### 6. 审题与模块划分
- **问题**: 审题不仔细，将第三问的特征分析（名人特征对表现的影响）提前与第一问一起做
- **后果**:
  - **循环论证风险**：Q1 用淘汰结果推断粉丝份额，Q3 用粉丝份额分析特征影响，若混在一起容易出现"用结果推结果"
  - **论文结构混乱**：方法论和结果的边界模糊，难以清晰表述
- **改进**:
  - 开题前仔细通读所有子问题，理清**数据流向**和**因果链条**
  - 明确每个问题的**输入**和**输出**，避免跨问题依赖
  - 按问题顺序逐步推进，前一问的输出作为后一问的输入

### 未来可扩展方向

| 方向 | 描述 | 优先级 |
|------|------|--------|
| 深度学习 | 使用 LSTM/Transformer 预测粉丝投票趋势 | 中 |
| 实时系统 | 构建投票结果实时监控仪表盘 | 低 |
| A/B 测试框架 | 设计不同计分规则的对照实验 | 高 |
| 社交媒体分析 | 整合 Twitter/Instagram 情感数据 | 中 |

### 代码质量改进清单

- [x] 统一使用绝对路径
- [x] 添加详细中英文注释
- [x] 创建 `.gitignore` 文件
- [ ] 添加单元测试 (`pytest`)
- [ ] 使用 `requirements.txt` 锁定依赖版本
- [ ] 添加类型注解 (`typing`)
- [ ] 使用 `logging` 替代 `print` 语句
- [ ] 将重复代码抽象为公共模块

### AI 工具使用反思

本项目使用了 AI 辅助工具，详见 `Paper/ai_use_report.tex`：

**有效使用场景**:
- 代码调试与错误排查
- LaTeX 公式格式化
- 文档翻译与润色

**需谨慎的场景**:
- 核心算法设计（需人工验证正确性）
- 统计结论解读（AI 可能过度自信）
- 创新点提炼（需结合领域知识）

---

## 写在最后

> *"I came, I divided, I conquered!"*

四天四夜。窗外的月亮圆了又缺，屏幕前的咖啡凉了又热。

我们在数据的海洋里打捞真相，在公式的迷宫中寻找出口。有过凌晨三点的崩溃，也有过收敛成功时的欢呼。代码报错时互相甩锅，结果跑通时击掌相庆——这大概就是建模的浪漫。

> *"We are all in the gutter, but some of us are looking at the stars."*

也许结果不尽如人意，但那又怎样？我们曾仰望同一片星空，追逐同一个目标。过程本身，就是意义。

> *"The journey is the reward."*

**Let's set a bit and flow!**

---

## 许可证

本项目仅用于2026 MCM学术竞赛目的。

---

## 团队信息

- **队伍编号**: 2622622
- **比赛**: 2026 Mathematical Contest in Modeling (MCM)
- **题目**: Problem C - Data With The Stars

---

*本README最后更新于2026年2月*
