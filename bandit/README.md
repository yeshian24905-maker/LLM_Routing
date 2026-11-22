# FT-UCB 离线路由实验

实现论文 [Appendix H, Algorithm 6](https://users.ece.cmu.edu/~oyagan/Conferences/ICLR2025.pdf#page=54)，
以当前 `result.csv` 中的 **3000 道题、4 个模型（12000 条已有回答）**做离线回放，
每道题先选一个模型，再读取该题该模型的已有回答；每个阈值独立运行 3000 轮。
不调用模型 API。四个阈值为 **89%、85%、80%、82.5%**。
需要 Python 3.10+ 和 Matplotlib。

## 使用

### 比较三种置信界改法

```powershell
python -m bandit.compare --empirical-costs --repeats 30 --horizons 3000 10000 --thresholds 0.89 0.85 0.80 0.825 --seed 42
```

保留原始 FT-UCB 作为基线，新增：

- 收紧 UCB：`mean + sqrt(0.5 * ln(t) / n)`。
- Bernoulli KL-UCB：求 `n * KL(mean, q) <= ln(t) + c * ln(ln(t))` 的最大 `q`；测试实用参数 `c=0`，并以 `c=3` 作敏感性对照。
- LCB：仅把原始置信界的加号改成减号，`mean - sqrt(2 * ln(t) / n)`。

所有方法仍然使用 `bound >= mu0` 的候选集合、最低成本选择、前四轮初始化以及候选集为空时的均匀随机回退。
LCB 没有偷偷改成“总选最高准确率模型”。每个策略只用本轮选中模型的已有结果更新自身。

3000 轮实验每题只使用一次；10000 轮实验从相同的 3000 道题中有放回抽样，**不是新增 10000 道题或新增模型回答**。
所有方法共享每个种子的题目顺序和反馈评分，默认 30 组配对种子。均值、标准差和配对增益区间只反映这个固定数据集上的顺序/抽样随机性，不代表新题的泛化置信区间。

该命令在运行期间阻断 socket 连接，不调用模型 API。输入 CSV 和归档评分保持不变。
结果输出到 `bandit/output/confidence-comparison/`：`runs.csv`、`aggregates.csv`、每 50 轮的均值曲线 `curves.csv`、首个种子的完整逐轮记录 `primary_trace.csv` 以及溯源信息 `summary.json`。
英文 PDF 输出到 `artifacts/Confidence_Methods_Comparison.pdf`；PDF 源代码使用 UTF-8，并嵌入字体。

本比较不宣称修改后的成本约束策略自动继承原论文或标准最大化奖励 KL-UCB 的全部保证；这里只验证实现并报告离线表现。

### 运行原始算法

使用当前的完整 CSV，并复用其中已经归档、可核对的评分：

```powershell
python -m bandit --results-csv result.csv --scores-csv bandit/output/ft-ucb/scored_results.csv --expected-prompts 3000 --empirical-costs --repeats 1 --seed 42 --output-dir bandit/output/ft-ucb-verified
```

该模式从 `result.csv` 核对归档子集的每个原始字段，仅允许换行符差异，保留原来的评分和人工复核标记。
不重新下载题目、不调用模型 API。输出目录必须与归档评分输入分开。
**题目集合由完整的 `result.csv` 决定，不会被旧的 1000 题归档缩减。**
其余 2000 道题按 CSV 标准答案自动评分；开放题和无法识别选项的回答保留人工复核标记。
`--scores-csv` 可省略，此时全部回答重新评分；原 CSV 不含选项文本，数字/文本形式的选择题答案无法映射时计错并标记复核。
`--expected-prompts` 是可选的题数校验，CSV 模式省略时使用全部题目。

`mu0` 是每一轮构造候选集合 `UCB >= mu0` 时使用的固定输入阈值。
最终正确与否由数据集的标准答案决定；输出中的最终准确率是否达到 `mu0` 仅为描述性统计，不参与选臂。

原始 MMMU 文件夹模式也保留；未指定 `--results-csv` 时默认读取项目根目录的 `mmmu_test_sample/`：

```powershell
python -m pip install -r bandit/requirements.txt
python -m bandit --empirical-costs
```

默认每个阈值只做一次实验，每轮只选择一个模型，种子 42。上述完整 CSV 命令是四次独立的 **3000 轮实验**。
原始文件夹模式的默认题数仍为 1000，默认输出到 `bandit/output/ft-ucb/`；当前 CSV 命令另外输出到 `bandit/output/ft-ucb-verified/`。
`--repeats 30` 可增加重复回放；`--seed`、`--output-dir`、`--thresholds` 可覆盖默认设置。
数据位置变化时使用 `--data-dir <目录>`；`--run <名称>` 选择该目录 `runs/` 中的其他采集批次。

必须明确选择一种成本来源：

- `--empirical-costs`：以各模型已知 CSV 费用的均值作为固定成本，缺失项不参与计算。
  这使用全量费用元数据，是对论文“成本事先已知”假设的离线近似，报告会注明。
- `--arm-costs arm_costs.json`：使用事先给定的固定美元成本。JSON 的键必须是 CSV 中
  四个模型的完整名称，值为非负有限数值。

成本向量在回放期间保持不变。缺失的实际费用不会被替换为零。

## 题目来源与评分

```text
mmmu_test_sample/
├── questions.jsonl         # 决定要处理的 1000 道题，提供标准答案
├── dataset_manifest.json  # 题目数量、文件校验信息
├── images/                # 原始题目图片
└── runs/sequential/
    ├── run_manifest.json  # 校验回答是在同一份题目上采集的
    └── results.csv        # 四个模型已收集的回答，仅作结果查询及事后评估
```

加载器检查题目文件哈希、题目数量、图片是否存在，以及采集批次是否对应同一份题目。
结果表的题号、学科、题型和标准答案必须与原始题目一致；缺少题目或模型结果会报错。

CSV 必须包含：

```text
question_id,subject,question_type,model,raw_response,correct_answer,cost,error
```

每题每个模型恰好一条已有记录；当前 CSV 共 12000 条回答记录，**它们不是 12000 个决策轮次**。
当前每个阈值只处理 3000 轮，每轮只用所选模型的结果更新 UCB。
加载器拒绝重复记录和无效费用。其他题量可用 `--expected-prompts N`。
`question_type` 为 `multiple-choice` 或 `open`。

评分优先提取最后一个完整 `<answer>...</answer>`。选择题识别选项字母、`C. 选项文字`，
或与原始题目某个选项唯一匹配的答案值（例如 `147`）；不使用标准答案来猜选项，也不把 `B, C` 当作单选。
开放题支持备选答案、
文本归一化与数字保留两位小数比较。空回答、解析失败和请求错误计 0，
`accounting_error:` 不影响已有回答的评分。评分参考 MMMU，但不等价于官方评估器。
开放题及回退解析标记 `needs_review=1`，正式报告前应复核。

`--labels reviewed_labels.csv` 可用人工标签覆盖评分，格式为 `question_id,model,correct`，
其中 `correct` 为 0 或 1；重复键和未知记录会报错。原始 CSV 不会被修改。

## 算法与指标

模型按固定成本升序编号，相同成本按模型名排序：

```text
前 K 轮依次选择每个模型一次。
C_t = {m: mean_m + sqrt(2 * ln(t) / N_m) >= mu0}
若 C_t 非空，选择其中成本最低的模型；否则从所有模型均匀随机选择。
```

UCB 不截断为 1。先选模型，再读取该题该模型的结果，只更新被选模型。
每题在一次回放中只出现一次，探索轮计入所有指标；同一种子下四个阈值使用相同题目顺序。
首个种子作为主实验，报告给出模型选择的整数次数；其余回放用于汇总均值和标准差。

设 `mu_i` 为模型在全量数据上的准确率估计，`a*` 为满足 `mu_i >= mu0` 的最便宜模型：

```text
cost_regret(t)    = sum_{s=1}^t max(0, c[chosen_s] - c[a*])
quality_regret(t) = sum_{s=1}^t max(0, mu0 - mu[chosen_s])
```

这是论文第 2 页定义的经验版本，使用模型均值而非逐题 0/1 结果。全量准确率只用于事后评估，
不会影响路由。成本 regret 单位为美元，质量 regret 单位为预期答对数缺口。
没有可行模型时，成本 regret 为 `N/A`，质量 regret 仍然计算。

报告分别给出被选 CSV 记录的实际费用与固定成本合计。被选记录有缺失费用时，实际总额为
`N/A`，同时保留已知费用下界和缺失条数；固定成本合计与成本 regret 仍使用 `c_i`。
FT-UCB 不保证有限轮次（包括当前的 3000 轮）的实际准确率达到阈值；乐观上界可能让便宜但不可行的模型长期被选中。
重复回放的标准差仅反映同一数据上的顺序与随机性，不能视为对新题的置信区间。

## 输出

| 文件 | 内容 |
| --- | --- |
| `scored_results.csv` | 当前 12000 条已有模型回答的评分，供事后比较和复核；不是路由决策记录 |
| `trace.csv` | 每个阈值、种子、轮次的选择、候选集合、选择前统计、累计指标与两种 regret |
| `runs.csv` | 每次回放的准确率、费用、缺失条数、两种 regret 与模型选择次数 |
| `curves.csv` | 每个阈值、每轮 regret 的均值与标准差 |
| `summary.json` | 固定成本、模型准确率、可行模型、汇总及输入文件哈希 |
| `report.md` | 主实验指标、整数选择次数、重复实验汇总、公式与讨论 |
| `regret.png` / `.pdf` | 两种累计 regret 曲线，均值 ± 标准差 |
| `arm_selections.png` / `.pdf` | 主实验的模型选择次数图 |

当前 CSV 命令的 `runs.csv` 为 4 条汇总、`trace.csv` 为 4 × 3000 条决策记录、`curves.csv` 为 4 × 3000 条曲线记录。
每个阈值的模型选择次数之和为 3000。

再次使用同一输出目录会覆盖同名文件。PNG/PDF 可用于幻灯片。
`bandit/output/ft-ucb/` 中的正式结果随代码提交到 GitHub；其他实验输出目录仍被忽略。

算法边界、手算 regret、缺失费用及标签隔离的测试代码位于 `bandit/tests/`，需要时手动运行：

```powershell
python -m unittest discover -s bandit/tests
```
