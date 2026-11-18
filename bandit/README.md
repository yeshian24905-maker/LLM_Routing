# UCB1 离线路由实验

直接复用已收集的回答，不调用模型 API，不增加 API 费用。只依赖 Python 3.10+ 标准库。

## 运行

在项目根目录执行。默认读取指定的 `data_prep/output/run/result.csv`：

```powershell
python -m bandit
```

当前工作区的完整数据实际位于下面的路径，可以直接指定，无需复制文件：

```powershell
python -m bandit --input data_prep/output/mmmu_test_sample/runs/sequential/results.csv
```

显式指定重复次数、种子和输出目录：

```powershell
python -m bandit --input data_prep/output/run/result.csv --repeats 30 --seed 42 --output-dir bandit/output/ucb1
```

输入必须包含 `question_id,subject,question_type,model,raw_response,correct_answer,cost,latency,error`。
默认要求 **1000 个不同题目，每题每个模型恰好一条记录**，例如四模型共 4000 条。
脚本拒绝重复记录、不完整的模型覆盖、不一致的标准答案和无效数值；不会静默丢题。
其他大小的数据可用 `--expected-prompts N`。

## 算法与测试协议

一个模型就是一个 arm，一道题就是一轮，奖励为答对 `1`、答错 `0`。
每次回放随机打乱题目顺序，策略状态从零开始；前 K 轮将每个模型各选一次。
之后选择下面的分数最大的模型，平分时随机选择：

```text
t = 当前轮数，从 1 开始
N_m = 前 t-1 轮中模型 m 被选择的次数
mean_m = 模型 m 在被选择的轮次中获得的平均奖励
UCB_m(t) = mean_m + sqrt(2 * ln(t) / N_m)
```

先选择模型，再查 CSV 中该题该模型的结果，**只用这一个结果更新 UCB**。
其他模型的结果只用于事后比较。每次回放中，每题仅出现一次，探索轮也计入结果。
默认使用种子 42 至 71 做 30 次回放；每次回放的 UCB、随机路由和固定模型使用相同题目顺序。

比较对象：UCB1、均匀随机路由、始终使用每一个固定模型。
另计算逐题选择答对模型的 oracle 准确率上界。
`regret_vs_best_fixed` 是“最佳固定模型答对数 − 当前策略答对数”；可能为负。
`gap_to_oracle` 使用逐题 oracle，为非负。trace 中的最佳固定模型按当前前缀计算。

这评估的是**有即时标准答案反馈时的在线学习过程**，不需要先将 1000 题划成训练集和测试集。
它不能代替冻结路由器后在独立测试集上的泛化评估。重复回放的标准差只反映同一批数据上顺序与随机性的影响。
UCB1 不使用题目内容或学科特征，目标是寻找整体表现较好的模型。`ucb1` 的奖励只看准确率，费用、延迟单独统计。

## 成本加权奖励

`--value-per-correct V` 为每个 V 增加一个 `ucb1-cost[V=…]` 策略。V 是“答对一题值多少美元”，奖励为：

```text
净值 = correct − min(cost / V, 1)        取值 [-1, 1]
reward = (净值 + 1) / 2                  映射到 UCB1 要求的 [0, 1]
```

两个模型之间，多出的准确率乘以 V 大于多出的平均费用时，策略才偏向更贵的模型。
例如在多花 0.0026 美元、多答对 8.5% 时，V 约为 0.031 是分界点。
单次费用超过 V 时惩罚截断为 1。缺失费用按满惩罚 1 处理，不当成免费。
延迟不进入奖励。

```powershell
python -m bandit --input data_prep/output/run/result.csv --value-per-correct 0.01 0.03 0.1
```

## 准确率目标下降低成本

固定的 V 无法直接对应一个准确率。`--target-accuracy τ` 为每个 τ 增加一个 `target-ucb[tau=…]` 策略
（[target_ucb.py](target_ucb.py)），目标是在整次回放准确率不低于 τ 的前提下尽量降低费用：

```text
score_m = KL-UCB(模型 m 的准确率) − 模型 m 的平均已知费用 / V
答完一题后：log V += step × (τ − correct)
```

- 答错时 V 变大，策略偏向准确的模型。答对时 V 变小，策略偏向便宜的模型。
  因此策略会在两个模型之间混合，而不是只收敛到一个模型。
- 对 T 道题有恒等式：`准确率 = τ + ln(V_初始 / V_最终) / (step × T)`。
  只要最终 V 不超过初始 V，这次回放就达到目标。初始 V 越大、step 越小，达标越稳，但多花的费用也越多。
- 准确率上界使用 KL-UCB（[Garivier and Cappé, 2011](https://arxiv.org/abs/1102.2490)），不用 UCB1 的上界。
  在 1000 题内，UCB1 的探索项过宽，即使只看准确率也只能达到约 87%，接近最佳模型的目标无法达到。
- 费用不加探索项，只用已知费用的平均值。还没有已知费用的模型，按已知平均费用最高的模型估计。
- `--target-initial-value`（默认 1 美元）和 `--target-step`（默认 0.3）调节上述取舍。

报告另列“Accuracy targets”表：各次回放中达标的次数，以及 best mix 费用。
best mix 是用全量数据算出的、平均准确率不低于 τ 的最便宜的固定随机模型混合，只作参照，不能部署。
在线策略的费用高于 best mix 的部分，是探索和学习的代价。

```powershell
python -m bandit --input data_prep/output/run/result.csv --target-accuracy 0.84 0.86 0.88
```

UCB1 公式依据：[Auer et al., 2002](https://people.eecs.berkeley.edu/~russell/classes/cs294/s11/readings/Auer%2Bal%3A2002.pdf)。

## 回答评分

原始 CSV 没有 `correct` 列，必须先将回答转成奖励。脚本采用可审阅的确定性规则：

- 优先取最后一个完整的 `<answer>...</answer>`，避免把推理中碰巧出现的正确答案算作回答。
- 选择题解析选项字母；没有标签时，只接受独立选项或明确的 answer/option/choice 表述。
  解析失败、空回答和请求错误计 0，不随机猜一个答案。`accounting_error` 不影响已有回答的评分。
- 开放题支持标准答案中的备选答案列表，文本大小写与空白归一化，以及带单位数字、千位分隔符和科学计数法。
  数字按 MMMU 常用规则保留两位小数后比较；文本使用词边界匹配。
  不做符号运算、单位换算、语义同义判断。
- 评分参考 [MMMU 官方评估代码](https://github.com/MMMU-Benchmark/MMMU/blob/main/mmmu/utils/eval_utils.py)，
  但额外优先使用 answer 标签、不随机猜选项，文本边界规则也不同，因此**不是官方评估器的等价实现**。
  两位小数规则可能误判很小的数值或精度要求高的答案；所有开放题和回退解析均标记 `needs_review=1`。

自动评分适合先检查实验流程；正式报告前应人工审阅 `scored_results.csv` 中的标记项。
可以提供部分或全部人工核验标签覆盖自动评分，不修改原始 CSV：

```csv
question_id,model,correct
test_Accounting_100,anthropic/claude-haiku-5.5,1
```

```powershell
python -m bandit --input data_prep/output/run/result.csv --labels reviewed_labels.csv
```

`--labels` 中 `correct` 必须为 `0` 或 `1`，重复键和原始数据不存在的键会报错。
标签应在比较算法前独立确定，避免按实验结果反复调整评分标准。

## 输出

默认输出到 `bandit/output/ucb1/`，再次使用同一目录会覆盖这些报告文件：

| 文件 | 内容 |
| --- | --- |
| `scored_results.csv` | 原始字段、抽取答案、0/1 得分、评分来源、人工复核标记 |
| `runs.csv` | 各次回放、各策略的准确率、已知费用、未知费用条数、延迟、regret、模型选择次数 |
| `trace.csv` | 各 UCB 策略与随机路由每轮选择、奖励、累计准确率、累计已知费用、regret；可用于画学习曲线 |
| `summary.json` | 均值与标准差、模型选择次数均值、输入文件哈希、评分统计 |
| `report.md` | 可读的实验比较表和评估限制 |

费用只累计被路由选中的 CSV 记录。缺失费用保留为未知，不自动当成 0；有缺失时，已知费用总和仅为下界。
CSV 的费用和延迟不完整覆盖历史失败重试、排队等待等开销，不能视为真实部署成本与端到端延迟的精确预测。
