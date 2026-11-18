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
UCB1 不使用题目内容或学科特征，目标是寻找整体表现较好的模型；当前奖励只优化准确率。
费用、延迟单独统计。如需优化费用，必须先另外确定奖励中的费用权重和范围。

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
| `trace.csv` | UCB 与随机路由每轮选择、奖励、累计准确率、累计已知费用、regret；可用于画学习曲线 |
| `summary.json` | 均值与标准差、模型选择次数均值、输入文件哈希、评分统计 |
| `report.md` | 可读的实验比较表和评估限制 |

费用只累计被路由选中的 CSV 记录。缺失费用保留为未知，不自动当成 0；有缺失时，已知费用总和仅为下界。
CSV 的费用和延迟不完整覆盖历史失败重试、排队等待等开销，不能视为真实部署成本与端到端延迟的精确预测。
