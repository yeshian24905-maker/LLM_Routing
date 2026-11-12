# MMMU 数据准备

在项目根目录运行 `python -m data_prep`。脚本下载 MMMU 题目和图片，调用四个模型并评分，输出每个「题目 × 模型」的一条记录。

核心代码只保留 5 个文件，每个都有实际逻辑：

| 文件 | 职责 |
| --- | --- |
| `__main__.py` | 命令行入口，直接处理四个命令 |
| `dataset.py` | 数据下载、校验、图片处理和提示词构建 |
| `collector.py` | 模型调用、实际费用查询和断点续跑 |
| `scoring.py` | 答案解析和评分 |
| `common.py` | 配置、文件读写、日志恢复及结果导出 |

四个模型及请求参数见 `config.json`。不写死价格：运行时读取最新模型目录，费用使用 API 实际账单。

## 使用

需要 Python 3.10+。先用少量题目检查：

```powershell
python -m pip install -r data_prep/requirements.txt
python -m data_prep check-models
python -m data_prep download --splits dev --subjects Accounting Physics --limit-per-subject 1 --output-dir data_prep/output/smoke
python -m data_prep run --data-dir data_prep/output/smoke --dry-run

# 真实采集：2 道题 × 4 个模型，共 8 次 completion 请求
$env:OPENROUTER_API_KEY = "替换为自己的 Key"
python -m data_prep run --data-dir data_prep/output/smoke --delay 1
```

`--dry-run` 不联网、不调用模型，也不生成虚构结果。Key 从环境变量读取，不写入产物，不自动加载 `.env`。

准备全部 30 个学科的 validation 数据：

```powershell
python -m data_prep download
python -m data_prep run --dry-run
python -m data_prep run --delay 1
```

默认数据目录为 `data_prep/output/mmmu_validation`，结果保存在该目录的 `runs/default/`。

常用选项见 `python -m data_prep <命令> --help`：

- `download --splits dev validation test --output-dir ...` 合并多个 split。
- `download --subjects Accounting Physics Art_Theory` 选择学科，名称区分大小写。
- `download --limit-per-subject N` 按原始顺序取每个「学科 × split」的前 N 题。
- `download --revision ...` 指定版本；实际下载固定在解析出的 commit SHA。
- `run --models anthropic/claude-haiku-5.5 --limit N` 只采集指定模型和前 N 题。
- `run --output-dir ...` 为另一组实验选择独立的输出目录。

## 输出

`results.csv` 和 `results.jsonl` 包含相同的 12 列：

| 字段 | 含义 |
| --- | --- |
| `question_id` | 原始 MMMU ID |
| `subject` | 学科 |
| `model` | 请求使用的完整模型 ID |
| `prediction` | 解析出的选项字母或开放题最终答案 |
| `correct_answer` | Ground truth |
| `correct` | 整数 0 / 1；API 或答案解析失败为 0 |
| `input_tokens` | API 报告的输入用量，包含图片 |
| `output_tokens` | API 报告的输出用量，不另行重复累加 reasoning tokens |
| `cost` | 该次 completion 实际报告费用，单位 USD |
| `latency` | 非流式 completion HTTP 请求耗时，单位秒 |
| `raw_response` | 未修改的模型回复文本 |
| `error` | API、解析或账单缺失错误；正常为 null |

费用优先使用 `usage.cost`，缺失时查询 `/generation` 的 `total_cost`。token 数优先使用 `usage.prompt_tokens/completion_tokens`，缺失时查询 `native_tokens_prompt/native_tokens_completion`。仍未知的值保留 `null` 并记录错误；实际零费用保留为 `0`。CSV 的 null 为空单元格，列表答案用 JSON 保存。

耗时不包含图片编码、请求间等待和后续账单查询。账单信息缺失不改变已完成的答案评分，因此 `correct=1` 时也可能有 `accounting_error`。

选择题只接受可唯一解析的答案，不随机猜测。开放题只评最终答案，忽略大小写并按词边界匹配文本；数字按 MMMU 约定保留两位小数，支持单位、千位分隔符、科学计数法和答案列表。此解析方式与官方自由文本回退行为有所区别。

题目、选项、图片路径及校验值保存在 `questions.jsonl`；图片按 `<image n>` 标记插入请求，同一张图只附加一次，其他非空图片也会附加。不会发送答案或 explanation。原始 API JSON、账单查询和字段来源保存在 `attempts.jsonl`，模型目录快照保存在 `catalogs/`。

## 续跑、重试与导出

运行同一条 `run` 命令会跳过已记录的组合，包括错误行。每次调用后同步写入日志，CSV/JSONL 按 `(question_id, model)` 保留最新结果。文件锁防止同一目录并发采集；数据和配置指纹防止不同实验混合。已完成的数据目录不会被覆盖。

```powershell
# 显式重新调用错误行，会产生新的请求和可能的费用
python -m data_prep run --data-dir data_prep/output/smoke --retry-errors --delay 1

# 从已有日志重新生成结果，不调用模型
python -m data_prep export --run-dir data_prep/output/smoke/runs/default

# 离线测试，包含模拟的下载 → 四模型采集 → 导出流程
python -m unittest discover -s data_prep/tests -v
```

旧尝试仍保留在日志中。`summary.json` 同时报告最新结果费用、所有尝试累计已知费用，以及未知费用条数。缺账单信息的错误行也会被 `--retry-errors` 重新调用。

HTTP 401/402/403 保存当前错误后停止；其他单次错误记录后继续。当次采集有错误时退出码为 1。若服务器已完成请求、本地尚未写入日志时退出，续跑无法判断该次是否已计费。训练时应筛选错误行，避免将接口故障当成模型能力不足。

参考：[MMMU 数据集](https://huggingface.co/datasets/MMMU/MMMU)、[官方评分实现](https://github.com/MMMU-Benchmark/MMMU/blob/main/mmmu/utils/eval_utils.py)、[OpenRouter 用量与费用](https://openrouter.ai/docs/cookbook/administration/usage-accounting)。
