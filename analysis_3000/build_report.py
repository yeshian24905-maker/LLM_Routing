"""Build the Chinese report and an Excel workbook from the verified analysis."""

import csv
import json
from pathlib import Path

from export import Report, render_pdf, workbook


OUT = Path(__file__).resolve().parent
S = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
MODELS = [r["model"] for r in S["models"]]
NAMES = dict(zip(MODELS, ["Haiku", "Luna", "Gemini", "Sonnet"]))
FULL = dict(zip(MODELS, ["Claude Haiku 5.5", "GPT-5.6 Luna", "Gemini 3.8 Flash", "Claude Sonnet 5.5"]))
M = {r["model"]:r for r in S["models"]}
P = lambda x: f"{100*x:.2f}%"
USD = lambda x: f"${x:.6f}"
CI = lambda r: f"{100*r['ci95_low']:.2f}–{100*r['ci95_high']:.2f}%"


def csv_rows(name):
    with (OUT / "tables" / f"{name}.csv").open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))


def by(field, value, model):
    return next(r for r in S["groups"][field] if r[field]==value and r["model"]==model)


def main():
    d=Report("MMMU 3,000 题：四模型与 FT-UCB 路由分析",
             "30 个 area × 100 题 · 12,000 条已有模型记录 · 2026-11-22\n全部为本地离线统计，无新增 API 调用")
    d.h1("主要发现")
    d.p("Gemini 的总体准确率最高，为 89.63%（2689/3000）；Haiku 的总费用最低，为 $0.977823，准确率 81.67%。在本样本的平均成本与总体准确率两个指标上，Haiku 和 Gemini 构成观察到的 Pareto 前沿。Luna 和 Sonnet 分别比 Haiku 和 Gemini 更贵、总体正确数更少，但仍有其他三个模型都答错、仅它们答对的题目。")
    d.table(["模型","正确 / 3000","准确率","95% Wilson CI","总费用 USD","均费 / 题 USD","费用 / 答对 USD"],
            [[NAMES[m],M[m]["correct"],P(M[m]["accuracy"]),CI(M[m]),USD(M[m]["latest_outcome_cost_usd"]),USD(M[m]["mean_outcome_cost_usd"]),USD(M[m]["cost_per_correct_usd"])]
             for m in sorted(MODELS,key=lambda m:-M[m]["accuracy"])], [1300,1250,1100,1850,1450,1500,1550])
    d.p("四模型全部答对 2149 题，全部答错 172 题；679 题的正确性存在模型间分歧。至少一个模型答对 2828 题，对应事后 oracle 上界 94.27%，比单用 Gemini 多答对 139 题。能否实现其中的增益，取决于路由器是否能在看到真实答案之前识别适合的模型。")
    d.p("FT-UCB 的主回放中，80% 阈值达到目标；82.5%、85% 与 89% 阈值均未达到目标。有限的 3000 轮内，UCB 仍让便宜模型保持乐观可行，低成本 regret 不能单独说明路由质量足够。")
    d.note("评分口径：统一采用仓库已有的确定性评分器 v2，属于 MMMU-style 自动评分，并非官方评测器。原 Word 没有逐题标签或评分代码，前 1000 题的正确数有 1–2 题差异；本报告重新评定全部 3000 题，并保留逐条评分和复核清单。")
    d.image(OUT/"figures/overall.png","图 1｜总体准确率、95% Wilson 区间与 3000 条最新结果的费用。费用不包含被替换的历史尝试。")

    d.page();d.h1("1. 数据范围、完整性与评分方法")
    d.table(["核对项","结果"],[
        ["题目数量","原 1000 题 + 新增 2000 题 = 3000 道不同题目"],
        ["area 覆盖","30 个 area，各 100 题；因此 area 宏平均准确率等于总体微平均"],
        ["模型覆盖","四个模型各 3000 条，合计 12000 条；每个 question_id 均覆盖四模型"],
        ["重复检查","无重复 (question_id, model)，两批题目 ID 无交集；题干、选项和图像哈希组成的完全相同内容重复数为 0"],
        ["元数据与源文件","subject、question_type、correct_answer 与 questions.jsonl 一致；合并 CSV 与两份原 CSV 的逐字段拼接完全相同"],
        ["图片与文件保护","3383 个图像引用的内容哈希验证通过；原 Word、三份 CSV、题目与采集日志等 14 个输入文件分析前后哈希一致"],
        ["选择题 / 开放题","2841 / 159（94.70% / 5.30%）"],
        ["难度标签","Easy 944，Medium 1378，Hard 678；使用数据集已有标签"],
    ],[2600,7400])
    d.p("输入使用 mmmu_test_sample/result.csv，以及原始和 additional_2000 两份 questions.jsonl。模型 ID 按原始记录保留：")
    d.table(["简称","记录中的模型 ID"],[[NAMES[m],m] for m in MODELS],[1700,8300])
    d.h2("统一评分")
    d.p("优先读取最后一个完整 <answer> 标签。选择题提取选项字母，或把与某个选项唯一匹配的选项文本映射为字母；带解释的明确选项标签也可识别。缺少标签时只使用已有评分器的明确选项回退，不随机猜选项。无法解析、空回答或请求失败均记为未答对。")
    d.p("开放题使用标签内内容，或无标签时的回答文本；文本规范化后匹配参考答案，数值按既有实现四舍五入到两位小数后比较。该规则可能无法识别数学等价式，也可能因回答包含多个数值而宽松命中。全部 636 条开放题回答，以及 12 条其他解析例外，共 648 条记录列入 review_queue.csv；这是待人工确认清单，不表示全部评分有误。")
    d.p("总体主结果保留这套统一自动评分，没有在本次分析中按预期结果更改标签。开放题与解析异常的人工确认可能改变准确率、oracle 和 regret；选择题单独结果也在后文列出。所谓模型一致/分歧均指二元正确性，不表示原始答案文字相同/不同。")
    d.h2("不确定性的解释")
    d.p("准确率区间采用 95% Wilson 二项区间，是对本样本及统一评分口径的描述性估计。模型比较使用同题配对的精确 McNemar 检验，6 组比较采用 Holm 校正。区间与检验未包含题目来源相关性、评分误差、模型采集时间差异或测试集选择偏差，不能视作所有未来请求上的服务保证。")

    d.page();d.h1("2. 成本、时延与调用可靠性")
    total=sum(M[m]["latest_outcome_cost_usd"] for m in MODELS)
    historical=sum(r["known_request_cost_usd"] for r in S["collection_billing"])
    d.p(f"12000 条最新结果的已记录费用合计为 ${total:.8f}。这相当于对每道题运行四个模型的数据采集成本；一次单模型路由应只累计被选中模型的那条记录费用。")
    d.p("原 CSV 有 9 条费用为空：Sonnet 7 条、Luna 2 条，均为 moderation_error，且错误信息明确写明 No credits were charged。原始空值保持不变；报告中的 latest-outcome 费用仅对这 9 条使用有依据的零费用，并保留 cost_source=provider_no_charge。其余费用均取原始记录。")
    d.table(["模型","请求错误","空回答","content_filter","截断 length","p50 时延 / 秒","p95 时延 / 秒"],
            [[NAMES[m],M[m]["request_errors"],M[m]["empty_responses"],M[m]["content_filter_count"],M[m]["length_count"],f'{M[m]["latency_p50_s"]:.2f}',f'{M[m]["latency_p95_s"]:.2f}'] for m in MODELS],
            [1300,1200,1200,1500,1400,1700,1700])
    d.p("空回答共 16 条：Haiku 有 7 条 content_filter 空回答，Sonnet 有 7 条输入拒答，Luna 有 2 条输入拒答。它们都保留在 3000 题分母内，按未答对处理。Gemini 有 1 条 length 截断，仍按保存的回答评分。空回答和请求错误存在重叠，不能把上表各列相加作为失败总数。")
    d.p("Sonnet 的 p95 时延最短，Gemini 最长。时延来自当时的调用记录，会受到提供方、重试、并发与网络影响；它不是隔离环境中的纯模型速度，也不能用 3000 × 平均时延估算并发采集的实际墙钟时间。")
    d.table(["模型","平均输入 tokens","平均输出 tokens","平均 reasoning tokens","平均时延 / 秒"],
            [[NAMES[m],f'{M[m]["input_tokens_mean"]:.1f}',f'{M[m]["output_tokens_mean"]:.1f}',f'{M[m]["reasoning_tokens_mean"]:.1f}',f'{M[m]["latency_mean_s"]:.2f}'] for m in MODELS])
    d.note("token 均值仅基于该字段有记录的条目。不同提供方的输入图像、输出与 reasoning token 计数口径可能不同；不应把输出与 reasoning token 直接相加，也不能从 Sonnet 的 reasoning_tokens=0 推断其没有内部推理。")
    d.h2("采集账目与当前结果费用的区别")
    d.table(["批次","带账目完成请求数","已知历史请求费用 USD","完成请求费用缺失","无费用失败日志数"],
            [[r["cohort"],r["completion_requests"],f'${r["known_request_cost_usd"]:.8f}',r["unknown_completion_cost_count"],r["failed_requests_without_cost"]] for r in S["collection_billing"]])
    d.p(f"按 generation_id 去重并应用 accounting_refresh 后，历史日志中的已知请求费用合计 ${historical:.8f}，比最新结果费用多 ${historical-total:.7f}，反映被重试替换的尝试。另有 46 条没有费用的失败日志；这里没有把它们统一补为零。因此该历史合计是日志中可核实的已知费用，不等同于已核对完整 OpenRouter 账单。")

    d.page();d.h1("3. 准确率与成本的权衡")
    d.image(OUT/"figures/cost_quality.png","图 2｜每 1000 题费用与总体准确率；右下方的模型在这两个样本指标上受到支配。")
    d.p(f"Gemini 相比 Haiku 多答对 {M[MODELS[2]]['correct']-M[MODELS[0]]['correct']} 题，额外费用 ${M[MODELS[2]]['latest_outcome_cost_usd']-M[MODELS[0]]['latest_outcome_cost_usd']:.6f}，相当于每多答对一题增加约 ${(M[MODELS[2]]['latest_outcome_cost_usd']-M[MODELS[0]]['latest_outcome_cost_usd'])/(M[MODELS[2]]['correct']-M[MODELS[0]]['correct']):.4f}。这是两个完整模型结果的总体增量比值，不是逐题可实现的升级价格。")
    d.p("Luna 比 Haiku 贵约 46%，少答对 23 题；这两者的准确率差异在配对检验中不显著。Sonnet 比 Gemini 贵约 56%，少答对 82 题；Gemini 的总体优势在配对检验中显著。若目标只关心总体准确率与平均费用，当前样本支持先关注 Haiku 与 Gemini；若还关心开放题、个别 area、时延或互补能力，仍需查看其余维度。")
    d.h2("同一批 3000 题上的配对比较")
    d.table(["A / B","仅 A 对","仅 B 对","B − A / pp","Holm 校正 p","双模型 oracle"],
            [[f'{NAMES[r["model_a"]]} / {NAMES[r["model_b"]]}',r["a_only_correct"],r["b_only_correct"],f'{r["b_minus_a_accuracy_pp"]:+.2f}',f'{r["p_holm"]:.3g}',P(r["pair_oracle_accuracy"])] for r in S["paired_comparisons"]],
            [2400,1100,1100,1450,1550,1500])
    d.p("McNemar 检验只利用两模型结果不同的题目。Haiku/Luna 的 p=0.288，不能据此断言两者真实准确率相等，也不能宣称 Haiku 显著优于 Luna。其余五组在 Holm 校正后 p<0.05；这些比较以自动评分标签可靠、题目间适合该近似为前提。")

    d.page();d.h1("4. 题型、难度与多图题")
    d.table(["模型","选择题：正确 / 2841","选择题准确率（95% CI）","开放题：正确 / 159","开放题准确率（95% CI）"],
            [[NAMES[m],by("question_type","multiple-choice",m)["correct"],f'{P(by("question_type","multiple-choice",m)["accuracy"])}\n{CI(by("question_type","multiple-choice",m))}',by("question_type","open",m)["correct"],f'{P(by("question_type","open",m)["accuracy"])}\n{CI(by("question_type","open",m))}'] for m in MODELS],
            [1100,1800,2700,1700,2700])
    d.p("开放题明显更难：Sonnet 69.18%、Gemini 67.92%、Haiku 64.15%、Luna 54.09%。Sonnet 只比 Gemini 多答对 2 道开放题，不能凭这个小差异宣称其开放题优势稳定。样本从原来的 55 道增加到 159 道，但仍远少于选择题，且全部开放题标签需要人工复核。")
    d.p("在占 94.7% 的选择题上，Gemini 仍领先，Haiku 与 Luna 几乎持平。因此总体排序并非只由开放题造成。选择题的数值或文本格式例外仍可能影响少量标签。")
    d.table(["难度","题数","Haiku","Luna","Gemini","Sonnet"],
            [[v,by("difficulty",v,MODELS[0])["n"]]+[P(by("difficulty",v,m)["accuracy"]) for m in MODELS] for v in ["Easy","Medium","Hard"]])
    d.p("难度从 Easy 到 Hard 时，Gemini 从 93.33% 降至 85.40%，Haiku 从 88.35% 降至 75.07%，Luna 从 87.61% 降至 73.16%。难题是扩大模型质量差距的主要观察维度之一，但 difficulty 来自数据集标签；不同难度的 area、题型构成也不相同，不能把相关性解释为单一因果效果。")
    d.image(OUT/"figures/format_difficulty.png","图 3｜按题型与数据集难度标签划分的准确率。完整分组区间与费用见表格数据。")
    d.page();d.h1("4.1 多图题的样本限制")
    d.table(["每题图片数","题数","Haiku","Luna","Gemini","Sonnet"],
            [[v,by("image_count",v,MODELS[0])["n"]]+[P(by("image_count",v,m)["accuracy"]) for m in MODELS] for v in range(1,7)])
    d.p("2807 道题只有一张图；6 张图仅 3 题。多图组准确率没有单调下降，说明不能仅凭这份分组表推断图像数量对模型能力的影响。小组样本的极高准确率尤其不应作为稳定结论。")

    d.page();d.h1("5. 30 个 area 的表现")
    d.image(OUT/"figures/area_accuracy.png","图 4｜每个 area 均为 100 题，因此表内百分数也等于答对题数；色标跨所有 area 保持一致。")
    subjects=sorted({r["subject"] for r in S["groups"]["subject"]})
    for block,part in enumerate([subjects[:15],subjects[15:]],1):
        d.page();d.h1(f"5.{block} Area 准确率与总费用（{1 if block==1 else 16}–{15 if block==1 else 30}）")
        d.note("每格为“准确率 / 该 area 的 100 条结果费用 USD”。费用遵循前述已记录费用与明确未扣费处理。各 area 的 95% 区间详见 Excel 的 area 表。")
        d.table(["Area（n=100）","Haiku","Luna","Gemini","Sonnet"],
                [[a.replace("_"," ")]+[f'{P(by("subject",a,m)["accuracy"])}\n${by("subject",a,m)["latest_outcome_cost_usd"]:.5f}' for m in MODELS] for a in part],
                [2800,1800,1800,1800,1800])
    d.page();d.h1("5.3 Area 层面的主要现象")
    d.p("按四模型准确率的平均值，Music 最难（57.00%），其次为 Diagnostics and Laboratory Medicine（70.00%）、Geography（75.25%）与 Electronics（75.75%）。Music 中 Gemini 70%、Sonnet 64%、Haiku 52%、Luna 42%；即使使用最强单模型，也仍有较大失败空间。")
    d.p("Gemini 在 24/30 个 area 中达到最高准确率（含并列），其中 21 个为独占第一；Sonnet 为 8 个（独占 5 个），Haiku 为 2 个（独占 1 个），Luna 没有 area 第一。并列会重复计数，四个数量不应相加得到 30。")
    d.p("Sonnet 在 Architecture and Engineering、Diagnostics and Laboratory Medicine、Economics、Mechanical Engineering 各比 Gemini 多答对 1 题，在 Psychology 多 2 题；Haiku 在 Marketing 为 95%，比 Gemini 多 2 题。每个 area 只有 100 题，这类 1–2 题的样本优势不足以支持直接部署固定领域路由。")
    # An optimistic area-only reference, fitted and evaluated on the same data.
    area_choices=[]
    for a in subjects:
        best=min([r for r in S["groups"]["subject"] if r["subject"]==a],key=lambda r:(-r["accuracy"],M[r["model"]]["mean_known_cost_usd"]))
        area_choices.append({"subject":a,"selected_model":best["model"],"correct":best["correct"],"n":100,"outcome_cost_usd":best["latest_outcome_cost_usd"]})
    area_correct=sum(r["correct"] for r in area_choices)
    area_cost=sum(r["outcome_cost_usd"] for r in area_choices)
    d.p(f"一个在同一份数据上事后选出每个 area 最准模型的 area-only oracle 可答对 {area_correct}/3000（{P(area_correct/3000)}），费用 ${area_cost:.6f}。它仅比单用 Gemini 多答对 {area_correct-M[MODELS[2]]['correct']} 题，且费用更高；这提示从全局 area 标签获得的观察到的增益远小于逐题 oracle。但这个参照在原样本上选模型并评分，有选择偏差，不能当作泛化结果。")
    with (OUT/"tables/area_only_oracle.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(area_choices[0]));w.writeheader();w.writerows(area_choices)

    d.page();d.h1("6. 模型互补性与 routing opportunity")
    d.table(["答对模型数","题目数","占比"],[[r["correct_models"],r["questions"],P(r["fraction"])] for r in S["overlap"]])
    d.p("四模型全对 2149 题（71.63%）；全错 172 题（5.73%）；存在正确性分歧 679 题（22.63%）。恰好一个模型答对 123 题，其中 Gemini 独占正确 67 题、Haiku 22 题、Luna 19 题、Sonnet 15 题。总体较弱的模型仍可能提供独特的正确答案，不能把总体支配关系等同于逐题无用。")
    d.p("Haiku 与 Gemini 合起来至少一个答对 2776/3000（92.53%），比单用 Gemini 多 87 题；Gemini 可救回 Haiku 的 326/550 个错误（59.27%）。反过来，Haiku 可救回 Gemini 的 87/311 个错误（27.97%）。这些是已知正确性后的覆盖关系，不是已经实现的自动升级策略。")
    d.image(OUT/"figures/complementarity.png","图 5｜左：正确性重叠；右：列模型能答对多少比例的行模型错误。救回率需要真实答案才能事后计算。")
    d.h2("两个事后 oracle：相同上界，不同成本假设")
    d.table(["规则","正确数 / 准确率","费用 USD","Haiku / Luna / Gemini / Sonnet 选择数"],
            [["固定平均费用顺序选最便宜正确模型",f'{S["oracles"][0]["correct"]} / {P(S["oracles"][0]["accuracy"])}',USD(S["oracles"][0]["cost_usd"]),"2622 / 203 / 160 / 15"],
             ["按每题实际记录费用选最便宜正确模型",f'{S["oracles"][1]["correct"]} / {P(S["oracles"][1]["accuracy"])}',USD(S["oracles"][1]["cost_usd"]),"1935 / 889 / 150 / 26"]],
            [3000,2000,1500,3500])
    d.p("若四模型都错，第一个 oracle 回落到固定平均费用最低的模型；第二个回落到该题实际费用最低的模型。两者都只计被最终选中那一个回答的费用，不计试遍所有模型后发现正确答案的费用。第二个还预知了各模型在该题的实际费用，因此其信息假设更强。")
    d.p("两者都达到 2828/3000（94.27%），但都需要预知正确性，不能部署。它们是这四份已保存回答在当前评分下的覆盖上界，不是所有可能模型、所有重采样回答或检索/工具系统的理论能力上界。172 道全错题与 123 道唯一正确题已分别导出，便于后续审查。")

    d.page();d.h1("7. 前 1000 题与新增 2000 题的比较")
    d.table(["模型","原 Word 正确数 / 1000","统一重评 / 1000","新增正确数 / 2000","新增准确率","变化 / pp","校正 p"],
            [[NAMES[r["model"]],r["old_correct_reference_word"],r["old_correct_regraded"],r["new_correct"],P(r["new_accuracy"]),f'{r["new_minus_old_pp"]:+.2f}',f'{r["p_holm"]:.3f}'] for r in S["cohort_comparison"]],
            [1100,1700,1500,1800,1400,1300,1200])
    d.p("原 Word 中 Haiku 824、Luna 827、Sonnet 879、Gemini 908；仓库评分器统一重评得到 826、825、878、908。没有原报告逐题标签，因此无法确定差异来自哪些人工判断或解析规则。本报告不会把这两套计数直接拼接；新增与原批次都用同一 v2 评分。")
    d.p("新增 2000 题中，Haiku 81.20%、Luna 80.10%、Gemini 89.05%、Sonnet 86.45%；四者都比原批次低约 1.35–2.40 个百分点。但两批不同题目的双侧 Fisher 检验经 4 组 Holm 校正后均不显著，Newcombe/Wilson 差值区间也均跨 0。数据不支持仅凭这些差异宣布模型退化。")
    d.table(["模型","新增 − 原批次 / pp","差值的 95% 区间 / pp"],
            [[NAMES[r["model"]],f'{r["new_minus_old_pp"]:+.2f}',f'{r["difference_ci95_low_pp"]:+.2f} 至 {r["difference_ci95_high_pp"]:+.2f}'] for r in S["cohort_comparison"]])
    d.image(OUT/"figures/cohort_accuracy.png","图 6｜同一评分口径下，两批数据的模型准确率。题目不同，不能当作同题重复实验。")
    d.p("前 1000 题各 area 只有 33 或 34 题；新增题按原样本的补足量抽样到每 area 总计 100 题。两批在题型、难度、内容和采集时间上仍有差异。新增与原批次的比较是描述性稳定性检查，无法单独区分抽样波动、题目构成和调用提供方变化。")

    d.page();d.h1("8. 四个阈值下的静态可行模型")
    d.p("把四个模型各自的全 3000 题准确率作为离线均值估计，可得到下面的事后可行集合。此表是比较基准，未向 FT-UCB 的选择步骤提供真实均值。")
    d.table(["固定阈值","样本均值满足阈值的模型","平均费用最低的可行模型","其准确率","其 95% 区间下界是否过阈值"],
            [[P(r["threshold"]),r["feasible_models"].replace("anthropic/claude-haiku-5.5","Haiku").replace("openai/gpt-5.6-luna","Luna").replace("google/gemini-3.8-flash","Gemini").replace("anthropic/claude-sonnet-5.5","Sonnet"),NAMES[r["cheapest_feasible_model"]],P(r["model_accuracy"]),"是" if r["ci95_low_meets_threshold"] else "否"] for r in S["static_thresholds"]],
            [1100,3100,2300,1300,2200])
    d.p("89% 阈值只有 Gemini 在样本均值上可行；其 95% Wilson 区间为 88.49–90.67%，下界仍低于 89%，所以这份样本尚不足以把“真实准确率至少 89%”作为置信保证。85% 与 82.5% 时 Gemini 和 Sonnet 可行，但 Gemini 更便宜。80% 时四模型的点估计都过线，Haiku 最便宜；Luna 自身的 95% 区间下界仍低于 80%。")
    d.h2("FT-UCB 的实验设置")
    d.p("每次回放共 3000 轮，每道题只出现一次；四个模型是四个 arm。按固定费用从低到高编号并各初始化一次，之后每轮计算经验准确率 + sqrt(2 ln(t) / 已采样次数)，从 UCB 不低于阈值的模型中选择费用最低者；候选集为空时均匀随机选取。UCB 不截断到 1。只读取被选中模型当前题的正确性来更新该 arm。算法依据用户提供的 Algorithm 6，并对照来源 [1]。")
    d.p("主结果使用 seed=42；题目顺序为拼接两个 questions.jsonl 后，以 Random('order:42') 打乱。另用 seeds 42–141 各回放一次，四个阈值共享各自 seed 对应的同一题目顺序。这些重排不是重新调用模型，也不是 100 份独立数据集。")
    d.table(["模型","固定费用 c：USD / 调用","费用可用记录数","离线评价均值 μ"],
            [[NAMES[m],f'{M[m]["mean_known_cost_usd"]:.10f}',3000-M[m]["raw_missing_cost_count"],P(M[m]["accuracy"])] for m in MODELS])
    d.note("论文采用事先已知的固定 arm 费用。本实验沿用仓库的离线近似：c 为该模型全部非空 CSV 费用的平均值，属于全数据校准，不是上线前独立标定。实际累计费用使用逐题保存的费用；regret 使用固定 c，两者必须分开。数据为无放回有限题库，且提供即时正确性反馈，与论文的平稳随机 bandit 假设有差别。")

    d.page();d.h1("9. FT-UCB：3000 轮主回放")
    d.table(["阈值","Haiku","Luna","Gemini","Sonnet","正确数","准确率","逐题费用 USD","达标"],
            [[P(r["threshold"]),r["haiku_count"],r["luna_count"],r["gemini_count"],r["sonnet_count"],r["correct"],P(r["accuracy"]),USD(r["observed_cost_usd"]),"是" if r["meets_threshold"] else "否"] for r in S["ft_ucb_primary"]],
            [1000,1000,900,1000,1000,1100,1300,1700,1000])
    d.p("89% 阈值下，Haiku 在第 2807 轮首次于初始化之后被其他模型替代。主回放最终仍选 Haiku 2818 次、Luna 180 次，Gemini 和 Sonnet 各 1 次。Luna 的总体准确率也低于 89%；它得到更多选择，是因为低采样量带来的宽 UCB 让它仍处于候选集，而非算法已经识别它达到目标。")
    d.p("其余三个阈值都选 Haiku 2997 次，另三个模型各 1 次。Haiku 在这条主路径中初始化后的最小 UCB 约 88.77%，始终高于 85%、82.5% 与 80%，因此这些阈值产生完全相同的选择序列与实际结果。")
    d.p("3000 轮末 n≈3000 时，sqrt(2 ln(3000)/3000)≈7.31 个百分点，仍然很宽。Haiku 的全数据均值约 81.67%，加上该数量级的探索项约为 88.98%；这解释了为什么 89% 附近才开始排除它。该计算只是对探索项尺度的说明，实际决策严格使用该路径当时的均值、t 和 n。")
    d.image(OUT/"figures/ft_ucb_behavior.png","图 7｜主回放的模型选择数与累计准确率；85%、82.5%、80% 的路径重合。")
    d.p("FT-UCB 的输入阈值是可行性学习的目标，不是每轮或有限总轮次准确率的硬约束。这里不应把“运行了阈值 89% 的算法”写成“路由器达到 89%”。本次结果表明高阈值下仍有显著质量缺口；也不能由单个短时实验推出算法在所有长时随机环境中失效。")

    d.page();d.h1("10. Cost / quality regret 曲线")
    d.p("使用来源 [1] 的正部 gap 定义，并在固定阈值设置中用 μ₀ 代替可行阈值。a* 是全数据均值达到阈值的模型中固定费用最低者；[x]+ = max(0,x)。")
    d.note("R_C(t) = Σ(s=1…t) [c(k_s) − c(a*)]+\nR_Q(t) = Σ(s=1…t) [μ₀ − μ̂_full(k_s)]+")
    d.p("μ̂_full 是每个模型在全 3000 题上的统一评分准确率，只用于评价 gap。这里是单条回放路径上的经验 plug-in pseudo-regret，使用估计均值，未计算论文对真实随机环境的精确期望。100 次重排均值用于顺序敏感性分析，不是新的真实 reward 分布。")
    d.table(["阈值","离线 a*","R_C(3000) / USD","R_Q(3000)","按固定 c 累计费用 USD"],
            [[P(r["threshold"]),NAMES[r["optimal_feasible_model"]],f'{r["cost_regret_usd"]:.8f}',f'{r["quality_regret"]:.6f}',USD(r["fixed_cost_usd"])] for r in S["ft_ucb_primary"]])
    d.image(OUT/"figures/ft_ucb_regret.png","图 8｜t=1…3000 的经验 cost 与 quality pseudo-regret。cost 图中 89%、85%、82.5% 的路径重合。")
    d.p("Cost regret 不奖励选择比 a* 更便宜的模型，也不惩罚这些便宜模型的质量不足；因此高阈值下 cost regret 仅约 $0.001647，并不表示质量达标。82.5% 与 85% 选择序列相同，但阈值不同，quality regret 分别为 24.991 与 99.941。")
    d.p("Quality regret 的单位是累计概率差，不是美元，也不是实际答错题数。它不能替换为 max(0, μ₀×t−累计正确数)。所有 regret 均采用四模型平均费用与全数据均值这套共同参照，未与逐题最便宜正确模型的 oracle 混用。")

    d.page();d.h1("11. 题目顺序敏感性：100 次离线回放")
    d.table(["阈值","平均准确率","准确率 P05–P95","平均逐题费用 USD","达标回放数 / 100","Haiku 选择数范围"],
            [[P(r["threshold"]),P(r["accuracy_mean"]),f'{P(r["accuracy_p05"])}–{P(r["accuracy_p95"])}',USD(r["observed_cost_usd_mean"]),round(100*r["fraction_meeting_threshold"]),f'{int(r["haiku_count_min"])}–{int(r["haiku_count_max"])}'] for r in S["ft_ucb_sensitivity"]],
            [1100,1450,2300,1700,1800,1650])
    d.p("100 次回放中，80% 阈值全部达标，其余三个阈值均未达标。89% 阈值下，Haiku 被选 1869–2997 次，变化由前序题目反馈导致；尽管选择数会变，Gemini 在这 100 次回放中均只被初始化选择一次，Luna 的乐观 UCB 足以长期吸收被 Haiku 让出的选择。")
    d.p("表中的 P05–P95 是同一批已保存回答在 100 个顺序下的经验分位数，不能解释为模型泛化准确率的 90% 置信区间。85%、82.5% 和 80% 下，选择数不变，只有初始化时其他模型对应的题目及其逐题费用略有变化。")
    d.h2("对后续 routing 研究的含义")
    d.p("当前数据支持两个不同问题：一是是否能以更低费用选择一个达到总体质量要求的 arm；二是是否能利用逐题互补性超过最强单模型。FT-UCB 是使用 arm 历史均值的非上下文策略，不直接利用题目内容、area 或候选答案差异，因此不能期待它自动达到 94.27% 的逐题 oracle 上界。")
    d.p("若继续研究，应先完成开放题与解析异常的人工复核，冻结逐题标签；随后预先规定训练/校准与评估划分，独立标定固定费用，再比较 FT-UCB、单用 Haiku、单用 Gemini 与新的路由方法。阈值、探索强度或领域规则若依据同一测试集调参，需要独立评估集或交叉验证。此处没有修改用户指定的 FT-UCB，也没有新增模型采集。")

    d.page();d.h1("12. 输出说明与复现")
    d.table(["文件 / 目录","用途"],[
        ["MMMU_3000_Analysis.docx / .html / .pdf","同内容报告；PDF 由本地浏览器导出（若可用）"],
        ["analysis_tables.xlsx","完整模型、分组、配对、oracle、FT-UCB 及逐条评分表"],
        ["tables/model_summary.csv / by_subject.csv","模型总表与 30 area 明细，包含分组 95% 区间、费用及质量字段"],
        ["tables/scored_results.csv / review_queue.csv","12000 条评分记录，以及 648 条待复核记录；可用 question_id+model 关联原始回答"],
        ["tables/question_outcomes.csv","3000 道题的四模型正确性与两个 oracle 的选择和费用"],
        ["tables/all_models_wrong.csv / only_one_model_correct.csv","172 道全错题与 123 道唯一正确题"],
        ["tables/ft_ucb_trace.csv","四阈值各 3000 轮，共 12000 条主路径选择、反馈、UCB、累计费用及 regret"],
        ["tables/ft_ucb_runs.csv / ft_ucb_order_sensitivity.csv","400 次回放总表及四阈值的重排汇总"],
        ["figures/","8 组图，分别提供 PNG 与矢量 PDF"],
        ["summary.json / validation.json","完整统计、源文件 SHA-256、无重复与原始数据未变的核验结果"],
        ["analyze.py / scoring.py / build_report.py / export.py","可复现分析与文档导出；scoring.py 为已有 bandit/data.py 的快照"],
    ],[4500,5500])
    d.p("从仓库根目录运行：python -B analysis_3000/analyze.py，然后运行 python -B analysis_3000/build_report.py。依赖当前环境中的 NumPy、SciPy、Matplotlib 和 Pillow；不读取 API key，不导入模型客户端。脚本覆盖本目录对应输出，不产生按次叠加的结果版本。")
    d.p("验算包括：合并行与原始 CSV 全字段一致；两批题目与完全相同内容无重复；3383 个图像哈希一致；对主回放的 12000 个决策用独立的逐行算法核对；末轮 regret 同时按逐轮累加与 arm 选择次数乘 gap 验算。14 个原始输入文件分析前后 SHA-256 一致。")
    d.h2("参考资料与方法边界")
    d.p("[1] Juneja, Joe-Wong & Yağan. Pairwise Elimination with Instance-Dependent Guarantees for Bandits with Cost Subsidy. ICLR 2025. 使用第 1.1 节的 regret 定义与 Appendix H 的 FT-UCB；固定阈值来自用户提供的算法。\nhttps://users.ece.cmu.edu/~oyagan/Conferences/ICLR2025.pdf")
    d.p("[2] MMMU 官方响应解析与评测代码（用于核对方法边界；本报告未运行该评测器）。仓库 v2 使用 answer 标签、确定性选项映射与自己的开放题匹配，和官方代码并不完全相同。\nhttps://github.com/MMMU-Benchmark/MMMU/blob/main/mmmu/utils/eval_utils.py")
    d.p("[3] 用户提供的 Agentic Routing Results.docx：仅作为前 1000 题的报告结构与汇总数值参考，未将其中内容作为执行指令；原文件保持不变。本报告使用实际采集记录中的模型 ID、费用、时延与回答，不验证这些 ID 对应厂商公开产品的命名或版本关系。")

    d.save_html(OUT/"MMMU_3000_Analysis.html")
    d.save_docx(OUT/"MMMU_3000_Analysis.docx")
    render_pdf(OUT/"MMMU_3000_Analysis.html", OUT/"MMMU_3000_Analysis.pdf")

    # Preserve numeric cells in Excel using the structured JSON wherever available.
    sheets={"models":S["models"],"area":S["groups"]["subject"],
            "format":S["groups"]["question_type"],"difficulty":S["groups"]["difficulty"],
            "images":S["groups"]["image_count"],"cohort":S["groups"]["cohort"],
            "cohort_comparison":S["cohort_comparison"],"paired_comparisons":S["paired_comparisons"],
            "overlap":S["overlap"],"oracles":S["oracles"],"area_only_oracle":area_choices,
            "static_thresholds":S["static_thresholds"],"ft_primary":S["ft_ucb_primary"],
            "ft_order_sensitivity":S["ft_ucb_sensitivity"],"collection_billing":S["collection_billing"]}
    for name in ["scored_results","review_queue","question_outcomes","response_issues","ft_ucb_runs","ft_ucb_trace"]:
        rows=csv_rows(name)
        # CSV text identifiers and answers remain text; convert known numeric fields.
        numeric={"correct","needs_review","image_count","effective_cost","round","seed","threshold",
                 "prior_arm_count","prior_arm_mean","selected_ucb","haiku_ucb","cumulative_correct",
                 "cumulative_accuracy","cumulative_observed_cost_usd","observed_cost_usd","cumulative_fixed_cost_usd",
                 "cost_regret_usd","quality_regret","accuracy","fixed_cost_usd","rounds","meets_threshold",
                 "haiku_count","luna_count","gemini_count","sonnet_count","correct_models","haiku_correct",
                 "luna_correct","gemini_correct","sonnet_correct","oracle_fixed_order_cost_usd","oracle_actual_min_cost_usd"}
        for r in rows:
            for k in numeric.intersection(r):
                if r[k]=="":r[k]=None
                else:
                    x=float(r[k]);r[k]=int(x) if x.is_integer() else x
        sheets[name]=rows
    sheets["data_dictionary"]=[
        {"field":"accuracy / ci95_low / ci95_high","meaning":"Proportions in [0,1]; multiply by 100 for percent. Wilson intervals are descriptive."},
        {"field":"correct / needs_review","meaning":"Binary automatic v2 correctness; review flag is not an adjudicated error."},
        {"field":"known_cost_usd","meaning":"Sum of nonempty original latest-outcome cost values, USD."},
        {"field":"latest_outcome_cost_usd / effective_cost","meaning":"Recorded cost; missing explicit no-charge moderation outcomes resolved as zero only in derived outputs."},
        {"field":"mean_known_cost_usd","meaning":"Mean over rows with a cost in the original CSV; fixed FT-UCB cost, not mean of all outcomes."},
        {"field":"quality_regret","meaning":"Sum max(0, threshold - full-data arm accuracy); probability-gap units, not observed errors."},
        {"field":"cost_regret_usd","meaning":"Sum max(0, fixed chosen arm cost - fixed cheapest feasible arm cost), USD."},
        {"field":"p_value / p_holm","meaning":"Exact McNemar for paired models; Fisher exact for disjoint cohorts; Holm within each family."},
        {"field":"*_p05 / *_p95","meaning":"Empirical permutation-order quantiles over 100 replays, not statistical confidence limits."},
        {"field":"latency_*","meaning":"Recorded call latency in seconds; token/latency means use available values only."},
        {"field":"cost / extracted_answer / correct_answer","meaning":"Original cost and answer strings remain text where preserving formatting is necessary."},
    ]
    workbook(OUT/"analysis_tables.xlsx",sheets)
    print(f"Created report and workbook ({len(sheets)} sheets).")


if __name__=="__main__":
    main()
