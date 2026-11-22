---
layout: post
title: "分辨率假说（二）：高噪声是个高通滤波器——那些暴涨 30 分的 Agent 到底学到了什么？"
title_en: "The Resolution Hypothesis (Part 2): High Noise Is a High-Pass Filter—What Did Those +30-Point Agents Actually Learn?"
date: 2026-11-17 19:00:00 +0800
description: "对 12 个基准的评测器源码与进化产物逐行审计：一句话就能复现 90% 的惊人涨幅，修好评测漏洞后中位涨幅从 17.6 pp 跌到 0.8 pp。"
description_en: "A line-by-line audit of graders and evolved artifacts across 12 benchmarks: one sentence recovers 90% of the headline gains, and repairing the graders drops the median from 17.6 pp to 0.8 pp. (Bilingual EN / 中文 with TL;DR)"
tags: agent-evolution evaluation research
categories: research-notes
thumbnail: /resolution-hypothesis/assets/gain_ledger.png
related_posts: true
---

<div class="zine-dispatch" data-title-zh="分辨率假说（二）：高噪声是个高通滤波器——那些暴涨 30 分的 Agent 到底学到了什么？" data-title-en="The Resolution Hypothesis (Part 2): High Noise Is a High-Pass Filter—What Did Those +30-Point Agents Actually Learn?">
  <a class="zine-pill-link" data-i18n-zh="论文 arXiv:2611.23068 ↗" data-i18n-en="Paper · arXiv:2611.23068 ↗" href="https://arxiv.org/abs/2611.23068" target="_blank" rel="noopener">论文 arXiv:2611.23068 ↗</a>
  <div class="zine-lang-switch" role="group" aria-label="Language">
    <button type="button" class="zine-lang-btn" data-lang-target="en" aria-pressed="false">EN</button>
    <button type="button" class="zine-lang-btn is-active" data-lang-target="zh" lang="zh-CN" aria-pressed="true">中文</button>
  </div>
</div>

<div class="zine-lang-panel" data-lang-panel="zh" lang="zh-CN" markdown="1">

<div class="zine-tldr" markdown="1">

**TL;DR：** 高噪声评估在进化循环中扮演了一台“高通滤波器”：只帮少数题目提分 2–5 pp 的扎实技巧会被噪声淹没，只有跨全题库生效的评测捷径才能刺穿阈值。通过对 12 个基准的逐行审计，我们在全部 27 组原涨幅超 3 pp 的配置中仅往空基线追加一句话，就复现了中位 90% 的收益；修好评测漏洞后，中位涨幅从 17.6 pp 暴跌至 0.8 pp。无论是 SpreadsheetBench 上“30 分百发百中、6 分淹没在噪声里”的对照，还是 LiveMathematicianBench 里进化自动磨出“劝降话术”攻破模型数学自尊心的案例，都直接印证了：分辨率决定了进化到底能留住什么。

</div>

> **《分辨率假说》系列（2/3）** · [论文 arXiv:2611.23068](https://arxiv.org/abs/2611.23068) · [项目主页与 30s 互动小游戏](/resolution-hypothesis/) · [幕后手记](/blog/2026/resolution-hypothesis-behind-the-scenes/)
> **连载导航**：[（一）原地踏步与方差账](/blog/2026/resolution-hypothesis-part-1/) · **（二）高通滤波器与捷径审计** · [（三）好 Skill 与失明的选择器](/blog/2026/resolution-hypothesis-part-3/)

在[上一篇](/blog/2026/resolution-hypothesis-part-1/)中，我们通过方差分解 $$\text{Var}(\hat{Y}) = \sigma^2 + \tau^2$$ 算清楚了一件事：常规 Agent 评估的单次标准差在 $$1.7\sim 4.1\text{ pp}$$ 左右，这导致两三个百分点的方法差距完全被淹没在了噪声里。

但正如我们在上一篇末尾提到的，文献和我们的复现网格里明明存在另一类现象：在 SpreadsheetBench、LiveMathematicianBench、AIME 2025 等基准上，进化后的产物相比 Baseline 动辄暴涨 $$10\sim 40\text{ pp}$$，远远超出了单次运行的噪声范围。

这是否说明进化算法在这些任务上真正涌现出了有效的提升？

当我们顺着“分辨率假说”的逻辑去审视这些暴涨，并把进化出来的产物逐行拆开、对照评测器源码做完审计后，我们发现了一个非常离谱的情况：**高噪声的评估系统，在进化循环中实际上扮演了一个“高通滤波器（High-Pass Filter）”的角色。**

---

### 低分辨率评估偏爱“捷径”

不妨先从直觉上推演一下：假设验证集有 40 道各不相同的任务，评估噪声的标准差是 $$3\sim 4\text{ pp}$$。

如果提议器（Proposer）总结出了一条**真正扎实但只针对某类具体任务**的技巧（例如“处理 CSV 缺失值时先检查编码格式”），它可能只会帮智能体多做对 1 到 2 道题，带来的真实提升 $$\Delta \approx 2.5\sim 5\text{ pp}$$。在单次评估的高噪声下，这么小的提升很容易被一次偶然的采样失败抵消掉，选择器根本认不出它。

反过来，什么样的修改能够产生 $$+20\sim 40\text{ pp}$$ 的单步跳跃，从而毫无悬念地刺穿噪声阈值、被选择器一把抓住呢？

**它必须是一条能够跨过具体任务内容的差异、一次性让测试集里几乎所有题目同时提分的全局规则。** 而在现有的基准测试中，能够做到这一点的，往往不是领域能力的突破，而是迎合评测器缺陷的**评估捷径（Evaluation Shortcuts）**。

为了验证这个推论，我们对 12 个基准上报告的最大涨幅做了两项对照测试：

1. **一句话测试（The One-Rule Test）**：进化出来的 Skill 或 Prompt 往往包含几十条指令。我们通过 LLM 辅助筛查加人工核对，从中只挑出**一句话**，直接拼到未进化的 Baseline 后面（其余一字不改）；
2. **修复评估测试（Repaired Evaluation）**：我们将评测脚本中违背基准原意的漏洞修好（例如恢复官方公式重算、将死板的字符串匹配改为语义等价判分、剔除题目泄露项），再用同一批进化产物重新测分。

![一句话复现与修复评估后的涨幅对比](/resolution-hypothesis/assets/gain_ledger.png){: style="display: block; max-width: 100%; width: auto; height: auto; max-height: 620px; margin: 1.5rem auto; object-fit: contain; border-radius: 8px;" }

实验结果非常离谱地整齐：在所有原涨幅超过 $$3\text{ pp}$$ 的 27 个“模型-基准”对中，**仅往 Baseline 后面追加一句话，就复现了完整进化算法中位数 90% 的涨幅**（图 3a）；而一旦把评测器的漏洞修好，**中位数涨幅立刻从 17.6 pp 跌到了 0.8 pp**（图 3b）。哪怕我们在修复后的评测器下从头重新运行进化算法，昔日的暴涨也再没有回来。

---

### 来鉴赏下非常离谱的案例

比起宏观的统计数字，我们在审计具体基准时看到的几个微观案例，更能直观地展示“进化算法是如何与评估分辨率互动的”（可以在我们论文 [arXiv:2611.23068](https://arxiv.org/abs/2611.23068) 的附录 E 中看到更多细节）：

#### 1. SpreadsheetBench 上的“30 分 vs 6 分”分辨率阈值

在 SpreadsheetBench 上，无论是原论文还是我们的复现，进化方法都能拿到 $$+30\sim 40\text{ pp}$$ 的惊人提升。检查进化产物后我们发现，核心全靠一句话：_“严禁向单元格写入 Excel 公式字符串，必须在 Python 中算出结果数值后直接填入纯数字或文本。”_

为什么写 Excel 公式会被判错？因为伴随某方法开源的评测脚本在读取生成的 `.xlsx` 文件时，**没有调用电子表格引擎去重新计算公式（Recalculate），导致所有未缓存求值的公式单元格全被读成了空值（0 分）**。

这里最值得玩味的是修复前后的对比实验：

当我们按照官方评测标准，引入 LibreOffice 在判分前对工作簿做公式重算后，上述进化产物的涨幅从 $$+31.1\text{ pp}$$（以 GPT-5.4 nano 为例）降到了 $$+6.7\text{ pp}$$。为什么还剩大约 $$6\text{ pp}$$ 的真实正收益？因为有少部分题目用到了较新的 Excel 函数，LibreOffice 也不支持求值，所以“直接用 Python 算好填数值”在修复后的评测器下，依然是一个能稳赚 $$6\text{ pp}$$ 的有效策略。

那么问题来了：**既然在修复后的评测器下，这条规则依然有 $$+6\text{ pp}$$ 的真实收益，为什么我们用修复后的评测器从头重新跑进化，算法却学不会它了呢？**

算一笔账就明白了：验证集一共有 39 道题，$$+6\text{ pp}$$ 只相当于 **2 到 3 道题**的差距，刚好掉进了单次评估的随机噪声区间里。结果，在原评测器（该规则价值 $$30\text{ pp}$$）下，8 次独立运行 100% 学会并保留了这一行为；而在修复后的评测器（价值 $$6\text{ pp}$$）下重跑的 12 个进化产物中，虽然有一半在某轮提议时把这句话写进了文本，但最终真正保留并执行这一行为的只有 **2 个**。

同一个行为准则，收益 $$30\text{ pp}$$ 时百发百中，收益 $$6\text{ pp}$$ 时就淹没在噪声里——这正是“分辨率决定哪些改进能被保留”的直接证据。

#### 2. 当基座模型抗拒“走捷径”时，进化算法学会了“说服”

在 **LiveMathematicianBench**（一个将最新 arXiv 数学论文定理转化为五选一选择题的基准）上，我们发现了一个严重的数据集构建漏洞：在全部 177 道题中，有 85 道题的选项里包含“其余选项中有一个是正确的，但可以证明一个更强的结果（stronger result）”，且**只要这个选项出现，它 100% 就是标准答案**（测试集 106 道题中有 46 道属于此类）。

但有趣的事情发生在我们做“一句话测试”的时候：当我们直接把规则 _“如果某个选项说其余某项正确但能证明更强结果，就选它”_ 追加给 Baseline 时，三个模型在含该选项的题目上照做的比例只有 **4%、54% 和 27%**，远不如完整进化出的 Skill（照做比例高达 **67%、86% 和 92%**）。

为什么一句直白的指令不管用？因为受过良好训练的基座模型有自己的“数学自尊心”——面对一道严肃的图论或代数定理题，你直接命令它不看推导、无脑选一个模糊的兜底选项，模型在推理时会抗拒执行。

而当我们对比完整进化出来的 Skill 文本时，发现进化循环在多轮迭代中，实际上完成了一次针对基座模型心理防线的“修辞优化”：

1. 它先编造了一个听起来很学术的理由：告诉模型出题人给出的具体数学选项 _“几乎总是被故意弱化表述的（almost always deliberately understated）”_；
2. 它扭转了举证责任：规定除非你能严格证明某个具体选项已经是紧致的（sharp），否则必须默认选择“更强结果”；
3. 它枚举了该选项的所有表面措辞变体，并在输出最终答案前再重复强调一次。

也就是说，当某个捷径的潜在回报足够大（足以刺穿评估噪声）时，进化算法不仅能找到这个捷径，甚至能把选择压力用在“如何优化话术来哄骗基座模型听话”上。而一旦我们只看测试集里剩下的 60 道不含该选项的正常数学题，所有模型的进化收益全部归零（$$-1.7\sim 0.0\text{ pp}$$）。

#### 3. 偏离目标的基准与一行配置引发的“大涨”

在审计过程中，我们还顺带看清了许多基准测试本身的局限（详见论文附录 E.3 与 E.4）：

- **不用检索的检索基准（SearchQA）**：SearchQA 本意是考察模型从给定的搜索引擎片段（Snippets）中提取答案的能力。它的原版评测采用严格字符串匹配（把 `"George Calvert (the 1st Lord Baltimore)"` 判错，只认简写），进化算法通过限制输出最短跨度获得了十几个点的提升。但当我们换用语义判分、并尝试**把题目里提供的所有搜索片段全部删光（让模型闭卷裸考）**时，DeepSeek-V4-Flash 的得分从 89.4% 变成了 90.2%，GPT-5.6 Luna 从 89.8% 变成了 90.7%——模型根本不需要看检索内容就能靠预训练知识答对，剩下的错题里反而有 36% 是因为原数据集中 Jeopardy 题目丢失了分类上下文导致的死题。
- **一行历史窗口配置带来的 $$+30\text{ pp}$$（ALFWorld）**：在 ALFWorld 上，Meta-Harness 和 AHE 两个代码进化方法在 DeepSeek-V4-Flash 和 GPT-5.6 Luna 上暴涨了 $$16\sim 32\text{ pp}$$。我们去检查生成的代码才发现，初始的 Baseline Agent 代码里写死了一个极不合理的超参：**只保留最近 2 步的交互历史（History）**，导致智能体走两步就忘了自己搜过哪儿。其中一次进化运行什么别的代码都没动，仅仅把这个历史窗口放开，单步就暴涨了 $$29\text{ pp}$$。

---

### 娜拉出走之后怎么办？

把这些评测捷径和基准缺陷清理干净之后，我们看到了一幅很真实的图景：

- 在 DocVQA、AIME 2025、PUPA 等基准上，修复评测后的 Baseline 本身就已经拿到了 $$88\sim 97$$ 分，留给进化的天花板（Headroom）本来就所剩无几；
- 而在 **Terminal-Bench**、**OfficeQA** 等真正困难的基准上，Baseline 离满分还有 $$30\sim 45\text{ pp}$$ 的巨大空间，且没有表面捷径可钻，此时现有的进化算法就集体陷入了停滞（增益仅为 $$1.0\sim 3.6\text{ pp}$$，低于单次运行标准差）。

这就把我们逼到了最核心的追问上：**在排除了评测捷径之后，面对像 Terminal-Bench 这样真正的硬骨头，大模型到底能不能写出真正有用的好 Skill？如果能，它们是怎么在进化循环里被弄丢的？**

在[最后一篇](/blog/2026/resolution-hypothesis-part-3/)里，我们就把提议器（Proposer）和选择器（Selector）彻底拆开，到候选池的内部去找答案。

</div>

<div class="zine-lang-panel is-hidden" data-lang-panel="en" lang="en" markdown="1">

<div class="zine-tldr" markdown="1">

**TL;DR:** Inside an evolution loop, noisy evaluation acts as a high-pass filter: solid task-specific techniques worth 2–5 pp drown in variance, while benchmark-wide evaluation shortcuts punch straight through the floor. Auditing graders and artifacts line by line across 12 benchmarks, appending a single sentence to the unevolved baseline recovers a median 90% of all gains above 3 pp across 27 model–benchmark pairs—and repairing the graders collapses the median gain from 17.6 pp to 0.8 pp. From SpreadsheetBench's 30 pp vs. 6 pp resolution threshold to LiveMathematicianBench's evolved rhetorical persuasion overcoming models' mathematical pride, the cases prove one point directly: resolution determines what evolution can keep.

</div>

> **_The Resolution Hypothesis_ Series (2/3)** · [Paper arXiv:2611.23068](https://arxiv.org/abs/2611.23068) · [Project Page & 30s Interactive Game](/resolution-hypothesis/) · [Behind the Scenes](/blog/2026/resolution-hypothesis-behind-the-scenes/)
> **Series Navigation**: [(1) Running in Place & the Variance Ledger](/blog/2026/resolution-hypothesis-part-1/) · **(2) The High-Pass Filter & Shortcut Audit** · [(3) Good Skills & Blind Selectors](/blog/2026/resolution-hypothesis-part-3/)

In [Part 1](/blog/2026/resolution-hypothesis-part-1/), our variance decomposition $$\text{Var}(\hat{Y}) = \sigma^2 + \tau^2$$ established a sobering baseline: standard single-run agent evaluations carry a standard deviation of $$1.7\sim 4.1\text{ pp}$$, completely swallowing two-to-three-point differences between methods.

Yet as we noted at the close of Part 1, a striking counterweight sits right there in both the literature and our own reproduction grid: on benchmarks like SpreadsheetBench, LiveMathematicianBench, and AIME 2025, evolved artifacts routinely jump $$10\sim 40\text{ pp}$$ over the baseline—far outside any single-run noise band.

Does that mean evolution is genuinely unlocking emergent capability on those tasks?

When we followed the logic of _The Resolution Hypothesis_ into those surges—dissecting the evolved artifacts line by line against the actual grader source code—we uncovered something remarkably stark: **inside an evolution loop, a noisy evaluation pipeline behaves like a high-pass filter.**

---

### Why Low-Resolution Evaluation Selects for Shortcuts

Follow the intuition for a second. Suppose a validation split has 40 diverse tasks and an evaluation noise SD of $$3\sim 4\text{ pp}$$.

If the proposer distills a **genuinely solid, task-specific engineering habit** (say, "inspect file encoding before handling missing CSV values"), it might flip 1 or 2 additional tasks from fail to pass—a true gain of $$\Delta \approx 2.5\sim 5\text{ pp}$$. Under single-run noise, a gain that modest is effortlessly wiped out by one unlucky rollout, and the selector never even sees it.

Conversely, what kind of edit can deliver a $$+20\sim 40\text{ pp}$$ single-step jump large enough to pierce the noise floor without fail and get locked in by the selector?

**It has to be a global rule that cuts across task content and lifts almost every problem in the test set at once.** On existing benchmarks, what delivers that kind of across-the-board lift is rarely a breakthrough in domain skill—it is an **Evaluation Shortcut** tailored to a quirk in the grader.

To test this prediction, we ran two controlled audits across the largest reported gains on 12 benchmarks:

1. **The One-Rule Test**: Evolved skills or prompts typically contain dozens of directives. Using LLM-assisted screening followed by manual verification against the grader, we extracted **a single sentence** and appended it verbatim to the unevolved baseline (changing nothing else).
2. **Repaired Evaluation**: We fixed defects where the evaluation script departed from the benchmark's intent (restoring official spreadsheet recalculation, replacing brittle string matching with semantic equivalence, or auditing leaked/broken items) and re-scored the **exact same evolved artifacts**.

![Comparison of reported gains vs. one-rule baseline recovery and repaired evaluation](/resolution-hypothesis/assets/gain_ledger.png){: style="display: block; max-width: 100%; width: auto; height: auto; max-height: 620px; margin: 1.5rem auto; object-fit: contain; border-radius: 8px;" }

The results lined up with eerie precision: across all 27 model–benchmark pairs with an original gain above $$3\text{ pp}$$, **appending a single sentence to the baseline recovered a median 90% of the full evolutionary gain** (Fig. 3a). And the moment the grader defects were repaired, **the median gain plunged from 17.6 pp to 0.8 pp** (Fig. 3b). Even when we re-ran evolution from scratch under the repaired graders, those headline surges never came back.

---

### A Gallery of Remarkably Absurd Cases

More than the aggregate numbers, a handful of concrete cases from our audit make vivid how evolution interacts with evaluation resolution (full details in Appendix E of [arXiv:2611.23068](https://arxiv.org/abs/2611.23068)):

#### 1. The "30 pp vs. 6 pp" Resolution Threshold on SpreadsheetBench

On SpreadsheetBench, both the original paper and our reproductions saw evolution surge by $$+30\sim 40\text{ pp}$$. Opening the evolved artifacts revealed that the entire gain hinged on one instruction: _"Never write Excel formula strings into cells; compute the results in Python and write raw numbers or text directly."_

Why were valid Excel formulas getting marked wrong? Because the evaluation script released alongside a method read the generated `.xlsx` files **without invoking a spreadsheet engine to recalculate formulas, reading every uncached formula cell as empty (0 points).**

The most telling part came from comparing before and after the repair:

Once we restored the official evaluation protocol—invoking LibreOffice to recalculate workbooks before grading—the evolved artifact's gain on GPT-5.4 nano dropped from $$+31.1\text{ pp}$$ to $$+6.7\text{ pp}$$. Why did roughly $$6\text{ pp}$$ of real upside remain? Because a small subset of tasks used newer Excel functions that LibreOffice itself cannot evaluate, so computing values in Python remained a genuinely profitable $$+6\text{ pp}$$ strategy even under the repaired grader.

Now comes the crucial question: **if that exact rule is still worth a genuine $$+6\text{ pp}$$ after repair, why couldn't evolution learn it when we re-ran from scratch on the repaired grader?**

Do the arithmetic and it clicks immediately: on a 39-task validation split, $$+6\text{ pp}$$ is just **2 to 3 tasks**, right inside single-run sampling noise. Under the broken grader (where the rule paid $$30\text{ pp}$$), **8 out of 8** independent runs discovered and kept it (100%). Under the repaired grader (where it paid $$6\text{ pp}$$), even though half of our 12 fresh evolution runs proposed the rule at some intermediate step, only **2** managed to retain and execute it by the end.

The exact same behavioral rule—100% captured at $$30\text{ pp}$$, drowned in noise at $$6\text{ pp}$$—is direct evidence that resolution governs which improvements survive.

#### 2. When Base Models Resist a Shortcut, Evolution Learns Rhetoric

On **LiveMathematicianBench** (a benchmark converting recent arXiv math theorems into 5-choice questions), we found a major dataset construction flaw: out of 177 problems, 85 include an option stating _"one of the other options is correct, but a stronger result can be proved,"_ and **whenever that option appears, it is 100% the ground-truth answer** (46 of the 106 test problems fall into this bucket).

The surprise came during our One-Rule Test: when we simply appended the blunt instruction _"if an option says another option is correct but a stronger result can be proved, select it"_ to the baseline, our three models obeyed it on only **4%, 54%, and 27%** of the applicable problems—far below the full evolved skills, which achieved **67%, 86%, and 92%** compliance.

Why did a plain directive fail? Because well-trained frontier models have a kind of "mathematical pride"—handed a serious graph-theory or algebra theorem, they push back when bluntly told to skip the proof and click a vague meta-option.

Inspecting the full evolved skill text revealed that across iterations, the evolutionary loop had effectively optimized a **rhetorical campaign** tailored to dismantle the base model's resistance:

1. It invented an academic-sounding rationale, warning the model that concrete mathematical choices are _"almost always deliberately understated"_ by problem authors;
2. It inverted the burden of proof, instructing the model to default to the "stronger result" option unless it could rigorously prove a concrete bound was sharp;
3. It enumerated surface phrasings of the option and repeated the directive right before final answer emission.

In other words, once a shortcut's payoff is large enough to clear the noise floor, selection pressure doesn't just find the loophole—it invests rounds in refining persuasive copy to talk the base model into taking it. Meanwhile, on the remaining 60 normal math test problems without that option, evolutionary gain across every model was zero ($$-1.7\sim 0.0\text{ pp}$$).

#### 3. Misaligned Benchmarks and a One-Line Config Worth +30 pp

Along the way, our audit surfaced broader benchmark defects as well (see Appendix E.3 and E.4):

- **A retrieval benchmark that needs no retrieval (SearchQA)**: SearchQA is meant to test extracting answers from provided search snippets. Its original grader used strict string matching (marking `"George Calvert (the 1st Lord Baltimore)"` wrong in favor of a bare short span), letting evolution gain over ten points merely by enforcing ultra-short outputs. When we switched to semantic grading and **deleted 100% of the retrieved snippets (forcing closed-book inference)**, DeepSeek-V4-Flash moved from 89.4% to 90.2%, and GPT-5.6 Luna moved from 89.8% to 90.7%. Models answer the questions straight from pretraining weights—and 36% of the remaining errors stem from Jeopardy clues missing their category context in the original dataset.
- **A one-line history cap worth $$+30\text{ pp}$$ (ALFWorld)**: On ALFWorld, the code-evolution methods Meta-Harness and AHE jumped $$16\sim 32\text{ pp}$$ on DeepSeek-V4-Flash and GPT-5.6 Luna. Inspecting the diffs showed that the starter baseline agent hardcoded a bizarre hyperparameter: **retaining only the last 2 turns of interaction history**, causing the agent to forget where it had just searched two steps ago. One evolution run changed nothing else—simply lifting that history window produced a $$+29\text{ pp}$$ single-step jump.

---

### After the Mirage Clears: What Remains?

Strip away the evaluation shortcuts and benchmark quirks, and a clear picture emerges:

- On DocVQA, AIME 2025, and PUPA, the repaired baseline already sits at $$88\sim 97\%$$, leaving almost no headroom for evolution in the first place.
- On genuinely hard benchmarks like **Terminal-Bench** and **OfficeQA**, baselines still sit $$30\sim 45\text{ pp}$$ below the ceiling with no surface shortcuts to exploit—and there, existing evolution algorithms stall out across the board (gains of just $$1.0\sim 3.6\text{ pp}$$, below a single run's standard deviation).

That brings us to the heart of the matter: **once shortcuts are off the table, on real hard problems like Terminal-Bench, can LLMs even write genuinely useful skills? And if they can, how does the evolution loop manage to lose them?**

In [Part 3](/blog/2026/resolution-hypothesis-part-3/), we decouple the proposer from the selector completely and look inside the candidate pool for the answer.

</div>

<script defer src="/assets/js/bilingual-post.js"></script>
