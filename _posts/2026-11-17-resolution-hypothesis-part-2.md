---
layout: post
title: "分辨率假说（二）：高噪声是个高通滤波器——那些涨了 30 分的 Agent 到底学到了什么？"
title_en: "The Resolution Hypothesis (Part 2): High Noise Is a High-Pass Filter—What Did Those +30-Point Agents Actually Learn?"
date: 2026-11-17 19:00:00 +0800
description: "逐行审计 12 个基准的评测器源码与进化产物：往基线后补一句话就能复现 90% 的涨幅，修好评测问题后，中位涨幅从 17.6 pp 降到 0.8 pp。（支持 EN / 中文 一键切换，含 TL;DR）"
description_en: "A line-by-line audit of graders and evolved artifacts across 12 benchmarks: one sentence recovers 90% of the headline gains, and repairing the graders drops the median from 17.6 pp to 0.8 pp. (Bilingual EN / 中文 with TL;DR)"
tags: agent-evolution evaluation research
categories: research-notes
thumbnail: /resolution-hypothesis/assets/gain_ledger.png
related_posts: true
---

<div class="zine-dispatch" data-title-zh="分辨率假说（二）：高噪声是个高通滤波器——那些涨了 30 分的 Agent 到底学到了什么？" data-title-en="The Resolution Hypothesis (Part 2): High Noise Is a High-Pass Filter—What Did Those +30-Point Agents Actually Learn?">
  <a class="zine-pill-link" data-i18n-zh="论文 arXiv:2611.23068 ↗" data-i18n-en="Paper · arXiv:2611.23068 ↗" href="https://arxiv.org/abs/2611.23068" target="_blank" rel="noopener">论文 arXiv:2611.23068 ↗</a>
  <div class="zine-lang-switch" role="group" aria-label="Language">
    <button type="button" class="zine-lang-btn" data-lang-target="en" aria-pressed="false">EN</button>
    <button type="button" class="zine-lang-btn is-active" data-lang-target="zh" lang="zh-CN" aria-pressed="true">中文</button>
  </div>
</div>

<div class="zine-lang-panel" data-lang-panel="zh" lang="zh-CN" markdown="1">

<div class="zine-tldr" markdown="1">

**TL;DR：** 高噪声评估在进化循环里像一台高通滤波器：只帮一两道题提分 2.5–5 pp 的扎实技巧很容易被方差盖住，真正能稳稳穿过噪声底板的，往往是跨全题库生效的评测捷径。逐行核对 12 个基准的评测器源码与进化产物后，在全部 27 组原涨幅超 3 pp 的配置上，仅往未进化基线后补一句话，就能复现中位数 90% 的收益；把评测问题修好之后，中位涨幅从 17.6 pp 降到了 0.8 pp。进化本身其实是在老老实实优化最响的信号——SpreadsheetBench 上同一条规则“30 分时 8/8 留下、6 分时 12 里只剩 2”，以及 LiveMathematicianBench 上进化花轮次打磨话术去说服模型走捷径，都很直观地说明了这一点。清理掉这些表面增益后，真正困难的基准还留着大块空间，也把问题推向下一篇：在没有捷径可走时，好 Skill 究竟写不写得出来，又是在哪一步弄丢的。

</div>

> **《分辨率假说》系列（2/3）** · [论文 arXiv:2611.23068](https://arxiv.org/abs/2611.23068) · [项目主页与 30s 互动小游戏](/resolution-hypothesis/) · [幕后手记](/blog/2026/resolution-hypothesis-behind-the-scenes/)
> **连载导航**：[（一）原地踏步与方差账](/blog/2026/resolution-hypothesis-part-1/) · **（二）高通滤波器与捷径审计** · [（三）好 Skill 与挑不出来的选择器](/blog/2026/resolution-hypothesis-part-3/)

在[上一篇](/blog/2026/resolution-hypothesis-part-1/)里，我们通过方差分解 $$\text{Var}(\hat{Y}) = \sigma^2 + \tau^2$$ 算清了一笔账：常规 Agent 评估的单次标准差大概在 $$1.7\sim 4.1\text{ pp}$$ 之间，换句话说，两三个百分点的方法差距，基本都会被单次采样的噪声盖过去。

不过上一篇末尾也留了个悬念。不管是在文献里，还是在我们自己跑出的复现网格里，确实有不少看上去非常硬气的数字：在 SpreadsheetBench、LiveMathematicianBench、AIME 2025 这些基准上，进化后的产物相比 Baseline 动辄高出 $$10\sim 40\text{ pp}$$，早就远远超出了单次运行的噪声区间。

看到这种涨幅，人的第一反应自然是：至少在这些任务上，进化算法是真的学到了点东西吧？

顺着“分辨率假说”的思路，我们把这些涨幅最大的进化产物一行一行拆开，再对照着评测器的源码逐条核对了一遍。查完之后，背后的机制其实相当直白：**在进化循环里，高噪声的评估系统，干的就是一台高通滤波器（High-Pass Filter）的活。**

---

这件事其实不难凭直觉想通。不妨假设验证集里有 40 道各不相同的任务，单次评估的噪声标准差在 $$3\sim 4\text{ pp}$$ 左右。

如果提议器（Proposer）总结出了一条**很扎实、但只适用于某类具体问题**的工程习惯——比如“处理 CSV 缺失值前先检查文件编码”——它也许能帮智能体多做对 1 到 2 道题，带来的真实提升大约是 $$\Delta \approx 2.5\sim 5\text{ pp}$$。放在单次评估的噪声里，这点收益很容易被某一次偶然的采样失手抹平，选择器（Selector）大概率根本认不出它。

反过来想，什么样的改动能够单步拉出 $$+20\sim 40\text{ pp}$$ 的差距，稳稳穿过噪声底板、让选择器一眼相中？

它几乎必然是一条**跨过具体题目内容、能同时撬动整个测试集里大多数题目的全局规则**。而在现有的基准测试里，能做到这一点的，往往不是什么领域能力的跃迁，而是刚好踩中评测器习惯或漏洞的**评估捷径（Evaluation Shortcuts）**。

话也得说公道一点：进化算法找到这种捷径，并不是它在“作弊”，更不是优化器坏掉了。恰恰相反，这说明优化器正在老老实实地干活——循环里什么信号高过了噪声底板，选择压力就会顺着什么信号往下推。噪声越高，留给细水长流的局部改进的空间就越窄，最后能活下来的，自然只剩嗓门最大的那几条全局捷径。

为了验证这个判断，我们对 12 个基准上报告的最大涨幅做了两项对照审计：

1. **一句话测试（The One-Rule Test）**：进化出来的 Skill 或 Prompt 动辄写满几十条指令。我们先用 LLM 辅助筛选，再对照评测器源码逐条人工核对，从中只抽出**一句话**，原封不动地拼在未进化的 Baseline 后面，其余一字不改；
2. **修复评估测试（Repaired Evaluation）**：把评测脚本里偏离基准原意的问题修好——比如恢复官方公式重算、把死板的字符串匹配换成语义等价判分、核查题目里的泄露或残缺项——然后拿同一批进化产物重新测分。

![一句话复现与修复评估后的涨幅对比](/resolution-hypothesis/assets/gain_ledger.png){: style="display: block; max-width: 100%; width: auto; height: auto; max-height: 620px; margin: 1.5rem auto; object-fit: contain; border-radius: 8px;" }

跑出来的数字整齐得几乎有点不好意思：在全部 27 个原涨幅超过 $$3\text{ pp}$$ 的“模型-基准”对里，**只往 Baseline 后面补这一句话，就复现了完整进化算法中位数 90% 的涨幅**（图 3a）；而只要把评测器的问题修好，**中位涨幅立刻从 17.6 pp 降到了 0.8 pp**（图 3b）。哪怕我们在修复后的评测器上把进化算法从头再跑一遍，昔日那些动辄二三十分的涨幅，也再没回来过。

---

比起汇总柱状图，审计过程中翻出来的几个具体案例，其实更能把“进化是怎么跟着分辨率走的”讲透。更多细节都记在我们论文 [arXiv:2611.23068](https://arxiv.org/abs/2611.23068) 的附录 E 里，这里挑三个最有意思的看看。

**先看 SpreadsheetBench 上那道“30 分 vs 6 分”的门槛。** 在 SpreadsheetBench 上，不管是原论文还是我们的复现，进化方法都能拿到 $$+30\sim 40\text{ pp}$$ 的提升。把进化产物打开一看，真正起作用的核心就一句话：_“严禁向单元格写入 Excel 公式字符串，必须在 Python 中算出结果数值后直接填入纯数字或文本。”_

写好好的 Excel 公式怎么会被判错呢？查完评测脚本就明白了：随某个方法一起开源的评测脚本在读取生成的 `.xlsx` 文件时，没有调用电子表格引擎去重新计算公式（Recalculate）。于是，所有没带缓存结果的公式单元格读出来全是空的，直接记 0 分。这种评测脚本里漏掉重算步骤的情况，在工程上其实挺容易不小心写出来；只不过落进进化循环里，就成了最显眼的优化目标。

更有意思的还在修完之后。我们按官方评测标准接入 LibreOffice，在判分前对工作簿做公式重算，同一批进化产物的涨幅（以 GPT-5.4 nano 为例）就从 $$+31.1\text{ pp}$$ 降到了 $$+6.7\text{ pp}$$。为什么还剩大约 $$6\text{ pp}$$ 的正收益？因为有几道题用到了较新的 Excel 函数，连 LibreOffice 也算不出来，所以“用 Python 算好再填数值”在修好的评测器下，依然是一条货真价实能稳赚 $$6\text{ pp}$$ 的策略。

问题也跟着来了：既然修完评测器后，这条规则还是有 $$+6\text{ pp}$$ 的真实收益，为什么我们拿修好的评测器从头重跑进化，算法却学不会它了？

算笔账就清楚了。验证集一共 39 道题，$$+6\text{ pp}$$ 折算下来也就 **2 到 3 道题**的差别，刚好落进单次评估的随机噪声里。结果就是：在旧评测器下（价值 $$30\text{ pp}$$），8 次独立运行全部学会并留住了这一行为（8/8）；换到修复后的评测器下（价值 $$6\text{ pp}$$），我们从头重跑了 12 次，其中一半确实在某一轮提议过这句话，但走到最后真正留下来并照着执行的，只剩 **2 个**（2/12）。一模一样的一条规则，值 $$30\text{ pp}$$ 时次次都能被稳稳接住，降到 $$6\text{ pp}$$ 就悄悄淹没在噪声里——分辨率决定进化能留住什么，大概没有比这更直接的对照了。

**再看 LiveMathematicianBench：模型不肯走捷径，进化就学着去做思想工作。** LiveMathematicianBench 把最新的 arXiv 数学论文定理改写成了五选一选择题。我们在核对数据时注意到一个构建上的巧合：全部 177 道题里，有 85 道题包含一个选项——“其余选项中有一个是正确的，但可以证明一个更强的结果（stronger result）”。只要这个选项出现，它 100% 就是标准答案（测试集 106 道题里占了 46 道）。

原本以为“一句话测试”在这里会直接通关，结果跑出来反而出了个小意外。当我们直接把大白话规则——_“如果某个选项说其余某项正确但能证明更强结果，就选它”_——拼到 Baseline 后面时，三个模型在含该选项的题目上照做的比例分别只有 **4%、54% 和 27%**；而完整进化出来的 Skill，照做比例能达到 **67%、86% 和 92%**。

为什么把答案喂到嘴边，模型还不肯吃呢？大概是因为训练得足够好的基座模型，多少有点“数学自尊心”：面前摆着一道正经的图论或代数定理题，你直白地命令它别管推导、直接挑个含糊的兜底选项，模型在推理时会自己犯嘀咕，干脆不听你的。

等我们把完整进化出来的 Skill 文本逐段拆开，才发现进化循环在多轮迭代里，居然对着基座模型的心理防线做了一整套“修辞优化”：

1. 它先给出一个听起来很像那么回事的学术理由：提醒模型，出题人写的具体数学选项 _“几乎总是被故意弱化表述的（almost always deliberately understated）”_；
2. 接着把举证责任倒置过来：规定除非你能严格证明某个具体选项的界已经是紧致的（sharp），否则默认就该选“更强结果”；
3. 最后把这类选项常见的各种措辞变体全列出来，还在输出最终答案前再叮嘱一遍。

这事想想其实挺耐人寻味的：只要一个捷径的收益大到足以刺穿噪声，选择压力不仅能把它找出来，还会心甘情愿地花上好几轮迭代，专门打磨话术去把基座模型哄顺了。当然，要是把这 46 道题拿掉，只看测试集里剩下的 60 道不含该选项的正常数学题，所有模型的进化收益就全都回到了零附近（$$-1.7\sim 0.0\text{ pp}$$）。

**最后顺带说说审计中碰到的两个基准插曲：SearchQA 和 ALFWorld**（细节见论文附录 E.3 与 E.4）。

SearchQA 的设计初衷，是考察模型从给定的搜索引擎片段（Snippets）里抽取答案的能力。原版评测用了很严格的字面匹配——比如 `"George Calvert (the 1st Lord Baltimore)"` 会被判错，只认更短的写法——于是进化算法靠约束模型输出最短跨度，轻轻松松拿了十几个点的涨幅。可当我们换成语义判分，顺手把题目里提供的**所有搜索片段全部删干净、让模型闭卷裸考**时，DeepSeek-V4-Flash 的得分从 89.4% 变成了 90.2%，GPT-5.6 Luna 从 89.8% 变成了 90.7%。也就是说，把检索证据全撤掉，分数非但没跌，还略微涨了一点点：模型早就能凭预训练记忆答出这些题，而剩下的错题里，倒有 36% 是因为原数据集中 Jeopardy 题目丢掉了分类上下文，成了谁看都猜不准的死题。

ALFWorld 那边则是另一种画风。在 ALFWorld 上，Meta-Harness 和 AHE 这两个代码进化方法在 DeepSeek-V4-Flash 和 GPT-5.6 Luna 上都涨了 $$16\sim 32\text{ pp}$$。我们翻出生成的代码 diff 一看，原因简单得让人没脾气：初始 Baseline Agent 的代码里写死了一个很窄的超参，**只保留最近 2 步的交互历史（History）**。智能体刚走两步，就忘了自己前脚刚搜过哪个抽屉。其中有一次进化运行别的什么逻辑都没改，仅仅把这行历史窗口的限制放开，单步就多拿了 $$+29\text{ pp}$$。起步脚手架顺手写个保守默认值本来也很常见，只是放在进化循环里，最先被优化出来的自然是这行配置，而不是什么高深的空间规划策略。

---

聊完这些案例，倒也没必要把板子打在具体哪篇论文或哪个评测脚本上。写基准、搭脚手架本来就是桩繁琐的脏活，漏掉公式重算、写窄一个历史窗口，或者在几百道题里留下出题模板痕迹，都是研究工程里再正常不过的疏忽。更值得留意的，其实是背后那条很实在的筛选机制：进化循环本身并不区分“真本事”和“评测捷径”，它只认信噪比。哪条改动的收益能稳稳高过噪声底板，优化就会往哪边用力。

而把这些捷径和配置插曲都清理干净之后，摆在面前的图景反而清晰多了：

- 在 DocVQA、AIME 2025、PUPA 这些基准上，修好评测后的 Baseline 本身就已经考到了 $$88\sim 97$$ 分，天花板（Headroom）本来就没剩多少；
- 而在 **Terminal-Bench**、**OfficeQA** 这类真正难啃的基准上，Baseline 离满分还有 $$30\sim 45\text{ pp}$$ 的大片空间，同时也没有什么表面捷径好钻。到了这儿，现有的进化算法就集体慢了下来，增益只剩 $$1.0\sim 3.6\text{ pp}$$，又掉回了单次运行的标准差里。

到这一步，真正核心的问题才算浮出水面：把评测捷径都撤掉之后，面对 Terminal-Bench 这样的硬骨头，大模型到底还写不写得出真正有用的好 Skill？如果它其实写得出来，那些好 Skill 又是在进化循环的哪一步被弄丢的？

在[最后一篇](/blog/2026/resolution-hypothesis-part-3/)里，我们就把提议器（Proposer）和选择器（Selector）彻底拆开，钻进候选池里面找答案。

</div>

<div class="zine-lang-panel is-hidden" data-lang-panel="en" lang="en" markdown="1">

<div class="zine-tldr" markdown="1">

**TL;DR:** Inside an evolution loop, noisy evaluation behaves like a high-pass filter: solid task-specific habits worth 2.5–5 pp get lost in rollout variance, while benchmark-wide evaluation shortcuts clear the noise floor with room to spare. Auditing graders and evolved artifacts line by line across 12 benchmarks, appending a single sentence to the unevolved baseline recovers a median 90% of gains above 3 pp across 27 model–benchmark pairs, and repairing the graders brings the median gain down from 17.6 pp to 0.8 pp. This is not the optimizer misbehaving—it is doing real optimization against whatever signal rises above the noise, whether that means retaining a rule 8/8 times at 30 pp but only 2/12 times at 6 pp on SpreadsheetBench, or spending rounds polishing persuasive copy to talk models into a shortcut on LiveMathematicianBench. Once those artifacts are cleared away, the genuinely hard benchmarks still sit wide open, which sets up the next question: without shortcuts on the table, can models propose useful skills at all, and where do they get lost?

</div>

> **_The Resolution Hypothesis_ Series (2/3)** · [Paper arXiv:2611.23068](https://arxiv.org/abs/2611.23068) · [Project Page & 30s Interactive Game](/resolution-hypothesis/) · [Behind the Scenes](/blog/2026/resolution-hypothesis-behind-the-scenes/)
> **Series Navigation**: [(1) Running in Place & the Variance Ledger](/blog/2026/resolution-hypothesis-part-1/) · **(2) The High-Pass Filter & Shortcut Audit** · [(3) Good Skills & Selectors That Miss Them](/blog/2026/resolution-hypothesis-part-3/)

In [Part 1](/blog/2026/resolution-hypothesis-part-1/), the variance decomposition $$\text{Var}(\hat{Y}) = \sigma^2 + \tau^2$$ left us with a fairly plain accounting fact: a single standard run on an agent benchmark carries a standard deviation of roughly $$1.7\sim 4.1\text{ pp}$$. Two or three percentage points of separation between methods simply disappear into rollout noise.

Still, we left one loose thread at the end of Part 1. Sitting right there in both prior papers and our own reproduction grid are gains that look hard to dismiss: on SpreadsheetBench, LiveMathematicianBench, and AIME 2025, evolved artifacts regularly come out $$10\sim 40\text{ pp}$$ ahead of the baseline—well past any single-run variance band.

Seeing jumps that large, the natural first thought is that evolution must be learning something real on those benchmarks.

Following the logic of _The Resolution Hypothesis_, we opened up the artifacts behind the largest gains and walked through them line by line alongside the actual grader source code. What fell out was a surprisingly plain mechanical effect: **inside an evolution loop, a noisy evaluator works as a high-pass filter.**

---

The intuition takes only a minute of arithmetic. Suppose your validation split has 40 distinct tasks and an evaluation noise standard deviation of $$3\sim 4\text{ pp}$$.

If the proposer distills a **genuinely useful, task-specific engineering habit**—say, checking file encoding before handling missing CSV values—it might turn 1 or 2 failing tasks into passes, a real gain of $$\Delta \approx 2.5\sim 5\text{ pp}$$. Under single-run variance, an improvement that size is easily erased by one unlucky rollout, so the selector rarely notices it was there.

Now ask the flip side: what kind of edit can produce a $$+20\sim 40\text{ pp}$$ jump in a single step, clearing the noise floor cleanly enough for the selector to lock it in?

Almost by necessity, it has to be **a global rule that cuts across task content and moves most of the test set at once.** On existing benchmarks, what delivers that sort of blanket lift is rarely a leap in domain capability—it is usually an **Evaluation Shortcut** matched to a quirk in the grader.

To be fair to the algorithms, finding a shortcut is not "cheating," nor does it mean the optimizer is broken. Quite the opposite: it is real optimization doing exactly what it was asked to do. Whatever signal rises above the noise floor is what selection pressure will push on. The noisier the evaluation, the narrower the odds for modest, local improvements—until the only things that survive are the loudest global shortcuts in the room.

To test that reading, we ran two controlled checks across the largest reported gains on 12 benchmarks:

1. **The One-Rule Test**: Evolved skills and prompts often run to dozens of instructions. Using LLM-assisted screening followed by manual verification against the grader code, we pulled out **a single sentence** and appended it verbatim to the unevolved baseline, touching nothing else.
2. **Repaired Evaluation**: We patched places where the evaluation script departed from the benchmark's intent—restoring official spreadsheet recalculation, swapping brittle string matching for semantic equivalence, and auditing leaked or broken items—then re-scored the exact same evolved artifacts.

![Comparison of reported gains vs. one-rule baseline recovery and repaired evaluation](/resolution-hypothesis/assets/gain_ledger.png){: style="display: block; max-width: 100%; width: auto; height: auto; max-height: 620px; margin: 1.5rem auto; object-fit: contain; border-radius: 8px;" }

The numbers lined up almost too neatly. Across all 27 model–benchmark pairs with an original gain above $$3\text{ pp}$$, **appending that one sentence to the baseline recovered a median 90% of the full evolutionary gain** (Fig. 3a). Once the grader issues were repaired, **the median gain dropped from 17.6 pp to 0.8 pp** (Fig. 3b). Even when we re-ran evolution from scratch under the repaired graders, those headline jumps did not come back.

---

Summary bars tell the macro story, but walking through a few individual benchmarks makes the interaction between resolution and optimization much easier to feel. The full audit sits in Appendix E of [arXiv:2611.23068](https://arxiv.org/abs/2611.23068); three cases in particular are worth lingering over.

**Start with the 30 pp vs. 6 pp threshold on SpreadsheetBench.** On SpreadsheetBench, both earlier work and our reproductions saw evolution add $$+30\sim 40\text{ pp}$$. Opening the evolved artifacts showed that practically the entire lift rested on one instruction: _"Never write Excel formula strings into cells; compute the results in Python and write raw numbers or text directly."_

Why would valid Excel formulas lose points? Reading the evaluation script released alongside one method made it clear: when opening the generated `.xlsx` files, the script never invoked a spreadsheet engine to recalculate formulas. Any formula cell without a cached value was read as empty and scored 0. Forgetting a recalculation pass is an easy enough plumbing slip in an evaluation script; inside an evolution loop, though, it turns into the loudest signal available.

The instructive part came after the repair. Once we restored the official protocol—running LibreOffice to recalculate workbooks before grading—the same evolved artifact's gain on GPT-5.4 nano fell from $$+31.1\text{ pp}$$ to $$+6.7\text{ pp}$$. Why did roughly $$6\text{ pp}$$ of real upside remain? Because a handful of tasks use newer Excel functions that LibreOffice itself cannot evaluate, so computing answers in Python and writing raw values stayed a legitimately useful $$+6\text{ pp}$$ habit even under the repaired grader.

Which leads to the real test: if that exact rule is still worth a genuine $$+6\text{ pp}$$ after the fix, why did fresh evolution runs on the repaired grader stop learning it?

Do the arithmetic and it makes immediate sense. On a 39-task validation split, $$+6\text{ pp}$$ is just **2 to 3 tasks**, sitting right inside single-run sampling variance. Under the broken grader (where the rule paid $$30\text{ pp}$$), **8 out of 8** independent runs discovered and kept it (100%). Under the repaired grader (where it paid $$6\text{ pp}$$), half of our 12 fresh runs did propose the rule at some intermediate step, yet only **2** managed to keep and act on it by the end. The exact same behavioral rule—captured every time at $$30\text{ pp}$$, mostly lost to noise at $$6\text{ pp}$$—is about as direct a demonstration of how resolution governs selection as you could ask for.

**Next, LiveMathematicianBench—where the model hesitated to take a shortcut, so evolution learned persuasion.** LiveMathematicianBench converts theorems from recent arXiv math papers into 5-choice questions. Auditing the dataset turned up a structural quirk: out of 177 problems, 85 include an option reading _"one of the other options is correct, but a stronger result can be proved,"_ and whenever that option appears, it is 100% the ground-truth answer (46 of the 106 test problems fall into this group).

We expected the One-Rule Test to wrap this up in one line, and got a mild surprise instead. When we appended the plain instruction—_"if an option says another option is correct but a stronger result can be proved, select it"_—to the baseline, three models followed it on only **4%, 54%, and 27%** of the applicable problems. The full evolved skills, by contrast, got **67%, 86%, and 92%** compliance.

Why ignore an answer handed over on a plate? Well-trained base models seem to have a quiet streak of mathematical pride: put a serious graph-theory or algebra theorem in front of them and bluntly tell them to skip the derivation and click a vague meta-option, and they tend to push back during reasoning.

Reading the full evolved skill text showed what the loop had actually been doing across rounds—running a miniature **rhetorical optimization** against the base model's hesitation:

1. It offered an academic-sounding premise, informing the model that concrete mathematical options are _"almost always deliberately understated"_ by the problem authors.
2. It flipped the burden of proof, instructing the model to default to the "stronger result" option unless it could rigorously prove a concrete bound was sharp.
3. It listed surface phrasing variants of that option and repeated the instruction right before the final answer was emitted.

There is something dryly funny about watching that play out: once a shortcut's payoff is large enough to clear the noise floor, selection pressure does not just locate the loophole—it happily spends iterations polishing persuasive copy to talk the base model into taking it. Meanwhile, on the remaining 60 normal test problems without that option, evolutionary gains across every model sat right around zero ($$-1.7\sim 0.0\text{ pp}$$).

**Finally, two quieter surprises from the benchmarks themselves: SearchQA and ALFWorld** (full details in Appendix E.3 and E.4).

SearchQA is meant to measure whether a model can extract answers from provided search snippets. Its original grader used strict string matching—marking `"George Calvert (the 1st Lord Baltimore)"` wrong in favor of a bare short span—which let evolution pick up double-digit gains simply by telling the model to keep answers ultra-short. When we switched to semantic grading and **deleted all of the retrieved snippets (forcing closed-book inference)**, DeepSeek-V4-Flash moved from 89.4% to 90.2%, and GPT-5.6 Luna moved from 89.8% to 90.7%. Removing the retrieval evidence nudged scores slightly _up_: models already know these answers from pretraining, and 36% of the remaining errors come from Jeopardy clues in the original dataset that are missing their category context.

ALFWorld turned out to be a different kind of surprise. There, the code-evolution methods Meta-Harness and AHE gained $$16\sim 32\text{ pp}$$ on DeepSeek-V4-Flash and GPT-5.6 Luna. Opening the diffs showed a disarmingly simple cause: the starter baseline agent hardcoded a tight hyperparameter that **kept only the last 2 turns of interaction history**, so after two steps the agent forgot where it had just looked. One evolution run touched nothing else—simply lifting that history window produced a $$+29\text{ pp}$$ gain in a single step. Shipping a starter harness with a conservative default is an ordinary enough engineering artifact, and when an optimizer sees a 29-point lever sitting in plain sight, it will naturally pull it first.

---

None of this is a reason to point fingers at specific papers or benchmark authors. Building evaluation harnesses is messy work; missing a spreadsheet recalculation pass, shipping a narrow history window, or leaving a template artifact across a hundred questions are ordinary hazards of research engineering. The more useful takeaway is the selection dynamic underneath: an evolution loop does not care whether a gain comes from domain mastery or grader plumbing. It only sees signal-to-noise ratio, and it will faithfully optimize whatever clears the floor.

Once those shortcuts and harness quirks are cleared off the table, what remains is much easier to read:

- On DocVQA, AIME 2025, and PUPA, the repaired baselines already score $$88\sim 97\%$$, leaving very little headroom for evolution to begin with.
- On genuinely hard benchmarks like **Terminal-Bench** and **OfficeQA**, baselines still have $$30\sim 45\text{ pp}$$ of real headroom and no surface shortcuts to lean on. There, existing evolution methods mostly stall out, with gains of $$1.0\sim 3.6\text{ pp}$$—back inside a single run's standard deviation.

That leaves us with the question that actually matters: with shortcuts off the table, on hard problems like Terminal-Bench, can LLMs write genuinely useful skills at all? And if they can, where inside the evolution loop do those skills get lost?

In [Part 3](/blog/2026/resolution-hypothesis-part-3/), we take the proposer and the selector apart and look inside the candidate pool.

</div>

<script defer src="/assets/js/bilingual-post.js"></script>
