---
layout: post
title: "我们本来想做一个 Agent 进化算法，最后却给这个领域做了一次全面体检"
title_en: "We Set Out to Build an Agent Evolution Algorithm—and Ended Up Running a Full-Body Diagnostic on the Field"
date: 2026-11-18 12:00:00 +0800
description: "从尝试设计新的 Agent 元进化算法，到在百万级容器轨迹里抓出评估盲区：一份《分辨率假说》的幕后手记与三个个人判断。"
description_en: "From trying to design a new meta-evolution algorithm to uncovering evaluation blind spots across 2.1 million trajectories: the story behind The Resolution Hypothesis, plus three candid personal takes. (Bilingual EN / 中文 with TL;DR)"
tags: agent-evolution evaluation research
categories: research-notes
thumbnail: /resolution-hypothesis/assets/optimism_drift.png
related_posts: true
---

<div class="zine-dispatch" data-title-zh="我们本来想做一个 Agent 进化算法，最后却给这个领域做了一次全面体检" data-title-en="We Set Out to Build an Agent Evolution Algorithm—and Ended Up Running a Full-Body Diagnostic on the Field">
  <a class="zine-pill-link" data-i18n-zh="论文 arXiv:2611.23068 ↗" data-i18n-en="Paper · arXiv:2611.23068 ↗" href="https://arxiv.org/abs/2611.23068" target="_blank" rel="noopener">论文 arXiv:2611.23068 ↗</a>
  <div class="zine-lang-switch" role="group" aria-label="Language">
    <button type="button" class="zine-lang-btn" data-lang-target="en" aria-pressed="false">EN</button>
    <button type="button" class="zine-lang-btn is-active" data-lang-target="zh" lang="zh-CN" aria-pressed="true">中文</button>
  </div>
</div>

<div class="zine-lang-panel" data-lang-panel="zh" lang="zh-CN" markdown="1">

<div class="zine-tldr" markdown="1">

**TL;DR：** 我们本来想做一个真正管用的 Agent 进化算法，却在硬核编程任务上集体撞墙，逼着自己做了一场“扔掉选择器、盲抽 100 个候选”的实验——结果发现池子里早躺着跨任务稳涨 +11.4 分的好 Skill，真正失明的是堪比抛硬币的选择器。顺藤摸瓜审计 12 个暴涨基准后，我们又揪出一连串评测捷径（一句话复现 90% 涨幅，修好后从 17.6 分跌到 0.8 分），索性放弃 ICLR 截稿、把百万轨迹推倒重跑写成《分辨率假说》。文末还摊开了三个更主观的判断：离散文本没有参数空间的记忆性、失明门控会把好 Prompt 磨成“高斯模糊”，以及短期避坑老范式、长期押注高分辨率过程尺子与上下文众包的理由。

</div>

> **论文**：_The Resolution Hypothesis: Agent Evolution's Key Is Identifying Improvements_
> **arXiv**：[2611.23068](https://arxiv.org/abs/2611.23068) · **PDF**：[arXiv PDF](https://arxiv.org/pdf/2611.23068) · **项目主页（含 30s 互动小游戏）**：[Resolution Hypothesis Project Page](/resolution-hypothesis/)
> **三篇连载长文**：[（一）原地踏步与方差账](/blog/2026/resolution-hypothesis-part-1/) · [（二）高通滤波器与捷径审计](/blog/2026/resolution-hypothesis-part-2/) · [（三）好 Skill 与失明的选择器](/blog/2026/resolution-hypothesis-part-3/)

半年前刚立项的时候，我们压根没想过要写一篇给整个领域“泼冷水”的论文。我们最初的目标特别单纯：做出一个真正管用的 Agent 进化算法。

当时社区里关于 Self-Evolving Agents（现在更时髦的叫法是 RSI，递归自我改进）的工作正扎堆涌现。我们顺着最火的 SkillOpt 一路调研下来，心里其实一直犯嘀咕：在模型权重冻结的前提下，靠自然语言反思去改写几段 Prompt 或 Skill 文档，这种所谓的“文本空间优化”，跟直接让大模型把思维链和自我反思写扎实一点到底有什么本质区别？看多了总有一种强烈的既视感——很像把 2024 年 TextGrad 那批工作换了个壳，端上来重炒了一遍冷饭。

因为骨子里不太喜欢这种纯讲故事的路数，我们起初想走得更有“机器学习原则”一点：比如借类似 EM（期望最大化）的思路来组织多轮迭代，或者干脆把文本修改做某种合理的参数化来训练。结果试了一圈全撞了南墙——多轮环境交互的归因信号太稀薄，离散文本参数化极其别扭，训练数据也捉襟见肘。

几条硬桥硬马的路走死之后，我们退了一步：既然试来试去只有类 AutoResearch 的路子效果看着最像样，不如在这上面打磨一套“元进化（Meta-Evolving）”算法。前期的曲线相当诱人——随着迭代轮次推进，只要对每轮生成的候选得分取个 max，你总能画出一条一路向右上方爬升的漂亮折线图。靠着这个苗头，阶段汇报很顺利，我们也拿到了继续扩量的支持。

直到真正的麻烦找上门来。

当我们开始扩量，并且把这套策略搬到成熟起来的实战基准和 Terminal-Bench 这种硬核容器工程任务上时，所有招数——不管是我们自己攒的，还是社区论文里吹得神乎其神的框架——全都失灵了。表面上循环里的最高分还在爬坡，可把进化出来的 Skill 拿去实测，分数永远在初始基线上下晃荡两三个点，实际用起来毫无可见影响。眼看节点临近，交不出真正在硬任务上站得住脚的策略，所有人都坐不住了。

### 一、从“先 Overfit”到那场盲抽实验

走投无路的时候，我们想起了 Karpathy 多年前写过的一条老派调试规矩：**怀疑机器学习系统有问题时，先别谈泛化，先去 Overfit；如果连单题过拟合都做不到，说明管线本身就是坏的。**

受早期做快速验证时的启发，我们干脆放下身段，对着具体错题手写极度直白的提示（Hint）硬喂给模型。结果发现了一件怪事：加上特定 Hint 后某几道题确实稳稳过了；可回头翻其他自动进化出来的普通 Skill 日志时，Hint 促成的关键正确动作，其实在普通 Skill 的某几次运行里也出现过！

为什么同一个正确动作，手写 Hint 能拿到稳稳的高分，普通进化产物却拿不到？

想通的那一刻真有点懊恼：**手写 Hint 做到的是“稳定扭转模型行为”，而普通 Skill 只是“偶尔撞上那个行为”，方差大得惊人。** 当时无论是文本派还是外面炒得很热的改脚手架代码（Harness）的 RSI，核心共识都是“拿实机跑一次验证集的分数做门控（Gate）”。但在这层高方差底下，能跨过 Gate 的无非两种情况：要么真改对了行为，要么纯粹是采样撞了大运；没跨过去的，可能是手气背，也可能是改对了三个必要动作里的两个、偏偏倒在了最后一步。你在循环里画出一条单调不减的曲线，什么都证明不了。

恰好那周离组会还有几天空档，手头一时没新方向可试，池子里又闲着一批没人用的 API 额度。我们干脆做了一个带点“行为艺术”色彩的实验：**把选择器彻底扔掉，就在 Terminal-Bench 上让模型纯瞎提议（Propose）100 个候选 Skill，然后砸算力对每个候选重测十几遍，用经验贝叶斯模型把真实底牌洗出来。**

做之前心里完全没底——既然选择器这么瞎，提议器会不会也一样菜？（当然，如果你拿千禧年数学难题让模型瞎提议，抽一万次也是零；好在常规软件工程任务还在当前基座模型的能力射程之内。）结果跑出来的那一刻，全组都震住了：**哪怕用最朴素的提议器，候选池里明明就躺着能跨 11 个编程任务平均稳涨 +11.4 个百分点的真东西！**（比如排头名的 `seed-027`，精准摁住了模型在自动化测试下擅自把导出类名 `Server` 美化成 `Servicer`、手搓空格破坏 BPE 换行分词计数等违约毛病。）

金子早就被写出来了，但回过头考考市面上的选择器：单次执行门控挑准更优候选的概率是 **49.7%**（抛硬币是 50.0%），放行接受的修改里 **62.1%** 其实是负优化；哪怕请 LLM 当裁判把几万字轨迹从头读到尾，三种主流裁判模板的准确率也只有 **45.4%–49.0%**——裁判自己的眼花叠加上环境随机波动，反而更瞎了。

那一刻我们意识到，这大概是一篇必须写出来给社区看的论文。

### 二、画不出的“Evolving Law”与推倒重跑的决定

真正按论文标准铺开 7 种主流方法、6 款模型去跑十几个公开基准时，一个差点掀翻我们立论的现象出现了：**在硬核编程基准上大家确实集体卡壳，可在 SpreadsheetBench、LiveMathematicianBench、AIME 这一批公开基准上，进化方法真的能哗哗暴涨 20 到 40 分！**

起初我们猜“是不是 SWE 任务方差大、那些暴涨基准方差小”？组里同学跑完统计回来直摇头——那些基准的单次方差一点也不小。当时为了赶 ICLR 截稿，我们甚至花了一周时间，想用基础正确率和方差拟合一条类似 Scaling Law 的 **“Evolving Law”** 去预测进化收益，结果画出来的散点图永远是一团乱麻。

机器学习最朴素的常识又救了我们一次：**你不可能在一个没搞懂底层分布的混合脏数据上拟合出定律。**

那位同学停下画图，直接扎进那 12 个基准的评测器源码和进化产物里逐行对账，结果抓出了一连串让人哭笑不得的现场：

- **SpreadsheetBench** 暴涨三四十分，是因为伴随开源的评测脚本读 `.xlsx` 时漏了公式重算，写正规公式全判 0 分，进化只是学会了一句“Python 算好填纯数字，别写公式”。最绝的是接入 LibreOffice 修好重算之后：因极少数新函数不支持求值，这句话其实还剩 **+6.7 分**的真收益；但在 39 道验证题上，6 分刚巧跌回两三道题的采样噪声带——原评测器下（值 30 分）8 次进化 **8/8** 全学会，修好后重跑 12 次进化，最终保住这一习惯的只剩 **2/12**！
- **LiveMathematicianBench** 里有 85 道题包含“其余某项正确，但可证明更强结果”且 **100% 是标准答案**。直接用一句话命令基座模型盲选它时，模型端着“数学自尊心”最低只有 **4%** 肯听话；多轮进化却自动磨出一套学术劝降话术（先骗模型具体选项都被出题人故意弱化了，再倒置证明责任），硬把盲选服从率哄到了 **92%**。
- **SearchQA** 把检索资料 100% 删空逼模型闭卷裸考，DeepSeek-V4-Flash 和 GPT-5.6 Luna 分数不降反升；**ALFWorld** 代码进化单步暴涨 29 分，仅因解除了初始代码硬编码的“只保留 2 步历史”。

在全部 27 组高涨幅配置里，往空基线后**只加一句话**就能复现中位 **90%** 的收益；修好评测漏洞后，12 个基准中位涨幅直接从 **17.6 分跌到 0.8 分**。

这时候两块拼图彻底咬合死了：**评估系统的分辨率决定了进化能留住什么。** 在单次对比标准差动辄 2~4 分（文献 171 列打榜结果的中位 SOTA 优势恰好只有 3.4 分，且 0 列给误差棒，Terminal-Bench 单次挑对第一名概率仅 33%）的粗糙尺子下，细水长流的真改进全被滤掉了，只有通吃全场的评测漏洞能刺穿噪声活下来。

看着越挖越深的数据，我们做了一个很干脆的决定：**不赶 ICLR 死线了，改投 ARR。** 同时，为了彻底排除早期我们在单台 64 核 256G 开发机上硬塞 64 个并发 Docker 带来的出口代理超时和缓存争抢干扰，全组咬牙把管线搬上公司几百核几 T 内存的大集群，配足单容器资源与网络白名单，趁着长假把几乎所有实验推倒重跑了一遍（重跑后发现光把基建修干净，整体通过率就涨了一截——我现在严重怀疑市面上不少论文里奇低无比的 Baseline，测出来的其实是一半校园网 NAT 超时率）。这 1,030 次独立进化、210 万条轨迹攒下来的账本，就是现在的 **《分辨率假说（The Resolution Hypothesis）》**。

> **如果你只有 1 分钟：这篇工作和此前说“评测有噪声”的研究到底差在哪？**
>
> 1. **不是门控规则问题，而是测量分辨率下限**：贪心门控会误接受漂移（循环内 +6.9 至 +9.0 分，8× 重测为 +0.4 至 −3.2 分），LLM 裁判会叠加阅读噪声（准确率 45.4%–49.0%），而把门控改保守只会停滞于起点（修复评测下 HarnessLens 71 次运行有 47 次退回初始版本）。
> 2. **一台高通滤波器统一解释三大悬案**：低分辨率把 +1 至 +3 分的真实进步滤成抛硬币（导致演化停滞与跨运行榜单洗牌），却专门放行 +10 至 +40 分的全局评测捷径（一句话复现 90%，修好评测从 17.6 分跌到 0.8 分）。
> 3. **提议已够强，且因果验证成立**：盲抽 100 个候选里已经躺着跨任务稳涨 +11.4 分的好 Skill；直接用 16× 验证采样拉高分辨率后，最优运行从 64.8% 跃升至 72.7%。真正该投入的不是更复杂的 Proposer，而是低方差的过程诊断（降噪 13×–57×）与提议/选择解耦评估。

### 三、账本之外，三个更主观的判断

实验数据论文里都有，最后我想把这半年折腾下来几个带点个人偏见的结论摊开聊聊：

**第一，别拿《The Bitter Lesson》给现在的文本 RSI 贴金——离散文本根本没有参数空间的“记忆性”。**
靠终端跑分反推修改本质上是强化学习，样本效率本就极低。传统 RL 在连续权重空间里之所以能慢慢攒出本事，是因为几百亿维向量更新有近似正交的几何性质，新方向叠上去不会轻易抹平旧方向，**参数天然是有记忆的**。离散文本和代码没有这种记忆性，重写两轮就把前人覆盖了；指望在极少样本、极高噪声下靠无记忆覆写攒出复利，本身就违背直觉。

**第二，好提示词需要“承担风险（Take Risk）”，而失明的门控只会制造“高斯模糊”。**
聪明人写 Prompt 管用，是因为人懂整个系统的运行机理，敢下极具体、极锋利的死命令，同时自己兜得住不误伤别处的风险。自动进化系统每轮只拿到 0/1 那一丁点环境反馈，根本消不掉风险——锋利的修改碰上两次采样坏运气就被淘汰，最终能安全混过门控的，全是四平八稳、谁也不得罪、像加了一层“高斯模糊”一样的片汤话。

**第三，短期劝新人避开老范式，但长期我依然相信“文本”与任务特化。**
如果今天有刚进组的同学问要不要顺着“提议—跑分—替换”的老路子开新坑，我的实话是**千万别碰**——留给你的空间太窄（哪怕暴力加码到 16 倍验证采样，救活分数的同时又会陷入拒斥多步探索的过度保守），辛辛苦苦抠出两三分，下一代基座模型一发版顺手就吃掉了。

但往远看，两条真正有戏的路已经露头了：一是我们在跑完 7 种方法后发现，改代码脚手架（Harness）之所以比纯改文本略占上风，是因为它逼着工具和检查流针对具体题目做了**特化**——顺着这个逻辑把评估也下沉到因题而异的确定性过程指标上（比如改完代码是否立刻跑了未掩盖报错的测试），噪声方差当场就能砍掉 **13 到 57 倍**；二是**上下文的单样本效率与众包潜力**。喂海量语料训权重换来的只是可怜的边际提升，而一段文本塞进上下文当场就能生效，且全世界每个正在用大模型的普通用户每天都在天然生产文本经验。

等哪天社区真正造出了高分辨率的新尺子，解决了怎么筛、怎么在离散空间里无损积累的问题，这股乘上用户规模的力量才算真正上了牌桌。

---

- **论文全文（arXiv）**：[https://arxiv.org/abs/2611.23068](https://arxiv.org/abs/2611.23068)
- **项目主页、交互式数据浏览器与 30 秒互动小游戏**：[/resolution-hypothesis/](/resolution-hypothesis/)
- **完整实验三部曲**：
  1. [Agent 自进化为什么总在“原地踏步”？](/blog/2026/resolution-hypothesis-part-1/)
  2. [高噪声是个高通滤波器：那些暴涨 30 分的 Agent 到底学到了什么？](/blog/2026/resolution-hypothesis-part-2/)
  3. [LLM 写出了好 Skill，我们却挑不出来](/blog/2026/resolution-hypothesis-part-3/)

</div>

<div class="zine-lang-panel is-hidden" data-lang-panel="en" lang="en" markdown="1">

<div class="zine-tldr" markdown="1">

**TL;DR:** We started out trying to build an agent evolution algorithm that actually worked, hit a wall on real software-engineering tasks, and ended up throwing away the selector to blindly sample 100 candidate skills—only to find genuine +11.4 pp gems already sitting in the pool while every selector was effectively flipping a coin. Auditing 12 benchmarks with massive reported gains then uncovered a trail of evaluation shortcuts (a single sentence recovered 90% of the gain; repairing the graders dropped the median from 17.6 pp to 0.8 pp), convincing us to skip the ICLR deadline, rebuild the infrastructure, and re-run 2.1M trajectories from scratch as _The Resolution Hypothesis_. At the end, I lay out three blunter personal takes: discrete text lacks the memory geometry of continuous weights, blind gating blurs sharp prompts into bland mush, and the real long-term bet is high-resolution process diagnostics paired with in-context adaptation.

</div>

> **Paper**: _The Resolution Hypothesis: Agent Evolution's Key Is Identifying Improvements_
> **arXiv**: [2611.23068](https://arxiv.org/abs/2611.23068) · **PDF**: [arXiv PDF](https://arxiv.org/pdf/2611.23068) · **Project Page (with 30s interactive game)**: [Resolution Hypothesis Project Page](/resolution-hypothesis/)
> **Three-Part Series**: [(1) Running in Place & the Variance Ledger](/blog/2026/resolution-hypothesis-part-1/) · [(2) The High-Pass Filter & Shortcut Audit](/blog/2026/resolution-hypothesis-part-2/) · [(3) Good Skills & Blind Selectors](/blog/2026/resolution-hypothesis-part-3/)

When we kicked off this project six months ago, the last thing on our minds was writing a paper that pours cold water on an entire subfield. Our goal was dead simple: build an agent evolution algorithm that actually works.

At the time, papers on Self-Evolving Agents—now dressed up under the flashier label "RSI," recursive self-improvement—were coming out in droves. Reading through the hottest entry, SkillOpt, and everything around it, a nagging doubt kept creeping in: with model weights frozen, using natural-language reflection to edit a few paragraphs of a prompt or skill file—how is this "optimization in text space" fundamentally different from just asking the model to think harder and reflect more carefully? The more papers we read, the stronger the déjà vu became. It felt a lot like taking the 2024 wave of TextGrad papers, swapping out the packaging, and reheating the same dish.

Because none of us were particularly fond of pure storytelling, we first tried to take a more principled machine-learning route: structuring multi-round updates in an EM-like loop, or finding some clean way to parameterize text edits so we could actually train them. Every single attempt ran straight into a brick wall. Credit assignment across multi-turn environment trajectories is razor-thin, parameterizing discrete text is deeply awkward, and training data was nowhere near enough.

Once the principled routes hit dead ends, we took a step back. Since AutoResearch-style loops seemed to be the only thing producing plausible numbers, we figured we would build a sharper "meta-evolving" algorithm on top of them. Early plots looked intoxicating: round after round, as long as you took the running max of candidate scores, you got a crisp line marching steadily up and to the right. That early promise got us smoothly through milestone reviews and earned us the compute budget to scale up.

Then the real trouble showed up.

The moment we scaled up and moved our strategy onto more mature, battle-tested benchmarks and hard containerized engineering suites like Terminal-Bench, everything fell apart—not just our own method, but every celebrated framework from the literature too. Inside the loop, the best-so-far curve kept climbing; on held-out evaluation, the evolved skills just wobbled two or three points around the baseline, with zero perceptible difference in practice. With deadlines closing in and nothing standing up on hard tasks, everyone in the room started getting restless.

### I. From "Overfit First" to the Blind-Sampling Experiment

Cornered, we fell back on an old debugging maxim Andrej Karpathy wrote years ago: **when you suspect a machine learning system is broken, forget generalization and try to overfit first; if you can't even overfit a single example, the pipeline itself is broken.**

Taking a cue from our early sanity checks, we swallowed our pride and hand-wrote brutally explicit hints for specific failed tasks, force-feeding them straight into the model. What came back was bizarre. With the hand-written hint, those tasks started passing reliably—yet when we dug back through logs of ordinary auto-evolved skills, the exact right action triggered by our hint had already shown up in some of their rollouts too!

Why did the same winning action produce rock-solid scores with a hand-written hint, while ordinary evolved skills went nowhere?

The moment it clicked was genuinely frustrating: **the hand-written hint reliably shifted the model's behavior, whereas an ordinary skill only stumbled into that behavior occasionally, buried under huge variance.** At the time, the shared dogma across both text-evolution methods and the heavily hyped harness-editing RSI frameworks was simple: run one pass on a validation set and use the score as an acceptance gate. Under that much noise, crossing the gate only means one of two things—either you actually improved the behavior, or you got lucky on a rollout. Failing the gate might mean bad luck, or it might mean you fixed two out of three necessary steps and tripped on the last one. Plotting a monotonically non-decreasing in-loop curve proves nothing at all.

As luck would have it, we had a few spare days before group meeting, no obvious next idea, and a pile of idle API credits sitting around. So we ran a slightly absurd, performance-art kind of experiment: **throw the selector in the trash, let the model blindly propose 100 candidate skills on Terminal-Bench, throw heavy compute at evaluating every single one a dozen-plus times, and use an Empirical-Bayes model to wash out the noise and see what was really in the pool.**

Going in, we had no idea what would come out—if the selector was blind, maybe the proposer was useless too. (Obviously, if you ask a model to blindly propose proofs for Millennium Prize Problems, 10,000 draws will still yield zero; fortunately, day-to-day software engineering is well within frontier models' reach.) When the numbers landed, the whole group went quiet: **even with the most bare-bones proposer imaginable, the pool already contained genuine skills that beat the baseline by an average of +11.4 percentage points across 11 coding tasks!** (The top skill, `seed-027`, nailed concrete contract violations—stopping the model from creatively renaming an exported `Server` class to `Servicer` under automated tests, or hand-joining strings with spaces in ways that silently broke BPE newline tokenization counts.)

The gold was already sitting in the pan. So we turned around and tested the field's selectors on that exact pool: a single-rollout execution gate picked the better skill **49.7%** of the time (a coin flip is 50.0%), and **62.1%** of the edits it accepted were actually worse than the baseline. Even when we handed tens of thousands of words of full execution trajectories to an LLM judge, three standard judging templates scored just **45.4%–49.0%**—the judge's own reading noise stacked on top of environment randomness, making it even blinder than before.

Right then we knew: this was a paper we had to write for the community.

### II. The "Evolving Law" We Couldn't Fit—and the Decision to Re-Run Everything

Once we scaled up properly—7 mainstream methods, 6 models, over a dozen public benchmarks—a result showed up that nearly blew up our entire thesis: **everyone stalled on hard coding benchmarks, yet on SpreadsheetBench, LiveMathematicianBench, and AIME, evolution methods genuinely surged by +20 to +40 points!**

At first we guessed that SWE tasks simply had higher variance while those surging benchmarks had lower variance. A teammate ran the numbers and shook his head—their single-run variance wasn't small at all. Trying to make the ICLR deadline, we even spent a week trying to fit a Scaling-Law-style **"Evolving Law"** from base accuracy and variance to predict evolutionary gains. The scatter plot stayed an indecipherable cloud.

Once again, the most basic instinct in machine learning bailed us out: **you cannot fit a clean law to a contaminated mixture whose underlying distribution you haven't inspected.**

That teammate stopped plotting, dove straight into the grader source code and evolved artifacts of all 12 benchmarks line by line, and came back with a gallery of absurd discoveries:

- **SpreadsheetBench** jumped 30–40 points because the open-source evaluation script shipped with a method read `.xlsx` files without triggering formula recalculation, grading every valid Excel formula as 0. Evolution simply learned one rule: "compute the numbers in Python and write raw values, never formulas." The punchline came after we repaired the grader with LibreOffice recalculation: because a handful of newer Excel functions still aren't supported by LibreOffice, that exact rule kept a real **+6.7 pp** advantage. Yet across 39 validation tasks, 6 points is just two or three tasks—right back inside the sampling noise floor. Under the broken grader (worth 30 pp), **8 out of 8** independent runs discovered and kept the rule; under the repaired grader (worth 6 pp), only **2 out of 12** runs managed to hold onto it!
- **LiveMathematicianBench** had 85 questions containing an option phrased as "one of the other options is correct, but a stronger result can be proved"—and whenever it appeared, it was **100% the answer key**. When we bluntly appended a one-sentence command telling the base model to always pick it, the models' "mathematical pride" kicked in, with compliance dropping as low as **4%**. Multi-round evolution automatically refined a piece of academic persuasion—telling the model that concrete options were almost always deliberately understated by the problem authors, then flipping the burden of proof—coaxing compliance all the way up to **92%**.
- **SearchQA**: when we deleted 100% of the retrieved search snippets and forced the models to take the test closed-book, both DeepSeek-V4-Flash and GPT-5.6 Luna actually scored _higher_. **ALFWorld**: code evolution jumped 29 points in a single step purely by removing a bizarre hardcoded cap in the starter agent that threw away all interaction history beyond the last 2 turns.

Across all 27 high-gain model–benchmark pairs, appending **one single sentence** to the unevolved baseline recovered a median **90%** of the full evolutionary gain. Once the graders were repaired, the median gain across all 12 benchmarks collapsed from **17.6 pp to 0.8 pp**.

That snapped the two halves of the puzzle together: **the resolution of the evaluation system dictates what evolution can keep.** Measured with a ruler whose single-comparison standard deviation is 2–4 points (the median SOTA margin across 171 reported leaderboard entries in 9 papers is just 3.4 pp, with 0 reporting error bars, and single-run top-1 identification among 4 methods on Terminal-Bench is only 33%), steady, genuine improvements get washed out, and only benchmark-wide loopholes can punch through the noise floor.

Staring at how deep the rabbit hole went, we made a clean call: **drop the ICLR deadline and target ARR instead.** At the same time, to completely rule out proxy timeouts and cache contention from our early days of cramming 64 concurrent Docker containers onto a single 64-core, 256 GB dev box, the whole team bit the bullet, migrated the pipeline onto a multi-hundred-core, multi-terabyte cluster with generous per-container resources and clean network whitelists, and spent the holiday re-running almost every experiment from scratch. (Simply fixing the infrastructure bumped overall pass rates noticeably—I now strongly suspect a non-trivial share of the suspiciously low baselines in published papers are really measuring campus NAT timeout rates.) That ledger—1,030 independent evolution runs and 2.1 million trajectories—became **_The Resolution Hypothesis_**.

> **If you only have 1 minute: how does this differ from earlier papers saying "evaluation is noisy"?**
>
> 1. **It is a measurement resolution floor, not a gating-rule tweak**: greedy gates drift via false acceptance (+6.9 to +9.0 pp in-loop vs. +0.4 to −3.2 pp on 8× re-eval), LLM judges add independent reading noise (45.4%–49.0% accuracy), and making gates more conservative simply freezes evolution at initialization (47/71 HarnessLens runs returned to the initial artifact under repaired eval).
> 2. **One high-pass filter explains three mysteries at once**: low resolution turns real +1 to +3 pp improvements into coin flips (causing stagnation and unstable rankings across runs) while selectively passing +10 to +40 pp global shortcuts (90% recovered by one sentence; 17.6 pp → 0.8 pp after repair).
> 3. **Proposals already work, and the causal test holds**: 100 blind draws already contain +11.4 pp cross-task skills, and raising validation resolution directly to 16× rollouts lifts the best run from 64.8% to 72.7%. The bottleneck to invest in isn't fancier proposers, but low-variance process diagnostics (13×–57× lower noise) and decoupled proposer/selector evaluation.

### III. Beyond the Ledger: Three More Subjective Takes

All the numbers are in the paper. To wrap up, let me put three frankly opinionated takeaways from the past six months on the table:

**First, stop invoking _The Bitter Lesson_ to flatter today's text-space RSI—discrete text has none of the memory geometry of parameter space.**
Reverse-engineering edits from terminal scores is reinforcement learning, which is notoriously sample-inefficient. The reason standard RL can still slowly accumulate capabilities in continuous weight space is that updates in tens-of-billions-dimensional vectors are nearly orthogonal—stacking a new direction doesn't easily erase old ones. **Weights have built-in memory.** Discrete text and code do not; two rounds of rewriting overwrite whatever came before. Expecting compounding returns from memoryless overwrites under tiny sample budgets and massive noise runs straight against basic intuition.

**Second, great prompts have to take risks, while blind gates grind everything into a "Gaussian blur."**
Sharp engineers write effective prompts because they understand the machinery underneath: they dare to issue razor-specific, opinionated directives while bearing the judgment call not to break everything else. An automated loop receiving a single 0/1 bit of environment feedback per task cannot price that risk—sharp edits get killed off by two unlucky rollouts, and what survives the gate is safe, inoffensive boilerplate, smoothed over like a layer of Gaussian blur.

**Third, in the short term I'd tell newcomers to steer clear of the old paradigm—but long term, I still believe in text and task specialization.**
If a new student joined the lab today asking whether to open a project on the classic "propose–score–replace" loop, my honest answer is **don't touch it**. The margin left there is razor-thin (even brute-forcing 16× validation rollouts rescues accuracy only to introduce conservatism that rejects multi-step exploration), and whatever two or three points you scrape out will get swallowed whole by the next frontier base model release.

Looking further out, though, two genuinely promising paths are emerging. First, across our 7 methods, the reason harness-editing slightly edged out pure text editing is that it forced tools and checks to **specialize** to the task at hand—carry that same logic down into evaluation with task-specific, deterministic process metrics (such as checking whether the agent ran an unmasked test command right after editing code), and evaluation noise variance drops by **13× to 57×** on the spot. Second, **the one-shot sample efficiency and crowdsourcing potential of context**. Feeding mountains of tokens into weight updates buys tiny marginal gains, whereas a single paragraph dropped into context takes effect immediately—and millions of everyday LLM users around the world are already generating textual know-how for free every day.

The day the community builds high-resolution rulers—solving how to filter signal from noise and accumulate knowledge without loss in discrete space—that force, multiplied by user scale, will finally take its seat at the table.

---

- **Full Paper (arXiv)**: [https://arxiv.org/abs/2611.23068](https://arxiv.org/abs/2611.23068)
- **Project Page, Interactive Explorer & 30s Game**: [/resolution-hypothesis/](/resolution-hypothesis/)
- **The Three-Part Deep Dive**:
  1. [Why Does Agent Self-Evolution Keep Running in Place?](/blog/2026/resolution-hypothesis-part-1/)
  2. [High Noise Is a High-Pass Filter: What Did Those +30-Point Agents Actually Learn?](/blog/2026/resolution-hypothesis-part-2/)
  3. [LLMs Already Write Good Skills—We Just Can't Pick Them Out](/blog/2026/resolution-hypothesis-part-3/)

</div>

<script defer src="/assets/js/bilingual-post.js"></script>
