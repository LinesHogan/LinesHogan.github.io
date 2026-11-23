---
layout: post
title: "我们本来想做一个 Agent 进化算法，最后却给这个领域做了一次全面体检"
title_en: "We Set Out to Build an Agent Evolution Algorithm—and Ended Up Running a Full-Body Diagnostic on the Field"
date: 2026-11-18 12:00:00 +0800
description: "从折腾新的 Agent 元进化算法，到在两百多万条容器轨迹里把评估分辨率拆解清楚：一份《分辨率假说》的幕后手记，以及三个带点个人偏见的判断。（支持 EN / 中文 一键切换，含 TL;DR）"
description_en: "From trying to build a new meta-evolution algorithm to auditing evaluation resolution across 2.1 million trajectories: the story behind The Resolution Hypothesis, plus three personal takes. (Bilingual EN / 中文 with TL;DR)"
tags: agent-evolution evaluation research
categories: research
blog_track: research
blog_badge_zh: "研究幕后"
blog_badge_en: "Behind the Scenes"
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

**TL;DR：** 我们原本想做一个管用的 Agent 进化算法，却先被“循环内曲线一路上涨、公开榜单动辄涨上几十分”的假象困住了。撤掉选择器、盲抽 100 个候选并逐行审计 12 个基准后，背后的测量机制浮出水面：提议池里早就有跨任务平均提升 +11.4 分的好 Skill，失效的是低分辨率选择器——它把小幅度的真实改进滤成了抛硬币，却专门放行一句话就能复现 90% 涨幅的评测漏洞（修好后中位涨幅从 17.6 分跌到 0.8 分）。于是我们放弃 ICLR 截稿、把基建搬上大集群重跑了 210 万条轨迹，写成《分辨率假说》。文末顺带摊开的三个主观判断——离散文本缺少参数空间的记忆性、失明门控会把好 Prompt 磨成“高斯模糊”，以及短期避开老范式、长期留意过程诊断与上下文众包——谈不上看清终局，只是这半年吃瘪之后绕不开的几件事。

</div>

> **论文**：_The Resolution Hypothesis: Agent Evolution's Key Is Identifying Improvements_
> **arXiv**：[2611.23068](https://arxiv.org/abs/2611.23068) · **PDF**：[arXiv PDF](https://arxiv.org/pdf/2611.23068) · **项目主页（含 30s 互动小游戏）**：[Resolution Hypothesis Project Page](/resolution-hypothesis/)
> **三篇连载长文**：[（一）原地踏步与方差拆解](/blog/2026/resolution-hypothesis-part-1/) · [（二）高通滤波器与捷径审计](/blog/2026/resolution-hypothesis-part-2/) · [（三）好 Skill 与挑不出来的选择器](/blog/2026/resolution-hypothesis-part-3/)

半年前刚立项的时候，我们压根没想过要写一篇听起来像在给整个领域泼冷水的论文。最初的想法倒也单纯：做一个管用的 Agent 进化算法。

那阵子社区里关于 Self-Evolving Agents（现在更时髦的叫法是 RSI，递归自我改进）的工作正扎堆冒出来。顺着当时最火的 SkillOpt 一路读下去，我心里一直有点犯嘀咕：在模型权重冻结的前提下，靠自然语言反思去改几段 Prompt 或 Skill 文档，这种所谓的“文本空间优化”，跟直接让大模型把思维链和自我反思写充分一点，到底有什么实质区别？看多了总有一种既视感——像是把 2024 年 TextGrad 那批工作换了个包装，端上来又炒了一遍冷饭。

因为不太习惯这种偏重讲故事的路数，我们一开始想走得更有“机器学习原则”一点：比如借 EM（期望最大化）的思路组织多轮迭代，或者把文本修改做参数化来训练。然而一轮试下来，每条路都没走通——多轮环境交互的归因信号稀薄，离散文本做参数化处处别扭，手头的训练数据也不够。

几条硬桥硬马的路走死之后，我们退了一步：既然试来试去只有类 AutoResearch 的路子跑出来的数字最像样，不如在这上面打磨一套“元进化（Meta-Evolving）”算法。前期的曲线确实让人动心——随着迭代轮次推进，只要对每轮候选得分取个 max，总能画出一条持续向右上方爬升的折线。靠着这点苗头，阶段汇报过得顺利，我们也拿到了继续扩量的预算。

然而一进入扩量阶段，麻烦就来了。

等我们把这套策略搬到更成熟的实战基准，尤其是 Terminal-Bench 这类容器工程任务上时，所有招数——不管是我们自己攒的，还是社区论文里讲得热闹的框架——差不多一起哑了火。表面上循环里的最高分还在慢慢往上走，可把进化出来的 Skill 拿去实测，分数总在初始基线上下晃两三个点，实际用起来几乎看不出区别。眼看节点临近，手里却拿不出在硬任务上站得住脚的策略，组里难免开始犯愁。

---

没辙的时候，我们翻出了 Karpathy 多年前写过的一条老派调试规矩：**怀疑机器学习系统有问题时，先别急着谈泛化，先去 Overfit；如果连单题过拟合都做不到，那多半是管线本身就坏了。**

顺着早期快速验证的思路，我们干脆放下身段，对着具体错题手写直白的提示（Hint）喂给模型。结果出现了一个反常现象：加上特定 Hint 后，那几道题可以稳定通过；可回头翻自动进化出来的普通 Skill 日志时，Hint 触发的那个关键正确动作，在普通 Skill 的某些运行里同样出现过。

同一个正确动作，为什么手写 Hint 能换来稳定的高分，普通进化产物却带不动分数？

问题出在采样方差上，道理简单得令人懊恼：**手写 Hint 能稳定改变模型行为，普通 Skill 只是偶尔撞上那个动作，背后藏着高采样方差。** 当时无论是文本派，还是外面正火的改脚手架代码（Harness）的 RSI，默认做法都是拿验证集跑一次的分数当门控（Gate）。在这层高方差底下，能跨过 Gate 的无非两种情况：要么真改对了行为，要么纯粹是那一次采样手气好；没跨过去的，可能是手气背，也可能是改对了三个必要步骤里的两个、偏偏倒在最后一步。你在循环里画出一条单调不减的折线，说明不了什么。

恰好那周离组会还有几天空档，手头一时没新方向可试，池子里又闲着一批没人用的 API 额度。我们索性做了场带点“行为艺术”意味的实验：**把选择器干脆撤掉，就在 Terminal-Bench 上让模型盲抽提议（Propose）100 个候选 Skill，再砸算力把每个候选重测十几遍，最后用经验贝叶斯模型把每个候选的真实水平估出来。**

跑之前心里也没底——既然选择器这么不靠谱，提议器会不会也一样使不上劲？（当然，如果你让模型对着千禧年数学难题盲提一万次，多半还是零；好在常规软件工程任务尚在当前基座模型的能力射程之内。）结果恰恰相反，数据出乎意料地清晰：**哪怕用最朴素的提议器，候选池里原本就躺着能跨 11 个编程任务平均提升 +11.4 个百分点的好 Skill。** 排头名的 `seed-027` 直指具体问题，摁住了模型在自动化测试下擅自把导出类名 `Server` 改成 `Servicer`、或者手搓空格破坏 BPE 换行分词计数这类违约毛病。

好 Skill 其实早就被写出来了。然而拿同一批候选去考市面上的选择器，结果就难看了：单次执行门控挑中更优候选的概率是 **49.7%**（抛硬币是 50.0%），放行接受的修改里 **62.1%** 是负优化；哪怕请 LLM 当裁判，把几万字的完整轨迹从头读到尾，三种主流裁判模板的准确率也只有 **45.4%–49.0%**——裁判自身的阅读偏差叠加环境随机波动，比直接看跑分还不准。

看到这组数字，我们基本确定：这件事得写出来给社区看看。

---

不过真按论文标准铺开实验——把 7 种主流方法、6 款模型丢进十几个公开基准里跑一遍——一个差点让我们推翻前面判断的现象冒了出来：**在硬核编程基准上大家普遍卡壳，可在 SpreadsheetBench、LiveMathematicianBench、AIME 这几个公开基准上，进化方法却能涨出 20 到 40 分。**

起初我们猜测，大概是软件工程任务方差大、那几个高涨幅基准方差小。然而统计结果恰恰相反——那些基准的单次方差一点也不小。那会儿为了赶 ICLR 截稿，我们甚至花了一周时间，试着用基础正确率和方差去拟合一条类似 Scaling Law 的 **“Evolving Law”** 来预测进化收益，结果画出来的散点图怎么看都是一团乱麻。

机器学习里最朴素的常识又救了我们一次：**底层数据分布都没摸清的时候，在一堆混杂的脏数据上是拟合不出定律的。**

那位同学停下画图，直接扎进这 12 个基准的评测器源码和进化产物里逐行核对，翻出了一连串多少有点幽默的现场：

- **SpreadsheetBench** 涨出三四十分，是因为随开源代码附带的评测脚本在读 `.xlsx` 时漏了公式重算，老老实实写正规公式全判 0 分，进化只是学会了一句“用 Python 算好填纯数字，别写公式”。更有意思的是接入 LibreOffice 修好重算之后：因为极少数新函数仍不支持求值，这句话还留着 **+6.7 分**的真实收益；但在 39 道验证题上，6 分多恰巧跌回了两三道题的采样噪声带——原评测器下（值 30 分）8 次进化 **8/8** 全学会了，修好后重跑 12 次，最终保住这条习惯的只剩 **2/12**。
- **LiveMathematicianBench** 里有 85 道题带着“其余某项正确，但可证明更强结果”这个选项，且它 **100% 是标准答案**。直接附一句话命令基座模型盲选时，模型端着一股“数学自尊心”，服从率最低只有 **4%**；多轮进化却自动磨出了一套颇具学术味的劝降话术——先告诉模型具体选项大都被出题人故意弱化了，再把证明责任倒置过来——愣是把盲选服从率拉到了 **92%**。
- **SearchQA** 里把检索资料 100% 删空、逼模型闭卷裸考，DeepSeek-V4-Flash 和 GPT-5.6 Luna 的分数不降反升；**ALFWorld** 的代码进化单步涨了 29 分，只是因为它删掉了初始代码里硬编码的“只保留 2 步历史”。

统计下来，在全部 27 组高涨幅配置里，往空基线后**只补一句话**就能复现中位 **90%** 的收益；把评测漏洞修好之后，12 个基准的中位涨幅直接从 **17.6 分跌到了 0.8 分**。

到这里，前面两块拼图对上了：**评估系统的分辨率，决定了进化循环究竟能留住什么。** 当一把尺子的单次对比标准差动辄落在 2 到 4 分（我们翻了 9 篇论文里的 171 列打榜结果，中位 SOTA 优势恰好只有 3.4 分，且 0 列给出误差棒；Terminal-Bench 上单次运行从 4 个方法里挑对第一名的概率也只有 33%），小幅度的真实改进会被噪声淹没，能在噪声中稳定可辨识、最终留存下来的，往往只剩通吃全场的评测漏洞。

眼看问题越挖越深，我们做了一个干脆的决定：**放弃 ICLR 截稿，改投 ARR。** 顺带着，为了把早期在单台 64 核 256G 开发机上硬塞 64 个并发 Docker 带来的代理超时和缓存争抢清干净，大家索性把整套管线搬上了公司几百核、几 T 内存的大集群，配足单容器资源与网络白名单，趁着长假把几乎所有实验推倒重跑了一遍。重跑完才发现，光把基建修干净，整体通过率就肉眼可见地往上抬了一截——我现在不免怀疑，市面上不少论文里低得不太合理的 Baseline，测出来的有一半是校园网 NAT 的超时率。这 1,030 次独立进化、210 万条轨迹汇总起来，就是现在的 **《分辨率假说（The Resolution Hypothesis）》**。

> **如果你只有 1 分钟：这篇工作和此前说“评测有噪声”的研究差在哪？**
>
> 1. **不是门控规则调参问题，而是测量分辨率下限**：贪心门控会因为误接受而漂移（循环内 +6.9 至 +9.0 分，8× 重测跌回 +0.4 至 −3.2 分）；LLM 裁判会叠加独立阅读噪声（准确率 45.4%–49.0%）；把门控改保守又会让系统停在原地（修复评测后，HarnessLens 的 71 次运行里有 47 次退回初始版本）。
> 2. **用一台高通滤波器统一解释三个现象**：低分辨率把 +1 至 +3 分的真实进步滤成抛硬币（表现为演化停滞和跨运行榜单洗牌），却稳定放行 +10 至 +40 分的全局评测捷径（一句话复现 90%，修好评测后从 17.6 分跌到 0.8 分）。
> 3. **提议器本来就够用，且因果验证成立**：盲抽 100 个候选里已包含跨任务平均提升 +11.4 分的好 Skill；直接把验证采样提到 16× 拉高分辨率，最优运行即从 64.8% 升至 72.7%。更划算的投入不是继续雕琢 Proposer，而是低方差的过程诊断（可降噪 13×–57×）以及提议与选择解耦的评估方式。

---

具体的实验细节论文里都写全了。收尾之前，我想把这半年折腾下来攒出的三个偏主观的判断也摊开聊聊——倒不是觉得它们有多铁板钉钉，只是踩过这半年坑之后，有些事已经没法装作没看见。

**第一，先别急着拿《The Bitter Lesson》给当下的文本 RSI 贴金——离散文本压根没有参数空间那种“记忆性”。**
靠终端跑分反推怎么改写文本，本质上还是强化学习，样本效率本来就低。传统 RL 在连续权重空间里之所以还能一点点攒出能力，很大程度上是因为几百亿维的向量更新天然带有近似正交的几何性质，新方向叠上去不容易把旧方向抹掉，**参数本身是有记忆的**。离散文本和代码没有这层结构，重写两三轮就把前人的积累覆盖干净了。指望在少样本、高噪声的环境里，靠无记忆的覆写滚出复利，从机制上就说不通。

**第二，好提示词往往得“承担风险（Take Risk）”，而失明的门控只会把东西磨成“高斯模糊”。**
熟手写的 Prompt 之所以管用，是因为人心里明白整个系统的脾气，敢下具体、锋利的硬规矩，同时也清楚边界在哪，不至于把别处改坏。自动进化循环每轮只能从环境里拿到 0 或 1 这么一点反馈，根本估算不出修改的风险边界——真正锋利的改动，只要碰上两次采样手气不好就被淘汰了；最后能混过门控的，多半是四平八稳、谁也不得罪、像蒙了一层“高斯模糊”一样的片汤话。

**第三，短期我会劝新人先避开老范式，但往远看，我依然看好“文本”与任务特化。**
如果今天有刚进组的同学问，要不要顺着“提议—跑分—替换”的老路子开个新坑，我的实话是尽量别往里扎。这条路上留给算法的空间太窄——哪怕暴力把验证采样拉到 16 倍，救回准确率的同时又会把系统推向过度保守，反过来拒斥需要多步推进的探索——辛辛苦苦抠出来的两三分，下一代基座模型一发版顺手就吃掉了。

但把目光放长一点，两条值得留意的小路已经露了头。一是我们在跑完 7 种方法后发现，改代码脚手架（Harness）之所以比纯改文本略占一点上风，是因为它逼着工具和检查流针对具体任务做了**特化**；顺着这个思路，把评估本身也下沉到因题而异的确定性过程指标上（比如改完代码后有没有立刻跑一条不掩盖报错的测试命令），噪声方差当场就能压低 **13 到 57 倍**。二是**上下文的单样本效率与众包潜力**。喂海量语料去训权重，换来的常常只是有限的边际提升，而一段写得对路的文本塞进上下文当场就能起效；更妙的是，全世界每天都在用大模型的普通用户，其实都在天然地生产这类文本经验——如果哪天社区真把筛选和离散空间积累的问题理顺了，这股乘上用户规模的力量恐怕才算上了牌桌。

当然，话说回来，我自己也谈不上看清了终局。怎么在离散空间里做出无损积累，高分辨率的过程尺子具体该怎么造，下一轮范式又会以哪种形态落地，我们手里现在同样没有现成答案。这半年吃瘪攒下来的教训，顶多是帮自己划掉了几条看起来热闹、实则容易原地打转的岔路。真要说眼下有什么踏实事可干，大概还是先把尺子磨准、把基线测干净，老老实实把每一步的真进步和噪声分清楚；至于更远的事，走一步再看一步吧。

---

- **论文全文（arXiv）**：[https://arxiv.org/abs/2611.23068](https://arxiv.org/abs/2611.23068)
- **项目主页、交互式数据浏览器与 30 秒互动小游戏**：[/resolution-hypothesis/](/resolution-hypothesis/)
- **完整实验三部曲**：
  1. [Agent 自进化为什么总在“原地踏步”？](/blog/2026/resolution-hypothesis-part-1/)
  2. [高噪声是个高通滤波器：那些涨了 30 分的 Agent 到底学到了什么？](/blog/2026/resolution-hypothesis-part-2/)
  3. [LLM 写出了好 Skill，我们却挑不出来](/blog/2026/resolution-hypothesis-part-3/)

</div>

<div class="zine-lang-panel is-hidden" data-lang-panel="en" lang="en" markdown="1">

<div class="zine-tldr" markdown="1">

**TL;DR:** We set out to build an agent evolution algorithm that worked, only to get tripped up by the comforting illusion of monotonically rising in-loop curves and headline benchmark jumps. Setting the selector aside to blindly sample 100 candidates—and auditing 12 high-gain benchmarks line by line—exposed the underlying measurement mechanism: genuine +11.4 pp skills were already sitting in the pool, while low-resolution selection turned real gains into coin flips and acted as a high-pass filter for evaluation shortcuts (a single sentence recovered 90% of the gain; repairing the graders dropped the median from 17.6 pp to 0.8 pp). We skipped the ICLR deadline, rebuilt our cluster infrastructure, and re-ran 2.1 million trajectories from scratch as _The Resolution Hypothesis_. The closing takes—on discrete-space memory, "Gaussian blur" gating, and where to place long-term bets—aren't offered as settled forecasts, only as what six months of bruised knuckles made hard to unsee.

</div>

> **Paper**: _The Resolution Hypothesis: Agent Evolution's Key Is Identifying Improvements_
> **arXiv**: [2611.23068](https://arxiv.org/abs/2611.23068) · **PDF**: [arXiv PDF](https://arxiv.org/pdf/2611.23068) · **Project Page (with 30s interactive game)**: [Resolution Hypothesis Project Page](/resolution-hypothesis/)
> **Three-Part Series**: [(1) Running in Place & Variance Decomposition](/blog/2026/resolution-hypothesis-part-1/) · [(2) The High-Pass Filter & Shortcut Audit](/blog/2026/resolution-hypothesis-part-2/) · [(3) Good Skills & Selectors That Miss Them](/blog/2026/resolution-hypothesis-part-3/)

When we started this project six months ago, the last thing we planned to write was a paper that sounded like pouring cold water on an entire subfield. Our goal was plain: build an agent evolution algorithm that works.

At the time, papers on Self-Evolving Agents—lately rebranded under the grander label "RSI," recursive self-improvement—were showing up every week. Reading through the hottest entry, SkillOpt, and the cluster of work around it, I kept running into the same nagging question: with model weights frozen, using natural-language reflection to rewrite a few paragraphs of a prompt or skill file—how is this "optimization in text space" substantively different from asking the model to think more carefully and reflect harder? The more papers we read, the harder it was to shake a familiar sense of déjà vu. It felt like taking the 2024 wave of TextGrad papers, swapping out the packaging, and reheating the same dish.

None of us were drawn to pure storytelling, so we first tried taking a more principled machine-learning route: structuring multi-round updates like an EM loop, or finding some clean way to parameterize text edits so we could train them properly. Every attempt stalled for mundane reasons: credit assignment across multi-turn environment trajectories is thin, parameterizing discrete text is awkward at every seam, and we had too little training data.

Once those principled detours ran dry, we stepped back. Since AutoResearch-style loops seemed to be the only setup producing plausible numbers, we decided to build a tighter "meta-evolving" algorithm on top of them. Early plots were easy to fall for: round after round, as long as you plotted the running max of candidate scores, you got a tidy line marching up and to the right. That early trajectory got us through milestone reviews and earned us the compute budget to scale up.

Unfortunately, trouble arrived the moment we scaled up.

As soon as we moved the loop onto mature benchmarks and containerized engineering suites like Terminal-Bench, everything stalled together—not just our own method, but the widely cited frameworks from the literature too. Inside the loop, the best-so-far curve kept nudging upward; on held-out evaluation, the evolved skills hovered two or three points around the baseline, with no noticeable difference in practice. With deadlines getting closer and nothing holding up on hard tasks, the mood in the room turned uneasy.

---

Out of ideas, we fell back on an old debugging rule Andrej Karpathy wrote years ago: **when you suspect a machine learning system is broken, stop worrying about generalization and try to overfit first; if you cannot even overfit a single example, the pipeline itself is broken.**

Borrowing from our early sanity checks, we swallowed our pride and hand-wrote specific hints for individual failed tasks, feeding them straight into the model. What came back was an odd contrast. With the hand-written hint, those tasks started passing reliably—yet when we dug back through the logs of ordinary auto-evolved skills, the same correct action triggered by our hint had already shown up in some of their rollouts too.

Why did the same correct action produce steady scores behind a hand-written hint while ordinary evolved skills went nowhere?

The culprit was sampling variance, and the explanation was almost embarrassingly simple: **the hand-written hint reliably shifted the model's behavior, whereas an ordinary skill only stumbled into it occasionally.** At the time, the default recipe across both text-evolution methods and popular harness-editing RSI frameworks was the same: run a single pass on a validation set and use that score as an acceptance gate. Under that much noise, crossing the gate only tells you one of two things—either you improved the behavior, or you got lucky on that rollout. Failing the gate might mean bad luck, or it might mean you fixed two of three necessary steps and tripped on the last one. A monotonically non-decreasing in-loop curve proves little on its own.

As it happened, we had a few spare days before group meeting, no obvious next idea, and a pile of idle API credits sitting in the account. So we ran a slightly absurd performance-art experiment: **set the selector aside altogether, let the model blindly propose 100 candidate skills on Terminal-Bench, spend the compute to evaluate each one a dozen-plus times, and use an Empirical-Bayes model to wash out the noise and see what was in the pool.**

Going in, we had no idea what to expect—if the selector was blind, maybe the proposer was just as weak. (Obviously, if you ask a model to blindly propose proofs for Millennium Prize Problems, 10,000 draws will still yield zero; fortunately, routine software engineering is within frontier models' reach.) When the numbers settled, the verdict was plain: **even with a bare-bones proposer, the pool already contained skills that beat the baseline by an average of +11.4 percentage points across 11 coding tasks.** The top skill, `seed-027`, targeted concrete contract violations—stopping the model from renaming an exported `Server` class to `Servicer` under automated tests, or hand-joining strings with spaces in ways that silently broke BPE newline tokenization counts.

Good skills were already being written. When we turned around and tested the field's selectors on the same pool, however, the numbers were hard to defend: a single-rollout execution gate picked the better candidate **49.7%** of the time (a coin flip is 50.0%), and **62.1%** of the edits it accepted were worse than the baseline. Even when we handed tens of thousands of words of full execution trajectories to an LLM judge, three standard judging templates scored only **45.4%–49.0%**—the judge's own reading noise stacked right on top of environment randomness, leaving it even less reliable than the raw score.

With those numbers on the table, it was clear we needed to write this up for the community.

---

Once we expanded to full paper scale—7 mainstream methods, 6 models, and more than a dozen public benchmarks—a result turned up that nearly upended our thesis: **everyone stalled on hard coding benchmarks, yet on SpreadsheetBench, LiveMathematicianBench, and AIME, evolution methods climbed by +20 to +40 points.**

Our first guess was that software engineering tasks had higher variance while those high-gain benchmarks had lower variance. Unfortunately, the statistics said the opposite—their single-run variance was not small at all. Trying to make the ICLR deadline, we even spent a week trying to fit a Scaling-Law-style **"Evolving Law"** from base accuracy and variance to predict evolutionary gains. The scatter plot remained an unreadable cloud.

Once again, the most basic instinct in machine learning saved us: **you cannot fit a clean law to a contaminated mixture whose underlying distribution you have not inspected.**

That teammate stopped plotting, went straight into the grader source code and evolved artifacts of all 12 benchmarks line by line, and came back with findings that were, at the very least, mildly ironic:

- **SpreadsheetBench** climbed 30 to 40 points because the open-source evaluation script shipped with one method read `.xlsx` files without triggering formula recalculation, grading every valid Excel formula as 0. Evolution simply learned one habit: "compute the numbers in Python and write raw values, never formulas." The better twist came after we repaired the grader with LibreOffice recalculation: because a handful of newer Excel functions still aren't supported by LibreOffice, that same rule retained a real **+6.7 pp** advantage. Yet across 39 validation tasks, 6 points is only two or three tasks—right back inside the sampling noise band. Under the broken grader (worth 30 pp), **8 out of 8** independent runs discovered and kept the rule; under the repaired grader (worth 6 pp), only **2 out of 12** managed to hold onto it.
- **LiveMathematicianBench** had 85 questions containing an option phrased as "one of the other options is correct, but a stronger result can be proved"—and whenever it appeared, it was **100% the answer key**. When we bluntly appended a one-sentence command telling the base model to always pick it, the models' "mathematical pride" kicked in, with compliance dropping as low as **4%**. Multi-round evolution automatically refined a bit of academic persuasion—telling the model that concrete options were almost always deliberately understated by the problem authors, then inverting the burden of proof—coaxing compliance up to **92%**.
- **SearchQA**: when we deleted 100% of the retrieved search snippets and forced the models to take the test closed-book, both DeepSeek-V4-Flash and GPT-5.6 Luna scored _higher_. **ALFWorld**: code evolution jumped 29 points in a single step by removing a hardcoded cap in the starter agent that threw away all interaction history beyond the last 2 turns.

Across all 27 high-gain model–benchmark pairs, appending **a single sentence** to the unevolved baseline recovered a median **90%** of the full evolutionary gain. Once the graders were repaired, the median gain across all 12 benchmarks dropped from **17.6 pp to 0.8 pp**.

That brought both halves of the puzzle into the same frame: **the resolution of the evaluation system dictates what evolution can keep.** Measured with a ruler whose single-comparison standard deviation is 2 to 4 points (across 171 leaderboard entries in 9 papers, the median SOTA margin is just 3.4 pp, with 0 reporting error bars, and single-run top-1 identification among 4 methods on Terminal-Bench is only 33%), small, genuine improvements get washed out, and only benchmark-wide loopholes stand out reliably above the noise.

Seeing how deep the measurement issue ran, we made a clean call: **drop the ICLR deadline and target ARR instead.** At the same time, to rule out proxy timeouts and cache contention from our early setup—cramming 64 concurrent Docker containers onto a single 64-core, 256 GB dev box—we moved the pipeline onto a multi-hundred-core, multi-terabyte cluster with generous per-container resources and proper network whitelists, and spent the holiday re-running almost every experiment from scratch. Simply cleaning up the infrastructure lifted overall pass rates; I now suspect a fair share of the suspiciously low baselines in published papers are measuring campus NAT timeout rates. Those 1,030 independent evolution runs and 2.1 million trajectories became **_The Resolution Hypothesis_**.

> **If you only have 1 minute: how does this differ from earlier papers saying "evaluation is noisy"?**
>
> 1. **It is a measurement resolution floor, not a gating-rule tweak**: greedy gates drift via false acceptance (+6.9 to +9.0 pp in-loop vs. +0.4 to −3.2 pp on 8× re-eval); LLM judges add independent reading noise (45.4%–49.0% accuracy); and making gates more conservative simply stalls evolution at initialization (47/71 HarnessLens runs returned to the initial artifact under repaired eval).
> 2. **One high-pass filter explains three mysteries at once**: low resolution turns real +1 to +3 pp improvements into coin flips (causing stagnation and unstable rankings across runs) while selectively passing +10 to +40 pp global shortcuts (90% recovered by one sentence; 17.6 pp → 0.8 pp after repair).
> 3. **Proposals already work, and the causal test holds**: 100 blind draws already contain +11.4 pp cross-task skills, and raising validation resolution directly to 16× rollouts lifts the best run from 64.8% to 72.7%. The better investment is not fancier proposers, but low-variance process diagnostics (13×–57× lower noise) and decoupled proposer/selector evaluation.

---

All the experimental numbers are laid out in the paper. Before wrapping up, I want to put three subjective takeaways from the past six months on the table—not as settled truths, just as things that six months of bruised knuckles made hard to unsee.

**First, stop borrowing _The Bitter Lesson_ to flatter today's text-space RSI—discrete text has none of the memory geometry of parameter space.**
Reverse-engineering edits from terminal scores is reinforcement learning, which is sample-inefficient by nature. The reason standard RL can gradually accumulate capabilities in continuous weight space is that updates in tens-of-billions-dimensional vectors are nearly orthogonal: stacking a new direction does not easily erase old ones. **Weights have built-in memory.** Discrete text and code do not; two or three rounds of rewriting overwrite whatever came before. Expecting compounding returns from memoryless overwrites under tiny sample budgets and heavy noise runs against the basic mechanics of the problem.

**Second, good prompts have to take risks, while blind gates grind everything into a "Gaussian blur."**
Experienced engineers write effective prompts because they understand the machinery underneath: they dare to give specific, opinionated directives while knowing where the boundaries are so they do not break everything else. An automated loop receiving a single 0 or 1 bit of environment feedback per task cannot price that risk. Sharp edits get filtered out after two unlucky rollouts, and what safely clears the gate is cautious, inoffensive boilerplate—smoothed over like a layer of Gaussian blur.

**Third, in the short term I would steer newcomers away from the old paradigm—yet long term, I still believe in text and task specialization.**
If a new student joined the lab today asking whether to start a project on the classic "propose–score–replace" loop, my honest advice would be to stay clear of it. The headroom left there is too narrow—even brute-forcing 16× validation rollouts rescues accuracy only by introducing conservatism that rejects multi-step exploration—and whatever two or three points you squeeze out will likely be absorbed by the next base model release anyway.

Looking further out, though, two quieter paths look worth watching. First, across our 7 methods, the reason harness-editing slightly edged out pure text editing is that it forced tools and checks to **specialize** to the task at hand; carry that same logic down into evaluation with task-specific, deterministic process metrics (such as checking whether the agent immediately ran an unmasked test command after editing code), and evaluation noise variance drops by **13× to 57×** on the spot. Second, **the one-shot sample efficiency and crowdsourcing potential of context**. Feeding mountains of tokens into weight updates buys modest marginal gains, whereas a single well-aimed paragraph dropped into context takes effect right away—and millions of everyday LLM users around the world are already generating textual know-how for free every day. If the community eventually figures out how to filter signal from noise and accumulate knowledge in discrete space, that force, multiplied by user scale, could become real.

Even so, stepping back, I would not pretend to know how the endgame plays out. How to achieve lossless accumulation in discrete space, what high-resolution process rulers will look like in practice, or what shape the next paradigm will take—we do not have ready-made answers to any of those yet. At best, the past six months of bruised knuckles crossed a few busy-looking dead ends off our map. In the meantime, sharpening the ruler, keeping baselines honest, and separating real gains from sampling noise one step at a time is probably enough to keep us busy.

---

- **Full Paper (arXiv)**: [https://arxiv.org/abs/2611.23068](https://arxiv.org/abs/2611.23068)
- **Project Page, Interactive Explorer & 30s Game**: [/resolution-hypothesis/](/resolution-hypothesis/)
- **The Three-Part Deep Dive**:
  1. [Why Does Agent Self-Evolution Keep Running in Place?](/blog/2026/resolution-hypothesis-part-1/)
  2. [High Noise Is a High-Pass Filter: What Did Those +30-Point Agents Actually Learn?](/blog/2026/resolution-hypothesis-part-2/)
  3. [LLMs Already Write Good Skills—We Just Can't Pick Them Out](/blog/2026/resolution-hypothesis-part-3/)

</div>

<script defer src="/assets/js/bilingual-post.js"></script>
