---
layout: post
title: "解码尾声的带宽账，与 5 行 IEEE 754 保序位变换"
title_en: "Bandwidth Accounting at the Decode Tail—and a 5-Line IEEE 754 Order-Preserving Bit Trick"
date: 2026-11-23 03:07:00 +0800
description: "从 top-p 过滤后平均仅剩 4.83 个幸存词出发降维词表计算；对照 IEEE 754 位图推导单条 uint64 整数比较实现“浮点降序 + 同分下标升序”，并提炼“锁种子逐词对账”与“盯批次最大耗时行”两把排障尺子（pLLM 技术专栏 · 支持 EN / 中文 一键切换）"
description_en: "Autoregressive decode ends in a memory-bandwidth bottleneck over the vocabulary matrix, where a few innocent lines of logit rescoring can run slower than lm_head itself. (pLLM Engineering Series · Bilingual EN / 中文)"
tags: pllm vllm inference-systems
categories: systems
pllm_synced: true
pllm_order: 4
pllm_badge_zh: "极客拆解 · 尾声与位运算"
pllm_badge_en: "Deep Dive · Bit Tricks & Tail Bandwidth"
related_posts: true
---

<div class="zine-dispatch" data-title-zh="解码尾声的带宽账，与 5 行 IEEE 754 保序位变换" data-title-en="Bandwidth Accounting at the Decode Tail—and a 5-Line IEEE 754 Order-Preserving Bit Trick">
  <div class="zine-dispatch-links">
    <a class="zine-pill-link" data-i18n-zh="pLLM 项目主页与文档" data-i18n-en="pLLM Project &amp; Docs" href="/pllm/">pLLM 项目主页与文档</a>
    <a class="zine-pill-link" data-i18n-zh="GitHub · pLLM ↗" data-i18n-en="GitHub &middot; pLLM ↗" href="https://github.com/LinesHogan/pLLM" target="_blank" rel="noopener">GitHub · pLLM ↗</a>
  </div>
  <div class="zine-lang-switch" role="group" aria-label="Language">
    <button type="button" class="zine-lang-btn" data-lang-target="en" aria-pressed="false">EN</button>
    <button type="button" class="zine-lang-btn is-active" data-lang-target="zh" lang="zh-CN" aria-pressed="true">中文</button>
  </div>
</div>

<div class="zine-lang-panel" data-lang-panel="zh" lang="zh-CN" markdown="1">

<div class="zine-tldr" markdown="1">

**TL;DR：** 自回归解码的最后一步被词表矩阵的显存带宽卡死，几行看似无害的词表重打分代码就能跑得比 `lm_head` 投影本身还慢。顺着剖析器定位瓶颈、统计 `top-p` 过滤后的真实幸存候选数，可以把全词表矩阵乘降维成对少数幸存行的按需点积；而为了在 GPU 寄存器内用单条整数指令完成“浮点得分降序 + 同分下标升序”的复合比较，借用基数排序里的 IEEE 754 保序位变换即可在 5 行代码内解决。文末复盘如何靠“锁种子逐词对账”与“盯批次最大值而非均值”两项习惯，把隐蔽的舍入错位和长尾毛刺提前揪出来。

</div>

> **pLLM 技术随笔与机制拆解（4/10 · 极客拆解 · 尾声与位运算）** · [pLLM 项目介绍与文档中心](/pllm/) · [专栏目录与系统解剖图](/pllm/#blog) · [仓库源码原文 ↗](https://github.com/LinesHogan/pLLM/blob/main/blog/zh/04-sampler-tail-and-bit-hack.md)

往大模型解码循环末尾加一点自定义采样逻辑——比如用在线训练的轻量小网络预测值 $$g$$ 去压低主模型隐藏状态 $$h$$ 里的平庸候选词——公式通常只有一行：在主模型做完 `top-k` / `top-p` 过滤留下的候选词上，按 $$(1 + \beta) s_i - \beta g_i$$ 重新打分，再抽样生成下一个词。

在研究脚本里，最顺手的写法是让 $$h$$ 和 $$g$$ 各自乘一次词表投影矩阵 `lm_head`，等过滤器把淘汰位置填成 `-inf` 后，调一行 `torch.where` 算出新得分：

```python
scores = torch.where(torch.isfinite(s), (1.0 + beta) * s - beta * pred, s)
```

把这段代码挂进并发推理引擎跑一轮性能剖析，时间线会立刻给你一个下马威：前面二十多层 Transformer 都跑完了，最后这几行和词表打交道的收尾代码，花掉的 GPU 时间居然比跑一次 `lm_head` 本身还要长。

---

怎么从现象看穿病根？先把解码最后一步的数据流与显存体积画出来。

拿 7B 模型、128 条并发解码请求来说，词表大小约 15.2 万，隐藏层维度 3584。数据在进入最终抽样前要穿过这样一条带宽漏斗：

```text
  隐藏状态 h [128, 3584] (不足 1 MB)
        │
        ▼  乘 lm_head 权重 [3584, 152064] (BF16, 约 1.1 GB 显存读取!)
  原始词表得分 [128, 152064]
        │
        ▼  转为采样工作区 FP32 张量 (约 78 MB 显存读写)
  温度缩放 + top-k / top-p 过滤 (淘汰词置为 -inf)
        │
        ▼
  Gumbel-Max 随机抽样 ──► 产出 128 个新 token_id
```

盯着图上的字节数看，瓶颈一目了然。每步只有 128 行时，`[128, 3584] × [3584, 152064]` 的浮点乘加次数对现代 Tensor Core 来说根本塞不满牙缝，耗时几乎全花在把 **1.1 GB** 权重从显存搬进芯片上。出了 `lm_head` 之后，采样器面对的 `[128, 152064]` 单精度工作张量也有近 **80 MB**。

朴素写法等于在最拥堵的路段连跑了两趟冤枉路：先为引导向量 $$g$$ 重读了一遍 1.1 GB 的 `lm_head` 权重；接着那一行 `torch.where` 在未融合的即时执行下被拆成 `isfinite`、类型转换、乘法、减法、选择等七个逐元素小内核，在 80 MB 的大表上来回读写了七遍。

问题建模到这里就清楚了：在被显存带宽锁死的解码尾声，**优化目标不是减少算术指令，而是把新增的全词表显存往返次数压成零**——凡是能塞进现有必经访存寄存器里顺手算完的算术，几乎都是免费的。

---

怎么省掉第二遍读那 1.1 GB 权重？如果换个问题问自己：**经过 `top-p = 0.95`（温度 0.8）过滤之后，那 152,064 列里到底还有几个词活着？**

与其拍脑袋猜，不如直接在真实生成流里插个计数器。我们在十几万行解码数据上统计了每行过滤后非 `-inf` 的候选词数量，分布高度偏斜：

```text
152,064 个词经过 top-p = 0.95 过滤后的幸存词数量分布：
  • 平均值：        4.83 个词  （超过 99.99% 的列已被置为 -inf）
  • 65 万行最大值： 2,785 个词 （即便在最发散的位置，也不足总词表的 2%）
```

看到 `4.83` 这个数字，算法结构自然就翻过来了：既然绝大多数时候只有四五个候选词能活过过滤，先用 $$g$$ 去乘全部 15 万列权重、再把 99.997% 的乘积扔掉，纯属无用功。

把计算顺序倒过来——等上游过滤器跑完，趁着扫描过滤结果的那一趟，把幸存词的得分和下标收集进一小块固定暂存区；随后只拿这四五个下标去 `lm_head` 权重矩阵里按行读取（5 行权重一共才 35 KB），跟 $$g$$ 做五次点积改写得分，紧接着在同一个内核里完成随机抽样。

---

顺着这条路动手写 Triton 内核时，会遇到一个具体的底层障碍。

在收集幸存词、求峰值、以及当某行幸存词偏多需要调 `tl.sort` 截断前 $$K$$ 名时，必须遵守确定性的复合决胜规则（tie-breaking）：**先按浮点得分 `logit`（`float32`）从大到小排；若两个词得分相等，按词表下标 `token_id`（`uint32`）从小到大排。**

麻烦在于，GPU 寄存器原语（如 Triton 内置的 `tl.maximum` 和 `tl.sort`）只支持对**单个标量张量**做比较，没法像写 C++ `std::sort` 那样传一个比较两个字段的自定义 lambda。如果拆成两轮掩码比较，寄存器占用和分支指令都会膨胀。

怎么让硬件用一条普通的整数比较指令 `a > b`，同时排对一个有符号浮点数和一个反向决胜的整数？

这个问题的灵感来自两个经典的底层工程领域：一个是图形学与高性能计算里的**浮点数基数排序**（Michael Herf 在 2001 年总结的 `FloatFlip` 技巧，也是 NVIDIA CUB `RadixSort` 处理浮点数的标准做法），另一个是数据库存储引擎（如 RocksDB / MySQL）构造**可按字节 `memcmp` 比较的复合索引键（memcomparable key）**——核心思想都是构造一个从 IEEE 754 浮点数到无符号整数的**保序双射（order-preserving bijection）**。

画一张 IEEE 754 `float32` 的内存比特布局图，推导过程一眼就能看清：

```text
IEEE 754 float32 比特布局（共 32 位）：
  31 30          23 22                                           0
 ┌──┬──────────────┬──────────────────────────────────────────────┐
 │S │ Exponent(8b) │                 Mantissa (23b)               │
 └──┴──────────────┴──────────────────────────────────────────────┘
  ▲
  └── 符号位 S：0 代表正数，1 代表负数（注意：不是补码，而是“符号 + 绝对值”！）

把原始比特直接当成无符号整数 uint32 看时，实数轴是拧着的：
  数值从小到大：   -inf  ──────►  -0.0   │   +0.0  ──────►  +inf
  原始比特(hex)： 0xFF800000 ──► 0x80000000 │ 0x00000000 ──► 0x7F800000
                  (S=1，越负比特反而越大，反序！) (S=0，越正比特越大，本来就保序！)

做一次对称位变换后，整根实数轴变成单调递增的无符号整数：
  • 负数 (S=1) -> 32 位全部按位取反 (~bits)：最高位变 0 落入下半区，且反序变正序！
  • 正数 (S=0) -> 仅最高位异或翻转 (^ 0x80000000)：最高位变 1 升入上半区，保持正序！

  变换后(hex)：   0x007FFFFF ──► 0x7FFFFFFF │ 0x80000000 ──► 0xFF800000
                     (-inf)        (-0)         (+0)           (+inf)
```

看上面这张图，为什么这套位运算如此工整？因为 IEEE 754 设计指数偏置（exponent bias）的初衷，本来就是为了让正浮点数能直接用整数比较器比大小；唯一的障碍是符号位和负数的绝对值反序。把正数翻最高位送去上半区、把负数 32 位全取反送去下半区（变换前先把 `-0.0` 归成 `+0.0` 消除正负零双表示），所有实数就严格按大小排在了 `uint32` 上。

再往外走一步，把变换后的 32 位保序浮点左移塞进 `uint64` 的高 32 位，低 32 位塞进 `0xFFFFFFFF - token_id`（把“下标越小越优先”翻成无符号整数越大），顺手把 `0` 留给已被过滤淘汰的 `-inf`：

```text
复合排序键 uint64（单条整数指令 key_a > key_b 即可完成复合决胜）：
  63                              32 31                               0
 ┌──────────────────────────────────┬──────────────────────────────────┐
 │     保序变换后的 float32 得分     │       0xFFFFFFFF - token_id      │
 └──────────────────────────────────┴──────────────────────────────────┘
        高 32 位决定得分降序                同分时低 32 位决定下标升序
                     （全 0 专用于已淘汰位置，自动垫底）
```

把整套推导落成 Triton 代码，编码与解码对称工整，加起来不到十行：

```python
@triton.jit
def _keys(logits: tl.tensor, tokens: tl.tensor, finite: tl.tensor) -> tl.tensor:
    logits = tl.where(logits == 0, 0.0, logits)
    bits = logits.to(tl.uint32, bitcast=True)
    ordered = tl.where(bits & 0x80000000 != 0, ~bits, bits ^ 0x80000000)
    return tl.where(finite, (ordered.to(tl.uint64) << 32) | (0xFFFFFFFF - tokens.to(tl.uint32)), 0)

@triton.jit
def _decode(keys: tl.tensor) -> tuple[tl.tensor, tl.tensor]:
    ordered = (keys >> 32).to(tl.uint32)
    bits = tl.where(ordered & 0x80000000 != 0, ordered ^ 0x80000000, ~ordered)
    return bits.to(tl.float32, bitcast=True), (0xFFFFFFFF - keys.to(tl.uint32)).to(tl.int32)
```

---

内核跑通之后，怎么确信它在线上几十万步并发生成里既没算错、又没有隐藏的性能毛刺？这里分享两条在手写定制 GPU 内核时普遍管用的排障方法——我们后面抓到的所有隐蔽问题，全是被这两把尺子量出来的。

**第一把尺子：用“锁死随机种子的逐词精确对账”，代替肉眼看生成通不通顺。**

写完融合内核后，如果只是随便跑几条提示词看英文通不通顺，99% 的数值瑕疵都会蒙混过关。可靠的做法是锁死随机种子，拿融合内核跟朴素的全词表参考实现连跑几十万个词做逐词比对，只要有一个词分叉就停下来看那一行发生了什么。

刚对账时，前面几千个词一字不差，跑到某一步突然整句分道扬镳。把分叉那一行的候选分布拉出来才看明白：我们的内核在寄存器里用高精度 `float32` 算完了点积（入口偏置融合内核同理），而原厂未量化的 `lm_head` 输出在落盘时经历过一次 `bfloat16` 舍入。平时这点尾数差无所谓，偏偏那一行有个边缘词的累积概率刚好卡在 `top-p = 0.95` 截断线上——少了一次 `bfloat16` 舍入，原本该出局的词留在了池子里，导致下游 Gumbel 随机抽样多消耗或错开了一个随机数，后面整句话自然全变了。

定位到根因后，修法只在寄存器里加半行：点积或累加完故意做一次 `.to(bf16).to(fp32)` 还原原厂的舍入台阶，并直接内联调用上游现成的 `gumbel_block_argmax` 抽样函数。重跑近三十万个词的对账，不匹配数归零。

**第二把尺子：在批处理 GPU 内核里，永远盯“批次最大耗时行（p99.9）”，别被平均值骗了。**

前面统计过，平均每行只有 `4.83` 个幸存词。按平均思维，把每行暂存区静态容量 `K` 设成 `256` 看似富余了五十倍。可拿真实语料做吞吐压测时，剖析器里隔几十步就会冒出一个拉长整步耗时的毛刺，把 `K` 放大到 `4096` 反倒比 `256` 更快。

把拖慢整批的那几行单独拎出来检查，同时抓出了两件光看平均值绝对想不到的事：

- 一是**批处理的木桶效应**：128 行请求在同一个内核网格里并发执行，整步完成时间取决于 `max(row_0, ..., row_127)` 而非均值。哪怕 127 行都只有 5 个词，只要撞上 1 行发散词数超过 `256` 掉进多趟 `tl.sort` 溢出慢路，其余 127 个线程块全得陪着干等。多花 `4 MiB` 显存把 `K` 设到盖住经验上界（`2,785`）的 `4096`，本质上是花小钱买断长尾慢分支；
- 二是抓出了**占千分之一的“未掩码全满行”**：有少部分触发慢分支的行，打开一看根本不是模型发散，恰恰相反，模型在那个位置笃定到了极点——第一名的概率超过 `0.9999999`，光它自己就跨过了 `top_p = 0.95`，于是上游过滤器走了提前返回的捷径，压根没往数组里写 `-inf`！看穿上游这条捷径后，在扫描时顺手累加指数和 `mass`，发现整行无 `-inf` 且峰值单点满足 `1.0 / mass >= top_p` 时直接取唯一峰值词点积，毛刺随之消失。

_(附带一个用同样方式抓显存水位爬升查出来的 PyTorch 注册陷阱：如果你把变长阶段的 Triton 内核用 `torch.library.wrap_triton` 包装并传入 `tuple[Tensor, ...]`，底层分发器会把张量元组当成编译期常量缓存在全局侧表里，每跑一步就永久攥住那一步的临时张量；改用标准 `@torch.library.custom_op` 接收 `list[Tensor]` 才会被当作普通运行期操作数释放。)_

---

这套从带宽漏斗出发的尾声融合，目前作为 `@pllm.guide_sampling` 与 `@pllm.add_logits` 的默认路径跑在我们的开源 vLLM 扩展层 **[pLLM](https://github.com/LinesHogan/pLLM)**（`pllm`）里，让 7B 模型在 128 并发下边跑每请求在线训练边做采样引导，仍能维持原生引擎 96% 以上的吞吐。

最后看一眼这套方案失效的边界：按词表行号去 `lm_head` 稀疏取行，前提是完整的 `lm_head` 权重未切分、未做块交错量化地躺在当前卡上。一旦开启多卡张量并行（`TP > 1`）切分了词表维，或使用了块打包量化权重，单卡线程块无法独立完成那几次稀疏点积；此时就得切回“把 $$h$$ 与 $$g$$ 在行维拼成两倍长、合乘一次 `lm_head`”的稠密回退路径——虽然省不掉矩阵乘，但至少保住了 1.1 GB 权重只从显存搬运一次的底线。

</div>

<div class="zine-lang-panel is-hidden" data-lang-panel="en" lang="en" markdown="1">

<div class="zine-tldr" markdown="1">

**TL;DR:** Autoregressive decode ends in a memory-bandwidth bottleneck over the vocabulary matrix, where a few innocent lines of logit rescoring can run slower than `lm_head` itself. Measuring how many tokens actually survive `top-p` filtering turns a full-vocabulary GEMM into on-demand dot products over a handful of surviving rows; packing `(float32 logit, uint32 token_id)` into a single `uint64` integer comparison takes a 5-line IEEE 754 order-preserving bijection borrowed from radix sort. We close with two reusable kernel-debugging habits—seed-locked token-by-token parity and profiling batch `max` rather than batch mean—that surface subtle BF16 rounding shifts and tail stragglers early.

</div>

> **pLLM Engineering Essays & Deep Dives (4/10 · Deep Dive · Bit Tricks & Tail Bandwidth)** · [pLLM Project & Docs Hub](/pllm/) · [Series Index & Anatomy Map](/pllm/#blog) · [Markdown Source on GitHub ↗](https://github.com/LinesHogan/pLLM/blob/main/blog/en/04-sampler-tail-and-bit-hack.md)

Adding custom sampling guidance to the end of an LLM decode loop—say, using an online-trained lightweight predictor $$g$$ to suppress bland candidate tokens in the trunk hidden state $$h$$—usually takes one line of math: over the candidate set surviving the trunk's `top-k` / `top-p` filter, rescore each surviving token as $$(1 + \beta) s_i - \beta g_i$$ before drawing the next token.

In a research script, the natural implementation projects both $$h$$ and $$g$$ through `lm_head`, waits for the sampler filter to mask rejected columns to `-inf`, and calls `torch.where`:

```python
scores = torch.where(torch.isfinite(s), (1.0 + beta) * s - beta * pred, s)
```

Drop that snippet into a concurrent serving engine and open a profiler trace, and the timeline delivers an immediate surprise: after more than twenty Transformer layers finish, those last few vocabulary lines burn more GPU time than running `lm_head` itself.

---

To diagnose why, draw the dataflow and byte volumes at the end of a decode step before reading any kernel code.

Take a 7B model serving 128 concurrent decode requests with a vocabulary of `152,064` and hidden dimension `3,584`. Every step pushes data through this bandwidth funnel:

```text
  Hidden state h [128, 3584] (< 1 MB)
        │
        ▼  Multiply by lm_head weights [3584, 152064] (BF16, ~1.1 GB HBM read!)
  Raw vocabulary logits [128, 152064]
        │
        ▼  Materialize FP32 sampler workspace (~78 MB HBM read/write)
  Temperature scaling + top-k / top-p mask (rejected columns set to -inf)
        │
        ▼
  Gumbel-Max stochastic draw ──► 128 next token_ids
```

Staring at the byte counts makes the bottleneck obvious. At 128 rows per step, the arithmetic in `[128, 3584] × [3584, 152064]` barely warms up modern Tensor Cores; runtime is dominated by streaming **1.1 GB** of weights out of HBM. Once past `lm_head`, the FP32 `[128, 152064]` sampler workspace spans nearly **80 MB**.

The naive snippet pays that toll twice over on the most congested stretch of the step: first streaming the 1.1 GB `lm_head` matrix a second time for $$g$$, then letting eager PyTorch decompose `torch.where` into seven elementwise kernels (`isfinite`, cast, multiply, subtract, select, and friends) that sweep back and forth across the 80 MB workspace seven times.

That reframes the optimization problem cleanly: at the bandwidth-bound decode tail, **the goal is not trimming arithmetic instructions—it is driving extra full-vocabulary HBM round-trips to zero**. Any math piggybacked inside registers during an already-mandatory memory pass is effectively free.

---

How do you avoid reading that 1.1 GB `lm_head` matrix a second time? Flip the question around: **after `top-p = 0.95` (at temperature `0.8`) finishes masking the vocabulary, how many of those `152,064` columns are actually still alive?**

Instead of guessing, instrument a live decode stream and count finite entries per row. Across `652,000` real decode rows, the survivor count is heavily skewed:

```text
Surviving candidates out of 152,064 after top-p = 0.95 filtering:
  • Mean:               4.83 tokens  (> 99.99% of columns are already -inf)
  • Max over 652k rows: 2,785 tokens (< 2% of the vocabulary even at peak entropy)
```

Seeing `4.83` turns the algorithm upside down: when four or five candidates survive on a typical step, multiplying $$g$$ against all `152,064` columns only to throw away `99.997%` of the dot products is pure waste.

Invert the execution order instead: wait until upstream filtering finishes, collect the surviving scores and indices into a small fixed scratchpad during a single scan over the filtered row, fetch only those four or five rows from `lm_head` (5 rows of BF16 weights total **35 KB**), compute five dot products with $$g$$, and draw the Gumbel sample inside the same Triton kernel.

---

Writing that Triton kernel hits a concrete low-level obstacle right away.

Whenever the kernel scans survivors, finds a row maximum, or invokes `tl.sort` to cap rare high-entropy rows to the top-$$K$$ candidates, deterministic tie-breaking requires a composite ordering: **sort by `float32` `logit` descending first; when two tokens tie in score, prefer the smaller `uint32` `token_id` ascending.**

GPU register primitives like Triton's `tl.maximum` and `tl.sort` only compare a **single scalar tensor**—you cannot hand them a two-field C++ `std::sort` comparator lambda, and splitting comparisons into multi-pass masks bloats register pressure and branching.

How do you make hardware execute a composite `(float32_desc, uint32_asc)` comparison using a single plain integer `a > b` instruction?

The inspiration comes from two classic systems domains: **floating-point radix sort** in graphics and HPC (summarized by Michael Herf's 2001 `FloatFlip` note and used inside NVIDIA CUB's `RadixSort`), and **memcomparable composite index keys** in storage engines like RocksDB and MySQL. Both construct an **order-preserving bijection** from IEEE 754 floats onto unsigned integers.

Draw the bit layout of an IEEE 754 `float32`, and the derivation falls out visually:

```text
IEEE 754 float32 bit layout (32 bits total):
  31 30          23 22                                           0
 ┌──┬──────────────┬──────────────────────────────────────────────┐
 │S │ Exponent(8b) │                 Mantissa (23b)               │
 └──┴──────────────┴──────────────────────────────────────────────┘
  ▲
  └── Sign bit S: 0 for positive, 1 for negative (sign-magnitude, NOT two's complement!)

Viewed directly as an unsigned integer uint32, the real line is folded backward on negatives:
  Numeric order:   -inf  ──────►  -0.0   │   +0.0  ──────►  +inf
  Raw bits (hex): 0xFF800000 ──► 0x80000000 │ 0x00000000 ──► 0x7F800000
                  (S=1: more negative = larger bits, inverted!) (S=0: already monotonic!)

A symmetric bit transform unfolds the entire real line monotonically onto uint32:
  • Negatives (S=1) -> bitwise NOT all 32 bits (~bits): MSB flips to 0 (lower half) & order reverses!
  • Positives (S=0) -> XOR only the sign bit (^ 0x80000000): MSB flips to 1 (upper half), order stays!

  Transformed:    0x007FFFFF ──► 0x7FFFFFFF │ 0x80000000 ──► 0xFF800000
                     (-inf)        (-0)         (+0)           (+inf)
```

Why does this work so cleanly? IEEE 754 designed its biased exponent specifically so non-negative floats could be compared with integer hardware; the only obstacles are the sign bit and the reversed magnitude of negatives. Canonicalizing `-0.0` to `+0.0` first, flipping the sign bit on non-negatives to lift them into the upper half, and complementing all 32 bits on negatives to drop and reverse them into the lower half maps every finite float strictly in order onto `uint32`.

Take one more step: shift that 32-bit ordered float into the upper half of a `uint64`, pack `0xFFFFFFFF - token_id` into the lower 32 bits (turning smaller token IDs into larger unsigned values), and reserve `0` for masked `-inf` entries:

```text
Composite uint64 sort key (a single integer key_a > key_b resolves score + tie-break):
  63                              32 31                               0
 ┌──────────────────────────────────┬──────────────────────────────────┐
 │    Order-preserving float32      │       0xFFFFFFFF - token_id      │
 └──────────────────────────────────┴──────────────────────────────────┘
       High 32 bits: logit descending     Low 32 bits: token_id ascending
                   (All-zero key 0 reserved for -inf, sorting last)
```

Translated into Triton, encoding and decoding are symmetric and fit in ten lines:

```python
@triton.jit
def _keys(logits: tl.tensor, tokens: tl.tensor, finite: tl.tensor) -> tl.tensor:
    logits = tl.where(logits == 0, 0.0, logits)
    bits = logits.to(tl.uint32, bitcast=True)
    ordered = tl.where(bits & 0x80000000 != 0, ~bits, bits ^ 0x80000000)
    return tl.where(finite, (ordered.to(tl.uint64) << 32) | (0xFFFFFFFF - tokens.to(tl.uint32)), 0)

@triton.jit
def _decode(keys: tl.tensor) -> tuple[tl.tensor, tl.tensor]:
    ordered = (keys >> 32).to(tl.uint32)
    bits = tl.where(ordered & 0x80000000 != 0, ordered ^ 0x80000000, ~ordered)
    return bits.to(tl.float32, bitcast=True), (0xFFFFFFFF - keys.to(tl.uint32)).to(tl.int32)
```

---

Once a custom GPU kernel compiles and runs fast, how do you verify that it neither drifts numerically nor hides latency spikes over hundreds of thousands of production steps? Two general diagnostic habits caught every single edge case we hit afterward.

**Habit 1: Replace eyeball fluency checks with seed-locked token-by-token parity.**

Eyeballing generated prose lets 99% of numerical discrepancies slip right past you. Lock the random seed instead, run the fused kernel side by side against the naive full-vocabulary reference across hundreds of thousands of generated tokens, and halt at the very first token divergence.

On our first parity run, several thousand tokens matched verbatim before an entire sentence suddenly branched off. Inspecting the candidate distribution on the diverging row revealed why: our fused kernel kept dot products in `float32` registers (as did the companion logit-bias kernel), whereas upstream's unquantized `lm_head` rounded its projection output through `bfloat16` on the way to memory. Almost everywhere that tiny mantissa difference is harmless—except when a borderline token sits right on the `top-p = 0.95` cumulative probability cutoff. Skipping the intermediate `bfloat16` rounding kept one extra token alive, shifting the downstream Gumbel draw and altering the rest of the sequence.

Once isolated, the fix took half a line in registers: apply an explicit `.to(bf16).to(fp32)` step to mirror upstream's rounding staircase and inline upstream's exact `gumbel_block_argmax` helper. Across nearly `300,000` generated tokens, mismatches dropped to zero.

**Habit 2: In batched GPU kernels, profile the batch `max` (`p99.9`), never just the mean.**

Earlier we saw that a typical row has `4.83` survivors. Thinking in averages, sizing the per-row scratchpad capacity `K` to `256` looks fifty times larger than necessary. Yet under realistic load, profiler traces showed periodic latency spikes every few dozen steps—and raising `K` to `4096` actually ran _faster_ than `256`.

Pulling out the specific rows behind those spikes exposed two phenomena that batch averages completely hide:

- **Batch straggler coupling**: 128 requests execute concurrently inside one kernel launch, so step latency equals `max(row_0, ..., row_127)`, not the mean. Even if 127 rows finish 5 dot products in microseconds, a single high-entropy row exceeding `256` survivors falls into the multi-pass `tl.sort` overflow path while the other 127 thread blocks wait. Spending `4 MiB` of scratch memory to set `K = 4096` above the empirical ceiling (`2,785`) buys out the straggler branch;
- **The `0.12%` unmasked confident rows**: A subset of slow-path rows were not uncertain at all—they were maximally confident! When the top token's probability exceeded `0.9999999`, it cleared `top_p = 0.95` by itself, so upstream's filter took an early-exit shortcut and returned the raw logits untouched without writing a single `-inf`. Tracking the softmax denominator `mass` during the scan and short-circuiting directly to the argmax dot product whenever no `-inf` appears and `1.0 / mass >= top_p` eliminated those spikes immediately.

_(Tracking steady-state memory watermarks the same way caught one bonus PyTorch registration trap: wrapping a variable-shape Triton launch with `torch.library.wrap_triton` and passing `tuple[Tensor, ...]` makes the dispatcher memoize the tensor tuple as a compile-time specialization key, pinning temporary tensors forever on every step; registering a `@torch.library.custom_op` taking `list[Tensor]` treats them as normal runtime operands.)_

---

This bandwidth-first tail pipeline powers `@pllm.guide_sampling` and `@pllm.add_logits` in **[pLLM](https://github.com/LinesHogan/pLLM)** (`pllm`), our open-source programmable extension layer on top of vLLM, keeping 7B serving with 128-way online per-request training and guided sampling above `96%` of baseline throughput.

Knowing the exact constraint model also tells you when this sparse gather stops applying: indexing individual vocabulary rows requires an unsharded, non-block-quantized `lm_head` matrix on the local GPU. Under tensor parallelism (`TP > 1`) across the vocabulary axis or block-packed quantized weights, local thread blocks cannot evaluate arbitrary vocabulary dot products independently; the engine then falls back to concatenating $$h$$ and $$g$$ along the batch axis (`lm_head(torch.cat((h, g), dim=0))`)—trading away sparsity, while still guaranteeing that the 1.1 GB weight matrix streams out of HBM only once.

</div>

<script defer src="/assets/js/bilingual-post.js"></script>
