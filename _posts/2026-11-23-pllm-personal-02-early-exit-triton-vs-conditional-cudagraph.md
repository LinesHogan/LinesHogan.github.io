---
layout: post
title: "不重录 CUDA Graph 实现动态切层：条件子图、掩码求值与早退 Triton 内核的比较"
title_en: "Dynamic Layer Switching Without Recapturing CUDA Graphs: Conditional Subgraphs, Masked Evaluation, and Early-Exit Triton Kernels"
date: 2026-11-23 15:48:00 +0800
description: "在多个候选层预埋动态钩子时，对比 CUDA 条件子图的工作区显存占用、朴素 torch.where 的无效访存带宽，以及基于片上 L2 标量判定的早退 Triton 内核与归一化端口原地累加方案；同时记录 rank > 256 回退路径中一行 bool(mask.any()) 引发图捕获报错的原因（工程与排障笔记 · 支持 EN / 中文 一键切换）"
description_en: "Once an online serving engine has captured CUDA Graphs across batch buckets, dynamically switching the target layer of an activation intervention, tap, or LoRA adapter by re-recording the... (Engineering & Debugging Notes · Bilingual EN / 中文)"
tags: pllm pytorch triton debugging
categories: engineering-notes
blog_track: engineering-notes
blog_series_zh: "pLLM"
blog_series_en: "pLLM"
pllm_synced: true
pllm_track: engineering-notes
pllm_order: 102
pllm_code: "N02"
pllm_badge_zh: "静态图与早退内核"
pllm_badge_en: "CUDA Graphs & Triton"
related_posts: true
---

<div class="zine-dispatch" data-title-zh="不重录 CUDA Graph 实现动态切层：条件子图、掩码求值与早退 Triton 内核的比较" data-title-en="Dynamic Layer Switching Without Recapturing CUDA Graphs: Conditional Subgraphs, Masked Evaluation, and Early-Exit Triton Kernels" data-desc-zh="在多个候选层预埋动态钩子时，对比 CUDA 条件子图的工作区显存占用、朴素 torch.where 的无效访存带宽，以及基于片上 L2 标量判定的早退 Triton 内核与归一化端口原地累加方案；同时记录 rank &gt; 256 回退路径中一行 bool(mask.any()) 引发图捕获报错的原因（工程与排障笔记 · 支持 EN / 中文 一键切换）" data-desc-en="Once an online serving engine has captured CUDA Graphs across batch buckets, dynamically switching the target layer of an activation intervention, tap, or LoRA adapter by re-recording the... (Engineering &amp; Debugging Notes · Bilingual EN / 中文)">
  <a class="zine-pill-link" data-i18n-zh="pLLM 项目主页与文档 →" data-i18n-en="pLLM Project &amp; Docs →" href="/pllm/">pLLM 项目主页与文档 →</a>
  <a class="zine-pill-link" data-i18n-zh="全部博客分类索引 →" data-i18n-en="Blog Categories →" href="/blog/">Blog Categories →</a>
</div>

<div class="zine-lang-panel" data-lang-panel="zh" lang="zh-CN" markdown="1">

<div class="zine-tldr" markdown="1">

**TL;DR：** 在线推理服务完成 CUDA Graph 捕获后，如果希望动态切换激活干预、探针或 LoRA 的目标层，重新捕获整套分档位静态图会带来数秒停顿；而在所有候选层做全量计算再 `torch.where` 掩码，又会让未命中层额外消耗显存带宽。对比 CUDA 条件子图的私有工作区显存开销后，我们采用了基于片上 L2 标量判定的早退 Triton 内核，并复用层间归一化输出端口原地累加，使未命中候选层跳过权重池与激活池的 HBM 读写。文末记录了 `rank > 256` 回退分支里 `bool(mask.any())` 导致图捕获失败的排查过程。

</div>

> **工程与排障笔记（#2 · 静态图与早退内核）** · [pLLM 项目主页与文档](/pllm/) · [全站博客分类目录](/blog/) · [GitHub 原文 ↗](https://github.com/LinesHogan/pLLM/blob/main/blog/personal/02-early-exit-triton-vs-conditional-cudagraph.md)

在做模型内部表征分析或在线激活引导（Steering）时，经常会遇到这样的场景：模型已经加载并完成了各个批次档位的 CUDA Graph 预热捕获，此时希望把读取残差或注入干预向量的位置从第 8 层调整到第 20 层，或者在同一个连续批处理步中让请求 A 走第 4 层 LoRA、请求 B 走第 18 层 LoRA、请求 C 传 `-1` 保持原模型输出。

如果把层号作为静态图结构的一部分，每次换层都需要卸载旧钩子、重新走一遍编译并重录 `[1, 2, 4, 8, ..., 256]` 等全部批次档位的 CUDA Graph，一次切换通常耗时数秒到十余秒，不适合交互式实验或在线服务。

要避免重录图，自然需要在构建期把候选层集合 `over=(4, 12, 20, ...)` 的钩子预先放进图里，运行时通过状态池里的一个 `0-d int64` 标量控制当前哪一层生效。随之而来的工程问题是：当候选层较多时（例如在 28 层模型上做逐层扫描，`over=tuple(range(28))`），每一步只有 1 层真正生效，其余 27 个未命中层的运行开销该如何控制。

---

围绕“在已捕获的静态图里跳过未命中层计算”，我们比较了三种实现方案。

**第一种方案是使用 CUDA 12.4+ 的设备端条件图节点（Conditional CUDA Graph / `device_branch`）。**  
思路是在每个候选层放置一个硬件级 `IF (active_count > 0)` 条件节点，仅当本步存在命中该层的请求时才执行分支子图内的矩阵乘。

实际验证下来，它的主要限制在于显存占用：每个条件子图分支在捕获其内部算子（如 cuBLAS 批量矩阵乘）时，会在各自的分支流内存池上独立申请私有工作区（Workspace）。

![CUDA 条件子图在多批档位与多候选层下的工作区膨胀](/assets/img/pllm/blog/assets/personal-02-fig1-zh.svg){: style="display: block; max-width: 100%; height: auto; margin: 1.5rem auto;" }

此外，在异构连续批处理中，只要批次里有 1 个请求命中第 $$l$$ 层，整层的 `IF` 条件就为真，子图内部仍然需要做行级过滤。用层级条件分支既多占了留给 KV Cache 的显存，也省不掉异构批次的行掩码处理。

**第二种方案是全算后掩码（`torch.where(mask, h + delta, h)`）。**  
做法最直接：每个候选层照常按槽位收集参数并执行矩阵乘算出 `delta`，最后用 `torch.where` 将未命中行的增量置零。这种方式不引入额外工作区，也兼容 CUDA Graph，但未命中的 27 层仍然会把参数和隐藏状态从高带宽显存（HBM）完整读入计算一遍。排查这部分开销时，我们还注意到原有的静态适配器路径里有一句无条件的 `captured[:len(hidden)].copy_(hidden)`——原本只用于在线求导保留前向输入，纯推理模式下省去这步拷贝也能减少每层的写带宽。

**第三种方案是基于片上 L2 标量检查的线程块早退（CTA Early-Exit）Triton 内核，配合已有输出端口原地累加。**  
每步前向开始前，引擎在设备端生成一张几百字节的 `int64` 归属表 `owners[rows]`（未命中行记为 `-1`），这张小表在前向期间常驻片上 L2 缓存。自定义 Triton 内核按 `(row, col_block)` 划分网格，每个线程块启动后的第一条指令先从 L2 读取 `owner = tl.load(owners_ptr + row)`，若小于 `0` 则直接 `return`：

![全算后掩码与片上 L2 标量早退 Triton 内核对比](/assets/img/pllm/blog/assets/personal-02-fig2-zh.svg){: style="display: block; max-width: 100%; height: auto; margin: 1.5rem auto;" }

这里还有一个容易忽略的细节：如果把激活补丁算子设计成返回新张量的函数 `out = masked_patch_add(hidden, table, pool)`，即使未命中行跳过了参数池读取，为了保证返回的 `out` 中包含原有的 `hidden`，仍需把整行 `hidden` 从 HBM 读出再写入 `out`，相当于每个未命中层多做一次全宽拷贝。

结合层间 `RMSNorm` 边界的上下文看，外层在调用边界钩子时已经备好了 `(residual, normed)` 两个输出端口，并已将分支输出拷入 `normed`。因此把补丁算子改为对 `normed` 原地累加的 `masked_patch_add_into(normed, table, pool)`，未命中行就可以不触碰 `normed` 所在的 HBM 地址：

```python
@triton.jit
def _masked_patch_add_into_kernel(
    target_ptr, table_ptr, pool_ptr,
    table_stride, target_stride, pool_stride, width, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    # 先读 L2 中的掩码标志；未命中时直接返回，不读写 target_ptr 与 pool_ptr
    if tl.load(table_ptr + row) == 0:
        return
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    col_mask = cols < width
    slot = tl.load(table_ptr + table_stride + row)
    dst_ptrs = target_ptr + row.to(tl.int64) * target_stride + cols
    h = tl.load(dst_ptrs, mask=col_mask, other=0.0)
    delta = tl.load(pool_ptr + slot.to(tl.int64) * pool_stride + cols, mask=col_mask, other=0.0)
    tl.store(dst_ptrs, h + delta.to(target_ptr.dtype.element_ty), mask=col_mask)
```

低秩 LoRA 适配器也采用了同样的思路：预留一块 `[max_rows, rank]` 的 `float32` 中间缓冲 `routed_mid`，分两阶段执行早退内核并在第二阶段直接累加到 `output[:, offset:offset+width]`。当外层运行在 `@pllm.custom_forward` 的子批次重排（`call_rows != None`）模式下时，只需对 `owners` 做一次 `index_select(0, call_rows)` 即可复用同一套内核。

---

在审查上述内核的外围分派逻辑时，我们发现了一处藏在低频参数分支里的图捕获问题。

由于两阶段低秩内核设定了 `rank <= 256` 的快路径上限，当 `rank > 256`、输入为 `float64` 或末维步长非连续时，代码会回退到常规 PyTorch 算子组合。为了在回退路径下少做无用计算，初版代码直接沿用了 CPU 分支的短路判断：

```python
if target.dtype == torch.float64 or rank > 256 or target.stride(-1) != 1:
    mask = table[0, :rows].bool()
    if bool(mask.any()):  # rank <= 256 时不会走到这里；一旦 rank = 260，图捕获会报错
        ...
```

平时常规测试使用的都是 `rank = 4` 或 `8`，GPU 执行一直走上面的 Triton 快路径，这段回退分支只在 CPU 测试里跑过；而在 CPU 上调 `bool(mask.any())` 是完全正常的。一旦用户在 GPU 上声明了 `rank = 260` 并开启 CUDA Graph，预热捕获走到 `bool(mask.any())` 时，PyTorch 需要同步流把布尔规约结果拷回主机转成 Python `bool`，CUDA 驱动就会报 `cudaErrorStreamCaptureUnsupported`。

修复方式是将 CPU 短路与 CUDA 兜底明确分开：CUDA 路径即使退回慢速实现，也只使用设备端的 `torch.where`，不把张量转换成 Python 标量；同时在 GPU 测试集里补一条 `rank = 260` 的图捕获用例覆盖该分支：

```python
if not target.is_cuda or target.dtype == torch.float64 or rank > 256 or target.stride(-1) != 1:
    mask = table[0, :rows].bool()
    if not target.is_cuda and not bool(mask.any()):
        # 仅在 CPU 路径使用 Python bool 短路
        return
    # CUDA 回退路径保持纯设备端掩码更新，避免流同步打断 CUDA Graph 捕获
    ...
    target.copy_(torch.where(mask[:, None], target + delta, target))
    return
```

---

在单张 RTX 4090 上以 `Qwen2.5-7B-Instruct`（`bfloat16`，开启 CUDA Graph）测试这组实现的实际开销：

- 在 4 个候选层（`over=(4, 12, 20, 26)`）挂载动态激活引导与异步残差导出，并在轮次间调用 `import_state` 实时切层：128 并发（`batch=8, n=16`）下解码 TPOT 为 `24.6 ms`，静态单层版本为 `24.4 ms`，无钩子的原版 vLLM 为 `23.2 ms`，动态门控相比静态单层增加约 `0.2 ms`；
- 将低秩 LoRA 挂在全部 28 层（`over=tuple(range(28))`，共 56 个 `q_proj` / `v_proj` 投影钩子，每步激活 1 层）时，整套分档位 CUDA Graph 共占用 `0.30 GiB` 显存，128 并发下 TPOT 为 `24.8 ms`（静态单层 LoRA 为 `24.0 ms`，27 个未命中层合计增加约 `0.8 ms`）。

相关代码已合入 **[pLLM](https://github.com/LinesHogan/pLLM)** 的 `pllm.model.LayerRoute`。在使用上仍需注意它的边界成本：早退内核省去了未命中层的 HBM 读写与矩阵乘，但每个候选层仍保留一次轻量内核发射（单层十余微秒量级），且层间补丁所在的归一化边界需传递残差与分支端口，`per_head`、`rank > 256` 及末维非连续补丁输入则会回退至掩码求值。因此 API 上将 `LayerRoute(..., over=(...))` 设为独立入口，要求显式列出实际需要切换的候选层，而不是默认挂满全模型。

</div>

<div class="zine-lang-panel is-hidden" data-lang-panel="en" lang="en" markdown="1">

<div class="zine-tldr" markdown="1">

**TL;DR:** Once an online serving engine has captured CUDA Graphs across batch buckets, dynamically switching the target layer of an activation intervention, tap, or LoRA adapter by re-recording the graphs stalls traffic for seconds, while evaluating every candidate layer and masking with `torch.where` wastes HBM bandwidth on inactive layers. After comparing the private cuBLAS workspace footprint of CUDA conditional subgraphs, we settled on an early-exit Triton kernel gated by an L2-resident scalar and paired with in-place accumulation on the existing inter-layer normalization output port, letting missed candidate layers skip both weight-pool and activation-pool HBM traffic. We close by tracing a CUDA Graph capture failure hidden inside the `rank > 256` fallback branch, triggered by a host-synchronizing `bool(mask.any())`.

</div>

> **Engineering & Debugging Notes (#2 · CUDA Graphs & Triton)** · [pLLM Project & Docs](/pllm/) · [All Blog Categories](/blog/) · [GitHub Source ↗](https://github.com/LinesHogan/pLLM/blob/main/blog/personal/en/02-early-exit-triton-vs-conditional-cudagraph.md)

When analyzing internal model representations or running online activation steering, a familiar requirement turns up quickly: the model is already loaded and CUDA Graphs are already warmed up across every batch-size bucket, and now you want to move a residual tap or steering vector from layer 8 to layer 20—or, inside a single continuous-batching step, route Request A through a layer-4 LoRA, Request B through a layer-18 LoRA, and Request C through `-1` to leave the base model untouched.

If the target layer ID is baked into the static graph structure, every layer switch forces you to detach the old hook, recompile, and recapture every CUDA Graph bucket (`[1, 2, 4, 8, ..., 256]`). A single switch takes anywhere from a few seconds to over ten seconds—far too slow for interactive experiments or live serving.

To avoid recapturing graphs, the hooks for every candidate layer in `over=(4, 12, 20, ...)` must be wired into the graph at build time, with a `0-d int64` scalar in the state pool controlling which layer takes effect at runtime. That raises an immediate engineering question: when the candidate set is large—say, sweeping every layer of a 28-layer model with `over=tuple(range(28))`—only 1 layer fires on a given step. How do you keep the other 27 missed layers from draining latency and memory?

---

We compared three approaches for skipping inactive-layer work inside an already captured static graph.

**Approach 1: Device-side conditional graph nodes (Conditional CUDA Graph / `device_branch`, CUDA 12.4+).**  
The idea is to place a hardware-level `IF (active_count > 0)` node at each candidate layer, executing the branch subgraph's matrix multiplications only when at least one request in the step hits that layer.

In practice, memory consumption is the main constraint: when each conditional branch captures its internal operators (such as batched cuBLAS GEMMs), it allocates a private workspace out of its own branch-stream memory pool.

![Conditional CUDA subgraphs multiplying cuBLAS workspace across batch buckets and candidate layers](/assets/img/pllm/blog/assets/personal-02-fig1-en.svg){: style="display: block; max-width: 100%; height: auto; margin: 1.5rem auto;" }

On top of that, under heterogeneous continuous batching, a single request hitting layer $$l$$ flips the entire layer's `IF` condition to true, so the subgraph still has to perform row-level filtering inside anyway. Layer-level conditional branching consumes VRAM meant for the KV cache while failing to remove row masking for mixed batches.

**Approach 2: Full evaluation followed by masking (`torch.where(mask, h + delta, h)`).**  
This is the most straightforward option: every candidate layer gathers parameters by slot and runs the matrix multiplication to compute `delta`, then `torch.where` zeroes out updates for inactive rows. It adds no private workspace and captures cleanly into CUDA Graphs, yet the 27 missed layers still load parameters and hidden states out of HBM and compute through them. While auditing that overhead, we also caught an unconditional `captured[:len(hidden)].copy_(hidden)` in the existing static adapter path—originally there to retain forward inputs for online autograd—and skipping that copy in inference-only mode trimmed per-layer write bandwidth as well.

**Approach 3: CTA Early-Exit Triton kernels gated by an L2-resident scalar, paired with in-place accumulation on existing output ports.**  
Before each forward step, the engine builds a few-hundred-byte `int64` ownership table `owners[rows]` on device (`-1` for inactive rows), small enough to stay resident in on-chip L2 cache throughout the pass. Our custom Triton kernel partitions the grid by `(row, col_block)`, and each thread block's very first instruction loads `owner = tl.load(owners_ptr + row)` from L2 and returns immediately if it is below `0`:

![Full HBM evaluation + torch.where vs. L2 scalar early-exit Triton kernel](/assets/img/pllm/blog/assets/personal-02-fig2-en.svg){: style="display: block; max-width: 100%; height: auto; margin: 1.5rem auto;" }

There is an easy-to-miss detail here: if the activation-patch op is written as an out-of-place function returning a new tensor (`out = masked_patch_add(hidden, table, pool)`), inactive rows skip reading the parameter pool, yet they still have to read the entire `hidden` row from HBM and write it into `out` just to preserve the original activations—effectively adding a full-width memory copy on every missed layer.

Looking at the surrounding inter-layer `RMSNorm` boundary solves this cleanly: the outer wrapper already allocates both output ports `(residual, normed)` before calling the boundary hook and has already copied the branch output into `normed`. Turning the patch op into an in-place accumulation onto `normed` (`masked_patch_add_into(normed, table, pool)`) lets missed rows return without touching `normed` in HBM at all:

```python
@triton.jit
def _masked_patch_add_into_kernel(
    target_ptr, table_ptr, pool_ptr,
    table_stride, target_stride, pool_stride, width, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    # Check the L2-resident mask first; return immediately on a miss without touching target_ptr or pool_ptr
    if tl.load(table_ptr + row) == 0:
        return
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    col_mask = cols < width
    slot = tl.load(table_ptr + table_stride + row)
    dst_ptrs = target_ptr + row.to(tl.int64) * target_stride + cols
    h = tl.load(dst_ptrs, mask=col_mask, other=0.0)
    delta = tl.load(pool_ptr + slot.to(tl.int64) * pool_stride + cols, mask=col_mask, other=0.0)
    tl.store(dst_ptrs, h + delta.to(target_ptr.dtype.element_ty), mask=col_mask)
```

Low-rank LoRA adapters follow the same structure: pre-allocate a `[max_rows, rank]` `float32` scratch buffer `routed_mid`, run a two-stage early-exit kernel, and accumulate straight into `output[:, offset:offset+width]` in stage two. When an outer `@pllm.custom_forward` reorders a sub-batch (`call_rows != None`), a single `index_select(0, call_rows)` on `owners` lets the exact same kernels handle it.

---

Reviewing the dispatch wrapper around these kernels uncovered a graph-capture bug tucked inside a low-frequency parameter branch.

Because the two-stage low-rank kernel caps its fast path at `rank <= 256`, calls with `rank > 256`, `float64` inputs, or a non-contiguous trailing stride fall back to standard PyTorch ops. To avoid extra work on that fallback path, the initial version reused the CPU branch's short-circuit guard:

```python
if target.dtype == torch.float64 or rank > 256 or target.stride(-1) != 1:
    mask = table[0, :rows].bool()
    if bool(mask.any()):  # Never reached for rank <= 256; breaks graph capture as soon as rank = 260
        ...
```

Routine tests all ran at `rank = 4` or `8`, so GPU execution always took the Triton fast path and this fallback only ever ran on CPU—where calling `bool(mask.any())` is completely fine. The moment a user declared `rank = 260` on GPU with CUDA Graphs enabled, warmup capture hit `bool(mask.any())`, forcing PyTorch to synchronize the stream and copy the boolean reduction back to the host to construct a Python `bool`. The CUDA driver failed immediately with `cudaErrorStreamCaptureUnsupported`.

The fix separates the CPU short-circuit from the CUDA fallback explicitly: even when taking the slow path, CUDA stays purely on device via `torch.where` without converting any tensor into a Python scalar, backed by a GPU graph-capture test at `rank = 260`:

```python
if not target.is_cuda or target.dtype == torch.float64 or rank > 256 or target.stride(-1) != 1:
    mask = table[0, :rows].bool()
    if not target.is_cuda and not bool(mask.any()):
        # Use Python bool short-circuit on CPU only
        return
    # Keep the CUDA fallback strictly on-device via masking so stream sync never breaks CUDA Graph capture
    ...
    target.copy_(torch.where(mask[:, None], target + delta, target))
    return
```

---

Measured on a single RTX 4090 running `Qwen2.5-7B-Instruct` (`bfloat16`, CUDA Graphs enabled), the actual overhead breaks down as follows:

- With dynamic activation steering and asynchronous residual export attached to 4 candidate layers (`over=(4, 12, 20, 26)`) and live layer switching via `import_state` between turns: decode TPOT at 128 concurrent requests (`batch=8, n=16`) is `24.6 ms`, compared to `24.4 ms` for a static single-layer hook and `23.2 ms` for stock vLLM with no hooks—dynamic gating adds roughly `0.2 ms` over the static single-layer baseline.
- With low-rank LoRA attached across all 28 layers (`over=tuple(range(28))`, 56 `q_proj` / `v_proj` projection hooks total, 1 layer active per step), the entire set of bucketed CUDA Graphs uses `0.30 GiB` of VRAM, and decode TPOT at 128 concurrency is `24.8 ms` (versus `24.0 ms` for a static single-layer LoRA—roughly `0.8 ms` across all 27 missed layers combined).

The code is merged into `pllm.model.LayerRoute` in **[pLLM](https://github.com/LinesHogan/pLLM)**. Its boundary costs remain worth keeping in mind: early-exit kernels eliminate HBM traffic and GEMMs on missed layers, yet each candidate layer still retains one lightweight kernel launch (on the order of ten-plus microseconds per layer), inter-layer patch boundaries still plumb residual and branch ports, and `per_head`, `rank > 256`, or non-contiguous trailing-stride patch inputs fall back to masked evaluation. That is why the API exposes `LayerRoute(..., over=(...))` as an explicit opt-in requiring callers to declare only the candidate layers they actually intend to switch across, rather than blanketing the full model by default.

</div>

<script defer src="/assets/js/bilingual-post.js"></script>
