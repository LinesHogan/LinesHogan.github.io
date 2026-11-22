---
layout: page
title: "explain Report and Runtime Diagnostics"
description: "pLLM Documentation · API, Config & Support Reference"
permalink: /pllm/docs/reference/diagnostics/
nav: false
---

<link rel="stylesheet" href="{{ '/assets/css/pllm-hub.css' | relative_url | bust_file_cache }}">

<div class="zine-dispatch" data-title-zh="explain 报告与运行时诊断字段" data-title-en="explain Report and Runtime Diagnostics">
  <div class="zine-dispatch-links">
    <a class="zine-pill-link primary" data-i18n-zh="← 返回 pLLM 文档中心" data-i18n-en="← pLLM Docs Hub" href="/pllm/#docs">← pLLM Docs Hub</a>
    <a class="zine-pill-link" data-i18n-zh="文档导航地图" data-i18n-en="Docs Roadmap" href="/pllm/docs/">Docs Roadmap</a>
    <span class="zine-lang-badge" data-i18n-zh="API 与配置参考手册" data-i18n-en="API, Config &amp; Support Reference">API, Config &amp; Support Reference</span>
  </div>
  <a class="zine-pill-link" data-i18n-zh="GitHub 源码 ↗" data-i18n-en="View on GitHub ↗" href="https://github.com/LinesHogan/pLLM/blob/main/doc/reference/diagnostics.md" target="_blank" rel="noopener">View on GitHub ↗</a>
</div>

<div id="markdown-content">

<div class="zine-lang-panel" data-lang-panel="en" lang="en" markdown="1">

<div id="en-explain-report-and-runtime-diagnostics"></div>

This page documents every line and field of the `pllm.explain(workload, config)` report and the `llm.diagnostics()` return value. To learn how to use them for troubleshooting, see [Inspecting Costs: pllm.explain](/pllm/docs/guides/explain/) and [Making Workloads Faster](/pllm/docs/howto/faster/).

```python
import torch

import pllm


class StreamNorm(pllm.Workload):
    @pllm.export(name="stream_norm", max_rows=8, max_cols=1)
    def norm(self, r: torch.Tensor = pllm.residual(12)) -> torch.Tensor:
        return r.float().norm(dim=-1, keepdim=True)


report = pllm.explain(StreamNorm(), pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=8))
print(report.eager_reason == "", report.workload_host_syncs, [export.rows for export in report.exports])
```

## Text Report from explain {#en-text-report-from-explain}

Each line in `print(report)` starts with a leading token indicating what that line describes.

| Line Prefix | Content |
| --- | --- |
| `model` | Model class, hidden width, layer count, vocabulary size, and dtype from `config.json`. Read-only models include an extra `model read-only: ...` line explaining why |
| `parallel` | Tensor parallelism degree; all subsequent reservations are per GPU |
| `forward` | Whether the model forward runs inside a CUDA graph: `CUDA graphs: no effect needs eager`, or `eager: <reason>` |
| `base` | Estimated per-token dense projection and LM-head FLOPs for the base model, excluding attention over KV history |
| Each line after the `member` header | One computation segment of one of your methods: `method_name  effect  anchor / lane / graph  state_read (origin, distance)`. A single method may span multiple segments |
| Anchor | Where the segment runs within a step: `request.prepare`, `model.execute`, `state.publish`, `sampler.pre_draw`, `sampler.complete`, `token.commit`, `request.finish`, or `next.prepare` (see [Core Concepts](/pllm/docs/explain/core-concepts/)) |
| lane | `main`: main stream; `side`: side stream overlapped with the forward pass |
| graph | `captured`: captured into a CUDA graph; `eager`: executed op-by-op |
| `(origin, distance)` | State version read: `latest` for the newest version, `boundary` for the previous publish boundary, `entry` for the version at step start; distance is the number of steps behind latest |
| `model site or inactive method` | A declaration acting directly inside the model (`Adapter`, `ActivationPatch`) or an unscheduled helper method, with no anchor |
| `forward eager because <method>` | This declaration forces the model forward back to eager mode |
| `CUDA-graph pool reservation` | Private CUDA-graph memory pool reservation for this segment |
| `estimate / step (N rows)` | Estimated per-step matmul/projection FLOPs, state read/write bytes (including optimizer fields), and sampler bytes at `max_num_seqs` rows |
| `kernels` | Estimated kernels launched per step (before fusion); `graph` indicates graph replay |
| `tap` | Hidden positions tapped from the model, which tokens (`decode` / `prefill` / `all`), buffer bytes, and the methods sharing that tap |
| Each line after the `state` header | `state_name  scope  bytes/root  reserved bytes`, followed by each field's storage dtype and reserved pages (`x N pages`), initializer source (`source`), `placement`, host memory bytes, and PCIe bytes per row and per step. Page counts exceed concurrent requests because old and new versions coexist; AdamW step counts and moments are read only by the optimizer without retaining old versions, so they use fewer pages than parameter fields |
| `memory` | Peak VRAM required by pLLM itself (state pages, task workspaces, CUDA graph pools, etc., excluding weights and KV cache); the second line gives two thresholds: below the first, side-stream tasks fall back to main-stream inline execution; below the second, planning fails |
| `budget` | Exact error text when your configured budget cannot fit (identical to `make_llm`); the report still prints the unrestricted plan |
| `preempt` | Whether state is kept (`keep`) or reset (`reset`) on request preemption, and why recomputation is rejected when reset |
| `host` | Number of GPU-to-CPU synchronizations in your math; methods inside the model are marked `unverified` and must be confirmed on GPU with `check_on_gpu` |
| `export` | Rows, columns, bytes, and event count per step for each export; notes when rows exceeding `max_rows` are dropped |
| `timing` | Values and consequences of timing settings, tagged `(default)` or `(you set)` |
| `op` | Custom ops: call sites, per-row FLOPs (when `register_flop_formula` is present), per-row byte lower bound, and whether a CPU reference and autograd exist |
| `attention layer` | Per-layer read pattern and estimated per-token FLOPs and KV bytes for `@pllm.custom_attention` |
| `KV layer` | Maximum KV blocks and bytes per request per layer under static KV layouts |
| `note` | Things omitted from or unknown to this report (such as GPU BLAS workspaces, model weights, and KV cache) |

## ExplainResult Fields {#en-explainresult-fields}

`pllm.explain` returns an `ExplainResult`; `dataclasses.asdict(report)` can be passed directly to `json.dumps`. Commonly read fields:

| Field | Meaning |
| --- | --- |
| `eager_reason` | Why the model forward is not inside a CUDA graph; empty string if it is |
| `members` | Effect, per-segment anchor, lane, graph capture status, state reads, and estimates for each method |
| `states` | Scope, per-field page count (`pages`), bytes per root, `reserved_bytes`, field dtypes, initializer source, and placement for each state |
| `taps` | Tapped hidden locations and buffer bytes |
| `peak_memory_bytes`, `overlap_threshold_bytes`, `inline_peak_memory_bytes` | The three numbers printed on the `memory` lines |
| `budget_errors` | Errors when configured budgets are exceeded; empty tuple if everything fits |
| `preemption`, `recompute_rejection` | The `preempt` line |
| `workload_host_syncs` | Synchronization count; `None` if it cannot be verified on CPU |
| `exports` | Row count, column count, bytes, and event count for each export |
| `escape_hatches` | The `timing` lines: setting, value, whether default, and consequence |
| `custom_ops`, `attention`, `kv_capacity`, `kernels` | Corresponding `op`, `attention layer`, `KV layer`, and `kernels` lines |
| `limitations` | The `note` lines |
| `flops_per_token_estimate` | Estimated per-token FLOPs for base model plus all methods; `None` if base model is unknown |

## llm.diagnostics() Fields {#en-llmdiagnostics-fields}

Callable at any time while the engine is running; returns values at the moment of the query after pLLM waits for device counters to flush.

| Field | Meaning |
| --- | --- |
| `eager_reason` | Same as in `explain`; empty string when the model runs inside a CUDA graph |
| `loss_count` | Number of training method invocations |
| `update_count` | Number of optimizer updates actually published |
| `trained_rows` | Cumulative rows trained |
| `active_states` | Number of currently live state roots |
| `state_usage` | Snapshot of state VRAM usage; see table below |
| `numeric_health` | L2 norm, maximum absolute value, and non-finite count (`norm`, `max_abs`, `nonfinite_count`) for each live published field; `None` if norm or max is non-finite. Computed only on query without adding per-step syncs; no history is retained after a field retires |
| `kv_cache_blocks`, `kv_cache_max_concurrency` | Actual KV blocks allocated by vLLM and maximum concurrent requests sized by `max_model_len` (per GPU) |
| `hold_count`, `last_hold` | Count and most recent details of `ActivationPatch` with `on_missing="hold"` reusing stale inputs |
| `inference_pins`, `completed_sampler_pins`, `pending_state_releases` | Internal counters for framework debugging |

`state_usage` fields:

| Field | Meaning |
| --- | --- |
| `reserved_bytes` | VRAM reserved for state |
| `logical_live_bytes` | Sum of field bytes across all live versions; fields shared between old and new versions are counted multiple times |
| `allocated_payload_bytes` | Actual allocated field bytes; shared fields are counted once, so this can be much smaller than `logical_live_bytes` |
| `reserved_padding_bytes` | Bytes reserved for alignment padding |
| `free_field_pages`, `live_field_pages`, `private_field_pages`, `retiring_field_pages` | Field page counts: free, published, currently being written, and awaiting reclamation |
| `private_versions`, `retiring_versions`, `pinned_versions` | Version counts: currently being written, awaiting reclamation, and pinned by readers (plus a pending-device-completion publish count for framework debugging) |
| `admission_capacity_failures` | Number of requests rejected at admission because state capacity was exhausted |

## Export Events {#en-export-events}

Commonly read fields on `pllm.io.TraceEvent` delivered to `on_export`: `tensor_name` (export name; component prefixes are stripped for component exports), `tensor()` (detached CPU tensor), `identities` (one `pllm.testing.RequestIdentity` per row: `prompt_index`, `sample_index`, and pLLM internal `request_id`), `positions` (each row's token position in its request), `step_id`, and `dropped_rows` (rows dropped on this step due to `max_rows`). `llm.trace_snapshot()` returns a `pllm.io.TraceSnapshot` with retained `events`, `dropped` (oldest events evicted when the snapshot buffer filled), and `export_dropped` (events dropped when the export delivery queue filled).

</div>

<div class="zine-lang-panel is-hidden" data-lang-panel="zh" lang="zh-CN" markdown="1">

<div id="zh-sec-1"></div>

本页逐行、逐字段解释 `pllm.explain(workload, config)` 的报告和 `llm.diagnostics()` 的返回值。怎么用它们找问题，见
[看成本：pllm.explain](/pllm/docs/guides/explain/) 和 [让负载更快](/pllm/docs/howto/faster/)。

```python
import torch

import pllm


class StreamNorm(pllm.Workload):
    @pllm.export(name="stream_norm", max_rows=8, max_cols=1)
    def norm(self, r: torch.Tensor = pllm.residual(12)) -> torch.Tensor:
        return r.float().norm(dim=-1, keepdim=True)


report = pllm.explain(StreamNorm(), pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=8))
print(report.eager_reason == "", report.workload_host_syncs, [export.rows for export in report.exports])
```

## explain 的文本报告 {#zh-sec-2}

`print(report)` 每行以一个词开头，这个词说明这一行讲什么。

| 行首 | 内容 |
| --- | --- |
| `model` | 模型类、hidden 宽度、层数、词表大小、精度，来自 `config.json`。只读模型另有一行 `model read-only: ...` 写明原因 |
| `parallel` | 张量并行卡数；后面所有预留都是每张卡的数 |
| `forward` | 模型 forward 是否在 CUDA graph 里：`CUDA graphs: no effect needs eager`，或 `eager: <原因>` |
| `base` | 底座模型每个 token 的稠密投影与 LM head FLOP 估计，不含对 KV 历史的 attention |
| `member` 表头之后的每一行 | 你的一个方法的一段计算：`方法名  效果  锚点 / lane / graph  读的状态 (origin, distance)`。一个方法可能有几段 |
| 锚点 | 这段计算在一步里的位置：`request.prepare`、`model.execute`、`state.publish`、`sampler.pre_draw`、`sampler.complete`、`token.commit`、`request.finish`、`next.prepare`（含义见[核心概念](/pllm/docs/explain/core-concepts/)） |
| lane | `main`：主 stream；`side`：旁路 stream，与 forward 重叠 |
| graph | `captured`：录进 CUDA graph；`eager`：逐算子执行 |
| `(origin, distance)` | 读的状态版本：`latest` 最新版本，`boundary` 上一个发布边界的版本，`entry` 步开始时的版本；distance 是离最新差几步 |
| `model site or inactive method` | 直接作用在模型里的声明（`Adapter`、`ActivationPatch`）或没有被安排的辅助方法，没有锚点 |
| `forward eager because <方法>` | 这个声明让模型 forward 退回 eager |
| `CUDA-graph pool reservation` | 这段计算的 CUDA graph 私有内存池预留 |
| `estimate / step (N rows)` | 按 `max_num_seqs` 行估计的每步 matmul 与投影 FLOP、状态读写字节（含优化器字段）、采样器字节 |
| `kernels` | 每步发射的算子数估计（融合前），`graph` 表示在 graph 里重放 |
| `tap` | 从模型里取出的 hidden 位置、哪些 token（decode / prefill / all）、缓冲字节和共用它的方法 |
| `state` 表头之后的每一行 | `状态名  作用域  bytes/root  reserved bytes`，再列出每个字段的存储精度和预留页数（`x N pages`）、初值来源（`source`）、放置位置（`placement`）、主机内存字节、每行和每步跨 PCIe 的字节。页数比并发请求多，因为新旧版本要同时存在；AdamW 的步数和矩只有优化器读，旧版本不保留，页数比参数字段少 |
| `memory` | pLLM 自己要的显存峰值（状态页、任务工作区、CUDA graph 池等，不含权重和 KV）；第二行给出两个阈值：低于第一个，旁路任务改成主 stream 内联；低于第二个，规划失败 |
| `budget` | 你设的预算装不下时的报错原文（与 `make_llm` 相同）；报告仍给出不受预算限制的计划 |
| `preempt` | 请求被抢占后状态保留（`keep`）还是重置（`reset`），以及重置时为什么不能重算 |
| `host` | 你的数学里 GPU→CPU 同步的次数；模型内部的方法标 `unverified`，要上 GPU 用 `check_on_gpu` 确认 |
| `export` | 每个导出每步多少行、多少列、多少字节、拆成几个事件；行数超过 `max_rows` 时注明多出的行会被丢弃 |
| `timing` | 时机类设置的取值和后果，`(default)` 是默认值、`(you set)` 是你设的 |
| `op` | 自写算子：调用位置、每行 FLOP（有 `register_flop_formula` 时）、每行字节下界、有没有 CPU 参考实现和 autograd |
| `attention layer` | `@pllm.custom_attention` 每层的读法和每 token 的 FLOP、KV 字节估计 |
| `KV layer` | 静态 KV 布局下每层每个请求最多占多少 KV 块和字节 |
| `note` | 这份报告不知道或没计入的事（例如 GPU 上的 BLAS 工作区、模型权重与 KV） |

## ExplainResult 的字段 {#zh-sec-3}

`pllm.explain` 返回 `ExplainResult`，`dataclasses.asdict(report)` 可以直接 `json.dumps`。常用字段：

| 字段 | 含义 |
| --- | --- |
| `eager_reason` | 模型 forward 为什么不在 CUDA graph 里；空字符串表示在 |
| `members` | 每个方法的效果、各段计算的锚点、lane、是否捕获、读的状态、估计 |
| `states` | 每个状态的作用域、每个字段的页数（`pages`）、每份字节数、`reserved_bytes`、字段精度、初值来源、放置位置 |
| `taps` | 取出的 hidden 位置和缓冲字节 |
| `peak_memory_bytes`、`overlap_threshold_bytes`、`inline_peak_memory_bytes` | `memory` 行的三个数 |
| `budget_errors` | 预算装不下时的报错，空元组表示装得下 |
| `preemption`、`recompute_rejection` | `preempt` 行 |
| `workload_host_syncs` | 同步次数；`None` 表示无法在 CPU 上确认 |
| `exports` | 每个导出的行数、列数、字节、事件数 |
| `escape_hatches` | `timing` 行：设置、取值、是否默认、后果 |
| `custom_ops`、`attention`、`kv_capacity`、`kernels` | 对应的 `op`、`attention layer`、`KV layer`、`kernels` 行 |
| `limitations` | `note` 行 |
| `flops_per_token_estimate` | 底座加所有方法每 token 的 FLOP 估计；底座未知时为 `None` |

## llm.diagnostics() 的字段 {#zh-sec-4}

引擎运行中随时可以调用；返回的是查询那一刻的值，pLLM 会先等设备上的计数写完。

| 字段 | 含义 |
| --- | --- |
| `eager_reason` | 同 explain，空字符串表示模型在 CUDA graph 里 |
| `loss_count` | 训练方法运行的次数 |
| `update_count` | 实际发布的优化器更新次数 |
| `trained_rows` | 参与训练的行数累计 |
| `active_states` | 当前存活的状态副本数 |
| `state_usage` | 状态显存的快照，见下表 |
| `numeric_health` | 每个存活的已发布字段的 L2 范数、最大绝对值和非有限值个数（`norm`、`max_abs`、`nonfinite_count`）；范数或最大值不是有限数时为 `None`。只在查询时计算，不增加每步的同步；字段退役后不保留历史 |
| `kv_cache_blocks`、`kv_cache_max_concurrency` | vLLM 实际分配的 KV 块数，以及按 `max_model_len` 算能同时容纳的请求数（每张卡） |
| `hold_count`、`last_hold` | 带 `on_missing="hold"` 的激活补丁沿用旧输入的次数和最近一次的情况 |
| `inference_pins`、`completed_sampler_pins`、`pending_state_releases` | 框架调试用的内部计数 |

`state_usage` 的字段：

| 字段 | 含义 |
| --- | --- |
| `reserved_bytes` | 为状态预留的显存 |
| `logical_live_bytes` | 所有存活版本的字段字节之和；新旧版本共用的字段会被重复计入 |
| `allocated_payload_bytes` | 实际占用的字段字节；共用的字段只计一次，所以可能远小于 `logical_live_bytes` |
| `reserved_padding_bytes` | 为对齐填充的字节 |
| `free_field_pages`、`live_field_pages`、`private_field_pages`、`retiring_field_pages` | 字段页的使用情况：空闲、已发布、正在写、等待回收 |
| `private_versions`、`retiring_versions`、`pinned_versions` | 正在写、等待回收、被读者占用的版本数（另有一个等待设备完成的发布计数，供框架调试） |
| `admission_capacity_failures` | 因为状态容量用完而被拒绝准入的请求数 |

## 导出事件 {#zh-sec-5}

`on_export` 收到的 `pllm.io.TraceEvent` 中用户常读的字段：`tensor_name`（导出名；组件的导出去掉了字段名前缀）、
`tensor()`（独立的 CPU 张量）、`identities`（每行一个 `pllm.testing.RequestIdentity`：`prompt_index`、`sample_index`、
pLLM 内部的 `request_id`）、`positions`（每行在请求中的 token 位置）、`step_id`、`dropped_rows`（这一步因 `max_rows`
被丢弃的行数）。`llm.trace_snapshot()` 返回 `pllm.io.TraceSnapshot`：保留的事件 `events`、缓冲满时丢掉的最旧事件数 `dropped`，
以及导出投递队列满时丢掉的事件数 `export_dropped`。

</div>

</div>

<div class="zine-doc-footer">
  <a class="zine-pill-link" data-i18n-zh="← 返回 pLLM 项目与文档中心" data-i18n-en="← Back to pLLM Project &amp; Docs" href="/pllm/#docs">← Back to pLLM Project &amp; Docs</a>
  <a class="zine-pill-link" href="https://github.com/LinesHogan/pLLM/blob/main/doc/reference/diagnostics.md" target="_blank" rel="noopener">GitHub &middot; doc/reference/diagnostics.md ↗</a>
</div>

<script defer src="/assets/js/bilingual-post.js"></script>
