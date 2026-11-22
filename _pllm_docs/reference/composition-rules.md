---
layout: page
title: "Composition Rules"
description: "pLLM Documentation · API, Config & Support Reference"
permalink: /pllm/docs/reference/composition-rules/
nav: false
---

<link rel="stylesheet" href="{{ '/assets/css/pllm-hub.css' | relative_url | bust_file_cache }}">

<div class="zine-dispatch" data-title-zh="组合规则" data-title-en="Composition Rules">
  <div class="zine-dispatch-links">
    <a class="zine-pill-link primary" data-i18n-zh="← 返回 pLLM 文档中心" data-i18n-en="← pLLM Docs Hub" href="/pllm/#docs">← pLLM Docs Hub</a>
    <a class="zine-pill-link" data-i18n-zh="文档导航地图" data-i18n-en="Docs Roadmap" href="/pllm/docs/">Docs Roadmap</a>
    <span class="zine-lang-badge" data-i18n-zh="API 与配置参考手册" data-i18n-en="API, Config &amp; Support Reference">API, Config &amp; Support Reference</span>
  </div>
  <a class="zine-pill-link" data-i18n-zh="GitHub 源码 ↗" data-i18n-en="View on GitHub ↗" href="https://github.com/LinesHogan/pLLM/blob/main/doc/reference/composition-rules.md" target="_blank" rel="noopener">View on GitHub ↗</a>
</div>

<div id="markdown-content">

<div class="zine-lang-panel" data-lang-panel="en" lang="en" markdown="1">

<div id="en-composition-rules"></div>

Each effect in a workload declares its position within a step and the states it reads or writes. pLLM orders effects by position, rejects only genuinely conflicting combinations, and reports an error with remediation instructions at build time (`pllm.explain` or `pllm.make_llm`). This page lists which combinations can coexist, which are rejected, and why. For how to compose workloads, see [Composing Workloads](/pllm/docs/guides/composition/).

## Allowed Combinations {#en-allowed-combinations}

| Combination | Conditions & Notes |
| --- | --- |
| Multiple `@pllm.train` methods, each training its own state | Must use the same training mode (all on request rows, or all on host batches) |
| `@pllm.export` with any effect | Exports are read-only |
| Multiple `@pllm.add_logits` methods (including from different components) | Summed before filtering, then sampling guidance is applied; addition is order-independent |
| `@pllm.add_logits` / `@pllm.guide_sampling` with `Adapter`, `ActivationPatch`, `@pllm.train_in_forward` | Sampling effects run after the forward pass; model edits apply inside the forward pass |
| One `@pllm.custom_forward` with a training overlay and an `ActivationPatch` | Every layer executed by custom forward passes through them; `Adapter` deltas apply on each projection call |
| `@pllm.custom_residual` with `@pllm.custom_forward` | Every layer executed by custom forward also passes through the custom residual (except stateful sublayers; see table below) |
| `pllm.residual` / `pllm.layer_logits` with `@pllm.custom_forward` | Reads the last computed residual stream for each layer |
| Producer and consumer of the same state | For example, `@pllm.write` or `@pllm.train` writing and `Adapter` reading; each state may have at most one writer method per step |

## Rejected Combinations {#en-rejected-combinations}

| Combination | Why | Fix |
| --- | --- | --- |
| Stacking two decorators of the same kind on one method (e.g., `@pllm.export` and `@pllm.train`) | The later decorator overwrites the earlier one, silently dropping one effect; different effects also have different default phases | Use one method per effect and share computation via `pllm.result(name)` |
| Two `@pllm.guide_sampling` methods (including across components) | The sampler reweights against only one prediction per step; how two predictions should be combined is an algorithmic choice | Combine into one prediction in a single method; use `@pllm.add_logits` for additional logit adjustments |
| Request-row training and host-batch training in the same workload (including components) | The two training modes differ in update timing and memory reservation; a workload supports only one training mode | Split across two engines; state read only by host code can be trained separately with `pllm.host.make_host` |
| Standalone request-row training, or training alongside `@pllm.guide_sampling`, with `rows=` / `update_rows=` | This path performs one update and publishes a new version per state on every step, and does not support skipping rows | Select rows on `@pllm.guide_sampling(rows=...)`, or use `reduction="sum"` and multiply the loss of excluded rows by 0 (see [Online Training](/pllm/docs/guides/training/)) |
| `@pllm.custom_forward` with a method reading `pllm.hidden` / `pllm.final_hidden` | In a custom forward, a layer may execute multiple times or not at all, so "that layer's output" is not unique | Read `pllm.residual(layer)` or `pllm.layer_logits(...)` |
| `active=` used outside `tokens="decode"` | Skipped layers on prompt tokens leave only approximate K/V entries that every subsequent token will read | Use `@pllm.custom_forward(tokens="decode")` |
| Stateful `@pllm.custom_residual` (`state=`, `chunk=`) with `@pllm.custom_forward` | Memory advances once per token, whereas custom forward visits each layer a variable number of times | Choose one or the other |
| Two `@pllm.custom_residual` methods on the same sublayer; two `@pllm.custom_attention` methods on the same layer | A sublayer's residual update and a layer's attention read can each have only one owner | Merge into a single method |
| Static KV layouts (`keep="window"` / `"none"`) with `pllm.PositionState` or `@pllm.custom_forward` | Static layouts cause vLLM to partition KV into multiple groups, while both features require a single group | Choose one or the other |
| `@pllm.custom_attention` inside `@pllm.custom_forward` with `kv="peek"` | Read-only layer invocations without KV writes do not yet support custom attention reads | Use `kv="commit"` |
| Two `pllm.Fed` states; reading a fed state directly from a method | Data from an external process enters a state read by `ActivationPatch` only through a single `@pllm.write` | Use one fed state per workload and copy/transform it via a single `@pllm.write` |
| Two `init=pllm.Imported()` states | A single `generate(batch=StateImport(...))` call imports only one state | Merge the fields into a single state |
| A component reading another component's state (state names containing `.`) | A component must behave identically to when it runs standalone | Place methods that need to share state in the same workload |
| One class inheriting from two workloads | Declarations with the same name from the two base classes overwrite each other | Declare them as dataclass fields (components) |
| Two methods reading each other's results within the same step | Results within a step can only flow in one direction | Remove one read, or move it to the next step by reading from state |
| A component requiring a different `publish_lag` than the engine | The entire engine shares a single `publish_lag` | Use the same value across all components |
| Passing `Adapter`, `ActivationPatch`, `@pllm.custom_forward`, or `@pllm.custom_residual` to `pllm.host.make_host` | They operate inside the model forward pass, and host training has no model | Use `pllm.make_llm` |

## Combinations with Engine Settings {#en-combinations-with-engine-settings}

| Pattern | Requires | Reason |
| --- | --- | --- |
| `@pllm.custom_attention`, model-writing `@pllm.train_in_forward`, `ActivationPatch` with `last_tokens > 0`, or model edits reading `RequestState` / mutable `InstanceState` | `enable_prefix_caching=False` | Cached KV depends on per-request state, dynamic parameters, or trailing generated tokens; read-only workloads, stateless `@pllm.custom_residual`, and static edits reading only immutable `InstanceState` can safely enable prefix caching |
| `@pllm.custom_attention`, `@pllm.custom_residual` (without `pllm.value` / `pllm.input_tokens`), `ActivationPatch`, dense/low-rank `Adapter` on `FromModule`/`Pretrained`/`Computed`/`Imported`/`Zeros` | Supports `tensor_parallel_size > 1` (vLLM >= 0.32.0) | Executes per local head, residual slice, or projection row/column shard |
| `@pllm.custom_attention` | FlashAttention 2 (or DeepSeek V2/V3 latent reads supporting MLA), unquantized KV; MLA requires `keep='all'` and no `window` buffer; `pllm.attention(...)` fact reads require `tensor_parallel_size=1` | Custom reads directly access paged KV / MLA latent pages |
| `@pllm.custom_residual` fed `pllm.input_tokens` / `pllm.value`, `per_head` adapters, side-stream optimizers/exports | `tensor_parallel_size=1` | Requires a full token/head view or is not closed within local shards |
| `cuda_graphs=True` | No patterns requiring sliced graphs or eager execution (such as `@pllm.custom_forward`) | See [Support Matrix](/pllm/docs/reference/support-matrix/#en-cuda-graphs-and-eager) |

For additional per-model and parallelism restrictions, see the [Support Matrix](/pllm/docs/reference/support-matrix/).

</div>

<div class="zine-lang-panel is-hidden" data-lang-panel="zh" lang="zh-CN" markdown="1">

<div id="zh-sec-1"></div>

一个负载里的效果各自声明自己在一步中的位置和读写的状态，pLLM 按位置排序，只拒绝真正冲突的组合，并在构建时
（`pllm.explain` 或 `pllm.make_llm`）报错、说明改法。本页列出哪些组合可以并存、哪些被拒绝以及原因。组合的写法
见[组合负载](/pllm/docs/guides/composition/)。

## 可以并存 {#zh-sec-2}

| 组合 | 条件与说明 |
| --- | --- |
| 多个 `@pllm.train`，各训各的状态 | 同一种训练方式（都在请求行上，或都在宿主批次上） |
| `@pllm.export` 与任何效果 | 导出只读 |
| 多个 `@pllm.add_logits`（包括来自不同组件的） | 先在过滤前相加，再应用采样引导；加法与顺序无关 |
| `@pllm.add_logits` / `@pllm.guide_sampling` 与 `Adapter`、`ActivationPatch`、`@pllm.train_in_forward` | 采样效果在 forward 之后，模型改写在 forward 之中 |
| 一个 `@pllm.custom_forward` 与一个训练 overlay、一个 `ActivationPatch` | custom forward 运行的每一层都经过它们；`Adapter` 的增量在每次投影调用时施加 |
| `@pllm.custom_residual` 与 `@pllm.custom_forward` | custom forward 运行的每一层也经过定制残差（带状态的子层除外，见下表） |
| `pllm.residual` / `pllm.layer_logits` 与 `@pllm.custom_forward` | 读每层最后一次算出的残差流 |
| 同一状态的生产者与消费者 | 例如 `@pllm.write` 或 `@pllm.train` 写、`Adapter` 读；一个状态每步只能有一个写入方法 |

## 被拒绝的组合 {#zh-sec-3}

| 组合 | 为什么 | 怎么改 |
| --- | --- | --- |
| 一个方法上叠两个同类装饰器（如 `@pllm.export` 和 `@pllm.train`） | 后一个会覆盖前一个，其中一个效果悄悄失效；不同效果的默认时机也不同 | 每个效果一个方法，用 `pllm.result(name)` 共享计算 |
| 两个 `@pllm.guide_sampling`（包括来自不同组件的） | 采样器每步只按一个预测重新加权；两个预测怎样合成是算法的选择 | 在一个方法里合成一个预测；其余修正用 `@pllm.add_logits` |
| 请求行训练与宿主批训练在同一个负载（包括组件） | 两种训练的更新时机和显存预留不同，一个负载只有一种训练方式 | 分到两个引擎；只给宿主代码读的状态可以用 `pllm.host.make_host` 单独训练 |
| 单独的请求行训练、或与 `@pllm.guide_sampling` 并存的训练，加 `rows=` / `update_rows=` | 这条路径每步给每个状态做一次更新并发布新版本，不支持按行跳过 | 在 `@pllm.guide_sampling(rows=...)` 上选行，或用 `reduction="sum"` 并把要排除的行的 loss 乘 0（见[在线训练](/pllm/docs/guides/training/)） |
| `@pllm.custom_forward` 与读 `pllm.hidden` / `pllm.final_hidden` 的方法 | custom forward 里一层可能跑多遍或不跑，"这一层的输出"不唯一 | 读 `pllm.residual(layer)` 或 `pllm.layer_logits(...)` |
| `active=` 用在 `tokens="decode"` 之外 | 跳层的 prompt token 只留下近似的 K/V，之后每个 token 都会读到 | `@pllm.custom_forward(tokens="decode")` |
| 带状态的 `@pllm.custom_residual`（`state=`、`chunk=`）与 `@pllm.custom_forward` | 记忆每个 token 前进一次，而 custom forward 里一层的访问次数不定 | 二选一 |
| 同一子层上的两个 `@pllm.custom_residual`；同一层上的两个 `@pllm.custom_attention` | 一个子层的残差更新、一层的 attention 读法只能有一个所有者 | 合并成一个方法 |
| 静态 KV 布局（`keep="window"` / `"none"`）与 `pllm.PositionState` 或 `@pllm.custom_forward` | 布局让 vLLM 把 KV 分成多组，这两者要求只有一组 | 二选一 |
| `@pllm.custom_attention` 在 `@pllm.custom_forward` 里配 `kv="peek"` | 只读不写的层调用还不支持定制读法 | 用 `kv="commit"` |
| 两个用 `pllm.Fed` 的状态；fed 状态被方法直接读 | 外部进程送来的数据只经一个 `@pllm.write` 进入 `ActivationPatch` 读的状态 | 每个负载一个 fed 状态，用一个 `@pllm.write` 转写 |
| 两个 `init=pllm.Imported()` 的状态 | 一次 `generate(batch=StateImport(...))` 只导入一个状态 | 把字段合并进一个状态 |
| 组件读另一个组件的状态（状态名里带 `.`） | 组件必须和单独运行时行为一致 | 需要共享状态的方法写进同一个负载 |
| 一个类继承两个负载 | 两个父类的同名声明会互相覆盖 | 把它们声明成字段（组件） |
| 一步之内两个方法互读对方的结果 | 一步内的结果只能单向流动 | 去掉其中一个读取，或把它移到下一步读状态 |
| 组件要求的 `publish_lag` 与引擎不同 | 整个引擎只有一个 `publish_lag` | 所有组件用同一个值 |
| `Adapter`、`ActivationPatch`、`@pllm.custom_forward`、`@pllm.custom_residual` 交给 `pllm.host.make_host` | 它们作用在模型的 forward 里，宿主训练没有模型 | 用 `pllm.make_llm` |

## 与引擎设置的组合 {#zh-sec-4}

| 写法 | 需要 | 原因 |
| --- | --- | --- |
| `@pllm.custom_attention`、写入模型的 `@pllm.train_in_forward`、`last_tokens > 0` 的 `ActivationPatch`，或读取 `RequestState` / 可变 `InstanceState` 的模型改写 | `enable_prefix_caching=False` | 缓存的 KV 依赖于每请求状态、动态参数或生成后段 token；只读负载、无状态 `@pllm.custom_residual` 与只读不可变 `InstanceState` 的静态改写可安全开启前缀缓存 |
| `@pllm.custom_attention`、`@pllm.custom_residual`（无 `pllm.value` / `pllm.input_tokens`）、`ActivationPatch`、`FromModule`/`Pretrained`/`Computed`/`Imported`/`Zeros` 上的 dense/low-rank `Adapter` | 支持 `tensor_parallel_size > 1`（vLLM >= 0.32.0） | 按局部头、残差切片或投影行/列分片执行 |
| `@pllm.custom_attention` | FlashAttention 2（或支持 MLA 的 DeepSeek V2/V3 潜变量读取）、未量化 KV；MLA 需 `keep='all'` 且不能使用 `window` 缓冲；`pllm.attention(...)` 事实读取需要 `tensor_parallel_size=1` | 定制读法直接访问分页 KV / MLA 潜变量页 |
| `@pllm.custom_residual` 喂 `pllm.input_tokens` / `pllm.value`、`per_head` 适配器、侧链优化器/导出 | `tensor_parallel_size=1` | 需要全量 token/头视角或未在本地分片内闭环 |
| `cuda_graphs=True` | 没有需要切片图或 eager 的写法（如 `@pllm.custom_forward`） | 见[支持矩阵](/pllm/docs/reference/support-matrix/#zh-sec-4) |

更多按模型和并行方式的限制见[支持矩阵](/pllm/docs/reference/support-matrix/)。

</div>

</div>

<div class="zine-doc-footer">
  <a class="zine-pill-link" data-i18n-zh="← 返回 pLLM 项目与文档中心" data-i18n-en="← Back to pLLM Project &amp; Docs" href="/pllm/#docs">← Back to pLLM Project &amp; Docs</a>
  <a class="zine-pill-link" href="https://github.com/LinesHogan/pLLM/blob/main/doc/reference/composition-rules.md" target="_blank" rel="noopener">GitHub &middot; doc/reference/composition-rules.md ↗</a>
</div>

<script defer src="/assets/js/bilingual-post.js"></script>
