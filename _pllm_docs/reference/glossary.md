---
layout: page
title: "Glossary"
description: "pLLM Documentation · API, Config & Support Reference"
permalink: /pllm/docs/reference/glossary/
nav: false
---

<link rel="stylesheet" href="{{ '/assets/css/pllm-hub.css' | relative_url | bust_file_cache }}">

<div class="zine-dispatch" data-title-zh="术语表" data-title-en="Glossary" data-desc-zh="pLLM 文档 · API 与配置参考手册" data-desc-en="pLLM Documentation · API, Config &amp; Support Reference">
  <div class="zine-dispatch-links">
    <a class="zine-pill-link primary" data-i18n-zh="← 返回 pLLM 文档中心" data-i18n-en="← pLLM Docs Hub" href="/pllm/#docs">← pLLM Docs Hub</a>
    <a class="zine-pill-link" data-i18n-zh="文档导航地图" data-i18n-en="Docs Roadmap" href="/pllm/docs/">Docs Roadmap</a>
    <span class="zine-lang-badge" data-i18n-zh="API 与配置参考手册" data-i18n-en="API, Config &amp; Support Reference">API, Config &amp; Support Reference</span>
  </div>
  <a class="zine-pill-link" data-i18n-zh="GitHub 源码 ↗" data-i18n-en="View on GitHub ↗" href="https://github.com/LinesHogan/pLLM/blob/main/doc/reference/glossary.md" target="_blank" rel="noopener">View on GitHub ↗</a>
</div>

<div id="markdown-content">

<div class="zine-lang-panel" data-lang-panel="en" lang="en" markdown="1">

<div id="en-glossary"></div>

Terms that appear throughout the documentation and `pllm.explain` output, one meaning per term. Each entry links to the page where it is explained in detail.

## Authoring Workloads {#en-authoring-workloads}

- **workload**: A subclass of `pllm.Workload` containing only declarations and tensor math. Read once when building the engine and scheduled by pLLM thereafter; its methods are not runtime callbacks. See [Workload Anatomy](/pllm/docs/guides/workload-anatomy/).
- **declaration**: The three constructs in a workload that answer "what to read, what to store, and where results go": parameter defaults, state attributes, and effect decorators or model-edit attributes.
- **input marker**: A value placed in a parameter's default specifying where the input comes from, such as `pllm.hidden(0)` or `pllm.parameter("s", "w")`. Type annotations only document what the method body receives. See [Reading Inside the Model](/pllm/docs/guides/reading/).
- **effect**: A decorator specifying where a method's output goes: `@pllm.export`, `@pllm.add_logits`, `@pllm.guide_sampling`, `@pllm.write`, `@pllm.train`, and so on. See [Where Results Go](/pllm/docs/guides/results/).
- **model site**: A declaration acting directly inside the model forward pass: `pllm.model.ActivationPatch`, `pllm.model.Adapter`, `@pllm.train_in_forward`, `@pllm.custom_residual`, and `@pllm.custom_attention`. CPU checks do not execute model sites.
- **setting method**: A method passed to a decorator or input marker instead of a fixed literal, such as `beta=strength` with `def strength(self) -> float`. pLLM calls it once per workload instance.
- **row**: A single entry along dimension 0 of a tensor received by a method: one active request (or one token to be sampled, or one token computed in the step). Row order is determined by the scheduler. See [Core Concepts](/pllm/docs/explain/core-concepts/).
- **selector (`rows=`)**: A method returning a `bool [rows]` mask, passed to an effect's `rows=` argument so the effect applies only to selected rows.
- **component**: A workload held as a dataclass field inside another workload; its names carry the field name prefix, and its states are independent. See [Composing Workloads](/pllm/docs/guides/composition/).
- **custom op**: A kernel wrapped with `torch.library.custom_op` and called inside a workload method. See [Custom Ops](/pllm/docs/guides/custom-ops/).

## State {#en-state}

- **state**: The only mutable construct in a workload: a collection of named tensor fields whose VRAM is allocated and version-updated by pLLM. See [State](/pllm/docs/guides/state/).
- **RequestState / InstanceState**: `pllm.RequestState` maintains one copy per request (or group of requests); `pllm.InstanceState` maintains one copy for the entire engine.
- **`key_fields`**: Determines which requests share a `RequestState` copy: per request, samples of the same prompt, or requests sharing a route (`routes=`). See [Sharing Training Across Requests](/pllm/docs/howto/share-training-across-requests/).
- **field**: A single tensor inside a state. `shape=` declares a single field named `"value"`; `pllm.Field` specifies a projection-scaled shape or an explicit dtype.
- **initializer (`init=`)**: Where a state's initial values come from: `pllm.Zeros`, `pllm.Computed`, `pllm.Pretrained`, `pllm.ModelCheckpoint`, `pllm.FromModule`, `pllm.Fed`, or `pllm.Imported`. See [Loading Trained Weights](/pllm/docs/howto/load-trained-weights/).
- **frozen state**: A state without an optimizer: readable anywhere, and mutable only through import or `@pllm.write`.
- **version**: Each state update publishes a new version; readers observe published versions, while in-flight requests retain their starting version. `pllm.version(state)` gives the number of times each row's state has been updated.
- **`publish_lag`**: An update computed on a step becomes visible to readers starting at the `(1 + publish_lag)`-th step on which the workload actually executes afterward; steps where the workload reads zero rows (such as mid-prompt chunks or recomputed history) are skipped by the engine and do not count. Defaults to 1 to overlap training with the next forward pass.
- **carry field**: A field declared with `carry=` that is published alongside training updates without being updated by the optimizer; written via `pllm.carry`.
- **PositionState (`pllm.PositionState`)**: One hidden-width vector per KV block, read and written via `layers.load` / `layers.save` exclusively inside `@pllm.custom_forward`.
- **placement (`placement=`)**: Whether a read-only large table resides in GPU VRAM (`"device"`) or pinned host memory (`"host"`).

## Training {#en-training}

- **request-row training**: A `@pllm.train` method reads rows of active requests and returns a per-row loss; pLLM reduces per state, runs backward, and updates. See [Online Training](/pllm/docs/guides/training/).
- **reduction (`reduction=`)**: When multiple rows share one state in a step, whether losses are averaged (`"mean"`) or summed (`"sum"`).
- **host batch**: A training batch assembled by host code and passed in via `generate(batch=...)`: `WeightedRowBatch`, `CausalTrainBatch`, or `WeightedCEBatch`. See [Host Loops and Batch Training](/pllm/docs/guides/host-loops/).
- **causal model (`pllm.causal_model`)**: The full model callable inside a training method with specified `Adapter` overlays attached, allowing gradients to flow into adapter state.
- **`HostTrainer`**: The object returned by `pllm.host.make_host`, which trains a workload's state on host batches without loading a model.
- **in-forward training (`@pllm.train_in_forward`)**: Trains a `RequestState` inside the forward pass using a projection's inputs and outputs and adds its delta to that projection (as in TTT / In-Place TTT). See [Model Edits](/pllm/docs/guides/model-edits/).

## Model Edits & Forward Structure {#en-model-edits--forward-structure}

- **ActivationPatch**: Reads values from state and modifies the outputs of selected layers (steering vectors, ReFT).
- **Adapter**: Applies a state as a weight delta onto designated projections (e.g., LoRA), regardless of how the state is written.
- **custom residual (`@pllm.custom_residual`)**: Takes over the residual update at a specific site in selected layers, value rows, or the final stream. See [Customizing Forward Structure](/pllm/docs/guides/forward-structure/).
- **stateful sublayer**: A custom residual with `state=` and `chunk=`, maintaining a per-request carry updated in chunks (fast-weight memory).
- **custom attention (`@pllm.custom_attention`)**: Replaces how selected layers read KV; can also declare a static **KV layout** (sliding window, prompt + window, cross-layer sharing).
- **custom forward (`@pllm.custom_forward`)**: Uses Python to orchestrate which layers each token runs, how many times, and when to stop.

## Execution & Verification {#en-execution--verification}

- **planner**: The component of pLLM that reads workload declarations at engine build time, assigns each method's step position and stream, and computes memory reservations; `pllm.explain` prints its plan. See [Inspecting Costs: pllm.explain](/pllm/docs/guides/explain/).
- **admission**: The moment pLLM accepts something. Request admission is when a request enters the engine, receives a state copy, and binds the state version it reads; model or pattern admission checks at build time whether a model or feature is supported, raising an error if not.
- **Inductor**: PyTorch `torch.compile`'s code generator, which fuses adjacent small ops into a single kernel; vLLM uses it when compiling model CUDA graphs, and pLLM uses it to fuse read-only lookups.
- **phase (`pllm.Phase`)**: Time windows within a step: `BEFORE_FORWARD`, `IN_FORWARD`, `AFTER_FORWARD`, `BEFORE_SAMPLE`, `AFTER_SAMPLE`, plus `AUTOMATIC` (chosen by pLLM). Each effect has a default phase that `@pllm.on(phase)` can override.
- **anchor**: Named positions within a step printed by `pllm.explain`: `request.prepare`, `model.execute`, `state.publish`, `sampler.pre_draw`, `token.commit`, and so on.
- **main stream / side stream (`lane`)**: `main` runs serially with the forward pass; `side` overlaps with the next forward pass.
- **eager reason**: Why the model forward is not inside a CUDA graph (`eager_reason`); empty string if captured.
- **neutral configuration (`neutral=`)**: The same workload with zero intervention strength; `testing.check` verifies that it leaves sampling unchanged.
- **contract check**: Properties verified by `pllm.testing.check` (CPU) and `check_on_gpu` (GPU): row-order invariance, request isolation, chunking equivalence, preemption recomputation, version visibility, and neutrality. See [Checking Correctness](/pllm/docs/guides/checking/).
- **budget**: The three fields in `ExecutionOptions` that cap pLLM memory: `state_page_budget_bytes`, `execution_peak_budget_bytes`, and `cuda_graph_pool_budget_bytes`. See [Configuration Options](/pllm/docs/reference/config/).

</div>

<div class="zine-lang-panel is-hidden" data-lang-panel="zh" lang="zh-CN" markdown="1">

<div id="zh-sec-1"></div>

文档和 `pllm.explain` 输出里出现的词，一个词一个意思。每条后面的链接指向讲它的页面。

## 写负载 {#zh-sec-2}

- **负载（workload）**：`pllm.Workload` 的子类，只放声明和张量数学。构建引擎时被读一次，之后由 pLLM 安排执行；
  它的方法不是运行时回调。见[负载的组成](/pllm/docs/guides/workload-anatomy/)。
- **声明**：负载回答"读什么、存什么、结果去哪"的三种写法：参数默认值、状态属性、效果装饰器或模型改写属性。
- **输入标记**：写在参数默认值里、说明数据从哪来的值，如 `pllm.hidden(0)`、`pllm.parameter("s", "w")`。
  类型注解只说明方法体拿到什么。见[读取模型内部](/pllm/docs/guides/reading/)。
- **效果**：说明方法的结果去哪的装饰器：`@pllm.export`、`@pllm.add_logits`、`@pllm.guide_sampling`、`@pllm.write`、
  `@pllm.train` 等。见[结果去哪](/pllm/docs/guides/results/)。
- **模型站点**：直接作用在模型 forward 里的声明：`pllm.model.ActivationPatch`、`pllm.model.Adapter`、
  `@pllm.train_in_forward`、`@pllm.custom_residual`、`@pllm.custom_attention`。CPU 上的检查不执行它们。
- **设置方法**：交给装饰器或输入标记、代替固定值的方法，如 `beta=strength` 配 `def strength(self) -> float`。
  pLLM 对每个负载实例调用一次。
- **行（row）**：方法收到的张量第 0 维的一行：一个运行中的请求（或采样器要抽的一个 token、一步算出的一个 token）。
  行的顺序由调度器决定。见[核心概念](/pllm/docs/explain/core-concepts/)。
- **选择器（`rows=`）**：返回 `bool [rows]` 掩码的方法，交给效果的 `rows=`，只让选中的行生效。
- **组件**：作为 dataclass 字段放进另一个负载的负载；名字带字段名前缀，状态互相独立。见[组合负载](/pllm/docs/guides/composition/)。
- **自写算子**：用 `torch.library.custom_op` 包装、在方法里调用的 kernel。见[自写算子](/pllm/docs/guides/custom-ops/)。

## 状态 {#zh-sec-3}

- **状态**：负载里唯一可变的东西：一组有名字的张量字段，由 pLLM 分配显存、按版本更新。见[状态](/pllm/docs/guides/state/)。
- **请求状态 / 实例状态**：`pllm.RequestState` 每个请求（或每组请求）一份；`pllm.InstanceState` 整个引擎一份。
- **`key_fields`**：决定哪些请求共用一份请求状态：每个请求、同一 prompt 的样本，或同一路由值（`routes=`）。
  见[让一组请求共享训练成果](/pllm/docs/howto/share-training-across-requests/)。
- **字段**：状态里的一个张量。`shape=` 声明一个叫 `"value"` 的字段；`pllm.Field` 给出按投影宽度算的形状或明确的精度。
- **初值来源（`init=`）**：状态的起始值从哪来：`pllm.Zeros`、`pllm.Computed`、`pllm.Pretrained`、
  `pllm.ModelCheckpoint`、`pllm.FromModule`、`pllm.Fed`、`pllm.Imported`。见[加载训练好的权重](/pllm/docs/howto/load-trained-weights/)。
- **冻结状态**：没有优化器的状态：到处可读，只能由导入或写入改变。
- **版本**：状态每次更新都发布一个新版本；读的一方读已发布的版本，在途的请求保留开始时的版本。
  `pllm.version(state)` 是每行的状态已更新的次数。
- **`publish_lag`**：某一步算出的更新，从它之后负载实际运行的第 1 + `publish_lag` 步起被读到；负载读不到任何行的步（只有提示中段或重算的历史）被引擎跳过，不计数。默认 1，换来训练和下一步 forward 的重叠。
- **carry 字段**：`carry=` 声明、随训练更新一起发布但不被优化器更新的字段，由 `pllm.carry` 写入。
- **位置状态（`pllm.PositionState`）**：每个 KV 块一个 hidden 宽的向量，只在 `@pllm.custom_forward` 里用
  `layers.load` / `layers.save` 读写。
- **放置（`placement=`）**：只读的大表放在显存（`"device"`）还是锁页主机内存（`"host"`）。

## 训练 {#zh-sec-4}

- **请求行训练**：`@pllm.train` 方法读运行中请求的行、每行返回一个 loss，pLLM 按状态归约、反向、更新。
  见[在线训练](/pllm/docs/guides/training/)。
- **归约（`reduction=`）**：同一份状态在一步里有多行时，loss 取平均（`"mean"`）还是求和（`"sum"`）。
- **宿主批次**：宿主代码拼好、经 `generate(batch=...)` 送进来的训练批次：`WeightedRowBatch`、`CausalTrainBatch`、
  `WeightedCEBatch`。见[宿主循环与批训练](/pllm/docs/guides/host-loops/)。
- **因果模型（`pllm.causal_model`）**：训练方法里可调用的整模型，带上指定的 `Adapter`，梯度流向适配器的状态。
- **`HostTrainer`**：`pllm.host.make_host` 返回的对象，在没有模型的情况下用宿主批次训练一个负载的状态。
- **前向内训练（`@pllm.train_in_forward`）**：在 forward 运行时，用投影的输入输出训练一份请求状态，并把它的增量
  加到投影上（TTT、In-Place TTT 一类）。见[改模型](/pllm/docs/guides/model-edits/)。

## 改模型与 forward {#zh-sec-5}

- **激活补丁（`ActivationPatch`）**：从状态读值，改某些层的输出（引导向量、ReFT）。
- **Adapter**：把状态当作权重增量加到指定投影上（LoRA 一类），不管状态由谁写。
- **定制残差（`@pllm.custom_residual`）**：接管某些层某个位置的残差更新、value 行或最终的流。
  见[改 forward 结构](/pllm/docs/guides/forward-structure/)。
- **状态子层**：带 `state=`、`chunk=` 的定制残差，每个请求一份按块更新的携带量（快权重记忆）。
- **定制 attention（`@pllm.custom_attention`）**：替换某些层读 KV 的方式；可以同时声明静态 **KV 布局**
  （滑动窗口、prompt 加窗口、跨层共享）。
- **custom forward（`@pllm.custom_forward`）**：用 Python 编排每个 token 跑哪些层、跑几遍、何时停。

## 执行与检查 {#zh-sec-6}

- **规划器**：pLLM 构建引擎时读负载声明、决定每个方法在一步的什么位置、哪条 stream 上运行、预留多少显存的那一部分；
  `pllm.explain` 打印的就是它的计划。见[看成本：pllm.explain](/pllm/docs/guides/explain/)。
- **准入（admission）**：pLLM 接纳一样东西的时刻。请求准入是请求进入引擎、分到状态副本并绑定它读的状态版本的那一刻；
  模型或写法的准入是构建引擎时检查它是否受支持，不受支持就报错。
- **Inductor**：PyTorch `torch.compile` 的代码生成器，把相邻的小算子融合成一个 kernel；vLLM 用它编译模型的 CUDA graph，
  pLLM 用它融合只读查表。
- **阶段（`pllm.Phase`）**：一步里的时间窗口：`BEFORE_FORWARD`、`IN_FORWARD`、`AFTER_FORWARD`、`BEFORE_SAMPLE`、
  `AFTER_SAMPLE`，以及由 pLLM 决定的 `AUTOMATIC`。每个效果有默认阶段，`@pllm.on(phase)` 可以改。
- **锚点**：`pllm.explain` 打印的、一步里的位置名：`request.prepare`、`model.execute`、`state.publish`、
  `sampler.pre_draw`、`token.commit` 等。
- **主 stream / 旁路 stream（lane）**：`main` 与 forward 串行；`side` 与下一步 forward 重叠。
- **eager 原因**：模型 forward 不在 CUDA graph 里的原因（`eager_reason`），空表示在。
- **中性设置（`neutral=`）**：强度为零的同一个负载；`testing.check` 要求它不改变采样。
- **契约检查**：`pllm.testing.check`（CPU）和 `check_on_gpu`（GPU）检查的性质：行的顺序、请求隔离、分块、抢占重算、
  版本可见性、中性设置。见[检查正确性](/pllm/docs/guides/checking/)。
- **预算**：`ExecutionOptions` 里限制 pLLM 显存的三个值：`state_page_budget_bytes`、`execution_peak_budget_bytes`、
  `cuda_graph_pool_budget_bytes`。见[配置项](/pllm/docs/reference/config/)。

</div>

</div>

<div class="zine-doc-footer">
  <a class="zine-pill-link" data-i18n-zh="← 返回 pLLM 项目与文档中心" data-i18n-en="← Back to pLLM Project &amp; Docs" href="/pllm/#docs">← Back to pLLM Project &amp; Docs</a>
  <a class="zine-pill-link" href="https://github.com/LinesHogan/pLLM/blob/main/doc/reference/glossary.md" target="_blank" rel="noopener">GitHub &middot; doc/reference/glossary.md ↗</a>
</div>

<script defer src="/assets/js/bilingual-post.js"></script>
