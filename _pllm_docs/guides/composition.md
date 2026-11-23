---
layout: page
title: "Composing Workloads"
description: "pLLM Documentation · User Guides"
permalink: /pllm/docs/guides/composition/
nav: false
---

<link rel="stylesheet" href="{{ '/assets/css/pllm-hub.css' | relative_url | bust_file_cache }}">

<div class="zine-dispatch" data-title-zh="组合负载" data-title-en="Composing Workloads" data-desc-zh="pLLM 文档 · 用户指南" data-desc-en="pLLM Documentation · User Guides">
  <div class="zine-dispatch-links">
    <a class="zine-pill-link primary" data-i18n-zh="← 返回 pLLM 文档中心" data-i18n-en="← pLLM Docs Hub" href="/pllm/#docs">← pLLM Docs Hub</a>
    <a class="zine-pill-link" data-i18n-zh="文档导航地图" data-i18n-en="Docs Roadmap" href="/pllm/docs/">Docs Roadmap</a>
    <span class="zine-lang-badge" data-i18n-zh="用户指南" data-i18n-en="User Guides">User Guides</span>
  </div>
  <a class="zine-pill-link" data-i18n-zh="GitHub 源码 ↗" data-i18n-en="View on GitHub ↗" href="https://github.com/LinesHogan/pLLM/blob/main/doc/guides/composition.md" target="_blank" rel="noopener">View on GitHub ↗</a>
</div>

<div id="markdown-content">

<div class="zine-lang-panel" data-lang-panel="en" lang="en" markdown="1">

<div id="en-composing-workloads"></div>

In inference-time algorithm research, you often develop multiple independent modules side by side: for example, an algorithm that trains a lightweight network online during decode to guide sampling (such as ESamp), alongside a probe that monitors residual norms at a specific layer in the background (such as StreamNorm). To run them inside the same inference engine, you do not need to merge all state declarations and tensor methods into one monolithic class. In pLLM, you simply write an outer `@dataclass(frozen=True)` workload and declare each sub-workload as a field.

At engine build time, pLLM walks all fields and flattens them into a unified execution graph—allocating memory, capturing CUDA graphs, and scheduling them concurrently under continuous batching. Every component keeps its own isolated namespace and state with zero cross-contamination, and any conflicts are caught and named explicitly at build time.

---

## Declaring an Outer Workload: Dataclass Fields {#en-declaring-an-outer-workload-dataclass-fields}

**Motivation**: Combine two independent `pllm.Workload` classes without modifying a single line of either sub-workload.

**Pattern**: Define the outer workload as a `@dataclass(frozen=True)` subclass of `pllm.Workload`, and declare each sub-workload as a field using `field(default_factory=...)`.

Why `default_factory`? Assigning a class instance directly as a dataclass default would cause all outer workload instances to share the exact same sub-workload object; `default_factory=MyWorkload` ensures each outer workload instance receives its own independent sub-workload instance and configuration.

```python
from dataclasses import dataclass, field

import torch
from torch import nn

import pllm
from pllm import testing

CONFIG = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=8)


class StreamNorm(pllm.Workload):
    @pllm.export(name="stream_norm", max_rows=8, max_cols=1)
    def norm(self, r: torch.Tensor = pllm.residual(12)) -> torch.Tensor:
        return r.float().norm(dim=-1, keepdim=True)


@dataclass(frozen=True)
class ESampLite(pllm.Workload):
    beta: float = 0.5

    distiller = pllm.RequestState(
        init=pllm.FromModule(
            lambda m: nn.Sequential(nn.Linear(m.hidden_size, 4), nn.Linear(4, m.hidden_size))
        ),
        optimizer=pllm.AdamW(1e-3),
    )

    @pllm.train(state="distiller")
    def loss(
        self,
        shallow: torch.Tensor = pllm.hidden(0),
        deep: torch.Tensor = pllm.hidden(-1),
        f: nn.Module = pllm.module("distiller"),
    ) -> torch.Tensor:
        return (f(shallow.float()) - deep.float()).square().mean(-1)

    def strength(self) -> float:
        return self.beta

    @pllm.guide_sampling(beta=strength)
    def predict(
        self,
        shallow: torch.Tensor = pllm.hidden(0),
        f: nn.Module = pllm.module("distiller"),
    ) -> torch.Tensor:
        return f(shallow.float())


@dataclass(frozen=True)
class Both(pllm.Workload):
    esamp: ESampLite = field(default_factory=ESampLite)
    monitor: StreamNorm = field(default_factory=StreamNorm)
```

---

## Namespace Isolation and Export Routing {#en-namespace-isolation-and-export-routing}

When you compose `ESampLite` and `StreamNorm` into `Both`, the outer workload's field names (`esamp` and `monitor`) become the namespace prefixes for each independent component:

1. **State and method prefixes**:
   In the execution plan and memory report from [`pllm.explain`](/pllm/docs/guides/explain/), all states, methods, and task names are automatically prefixed by their field name. For example:
   - `ESampLite`'s state `distiller` becomes `esamp.distiller`;
   - training method `loss` becomes `esamp.loss`, and sampling guidance method `predict` becomes `esamp.predict`;
   - monitoring method `norm` becomes `monitor.norm`.
   Even if two sub-workloads happen to use identical internal names (such as `loss` or `state`), no symbol collision occurs.

2. **Export event routing (`on_export`)**:
   `StreamNorm` declares `@pllm.export(name="stream_norm", ...)`. Inside the engine, the global exported tensor name is qualified as `monitor.stream_norm`. However, when export events arrive from GPU to the CPU host process, pLLM strips the field prefix `monitor.` based on the originating component and delivers the local tag `stream_norm` to `monitor.on_export(event)`.
   **Benefit**: Component authors write `on_export` callbacks without needing to know what field name the component will be mounted under later—existing code works out of the box.

Let's inspect the plan on CPU with `pllm.explain` and verify the scheduling contracts with `pllm.testing.check`:

```python
# Inspect the plan on CPU: methods and states carry esamp.* and monitor.* prefixes
print(pllm.explain(Both(), CONFIG))

# Run scheduling contract checks on CPU (including component independence verification)
print(testing.check(Both(), CONFIG, neutral=Both(esamp=ESampLite(beta=0.0))).passed)
```

---

## Component Independence and State Privacy {#en-component-independence-and-state-privacy}

In `Both`, can `monitor` read `esamp`'s state—for example, by declaring `f: nn.Module = pllm.module("esamp.distiller")` inside one of `monitor`'s methods?

**No—this is strictly forbidden.** Using any dotted name (`.`) in an input marker or state reference raises an exception at build time:
`components are independent; to share state, put both methods in one workload`.

### Why Enforce State Privacy? {#en-why-enforce-state-privacy}

This rule comes directly from pLLM's **component independence guarantee**:
- **State trajectories must match standalone execution**: Any component inside a composed workload must produce the exact bitwise-identical computation and state evolution trajectory (every step's state tensor values, optimizer step counts, and version updates) as when run alone in its own engine.
- **Preventing implicit timing coupling**: Allowing cross-field state reads/writes would introduce implicit execution-order dependencies (who runs first, and which intermediate version gets read), breaking pipelined and CUDA-graph overlapped scheduling and causing hard-to-reproduce numerical drift when standalone-tested components are combined.

### Contract Check: `component independence` {#en-contract-check-component-independence}

When you run [`pllm.testing.check`](/pllm/docs/guides/checking/) on a composed workload, the checker verifies not only standard batching scheduling contracts (row permutation, request isolation, preemption replay, etc.), but also **`component independence`**:
1. Runs the full outer composed workload `Both` on CPU;
2. Runs each sub-component (`ESampLite` and `StreamNorm`) individually;
3. Compares every output, state, and export event of each component across both runs, request by request and token by token; any mismatch raises `ContractError`.

> [!TIP]
> **What if two methods genuinely need to share state?**
> If two methods must read and write the same state (for example, one method trains small weights via `@pllm.train` and another reads those weights to rewrite forward residuals), they are logically two steps of a single algorithm rather than two orthogonal components. Declare them inside the same `pllm.Workload` class instead of splitting them across components. See [Workload Anatomy](/pllm/docs/guides/workload-anatomy/) and [State](/pllm/docs/guides/state/).

---

## Merging Effects: Multiple `@pllm.add_logits` {#en-merging-effects-multiple-pllmadd_logits}

How do results from different components combine onto the model?

- **Multiple `@pllm.add_logits` can coexist**:
  If component A and component B both declare `@pllm.add_logits`, they do not conflict. During the sampler epilogue, pLLM accumulates each component's logit delta onto the model's base logits in the declaration order of the outer dataclass fields.
- **Logit additions run before sampling guidance**:
  All `@pllm.add_logits` deltas are accumulated before `@pllm.guide_sampling` runs. Thus, if one component shifts the logits, a downstream sampling guidance method still performs filtering and log-reweighting on top of the updated distribution. See [Where Results Go](/pllm/docs/guides/results/).

---

## Composition Conflicts and Build-Time Errors {#en-composition-conflicts-and-build-time-errors}

Not every pair of effects can be stacked unconditionally. When two components make mutually exclusive demands on the same physical resource of the base engine, pLLM **rejects the build immediately at build time (`pllm.explain` or `pllm.make_llm`) and names both conflicting sides**, rather than letting a later declaration silently overwrite an earlier one.

Here is an intentional conflict where two components both declare `@pllm.guide_sampling`:

```python
@dataclass(frozen=True)
class ConflictingBoth(pllm.Workload):
    a: ESampLite = field(default_factory=ESampLite)
    b: ESampLite = field(default_factory=ESampLite)


try:
    pllm.explain(ConflictingBoth(), CONFIG)
except ValueError as e:
    print(f"Caught expected build conflict: {e}")
```

Running this raises `ValueError`:
`a.predict and b.predict both declare @pllm.guide_sampling; keep at most one guidance method`. The message explicitly names both conflicting methods, `a.predict` and `b.predict`.

### Other Composition Conflicts {#en-other-composition-conflicts}

For error messages covering other composition conflicts, see the [Effects and results](/pllm/docs/reference/errors/#en-effects-and-results) and [Model sites and custom forwards](/pllm/docs/reference/errors/#en-model-sites-and-custom-forwards) sections of the Error Catalog; for the full matrix of what can coexist vs. what is rejected and why, see [Composition Rules](/pllm/docs/reference/composition-rules/).

---

## Self-Check {#en-self-check}

1. A composed workload `Both` contains a monitoring component `monitor: StreamNorm`, whose method declares `@pllm.export(name="stream_norm", ...)`. When consuming events on the host, does `monitor.on_export(event)` receive `event.tensor_name` as `"monitor.stream_norm"` or `"stream_norm"`? Why?

<details markdown="1">
<summary>Answer</summary>

`"stream_norm"`. Before dispatching a GPU export event into a component's own `on_export` callback, pLLM's export router strips the outer field prefix `monitor.`. This preserves component reusability: sub-components can be written and unit-tested without knowing what field name an outer workload will mount them under.

</details>

2. A researcher wants component A to compute per-row classification features into state `feat`, and component B to read `feat` to produce logit biases. They write A and B as two sub-workloads in an outer dataclass, and hit an error when declaring `pllm.parameter("a.feat", "value")` in B. Why does pLLM disallow this, and what is the right implementation?

<details markdown="1">
<summary>Answer</summary>

pLLM enforces component independence: each component's state trajectory must match running it standalone. Cross-component state reads/writes would introduce implicit timing dependencies and break the `component independence` contract check.

Right approach: Because `feat` is a tightly shared intermediate state between the two methods, they belong to the same algorithm semantically. Put both the producer method and the consumer method of `feat` inside a single `pllm.Workload` class.

</details>

</div>

<div class="zine-lang-panel is-hidden" data-lang-panel="zh" lang="zh-CN" markdown="1">

<div id="zh-sec-1"></div>

在很多推理算法的研究中，你往往会同时开发多个独立的模块：例如一个在解码时在线训练轻量网络以引导采样的算法（如 ESamp），和一个在后台监控模型特定层残差范数的探针（如 StreamNorm）。如果要把它们放进同一个推理引擎，你不需要把所有状态定义与张量方法手动合并到一个庞大的类中——在 pLLM 中，只需编写一个外层 `@dataclass(frozen=True)` 负载，将各个子负载声明为其字段。

pLLM 会在构建引擎时遍历所有字段，把它们扁平化为一个统一的执行图：自动分配显存、捕获 CUDA graph、并在连续批处理中同时调度它们。每个组件保持命名空间与状态独立，互不污染，并且冲突会在构建期直接报错点名。

---

## 声明外层负载：dataclass 字段 {#zh-sec-2}

动机：将两个独立的 `pllm.Workload` 组合在一起，同时保持子负载的代码完全不用修改。

写法：把外层负载定义为继承自 `pllm.Workload` 的 `@dataclass(frozen=True)` 类，每个子负载使用 `field(default_factory=...)` 声明为一个字段。

为什么使用 `default_factory`？因为 dataclass 字段如果是类实例，直接赋值默认实例会导致所有外层负载实例共享同一个子负载对象；`default_factory=MyWorkload` 保证每次实例化外层负载时，各个子负载都能获得独立的实例与配置。

```python
from dataclasses import dataclass, field

import torch
from torch import nn

import pllm
from pllm import testing

CONFIG = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=8)


class StreamNorm(pllm.Workload):
    @pllm.export(name="stream_norm", max_rows=8, max_cols=1)
    def norm(self, r: torch.Tensor = pllm.residual(12)) -> torch.Tensor:
        return r.float().norm(dim=-1, keepdim=True)


@dataclass(frozen=True)
class ESampLite(pllm.Workload):
    beta: float = 0.5

    distiller = pllm.RequestState(
        init=pllm.FromModule(
            lambda m: nn.Sequential(nn.Linear(m.hidden_size, 4), nn.Linear(4, m.hidden_size))
        ),
        optimizer=pllm.AdamW(1e-3),
    )

    @pllm.train(state="distiller")
    def loss(
        self,
        shallow: torch.Tensor = pllm.hidden(0),
        deep: torch.Tensor = pllm.hidden(-1),
        f: nn.Module = pllm.module("distiller"),
    ) -> torch.Tensor:
        return (f(shallow.float()) - deep.float()).square().mean(-1)

    def strength(self) -> float:
        return self.beta

    @pllm.guide_sampling(beta=strength)
    def predict(
        self,
        shallow: torch.Tensor = pllm.hidden(0),
        f: nn.Module = pllm.module("distiller"),
    ) -> torch.Tensor:
        return f(shallow.float())


@dataclass(frozen=True)
class Both(pllm.Workload):
    esamp: ESampLite = field(default_factory=ESampLite)
    monitor: StreamNorm = field(default_factory=StreamNorm)
```

---

## 命名空间隔离与导出路由 {#zh-sec-3}

当你将 `ESampLite` 和 `StreamNorm` 组合进 `Both` 时，外层负载的字段名（`esamp` 和 `monitor`）成为各个独立组件（component）的命名空间前缀：

1. **状态与方法前缀**：
   在 [`pllm.explain`](/pllm/docs/guides/explain/) 的执行计划和显存报告中，所有状态、方法和任务名都会自动加上字段名前缀。例如：
   - `ESampLite` 的状态 `distiller` 变为 `esamp.distiller`；
   - 训练方法 `loss` 变为 `esamp.loss`，引导采样方法 `predict` 变为 `esamp.predict`；
   - 监控方法 `norm` 变为 `monitor.norm`。
   这样即使两个子负载内部恰好使用了同名的方法或状态（例如都叫 `loss` 或 `state`），也不会发生符号冲突。

2. **导出事件路由（`on_export`）**：
   `StreamNorm` 声明了 `@pllm.export(name="stream_norm", ...)`。在引擎内部，导出的全局张量名会被修饰为 `monitor.stream_norm`。但当导出事件从 GPU 传输到 CPU 宿主进程时，pLLM 会根据事件来自哪个组件剥离掉字段名前缀 `monitor.`，将本地标签 `stream_norm` 传递给对应组件实例的 `monitor.on_export(event)`。
   **收益**：组件作者编写 `on_export` 回调时，无需知道这个组件未来会被挂载在哪个字段名下，原有代码直接复用。

下面在 CPU 上运行 `pllm.explain` 查看执行计划，并用 `pllm.testing.check` 验证调度契约：

```python
# 在 CPU 上查看计划：方法和状态带有 esamp.* 与 monitor.* 前缀
print(pllm.explain(Both(), CONFIG))

# 在 CPU 上运行调度契约检查（包含 component independence 验证）
print(testing.check(Both(), CONFIG, neutral=Both(esamp=ESampLite(beta=0.0))).passed)
```

---

## 组件独立性与状态私有 {#zh-sec-4}

在 `Both` 中，`monitor` 能不能去读取 `esamp` 的状态？例如在 `monitor` 的方法里声明 `f: nn.Module = pllm.module("esamp.distiller")`？

**答案是严格禁止的。** 如果你在任何输入标记或状态引用里使用了包含点号 `.` 的名称，pLLM 构建时会抛出异常：
`components are independent; to share state, put both methods in one workload`。

### 为什么强制状态私有？ {#zh-sec-5}

这是由 pLLM 的**组件独立性保证**决定的：
- **状态轨迹必须等价于单跑**：任何一个组件在组合负载里的计算过程、状态演化轨迹（每一步的状态张量取值、优化器步数与版本更新），必须与它单独在一个引擎里运行时严格逐位一致。
- **防止隐式时序耦合**：如果组件之间能够跨字段读写状态，就会引入隐式的执行时序依赖（谁先算、谁读到了谁的哪一个中间版本），破坏流水线与 CUDA graph 的重叠调度，使得单独测试组件时通过、拼在一起时却产生难以复现的数值漂移。

### 契约检查：`component independence` {#zh-sec-6}

当你对组合负载运行 [`pllm.testing.check`](/pllm/docs/guides/checking/) 时，测试器不仅会检查常规的批处理调度契约（行置换、请求隔离、抢占重放等），还会额外执行 **`component independence`（组件独立性）** 检查：
1. 在 CPU 上完整运行一次外层组合负载 `Both`；
2. 依次单独运行每个子组件（`ESampLite` 和 `StreamNorm`）；
3. 逐个请求、逐个 token 比较每个组件在两处的所有输出、状态与导出事件；若存在任何不匹配，抛出 `ContractError`。

> [!TIP]
> **如果两个方法确实需要共享状态怎么办？**
> 如果两个方法需要读写同一个状态（例如一个方法用 `@pllm.train` 训练小权重，另一个方法读这套权重去改写前向残差），说明它们在算法逻辑上本就是同一个算法的两个步骤，而不是两个正交的组件。此时应当将它们声明在同一个 `pllm.Workload` 类内部，而不是拆成两个组件跨界访问。详见[负载的组成](/pllm/docs/guides/workload-anatomy/)与[状态](/pllm/docs/guides/state/)。

---

## 效果的合并：多个 `@pllm.add_logits` {#zh-sec-7}

不同组件产生的结果如何作用到模型上？

- **`@pllm.add_logits` 可以并存**：
  若组件 A 和组件 B 都声明了 `@pllm.add_logits`，它们不会互相冲突。pLLM 会在采样前处理阶段（sampler epilogue）将各组件输出的 logits 增量按外层 dataclass 的字段声明顺序依次累加到模型的 base logits 上。
- **先加 logits，再做引导采样**：
  全部 `@pllm.add_logits` 的累加在采样引导（`@pllm.guide_sampling`）之前完成。这意味着如果有组件修改了 logits，后续的采样引导方法依然可以在修改后的分布基础上执行过滤与对数重加权。详见[结果去哪](/pllm/docs/guides/results/)。

---

## 组合冲突与构建报错 {#zh-sec-8}

并非所有算法效果都可以无条件叠加。当两个组件对底座引擎的同一物理资源提出互斥需求时，pLLM **在构建期（`pllm.explain` 或 `pllm.make_llm`）直接拒绝启动并明确点名冲突的双方**，不会让后声明的效果悄悄覆盖先声明的效果。

先看一个故意制造冲突的例子：两个组件都声明了 `@pllm.guide_sampling`。

```python
@dataclass(frozen=True)
class ConflictingBoth(pllm.Workload):
    a: ESampLite = field(default_factory=ESampLite)
    b: ESampLite = field(default_factory=ESampLite)


try:
    pllm.explain(ConflictingBoth(), CONFIG)
except ValueError as e:
    print(f"捕获到预期的构建冲突: {e}")
```

运行这段代码会直接抛出 `ValueError`：
`a.predict and b.predict both declare @pllm.guide_sampling; keep at most one guidance method`。报错明确点名了冲突的双方 `a.predict` 和 `b.predict`。

### 其他组合冲突 {#zh-sec-9}

其余组合冲突的报错见报错目录的[效果与结果](/pllm/docs/reference/errors/#zh-sec-5)、[改模型与 forward 结构](/pllm/docs/reference/errors/#zh-sec-8)两节；哪些写法可以并存、哪些被拒绝及原因见[组合规则](/pllm/docs/reference/composition-rules/)。

---

## 自测 {#zh-sec-10}

1. 组合负载 `Both` 中包含监控组件 `monitor: StreamNorm`，其内部方法声明了 `@pllm.export(name="stream_norm", ...)`。在宿主端消费事件时，`monitor.on_export(event)` 接收到的 `event.tensor_name` 是 `"monitor.stream_norm"` 还是 `"stream_norm"`？为什么？

<details markdown="1">
<summary>答案</summary>

是 `"stream_norm"`。pLLM 的导出分发机制在将 GPU 导出的事件送入组件自身的 `on_export` 回调前，会自动剥离外层字段名前缀 `monitor.`。这样设计的目的是实现组件复用：子组件编写和单测时无需感知外部挂载它的字段名。

</details>

2. 研究员想要让组件 A 计算出每一行的分类特征，并将其存入状态 `feat`，再让组件 B 读取 `feat` 生成对应的 logits 偏置。他们把 A 和 B 写成了两个独立的子负载拼入外层 dataclass，但在 B 中声明 `pllm.parameter("a.feat", "value")` 时报错。为什么 pLLM 不支持这种写法？正确的实现方式是什么？

<details markdown="1">
<summary>答案</summary>

pLLM 强制组件独立性：每个组件的状态轨迹必须与它单独运行时完全一致，跨组件读写状态会引入隐式时序依赖并破坏单测契约（`component independence` 检查会失败）。

正确做法：由于状态 `feat` 本就是这两个方法紧密共享的核心中间量，它们在算法语义上属于同一个算法。应该把产生 `feat` 的方法和读取 `feat` 的方法写在同一个 `pllm.Workload` 类中。

</details>

</div>

</div>

<div class="zine-doc-footer">
  <a class="zine-pill-link" data-i18n-zh="← 返回 pLLM 项目与文档中心" data-i18n-en="← Back to pLLM Project &amp; Docs" href="/pllm/#docs">← Back to pLLM Project &amp; Docs</a>
  <a class="zine-pill-link" href="https://github.com/LinesHogan/pLLM/blob/main/doc/guides/composition.md" target="_blank" rel="noopener">GitHub &middot; doc/guides/composition.md ↗</a>
</div>

<script defer src="/assets/js/bilingual-post.js"></script>
