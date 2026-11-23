---
layout: page
title: "Custom Operators"
description: "pLLM Documentation · User Guides"
permalink: /pllm/docs/guides/custom-ops/
nav: false
---

<link rel="stylesheet" href="{{ '/assets/css/pllm-hub.css' | relative_url | bust_file_cache }}">

<div class="zine-dispatch" data-title-zh="自写算子" data-title-en="Custom Operators" data-desc-zh="pLLM 文档 · 用户指南" data-desc-en="pLLM Documentation · User Guides">
  <div class="zine-dispatch-links">
    <a class="zine-pill-link primary" data-i18n-zh="← 返回 pLLM 文档中心" data-i18n-en="← pLLM Docs Hub" href="/pllm/#docs">← pLLM Docs Hub</a>
    <a class="zine-pill-link" data-i18n-zh="文档导航地图" data-i18n-en="Docs Roadmap" href="/pllm/docs/">Docs Roadmap</a>
    <span class="zine-lang-badge" data-i18n-zh="用户指南" data-i18n-en="User Guides">User Guides</span>
  </div>
  <a class="zine-pill-link" data-i18n-zh="GitHub 源码 ↗" data-i18n-en="View on GitHub ↗" href="https://github.com/LinesHogan/pLLM/blob/main/doc/guides/custom-ops.md" target="_blank" rel="noopener">View on GitHub ↗</a>
</div>

<div id="markdown-content">

<div class="zine-lang-panel" data-lang-panel="en" lang="en" markdown="1">

<div id="en-custom-operators"></div>

When an inference-time algorithm needs an operation that PyTorch cannot express efficiently—such as lookup-table hashing, chunked state scans, or sparse attention page scoring—researchers typically write dedicated Triton or CUDA kernels.

In pLLM, integrating custom kernels does not require learning a proprietary operator framework: wrap your kernel as an operator object using PyTorch's native `torch.library.custom_op`, and call it directly inside a `pllm.Workload` method body just like a regular function. pLLM handles capturing it into CUDA graphs under continuous batching, planning meta-tensor shapes and device memory on CPU, tracking GPU-to-CPU synchronizations, and verifying numerical equivalence between your kernel and its CPU reference implementation via contract checks.

## Quickstart: Using a Custom Operator in a Workload {#en-quickstart-using-a-custom-operator-in-a-workload}

Below is a complete minimal example. We define a pure-PyTorch row-normalization operator `row_norm`, register its meta shape inference and FLOP formula, and invoke it inside an `@pllm.export` method.

```python
import math
from dataclasses import dataclass

import torch
from torch.utils.flop_counter import register_flop_formula

import pllm
from pllm import testing


@torch.library.custom_op("my_ops::row_norm", mutates_args=())
def row_norm(x: torch.Tensor) -> torch.Tensor:
    """Pure-PyTorch implementation, which also serves as the CPU reference."""
    return torch.linalg.vector_norm(x, dim=-1, keepdim=True)


@row_norm.register_fake
def _row_norm_fake(x: torch.Tensor) -> torch.Tensor:
    """Meta-tensor shape inference: provides output shape and dtype to the planner."""
    return torch.empty(x.shape[:-1] + (1,), dtype=x.dtype, device=x.device)


@register_flop_formula(torch.ops.my_ops.row_norm)
def _row_norm_flops(x_shape, *args, **kwargs) -> int:
    """Declare per-call FLOPs: vector norm takes roughly 2 FLOPs per element."""
    return 2 * math.prod(x_shape)


@dataclass(frozen=True)
class ExportNorm(pllm.Workload):
    @pllm.export(name="norm", max_cols=1)
    def observe(self, h: torch.Tensor = pllm.hidden(0)) -> torch.Tensor:
        # Call the op object directly rather than torch.ops.my_ops.row_norm
        return row_norm(h)


config = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=4)

# 1. Analyze execution plan and operator cost on CPU (zero GPU overhead)
report = pllm.explain(ExportNorm(), config)

# 2. Verify scheduling contracts on CPU
result = testing.check(ExportNorm(), config)
```

When you call `pllm.explain`, the report lists the operator's call site, FLOPs per row, and a lower bound on logical memory traffic in bytes. Calling `testing.check` executes the operator on CPU to verify scheduling contracts such as row permutation isolation and recomputation consistency.

## Integration Rules and Rationale {#en-integration-rules-and-rationale}

Calling a custom operator inside a workload method requires following the rules below, each of which corresponds to an execution constraint of continuous batching or CUDA graphs:

### 1. `mutates_args=()` and Returning New Tensors {#en-sec-1}

You must declare `mutates_args=()` when defining the operator, perform no in-place mutations on input tensors inside the body (such as `x.add_()` or slice assignment), and always return new tensors.

- **Why**: Under continuous batching and CUDA graphs, the activations and state tensors passed to a method are shared or paged memory views borrowed by the current task. Other methods in the same step may be reading the same activation buffer, and earlier tasks may still hold references to prior state pages. In-place mutation bypasses pLLM's versioned state publication and corrupts data.
- **How**: Always allocate and return new tensors for operator outputs; publish any persistent state changes exclusively through framework effects such as `@pllm.write` or `@pllm.train`.

### 2. `register_fake` Is Required {#en-sec-2}

After defining the operator, you must register a meta-device shape-inference function via `@op.register_fake`.

- **Why**: `pllm.explain` and the engine's startup memory planner run entirely on CPU. The framework traces the full execution flow using PyTorch `meta` tensors (which track shape and dtype without allocating physical storage) to determine per-step peak memory, task buffers, and GPU-to-CPU syncs. Without a fake implementation, shape inference breaks and planning cannot proceed.
- **How**: Have the fake function take the same arguments as the real operator and return meta tensors via `torch.empty(...)` or `torch.empty_like(...)` with the exact output shape and dtype.

### 3. Output Shapes Must Depend Strictly on Input Shapes {#en-sec-3}

Every dimension of an operator's output tensors must be derivable from input tensor shapes alone—never from values inside the tensors (do not call `ctx.new_dynamic_size()` or slice dynamically based on tensor data).

- **Why**: Workload methods are captured into CUDA graphs. CUDA graphs require the memory topology and tensor sizes across the entire batch to be statically fixed at capture time and cannot accommodate data-dependent size changes.
- **How**: If a result is logically variable-length (for example, top-K sparse scoring), return a fixed maximum-length tensor paired with a boolean mask indicating valid entries.

### 4. Call the Op Object Directly, Not `torch.ops.<namespace>.<name>` {#en-sec-4}

Inside workload methods, invoke the operator object created by `torch.library.custom_op` directly (e.g., `row_norm(h)`) rather than going through the dispatcher (`torch.ops.my_ops.row_norm(h)`).

- **Why**: When compiling and analyzing a workload, pLLM statically inspects the global symbols referenced by each method to extract the operator definition and its underlying Triton/CUDA kernel source. Kernel source changes are hashed into the workload's identity fingerprint so cached graphs are automatically invalidated when the kernel changes. Looking up via `torch.ops` hides the source object and prevents the framework from tracking versions or shipping the registration to worker processes.
- **How**: Define the operator at module top level or import it via `from my_kernels import my_op`, and pass arguments directly to that object inside your method.

### 5. Register `register_autograd` for Online Training {#en-sec-5}

If the operator sits on the backward path of `@pllm.train` online training, you must register a backward rule with `op.register_autograd(...)`.

- **Why**: pLLM's online training relies on PyTorch's autograd engine to build the gradient graph. If gradients must flow through a custom operator that lacks a backward rule, autograd raises an error and aborts. Operators used only in forward inference, residual rewrites, sampling guidance, or activation exports do not need autograd registration.
- **How**: Write a backward function returning gradients for each input and bind it via `op.register_autograd(backward_fn)`.

### 6. Register `register_vmap` in Per-Token Cells {#en-sec-6}

If the operator is called inside an `@pllm.custom_forward` unit executing per token (`over="token"`), you must register a batching rule with `op.register_vmap(...)`.

- **Why**: Under `over="token"`, the framework uses `torch.vmap` to vectorize single-token logic across the whole token batch. `vmap` rejects any custom operator without a registered batching rule.
- **How**: Register the batching behavior with `op.register_vmap(vmap_fn)`, or switch to whole-span scheduling (`over="span"`) when per-token granularity is unnecessary.

### 7. Optional `register_kernel("cpu")` {#en-sec-7}

We recommend providing a pure-PyTorch CPU reference implementation for your GPU kernel. If `device_types="cuda"` is omitted, the Python function body itself serves as the fallback implementation; if `device_types="cuda"` is specified explicitly, register a CPU version via `@op.register_kernel("cpu")`.

- **Why**: `pllm.testing.check` verifies continuous-batching scheduling contracts (such as cross-row isolation and chunked prompt handling) quickly without touching a GPU or loading weights. Without a CPU reference, CPU contract checks cannot execute the operator's math, and subsequent GPU checks cannot compare across devices.
- **How**: Write and register a mathematically equivalent function using native PyTorch ops on CPU.

### 8. Optional `register_flop_formula` {#en-sec-8}

You can register a FLOP formula using `@register_flop_formula(torch.ops.<namespace>.<name>)`.

- **Why**: `pllm.explain` estimates the compute cost of each step in your workload. With a formula registered, the report displays `[declared] FLOP/row`; without one, it marks the call as `unknown [no flop formula, counted 0]`.
- **How**: Register a function that accepts tensor shapes and returns the total floating-point operations for that call.

## Verification and Device Equivalence (`check_on_gpu`) {#en-verification-and-device-equivalence-check_on_gpu}

Passing `testing.check` on CPU only confirms that the CPU reference implementation satisfies the scheduling contracts. When you write a custom Triton or CUDA kernel, the real risks are numerical divergence between the GPU kernel and CPU reference, or hidden GPU-to-CPU host synchronizations inside the kernel.

`pllm.testing.check_on_gpu` catches both issues on a real GPU engine:

```python
print(testing.check_on_gpu(ExportNorm(), config))
```

For workloads containing custom operators, `check_on_gpu` runs a **device equivalence** check: it copies the step-by-step input tensors from the CPU contract test script onto the GPU, executes the actual GPU kernel, and compares outputs against the CPU reference within the default tolerances for the dtype (`torch.testing.assert_close`). It also verifies that 8 consecutive steps of steady decode run without any host synchronization.

If the GPU kernel diverges mathematically from the CPU reference, `check_on_gpu` raises `ContractError`:

```text
1 contract failure(s)
[device equivalence] [device equivalence] add: Tensor-likes are not close!

Mismatched elements: 12715 / 13440 (94.6%)
Greatest absolute difference: 2.53125 at index (5, 581) (up to 1e-05 allowed)
Greatest relative difference: 5536.0 at index (8, 688) (up to 0.016 allowed)
```

The failure report pinpoints the fraction of mismatched elements along with the maximum absolute and relative errors, helping you track down missing terms or precision issues in the kernel.

## Measured Throughput and Practical Guidance {#en-measured-throughput-and-practical-guidance}

Before writing a custom kernel, keep one reality in mind: **custom kernels do not automatically speed up end-to-end inference**.

In [09 Custom Triton Kernels](/pllm/docs/examples/09_custom_kernels/), we inserted a relative steering operator after layer 12's MLP output (computing residual-stream norms and adding a scaled direction vector). On an RTX 4090 running vLLM 0.32.0 with Qwen2.5-7B-Instruct (bf16, CUDA graphs) across 128 concurrent requests (8 prompts × 16 samples, 256 tokens each, median of 3 rounds after 1 warmup round), we measured:

```text
Native vLLM   :   5872 token/s, 100% of native vLLM
Triton kernel :   5760 token/s,  98% of native vLLM
PyTorch op    :   5755 token/s,  98% of native vLLM
```

The handwritten Triton kernel and the pure-PyTorch op achieve virtually identical throughput (both 98% of native vLLM). A single-layer residual vector addition takes negligible time and is nowhere near the compute bottleneck of full-model inference.

The right motivation for writing a custom kernel is when an algorithm genuinely involves operations PyTorch cannot express efficiently—such as custom hash collision handling or fine-grained masked scans over token history—not rewriting basic linear algebra in hopes of free speedups.

For complete Triton kernel definitions, safetensors weight binding, and benchmark code, see [09 Custom Triton Kernels](/pllm/docs/examples/09_custom_kernels/). For details on method-body boundaries, see [Workload Anatomy](/pllm/docs/guides/workload-anatomy/); for pre-launch cost analysis, see [Seeing Costs Early: pllm.explain](/pllm/docs/guides/explain/).

## Common Errors and Troubleshooting {#en-common-errors-and-troubleshooting}

For exact error messages, root causes, and fixes when patterns on this page are rejected, see the [Custom operators](/pllm/docs/reference/errors/#en-custom-operators) and [Method bodies](/pllm/docs/reference/errors/#en-method-bodies) sections of the Error Catalog.

## Self-Check {#en-self-check}

<details markdown="1">
<summary>1. Why must a workload method call the custom op object directly instead of <code>torch.ops.my_lib.my_kernel(...)</code>?</summary>

pLLM statically inspects the global symbols referenced by each method so it can read the custom operator and its underlying Triton/CUDA kernel source directly. Kernel source changes are hashed into the workload's identity fingerprint to trigger graph recompilation automatically when a kernel is edited, and the op object carries its registration to worker processes. Going through the `torch.ops` dispatcher obscures the source object and prevents the framework from tracking kernel changes or shipping the registration cleanly.

</details>

<details markdown="1">
<summary>2. Suppose a custom operator filters out tokens based on scores so its output length varies per step. Can it return a variable-length tensor directly? Why or why not?</summary>

No. Workload methods run inside CUDA graphs, which require tensor memory layouts and every dimension size to be completely static at capture time. If an output shape depends on tensor data, graph capture will fail. Instead, return a fixed maximum-length tensor alongside a boolean mask, and ignore invalid positions in downstream operations using `torch.where` or masked multiplication.

</details>

<details markdown="1">
<summary>3. If a custom operator is used only for rewriting the residual stream (<code>@pllm.custom_residual</code>) and exporting observations (<code>@pllm.export</code>), does it need <code>register_autograd</code>?</summary>

No. `register_autograd` is only required when the operator lies on the backward pass of `@pllm.train` online training. Pure inference, activation exports, and forward-only rewrites do not invoke backpropagation and need no autograd rule.

</details>

</div>

<div class="zine-lang-panel is-hidden" data-lang-panel="zh" lang="zh-CN" markdown="1">

<div id="zh-sec-1"></div>

当推理时算法需要执行 PyTorch 无法高效表达的步骤——例如查表哈希、状态分块扫描（chunked scan）或稀疏注意力页打分时，研究者通常会编写专门的 Triton 或 CUDA 核函数。

在 pLLM 中，接入自定义核函数无需学习专有算子框架：使用 PyTorch 原生的 `torch.library.custom_op` 将核函数包装为算子对象，并在 `pllm.Workload` 的方法体中像调用普通函数一样直接调用它。pLLM 会负责将其捕获进连续批处理下的 CUDA graph、在 CPU 上完成元张量形状与显存规划、追踪 GPU 到 CPU 的同步，并通过契约检查验证核函数与 CPU 参考实现的数值等价性。

## 快速上手：在负载中使用自写算子 {#zh-sec-2}

下面是一个完整的最小示例。我们定义一个纯 PyTorch 实现的行归一化算子 `row_norm`，注册 meta 形状推导与 FLOP 计算公式，并在 `@pllm.export` 方法中调用它。

```python
import math
from dataclasses import dataclass

import torch
from torch.utils.flop_counter import register_flop_formula

import pllm
from pllm import testing


@torch.library.custom_op("my_ops::row_norm", mutates_args=())
def row_norm(x: torch.Tensor) -> torch.Tensor:
    """纯 PyTorch 实现，同时作为 CPU 参考实现。"""
    return torch.linalg.vector_norm(x, dim=-1, keepdim=True)


@row_norm.register_fake
def _row_norm_fake(x: torch.Tensor) -> torch.Tensor:
    """元张量形状推导：为规划器提供输出形状与 dtype。"""
    return torch.empty(x.shape[:-1] + (1,), dtype=x.dtype, device=x.device)


@register_flop_formula(torch.ops.my_ops.row_norm)
def _row_norm_flops(x_shape, *args, **kwargs) -> int:
    """声明每行计算 FLOP：向量范数大约对应每个元素 2 次浮点运算。"""
    return 2 * math.prod(x_shape)


@dataclass(frozen=True)
class ExportNorm(pllm.Workload):
    @pllm.export(name="norm", max_cols=1)
    def observe(self, h: torch.Tensor = pllm.hidden(0)) -> torch.Tensor:
        # 直接调用算子对象本身，不使用 torch.ops.my_ops.row_norm
        return row_norm(h)


config = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=4)

# 1. 在 CPU 上分析规划与算子成本（无 GPU 开销）
report = pllm.explain(ExportNorm(), config)

# 2. 在 CPU 上检查调度契约
result = testing.check(ExportNorm(), config)
```

调用 `pllm.explain` 时，报告中会列出该算子的调用位置、每行 FLOP 与逻辑访存字节下界；调用 `testing.check` 则会在 CPU 上执行该算子，验证调度契约（如行重排隔离与重算一致性）。

## 接入规则与设计原因 {#zh-sec-3}

在负载方法中调用自定义算子需要遵守以下规则，每条规则均对应底层连续批处理或 CUDA graph 的执行约束：

### 1. `mutates_args=()` 与返回新张量 {#zh-sec-4}

定义算子时必须声明 `mutates_args=()`，方法体内不要对入参张量做任何原地修改（如 `x.add_()` 或切片赋值），必须返回新张量。

- **原因**：在连续批处理与 CUDA graph 下，方法接收到的激活值和状态张量是框架借给当前任务的共享或分页内存视图。同一步内的其他方法可能正在读取同一块激活，而旧版本状态页面也可能仍被前序任务引用。原地修改会绕过 pLLM 的版本发布机制破坏数据。
- **做法**：算子内部始终将结果写入新分配的张量并返回；任何持久化状态变更一律通过 `@pllm.write` 或 `@pllm.train` 等框架效果发布。

### 2. 必须注册 `register_fake` {#zh-sec-5}

定义算子后，必须使用 `@op.register_fake` 注册 meta 设备张量的形状推导函数。

- **原因**：`pllm.explain` 以及引擎启动阶段的显存规划器完全在 CPU 上运行。框架使用 PyTorch 的 `meta` 设备张量（仅保留形状和数据类型，不分配物理存储）追踪整网执行流，以确定各步的显存峰值、任务缓冲区并检查 GPU→CPU 同步。若缺少 fake 函数，形状推导中断，规划无法进行。
- **做法**：fake 函数接收与真实算子相同的输入参数，使用 `torch.empty(...)` 或 `torch.empty_like(...)` 返回与真实输出形状、数据类型完全一致的 meta 张量。

### 3. 输出形状必须严格由输入形状决定 {#zh-sec-6}

算子输出张量的每一维大小必须由输入张量的形状推导得出，不能依赖输入张量内部的具体数值（不能调用 `ctx.new_dynamic_size()`，也不能根据内容做动态长度切片）。

- **原因**：负载方法会被捕获进 CUDA graph。CUDA graph 要求整批计算的内存拓扑与张量大小完全静态确定，无法容纳由数据内容动态决定的尺寸变动。
- **做法**：若计算结果在逻辑上是不定长的（例如稀疏打分保留 top-K），返回固定最大长度的张量，并附带一个布尔掩码（mask）指示有效元素。

### 4. 调用算子对象本身，而非 `torch.ops.<命名空间>.<名字>` {#zh-sec-7}

在负载方法中，必须直接调用通过 `torch.library.custom_op` 声明的算子对象（如 `row_norm(h)`），不要通过分发器写成 `torch.ops.my_ops.row_norm(h)`。

- **原因**：pLLM 在编译和分析负载时，会静态检查方法引用的全局对象，提取算子定义及其底层的 Triton/CUDA 核函数源码。核函数的源码变更会被计入负载身份指纹，从而在核函数修改时自动使旧图缓存失效。通过 `torch.ops` 分发器间接查找会遮蔽源码对象，使框架无法追踪其版本。
- **做法**：在模块顶层定义或通过 `from my_kernels import my_op` 导入算子对象，在方法内直接传参调用。

### 5. 在线训练时需注册 `register_autograd` {#zh-sec-8}

如果该算子处于 `@pllm.train` 在线训练的反向传播计算路径上，必须通过 `op.register_autograd(...)` 注册反向传播规则。

- **原因**：pLLM 的在线训练依赖 PyTorch 的 autograd 引擎构建梯度图。若梯度需要穿过自定义算子而算子没有反向规则，自动求导会在该节点报错中断。若算子仅用于推理前向、残差改写、采样引导或激活导出，则无需注册。
- **做法**：编写返回各入参梯度的 backward 函数，并通过 `op.register_autograd(backward_fn)` 完成绑定。

### 6. 在逐 token 单元中需注册 `register_vmap` {#zh-sec-9}

如果算子在 `@pllm.custom_forward` 且按逐 token（`over="token"`）执行的单元中调用，必须通过 `op.register_vmap(...)` 注册批处理规则。

- **原因**：在 `over="token"` 模式下，框架利用 `torch.vmap` 将针对单个 token 的运算自动矢量化到整批 token 上。未注册 vmap 规则的自定义算子会被 `vmap` 拒绝。
- **做法**：使用 `op.register_vmap(vmap_fn)` 注册批处理展开逻辑；或者在无需逐 token 细粒度控制时，改用整段调度的 `over="span"`。

### 7. 可选注册 `register_kernel("cpu")` {#zh-sec-10}

推荐为 GPU 核函数补充纯 PyTorch 的 CPU 参考实现。若未指定 `device_types="cuda"`，定义算子时的 Python 函数体本身即可作为通用实现；若显式指定了 `device_types="cuda"`，则通过 `@op.register_kernel("cpu")` 补充。

- **原因**：`pllm.testing.check` 坚持在不接触 GPU、不加载权重的环境下快速检查连续批处理调度契约（如跨行干扰、提示词分块处理等）。若缺乏 CPU 参考实现，CPU 契约检查无法执行算子的数学运算；后续 GPU 检查也无法进行跨设备比对。
- **做法**：编写数学等价的 PyTorch 原生 CPU 函数进行注册。

### 8. 可选注册 `register_flop_formula` {#zh-sec-11}

可通过 `@register_flop_formula(torch.ops.<命名空间>.<名字>)` 为算子注册 FLOP 计算公式。

- **原因**：`pllm.explain` 会估算负载中每个步骤的计算开销。注册公式后，报告会显示明确的 `[declared] FLOP/row`；未注册时，报告中标记为 `unknown [no flop formula, counted 0]`。
- **做法**：注册接收张量形状并返回该次调用总浮点运算次数的函数。

## 验证与设备等价检查（`check_on_gpu`） {#zh-sec-12}

在 CPU 上通过 `testing.check` 仅说明 CPU 参考实现符合调度契约。当编写了专门的 Triton 或 CUDA 核函数时，真正的隐患往往是 GPU 核函数与 CPU 参考实现算得不一致，或者核函数内部偷偷触发了 GPU 到 CPU 的同步。

`pllm.testing.check_on_gpu` 用于在真实 GPU 引擎上排查这些问题：

```python
print(testing.check_on_gpu(ExportNorm(), config))
```

在包含自定义算子的负载上，`check_on_gpu` 会执行**设备等价（device equivalence）**检查：将 CPU 契约检查脚本中的各步输入张量搬移到 GPU 上重放，调用实际的 GPU 核函数计算，并按对应数据类型的默认容差（`torch.testing.assert_close`）比较两者的输出。同时，它会在连续 8 个稳定 decode 步中确认没有任何主机同步。

如果 GPU 核函数与 CPU 参考实现存在数学逻辑差异，`check_on_gpu` 会抛出 `ContractError`：

```text
1 contract failure(s)
[device equivalence] [device equivalence] add: Tensor-likes are not close!

Mismatched elements: 12715 / 13440 (94.6%)
Greatest absolute difference: 2.53125 at index (5, 581) (up to 1e-05 allowed)
Greatest relative difference: 5536.0 at index (8, 688) (up to 0.016 allowed)
```

错误报告会明确指出不匹配的元素比例、最大绝对差与相对差，帮助定位核函数实现中的遗漏或精度问题。

## 实测性能与使用建议 {#zh-sec-13}

写自定义算子前需要明确：**自定义核函数并不保证端到端推理提速**。

在 [09 自写 Triton 算子](/pllm/docs/examples/09_custom_kernels/) 示例中，我们在第 12 层 MLP 输出后插入了一个相对引导算子（计算残差流范数并按比例加入方向向量）。在 RTX 4090、vLLM 0.32.0、Qwen2.5-7B-Instruct（bf16，CUDA graph）下对 128 个并发请求（8 个 prompt × 16 个样本，各 256 token，预热 1 轮后 3 轮取中位数）进行吞吐测试，测得数据如下：

```text
原生 vLLM   :   5872 token/s，为原生 vLLM 的 100%
Triton 算子 :   5760 token/s，为原生 vLLM 的 98%
PyTorch 算子:   5755 token/s，为原生 vLLM 的 98%
```

可以看到，手写的 Triton 核函数与纯 PyTorch 算子的吞吐几乎一致（均为原生 vLLM 的 98%）。这是因为单层残差向量加法本身耗时极短，根本不是整网推理的计算瓶颈。

编写自定义核函数的正确动机，是算法中确实包含了 PyTorch 难以高效表达的操作（如特殊的哈希碰撞处理、按 token 历史的细粒度掩码扫描等），而非盲目为了提速重写基本线性代数运算。

关于 Triton 核函数的完整定义、safetensors 权重绑定与吞吐基准代码，请参阅 [09 自写 Triton 算子](/pllm/docs/examples/09_custom_kernels/)。关于方法体边界的更多细节，可参阅[负载的组成](/pllm/docs/guides/workload-anatomy/)；关于启动前分析，可参阅[看成本：pllm.explain](/pllm/docs/guides/explain/)。

## 常见报错与排查 {#zh-sec-14}

这一页的写法被拒绝时的报错原文、原因和改法，见报错目录的[自写算子](/pllm/docs/reference/errors/#zh-sec-9)、[方法体](/pllm/docs/reference/errors/#zh-sec-4)几节。

## 自测 {#zh-sec-15}

<details markdown="1">
<summary>1. 为什么在负载方法体里必须直接调用自定义算子对象，而不能写 <code>torch.ops.my_lib.my_kernel(...)</code>？</summary>

pLLM 需要静态分析方法引用的全局符号，直接读取自定义算子及其底层的 Triton/CUDA 核函数源码。核函数的源码变更会被哈希进负载身份指纹，确保核函数修改后能够自动触发图重编。通过 `torch.ops` 分发器查找会绕过这一源码分析过程，使框架无法感知核函数代码的变化。

</details>

<details markdown="1">
<summary>2. 某个自定义算子需要根据打分剔除一部分 token，输出长度在每步并不固定，能否在算子内直接返回变长张量？为什么？</summary>

不能。负载方法运行在 CUDA graph 内，CUDA graph 要求录制时的张量内存布局和每维大小完全静态固定。如果输出形状由张量数据决定，图捕获会报错失败。正确的做法是返回固定最大长度的张量，同时返回一个布尔掩码（mask），在后续步骤中使用 `torch.where` 或掩码乘法忽略无效位置。

</details>

<details markdown="1">
<summary>3. 如果自定义算子仅用于改写残差流（<code>@pllm.custom_residual</code>）和导出观测（<code>@pllm.export</code>），需要注册 <code>register_autograd</code> 吗？</summary>

不需要。`register_autograd` 仅在算子处于 `@pllm.train` 在线训练的反向求导路径上时才是必需的。纯推理、激活观测或前向重写不涉及反向传播，无需注册自动求导规则。

</details>

</div>

</div>

<div class="zine-doc-footer">
  <a class="zine-pill-link" data-i18n-zh="← 返回 pLLM 项目与文档中心" data-i18n-en="← Back to pLLM Project &amp; Docs" href="/pllm/#docs">← Back to pLLM Project &amp; Docs</a>
  <a class="zine-pill-link" href="https://github.com/LinesHogan/pLLM/blob/main/doc/guides/custom-ops.md" target="_blank" rel="noopener">GitHub &middot; doc/guides/custom-ops.md ↗</a>
</div>

<script defer src="/assets/js/bilingual-post.js"></script>
