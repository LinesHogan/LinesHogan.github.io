---
layout: page
title: "Going to Production: Multiprocessing, Tensor Parallelism, and Quantized Base Models"
description: "pLLM Documentation · How-To & Serving Guides"
permalink: /pllm/docs/howto/production/
nav: false
---

<link rel="stylesheet" href="{{ '/assets/css/pllm-hub.css' | relative_url | bust_file_cache }}">

<div class="zine-dispatch" data-title-zh="上生产：多进程、张量并行与量化底座" data-title-en="Going to Production: Multiprocessing, Tensor Parallelism, and Quantized Base Models" data-desc-zh="pLLM 文档 · 操作与上线指南" data-desc-en="pLLM Documentation · How-To &amp; Serving Guides">
  <div class="zine-dispatch-links">
    <a class="zine-pill-link primary" data-i18n-zh="← 返回 pLLM 文档中心" data-i18n-en="← pLLM Docs Hub" href="/pllm/#docs">← pLLM Docs Hub</a>
    <a class="zine-pill-link" data-i18n-zh="文档导航地图" data-i18n-en="Docs Roadmap" href="/pllm/docs/">Docs Roadmap</a>
    <span class="zine-lang-badge" data-i18n-zh="操作与上线指南" data-i18n-en="How-To &amp; Serving Guides">How-To &amp; Serving Guides</span>
  </div>
  <a class="zine-pill-link" data-i18n-zh="GitHub 源码 ↗" data-i18n-en="View on GitHub ↗" href="https://github.com/LinesHogan/pLLM/blob/main/doc/howto/production.md" target="_blank" rel="noopener">View on GitHub ↗</a>
</div>

<div id="markdown-content">

<div class="zine-lang-panel" data-lang-panel="en" lang="en" markdown="1">

<div id="en-going-to-production-multiprocessing-tensor-parallelism-and-quantized-base-models"></div>

After validating an algorithm in a notebook or local script, promoting a `pllm.Workload` to production serving typically involves three engineering requirements: isolating the inference engine into a separate subprocess (`multiprocessing="auto"` or `True`) for online traffic, enabling Tensor Parallelism across multiple GPUs to serve larger models, and running efficiently on quantized base models (FP8, AWQ, GPTQ).

This guide covers how to configure these three capabilities, their constraints, and the mechanisms behind them.

---

## Multiprocessing: EngineCore Isolation and Serialization Requirements {#en-multiprocessing-enginecore-isolation-and-serialization-requirements}

`ModelConfig` sets `multiprocessing="auto"` by default: during local single-GPU debugging with `pllm.make_llm`, the workload and engine run in the same Python process; when using `pllm.make_async_llm`, `pllm serve`, or `tensor_parallel_size > 1`, pLLM automatically isolates the execution engine into a separate EngineCore worker subprocess (you can also pass `multiprocessing=True` / `False` explicitly to override the default policy). When multiprocessing is enabled, the following serialization rules apply:

The driver serializes the workload object via `pickle` and sends it to the EngineCore worker.

```python
from dataclasses import dataclass

import torch
from torch import nn
import pllm
from pllm import testing

CONFIG = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=8)


@dataclass(frozen=True)
class StreamMonitor(pllm.Workload):
    @pllm.export(name="norm", max_rows=8, max_cols=1)
    def norm(self, r: torch.Tensor = pllm.residual(12)) -> torch.Tensor:
        return r.float().norm(dim=-1, keepdim=True)


# Verify planning and scheduling contracts on CPU
report = pllm.explain(StreamMonitor(), CONFIG)
check_result = testing.check(StreamMonitor(), CONFIG)
```

To launch the engine in multiprocessing mode for production serving:

```python
from vllm import SamplingParams

from my_project.workloads import StreamMonitor

mp_config = pllm.ModelConfig(
    "Qwen/Qwen2.5-0.5B-Instruct",
    max_num_seqs=8,
    multiprocessing=True,
)

with pllm.make_llm(mp_config, workload=StreamMonitor()) as llm:
    outputs = llm.generate(["Hello world"], SamplingParams(max_tokens=8))
```

In multiprocessing mode, workload code must satisfy the following constraints:

1. **Workload fields must be picklable (no locks, open files, or live handles)**  
   **Why**: Workloads are sent to the worker subprocess via `cloudpickle`: classes and functions defined in `__main__` or notebook cells (including lambdas and closures) are serialized by value, whereas those in importable modules are serialized by import path. Process-bound objects such as locks, open file handles, and sockets cannot be serialized and fail during encoding.
2. **The workload module must be importable inside the worker**  
   **Why**: For classes passed by import path, the worker re-imports the defining module, so that module must be on the worker's `PYTHONPATH`.
3. **The module must be import-safe**  
   **Why**: When the worker subprocess starts up, it imports your module at top level before initializing the CUDA runtime or GPU memory. Any top-level code that allocates GPU tensors or touches the CUDA context will crash the subprocess before device initialization completes.
4. **The worker plans the workload itself**  
   **Why**: After unpickling the workload, the worker performs planning and binding in its own process, and the driver only retrieves the resulting plan. Modules sent by import path are imported directly by the worker, so—just as in any Python multiprocessing program—modifying those modules between calling `make_llm` and worker startup is undefined behavior.
5. **Model objects cannot be stored as workload fields**  
   **Why**: Base model weights typically span gigabytes to tens of gigabytes and are already loaded and resident in GPU memory by the worker. Workload fields must not hold duplicate model weights. If your algorithm needs to invoke the base model inside a method (for example, causal language model training), use `pllm.causal_model("adapter")`, which directly reuses the base model already loaded in the worker.

<details markdown="1">
<summary>Self-check: Why does storing a model object in a workload field fail in multiprocessing mode?</summary>

Because the base model occupies large amounts of VRAM and is bound to a specific CUDA device context, making it unsafe to transfer across processes via `pickle`. Moreover, the EngineCore worker already owns the loaded base model—duplicating it wastes memory and causes deserialization errors. Inside workload methods, use `pllm.causal_model` to access the already-loaded model.

</details>

---

## Multi-GPU Tensor Parallelism (TP > 1) {#en-multi-gpu-tensor-parallelism-tp--1}

When a single GPU does not have enough VRAM to hold the full model weights, or when higher memory bandwidth is needed, enable multi-GPU Tensor Parallelism via `ModelConfig(tensor_parallel_size=2, ...)`.

Tensor parallelism requires vLLM 0.32.0 or later.

If you need long-running concurrent requests, token-by-token streaming, or an OpenAI-compatible HTTP endpoint, see the [Online Serving Guide](/pllm/docs/howto/online-serving/).

```python
from vllm import SamplingParams

tp_config = pllm.ModelConfig(
    "Qwen/Qwen2.5-0.5B-Instruct",
    max_num_seqs=8,
    tensor_parallel_size=2,
)

with pllm.make_llm(tp_config, workload=StreamMonitor()) as llm:
    outputs = llm.generate(["Tensor parallel run"], SamplingParams(max_tokens=8))
```

### How It Works: Zero Cross-GPU Collectives and Staggered Publishing {#en-how-it-works-zero-cross-gpu-collectives-and-staggered-publishing}

Under Tensor Parallelism, vLLM must execute a cross-GPU synchronization (Custom All-Reduce) after every attention and MLP layer. If pLLM introduced its own cross-GPU collective communication, or if side-stream training compute contended with vLLM's All-Reduce for interconnect bandwidth, every GPU would stall severely.

To prevent this, pLLM uses two mechanisms:
1. **Zero pLLM cross-GPU collectives**: Hidden-state reads (`pllm.hidden`, `pllm.residual`, `pllm.final_hidden`), sampling guidance (`@pllm.guide_sampling`), logits adjustments (`@pllm.add_logits`), and request-level states maintain an identical replica on each TP rank. Every GPU executes the same deterministic computation independently without exchanging data.
2. **Stream priorities and staggered publishing**: The TP inference main stream runs at high priority (`-1`) while the training side stream runs at normal priority (`0`). State updates are written before sampling and published at token commit time, avoiding bus contention between side-stream work and All-Reduce.

### Support Boundaries for TP > 1 {#en-support-boundaries-for-tp--1}

| Feature Category | Behavior under TP > 1 | Details and Rationale |
| :--- | :--- | :--- |
| **Hidden states & sampling** | Fully supported | `pllm.hidden`, `pllm.residual`, `pllm.final_hidden`, `@pllm.guide_sampling`, and `@pllm.add_logits` execute deterministically on each rank's local replica. |
| **Request state & concurrent activation patches** | Fully supported | Request-state training such as ESamp and `ActivationPatch` run independently on each rank's local replica and versioned slot table with zero cross-GPU communication overhead. |
| **Custom forward & custom residual** | Supported | `@pllm.custom_forward` and `@pllm.custom_residual` without `pllm.value(...)` or `pllm.input_tokens` dependencies execute across each rank's local shards. |
| **Custom attention (local heads)** | Supported | `@pllm.custom_attention` recomputing directly over each rank's local KV-sharded heads (including DeepSeek V2/V3 latent heads); reading `pllm.attention(...)` facts or static `window` buffers requires `tensor_parallel_size=1`. |
| **Sharded adapter weights** | Supported (dense / low-rank) | Dense and low-rank deltas on `pllm.model.Adapter` (with `Zeros`, `Computed`, `Pretrained`, or `Imported` init) align with projection sharding: `q`/`k`/`v` and `gate`/`up` projections shard weights and factor B along output rows; `down` projections shard weights and factor A along input columns; the other factor is replicated on all ranks and can be updated on local shards via `@pllm.train_in_forward`. |
| **Exports (`@pllm.export`)** | Delivered by rank 0 only | Results are identical across all ranks, so export events are sent back to the CPU from rank 0 only to avoid redundant transfers. |
| **Features requiring TP=1** | Rejected at validation | `operator="per_head"` adapter deltas; adapter states with optimizers, side-stream writes, or `FromModule` / `ModelCheckpoint` / `Fed` init; whole-model training via `pllm.causal_model`; external streaming via `pllm.Fed`; `@pllm.custom_residual` depending on `pllm.value(...)` or `pllm.input_tokens`; `@pllm.custom_attention` using `pllm.attention(...)` or `window` buffers; and `pllm.testing.check_on_gpu` (must be verified at TP=1 first). |

> [!IMPORTANT]
> **TP does not reduce per-GPU VRAM used by request state**: `pllm.explain` notes `reservations are per rank` when reporting memory. Tensor parallelism shards base model weights across GPUs, but pLLM keeps a complete replica of request state on every GPU, so per-rank request-state VRAM does not decrease as you add GPUs.

<details markdown="1">
<summary>Self-check: Why doesn't per-GPU request-state memory drop by half when enabling TP=2?</summary>

Because Tensor Parallelism shards the base model's projection weights, whereas pLLM maintains an independent replica of request state on each TP rank to guarantee zero cross-GPU communication and deterministic execution. As a result, per-GPU request-state memory requirements remain unchanged.

</details>

---

## Quantized Base Models: FP8, AWQ, and GPTQ {#en-quantized-base-models-fp8-awq-and-gptq}

Production deployments commonly serve FP8, AWQ, or GPTQ quantized checkpoints to cut VRAM consumption and improve memory-bandwidth-bound decode throughput.

pLLM supports quantized base models as first-class citizens:
1. **Identical model admission rules**: Quantized checkpoints follow the same model-family admission rules as dense models (e.g., Qwen2/2.5, Llama, Qwen3; see the [Support Matrix](/pllm/docs/reference/support-matrix/)).
2. **LoRA-style deltas never dequantize the base weights**: Weight deltas applied through `pllm.model.Adapter` or `@pllm.train_in_forward` add high-precision (`BF16` or `FP32`) corrections `(x @ A.T) @ B.T` directly onto the output of the quantized GEMM kernel—just like vLLM's native LoRA over quantized models. Base weights stay quantized throughout and are never materialized as dense matrices in VRAM.
3. **On-demand dequantization for whole-model training**: Training through `pllm.causal_model` works similarly to QLoRA: the forward pass uses quantized kernels, while the backward pass temporarily dequantizes the participating weights to compute gradients, balancing VRAM savings with differentiability.
4. **Materializing dense weights**: If `@pllm.train_in_forward` declares a read of `pllm.operand("weight")`, pLLM materializes a dense floating-point copy of the weight in VRAM for that method to use.

---

## Engine Lifecycle and Resource Cleanup {#en-engine-lifecycle-and-resource-cleanup}

In long-running service processes or batch pipelines, you may create and destroy multiple engine instances sequentially within a single Python process. Always manage the engine lifecycle with a context manager:

```python
from vllm import SamplingParams

config = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=8)

with pllm.make_llm(config, workload=StreamMonitor()) as llm:
    llm.generate(["Prompt 1"], SamplingParams(max_tokens=8))

# Exiting the with block automatically calls llm.close()
```

Exiting the `with` block (or calling `llm.close()`) shuts down the engine and releases its compiled state and CUDA graph memory, allowing repeated engine creation and teardown within the same process.

### Environment Requirements for GPU Contract Checks {#en-environment-requirements-for-gpu-contract-checks}

`pllm.testing.check_on_gpu` internally constructs live engines and verifies scheduling contracts including eight-step steady-state decoding, forced KV eviction, and recomputation. Because it requires full control over the underlying engine lifecycle, it must be invoked in a standalone process with:
- `multiprocessing=False`
- `max_num_seqs >= 2`
- `max_model_len >= 32`

```python
testing.check_on_gpu(
    StreamMonitor(),
    pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=4, max_model_len=64),
)
```

See [Checking Correctness](/pllm/docs/guides/checking/) for details.

---

## Memory Knobs and Request Preemption {#en-memory-knobs-and-request-preemption}

In production, single-GPU VRAM is partitioned into four components:
$$\text{Total VRAM} = \text{Base Model Weights} + \text{CUDA Graph Pool} + \text{pLLM State \& Execution Workspace} + \text{vLLM KV Cache Blocks}$$

Model weights, the CUDA graph pool, and pLLM's state reservations claim fixed shares first; all remaining VRAM goes to vLLM's paged KV cache.

### Core Memory Knobs {#en-core-memory-knobs}

Configure memory bounds in `pllm.ModelConfig` and `pllm.ExecutionOptions`:

```python
# Tune execution budgets and scheduling limits
service_config = pllm.ModelConfig(
    "Qwen/Qwen2.5-0.5B-Instruct",
    max_num_seqs=16,
    max_num_batched_tokens=1024,
    gpu_memory_utilization=0.7,
    execution=pllm.ExecutionOptions(
        state_page_budget_bytes=0,
        cuda_graph_pool_budget_bytes=128 * 1024 * 1024,
    ),
)
service_plan = pllm.explain(StreamMonitor(), service_config)
```

- **`gpu_memory_utilization`** (default `0.5`): Fraction of GPU memory allocated to the inference engine. Raising it gives the KV cache more room, though you must guard against GPU OOM under high concurrency or large models.
- **`max_num_seqs`** (default `8`): Maximum number of concurrent sequences processed per step. This directly sets the number of state slots `RequestState` pre-allocates for concurrent requests.
- **`max_num_batched_tokens`** (default `512`): Maximum total tokens batched across concurrent requests in a single step (including both prefill and decode).
- **`state_page_budget_bytes`** (default `0`): Upper bound on device memory for request state pages; `0` uses the free capacity observed by the framework.
- **`execution_peak_budget_bytes`** (default `None`): Upper bound on combined peak memory across state pages, task workspaces, and intermediate outputs.
- **`cuda_graph_pool_budget_bytes`** (default `None`): Upper bound on private CUDA graph pools for tasks and the model. `None` sizes automatically: tensor tasks reserve according to traced needs and model graphs are managed by vLLM; explicit caps are still enforced.
- **`max_host_training_rows`** (default `1`): Maximum rows per state per step fed by host training batches (request-level decode row training is budgeted separately).

For full parameter details across all execution options, see [Configuration](/pllm/docs/reference/config/). Always check peak memory reservations on CPU with `pllm.explain` (see [Inspecting Cost: pllm.explain](/pllm/docs/guides/explain/)) before deploying.

### Request Preemption and Recomputation {#en-request-preemption-and-recomputation}

When a surge in concurrent requests exhausts KV cache blocks, the vLLM scheduler preempts active requests, evicts their KV cache blocks, and recomputes their history once memory frees up.

When a preempted request is rescheduled, vLLM recomputes its history. What this means for your workload is shown in the `preempt` line of `pllm.explain`:

- `state keep`: request state is preserved across preemption and is unaffected by history recomputation.
- `state reset`: state resets to its initial value and is re-derived as history recomputes. When request state is keyed by `request_id` with no mutable instance state, external feeds, or imports, recomputation reproduces the original state and the engine recomputes seamlessly. Otherwise, the `preempt` line states why recomputation is unsafe, and if preemption actually occurs at runtime, the engine raises:
  `vLLM preempted a request and would recompute its history; ... To avoid preemption, give vLLM more KV cache (gpu_memory_utilization) or fewer concurrent sequences (max_num_seqs)`

**Mitigation**: Inspect the `preempt` line before going live. If your workload cannot recompute safely, raise `gpu_memory_utilization` to give the KV cache sufficient headroom or lower `max_num_seqs` so preemption is not triggered.

---

## Boundaries Between Forward Edits and Prefix Caching {#en-boundaries-between-forward-edits-and-prefix-caching}

vLLM supports Prefix Caching (`enable_prefix_caching=True`) to reuse existing KV cache blocks across prompts that share a common prefix, significantly reducing prefill latency on long contexts.

In `ModelConfig`, `enable_prefix_caching` defaults to `False`. Whenever a workload modifies model computation, pLLM determines whether prefix caching is safe based on whether state varies across requests or steps:

### Allowed vs. Mutually Exclusive Boundaries {#en-allowed-vs-mutually-exclusive-boundaries}

- **Safe to enable**: Purely read-only or sampling-guidance workloads, stateless `@pllm.custom_residual`, and `Adapter` or `ActivationPatch(..., last_tokens=0)` reading only static immutable `InstanceState` (`Zeros` / `Computed` / `Pretrained`, without `Imported`/`Fed` or write tasks). In all of these cases, identical prompts produce identical KV tensors across all requests.
- **Must disable (rejected at startup admission)**: Model edits using `@pllm.custom_attention` (including reading `pllm.attention(...)` facts), `@pllm.train_in_forward`, `ActivationPatch(..., last_tokens > 0)`, or reading `RequestState` / mutable or externally written `InstanceState`. These patterns cause identical prefixes to produce different KV tensors across requests or state versions, so reusing cached prefixes would silently corrupt numerical semantics.

---

## Common Errors and Troubleshooting {#en-common-errors-and-troubleshooting}

For exact error messages, root causes, and fixes when patterns on this page are rejected, see [Engine settings, models and parallelism](/pllm/docs/reference/errors/#en-engine-settings-models-and-parallelism) and [Requests and run time](/pllm/docs/reference/errors/#en-requests-and-run-time) in the Error Catalog.

</div>

<div class="zine-lang-panel is-hidden" data-lang-panel="zh" lang="zh-CN" markdown="1">

<div id="zh-sec-1"></div>

在 Notebook 或本地脚本中验证完算法之后，将负载（`pllm.Workload`）推向生产服务通常面临三类工程需求：将推理引擎隔离进独立子进程（`multiprocessing="auto"` 或 `True`）以服务在线流量、在多张 GPU 上开启张量并行（Tensor Parallelism）以承载更大模型，以及在量化底座（FP8、AWQ、GPTQ）上保持高效运行。

本文介绍这三项能力的配置方法、约束条件及其背后的运行机制。

---

## 开启多进程：EngineCore 隔离与序列化要求 {#zh-sec-2}

`ModelConfig` 默认设置 `multiprocessing="auto"`：在本地用 `pllm.make_llm` 做单卡调试时，负载与引擎默认运行在同一个 Python 进程中；当使用 `pllm.make_async_llm`、`pllm serve` 或开启 `tensor_parallel_size > 1` 时，`pLLM` 会自动将执行引擎隔离进独立的 EngineCore 子进程中（也可显式传 `multiprocessing=True` / `False` 覆盖默认策略）。启用多进程时需满足以下序列化条件：

驱动端（driver）通过 `pickle` 将负载对象序列化传递给 EngineCore worker。

```python
from dataclasses import dataclass

import torch
from torch import nn
import pllm
from pllm import testing

CONFIG = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=8)


@dataclass(frozen=True)
class StreamMonitor(pllm.Workload):
    @pllm.export(name="norm", max_rows=8, max_cols=1)
    def norm(self, r: torch.Tensor = pllm.residual(12)) -> torch.Tensor:
        return r.float().norm(dim=-1, keepdim=True)


# 在 CPU 上检查规划和调度契约
report = pllm.explain(StreamMonitor(), CONFIG)
check_result = testing.check(StreamMonitor(), CONFIG)
```

要在生产服务中以多进程模式启动引擎：

```python
from vllm import SamplingParams

from my_project.workloads import StreamMonitor

mp_config = pllm.ModelConfig(
    "Qwen/Qwen2.5-0.5B-Instruct",
    max_num_seqs=8,
    multiprocessing=True,
)

with pllm.make_llm(mp_config, workload=StreamMonitor()) as llm:
    outputs = llm.generate(["Hello world"], SamplingParams(max_tokens=8))
```

在多进程模式下，负载代码必须遵守以下约束：

1. **负载字段必须能 pickle（不放锁、打开的文件或活动句柄）**  
   **原因**：负载用 cloudpickle 传给 worker 子进程：定义在 `__main__` 或 notebook 单元格里的类和函数（包括 lambda 与闭包）按值传过去，可导入模块里的按导入路径传。锁、文件、socket 这类绑定在本进程上的对象无法序列化，编码时会报错。
2. **负载所在模块要能在 worker 里导入**  
   **原因**：按导入路径传过去的类，worker 会重新 `import` 它所在的模块，所以那个模块要存在于 worker 的 Python 路径上。
3. **模块必须具备导入安全性（import-safe）**  
   **原因**：worker 子进程被拉起时，在初始化 CUDA 运行时与 GPU 显存之前就会在顶层 `import` 你的模块。如果在模块顶层执行了分配 GPU 张量或触碰 CUDA 上下文的代码，子进程会在设备上下文就绪前崩溃。
4. **worker 自己规划负载**  
   **原因**：worker 反序列化出负载后，在本进程内完成规划和绑定，驱动端只取回它的计划。按导入路径传过去的模块由 worker 自己 `import`，和任何 Python 多进程程序一样：从 `make_llm` 到 worker 启动之间改动这些模块，后果由作者负责。
5. **模型对象不可作为负载字段**  
   **原因**：基座模型权重通常达数 GB 至数十 GB，且已经由 worker 底层加载并驻留在 GPU 显存中。负载字段不能持有重复的模型权重。如果算法需要在方法内调用基座模型（如因果语言模型训练），应在方法内使用 `pllm.causal_model("adapter")`，它会直接复用 worker 中已加载的基座模型。

<details markdown="1">
<summary>自测：为什么把模型对象存入负载字段会在多进程模式下失败？</summary>

因为基座模型占用大量显存且与特定的 CUDA 设备上下文绑定，无法通过 `pickle` 跨进程安全传输；此外 EngineCore worker 已经拥有加载好的基座模型，重复传递不仅浪费资源，还会引发反序列化错误。方法内应当使用 `pllm.causal_model` 访问已加载的模型。

</details>

---

## 多卡张量并行（TP > 1） {#zh-sec-3}

当单张卡显存不足以容纳整个模型权重，或者需要更高显存带宽时，可以通过 `ModelConfig(tensor_parallel_size=2, ...)` 开启多卡张量并行（Tensor Parallelism）。

在张量并行下，需要 vLLM 0.32.0 及以上：

如果需要长期运行并发请求、逐 token 流式返回或暴露 OpenAI 兼容 HTTP 接口，请参阅 [在线服务指南](/pllm/docs/howto/online-serving/)。

```python
from vllm import SamplingParams

tp_config = pllm.ModelConfig(
    "Qwen/Qwen2.5-0.5B-Instruct",
    max_num_seqs=8,
    tensor_parallel_size=2,
)

with pllm.make_llm(tp_config, workload=StreamMonitor()) as llm:
    outputs = llm.generate(["Tensor parallel run"], SamplingParams(max_tokens=8))
```

### 运行机制：零跨卡集合通信与错峰发布 {#zh-sec-4}

张量并行下，vLLM 在每一层注意力与 MLP 之后都必须进行一次全卡同步（Custom All-Reduce）。如果 pLLM 自身也引入跨卡集合通信，或者旁路流上的训练计算与 vLLM 的 All-Reduce 产生总线带宽竞争，所有卡都会发生严重停顿。

为此，pLLM 采取两项设计：
1. **零 pLLM 跨卡集合通信**：隐藏状态读取（`pllm.hidden`、`pllm.residual`、`pllm.final_hidden`）、采样引导（`@pllm.guide_sampling`）、Logits 调整（`@pllm.add_logits`）和请求级状态在每个 TP rank 上各存一份完全相同的副本，各卡独立执行相同的确定性计算，彼此不交换数据。
2. **流优先级与错峰发布**：将张量并行推理主流设为高优先级（`-1`），旁路训练流设为普通优先级（`0`），并让状态更新在采样前写入、在 token 提交时发布，避免旁路计算与 All-Reduce 撞车。

### TP > 1 的支持边界 {#zh-sec-5}

| 特性分类 | 在 TP > 1 下的表现 | 说明与原因 |
| :--- | :--- | :--- |
| **隐藏状态与采样** | 完全支持 | `pllm.hidden`、`pllm.residual`、`pllm.final_hidden`、`@pllm.guide_sampling`、`@pllm.add_logits` 在各卡本地副本上确定性计算。 |
| **请求状态与并发激活补丁** | 完全支持 | 例如 ESamp 等请求状态训练，以及 `ActivationPatch`，在各卡本地副本与版本槽表上独立运行，无跨卡通信开销。 |
| **自定义前向与残差** | 支持 | `@pllm.custom_forward` 与无 `pllm.value(...)` / `pllm.input_tokens` 依赖的 `@pllm.custom_residual` 在各卡本地分片上推进执行。 |
| **自定义注意力（本地头）** | 支持 | 直接在各卡本地 KV 分片头（含 DeepSeek V2/V3 潜变量头）上重算的 `@pllm.custom_attention`；若读取 `pllm.attention(...)` 事实表或静态 `window` 缓冲则需 `tensor_parallel_size=1`。 |
| **适配器权重分片** | 支持 dense / low-rank | `pllm.model.Adapter` 的 dense 与 low-rank 增量（支持 `Zeros`、`Computed`、`Pretrained`、`Imported` 初始化）与投影分片对齐：q/k/v 和 gate/up 投影按输出行切分权重与因子 B；down 投影按输入列切分权重与因子 A；另一因子各卡复制，可由 `@pllm.train_in_forward` 在本地分片上更新。 |
| **导出（`@pllm.export`）** | 仅由 rank 0 交付 | 所有 rank 上的计算结果完全一致，因此导出事件只由 rank 0 送回 CPU，避免重复传输。 |
| **必须设为 TP=1 的特性** | 直接拒绝报错 | `operator="per_head"` 的适配器增量、带优化器 / 侧链读写或 `FromModule` / `ModelCheckpoint` / `Fed` 初始化的适配器状态、经 `pllm.causal_model` 的整模型训练、外部状态投喂（`pllm.Fed`）、依赖 `pllm.value(...)` / `pllm.input_tokens` 的 `@pllm.custom_residual`、使用 `pllm.attention(...)` 或 `window` 缓冲的 `@pllm.custom_attention`，以及 `pllm.testing.check_on_gpu`（须先在 TP=1 下验证）。 |

> [!IMPORTANT]
> **TP 不会减少单卡上请求状态的显存占用**：`pllm.explain` 在报告显存时会提示 `reservations are per rank`。张量并行平摊的是底座模型的权重显存，但 pLLM 的请求状态在每张卡上均保留完整副本，因此请求状态占用的显存不会随卡数增加而减少。

<details markdown="1">
<summary>自测：开启 TP=2 后，为什么单卡上的请求状态显存没有减半？</summary>

因为张量并行切分的是底座模型的投影层权重，而 pLLM 为了保证零跨卡通信与确定性执行，在每个 TP rank 上都独立维护一份请求状态副本。因此单卡上的请求状态显存需求保持不变。

</details>

---

## 量化底座：FP8、AWQ 与 GPTQ {#zh-sec-6}

在生产部署中，服务通常使用 FP8、AWQ 或 GPTQ 量化权重以削减显存占用并提升访存受限阶段的 decode 吞吐。

pLLM 将量化底座作为一等公民支持：
1. **模型准入规则相同**：量化检查点遵循与稠密模型相同的家族准入规则（如 Qwen2/2.5、Llama、Qwen3 等，详见[支持矩阵](/pllm/docs/reference/support-matrix/)）。
2. **LoRA 类增量不反量化底座**：通过 `pllm.model.Adapter` 或 `@pllm.train_in_forward` 施加的权重增量，就像 vLLM 原生对量化模型应用 LoRA 一样，在量化矩阵乘法算子的输出上直接累加高精度（BF16 或 FP32）的修正项 `(x @ A.T) @ B.T`，底座权重全程保持量化状态，不需要在显存中展开为稠密矩阵。
3. **整模型训练按需反量化**：经 `pllm.causal_model` 训练时，采用类似 QLoRA 的机制：前向使用量化算子计算，反向传播求导时临时对参与计算的权重进行反量化，兼顾显存节省与导数计算。
4. **物化稠密权重**：在 `@pllm.train_in_forward` 中如果声明读取 `pllm.operand("weight")`，pLLM 会在显存中物化出一份稠密的浮点权重副本供方法使用。

---

## 引擎生命周期与资源清理 {#zh-sec-7}

在长期运行的服务进程或批处理流水线中，可能需要在一个 Python 进程内先后创建并销毁多个引擎实例。务必使用上下文管理器管理引擎生命周期：

```python
from vllm import SamplingParams

config = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=8)

with pllm.make_llm(config, workload=StreamMonitor()) as llm:
    llm.generate(["Prompt 1"], SamplingParams(max_tokens=8))

# 离开 with 代码块会自动调用 llm.close()
```

离开 `with`（或调用 `llm.close()`）时，引擎关闭并释放它的编译状态和 CUDA graph 内存，所以同一个进程里可以反复创建、关闭引擎。

### GPU 契约检查的环境要求 {#zh-sec-8}

`pllm.testing.check_on_gpu` 在内部会构建真实引擎并验证八步稳态解码、强制 KV 驱逐与重算等调度契约。由于它需要完全控制底层引擎的生命周期，必须在独立的进程中调用，且要求：
- `multiprocessing=False`；
- `max_num_seqs >= 2`；
- `max_model_len >= 32`。

```python
testing.check_on_gpu(
    StreamMonitor(),
    pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=4, max_model_len=64),
)
```

详见[检查正确性](/pllm/docs/guides/checking/)。

---

## 显存旋钮与请求抢占（Preemption） {#zh-sec-9}

在生产环境中，单卡 GPU 显存主要由四部分构成：
$$\text{总显存} = \text{底座模型权重} + \text{CUDA Graph 内存池} + \text{pLLM 状态与执行工作区} + \text{vLLM KV Cache 块}$$

模型权重、CUDA Graph 内存池与 pLLM 的状态显存优先分走固定份额，剩下的显存全部归 vLLM 的分页 KV Cache 使用。

### 核心显存旋钮 {#zh-sec-10}

在 `pllm.ModelConfig` 与 `pllm.ExecutionOptions` 中配置显存边界：

```python
# 调整执行预算与调度限制
service_config = pllm.ModelConfig(
    "Qwen/Qwen2.5-0.5B-Instruct",
    max_num_seqs=16,
    max_num_batched_tokens=1024,
    gpu_memory_utilization=0.7,
    execution=pllm.ExecutionOptions(
        state_page_budget_bytes=0,
        cuda_graph_pool_budget_bytes=128 * 1024 * 1024,
    ),
)
service_plan = pllm.explain(StreamMonitor(), service_config)
```

- **`gpu_memory_utilization`**（默认 0.5）：分配给推理引擎的显存比例。调高该值可为 KV Cache 提供更多显存，但在大并发或大模型下需防止 GPU OOM。
- **`max_num_seqs`**（默认 8）：单步允许并发处理的序列数上限。它直接决定了 `RequestState` 为并发请求预先分配的槽位数。
- **`max_num_batched_tokens`**（默认 512）：单步内并发请求累计的最大 token 数量（包含 prefill 与 decode）。
- **`state_page_budget_bytes`**（默认 0）：限制请求状态分配的设备页面显存上限；设为 0 表示使用框架观测到的空闲容量。
- **`execution_peak_budget_bytes`**（默认 None）：限制状态页、任务工作区与中间输出的总峰值显存。
- **`cuda_graph_pool_budget_bytes`**（默认 None）：任务与模型的 CUDA Graph 私有内存池上限。None 自动定容：张量任务按追踪需求预留，模型图由 vLLM 管理；显式上限仍会检查。
- **`max_host_training_rows`**（默认 1）：每步每个状态由主机投喂训练的最大行数（请求解码行训练单独统计）。

所有执行选项的详细参数见[配置项](/pllm/docs/reference/config/)。在上线前，务必使用 `pllm.explain`（详见[看成本：pllm.explain](/pllm/docs/guides/explain/)）在 CPU 上核对显存预留峰值。

### 请求抢占与重算（Preemption & Recompute） {#zh-sec-11}

当并发请求激增导致 KV Cache 块耗尽时，vLLM 调度器会主动抢占（preempt）正在生成的请求，驱逐其 KV Cache 块，待显存空闲后再重新计算其历史。

被抢占的请求重新调度时，vLLM 会重算它的历史。这对负载意味着什么，`pllm.explain` 的 `preempt` 行已经写明：

- `state keep`：请求状态在抢占期间保留，重算历史不影响它；
- `state reset`：状态要回到起点、随历史重算重新推导。请求状态按 `request_id` 区分、没有可变的实例状态、没有外部 feed 或导入时，重算能复现原来的状态，引擎直接重算；否则 `preempt` 行写出不能重算的原因，真的发生抢占时报错：
  `vLLM preempted a request and would recompute its history; ... To avoid preemption, give vLLM more KV cache (gpu_memory_utilization) or fewer concurrent sequences (max_num_seqs)`

**防范措施**：上线前看 `preempt` 行；属于不能重算的负载时，提高 `gpu_memory_utilization` 给 KV cache 留足空间，或调小 `max_num_seqs`，避免触发抢占。

---

## 前向改写与 Prefix Caching 的精细边界 {#zh-sec-12}

vLLM 提供了前缀缓存功能（Prefix Caching，`enable_prefix_caching=True`），能够对相同前缀的 prompt 复用已有 KV Cache 块，从而显著降低长上下文 prefill 的计算耗时。

在 `ModelConfig` 中，`enable_prefix_caching` 默认为 `False`。当算法改写模型计算时，`pLLM` 按状态是否随请求或步数变化来判断是否允许开启前缀缓存：

### 允许与互斥边界 {#zh-sec-13}

- **可以安全开启**：纯只读或采样引导负载、无状态的 `@pllm.custom_residual`，以及仅读取静态不可变 `InstanceState`（使用 `Zeros` / `Computed` / `Pretrained`，非 `Imported`/`Fed`、无写入任务）的 `Adapter` 或 `ActivationPatch(..., last_tokens=0)`。因为相同 prompt 在所有请求上算出的 KV 张量完全一致。
- **必须关闭（启动准入拒绝）**：使用 `@pllm.custom_attention`（包含读取 `pllm.attention(...)` 事实）、`@pllm.train_in_forward`、`ActivationPatch(..., last_tokens > 0)`，或读取 `RequestState` / 可变与外部写入 `InstanceState` 的模型改写。这些写法会使相同前缀在不同请求或不同版本下得到不同的 KV 张量，复用缓存会导致数值语义错误。

---

## 常见报错与排查 {#zh-sec-14}

这一页的写法被拒绝时的报错原文、原因和改法，见报错目录的[引擎设置、模型与并行](/pllm/docs/reference/errors/#zh-sec-10)、[请求与运行时](/pllm/docs/reference/errors/#zh-sec-12)几节。

</div>

</div>

<div class="zine-doc-footer">
  <a class="zine-pill-link" data-i18n-zh="← 返回 pLLM 项目与文档中心" data-i18n-en="← Back to pLLM Project &amp; Docs" href="/pllm/#docs">← Back to pLLM Project &amp; Docs</a>
  <a class="zine-pill-link" href="https://github.com/LinesHogan/pLLM/blob/main/doc/howto/production.md" target="_blank" rel="noopener">GitHub &middot; doc/howto/production.md ↗</a>
</div>

<script defer src="/assets/js/bilingual-post.js"></script>
