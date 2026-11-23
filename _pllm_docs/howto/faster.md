---
layout: page
title: "Making Workloads Faster"
description: "pLLM Documentation · How-To & Serving Guides"
permalink: /pllm/docs/howto/faster/
nav: false
---

<link rel="stylesheet" href="{{ '/assets/css/pllm-hub.css' | relative_url | bust_file_cache }}">

<div class="zine-dispatch" data-title-zh="让负载更快" data-title-en="Making Workloads Faster" data-desc-zh="pLLM 文档 · 操作与上线指南" data-desc-en="pLLM Documentation · How-To &amp; Serving Guides">
  <div class="zine-dispatch-links">
    <a class="zine-pill-link primary" data-i18n-zh="← 返回 pLLM 文档中心" data-i18n-en="← pLLM Docs Hub" href="/pllm/#docs">← pLLM Docs Hub</a>
    <a class="zine-pill-link" data-i18n-zh="文档导航地图" data-i18n-en="Docs Roadmap" href="/pllm/docs/">Docs Roadmap</a>
    <span class="zine-lang-badge" data-i18n-zh="操作与上线指南" data-i18n-en="How-To &amp; Serving Guides">How-To &amp; Serving Guides</span>
  </div>
  <a class="zine-pill-link" data-i18n-zh="GitHub 源码 ↗" data-i18n-en="View on GitHub ↗" href="https://github.com/LinesHogan/pLLM/blob/main/doc/howto/faster.md" target="_blank" rel="noopener">View on GitHub ↗</a>
</div>

<div id="markdown-content">

<div class="zine-lang-panel" data-lang-panel="en" lang="en" markdown="1">

<div id="en-making-workloads-faster"></div>

Your algorithm workload runs end-to-end and produces correct results, yet autoregressive decode throughput is lower than expected—or the engine fails with out-of-memory errors at startup.

This how-to guide is a prioritized optimization checklist. Unlike offline training, autoregressive decode generates only 1 token per concurrent request per step: arithmetic intensity (FLOPs/byte) is extremely low, compute is usually abundant, and the true bottlenecks come down to three physical constraints: **GPU timeline bubbles (host synchronization or kernel launch stalls)**, **memory and bus bandwidth (full-vocabulary projections, state reads/writes, PCIe transfers)**, and **KV Cache VRAM capacity**.

For the underlying derivations and hardware models behind these constraints, see [Performance and Memory](/pllm/docs/explain/performance/); for definitions of the fields in the execution report, see [Reading Costs: pllm.explain](/pllm/docs/guides/explain/); and for how to benchmark fairly after making changes, see [Measuring Throughput Fairly](/pllm/docs/howto/measure-throughput/).

> [!NOTE] Benchmark Environment and Reference Numbers
> Unless stated otherwise, all measured numbers in this guide come from the same baseline environment: **single RTX 4090 (24 GiB), `Qwen/Qwen2.5-7B-Instruct` (BF16), vLLM 0.32.0, 128 concurrent sequences (8 prompts $$\times$$ 16 sampling branches), 256 generated tokens per sequence, CUDA Graphs enabled; median of 3 rounds after 1 warmup round via `pllm.benchmark.throughput`**. Throughput percentages are relative to native vLLM under the exact same settings with CUDA Graphs enabled (~$$5,894\text{ tok/s}$$, or ~$$21.7\text{ ms}$$ per step across 128 rows). With the same settings on a 0.5B model, one step takes only ~$$3.0\text{ ms}$$; below, both are contrasted wherever conclusions depend on model size.

```python
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

import pllm
from pllm import testing
```

---

## 1. Confirm the Forward Pass Is Still Inside a CUDA Graph {#en-sec-1}

### What to Look For {#en-what-to-look-for}

Run `pllm.explain(workload, config)` and inspect the `forward` line:

```text
forward  CUDA graphs: no effect needs eager
```

If it falls back to eager mode, this line displays `forward eager: <reason>`. After launching the engine, you can also inspect `llm.diagnostics().eager_reason` at runtime (an empty string `""` in graph mode; the concrete reason when falling back to eager).

### What It Means {#en-what-it-means}

Falling back to eager mode forces Python to launch hundreds of operators individually on every step. Whether that CPU overhead is fatal depends on how long a single GPU step takes: at 128 concurrent sequences on 7B, a step takes ~21.7 ms—mostly reading weights—so **falling back to eager still retains 98% of graph-mode throughput** ($$5,773$$ vs. $$5,894\text{ tok/s}$$). On 0.5B under the same settings, a step is only ~3 ms, so **eager drops to 49% of graph-mode throughput** ($$20,926$$ vs. $$42,355\text{ tok/s}$$). The smaller the model, the lower the concurrency, and the shorter the step, the more critical CUDA Graphs become; graph mode is also what lets pLLM capture training and model edits into the same graph.

### How to Fix It {#en-how-to-fix-it}

Verify that your workload preserves the whole-model CUDA Graph:
1. **Both per-request and shared adapters stay in-graph**: Whether `pllm.model.Adapter` is bound to `pllm.RequestState` (one state page per request) or `pllm.InstanceState` (shared or admission-version-isolated state slots), deltas are computed directly inside the CUDA Graph via versioned page tables without breaking whole-model capture.
2. **Activation patches and sliced graphs**: `pllm.model.ActivationPatch` also stays inside the CUDA Graph. What replaces the whole-model graph is `@pllm.custom_forward` (which switches to per-layer-slice captured graphs, with the reason recorded in `eager_reason`) or uncaptured/non-traceable operators.
3. **Fail fast via config**: Set `ModelConfig("Qwen/Qwen2.5-7B-Instruct", cuda_graphs=True)`. With the default `cuda_graphs="auto"`, constructs that cannot enter the whole-model graph fall back to eager or sliced mode (with the reason printed in `forward` and `eager_reason`). Setting it explicitly to `True` makes any declaration that breaks whole-model capture raise `ValueError` immediately in `explain` or `make_llm`, preventing silent throughput regressions.

```python
@dataclass(frozen=True)
class FastLoRA(pllm.Workload):
    # Use RequestState: each request holds its own state copy, and the adapter stays fully inside the CUDA Graph
    lora = pllm.RequestState(
        {
            "a": pllm.Field(lambda in_features, out_features: (4, in_features)),
            "b": pllm.Field(lambda in_features, out_features: (out_features, 4)),
        },
        init=pllm.Imported(),
    )
    apply = pllm.model.Adapter(
        "lora",
        (pllm.model.Projection("q_proj", 12), pllm.model.Projection("v_proj", 12)),
        operator="low_rank",
    )


# Enable cuda_graphs=True: any pattern that triggers eager fallback fails immediately at build time.
# With 128 requests each carrying a LoRA, the modified model's CUDA graph needs a larger capture pool
# (if insufficient, startup fails with an error stating the required byte budget).
config = pllm.ModelConfig(
    "Qwen/Qwen2.5-7B-Instruct",
    max_model_len=512,
    max_num_seqs=128,
    max_num_batched_tokens=8192,
    cuda_graphs=True,
    execution=pllm.ExecutionOptions(cuda_graph_pool_budget_bytes=512 << 20),
)
report = pllm.explain(FastLoRA(), config)
assert report.eager_reason == ""
```

### What It Costs {#en-what-it-costs}

`RequestState` reserves versioned pages per request slot for independent state (for example, on 7B at 128 concurrency, `explain` reports 256 pages per field, totaling ~22 MiB). If your algorithm only needs cross-request shared state (or state advancing via online `import_state` versions), you can place the `Adapter` on `InstanceState` instead and still stay inside the CUDA Graph (reserving `2 + max_num_seqs = 130` pages when in-flight updates are supported, or just 1 page for static read-only state). Both adapter kinds increase the modified model's CUDA Graph capture pool proportionally (startup raises an error naming the exact byte budget if it is too small). Using `@pllm.custom_forward` or falling back to eager costs only ~2% on 7B, whereas eager halves throughput on 0.5B; setting `cuda_graphs=True` adds zero runtime overhead.

---

## 2. Eliminate Host Synchronization in Workload Math {#en-sec-2}

### What to Look For {#en-what-to-look-for-1}

Inspect the `host` line in `pllm.explain` output:

```text
host     0 syncs in traced workload math
```

This line must read `0 syncs`. For in-layer model rewrites (`@pllm.custom_residual`, `@pllm.custom_attention`) and custom operators, CPU tracing reports `unverified syncs`; verify those with `pllm.testing.check_on_gpu(workload, config)`.

### What It Means {#en-what-it-means-1}

pLLM relies on the CPU running ahead of the GPU to prepare metadata asynchronously. Calling `.item()`, `.tolist()`, `.cpu()`, `print(tensor)`, `float(tensor)`, or a Python value branch `if tensor.any():` inside a workload method forces the CPU to block until the GPU finishes all previously launched kernels and copies the scalar back to host memory. That host sync completely drains the GPU execution pipeline, causing stalls of hundreds of microseconds to milliseconds.

### How to Fix It {#en-how-to-fix-it-1}

Rewrite scalar extraction and dynamic control flow as pure tensor math:
1. **Replace branches**: Convert Python `if condition:` into `torch.where` or boolean mask multiplication.
2. **Forbid scalar conversions**: Never call `.item()` or print tensors inside workload methods.
3. **Asynchronous observation and health diagnostics**: For debugging values or collecting statistics, use `@pllm.export` (which writes into a bounded pinned ring buffer and drops events on overflow by default without blocking the GPU or CPU). At runtime, check `llm.diagnostics().numeric_health` for NaN/Inf in state tensors—this check incurs 0 per-step syncs during steady-state decode.
4. **GPU contract check**: Run `pllm.testing.check_on_gpu(workload, config)`, which executes at least 8 steady-state decode steps under `torch.cuda.set_sync_debug_mode("error")` to catch any implicit in-layer synchronizations.

```python
@dataclass(frozen=True)
class ZeroSyncWorkload(pllm.Workload):
    # Replace Python if branches with torch.where and avoid .item() to guarantee zero host syncs
    @pllm.add_logits()
    def steer(self, logits: torch.Tensor = pllm.layer_logits((-1,))) -> torch.Tensor:
        last = logits[:, -1]  # [rows, vocab]: the model's own logits
        return torch.where(last > 0.0, 0.1 * last, torch.zeros_like(last))  # amplify positive logits only


sync_report = pllm.explain(ZeroSyncWorkload(), config)
assert sync_report.workload_host_syncs == 0
```

### What It Costs {#en-what-it-costs-1}

Tensor branching (such as `torch.where`) evaluates both branches mathematically, but given the abundant compute available during decode, that extra arithmetic is negligible compared to a millisecond-scale pipeline drain.

---

## 3. Ensure Side-Stream Tasks Overlap with the Forward Pass {#en-sec-3}

### What to Look For {#en-what-to-look-for-2}

1. Check the `lane` column for each task in `pllm.explain`'s `member` list: verify whether it shows `side` (side CUDA stream) or has fallen back to `main` (main stream).
2. Check whether the `timing` line shows `publish_lag=1 (default)`.
3. Check the two thresholds printed on the `memory` line:

```text
memory   peak 1512728864 bytes funded by pLLM [per-rank tensor reservations; excludes GPU-dependent BLAS workspace; execution_peak_budget_bytes]
         below 1512728864 bytes tasks run inline; below 1333743904 bytes planning fails
```

Check whether your configured `execution_peak_budget_bytes` is below the overlap threshold on the first line (the example above is `Distill` from [Workload Anatomy](/pllm/docs/guides/workload-anatomy/) on 7B at 128 concurrency: `1512728864 bytes`).

### What It Means {#en-what-it-means-2}

Heavier compute such as online training (`@pllm.train`) is scheduled by default on a side stream (`side`) for asynchronous execution:
* **Default `publish_lag=1`**: After step $$n$$ produces a token, the side stream asynchronously runs backpropagation and the optimizer update, which become visible at step $$n+2$$. Step $$n+1$$'s forward reads the state version from step $$n-1$$ without waiting for step $$n$$'s training, overlapping training with the forward pass.
* **Memory budget downgrade**: Side-stream overlap requires holding workspace buffers for two versions simultaneously, raising peak memory above serial execution. If your `ExecutionOptions(execution_peak_budget_bytes=...)` is below the overlap threshold (yet above the planning failure floor), the planner does not error out—it silently downgrades side-stream tasks to inline execution on the main stream (`inline`, changing `lane` from `side` to `main`), serializing them with the model forward and losing overlap.

### How to Fix It {#en-how-to-fix-it-2}

1. **Keep the lag**: Keep the default `publish_lag=1` rather than setting `publish_lag=0` unnecessarily (unless your algorithm mathematically requires the very next step to see the current step's parameter updates immediately).
2. **Raise the memory budget**: If `lane` fell back to `main`, check the `memory` line in `explain` and raise `execution_peak_budget_bytes` above the overlap threshold to restore side-stream overlap.

```python
exec_opts = pllm.ExecutionOptions(
    execution_peak_budget_bytes=1024 << 20,  # above overlap_threshold to preserve side-stream overlap
    publish_lag=1,  # default 1: step n updates take effect at step n+2, overlapping training with step n+1 forward
)
overlap_config = pllm.ModelConfig(
    "Qwen/Qwen2.5-7B-Instruct",
    max_num_seqs=128,
    max_num_batched_tokens=8192,
    execution=exec_opts,
)
```

### What It Costs {#en-what-it-costs-2}

How much overlap saves depends on how large training is relative to the whole step (`ESamp`, rank 4):
* **7B**: For pure online training ($$\beta=0$$), `publish_lag=1` achieves **97.4%** of native vLLM throughput vs. **97.2%** for `publish_lag=0`; with training + guided sampling ($$\beta=0.1$$), it is **96.4%** vs. **96.1%**. Each training step takes only ~0.6 ms of GPU time—against a 21.7 ms step, the cost of serial waiting is within measurement noise.
* **0.5B (same settings, only changing the model)**: For pure online training, `publish_lag=1` reaches **91.9%** vs. **89.8%** for `publish_lag=0`. Shorter steps make overlap significantly more valuable.

The cost is extra memory reservation, incurred only on states read at a lagged version: when a method reads a version lagged by `publish_lag` steps, each request slot must retain `publish_lag` extra old parameter versions (optimizer fields are not retained for old versions; see the formula in [Performance and Memory, Section 3.1](/pllm/docs/explain/performance/)). For `Distill` on 7B (`predict` reads the lagged version) at 128 concurrency, the parameter field grows from 256 to 384 pages and total state reservation rises from ~95 MiB to ~110 MiB; `ESamp` ($$\beta=0$$) has no lagged reader, so both settings use 256 pages per field.

---

## 4. Cut Bytes Moved per Step {#en-sec-4}

### What to Look For {#en-what-to-look-for-3}

Inspect the task estimate lines and state lines in `pllm.explain`:
* `projection FLOP`: whether extra full-vocabulary LM Head projections are being run.
* `sampler ... bytes`: bytes read and written for full-vocabulary tensors.
* `state reads (origin, distance)`: bytes read from state.
* `logical transfer bytes/row` and `decode bytes/step` on `state` lines: per-step transfer volume for host lookup tables over PCIe.

### What It Means {#en-what-it-means-3}

Decode is bound by VRAM and bus bandwidth. Every extra byte moved per step adds directly to latency:
1. **LM Head projections and full-vocabulary tensors**: The 7B LM Head weight (3584 $$\times$$ 152064 BF16) weighs 1.09 GB. Each extra full-vocabulary projection costs **~$$+1.35\text{ ms}$$** (~6% of a step; on 0.5B, the 272 MB weight takes ~0.36 ms, which is 12% of a step). Each pass in the sampler scanning a $$128 \times 152064$$ vocabulary tensor takes 78–283 μs.
2. **Large table transfers over PCIe**: On an RTX 4090, PCIe 4.0 x16 reads from host memory at ~23.7 GB/s—roughly 1/40th of GPU VRAM bandwidth. Gathering 38.9 MB of wide vectors across 128 rows takes a full **1.6 ms** (~7% of a 7B step; over half a step on 0.5B).

### How to Fix It {#en-how-to-fix-it-3}

#### 1) Merge Projections and Optimize Guided Sampling {#en-sec-5}

* **Use `@pllm.guide_sampling`**: Guided sampling returns a `[rows, hidden]` hidden-state prediction vector; by default the framework uses a bounded fused path (`guidance_capacity=4096`) that projects and reweights only the surviving top-K tokens, avoiding a separate full-vocabulary GEMM.
* **Avoid multi-layer `pllm.layer_logits`**: Every `layer_logits` entry triggers an independent full-vocabulary GEMM, re-reading the LM Head weights (1.09 GB on 7B). On 7B (32 concurrent requests), 9-layer DoLa drops to **66%** of native throughput; on smaller models, the penalty is even larger.
* **Filter by rows**: For non-global interventions, add a `rows=` filter to your effect method (e.g., `rows="trained"`) to shrink the number of active rows computed.

```python
@dataclass(frozen=True)
class FastGuidance(pllm.Workload):
    beta: float = 0.5

    def strength(self) -> float:
        return self.beta

    # Prefer @pllm.guide_sampling: returns [rows, hidden] and fuses deeply with the sampler
    @pllm.guide_sampling(beta=strength)
    def steer(self, h: torch.Tensor = pllm.final_hidden()) -> torch.Tensor:
        return h


guidance_report = pllm.explain(FastGuidance(), config)
```

#### 2) Use `placement="host"` with Clean Method Splitting for Huge Read-Only Tables {#en-sec-6}

For read-only lookup tables ranging from hundreds of MiB to several GiB, declare `pllm.Pretrained(..., placement="host")` to keep them in pinned host memory. To hide the ~1.6 ms PCIe transfer behind the forward pass, follow this split pattern:

* **Dedicated lookup method**: The lookup method must read only `pllm.input_tokens` and read-only state (with no effect decorator), passing the gathered result to downstream effect methods via `pllm.result("lookup")`.
* **Why this split is mandatory**: Only an independent method depending purely on tokens and read-only tables can be scheduled by the planner at `request.prepare / side` at the start of the step, using a bounded number of SMs (capped at 32 thread blocks on RTX 4090, i.e. 1/4 of SMs) to prefetch on the side stream in parallel with the backbone forward. If the same method also reads `pllm.hidden` or `pllm.final_hidden`, it must wait until the model forward finishes and run at `state.publish / main`, exposing the full 1.6 ms transfer latency on the timeline.

```python
@dataclass(frozen=True)
class FastHostLookup(pllm.Workload):
    buckets: int = 2048

    @property
    def memory(self) -> pllm.InstanceState:
        # One vocab-wide logits prior row per bucket; ~594 MiB on 7B (2048 x 152064 x bf16), placed in pinned host memory
        return pllm.InstanceState({"table": lambda m: (self.buckets, m.vocab_size)},
                                  init=pllm.Pretrained("ngram_table.safetensors", placement="host"))

    # Dedicated lookup method: reads only token history and the read-only table, no effect decorator;
    # the planner schedules it on request.prepare / side
    def lookup(self, ids: torch.Tensor = pllm.input_tokens(history=1),
               table: torch.Tensor = pllm.parameter("memory", "table")) -> torch.Tensor:
        keys = (ids[:, 0] * 31 + ids[:, 1] + 1) % self.buckets
        return table[keys]

    # Effect method consumes the gathered result via pllm.result
    @pllm.add_logits()
    def prior(self, logits: torch.Tensor = pllm.result("lookup")) -> torch.Tensor:
        return logits
```

### What It Costs {#en-what-it-costs-3}

In RTX 4090 benchmarks (7B, 128 concurrency, 594 MiB table, `NGramPrior` from Example 11):
* Native vLLM: 5,877 tok/s (~21.8 ms/step);
* Device-resident table (`placement="device"`): 5,742 tok/s (98%), `explain` peak pLLM memory 1,540 MiB;
* Host-resident table (`placement="host"`): 5,615 tok/s (96%), peak pLLM memory drops to **946 MiB** (only ~0.5 ms slower per step than device placement, saving the entire 594 MiB of VRAM);
* Guided sampling capacity (`ESamp`, $$\beta=0.5$$): default `guidance_capacity=4096` achieves 5,678 tok/s (96.3%), with 256 and 1024 within noise of it; exact dense guidance (`guidance_capacity=None`) drops to 5,533 tok/s (93.9%).

<details markdown="1">
<summary>Self-check: Why did a host lookup add 1–2 ms per step after also reading final_hidden in the same method?</summary>

Because the same method read <code>final_hidden</code>, it had to wait until the entire model forward completed (anchored at <code>state.publish / main</code>) instead of launching at <code>request.prepare / side</code> at the start of the step. As a result, the ~1.6 ms PCIe transfer was completely exposed outside the forward pass. The fix is to isolate the lookup—which depends only on <code>input_tokens</code>—into an undecorated helper method and pass its output to the effect method via <code>pllm.result</code>.

</details>

---

## 5. Optimize KV Memory: Static Layouts vs. Dynamic Reads {#en-sec-7}

### What to Look For {#en-what-to-look-for-4}

1. Check the `kv_capacity` report in `pllm.explain` (which shows per-layer `KV bytes/request` and total capacity).
2. Inspect `llm.diagnostics().kv_cache_blocks` and `kv_cache_max_concurrency` at runtime.

### What It Means {#en-what-it-means-4}

In vLLM, memory is allocated in order: model weights + CUDA Graph capture pools + workload state reservations $$\rightarrow$$ **whatever remains goes to the KV Cache**. A smaller KV Cache lowers the maximum number of concurrent requests.

The two mechanisms in `@pllm.custom_attention` behave fundamentally differently:
1. **Static geometric layouts (`keep="window"`, `sink=True`, `keep="none", source=k`)**: Map directly onto vLLM's physical block manager. Physical blocks that slide out of the window are genuinely freed back to the memory pool, **actually saving KV VRAM capacity**, while running through native FlashAttention-2 / FlashInfer kernels to genuinely cut read bandwidth.
2. **Dynamic sparse reads (`kv.attend(q, topk=...)`)**: Every historical token's KV must still remain in VRAM for per-step scoring and selection, **saving zero KV VRAM capacity**; it reduces only logical attention reads. If scoring and gathering are written in plain PyTorch ops, kernel launch and fragmentation overhead far outweigh the saved reads. Score-based physical KV eviction (such as H2O or SnapKV) is unsupported because vLLM's physical block tables require sequential append-only allocation.

### How to Fix It {#en-how-to-fix-it-4}

When your goal is reducing VRAM footprint and raising maximum concurrency, use static layouts:
* **Local sliding window**: `@pllm.custom_attention(layers=tuple(range(0, 28, 2)), keep="window", window=256)` (7B has 28 layers);
* **Cross-layer KV sharing**: `@pllm.custom_attention(layers=tuple(range(1, 28, 2)), keep="none", source=1)`.

```python
@dataclass(frozen=True)
class LocalGlobalAttention(pllm.Workload):
    # Static sliding window: frees physical blocks sliding out of the window, genuinely saving KV capacity
    @pllm.custom_attention(layers=tuple(range(0, 28, 2)), keep="window", window=256)
    def read(self, q: torch.Tensor, kv: pllm.model.PagedKV) -> torch.Tensor:
        return kv.attend(q)[0]


attn_report = pllm.explain(LocalGlobalAttention(), config)
```

### What It Costs {#en-what-it-costs-4}

In benchmarks from [Example 10](/pllm/docs/examples/10_custom_attention/) (RTX 4090, 7B, `max_model_len=2048`, 128-token chunked prefill; KV caps from `explain`):
* **Per-request KV VRAM cap**: Full KV is 112.0 MiB; even-layer window 256 drops to **70.4 MiB**; odd-layer cross-layer sharing drops to **56.0 MiB** (doubling the number of requests that fit in the same KV pool).
* **Throughput**: At 128 concurrency and ~270 tokens per sequence, passthrough reads, even-layer windows, and cross-layer sharing all run at 99% of native vLLM (setup in Example 10).
* **Dynamic sparse read comparison**: `QuestRead` in the [Custom Attention Tutorial](/pllm/docs/examples/10_custom_attention/) (pure PyTorch) still consumes 112.0 MiB of KV VRAM and throughput plummets to **13%** of native (649 tok/s), because unfused PyTorch scoring and gathering cost far more than the saved reads.

---

## 6. Control Precision Finely: Balance Bandwidth and Numerical Stability {#en-sec-8}

### What to Look For {#en-what-to-look-for-5}

1. Check the resolved `dtype` for each field on `state` lines in `pllm.explain` (e.g., `field 0: bfloat16, field 1: float32`).
2. Check `llm.diagnostics().numeric_health` at runtime.

### What It Means {#en-what-it-means-5}

During bandwidth-bound autoregressive decode, state tensor precision directly sets memory traffic. BF16 halves tensor size compared to FP32, halving both read/write bandwidth and state-pool VRAM usage. However, across multi-step autoregressive recurrences, using low precision for sensitive accumulators introduces severe rounding drift, causing divergence or NaNs.

### How to Fix It {#en-how-to-fix-it-5}

1. **Keep `dtype="auto"` by default**: For read-only weights, single-step activations, or state fields not accumulated across steps, `dtype="auto"` aligns with the base model's dtype (BF16/FP16), cutting state bandwidth by 50%.
2. **Explicitly specify `torch.float32`** for:
   * Cross-step accumulators and optimizer first/second moments;
   * Recurrent carried state fields (`carry=`);
   * Coefficients and state transition matrices in linear prefix scans (`pllm.scan`);
   * Syntax: `pllm.Field(shape_fn, dtype=torch.float32)`.

```python
@dataclass(frozen=True)
class PrecisionAwareWorkload(pllm.Workload):
    state = pllm.RequestState(
        {
            "momentum": pllm.Field((896,), dtype=torch.float32),  # explicit FP32: avoid cumulative rounding error
            "cached_feature": pllm.Field((896,)),  # default auto: follows base model BF16
        }
    )

    @pllm.export(name="momentum_norm", max_rows=8, max_cols=1)
    def report(self, m: torch.Tensor = pllm.parameter("state", "momentum")) -> torch.Tensor:
        return m.norm(dim=-1, keepdim=True)


precision_report = pllm.explain(PrecisionAwareWorkload(), config)
```

### What It Costs {#en-what-it-costs-5}

Explicit FP32 doubles physical VRAM and memory traffic for that field, but on small state fields this is the necessary cost for numerical stability and avoiding divergence on long sequences.

---

## 7. When Operators Become the Bottleneck: Plug In Custom Kernels {#en-sec-9}

### What to Look For {#en-what-to-look-for-6}

1. Check the `custom_ops` report in `pllm.explain` (which shows operator names, `flops_per_row`, and CPU reference implementations).
2. Inspect profiler traces to see if the timeline is packed with tiny fragmented PyTorch kernels.

### What It Means {#en-what-it-means-6}

Writing multi-step tensor logic in plain PyTorch inside model layers (`@pllm.custom_residual`) or attention (`@pllm.custom_attention`) incurs kernel launch overhead and repeated VRAM round-trips on every call. In the [Custom Attention Tutorial](/pllm/docs/examples/10_custom_attention/), Quest's per-page scoring and gather written in PyTorch runs at only **13%** of native throughput on 7B.

### How to Fix It {#en-how-to-fix-it-6}

Implement compute-intensive or memory-fragmented logic as fused Triton or CUDA kernels registered through PyTorch's standard mechanism:
1. Register a named operator with `@torch.library.custom_op` and provide static shape inference via `register_fake`.
2. In your workload methods, **call the operator function object directly** (e.g., `fused_step(x, v)`) rather than `torch.ops.<namespace>.<op_name>`. Direct calls let the planner statically analyze the source and capture the operator straight into the CUDA Graph. See [Writing Custom Operators](/pllm/docs/guides/custom-ops/) for detailed authoring and verification steps.

```python
@torch.library.custom_op("faster_guide::fused_step", mutates_args=(), device_types="cuda")
def fused_step(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return x + v


@fused_step.register_fake
def _fused_step_fake(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)
```

### What It Costs {#en-what-it-costs-6}

In RTX 4090 benchmarks:
* Example 09 attaches a Triton kernel registered as a `custom_op` at layer 12 and achieves **98%** of native vLLM throughput ($$5,760$$ vs. $$5,872\text{ tok/s}$$);
* The equivalent PyTorch math also achieves **98%** ($$5,755\text{ tok/s}$$): a single elementwise add is not a bottleneck in the first place;
* Writing a custom op requires authoring and maintaining a Triton/CUDA kernel and fake shape function. It pays off on steps that are genuinely slow in PyTorch—like Example 10's PyTorch Quest page scoring, which drops to 13% of native.

---

## 8. Rewrite Residuals Only on the Layers That Need It {#en-sec-10}

### What to Look For {#en-what-to-look-for-7}

The `forward` line in `pllm.explain` and the number of layers carrying `@pllm.custom_residual` or `ActivationPatch`.

### What It Means {#en-what-it-means-7}

In a model graph carrying residual rewrites, vLLM captures the model's un-fused original operators rather than Inductor-fused operators, so even an identity rewrite pays that cost (~3% on 7B at 128 concurrency; see [Performance and Memory, Section 4.4](/pllm/docs/explain/performance/#en-sec-14)), and the rewrite's own compute accumulates per attached layer.

### How to Fix It {#en-how-to-fix-it-7}

Use `layers=` to attach only to the layers that actually need rewriting. To measure the cost of the rewrite itself, compare against native vLLM in eager mode or run `testing.check_on_gpu` (which builds a matching native engine). Stateful sublayers also require a larger CUDA Graph capture pool—increase `ExecutionOptions(cuda_graph_pool_budget_bytes=...)` to the byte budget reported in the error message.

---

## 9. Make Layer Skipping Actually Save Time {#en-sec-11}

### What to Look For {#en-what-to-look-for-8}

The skip rate of `layers(..., active=mask)` inside `@pllm.custom_forward(tokens="decode")`, and whether throughput scales with it.

### What It Means {#en-what-it-means-8}

Decode GEMMs are weight-bandwidth bound: as long as even one row in the batch runs a layer, that layer's entire weight matrix must be read once. Partial-row skipping saves only the attention reads for the skipped rows; you only save weight-read time when the entire batch skips a block of layers together (for the mechanism and measurements, see [Performance and Memory, Section 4.5](/pllm/docs/explain/performance/)).

### How to Fix It {#en-how-to-fix-it-8}

Design skip predicates so the whole batch skips together more often (for example, fixed layer blocks or step-level rather than row-level thresholds), or apply layer skipping in long-context regimes where attention reads dominate. Prompt tokens never skip layers; `active=` applies only to `tokens="decode"`.

---

## 10. Sampler: Enable FlashInfer Sampling {#en-sec-12}

At batch sizes of 128+ rows or when a workload modifies logits, vLLM's default sampler can take noticeable time. When FlashInfer is installed, set the environment variable below before launch to switch to its fused sampling kernel:

```bash
export VLLM_USE_FLASHINFER_SAMPLER=1
```

Because this changes sampling time on both the native baseline and the workload engine, always measure both under the identical setting and record it in your report (see [Measuring Throughput Fairly](/pllm/docs/howto/measure-throughput/)).

---

## Summary: Performance Optimization Cheat Sheet {#en-summary-performance-optimization-cheat-sheet}

| Check | Metric (`explain` or Runtime) | Target State | Optimization |
| :--- | :--- | :--- | :--- |
| **1. Graph Mode** | `forward` line / `eager_reason` | `CUDA graphs: no effect needs eager` / `""` | Move mutable shared state to `RequestState`; set `cuda_graphs=True` to catch regressions |
| **2. Host Syncs** | `host` line | `0 syncs in traced workload math` | Replace Python `if` with `torch.where`; remove `.item()`; observe asynchronously with `@pllm.export` |
| **3. Side-Stream Overlap** | `member.lane` / `memory` thresholds | `side` / budget above `overlap_threshold` | Keep `publish_lag=1`; raise `execution_peak_budget_bytes` to preserve overlap |
| **4. Bytes Moved** | `projection FLOP` / `sampler bytes` | Minimal full-vocab GEMMs / high lookup overlap | Replace multi-layer `layer_logits` with `@pllm.guide_sampling`; split host tables into a step-start helper |
| **5. KV Memory** | `kv_capacity` / `kv_cache_blocks` | Real VRAM capacity freed | Use static layouts (sliding window or cross-layer sharing); remember dynamic Top-K saves no KV capacity |
| **6. State Precision** | Per-field `dtype` on `state` lines | `auto` -> BF16 for read-only; explicit FP32 for sensitive state | Declare `torch.float32` on accumulators, momentum, and carry fields; keep `auto` elsewhere |
| **7. Kernel Fusion** | `custom_ops` report / small ops in profiler | Fused in-graph, throughput $$\ge 95\%$$ | Register fused Triton kernels via `@torch.library.custom_op` and call the op object directly |
| **8. Residual Rewrites** | `forward` line / attached layer count | Only required layers attached | Restrict with `layers=`; compare against eager native or `check_on_gpu` to isolate rewrite cost |
| **9. Layer Skipping** | Skip rate vs. throughput | Whole batch skips together | Decide skipping by layer block or step; use `active=` only on decode |
| **10. Sampler** | Sampler time | Fused sampling | Set `VLLM_USE_FLASHINFER_SAMPLER=1` identically on both arms |

</div>

<div class="zine-lang-panel is-hidden" data-lang-panel="zh" lang="zh-CN" markdown="1">

<div id="zh-sec-1"></div>

你的算法负载已经跑通、结果正确，但发现自回归解码（decode）的端到端吞吐低于预期，或者服务刚启动就报显存不足。

这篇操作指南是一份按排查优先级排序的优化清单。自回归解码与离线训练不同：GPU 每步只为每个并发请求生成 1 个 token，计算强度（FLOPs/字节）极低，算力通常过剩，真正的瓶颈集中在三个物理制约：**GPU 时间线气泡（主机同步或算子发射阻塞）**、**显存与总线搬运带宽（全词表投影、状态读写、PCIe 传输）**，以及 **KV Cache 显存容量**。

关于这些物理制约的底层推导与硬件模型，参见 [性能与显存](/pllm/docs/explain/performance/)；阅读执行报告的字段定义，参见 [看成本：pllm.explain](/pllm/docs/guides/explain/)；改完之后怎样公平地测，参见 [公平地测吞吐](/pllm/docs/howto/measure-throughput/)。

> [!NOTE] 基准测试环境与标尺数字
> 除非另有说明，本文引用的实测数字都来自同一基准环境：**单卡 RTX 4090（24 GiB）、`Qwen/Qwen2.5-7B-Instruct`（BF16）、vLLM 0.32.0、128 条并发序列（8 条 prompt $$\times$$ 16 个采样分支）、每条生成 256 个 token、开启 CUDA Graph；`pllm.benchmark.throughput` 预热 1 轮后取 3 轮中位数**。吞吐百分比均相对于同样设置下原生 vLLM（开启 CUDA Graph）的吞吐（约 $$5,894\text{ tok/s}$$，按 128 行折算单步约 $$21.7\text{ ms}$$）。同样设置换成 0.5B 模型，一步只有约 $$3.0\text{ ms}$$；下文在结论随模型大小改变的地方给出两者对照。

```python
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

import pllm
from pllm import testing
```

---

## 1. 确认前向是否仍在 CUDA Graph 中 {#zh-sec-2}

### 看什么 {#zh-sec-3}

运行 `pllm.explain(workload, config)`，观察输出的 `forward` 行：

```text
forward  CUDA graphs: no effect needs eager
```

若退回 eager 模式，该行会显示 `forward eager: <reason>`。在启动引擎后，也可以通过查询运行时对象 `llm.diagnostics().eager_reason` 确认（图模式下为空字符串 `""`；退回 eager 时为具体原因）。

### 意味着什么 {#zh-sec-4}

如果退回 eager 模式，每一步都要从 Python 逐个发射几百个算子。这份 CPU 开销是否致命，取决于一步 GPU 执行有多长：7B 在 128 并发下一步约 21.7 ms，主要花在读权重上，**退回 eager 后吞吐仍有图模式的 98%**（$$5,773$$ 对 $$5,894\text{ tok/s}$$）；同样设置的 0.5B 一步只有约 3 ms，**eager 掉到图模式的 49%**（$$20,926$$ 对 $$42,355\text{ tok/s}$$）。模型越小、并发越低、步越短，CUDA Graph 越要紧；图模式还是 pLLM 把训练和改写录进同一张图的前提。

### 怎么改 {#zh-sec-5}

检查负载是否保持了整模型 CUDA Graph：
1. **每请求与共享适配器均可进图**：无论 `pllm.model.Adapter` 绑定在 `pllm.RequestState`（每请求一份状态页）还是 `pllm.InstanceState`（共享或按准入版本隔离的状态槽），都会通过版本页表直接在 CUDA Graph 内部计算增量，不会破坏整模型捕获。
2. **激活补丁与切片图**：`pllm.model.ActivationPatch` 同样留在 CUDA Graph 内部。会替换整模型图的是 `@pllm.custom_forward`（改用按层段捕获的切片图，原因写在 `eager_reason` 里）或未捕获/不可追踪算子。
3. **配置强制报错**：在配置中声明 `ModelConfig("Qwen/Qwen2.5-7B-Instruct", cuda_graphs=True)`。默认的 `cuda_graphs="auto"` 遇到不能进整模型图的写法时以 eager 或切片模式运行（原因写在 `forward` 行和 `eager_reason` 里）；将其显式设为 `True` 后，任何破坏整模型图的声明都会在 `explain` 或 `make_llm` 阶段直接抛出 `ValueError`，杜绝吞吐隐蔽下降。

```python
@dataclass(frozen=True)
class FastLoRA(pllm.Workload):
    # 使用 RequestState：每个请求持有一份状态副本，适配器可完整录入 CUDA Graph
    lora = pllm.RequestState(
        {
            "a": pllm.Field(lambda in_features, out_features: (4, in_features)),
            "b": pllm.Field(lambda in_features, out_features: (out_features, 4)),
        },
        init=pllm.Imported(),
    )
    apply = pllm.model.Adapter(
        "lora",
        (pllm.model.Projection("q_proj", 12), pllm.model.Projection("v_proj", 12)),
        operator="low_rank",
    )


# 开启 cuda_graphs=True：若有任何导致退回 eager 的写法，构建期直接报错。
# 128 个请求各带一份 LoRA 时，改写后模型的 CUDA graph 要更大的捕获池（不够时启动报错并给出需要的字节数）。
config = pllm.ModelConfig(
    "Qwen/Qwen2.5-7B-Instruct",
    max_model_len=512,
    max_num_seqs=128,
    max_num_batched_tokens=8192,
    cuda_graphs=True,
    execution=pllm.ExecutionOptions(cuda_graph_pool_budget_bytes=512 << 20),
)
report = pllm.explain(FastLoRA(), config)
assert report.eager_reason == ""
```

### 代价是什么 {#zh-sec-6}

`RequestState` 会在每个请求槽位为独立状态预留版本页（例如 7B、128 并发下 `explain` 报每个字段 256 页，共约 22 MiB）；若算法只需跨请求共享（或随在线 `import_state` 版本推进），也可以把 `Adapter` 直接放在 `InstanceState` 上并同样留在 CUDA Graph 内（支持在途更新时预留 `2 + max_num_seqs = 130` 页，静态只读状态仅需 1 页）。两类适配器都会相应增大改写模型的 CUDA Graph 捕获池（不够时启动报错并给出需要的字节数）。如果改用 `@pllm.custom_forward` 或回退 eager，7B 上仅慢约 2%，而 0.5B 上 eager 只有图模式的一半；设置 `cuda_graphs=True` 没有任何运行时性能开销。

---

## 2. 消除数学运算中的主机同步 {#zh-sec-7}

### 看什么 {#zh-sec-8}

查看 `pllm.explain` 输出中的 `host` 行：

```text
host     0 syncs in traced workload math
```

该行必须显示为 `0 syncs`。对于模型层内部改写（`@pllm.custom_residual`、`@pllm.custom_attention`）和自定义算子，CPU 追踪会显示为 `unverified syncs`，需运行 `pllm.testing.check_on_gpu(workload, config)` 验证。

### 意味着什么 {#zh-sec-9}

pLLM 依赖 CPU 超前 GPU 异步准备元数据。一旦在负载方法中调用了 `.item()`、`.tolist()`、`.cpu()`、`print(tensor)`、`float(tensor)` 或 Python 值分支 `if tensor.any():`，CPU 就必须阻塞等待 GPU 将之前发射的所有算子执行完毕并把标量拷回主机内存。这一主机同步会彻底排空 GPU 执行流水线，造成数百微秒至毫秒级的 GPU 停顿。

### 怎么改 {#zh-sec-10}

将标量提取与动态条件分支改写为全张量数学：
1. **替代分支**：Python `if condition:` 改用 `torch.where` 或布尔掩码乘法。
2. **禁止标量转换**：不要在方法内调用 `.item()` 或在循环中打印张量。
3. **异步观察与健康诊断**：调试数值或收集统计量时，改用 `@pllm.export` 异步导出（数据写入有界锁页环形缓冲区，队列满时默认丢弃而不阻塞 GPU 或 CPU）；运行时可调用 `llm.diagnostics().numeric_health` 查询状态张量中是否存在 NaN/Inf，这一检查在稳态解码期间产生 0 次每步同步。
4. **GPU 契约检查**：运行 `pllm.testing.check_on_gpu(workload, config)`，它会在 `torch.cuda.set_sync_debug_mode("error")` 下实跑至少 8 个稳态解码步，抓出层内所有隐式同步。

```python
@dataclass(frozen=True)
class ZeroSyncWorkload(pllm.Workload):
    # 用 torch.where 替代 Python if 分支，不调用 .item()，确保零主机同步
    @pllm.add_logits()
    def steer(self, logits: torch.Tensor = pllm.layer_logits((-1,))) -> torch.Tensor:
        last = logits[:, -1]  # [rows, vocab]：模型自己的 logits
        return torch.where(last > 0.0, 0.1 * last, torch.zeros_like(last))  # 只放大正的 logits


sync_report = pllm.explain(ZeroSyncWorkload(), config)
assert sync_report.workload_host_syncs == 0
```

### 代价是什么 {#zh-sec-11}

张量分支（如 `torch.where`）会同时执行两个分支的数学计算，但在 GPU 充裕的算力下，这点计算耗时相比排空流水线的毫秒级停顿微不足道。

---

## 3. 确保旁路流任务与前向重叠 {#zh-sec-12}

### 看什么 {#zh-sec-13}

1. 查看 `pllm.explain` 的 `member` 列表中各任务的 `lane` 列是否显示为 `side`（旁路 CUDA stream）还是退回了 `main`（主 stream）。
2. 查看 `timing` 行是否显示 `publish_lag=1 (default)`。
3. 查看 `memory` 行打印的两个阈值：

```text
memory   peak 1512728864 bytes funded by pLLM [per-rank tensor reservations; excludes GPU-dependent BLAS workspace; execution_peak_budget_bytes]
         below 1512728864 bytes tasks run inline; below 1333743904 bytes planning fails
```

检查你配置的 `execution_peak_budget_bytes` 是否低于第一行提示的重叠阈值（上例是[负载的组成](/pllm/docs/guides/workload-anatomy/)里的 `Distill` 在 7B、128 并发下的输出：`1512728864 bytes`）。

### 意味着什么 {#zh-sec-14}

在线训练（`@pllm.train`）等较重的计算默认调度在旁路 stream（`side`）上异步执行。
* **默认 `publish_lag=1`**：第 $$n$$ 步 token 产生后，旁路流异步执行反向传播和优化器更新，更新在第 $$n+2$$ 步生效；第 $$n+1$$ 步前向直接读取第 $$n-1$$ 步的状态版本，无需等待第 $$n$$ 步的训练，实现训练与前向的并行重叠。
* **显存预算降级**：旁路重叠需要同时保留两个版本的工作区缓冲，显存峰值高于串行执行。如果你设置的 `ExecutionOptions(execution_peak_budget_bytes=...)` 低于重叠阈值（但高于规划失败下限），规划器不会直接报错，而是自动将旁路任务退回主 stream 内联执行（`inline`，`lane` 从 `side` 变为 `main`），任务与模型前向前向串行，失去重叠。

### 怎么改 {#zh-sec-15}

1. **保持滞后**：保持默认的 `publish_lag=1`，不要无故设为 `publish_lag=0`（除非算法在数学上严格要求下一步立即看到当前步的参数更新）。
2. **调大显存预算**：如果 `lane` 退回了 `main`，检查 `explain` 的 `memory` 行，将 `execution_peak_budget_bytes` 调大至超过重叠阈值，恢复旁路重叠。

```python
exec_opts = pllm.ExecutionOptions(
    execution_peak_budget_bytes=1024 << 20,  # 高于 overlap_threshold 阈值，保障旁路重叠
    publish_lag=1,  # 默认 1：第 n 步更新在第 n+2 步生效，训练与第 n+1 步前向重叠
)
overlap_config = pllm.ModelConfig(
    "Qwen/Qwen2.5-7B-Instruct",
    max_num_seqs=128,
    max_num_batched_tokens=8192,
    execution=exec_opts,
)
```

### 代价是什么 {#zh-sec-16}

重叠省下多少，取决于训练在整步里占多大比例（`ESamp`，rank 4）：
* **7B**：纯在线训练（$$\beta=0$$）时，`publish_lag=1` 为原生 vLLM 吞吐的 **97.4%**，`publish_lag=0` 为 **97.2%**；训练 + 引导采样（$$\beta=0.1$$）时为 **96.4%** 和 **96.1%**。每步训练只有约 0.6 ms GPU 时间，相对 21.7 ms 的一步，串行等待的代价在噪声附近。
* **0.5B（同样设置，只换模型）**：纯在线训练时 `publish_lag=1` 为 **91.9%**，`publish_lag=0` 为 **89.8%**。步越短，重叠越值钱。

代价是显存预留增加，但只落在读落后版本的状态上：有方法读落后 `publish_lag` 步的版本时，每个请求槽位要多留 `publish_lag` 份旧版本的参数（优化器字段不随旧版本保留；公式见[性能与显存 3.1 节](/pllm/docs/explain/performance/#zh-sec-10)）。对 7B 上的 `Distill`（`predict` 读落后的版本），128 并发下参数字段从 256 页增至 384 页，状态预留从约 95 MiB 增至约 110 MiB；`ESamp`（$$\beta=0$$）没有落后读者，两种设置都是每个字段 256 页。

---

## 4. 削减每步搬运字节数 {#zh-sec-17}

### 看什么 {#zh-sec-18}

查看 `pllm.explain` 的任务估算行与状态行：
* `projection FLOP`：是否有额外的 LM Head 全词表投影。
* `sampler ... bytes`：全词表张量的读写字节数。
* `state reads (origin, distance)`：状态读取字节数。
* `state` 行的 `logical transfer bytes/row` 与 `decode bytes/step`：跨 PCIe 主机查表的每步传输字节。

### 意味着什么 {#zh-sec-19}

解码阶段受限于显存与总线带宽。每步多搬运数据都会直接增加延迟：
1. **LM Head 投影与全词表张量**：7B 的 LM Head 权重（3584 $$\times$$ 152064 BF16）重达 1.09 GB。多算一次全词表投影耗时约 **$$+1.35\text{ ms}$$**（约一步的 6%；0.5B 的 272 MB 权重约 0.36 ms，却占它一步的 12%）；采样器扫描一遍 $$128 \times 152064$$ 的词表张量，每个阶段需 78–283 μs。
2. **大表跨 PCIe 传输**：4090 上 PCIe 4.0 x16 读主机内存实测约 23.7 GB/s，约为 GPU 显存带宽的 1/40。128 行查表搬运 38.9 MB 宽向量需要实打实的 **1.6 ms**（7B 一步的约 7%；0.5B 上超过半步）。

### 怎么改 {#zh-sec-20}

#### 1) 投影合并与引导采样优化 {#zh-sec-21}

* **使用 `@pllm.guide_sampling`**：引导采样返回 `[rows, hidden]` 隐藏层预测向量，框架默认使用 `guidance_capacity=4096` 的有界融合路径，只对过滤后保留的 top-K token 做投影和加权，不产生额外的独立全词表 GEMM。
* **避免多层 `pllm.layer_logits`**：每个 `layer_logits` 都会触发一次独立的全词表 GEMM，重复读取 LM Head 权重（7B 为 1.09 GB）。在 7B 上（32 个并发请求），读 9 层的 DoLa 吞吐为原生的 **66%**；模型越小，这部分占比越大。
* **按行过滤**：针对非全局干预，给效果方法加上 `rows=` 过滤器（如 `rows="trained"`），缩小参与计算的请求行数。

```python
@dataclass(frozen=True)
class FastGuidance(pllm.Workload):
    beta: float = 0.5

    def strength(self) -> float:
        return self.beta

    # 优先使用 @pllm.guide_sampling：只返回 [rows, hidden]，与采样器深度融合
    @pllm.guide_sampling(beta=strength)
    def steer(self, h: torch.Tensor = pllm.final_hidden()) -> torch.Tensor:
        return h


guidance_report = pllm.explain(FastGuidance(), config)
```

#### 2) 超大只读表使用 `placement="host"` 并规范方法切分 {#zh-sec-22}

对于几百 MiB 至几 GiB 的超大只读查找表，在 `pllm.Pretrained(..., placement="host")` 中声明放在主机内存。要让约 1.6 ms 的 PCIe 传输藏在前向计算中，必须遵循以下切分规则：

* **独立查表方法**：查表方法必须只读取 `pllm.input_tokens` 和只读状态（不带效果装饰器），通过 `pllm.result("lookup")` 把查表结果传给后续效果方法。
* **为什么必须这样写**：只有纯依赖 token 和只读表的独立方法，规划器才能将其调度在步首的 `request.prepare / side` 上，使用限制数量的 SM（RTX 4090 上限制为 32 个线程块，即 1/4 SM）在旁路拉取，让传输与主干模型前向重叠；若在查表方法中同时读取了 `pllm.hidden` 或 `pllm.final_hidden`，方法必须等待模型前向结束在 `state.publish / main` 执行，1.6 ms 的传输延迟将完全暴露在时间线上。

```python
@dataclass(frozen=True)
class FastHostLookup(pllm.Workload):
    buckets: int = 2048

    @property
    def memory(self) -> pllm.InstanceState:
        # 每个桶一行词表宽的 logits 先验；7B 上 2048 × 152064 × bf16 约 594 MiB，放在锁页主机内存
        return pllm.InstanceState({"table": lambda m: (self.buckets, m.vocab_size)},
                                  init=pllm.Pretrained("ngram_table.safetensors", placement="host"))

    # 独立查表方法：只读 token 历史和只读表，不加效果装饰器；规划器把它排在 request.prepare / side
    def lookup(self, ids: torch.Tensor = pllm.input_tokens(history=1),
               table: torch.Tensor = pllm.parameter("memory", "table")) -> torch.Tensor:
        keys = (ids[:, 0] * 31 + ids[:, 1] + 1) % self.buckets
        return table[keys]

    # 效果方法通过 pllm.result 读取查表结果
    @pllm.add_logits()
    def prior(self, logits: torch.Tensor = pllm.result("lookup")) -> torch.Tensor:
        return logits
```

### 代价是什么 {#zh-sec-23}

在 RTX 4090（7B，128 并发，594 MiB 表，示例 11 的 `NGramPrior`）基准测试中：
* 原生 vLLM：5,877 tok/s（约 21.8 ms/步）；
* 设备端查表（`placement="device"`）：5,742 tok/s（98%），`explain` 报的 pLLM 显存峰值 1,540 MiB；
* 主机端查表（`placement="host"`）：5,615 tok/s（96%），显存峰值降至 **946 MiB**（单步只比放显存慢约 0.5 ms，省下整张表的 594 MiB 显存）；
* 引导采样容量（`ESamp`，$$\beta=0.5$$）：默认 `guidance_capacity=4096` 吞吐为 5,678 tok/s（96.3%），256 和 1024 与它相差在噪声内；精确 dense 路径（`guidance_capacity=None`）为 5,533 tok/s（93.9%）。

<details markdown="1">
<summary>自测：为什么主机查表混读了 final_hidden 之后每步多出了 1–2 ms？</summary>

因为同一个方法读取了 <code>final_hidden</code>，该方法必须等待模型前向全部计算完毕才能启动（anchor 是 <code>state.publish / main</code>），无法在步首的 <code>request.prepare / side</code> 发射。结果约 1.6 ms 的 PCIe 传输完全暴露在前向之外。解决办法是将只依赖 <code>input_tokens</code> 的查表拆成独立方法，通过 <code>pllm.result</code> 传递给效果方法。

</details>

---

## 5. 优化 KV 显存：区分静态布局与动态读取 {#zh-sec-24}

### 看什么 {#zh-sec-25}

1. 查看 `pllm.explain` 的 `kv_capacity` 报告（显示各层的 `KV bytes/request` 与总容量）。
2. 运行时查看 `llm.diagnostics().kv_cache_blocks` 与 `kv_cache_max_concurrency`。

### 意味着什么 {#zh-sec-26}

在 vLLM 中，显存分配顺序是：模型权重 + CUDA Graph 捕获池 + 负载状态预留 $$\rightarrow$$ **剩下的显存全部归 KV Cache**。KV Cache 越小，最大并发请求数就越低。

`@pllm.custom_attention` 的两种机制有着根本区别：
1. **静态几何布局（`keep="window"`, `sink=True`, `keep="none", source=k`）**：直接映射到 vLLM 底层的物理块管理器。滑出窗口的物理块被真正回收给显存池，**真正释放 KV 显存容量**，并走原生 FlashAttention-2 / FlashInfer 内核，真正降低读取带宽。
2. **动态稀疏读取（`kv.attend(q, topk=...)`）**：所有历史 token 的 KV 仍然必须全部保存在显存中供每步打分挑选，**完全不节省 KV 显存容量**；它只省注意力的逻辑读取量。如果打分和挑选使用纯 PyTorch 算子写，碎片开销远超省下的读取收益。底层不支持基于分数的显存物理驱逐（如 H2O/SnapKV），因为 vLLM 的物理块表要求按顺序连续分配。

### 怎么改 {#zh-sec-27}

若目标是降低显存占用、提升系统并发上限，采用静态布局：
* **局部滑动窗口**：`@pllm.custom_attention(layers=tuple(range(0, 28, 2)), keep="window", window=256)`（7B 共 28 层）；
* **跨层共享 KV**：`@pllm.custom_attention(layers=tuple(range(1, 28, 2)), keep="none", source=1)`。

```python
@dataclass(frozen=True)
class LocalGlobalAttention(pllm.Workload):
    # 静态滑动窗口：真正释放滑出窗口的物理块，真省 KV 显存容量
    @pllm.custom_attention(layers=tuple(range(0, 28, 2)), keep="window", window=256)
    def read(self, q: torch.Tensor, kv: pllm.model.PagedKV) -> torch.Tensor:
        return kv.attend(q)[0]


attn_report = pllm.explain(LocalGlobalAttention(), config)
```

### 代价是什么 {#zh-sec-28}

在[示例 10](/pllm/docs/examples/10_custom_attention/) 的测试中（RTX 4090，7B，`max_model_len=2048`，128 token 分块 prefill；KV 上限来自 `explain`）：
* **单请求 KV 显存上限**：全量 112.0 MiB；偶数层窗口 256 降至 **70.4 MiB**；奇数层跨层共享降至 **56.0 MiB**（同样的 KV 池能装下的请求数翻倍）。
* **吞吐表现**：128 并发、每个序列约 270 token 时，原样读取、偶数层窗口和跨层共享都为原生 vLLM 的 99%（设置见示例 10）。
* **动态稀疏读取对比**：[自定义 attention 教程](/pllm/docs/examples/10_custom_attention/) 中的 `QuestRead`（纯 PyTorch 实现）显存依然为 112.0 MiB，吞吐断崖式下跌至原生的 **13%**（649 tok/s），因为未融合的 PyTorch 打分与 gather 开销远超节省的读取量。

---

## 6. 精细控制精度：平衡带宽与数值稳定性 {#zh-sec-29}

### 看什么 {#zh-sec-30}

1. 查看 `pllm.explain` 的 `state` 行中各个字段解析出的 `dtype`（如 `field 0: bfloat16, field 1: float32`）。
2. 运行时查看 `llm.diagnostics().numeric_health`。

### 意味着什么 {#zh-sec-31}

在带宽受限的自回归解码阶段，状态张量的精度直接决定显存读写搬运量。BF16 相比 FP32 体积减半，读写带宽和状态显存池占用均减半。但在自回归多步递推中，某些敏感变量若使用低精度会产生严重的累积舍入误差，引发数值发散或 NaN。

### 怎么改 {#zh-sec-32}

1. **默认使用 `dtype="auto"`**：对于只读权重、单步激活或不需要跨步累加的状态字段，`dtype="auto"` 会自动对齐模型的基座精度（BF16/FP16），节省 50% 的状态显存带宽。
2. **显式指定 `torch.float32`**：
   * 跨步累加器、优化器一阶/二阶矩状态；
   * 循环状态携带字段（`carry=`）；
   * 线性前缀扫描 `pllm.scan` 的系数与状态转移矩阵；
   * 写法：`pllm.Field(shape_fn, dtype=torch.float32)`。

```python
@dataclass(frozen=True)
class PrecisionAwareWorkload(pllm.Workload):
    state = pllm.RequestState(
        {
            "momentum": pllm.Field((896,), dtype=torch.float32),  # 显式 FP32：避免累加舍入误差
            "cached_feature": pllm.Field((896,)),  # 默认 auto：跟随模型基座自动选用 BF16
        }
    )

    @pllm.export(name="momentum_norm", max_rows=8, max_cols=1)
    def report(self, m: torch.Tensor = pllm.parameter("state", "momentum")) -> torch.Tensor:
        return m.norm(dim=-1, keepdim=True)


precision_report = pllm.explain(PrecisionAwareWorkload(), config)
```

### 代价是什么 {#zh-sec-33}

显式声明 FP32 使该字段的物理显存和搬运量增加一倍，但在小尺寸状态字段上，这点开销是保障数值稳定、避免长序列发散的必要成本。

---

## 7. 算子成为瓶颈时：接入自定义算子 {#zh-sec-34}

### 看什么 {#zh-sec-35}

1. 查看 `pllm.explain` 的 `custom_ops` 报告（显示算子名称、`flops_per_row` 和 CPU 参考实现）。
2. 查看性能剖析器中是否被大量细碎的 PyTorch 小算子填满时间线。

### 意味着什么 {#zh-sec-36}

在模型层内（`@pllm.custom_residual`）或注意力（`@pllm.custom_attention`）中直接用 PyTorch 编写多步张量操作，每次调用都会产生算子发射开销与显存往返写入。[自定义 attention 教程](/pllm/docs/examples/10_custom_attention/) 中的 Quest 算法用 PyTorch 编写每页打分和 gather，在 7B 上吞吐仅有原生的 **13%**。

### 怎么改 {#zh-sec-37}

将计算密集的逻辑用 Triton 或 CUDA 实现为融合 kernel，并用 PyTorch 标准机制注册：
1. 用 `@torch.library.custom_op` 注册具名算子，并通过 `register_fake` 提供静态形状推导。
2. 在负载方法中**直接调用算子函数对象本身**（例如 `fused_step(x, v)`），不要使用 `torch.ops.<namespace>.<op_name>`。直接调用能让规划器静态分析源码并把算子直接录入 CUDA Graph。详细编写与检查步骤参见 [自写算子](/pllm/docs/guides/custom-ops/)。

```python
@torch.library.custom_op("faster_guide::fused_step", mutates_args=(), device_types="cuda")
def fused_step(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return x + v


@fused_step.register_fake
def _fused_step_fake(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)
```

### 代价是什么 {#zh-sec-38}

在 RTX 4090 基准测试中：
* 示例 09 在第 12 层挂一个注册为 `custom_op` 的 Triton 算子，吞吐为原生 vLLM 的 **98%**（$$5,760$$ 对 $$5,872\text{ tok/s}$$）；
* 同样数学的 PyTorch 写法也是 **98%**（$$5,755\text{ tok/s}$$）：一次逐行加法本来就不是瓶颈；
* 编写算子需要额外手写 Triton 或 CUDA kernel 并维护 fake 形状函数。值得这样做的是 PyTorch 写起来太慢的步骤，例如示例 10 里用 PyTorch 写的 Quest 页打分，吞吐只有原生的 13%。

---

## 8. 只在需要的层上改写残差 {#zh-sec-39}

### 看什么 {#zh-sec-40}

`pllm.explain` 的 `forward` 行与挂了 `@pllm.custom_residual` 或 `ActivationPatch` 的层数。

### 意味着什么 {#zh-sec-41}

挂了残差改写的模型图里，vLLM 录的是原始算子而不是 Inductor 融合后的算子，即使恒等改写也要付这份代价（7B、128 并发下约 3%，见[性能与显存 4.4 节](/pllm/docs/explain/performance/#zh-sec-18)）；改写本身的计算按层数累加。

### 怎么改 {#zh-sec-42}

用 `layers=` 只挂真正需要改写的层；评估改写本身的代价时，和 eager 模式的原生 vLLM 对比，或直接跑 `testing.check_on_gpu`（它会建一个设置相同的原生引擎）。带状态的子层要更大的 CUDA graph 捕获池，按报错给出的字节数调大 `ExecutionOptions(cuda_graph_pool_budget_bytes=...)`。

---

## 9. 让跳层真的省下时间 {#zh-sec-43}

### 看什么 {#zh-sec-44}

`@pllm.custom_forward(tokens="decode")` 里 `layers(..., active=mask)` 的跳过比例，以及吞吐是否随它变化。

### 意味着什么 {#zh-sec-45}

decode 的矩阵乘受权重带宽限制：一批里只要还有一行要跑某层，这层的权重就要整份读一遍。部分行跳过只省下它们的 attention 读取；只有整批都跳过一段层时，才省下读权重的时间（原因与实测见[性能与显存 4.5 节](/pllm/docs/explain/performance/#zh-sec-19)）。

### 怎么改 {#zh-sec-46}

把跳层的判断设计成让整批更常一起跳（例如固定的层段、按步而不是按行的阈值），或者用在上下文很长、attention 读取占大头的场景；prompt 不跳层，`active=` 只用于 `tokens="decode"`。

---

## 10. 采样器：打开 FlashInfer 采样 {#zh-sec-47}

128 行以上的大批次、或负载改了 logits 时，vLLM 默认采样器的耗时会变得可观。装了 FlashInfer 时，在启动前设环境变量换成它的融合采样算子：

```bash
export VLLM_USE_FLASHINFER_SAMPLER=1
```

它同时改变原生与负载两边的采样耗时，所以对比吞吐时两边要在同一设置下测，并在报告里写明（见[公平地测吞吐](/pllm/docs/howto/measure-throughput/)）。

---

## 总结：性能优化速查清单 {#zh-sec-48}

| 检查项 | 观察指标（`explain` 或运行时） | 目标状态 | 优化手段 |
| :--- | :--- | :--- | :--- |
| **1. 图模式** | `forward` 行 / `eager_reason` | `CUDA graphs: no effect needs eager` / `""` | 共享可变状态改为 `RequestState`；设置 `cuda_graphs=True` 捕获异常 |
| **2. 主机同步** | `host` 行 | `0 syncs in traced workload math` | Python `if` 改用 `torch.where`；去除 `.item()`；用 `@pllm.export` 异步观察 |
| **3. 旁路重叠** | `member.lane` / `memory` 阈值 | `side` / 预算高于 `overlap_threshold` | 保持 `publish_lag=1`；调大 `execution_peak_budget_bytes` 保障重叠 |
| **4. 搬运字节** | `projection FLOP` / `sampler bytes` | 尽量少全词表 GEMM / 查表重叠率高 | 用 `@pllm.guide_sampling` 替代多层 `layer_logits`；主机表拆独立方法走步首拉取 |
| **5. KV 显存** | `kv_capacity` / `kv_cache_blocks` | 显存容量真正释放 | 采用滑动窗口或跨层共享等静态几何布局；认清动态 Top-K 不省显存容量 |
| **6. 状态精度** | `state` 各字段 `dtype` | 只读 auto 为 BF16，敏感变量显式 FP32 | 累加器、动量与 carry 显式声明 `torch.float32`，其余字段保持 `auto` |
| **7. 算子融合** | `custom_ops` 报告 / 剖析器小算子 | 融合进图，吞吐 $$\ge 95\%$$ | 用 `@torch.library.custom_op` 接入 Triton 融合 kernel，直接调用算子对象 |
| **8. 残差改写** | `forward` 行 / 挂载层数 | 只挂需要的层 | `layers=` 限定层；和 eager 原生或 `check_on_gpu` 对比改写本身的代价 |
| **9. 跳层** | 跳过比例与吞吐 | 整批一起跳 | 按层段或按步决定跳层；只在 decode 上用 `active=` |
| **10. 采样器** | 采样耗时 | 融合采样 | `VLLM_USE_FLASHINFER_SAMPLER=1`，两边同设置 |

</div>

</div>

<div class="zine-doc-footer">
  <a class="zine-pill-link" data-i18n-zh="← 返回 pLLM 项目与文档中心" data-i18n-en="← Back to pLLM Project &amp; Docs" href="/pllm/#docs">← Back to pLLM Project &amp; Docs</a>
  <a class="zine-pill-link" href="https://github.com/LinesHogan/pLLM/blob/main/doc/howto/faster.md" target="_blank" rel="noopener">GitHub &middot; doc/howto/faster.md ↗</a>
</div>

<script defer src="/assets/js/bilingual-post.js"></script>
