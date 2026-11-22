---
layout: page
title: "07 Rewriting Residual Connections: Value Residual, Hyper-Connections, and Memory Sublayers"
description: "pLLM Documentation · Algorithm Family Examples (06–13)"
permalink: /pllm/docs/examples/07_custom_residual/
nav: false
---

<link rel="stylesheet" href="{{ '/assets/css/pllm-hub.css' | relative_url | bust_file_cache }}">

<div class="zine-dispatch" data-title-zh="07 改写残差连接：value residual、超连接与记忆子层" data-title-en="07 Rewriting Residual Connections: Value Residual, Hyper-Connections, and Memory Sublayers">
  <div class="zine-dispatch-links">
    <a class="zine-pill-link primary" data-i18n-zh="← 返回 pLLM 文档中心" data-i18n-en="← pLLM Docs Hub" href="/pllm/#docs">← pLLM Docs Hub</a>
    <a class="zine-pill-link" data-i18n-zh="文档导航地图" data-i18n-en="Docs Roadmap" href="/pllm/docs/">Docs Roadmap</a>
    <span class="zine-lang-badge" data-i18n-zh="可执行教程 (.ipynb)" data-i18n-en="Notebook (.ipynb)">Notebook (.ipynb)</span>
  </div>
  <a class="zine-pill-link" data-i18n-zh="在 GitHub 打开 .ipynb ↗" data-i18n-en="Open .ipynb on GitHub ↗" href="https://github.com/LinesHogan/pLLM/blob/main/doc/examples/07_custom_residual.ipynb" target="_blank" rel="noopener">Open .ipynb on GitHub ↗</a>
</div>

<div id="markdown-content">

<div class="zine-lang-panel" data-lang-panel="en" lang="en" markdown="1">

<div id="en-sec-1"></div>

In Example 06, we read the model's residual stream. Many architectural proposals operate at this exact level: altering the values read by each attention layer (such as ResFormer's value residual), changing how sublayer outputs merge back into the residual stream (such as LAuReL or hyper-connections), or attaching a fast-weight memory beside a sublayer that reads and writes on the fly (such as TTT or Titans).

If you want to experiment with these structures inside an inference engine—for example, to load a checkpoint trained with the new architecture or to verify that the wiring matches—the conventional approach is to fork vLLM's model code and edit every layer's `forward` method. Once fast-weight memory enters the picture, you also have to manage isolated per-request state yourself so concurrent requests never corrupt one another.

In pLLM, `@pllm.custom_residual` lets a single method take over the residual update at chosen sites while native attention, MLP, and KV writes run as usual. When hyperparameters are set to an identity transform, the output matches the base model verbatim, making it easy to verify your wiring first. For stateful memory sublayers, each request owns an isolated copy of the memory updated at a fixed chunk length, so the mathematical result does not depend on how the scheduler splits a prompt across steps.

Before we start, note two limitations and costs:
First, `Qwen/Qwen2.5-7B-Instruct` used in this notebook was not trained with these structures (~15 GiB in bf16 weights, requiring a 24 GiB GPU). Setting non-identity parameters will change the model's outputs, not improve answer quality; our goal here is to demonstrate how to wire these structures into the inference engine and measure their runtime overhead.
Second, `@pllm.custom_residual` requires prefix caching to be disabled and runs on a single GPU. It executes inside CUDA graphs alongside the model; in Step 3 we measure how much the extra per-layer computation reduces throughput.

This notebook walks through four experiments:
1. **Value residual**: mix layer 0's values into subsequent layers' values.
2. **Hyper-connections**: expand one residual stream into multiple parallel streams, reading and mixing before and after each sublayer.
3. **Throughput cost**: benchmark these architectures running inside CUDA graphs.
4. **Stateful memory sublayer**: attach a per-request fast-weight matrix beside an MLP, reading and updating by fixed-size chunks.

Let's start by running native vLLM in eager mode as a baseline on two test prompts.

```python
import os

os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"  # Show only vLLM errors and suppress startup logs

from collections.abc import Mapping
from dataclasses import dataclass, replace

import torch
from transformers import AutoTokenizer
from vllm import SamplingParams

import pllm
from pllm.benchmark import throughput

MODEL = "Qwen/Qwen2.5-7B-Instruct"
# Rewritten residuals run inside CUDA graphs. Capture pools auto-size by default; set ExecutionOptions if you need an explicit cap.
CONFIG = pllm.ModelConfig(MODEL, max_model_len=512, max_num_seqs=32, max_num_batched_tokens=512,
                          gpu_memory_utilization=0.7, seed=0)
```

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL)
questions = ["Write one sentence about the ocean.", "Name three prime numbers."]
prompts = [tokenizer.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                         add_generation_prompt=True) for q in questions]
greedy = SamplingParams(max_tokens=40, temperature=0.0)


def answers(llm) -> list[str]:
    return [output.outputs[0].text for output in llm.generate(prompts, greedy, use_tqdm=False)]


def run(workload: pllm.Workload, config: pllm.ModelConfig = CONFIG) -> list[str]:
    with pllm.make_llm(config, workload=workload) as llm:
        texts = answers(llm)
        print("eager 原因:", repr(llm.diagnostics().eager_reason))
    return texts


# Control group: native vLLM without a workload, running in eager mode (see below for why).
with pllm.make_llm(replace(CONFIG, cuda_graphs=False)) as plain:
    baseline = answers(plain)
for text in baseline:
    print(repr(text))
```

```text
"The ocean covers approximately 71% of the Earth's surface and is home to a vast array of life forms, from microscopic plankton to massive whales."
'Sure, three prime numbers are 2, 3, and 5. Prime numbers are numbers greater than 1 that have no divisors other than 1 and themselves.'
```

## Step 1: Value Residual (Rewriting Each Layer's Value) {#en-step-1-value-residual-rewriting-each-layers-value}

In this step, we answer: how can we modify the value tensor that each attention layer reads and writes into the KV cache—without modifying model source code—and verify that our wiring is correct?

First, why did the baseline above explicitly disable CUDA graphs (`cuda_graphs=False`)? A model with `@pllm.custom_residual` still runs inside CUDA graphs, but vLLM disables inductor operator fusion so the graph captures the model's original operators, matching eager-mode arithmetic. By contrast, default native vLLM uses fused operators whose bf16 rounding differs slightly (Tutorial 00 encountered this same rounding difference at steering strength `0`). Therefore, to verify "verbatim equivalence to native" under an identity configuration, we compare against eager-mode native vLLM.

Our first architecture is ResFormer's value residual. The rule is simple: starting from layer 1, each attention layer mixes in a fraction of layer 0's value via $$V_l + \lambda_l (V_0 - V_l)$$. When the mixing coefficient `lam = 0`, the formula reduces to $$V_l$$, i.e., the original model.

In the code below, we intercept layers 1 through 27 (this model has 28 layers) using `@pllm.custom_residual(site="value", layers=tuple(range(1, 28)))`:
- A `site="value"` method receives the current layer's value tensor `v` of shape `[rows, kv_heads, head_dim]` and returns the replacement value (attention reads this returned tensor, and it is also what gets written to the KV cache).
- `pllm.value(0)` reads the same token's layer-0 value from the current forward pass.
- `pllm.depth()` returns the current layer index.
- Per-layer mixing coefficients `lam` live in an `InstanceState`, populated at initialization by `fill`.
- Incoming tensors have the model's `bfloat16` dtype; we cast with `.float()` for float32 math and cast back via `.to(v.dtype)`.

```python
@dataclass(frozen=True)
class ValueResidual(pllm.Workload):
    mixing: float = 0.0

    @property
    def gates(self) -> pllm.InstanceState:
        return pllm.InstanceState({"lam": lambda m: (m.layer_count,)}, init=pllm.Computed(self.fill))

    def fill(self, fields: Mapping[str, torch.Tensor], generator: torch.Generator) -> None:
        fields["lam"].fill_(self.mixing)

    @pllm.custom_residual(site="value", layers=tuple(range(1, 28)))
    def mix(self, v: torch.Tensor, first: torch.Tensor = pllm.value(0), layer: int = pllm.depth(),
            lam: torch.Tensor = pllm.parameter("gates", "lam")) -> torch.Tensor:
        return (v.float() + lam[layer] * (first.float() - v.float())).to(v.dtype)
```

Before generating, let's inspect the plan and check contracts on CPU: `explain` confirms the model remains in CUDA graphs, and `testing.check` uses `lam = 0` as the `neutral=` baseline. Because custom residuals operate inside the model and there is no model to edit on CPU, the check explicitly reports `SKIP` for model-site execution; bit-for-bit verification happens in GPU generation and `check_on_gpu`.

```python
from pllm import testing

print("CUDA graph:", "保留" if pllm.explain(ValueResidual(0.3), CONFIG).eager_reason == "" else "退回 eager")
print(testing.check(ValueResidual(0.3), CONFIG, neutral=ValueResidual(0.0)))
```

```text
CUDA graph: 保留
status  contract / reason
PASS    tensor_parallel_size=1: admission
PASS    compile
PASS    neutral
SKIP    model-site execution: CPU decoder does not execute residual hooks, overlays, spans or PositionState
```

```python
print("lam = 0，与原生逐字相同:", run(ValueResidual(0.0)) == baseline)
for text in run(ValueResidual(0.3)):
    print("lam = 0.3:", repr(text))
```

```text
eager 原因: ''
lam = 0，与原生逐字相同: True
eager 原因: ''
lam = 0.3: "The vast ocean covers over 70% of the Earth's surface and is home to a breathtakingly diverse array of marine life."
lam = 0.3: 'Certainly! Here are three prime numbers:\n\n1. 7\n2. 13\n3. 19\n\nPrime numbers are numbers that are greater than 1 and have no divisors'
```

From the output, `eager_reason` is an empty string on every run—the model executes inside CUDA graphs. When `lam = 0`, the outputs match eager-mode native vLLM verbatim (`True`), confirming the wiring is correct.
When `lam` is set to `0.3`, both responses change: the first rephrases the opening (`"The vast ocean covers over 70% of the Earth's surface..."`), and the second turns into a numbered list producing 7, 13, and 19 instead of 2, 3, and 5. The text remains fluent, which only shows the intervention took effect—not that the model improved, since the base model was never trained with this structure.

```python
@dataclass(frozen=True)
class HyperConnections(pllm.Workload):
    n: int = 4
    perturb: float = 0.0

    @property
    def routing(self) -> pllm.InstanceState:
        n = self.n
        return pllm.InstanceState(
            {"read": lambda m: (2 * m.layer_count, n), "mix": lambda m: (2 * m.layer_count, n, n),
             "write": lambda m: (2 * m.layer_count, n)},
            init=pllm.Computed(self.fill),
        )

    def fill(self, fields: Mapping[str, torch.Tensor], generator: torch.Generator) -> None:
        fields["read"].fill_(1 / self.n)
        fields["write"].fill_(1)
        eye = torch.eye(self.n, device=fields["mix"].device)
        noise = torch.randn(fields["mix"].shape, generator=torch.Generator(device=eye.device).manual_seed(5),
                            device=eye.device) * self.perturb
        fields["mix"].copy_(eye + noise - noise.mean(-1, keepdim=True))

    def choose(self, x: torch.Tensor, s: int = pllm.depth(),
               read: torch.Tensor = pllm.parameter("routing", "read")) -> torch.Tensor:
        return torch.einsum("n,rnd->rd", read[s].float(), x.float()).to(x.dtype)

    def streams(self) -> int:
        return self.n

    @pllm.custom_residual(site="both", read="choose", streams=streams)
    def update(self, x: torch.Tensor, y: torch.Tensor, s: int = pllm.depth(),
               mix: torch.Tensor = pllm.parameter("routing", "mix"),
               write: torch.Tensor = pllm.parameter("routing", "write")) -> torch.Tensor:
        mixed = torch.einsum("mn,rnd->rmd", mix[s].float(), x.float())
        return (mixed + write[s].float()[:, None] * y.float()[:, None]).to(x.dtype)

    @pllm.custom_residual(site="final")
    def collapse(self, x: torch.Tensor) -> torch.Tensor:
        return x.float().mean(1).to(x.dtype)
```

```python
print("恒等路由，与原生逐字相同:", run(HyperConnections()) == baseline)
for text in run(HyperConnections(perturb=0.5)):
    print("扰动 0.5:", repr(text))
```

```text
eager 原因: ''
恒等路由，与原生逐字相同: True
eager 原因: ''
扰动 0.5: "The ocean covers approximately ( ( 90 %) of the Earth's surface and is a source of wonder, biodiversity, and vital resources for our planet."
扰动 0.5: 'Certainly! Three prime numbers are 2, 3, and 5. Prime numbers are numbers that have exactly two distinct positive divisors: 1 and the number itself.'
```

Looking at the results: under identity routing (`perturb=0.0`, i.e. `read = 1/n`, `mix = identity`, `write = 1`), the 4 parallel streams still produce outputs verbatim identical to the eager base model (`True`). After adding `0.5` random perturbation to the mixing matrix, the first answer introduces an incorrect number and stray parentheses (`"The ocean covers approximately ( ( 90 %) of the Earth's surface..."`), and the second gives a different phrasing for prime numbers.

## Step 3: Measuring Throughput Overhead {#en-step-3-measuring-throughput-overhead}

In this step, we answer: when these architectures run inside CUDA graphs, how much does the extra per-layer tensor math reduce inference throughput?

To measure realistic throughput, we construct 32 concurrent requests (8 prompts × 4 samples each × 256 generated tokens) and benchmark with `pllm.benchmark.throughput`: 1 warmup round followed by 3 timed rounds taking the median, timing each full `generate` call (including prefill). We compare four configurations side by side: native vLLM with CUDA graphs, native vLLM in eager mode (as a reference for what losing CUDA graphs costs), `ValueResidual(0.3)` from Step 1, and 4-stream `HyperConnections(perturb=0.5)`.

```python
many = [f"Explain the mathematical idea numbered {i}:" for i in range(8)]
sampling = SamplingParams(n=4, max_tokens=256, temperature=0.8, top_p=0.95, ignore_eos=True, seed=7)

rates = {}
for name, config in (("原生 vLLM，CUDA graph", CONFIG), ("原生 vLLM，eager", replace(CONFIG, cuda_graphs=False))):
    with pllm.make_llm(config) as plain:
        rates[name] = throughput(plain, many, sampling)
for name, workload in (("value residual", ValueResidual(0.3)), ("超连接 n=4", HyperConnections(perturb=0.5))):
    with pllm.make_llm(CONFIG, workload=workload) as llm:
        rates[name] = throughput(llm, many, sampling)
graph = rates["原生 vLLM，CUDA graph"]
for name, rate in rates.items():
    print(f"{name:22s}: {rate:6.0f} token/s，为 CUDA graph 原生的 {rate / graph:.0%}")
```

```text
原生 vLLM，CUDA graph    :   1728 token/s，为 CUDA graph 原生的 100%
原生 vLLM，eager         :   1703 token/s，为 CUDA graph 原生的 99%
value residual        :   1703 token/s，为 CUDA graph 原生的 99%
超连接 n=4               :   1610 token/s，为 CUDA graph 原生的 93%
```

Here are the numbers from this run (RTX 4090, `Qwen2.5-7B-Instruct` bf16, vLLM 0.32.0, 32 concurrent requests, 256 tokens each, median of 3 rounds after 1 warmup):
- **Native vLLM, CUDA graph**: `1728 token/s` (`100%` baseline)
- **Native vLLM, eager**: `1703 token/s`, `99%` of native CUDA graph
- **value residual**: `1703 token/s`, `99%` of native CUDA graph
- **hyper-connections n=4**: `1610 token/s`, `93%` of native CUDA graph

On 7B, a single decode step takes roughly 19 ms and is largely memory-bandwidth bound on weight reads, so even eager native reaches `99%` of graph mode—this dataset does not highlight the benefit of staying in CUDA graphs (on smaller models with few-millisecond steps, eager often drops to roughly half of graph throughput). The gap comes from the added per-layer math: value residual modifies the value tensor across 27 layers (`99%`), while hyper-connections read and matrix-mix 4 residual streams at all 56 sublayers (28 layers × 2 sublayers each), reaching `93%`.

## Step 4: Stateful Memory Sublayer (Chunk-Updated Fast Weights) {#en-step-4-stateful-memory-sublayer-chunk-updated-fast-weights}

In this step, we answer: if a residual branch carries per-request "memory" that is read and updated as tokens arrive, how do we write it so concurrent requests stay isolated and the mathematical result is invariant to how the scheduler chunks execution into steps?

In plain terms, here is what this memory sublayer computes: beside layer 12's MLP, we place a `[hidden, hidden]` fast-weight matrix $$M$$ (one independent copy per request, initialized to zero).
- **Read**: each token passing through layer 12 normalizes its incoming residual stream $$x$$ to form key $$k$$, reads $$M k$$ from the matrix, scales it by a gating factor `gate` (`2` in this demo), and adds it into the residual stream after the MLP output.
- **Write (update)**: after accumulating a fixed-length chunk (`chunk=16`, i.e., 16 tokens), $$M$$ takes one gradient-descent step (learning rate `1.0`) toward the objective $$\|M k - y\|^2$$, where $$y$$ is the MLP output at that layer.

This is the core skeleton of "read-and-write fast-weight memory" architectures such as TTT (Test-Time Training) and Titans. When `gate = 0`, the read term is zeroed out and the model reduces to the base model.

In code, two parameters hand this rule over to the framework:
- `state="memory"` binds a single-field `RequestState` (`M`), which the framework allocates and isolates per request.
- `chunk=chunk_length` sets the chunk length to 16 (`chunk_length` is a method called once when pLLM binds the workload to read the instance's `chunk` attribute). The decorated `remember` method returns `(new_stream, new_M)`.

Chunk-update visibility follows a strict rule: every row within a chunk (16 rows) reads the **$$M$$ from the start of that chunk**; only when the last row of the chunk finishes does the returned **new $$M$$** take effect for the next chunk. If a request has fewer than 16 tokens remaining in a given engine forward step, the framework automatically buffers those partial rows until subsequent steps complete a full chunk. Mathematically, the final output therefore does not depend on how the vLLM scheduler slices a long prompt across steps.

```python
@dataclass(frozen=True)
class FastWeightMemory(pllm.Workload):
    layer: int = 12
    chunk: int = 16
    lr: float = 1.0
    gate: float = 2.0

    @property
    def memory(self) -> pllm.RequestState:
        return pllm.RequestState({"M": lambda m: (m.hidden_size, m.hidden_size)})

    def where(self) -> tuple[int, ...]:
        return (self.layer,)

    def chunk_length(self) -> int:
        return self.chunk

    @pllm.custom_residual(site="mlp", layers=where, state="memory", chunk=chunk_length)
    def remember(self, x: torch.Tensor, y: torch.Tensor,
                 M: torch.Tensor = pllm.parameter("memory", "M")) -> tuple[torch.Tensor, torch.Tensor]:
        k = torch.nn.functional.normalize(x.float(), dim=-1)  # Key for this row
        read = k @ M.T                                         # Value read from memory
        stream = (x.float() + y.float() + self.gate * read).to(x.dtype)
        return stream, M - self.lr / len(x) * (read - y.float()).T @ k  # One gradient step on |M k - y|^2
```

```python
# One FP32 M per request (3584 x 3584 x 4 bytes, ~49 MiB), multiplied by in-flight versions. 32 concurrent requests
# do not fit on a 24 GiB GPU, so we use 8 concurrent requests here and compare against native vLLM at the same concurrency.
# Stateful sublayers also need VRAM for the model graph and scratch workspace; lower vLLM's memory utilization. Capture pool auto-sizes.
SMALL = replace(CONFIG, max_num_seqs=8, gpu_memory_utilization=0.65)
for config in (CONFIG, SMALL):
    state = pllm.explain(FastWeightMemory(), config).states[0]
    print(f"{config.max_num_seqs:2d} 个并发请求：记忆状态 {state.bytes_per_root / 2**20:.1f} MiB × {state.roots} 份 = "
          f"{state.reserved_bytes / 2**30:.2f} GiB")
print("gate = 0，与原生逐字相同:", run(FastWeightMemory(gate=0.0), SMALL) == baseline)
for text in run(FastWeightMemory(), SMALL):
    print("gate = 2:", repr(text))
single = SamplingParams(n=1, max_tokens=256, temperature=0.8, top_p=0.95, ignore_eos=True, seed=7)
with pllm.make_llm(SMALL) as plain:
    small_graph = throughput(plain, many, single)
with pllm.make_llm(SMALL, workload=FastWeightMemory()) as llm:
    rate = throughput(llm, many, single)
print(f"8 个并发请求：原生 vLLM {small_graph:6.0f} token/s，记忆子层 {rate:6.0f} token/s，为原生的 {rate / small_graph:.0%}")
```

```text
32 个并发请求：记忆状态 49.2 MiB × 130 份 = 6.25 GiB
 8 个并发请求：记忆状态 49.2 MiB × 34 份 = 1.63 GiB
eager 原因: ''
gate = 0，与原生逐字相同: True
eager 原因: ''
gate = 2: "The ocean covers approximately 71% of the Earth's surface and plays a crucial role in regulating the planet's climate."
gate = 2: 'Sure, here are three prime numbers:\n\n1. 2\n2. 3\n3. 5\n\nPrime numbers are numbers greater than 1 that have no divisors other than 1'
8 个并发请求：原生 vLLM    484 token/s，记忆子层    362 token/s，为原生的 75%
```

From the output, when `gate = 0`, the results remain verbatim identical to the eager base model (`True`). When `gate` is set to `2`, the second half of the first answer changes (`"...plays a crucial role in regulating the planet's climate."`), and the second answer becomes a numbered list. The stateful memory sublayer also runs inside CUDA graphs (`eager_reason` is empty).

Each request's `M` is a `[hidden, hidden]` FP32 matrix, about 49 MiB on 7B; including in-flight versions, 32 concurrent requests would require 6.25 GiB, which does not fit alongside everything else on a 24 GiB GPU. Running at concurrency 8 instead still needs ~1.9 GiB for the capture pool. At 8 concurrent requests, the memory sublayer reaches `362 token/s`, or `75%` of native vLLM at concurrency 8 (`484 token/s`): every token reads the 49 MiB matrix once and updates it every 16 tokens, competing directly with model weight reads for memory bandwidth.

## Summary {#en-summary}

Looking back at the four experiments in this notebook, here are the key usages and boundaries of `@pllm.custom_residual`:
1. **Five attachment sites (`site`)**: attach at `attention`, `mlp`, `both` (applied to both attention and MLP at each layer), `value` (replacing the value read by attention and written to KV cache), or `final` (after all layers and before the final norm, typically to collapse multiple streams back into one).
2. **Verify with identity settings first**: whether mixing layer-0 values, splitting into parallel streams, or attaching fast-weight memory, configure parameters for an identity transform and verify verbatim equivalence against eager-mode native vLLM before testing non-trivial parameters.
3. **Stateful memory sublayers**: binding `state` (a single-field `RequestState`) and `chunk` (chunk length) gives each request an isolated state that reads and updates at a fixed chunk cadence.
4. **Current limitations and overhead**: `@pllm.custom_residual` runs inside CUDA graphs alongside the model, at the cost of the added per-layer compute (`99%`, `93%`, and `75%` measured on 7B here, with the memory sublayer measured at concurrency 8). It requires a single GPU and prefix caching disabled. In addition, a stateful (`state`) memory sublayer cannot be combined with `@pllm.custom_forward`, because custom forward passes do not guarantee a fixed visit count per layer per token.

This notebook demonstrated the skeleton of fast-weight memory on a standard base model. In the next example, [08 In-Place TTT](/pllm/docs/examples/08_inplace_ttt/), we load a checkpoint actually trained with chunked fast-weight updates—In-Place TTT—and inspect its behavior with real trained weights.

</div>

<div class="zine-lang-panel is-hidden" data-lang-panel="zh" lang="zh-CN" markdown="1">

<div id="zh-sec-1"></div>

在示例 06 中，我们读了模型的残差流。不少改模型结构的论文关注的也是这一层面的问题：比如改变每个 attention 层读到的 value（如 ResFormer 的 value residual），改变子层输出并入残差流的方式（如 LAuReL、超连接 hyper-connections），或者在子层旁边挂一个边读边写的快权重记忆（如 TTT、Titans）。

如果你想在推理服务里试验这类结构（例如拿到一个按新结构训练的检查点，或者想先验证结构接得对不对），传统做法通常是 fork 一份 vLLM 的模型代码，逐层修改 `forward` 函数；一旦涉及快权重记忆，还得自己管理每个请求独立的状态，防止并发请求之间互相串数据。

在 pLLM 中，`@pllm.custom_residual` 可以让一个方法接管指定位置的残差更新，而原生的 attention、MLP 和 KV 写入照常运行。当把超参数设为恒等变换时，输出与原模型逐字相同，方便你先确认结构没有接错；对于带状态的记忆子层，每个请求拥有独立的一份记忆并按固定块长更新，数学上的结果不依赖 prompt 怎样被调度器切成步。

在开始之前需要说明两点限制与代价：
第一，本篇使用的 `Qwen/Qwen2.5-7B-Instruct` 并没有按这些新结构训练过（bf16 权重约 15 GiB，需要一张 24 GiB 的卡）。因此，把参数设为非恒等值只会让模型的输出发生变化，不会让回答质量变好。本篇演示的是如何把这些结构接进推理引擎，以及它们带来的运行开销。
第二，`@pllm.custom_residual` 需要关闭前缀缓存、在单卡上运行。它和模型一起在 CUDA graph 里运行；每层多出来的计算会让吞吐下降多少，我们在第三步实测。

本篇分四步来进行实验：
1. **value residual**：把第 0 层的 value 混入后续各层的 value。
2. **超连接（hyper-connections）**：把一条残差流拆成多条并行的流，在子层前后做读取与混合。
3. **吞吐代价**：实测这几种结构在 CUDA graph 里运行时的吞吐。
4. **带状态的记忆子层**：在 MLP 旁挂载每个请求独立的快权重矩阵，实现按块读写更新。

下面先启动一个以 eager 模式运行的原生 vLLM 作为对照组，看看它在两个测试问题上的输出。

```python
import os

os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"  # 只显示 vLLM 的报错，省掉启动日志

from collections.abc import Mapping
from dataclasses import dataclass, replace

import torch
from transformers import AutoTokenizer
from vllm import SamplingParams

import pllm
from pllm.benchmark import throughput

MODEL = "Qwen/Qwen2.5-7B-Instruct"
# 改写后的残差在 CUDA graph 里运行。捕获池默认自动定容；需要显式上限时再设置执行选项。
CONFIG = pllm.ModelConfig(MODEL, max_model_len=512, max_num_seqs=32, max_num_batched_tokens=512,
                          gpu_memory_utilization=0.7, seed=0)
```

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL)
questions = ["Write one sentence about the ocean.", "Name three prime numbers."]
prompts = [tokenizer.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                         add_generation_prompt=True) for q in questions]
greedy = SamplingParams(max_tokens=40, temperature=0.0)


def answers(llm) -> list[str]:
    return [output.outputs[0].text for output in llm.generate(prompts, greedy, use_tqdm=False)]


def run(workload: pllm.Workload, config: pllm.ModelConfig = CONFIG) -> list[str]:
    with pllm.make_llm(config, workload=workload) as llm:
        texts = answers(llm)
        print("eager 原因:", repr(llm.diagnostics().eager_reason))
    return texts


# 对照组：不带负载的原生 vLLM，用 eager 模式运行（理由见下文）。
with pllm.make_llm(replace(CONFIG, cuda_graphs=False)) as plain:
    baseline = answers(plain)
for text in baseline:
    print(repr(text))
```

```text
"The ocean covers approximately 71% of the Earth's surface and is home to a vast array of life forms, from microscopic plankton to massive whales."
'Sure, three prime numbers are 2, 3, and 5. Prime numbers are numbers greater than 1 that have no divisors other than 1 and themselves.'
```

## 第一步：value residual（改写每层的 value） {#zh-sec-2}

这一步我们要回答的问题是：怎样在不改模型源码的情况下，修改每一层 attention 读到并写入 KV cache 的 value，并且确认代码接法是对的？

先解释一下上面的对照组为什么特意关掉了 CUDA graph（`cuda_graphs=False`）。带 `@pllm.custom_residual` 的模型仍然在 CUDA graph 里运行，但 vLLM 不再用 inductor 融合算子，graph 里捕获的是模型原本的算子，算术和 eager 模式一致；默认的原生 vLLM 用的是融合后的算子，bf16 舍入略有不同（教程 00 里强度 0 的引导就遇到过这种舍入差别）。所以要验证恒等设置下“与原生逐字相同”，我们和 eager 模式的原生 vLLM 对比。

我们要实现的第一个结构是 ResFormer 提出的 value residual。它的计算规则很直观：从第 1 层起，每层 attention 的 value 混入一部分第 0 层的 value，公式为 $$V_l + \lambda_l (V_0 - V_l)$$。当混合系数 `lam = 0` 时，公式退化为 $$V_l$$，也就是原模型。

在下面的代码中，我们用 `@pllm.custom_residual(site="value", layers=tuple(range(1, 28)))` 拦截第 1 到 27 层的 value（这个模型共 28 层）：
- `site="value"` 的方法会收到当前层的 value 张量 `v`，形状为 `[行, kv 头, 头维]`，方法返回替换后的 value（当前层的 attention 会读取这个新返回值，同时它也会被写进 KV cache）。
- `pllm.value(0)` 读取同一个 token 在这次前向传播里第 0 层的 value。
- `pllm.depth()` 返回当前层号。
- 各层的混合系数 `lam` 放在一份 `InstanceState` 里，在初始化时由 `fill` 方法填入设定值。
- 传入的张量是模型的 bf16 类型；方法里先用 `.float()` 转成 float32 计算，算完再转回原类型 `.to(v.dtype)`。

```python
@dataclass(frozen=True)
class ValueResidual(pllm.Workload):
    mixing: float = 0.0

    @property
    def gates(self) -> pllm.InstanceState:
        return pllm.InstanceState({"lam": lambda m: (m.layer_count,)}, init=pllm.Computed(self.fill))

    def fill(self, fields: Mapping[str, torch.Tensor], generator: torch.Generator) -> None:
        fields["lam"].fill_(self.mixing)

    @pllm.custom_residual(site="value", layers=tuple(range(1, 28)))
    def mix(self, v: torch.Tensor, first: torch.Tensor = pllm.value(0), layer: int = pllm.depth(),
            lam: torch.Tensor = pllm.parameter("gates", "lam")) -> torch.Tensor:
        return (v.float() + lam[layer] * (first.float() - v.float())).to(v.dtype)
```

生成之前，先在 CPU 上看计划并检查：`explain` 确认模型仍在 CUDA graph 里；`testing.check` 以 `lam = 0` 的负载作
`neutral=` 对照。定制残差作用在模型内部，CPU 上没有模型可以改，所以检查会写明跳过了这部分（`SKIP`），
逐位对照要等 GPU 上的生成和 `check_on_gpu`。

```python
from pllm import testing

print("CUDA graph:", "保留" if pllm.explain(ValueResidual(0.3), CONFIG).eager_reason == "" else "退回 eager")
print(testing.check(ValueResidual(0.3), CONFIG, neutral=ValueResidual(0.0)))
```

```text
CUDA graph: 保留
status  contract / reason
PASS    tensor_parallel_size=1: admission
PASS    compile
PASS    neutral
SKIP    model-site execution: CPU decoder does not execute residual hooks, overlays, spans or PositionState
```

```python
print("lam = 0，与原生逐字相同:", run(ValueResidual(0.0)) == baseline)
for text in run(ValueResidual(0.3)):
    print("lam = 0.3:", repr(text))
```

```text
eager 原因: ''
lam = 0，与原生逐字相同: True
eager 原因: ''
lam = 0.3: "The vast ocean covers over 70% of the Earth's surface and is home to a breathtakingly diverse array of marine life."
lam = 0.3: 'Certainly! Here are three prime numbers:\n\n1. 7\n2. 13\n3. 19\n\nPrime numbers are numbers that are greater than 1 and have no divisors'
```

从运行输出可以看到，每次运行的 `eager 原因` 都是空字符串：模型在 CUDA graph 里运行。当 `lam = 0` 时，输出与 eager 模式的原生 vLLM 逐字相同（`True`），这说明结构接对了。
当把 `lam` 设为 `0.3` 时，两个回答都变了：第一个回答换了说法（`"The vast ocean covers over 70% of the Earth's surface..."`），第二个变成了带编号的列表，而且列出的 7、13、19 和原来的 2、3、5 不同。回答依然通顺，但这只说明修改已经生效，不说明模型变好，因为底座模型没有按这个结构训练过。

## 第二步：超连接（把一条残差流扩成多条） {#zh-sec-3}

这一步我们要回答的问题是：如果一个结构不仅改子层的输出，还要把单条残差流扩展成多条并行流，并在子层前后分别做组合，该怎样用 `@pllm.custom_residual` 表达？

超连接（hyper-connections，Zhu et al., 2024）的做法是把原本的一条残差流换成 `n` 条并行的流：
1. 每个子层开始前，从 `n` 条流里按权重 `read` 加权求和读出一条，作为该子层的输入。
2. 子层算完得到输出 `y` 后，`n` 条流先按 `mix` 矩阵互相混合，再按权重 `write` 把子层输出 `y` 写回各条流。
3. 跑完所有层之后，在进入最终的 norm 之前，把 `n` 条流合成一条。

下面演示的是静态版本的超连接（权重是常数）。当取 `read = 1/n`、`mix = 单位阵`、`write = 1` 时，多条流里始终装着相同的值，行为就退化回原模型。

在代码中，我们配合使用几个参数来完成这套连接：
- `streams=streams` 让残差流的形状从 `[行, hidden]` 变成 `[行, n, hidden]`。
- `read="choose"` 指定在子层运行前调用 `choose` 方法，把 `[行, n, hidden]` 按权重压成 `[行, hidden]` 送给子层。
- `site="both"` 表示这个钩子同时作用于每一层的 attention 和 MLP 两个子层。此时 `pllm.depth()` 返回的是子层序号：attention 子层为 `2 × 层号`，MLP 子层再加 1（即 `2 × 层号 + 1`）。
- `site="final"` 挂载在所有层跑完之后、最终 norm 之前，这里的 `collapse` 方法对第 1 维求均值，把 `n` 条流重新合成一条。

顺带一提，像 LAuReL（Menghani et al., 2024）这类无需拆分成多条流、只在每个子层做残差组合的结构，同样可以用 `site="both"` 来写：直接返回 `alpha * y + beta * x + 低秩项` 即可。

```python
@dataclass(frozen=True)
class HyperConnections(pllm.Workload):
    n: int = 4
    perturb: float = 0.0

    @property
    def routing(self) -> pllm.InstanceState:
        n = self.n
        return pllm.InstanceState(
            {"read": lambda m: (2 * m.layer_count, n), "mix": lambda m: (2 * m.layer_count, n, n),
             "write": lambda m: (2 * m.layer_count, n)},
            init=pllm.Computed(self.fill),
        )

    def fill(self, fields: Mapping[str, torch.Tensor], generator: torch.Generator) -> None:
        fields["read"].fill_(1 / self.n)
        fields["write"].fill_(1)
        eye = torch.eye(self.n, device=fields["mix"].device)
        noise = torch.randn(fields["mix"].shape, generator=torch.Generator(device=eye.device).manual_seed(5),
                            device=eye.device) * self.perturb
        fields["mix"].copy_(eye + noise - noise.mean(-1, keepdim=True))

    def choose(self, x: torch.Tensor, s: int = pllm.depth(),
               read: torch.Tensor = pllm.parameter("routing", "read")) -> torch.Tensor:
        return torch.einsum("n,rnd->rd", read[s].float(), x.float()).to(x.dtype)

    def streams(self) -> int:
        return self.n

    @pllm.custom_residual(site="both", read="choose", streams=streams)
    def update(self, x: torch.Tensor, y: torch.Tensor, s: int = pllm.depth(),
               mix: torch.Tensor = pllm.parameter("routing", "mix"),
               write: torch.Tensor = pllm.parameter("routing", "write")) -> torch.Tensor:
        mixed = torch.einsum("mn,rnd->rmd", mix[s].float(), x.float())
        return (mixed + write[s].float()[:, None] * y.float()[:, None]).to(x.dtype)

    @pllm.custom_residual(site="final")
    def collapse(self, x: torch.Tensor) -> torch.Tensor:
        return x.float().mean(1).to(x.dtype)
```

```python
print("恒等路由，与原生逐字相同:", run(HyperConnections()) == baseline)
for text in run(HyperConnections(perturb=0.5)):
    print("扰动 0.5:", repr(text))
```

```text
eager 原因: ''
恒等路由，与原生逐字相同: True
eager 原因: ''
扰动 0.5: "The ocean covers approximately ( ( 90 %) of the Earth's surface and is a source of wonder, biodiversity, and vital resources for our planet."
扰动 0.5: 'Certainly! Three prime numbers are 2, 3, and 5. Prime numbers are numbers that have exactly two distinct positive divisors: 1 and the number itself.'
```

来看运行结果：在恒等路由（`perturb=0.0`，即 `read = 1/n`、`mix = 单位阵`、`write = 1`）下，4 条并行流算出的结果依然与 eager 原生模型逐字相同（`True`）。当给混合矩阵加上 `0.5` 的随机扰动后，第一个回答出现了错误的数字和多余的括号（`"The ocean covers approximately ( ( 90 %) of the Earth's surface..."`），第二个换了一种解释质数的说法。

## 第三步：实测吞吐代价 {#zh-sec-4}

这一步我们要回答的问题是：这些结构在 CUDA graph 里运行时，每层新加的张量计算会让推理吞吐降低多少？

为了测出真实吞吐，我们构造 32 个并发请求（8 个 prompt × 每个采样 4 个样本 × 生成 256 个 token），用 `pllm.benchmark.throughput` 测吞吐：先预热一轮，再计时三轮取中位数，每轮计整次 `generate` 的时间（含 prefill）。我们把四组配置放在一起对比：开启 CUDA graph 的原生 vLLM、eager 模式的原生 vLLM（作为参照，看看如果失去 CUDA graph 会是多少）、刚才写的 `ValueResidual(0.3)`，以及 4 条流的 `HyperConnections(perturb=0.5)`。

```python
many = [f"Explain the mathematical idea numbered {i}:" for i in range(8)]
sampling = SamplingParams(n=4, max_tokens=256, temperature=0.8, top_p=0.95, ignore_eos=True, seed=7)

rates = {}
for name, config in (("原生 vLLM，CUDA graph", CONFIG), ("原生 vLLM，eager", replace(CONFIG, cuda_graphs=False))):
    with pllm.make_llm(config) as plain:
        rates[name] = throughput(plain, many, sampling)
for name, workload in (("value residual", ValueResidual(0.3)), ("超连接 n=4", HyperConnections(perturb=0.5))):
    with pllm.make_llm(CONFIG, workload=workload) as llm:
        rates[name] = throughput(llm, many, sampling)
graph = rates["原生 vLLM，CUDA graph"]
for name, rate in rates.items():
    print(f"{name:22s}: {rate:6.0f} token/s，为 CUDA graph 原生的 {rate / graph:.0%}")
```

```text
原生 vLLM，CUDA graph    :   1728 token/s，为 CUDA graph 原生的 100%
原生 vLLM，eager         :   1703 token/s，为 CUDA graph 原生的 99%
value residual        :   1703 token/s，为 CUDA graph 原生的 99%
超连接 n=4               :   1610 token/s，为 CUDA graph 原生的 93%
```

看一下这次运行的数据（RTX 4090、`Qwen2.5-7B-Instruct` bf16、vLLM 0.32.0、32 个并发请求、各 256 token，预热 1 轮后 3 轮取中位数）：
- **原生 vLLM，CUDA graph**：`1728 token/s`（作为 `100%` 基准）
- **原生 vLLM，eager**：`1703 token/s`，为 CUDA graph 原生的 `99%`
- **value residual**：`1703 token/s`，为 CUDA graph 原生的 `99%`
- **超连接 n=4**：`1610 token/s`，为 CUDA graph 原生的 `93%`

在 7B 上，一步 decode 约 19 ms，大部分时间花在读权重上，所以连 eager 原生也有图模式的 `99%`：这一组数字看不出“留在 CUDA graph 里”的好处（在一步只有几毫秒的小模型上，eager 往往只有图模式的一半左右）。差距来自每层多出来的计算：value residual 修改 27 层的 value 张量，吞吐为 `99%`；超连接在 56 个子层（28 层 × 每层 2 个子层）上各自做 4 条残差流的读取与矩阵混合，吞吐为 `93%`。

## 第四步：带状态的记忆子层（按块更新的快权重） {#zh-sec-5}

这一步我们要回答的问题是：如果残差分支里带有需要随 token 边读边写的“记忆”（每个请求独立一份），怎样写才能既让并发请求互不干扰，又保证计算结果不随调度器的分步方式而改变？

我们先用白话描述一下这个记忆子层在算什么：在第 12 层 MLP 旁边放一个形状为 `[hidden, hidden]` 的快权重矩阵 $$M$$（每个请求各自拥有一份，初始为零）。
- **读取**：每个经过第 12 层的 token，先把自己的残差流输入 $$x$$ 做归一化得到键 $$k$$，从矩阵里读出 $$M k$$，乘上一个门控系数 `gate`（演示中设为 `2`），加到 MLP 输出后的残差流里。
- **写入（更新）**：每凑满固定长度的一块（这里设 `chunk=16`，即 16 个 token），矩阵 $$M$$ 就对着目标 $$\|M k - y\|^2$$（其中 $$y$$ 是这一层 MLP 的输出）走一步梯度下降（学习率设为 `1.0`）。

这就是 TTT（Test-Time Training）或 Titans 一类“边读边写的快权重记忆”的骨架。当 `gate = 0` 时，读出的记忆项被乘零丢弃，模型退化为原模型。

在代码中，我们通过两个关键参数把这个规则交给框架管理：
- `state="memory"` 指定关联一份只有一个字段 `M` 的 `RequestState`，框架会为每个请求独立分配并隔离这块状态。
- `chunk=chunk_length` 指定块长为 16：`chunk_length` 是上面定义的方法，pLLM 绑定负载时调用一次，读到实例的 `chunk`。被装饰的 `remember` 方法返回元组 `(新的残差流, 新的 M)`。

关于按块更新的时机，有一条明确的规则：同一个块（16 行）里的每一行，在计算读取时用的都是**这一块开始时的 $$M$$**；只有当这一块的最后一行跑完时，方法返回的**新的 $$M$$** 才会生效并留给下一块使用。如果在某次引擎前向步中，某个请求剩下的 token 数没凑满 16 行，这些没凑满的行会由框架自动保存起来，等到后续步凑满一整块时再触发状态更新。因此从数学上讲，最终生成的结果不依赖 vLLM 调度器怎样把一段长 prompt 切成多个步来执行。

```python
@dataclass(frozen=True)
class FastWeightMemory(pllm.Workload):
    layer: int = 12
    chunk: int = 16
    lr: float = 1.0
    gate: float = 2.0

    @property
    def memory(self) -> pllm.RequestState:
        return pllm.RequestState({"M": lambda m: (m.hidden_size, m.hidden_size)})

    def where(self) -> tuple[int, ...]:
        return (self.layer,)

    def chunk_length(self) -> int:
        return self.chunk

    @pllm.custom_residual(site="mlp", layers=where, state="memory", chunk=chunk_length)
    def remember(self, x: torch.Tensor, y: torch.Tensor,
                 M: torch.Tensor = pllm.parameter("memory", "M")) -> tuple[torch.Tensor, torch.Tensor]:
        k = torch.nn.functional.normalize(x.float(), dim=-1)  # 这一行的键
        read = k @ M.T                                         # 从记忆里读出的值
        stream = (x.float() + y.float() + self.gate * read).to(x.dtype)
        return stream, M - self.lr / len(x) * (read - y.float()).T @ k  # 对 |M k - y|^2 走一步梯度
```

```python
# 每个请求一份 FP32 的 M（3584 × 3584 × 4 字节，约 49 MiB），再乘上在途版本的份数。32 个并发请求放不进
# 24 GiB 的卡，这一步改用 8 个并发请求，吞吐也和同样 8 个并发请求的原生 vLLM 比。
# 带状态子层还要留出模型图和临时工作区的显存，降低 vLLM 显存比例。捕获池自动定容。
SMALL = replace(CONFIG, max_num_seqs=8, gpu_memory_utilization=0.65)
for config in (CONFIG, SMALL):
    state = pllm.explain(FastWeightMemory(), config).states[0]
    print(f"{config.max_num_seqs:2d} 个并发请求：记忆状态 {state.bytes_per_root / 2**20:.1f} MiB × {state.roots} 份 = "
          f"{state.reserved_bytes / 2**30:.2f} GiB")
print("gate = 0，与原生逐字相同:", run(FastWeightMemory(gate=0.0), SMALL) == baseline)
for text in run(FastWeightMemory(), SMALL):
    print("gate = 2:", repr(text))
single = SamplingParams(n=1, max_tokens=256, temperature=0.8, top_p=0.95, ignore_eos=True, seed=7)
with pllm.make_llm(SMALL) as plain:
    small_graph = throughput(plain, many, single)
with pllm.make_llm(SMALL, workload=FastWeightMemory()) as llm:
    rate = throughput(llm, many, single)
print(f"8 个并发请求：原生 vLLM {small_graph:6.0f} token/s，记忆子层 {rate:6.0f} token/s，为原生的 {rate / small_graph:.0%}")
```

```text
32 个并发请求：记忆状态 49.2 MiB × 130 份 = 6.25 GiB
 8 个并发请求：记忆状态 49.2 MiB × 34 份 = 1.63 GiB
eager 原因: ''
gate = 0，与原生逐字相同: True
eager 原因: ''
gate = 2: "The ocean covers approximately 71% of the Earth's surface and plays a crucial role in regulating the planet's climate."
gate = 2: 'Sure, here are three prime numbers:\n\n1. 2\n2. 3\n3. 5\n\nPrime numbers are numbers greater than 1 that have no divisors other than 1'
8 个并发请求：原生 vLLM    484 token/s，记忆子层    362 token/s，为原生的 75%
```

从输出可以看到，当 `gate = 0` 时，输出依然与 eager 模式的原生模型逐字相同（`True`）。当把 `gate` 设为 `2` 时，第一个回答的后半句变了（`"...plays a crucial role in regulating the planet's climate."`），第二个变成了带编号的列表。带状态的记忆子层同样在 CUDA graph 里运行（`eager 原因` 为空）。

这个子层每个请求的 `M` 是 `[hidden, hidden]` 的 FP32 矩阵，7B 上约 49 MiB；加上在途版本，32 个并发请求要 6.25 GiB，放不进 24 GiB 的卡，所以这一步只开 8 个并发请求，它的 graph 也要约 1.9 GiB 的捕获池。8 个并发请求下，记忆子层的吞吐是 `362 token/s`，为同样 8 个并发请求的原生 vLLM（`484 token/s`）的 `75%`：每个 token 都要读一次 49 MiB 的矩阵，每 16 个 token 还要更新一次，这些读写和读模型权重抢的是同一份显存带宽。

## 小结 {#zh-sec-6}

回顾本篇的四个实验，关于 `@pllm.custom_residual` 的用法与边界可以总结为以下几点：
1. **五种挂载位置（`site`）**：可以挂载在 `attention`、`mlp`、`both`（同时作用于每层的 attention 和 MLP）、`value`（替换 attention 读取并写入 KV cache 的 value）以及 `final`（所有层之后、最终 norm 之前，常用于把多条流合并回一条）。
2. **先做恒等校验**：无论是混入首层 value、拆分多条并行流，还是外挂快权重记忆，都可以先通过参数设置成恒等变换，与 eager 模式的原生模型比对是否逐字相同，以此确认结构接法无误。
3. **带状态的记忆子层**：通过绑定 `state`（一份单字段的 `RequestState`）和 `chunk`（块长），可以实现每个请求一份状态、按固定块长读写更新的快权重结构。
4. **当前限制与开销**：`@pllm.custom_residual` 和模型一起在 CUDA graph 里运行，代价是每层多出来的计算（本篇这次在 7B 上实测为 99%、93% 和 75%，记忆子层是在 8 个并发请求下测的）；它需要单卡并关闭前缀缓存；此外，带状态（`state`）的记忆子层不能和 `@pllm.custom_forward` 一起使用，因为在自定义前向传播中，每个 token 访问同一层的次数是不固定的。

本篇是在普通模型上演示快权重记忆的骨架。在下一篇 [08 In-Place TTT](/pllm/docs/examples/08_inplace_ttt/) 中，我们将加载一个真正按“按块更新快权重”训练出来的模型——In-Place TTT，看看它在真实权重下的运行表现。

</div>

</div>

<div class="zine-doc-footer">
  <a class="zine-pill-link" data-i18n-zh="← 返回 pLLM 项目与文档中心" data-i18n-en="← Back to pLLM Project &amp; Docs" href="/pllm/#docs">← Back to pLLM Project &amp; Docs</a>
  <a class="zine-pill-link" href="https://github.com/LinesHogan/pLLM/blob/main/doc/examples/07_custom_residual.ipynb" target="_blank" rel="noopener">GitHub &middot; doc/examples/07_custom_residual.ipynb ↗</a>
</div>

<script defer src="/assets/js/bilingual-post.js"></script>
