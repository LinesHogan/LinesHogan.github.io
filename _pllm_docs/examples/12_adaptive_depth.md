---
layout: page
title: "12 Adaptive Depth: Letting Each Token Decide Which Layers to Skip"
description: "pLLM Documentation · Algorithm Family Examples (06–13)"
permalink: /pllm/docs/examples/12_adaptive_depth/
nav: false
---

<link rel="stylesheet" href="{{ '/assets/css/pllm-hub.css' | relative_url | bust_file_cache }}">

<div class="zine-dispatch" data-title-zh="12 自适应深度：让每个 token 自己决定跳过哪些层" data-title-en="12 Adaptive Depth: Letting Each Token Decide Which Layers to Skip" data-desc-zh="pLLM 文档 · 算法族示例与实测 (06–13)" data-desc-en="pLLM Documentation · Algorithm Family Examples (06–13)">
  <div class="zine-dispatch-links">
    <a class="zine-pill-link primary" data-i18n-zh="← 返回 pLLM 文档中心" data-i18n-en="← pLLM Docs Hub" href="/pllm/#docs">← pLLM Docs Hub</a>
    <a class="zine-pill-link" data-i18n-zh="文档导航地图" data-i18n-en="Docs Roadmap" href="/pllm/docs/">Docs Roadmap</a>
    <span class="zine-lang-badge" data-i18n-zh="可执行教程 (.ipynb)" data-i18n-en="Notebook (.ipynb)">Notebook (.ipynb)</span>
  </div>
  <a class="zine-pill-link" data-i18n-zh="在 GitHub 打开 .ipynb ↗" data-i18n-en="Open .ipynb on GitHub ↗" href="https://github.com/LinesHogan/pLLM/blob/main/doc/examples/12_adaptive_depth.ipynb" target="_blank" rel="noopener">Open .ipynb on GitHub ↗</a>
</div>

<div id="markdown-content">

<div class="zine-lang-panel" data-lang-panel="en" lang="en" markdown="1">

<div id="en-sec-1"></div>

> **This is a conceptual demonstration.** This section uses a model not trained for layer skipping (`Qwen2.5-7B-Instruct`) solely to show how `active=` is written, how pLLM executes and checks it, and where its costs lie; it does not demonstrate the quality of layer-skipping algorithms. Real deployments require models trained for skipping—such as LayerSkip models trained with early-exit losses and layer dropout, CALM-style models with fine-tuned exit gates, or looped-depth models like Ouro. Skipping layers on an untrained model shifts the output distribution away from the base model, and the more tokens skip, the further it drifts (see Step 3).

After Tutorial 04 (customizing `forward` and looped depth), you may be interested in another family of adaptive-compute methods: early exit (e.g., CALM, LayerSkip), or per-token layer selection such as Mixture-of-Depths and Mixture-of-Recursions. Here we look at how to let each token dynamically decide which layers to skip inside an inference engine.

In Hugging Face, a single `if` statement lets a token skip layers. Inside a continuous-batching inference engine, several engineering problems immediately appear:
- Within the same batch, different tokens skip different layers.
- Skipped layers must still populate KV entries for that token; otherwise, later tokens attending at that layer would read uninitialized slots or stale data from prior requests.
- Sending the count of skipped rows back to the CPU to branch would trigger a GPU-to-CPU synchronization every step.
- CUDA graphs require fixed tensor shapes.

In pLLM, you can pass a dynamically computed boolean tensor `active` (shape `[rows, 1]`) to a slice inside a `@pllm.custom_forward` method:
- Rows with `active == False` skip computing that layer slice and pass `hidden` through unchanged.
- pLLM populates KV for skipped layers according to the `Layers` contract: on the first skip, that row's `hidden` is propagated through each skipped layer's native norm, QKV projection, and RoPE to compute and write K/V.
- The skip decision stays on GPU: pLLM stably compacts active rows into power-of-two buckets and selects which bucket executes via CUDA graph conditional nodes, with zero blocking GPU-to-CPU syncs per step.

To demonstrate concretely, we implement a simplified skip rule:
- All tokens execute layers 0–15;
- Compare residual-stream cosine similarity before and after layers 12–15; if direction barely changed (similarity exceeds a threshold), skip layers 16–19;
- All tokens execute layers 20–27 (this 7B model has 28 layers; the final layers strongly shape outputs, so we keep them unskipped).

Facts and limitations before starting:
- `Qwen2.5-7B-Instruct` was not trained for layer skipping (methods like LayerSkip require dedicated training); the more tokens skip, the further output distributions drift (Step 3). This example demonstrates syntax, checks, and costs, not output quality.
- When some tokens skip and others do not, almost no wall-clock time is saved (measured here: never-skip runs at `99%` of vanilla vLLM CUDA graphs; ~1/4 of tokens skipping runs at `98%`); throughput jumps clearly (`112%`) only when all tokens skip those 4 layers (4 of 28) together. `explain` explicitly notes that decode GEMMs save most when all rows exit.
- `explain` cannot yet estimate the CUDA graph capture memory required by `custom_forward`; if the budget is too low, `generate` fails with the required byte count. At 7B with 32 concurrent requests it needs ~143 MiB (exceeding the 64 MiB default), so we set 512 MiB here.
- `active=` is currently supported only in `tokens="decode"` custom forwards (prompt layers do not skip).

Experiments run on RTX 4090 (24 GiB), vLLM 0.32.0, `Qwen2.5-7B-Instruct` (bf16, weights ~15 GiB) in five steps:
1. Author the layer-skipping workload;
2. Run `explain` and contract checks on CPU;
3. Measure how many tokens skip and how outputs shift;
4. Benchmark throughput across configurations;
5. Run GPU contract checks.

```python
import os

os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"  # Show only vLLM errors to suppress startup logs

from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from vllm import SamplingParams

import pllm
from pllm import testing
from pllm.benchmark import throughput

# 7B bf16 weights are ~15 GiB, requiring a 24 GiB GPU.
MODEL = "Qwen/Qwen2.5-7B-Instruct"
# Up to 32 concurrent requests; custom_forward CUDA graphs on 7B exceed the default 64 MiB capture budget (128 MiB reported
# ~143 MiB needed), so allocate 512 MiB here.
CONFIG = pllm.ModelConfig(MODEL, max_model_len=512, max_num_seqs=32, max_num_batched_tokens=512,
                          gpu_memory_utilization=0.7, seed=0,
                          execution=pllm.ExecutionOptions(cuda_graph_pool_budget_bytes=512 << 20))
```

## Step 1: Authoring the Workload {#en-step-1-authoring-the-workload}

The question here is: how do we express a per-token skip rule in code inside `@pllm.custom_forward`?

As covered in Tutorial 04, `layers(start, stop, hidden=...)` executes a slice of layers and returns per-layer outputs (e.g., `layers(0, 12)[11]` is layer 11's output). In the code below:
- All rows first run layers 0–11 to produce `h12`, then layers 12–15 to produce `h16`.
- We compute the cosine similarity between `h16` and `h12`, producing a `[rows, 1]` boolean tensor `moved`: rows with similarity `<= threshold` are `True` (still changing direction and needing more compute).
- Calling `layers(16, 20, hidden=h16, active=moved)` executes layers 16–19 only for rows where `moved` is `True`.
- The default `threshold=2.0` never skips (since cosine similarity never exceeds 1), serving as our "skipping disabled" control.

Note that this rule inspects only each row's own tensors and never reads data from other requests in the batch (verified by the CPU contract checks next). On GPU, varying batch widths introduce tiny floating-point rounding differences, so tokens whose similarity sits right near the threshold can make different skip decisions across different batches.

```python
@dataclass(frozen=True)
class SkipMiddle(pllm.Workload):
    threshold: float = 2.0  # Skip [16, 20) when cosine similarity exceeds this; 2.0 never skips

    @pllm.custom_forward(tokens="decode")
    def forward(self, layers: pllm.model.Layers) -> torch.Tensor:
        h12 = layers(0, 12)[11]                      # Layers 0-11: run for all tokens
        h16 = layers(12, 16, hidden=h12)[15]         # Layers 12-15: run for all tokens
        moved = F.cosine_similarity(h16.float(), h12.float(), dim=-1).unsqueeze(-1) <= self.threshold
        h20 = layers(16, 20, hidden=h16, active=moved)[19]  # Layers 16-19: compute only for rows where moved is True
        return layers(20, hidden=h20)[-1]            # Layers 20-27: run for all tokens
```

## Step 2: `explain` and Contract Checks (CPU) {#en-step-2-explain-and-contract-checks-cpu}

Before starting the full model on GPU, let's run `pllm.explain` and `testing.check` on CPU to inspect how pLLM schedules this custom forward and whether the skip logic satisfies isolation and recomputation contracts. Run the two cells below; we will walk through the outputs right after.

```python
for line in str(pllm.explain(SkipMiddle(0.85), CONFIG)).splitlines():
    if line.startswith(("forward ", "host ")) or "active=" in line:
        print(line)
```

```text
forward  eager: @pllm.custom_forward schedules model slices and owns their wave graphs
forward  custom_forward  model.execute / main / eager  - (entry, 0)
host     0 syncs in traced workload math
timing   forward: active=CUDA conditional nodes (default): stable compaction into power-of-two buckets plus exact all-active width; 0 blocking host syncs per step; first skipped commit propagates native norm/QKV/RoPE, later skips retain KV; recompute replays by token role; decode GEMMs save most when all rows exit; full-QKV propagation is about 7% of Qwen2.5 layer FLOPs (estimate)
```

```python
print(testing.check(SkipMiddle(0.85), CONFIG))
```

```text
status  contract / reason
PASS    tensor_parallel_size=1: admission
PASS    compile
PASS    adaptive compaction = masked oracle
PASS    adaptive row permutation / isolation
PASS    adaptive KV policy
PASS    never exit bitwise
PASS    custom-forward instance reads / CPU decoder
PASS    row permutation
PASS    request isolation
PASS    mixed prefill/decode
PASS    chunking / tokens=
PASS    preempted
SKIP    model arithmetic: CPU fake decoder; native kernels need GPU
SKIP    model attention / KV: CPU decoder checks token math and state reads; real attention needs GPU
SKIP    publish_lag: no row method reads versioned state
SKIP    neutral: pass neutral= with the author's disabled settings
```

Let's read both outputs carefully.

In `pllm.explain`:
- `forward  eager: @pllm.custom_forward schedules model slices and owns their wave graphs`: vLLM's whole-model CUDA graph is replaced by pLLM scheduling layer slices and capturing its own wave graphs for them (Step 4's throughput confirms it does not run at eager speed).
- `host     0 syncs in traced workload math; exports: none`: zero GPU-to-CPU syncs inside the method.
- `timing   forward: active=CUDA conditional nodes (default)...` spells out how `active=` executes and what it costs:
  - `stable compaction into power-of-two buckets plus exact all-active width`: active rows are compacted into power-of-two bucket widths, using the exact width when all rows are active;
  - `0 blocking host syncs per step`: zero blocking host syncs per step;
  - `first skipped commit propagates native norm/QKV/RoPE, later skips retain KV`: first skip populates KV via native norm, QKV projection, and RoPE, while subsequent skips retain previously written KV;
  - `recompute replays by token role`: preemption recomputation replays according to token role;
  - `decode GEMMs save most when all rows exit`: decode GEMMs save the most when all rows exit together;
  - `full-QKV propagation is about 7% of Qwen2.5 layer FLOPs (estimate)`: propagating QKV to backfill KV costs roughly 7% of a Qwen2.5 layer's FLOPs (labeled an estimate by `explain`).

In `testing.check`:
- `adaptive compaction = masked oracle`: compacted execution matches a masked row-by-row reference oracle;
- `adaptive row permutation / isolation`: skip decisions do not depend on a row's batch position or other requests;
- `adaptive KV policy`: skipped-layer KV backfill matches the contract;
- `never exit bitwise`: when never skipping, outputs are bitwise identical to the program without `active`;
- Remaining standard scheduling contracts (`row permutation`, `request isolation`, `mixed prefill/decode`, `chunking / tokens=`, `preempted`) all `PASS`.
- The two `SKIP` lines (`model arithmetic` and `model attention / KV`) note that the CPU checker uses a tiny synthetic decoder to verify token math and state reads; real model arithmetic and attention are checked on GPU in Step 5.

---

## Step 3: Measuring Skip Rates and Output Shift {#en-step-3-measuring-skip-rates-and-output-shift}

Next: what range do cosine similarities fall into during real generation? How many tokens skip layers 16–19 at different thresholds, and how does the generated text change?

We do this in two parts:
1. Run the never-skip version (`threshold=2.0`) for 64 tokens across 8 questions while exporting per-decode-token cosine similarity across layers 12–15 (`pllm.residual(11)` and `pllm.residual(15)`) via `@pllm.export`, and compute the fraction of tokens that would skip at thresholds `0.9`, `0.85`, and `0.8`. Note this is measured along the unskipped trajectory; once layers are actually skipped, subsequent token trajectories diverge, so these fractions are estimates.
2. Run thresholds `0.9`, `0.85`, `0.8`, and "always skip" (`threshold=-2.0`): print the first answer and measure how far skipping pushes the output distribution using `pllm.testing.check_distribution`. It generates 64 greedy tokens from the uninstrumented model for each question, teacher-forces those exact tokens through the skipping workload, and compares full next-token probability distributions at every position via KL divergence and total variation distance (TV: 0 means identical, 1 means disjoint). Because both sides see the exact same prefix, a single divergence does not cascade into counting every later position as different.

```python
cosines: list[float] = []


@dataclass(frozen=True)
class SkipMiddleWithStats(SkipMiddle):
    """Same workload plus an export of per-decode-token cosine similarity across [12, 16)."""

    @pllm.export(name="cosine", max_rows=32, max_cols=1)
    def cosine(self, before: torch.Tensor = pllm.residual(11), after: torch.Tensor = pllm.residual(15)) -> torch.Tensor:
        return F.cosine_similarity(before.float(), after.float(), dim=-1)[:, None]

    def on_export(self, event: pllm.io.TraceEvent) -> None:
        cosines.extend(row[0] for row in event.tensor().tolist())


tokenizer = AutoTokenizer.from_pretrained(MODEL)
questions = ["Write one sentence about the ocean.", "Name three prime numbers.", "What is 12 times 7?",
             "Give me a synonym for happy.", "Who wrote Hamlet?", "Translate 'thank you' into French.",
             "What color is the sky on a clear day?", "List two uses of a paperclip."]
prompts = [tokenizer.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
           for q in questions]
greedy = SamplingParams(max_tokens=64, temperature=0.0, ignore_eos=True)


def generate(llm) -> list[list[int]]:
    return [list(output.outputs[0].token_ids) for output in llm.generate(prompts, greedy, use_tqdm=False)]


with pllm.make_llm(CONFIG, workload=SkipMiddleWithStats(2.0)) as llm:
    generate(llm)
values = torch.tensor(cosines)
print(f"{len(values)} 个 decode token；余弦相似度的 10%/50%/90% 分位数:",
      [round(v, 3) for v in values.quantile(torch.tensor([0.1, 0.5, 0.9])).tolist()])
for threshold in (0.9, 0.85, 0.8):
    print(f"阈值 {threshold}: {float((values > threshold).float().mean()):.0%} 的 token 会跳过第 16-19 层")
```

```text
504 个 decode token；余弦相似度的 10%/50%/90% 分位数: [0.768, 0.824, 0.866]
阈值 0.9: 0% 的 token 会跳过第 16-19 层
阈值 0.85: 23% 的 token 会跳过第 16-19 层
阈值 0.8: 72% 的 token 会跳过第 16-19 层
```

```python
for threshold in (0.9, 0.85, 0.8, -2.0):
    with pllm.make_llm(CONFIG, workload=SkipMiddle(threshold)) as llm:
        first = tokenizer.decode(generate(llm)[0])
    distance = testing.check_distribution(SkipMiddle(threshold), CONFIG, prompts=prompts, tokens=64)
    print(f"阈值 {threshold:+.2f}: 与不跳层模型的下一 token 分布相比，KL 平均 {distance.kl_plain_to_workload.mean:.3f}，"
          f"TV 平均 {distance.tv.mean:.2f}（{distance.token_count} 个位置）")
    print("    ", repr(first))
```

```text
阈值 +0.90: 与不跳层模型的下一 token 分布相比，KL 平均 0.000，TV 平均 0.00（512 个位置）
     "The ocean covers approximately 71% of the Earth's surface and is home to a vast array of life forms, from microscopic plankton to massive whales.<|im_end|>\n<|im_start|><|endoftext|>Human: Can you provide more details about the life forms in the ocean?<|im_end|>\n<|im_start|>ynes\nCertainly! The ocean is teeming with"
阈值 +0.85: 与不跳层模型的下一 token 分布相比，KL 平均 0.131，TV 平均 0.07（512 个位置）
     "The ocean covers approximately 71% of the Earth's surface and is home to a vast array of marine life.<|im_end|>\n<|im_start|>user\nCan you provide more details on the different types of marine life found in the ocean?<|im_end|>\n<|im_start|>venida\nCertainly! The ocean is teeming with a diverse array of life"
阈值 +0.80: 与不跳层模型的下一 token 分布相比，KL 平均 0.562，TV 平均 0.18（512 个位置）
     "The ocean covers approximately 71% of the Earth's surface and is home to a vast diversity of life, from microscopic organisms to massive whales.<|im_end|>\n<|im_start|>user\nCan you provide more information about the different types of marine life in the ocean?<|im_end|>\n<|im_start|>\n<|im_start|>user\nSure! The ocean is full"
阈值 -2.00: 与不跳层模型的下一 token 分布相比，KL 平均 0.868，TV 平均 0.25（512 个位置）
     "The ocean covers a significant portion of the Earth's surface and is home to a vast array of marine life, including but not limited to, diverse ecosystems and species.<|im_end|>\n<|im_start|>user\nWrite a sentence about the ocean.<|im_end|>\n<|im_start|>sentence about the ocean.\n\nAssistant: Sure, here is a sentence about the ocean"
```

Inspect Step 3's results:
- Across 8 answers of 64 tokens each, we collected 504 decode tokens (`8 × 63`: each answer's 1st generated token comes from the last prompt step, bypassing a `tokens="decode"`-only forward).
- The 10%/50%/90% quantiles of cosine similarity across layers 12–15 are `[0.768, 0.824, 0.866]`, with median `0.824`. Along this unskipped trajectory, ~`0%` of tokens skip layers 16–19 at threshold `0.9`, ~`23%` at `0.85`, and ~`72%` at `0.8`.
- Now look at the actual distribution shift after skipping (mean KL and TV over the 512 teacher-forced positions across 8 questions):
  - Threshold `+0.90`: both KL and TV print as `0` (no tokens skip); the first answer opens with `"The ocean covers approximately 71% of the Earth's surface ..."`.
  - Threshold `+0.85`: mean KL `0.131`, mean TV `0.07`; the second half of the first sentence already changes (`"a vast array of marine life."`).
  - Thresholds `+0.80` and `-2.00` (always skip): mean KL climbs to `0.562` and `0.868`, mean TV to `0.18` and `0.25`. The answers still read fluently, yet per-position distributions drift steadily away from the base model.

This confirms the opening caveat: `Qwen2.5-7B-Instruct` was not trained for layer skipping, so more skipping pushes distributions further off; fluent prose does not mean quality is preserved—always evaluate on your target task.

---

## Step 4: Measuring Throughput {#en-step-4-measuring-throughput}

Next: does skipping 4 layers on this 7B model speed up actual throughput or slow it down?

Benchmark setup: 32 concurrent requests (8 prompts × 4 samples × 256 tokens), timed with `pllm.benchmark.throughput` (1 warmup round + median of 3 timed rounds, full `generate` including prefill). We compare 5 setups:
1. Vanilla vLLM with CUDA graphs;
2. Vanilla vLLM in eager mode (`cuda_graphs=False`);
3. `SkipMiddle` never skip (`threshold=2.0`);
4. `SkipMiddle` threshold `0.85`;
5. `SkipMiddle` always skip (`threshold=-2.0`).

```python
many = [f"Explain the mathematical idea numbered {i}:" for i in range(8)]
sampling = SamplingParams(n=4, max_tokens=256, temperature=0.8, top_p=0.95, ignore_eos=True, seed=7)

rates = {}
for name, config in (("原生 vLLM，CUDA graph", CONFIG), ("原生 vLLM，eager", replace(CONFIG, cuda_graphs=False))):
    with pllm.make_llm(config) as plain:
        rates[name] = throughput(plain, many, sampling)
for name, threshold in (("从不跳", 2.0), ("阈值 0.85", 0.85), ("总是跳", -2.0)):
    with pllm.make_llm(CONFIG, workload=SkipMiddle(threshold)) as llm:
        rates[name] = throughput(llm, many, sampling)
graph = rates["原生 vLLM，CUDA graph"]
for name, rate in rates.items():
    print(f"{name:20s}: {rate:6.0f} token/s，为 CUDA graph 原生的 {rate / graph:.0%}")
```

```text
原生 vLLM，CUDA graph  :   1727 token/s，为 CUDA graph 原生的 100%
原生 vLLM，eager       :   1703 token/s，为 CUDA graph 原生的 99%
从不跳                 :   1703 token/s，为 CUDA graph 原生的 99%
阈值 0.85             :   1692 token/s，为 CUDA graph 原生的 98%
总是跳                 :   1928 token/s，为 CUDA graph 原生的 112%
```

Reading the throughput numbers:
- **Never skip** (`1703 token/s`) hits `99%` of vanilla vLLM CUDA graphs (`1727 token/s`), showing pLLM's wave graphs for sliced execution add negligible overhead. Vanilla vLLM eager (`1703 token/s`) also reaches `99%` here: a 7B decode step across 32 concurrent requests takes ~19 ms, dominated by streaming weights, hiding per-op CPU launch overhead (on small models with few-millisecond steps, eager is far slower).
- **Threshold `0.85`** (~`23%` of tokens skipping) runs at `1692 token/s` (`98%` of vanilla CUDA graphs), virtually tied with never-skip; **Always skip** reaches `1928 token/s` (`112%`).
- Why? As noted upfront and in `explain`, decode layer latency is dominated by reading that layer's weights: as long as even one row executes layers 16–19, the full weights must be streamed, while skipping rows only save their own attention reads and still pay QKV backfill on first skip. Only when every row skips together are those 4 layers (4 of 28, ~14%) truly bypassed—matching the 12% throughput gain.

---

## Step 5: GPU Contract Checks {#en-step-5-gpu-contract-checks}

Finally: does this workload pass runtime contract checks on the real GPU model and kernels?

We call `testing.check_on_gpu` with a smaller `max_num_seqs=8` config and `neutral=SkipMiddle(2.0)`. On GPU, it verifies zero GPU-to-CPU syncs across 8 steady decode steps, exact preemption recomputation, and bitwise parity with vanilla vLLM for greedy and fixed-seed sampling when skipping is disabled.

```python
print(testing.check_on_gpu(SkipMiddle(0.85), replace(CONFIG, max_num_seqs=8), neutral=SkipMiddle(2.0)))
```

```text
status  contract / reason
PASS    explain/probe
PASS    host syncs: eight steady decode steps
PASS    preempted: fixed per-token inputs, token IDs and terminal request state
PASS    neutral: greedy and seeded sampling bitwise
```

All four GPU checks (`explain/probe`, `host syncs: eight steady decode steps`, `preempted`, `neutral`) `PASS`.

### Summary {#en-summary}

- Inside `@pllm.custom_forward`, `active` is a `[rows, 1]` boolean tensor computed dynamically to gate whether each row executes a layer slice.
- Rows with `active == False` skip computing the slice, preserve `hidden`, and have their KV backfilled by pLLM according to contract.
- Skip decisions remain on GPU via power-of-two compaction and CUDA graph conditional nodes, incurring zero blocking GPU-to-CPU syncs per step.
- `pllm.explain` surfaces this execution mechanism and the conditions under which compute is actually saved.
- Real throughput gains depend on how many layers are skipped, how many rows skip together, and whether the model was trained for skipping.
- This is a conceptual demo: real layer-skipping algorithms require specifically trained models, or quality degrades first.
- Current limits: `active=` applies only to `tokens="decode"` custom forwards, and captured custom-forward CUDA graphs may exceed the default 64 MiB capture budget (7B at 32 concurrent requests required ~143 MiB here, so we set 512 MiB).

### Next {#en-next}

Continue to [13 Agent-Driven Algorithm Search](/pllm/docs/examples/13_algorithm_search/), which builds an automated search loop where an agent edits workloads that pLLM checks and scores—starting from an early-exit workload as one of its seeds.

</div>

<div class="zine-lang-panel is-hidden" data-lang-panel="zh" lang="zh-CN" markdown="1">

<div id="zh-sec-1"></div>

> **这是一个概念性演示。** 这一节用一个没有为跳层训练过的模型（Qwen2.5-7B-Instruct），只为展示 `active=` 怎么写、pLLM 怎么执行和检查它、代价在哪里；它不代表跳层算法的效果。真正使用跳层，需要专门训练过的模型，例如用早退损失和 layer dropout 训练过的 LayerSkip 模型、微调过退出判断的 CALM 式模型，或按循环深度训练的 Ouro 一类模型。没训练过的模型一跳层，输出分布就会偏离原模型，跳得越多偏得越远（见第三步）。

读完教程 04（改写 forward、循环深度）之后，你可能会关心另一类让不同 token 使用不同计算量的做法：early exit（例如 CALM、LayerSkip），或者 Mixture-of-Depths、Mixture-of-Recursions 这类按 token 选择执行哪些层的算法。这一节我们来看：怎样在推理引擎里让每个 token 在运行过程中自己决定跳过哪些层。

在 Hugging Face 里，写一个 `if` 就能让某个 token 少跑几层。但到了推理引擎里，这件事会遇到几个工程问题：
- 同一个 batch 里，不同 token 跳过的层不同。
- 被跳过的层还得给这个 token 写入 KV，否则后面的 token 在这一层做 attention 时，读到的会是空位或者别的请求留下的旧数据。
- 如果要把“有多少行跳过”这个数字传回 CPU 来决定下一步怎么算，每一步都会多出一次 GPU 到 CPU 的同步。
- CUDA graph 要求张量形状固定。

在 pLLM 里，你可以在 `@pllm.custom_forward` 方法里给某一段层传入一个布尔张量 `active`（形状为 `[行, 1]`，在方法内部实时算出）：
- `active` 值为 `False` 的行不计算这一段层，`hidden` 原样传过去。
- 这些行在被跳过层上的 KV 由 pLLM 按 `Layers` 文档的规则补上：第一次跳过时，用这一行的 `hidden` 经过每个被跳过层自己的 norm、QKV 投影和 RoPE 算出 K/V 并写进去。
- 跳不跳的决定留在 GPU 上：pLLM 把需要计算的行压紧成 2 的幂大小的分组，用 CUDA graph 的条件节点选择执行哪一组，每步没有阻塞的 GPU 到 CPU 同步。

为了具体演示，这一节实现一个简化的跳层规则：
- 第 0–15 层所有 token 都跑；
- 比较第 12–15 层前后的残差流余弦相似度，如果这 4 层几乎没改变方向（相似度大于阈值），就跳过第 16–19 层；
- 第 20–27 层所有 token 都跑（这个 7B 模型共 28 层，最后几层对输出影响很大，这里不跳）。

开始前先说明几项事实与限制：
- `Qwen2.5-7B-Instruct` 没有按“跳层”训练过（LayerSkip 一类方法需要专门训练），跳得越多，输出分布偏离原模型越远（实测见第三步）；这个示例只演示写法、检查和代价，不演示生成质量。
- 一部分 token 跳、一部分不跳时，几乎省不下时间（这次实测：不跳是原生 vLLM CUDA graph 的 99%，约四分之一的 token 跳层时是 98%）；所有 token 一起跳过这 4 层（28 层里的 4 层）时，吞吐才明显上去（112%）。`explain` 的输出里也直接说明了“所有行都退出时 decode 的矩阵乘省得最多”。
- 自定义 forward 的 CUDA graph 需要的捕获内存，`explain` 目前估不出来；内存不够时 `generate` 会报错并给出需要的字节数。在 7B、32 个并发请求下要 143 MiB 左右（超过默认的 64 MiB），这里设了 512 MiB。
- `active=` 目前只能用在 `tokens="decode"` 的 custom forward 里（prompt 的层不跳）。

本节实验的运行环境为 RTX 4090（24 GiB）、vLLM 0.32.0、`Qwen2.5-7B-Instruct`（bf16，权重约 15 GiB）。我们将按五个步骤推进：
1. 编写跳层负载；
2. 在 CPU 上运行 `explain` 和契约检查；
3. 观察规则会让多少 token 跳层、输出变成什么样；
4. 测量不同配置下的吞吐；
5. 在 GPU 上运行契约检查。

```python
import os

os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"  # 只显示 vLLM 的报错，省掉启动日志

from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from vllm import SamplingParams

import pllm
from pllm import testing
from pllm.benchmark import throughput

# 7B 的 bf16 权重约 15 GiB，需要一张 24 GiB 的卡。
MODEL = "Qwen/Qwen2.5-7B-Instruct"
# 最多 32 个并发请求；自定义 forward 的 CUDA graph 在 7B 上超过默认的 64 MiB 捕获预算（给 128 MiB 时报错说要
# 143 MiB），这里给 512 MiB。
CONFIG = pllm.ModelConfig(MODEL, max_model_len=512, max_num_seqs=32, max_num_batched_tokens=512,
                          gpu_memory_utilization=0.7, seed=0,
                          execution=pllm.ExecutionOptions(cuda_graph_pool_budget_bytes=512 << 20))
```

## 第一步：写负载 {#zh-sec-2}

这一步我们要回答的问题是：怎样在 `@pllm.custom_forward` 里把“每个 token 跳不跳层”的规则写成代码？

教程 04 讲过，`layers(start, stop, hidden=...)` 用来跑一段层并返回每层的输出。比如 `layers(0, 12)[11]` 就是第 11 层的输出。在这段代码里：
- 我们先让所有行跑完第 0–11 层拿到 `h12`，再跑完第 12–15 层拿到 `h16`。
- 接着算 `h16` 和 `h12` 的余弦相似度，得到形状为 `[行, 1]` 的布尔张量 `moved`：余弦相似度不大于阈值的行为 `True`，代表方向还在变、需要继续算。
- 调用 `layers(16, 20, hidden=h16, active=moved)` 时，只有 `moved` 为 `True` 的行会计算第 16–19 层这 4 层。
- 默认参数 `threshold=2.0` 表示永远不跳（因为余弦相似度不会超过 1），用作“关闭跳层”的对照版本。

注意，这里的判定规则只依赖当前这一行自己的张量，不读取同一 batch 里别的请求的数据（下一步的 CPU 契约检查查的就是这一点）。在 GPU 上，batch 宽度不同会带来很小的浮点舍入差别，相似度正好落在阈值附近的 token 可能会在不同的 batch 里做出不同的跳层决定。

```python
@dataclass(frozen=True)
class SkipMiddle(pllm.Workload):
    threshold: float = 2.0  # 余弦相似度大于它就跳过 [16, 20)；2.0 表示永远不跳

    @pllm.custom_forward(tokens="decode")
    def forward(self, layers: pllm.model.Layers) -> torch.Tensor:
        h12 = layers(0, 12)[11]                      # 第 0-11 层，所有 token 都跑
        h16 = layers(12, 16, hidden=h12)[15]         # 第 12-15 层，所有 token 都跑
        moved = F.cosine_similarity(h16.float(), h12.float(), dim=-1).unsqueeze(-1) <= self.threshold
        h20 = layers(16, 20, hidden=h16, active=moved)[19]  # 第 16-19 层：只有 moved 为 True 的行计算
        return layers(20, hidden=h20)[-1]            # 第 20-27 层，所有 token 都跑
```

## 第二步：`explain` 和契约检查（CPU） {#zh-sec-3}

写好负载之后，我们先不在 GPU 上启动完整模型，而是在 CPU 上跑 `pllm.explain` 和 `testing.check`，看看 pLLM 怎么调度这个自定义 forward，以及跳层逻辑是否满足框架的隔离与重算契约。运行下面两个单元格，我们将在输出之后逐条解读结果。

```python
for line in str(pllm.explain(SkipMiddle(0.85), CONFIG)).splitlines():
    if line.startswith(("forward ", "host ")) or "active=" in line:
        print(line)
```

```text
forward  eager: @pllm.custom_forward schedules model slices and owns their wave graphs
forward  custom_forward  model.execute / main / eager  - (entry, 0)
host     0 syncs in traced workload math
timing   forward: active=CUDA conditional nodes (default): stable compaction into power-of-two buckets plus exact all-active width; 0 blocking host syncs per step; first skipped commit propagates native norm/QKV/RoPE, later skips retain KV; recompute replays by token role; decode GEMMs save most when all rows exit; full-QKV propagation is about 7% of Qwen2.5 layer FLOPs (estimate)
```

```python
print(testing.check(SkipMiddle(0.85), CONFIG))
```

```text
status  contract / reason
PASS    tensor_parallel_size=1: admission
PASS    compile
PASS    adaptive compaction = masked oracle
PASS    adaptive row permutation / isolation
PASS    adaptive KV policy
PASS    never exit bitwise
PASS    custom-forward instance reads / CPU decoder
PASS    row permutation
PASS    request isolation
PASS    mixed prefill/decode
PASS    chunking / tokens=
PASS    preempted
SKIP    model arithmetic: CPU fake decoder; native kernels need GPU
SKIP    model attention / KV: CPU decoder checks token math and state reads; real attention needs GPU
SKIP    publish_lag: no row method reads versioned state
SKIP    neutral: pass neutral= with the author's disabled settings
```

我们先来读上面两段输出。

在 `explain` 的输出中：
- `forward  eager: @pllm.custom_forward schedules model slices and owns their wave graphs`：表示不用 vLLM 自己的整模型 CUDA graph 了，模型的层由 pLLM 按段调度，pLLM 会为这些分段捕获自己的 graph（从第四步的吞吐测试能看出来它不是 eager 的速度）。
- `host     0 syncs in traced workload math; exports: none`：方法内部没有 GPU 到 CPU 的同步。
- `timing   forward: active=CUDA conditional nodes (default)...` 这一行列出了 `active=` 的执行机制与代价：
  - `stable compaction into power-of-two buckets plus exact all-active width`：需要计算的行被压紧到 2 的幂大小的分组中，当所有行都计算时则使用精确宽度；
  - `0 blocking host syncs per step`：每步 0 次阻塞的主机同步；
  - `first skipped commit propagates native norm/QKV/RoPE, later skips retain KV`：第一次跳过时按原生层的 norm、QKV 投影和 RoPE 补写 KV，之后再跳过时保留上次写入的 KV；
  - `recompute replays by token role`：请求被抢占后重算时，按 token 的角色重放；
  - `decode GEMMs save most when all rows exit`：所有行都退出时，decode 阶段的矩阵乘省得最多；
  - `full-QKV propagation is about 7% of Qwen2.5 layer FLOPs (estimate)`：补写 KV 所需的 QKV 计算大约占 Qwen2.5 一层 FLOP 的 7%（`explain` 标注为估算）。

在 `testing.check` 的输出中：
- `adaptive compaction = masked oracle`：把需要计算的行压紧后计算的结果，与“按掩码逐行计算”的参照结果一致；
- `adaptive row permutation / isolation`：跳层的决定不会随着行在 batch 中的位置改变，也不受别的请求影响；
- `adaptive KV policy`：被跳过层的 KV 补写规则符合契约；
- `never exit bitwise`：永远不跳时，计算结果与不带 `active` 的程序逐位相同；
- 其余常规调度契约（`row permutation`、`request isolation`、`mixed prefill/decode`、`chunking / tokens=`、`preempted`）也全部 `PASS`。
- 标记为 `SKIP` 的两行（`model arithmetic` 和 `model attention / KV`）说明 CPU 检查用的是假的小解码器，用来检查 token 算术和状态读取；真实的模型算术和 attention 需要到第五步在 GPU 上检查。

---

## 第三步：看规则会让多少 token 跳层、输出变成什么样 {#zh-sec-4}

这一步我们要回答的问题是：余弦相似度在实际生成时分布在什么范围？设不同的阈值会让多少 token 跳过第 16–19 层，生成的文字又会变成什么样？

做法分两步：
1. 先用永远不跳的版本（`threshold=2.0`）对 8 个问题各生成 64 个 token 的回答，同时用 `@pllm.export` 导出每个 decode token 在第 12–15 层前后的余弦相似度（读取 `pllm.residual(11)` 和 `pllm.residual(15)`），统计在 `0.9`、`0.85`、`0.8` 三个阈值下分别有多少比例的 token 会跳层。注意，这是按永远不跳时的轨迹统计的；真正跳层以后，后续 token 的轨迹会发生变化，所以这个比例只是估计。
2. 分别用 `0.9`、`0.85`、`0.8` 三个阈值以及“总是跳”（`threshold=-2.0`）各跑一次：打印第一个回答的文本，并用 `pllm.testing.check_distribution` 量出跳层把模型的输出分布推开了多远。它先让不带负载的模型对每个问题贪心生成 64 个 token，再把同样的 token 逐个喂给跳层负载（teacher forcing），在每个位置上比较两边下一个 token 的完整概率分布，给出 KL 散度和总变差距离（TV：0 表示两个分布相同，1 表示完全不重叠）。因为两边始终读同一段上下文，一处分歧不会让后面所有位置都算作不同。

```python
cosines: list[float] = []


@dataclass(frozen=True)
class SkipMiddleWithStats(SkipMiddle):
    """同一个负载，外加一个导出：每个 decode token 在 [12, 16) 前后的余弦相似度。"""

    @pllm.export(name="cosine", max_rows=32, max_cols=1)
    def cosine(self, before: torch.Tensor = pllm.residual(11), after: torch.Tensor = pllm.residual(15)) -> torch.Tensor:
        return F.cosine_similarity(before.float(), after.float(), dim=-1)[:, None]

    def on_export(self, event: pllm.io.TraceEvent) -> None:
        cosines.extend(row[0] for row in event.tensor().tolist())


tokenizer = AutoTokenizer.from_pretrained(MODEL)
questions = ["Write one sentence about the ocean.", "Name three prime numbers.", "What is 12 times 7?",
             "Give me a synonym for happy.", "Who wrote Hamlet?", "Translate 'thank you' into French.",
             "What color is the sky on a clear day?", "List two uses of a paperclip."]
prompts = [tokenizer.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
           for q in questions]
greedy = SamplingParams(max_tokens=64, temperature=0.0, ignore_eos=True)


def generate(llm) -> list[list[int]]:
    return [list(output.outputs[0].token_ids) for output in llm.generate(prompts, greedy, use_tqdm=False)]


with pllm.make_llm(CONFIG, workload=SkipMiddleWithStats(2.0)) as llm:
    generate(llm)
values = torch.tensor(cosines)
print(f"{len(values)} 个 decode token；余弦相似度的 10%/50%/90% 分位数:",
      [round(v, 3) for v in values.quantile(torch.tensor([0.1, 0.5, 0.9])).tolist()])
for threshold in (0.9, 0.85, 0.8):
    print(f"阈值 {threshold}: {float((values > threshold).float().mean()):.0%} 的 token 会跳过第 16-19 层")
```

```text
504 个 decode token；余弦相似度的 10%/50%/90% 分位数: [0.768, 0.824, 0.866]
阈值 0.9: 0% 的 token 会跳过第 16-19 层
阈值 0.85: 23% 的 token 会跳过第 16-19 层
阈值 0.8: 72% 的 token 会跳过第 16-19 层
```

```python
for threshold in (0.9, 0.85, 0.8, -2.0):
    with pllm.make_llm(CONFIG, workload=SkipMiddle(threshold)) as llm:
        first = tokenizer.decode(generate(llm)[0])
    distance = testing.check_distribution(SkipMiddle(threshold), CONFIG, prompts=prompts, tokens=64)
    print(f"阈值 {threshold:+.2f}: 与不跳层模型的下一 token 分布相比，KL 平均 {distance.kl_plain_to_workload.mean:.3f}，"
          f"TV 平均 {distance.tv.mean:.2f}（{distance.token_count} 个位置）")
    print("    ", repr(first))
```

```text
阈值 +0.90: 与不跳层模型的下一 token 分布相比，KL 平均 0.000，TV 平均 0.00（512 个位置）
     "The ocean covers approximately 71% of the Earth's surface and is home to a vast array of life forms, from microscopic plankton to massive whales.<|im_end|>\n<|im_start|><|endoftext|>Human: Can you provide more details about the life forms in the ocean?<|im_end|>\n<|im_start|>ynes\nCertainly! The ocean is teeming with"
阈值 +0.85: 与不跳层模型的下一 token 分布相比，KL 平均 0.131，TV 平均 0.07（512 个位置）
     "The ocean covers approximately 71% of the Earth's surface and is home to a vast array of marine life.<|im_end|>\n<|im_start|>user\nCan you provide more details on the different types of marine life found in the ocean?<|im_end|>\n<|im_start|>venida\nCertainly! The ocean is teeming with a diverse array of life"
阈值 +0.80: 与不跳层模型的下一 token 分布相比，KL 平均 0.562，TV 平均 0.18（512 个位置）
     "The ocean covers approximately 71% of the Earth's surface and is home to a vast diversity of life, from microscopic organisms to massive whales.<|im_end|>\n<|im_start|>user\nCan you provide more information about the different types of marine life in the ocean?<|im_end|>\n<|im_start|>\n<|im_start|>user\nSure! The ocean is full"
阈值 -2.00: 与不跳层模型的下一 token 分布相比，KL 平均 0.868，TV 平均 0.25（512 个位置）
     "The ocean covers a significant portion of the Earth's surface and is home to a vast array of marine life, including but not limited to, diverse ecosystems and species.<|im_end|>\n<|im_start|>user\nWrite a sentence about the ocean.<|im_end|>\n<|im_start|>sentence about the ocean.\n\nAssistant: Sure, here is a sentence about the ocean"
```

来看看第三步的实验结果：
- 8 个回答、每个回答生成 64 个 token，一共收集到 504 个 decode token（`8 × 63`：每个回答的第 1 个生成 token 来自 prompt 阶段的最后一步，不经过这个只作用于 `tokens="decode"` 的程序）。
- 第 12–15 层前后残差流余弦相似度的 10%/50%/90% 分位数是 `[0.768, 0.824, 0.866]`，中位数为 `0.824`。按这条不跳层的轨迹估计，阈值为 `0.9` 时 `0%` 的 token 会跳过第 16–19 层，阈值为 `0.85` 时约 `23%`，阈值为 `0.8` 时约 `72%`。
- 再看实际跳层后的变化。KL 和 TV 是在不跳层模型的 64 个贪心 token 上逐位置比较下一 token 分布得到的平均值（8 个问题共 512 个位置）：
  - 阈值 `+0.90`：KL 和 TV 在打印精度下都是 `0`（没有 token 跳层），第一个回答开头是 `"The ocean covers approximately 71% of the Earth's surface ..."`。
  - 阈值 `+0.85`：KL 平均 `0.131`、TV 平均 `0.07`，第一个回答的后半句已经变了（`"a vast array of marine life."`）。
  - 阈值 `+0.80` 和 `-2.00`（总是跳）：KL 平均升到 `0.562` 和 `0.868`，TV 平均 `0.18` 和 `0.25`。这几个回答读起来仍然通顺，但每个位置上的分布离原模型越来越远。

这印证了开头说明的事实：`Qwen2.5-7B-Instruct` 没有针对跳层训练过，跳过的 token 越多，输出分布偏得越远；句子还通顺不等于质量没受影响，要用自己的任务评测。

---

## 第四步：测量吞吐 {#zh-sec-5}

这一步我们要回答的问题是：在这个 7B 模型上跳过 4 层，实际吞吐会变快还是变慢？

测试方法：同时跑 32 个并发请求（8 个 prompt × 每个采样 4 个样本 × 每个样本生成 256 个 token），用 `pllm.benchmark.throughput` 测：先预热 1 轮，再计时 3 轮取中位数，每轮计整次 `generate` 的时间（含 prefill）。我们把 5 种配置放在一起对照：
1. 原生 vLLM，开启 CUDA graph；
2. 原生 vLLM，eager 模式（关闭 CUDA graph）；
3. `SkipMiddle` 从不跳（`threshold=2.0`）；
4. `SkipMiddle` 阈值 `0.85`；
5. `SkipMiddle` 总是跳（`threshold=-2.0`）。

```python
many = [f"Explain the mathematical idea numbered {i}:" for i in range(8)]
sampling = SamplingParams(n=4, max_tokens=256, temperature=0.8, top_p=0.95, ignore_eos=True, seed=7)

rates = {}
for name, config in (("原生 vLLM，CUDA graph", CONFIG), ("原生 vLLM，eager", replace(CONFIG, cuda_graphs=False))):
    with pllm.make_llm(config) as plain:
        rates[name] = throughput(plain, many, sampling)
for name, threshold in (("从不跳", 2.0), ("阈值 0.85", 0.85), ("总是跳", -2.0)):
    with pllm.make_llm(CONFIG, workload=SkipMiddle(threshold)) as llm:
        rates[name] = throughput(llm, many, sampling)
graph = rates["原生 vLLM，CUDA graph"]
for name, rate in rates.items():
    print(f"{name:20s}: {rate:6.0f} token/s，为 CUDA graph 原生的 {rate / graph:.0%}")
```

```text
原生 vLLM，CUDA graph  :   1727 token/s，为 CUDA graph 原生的 100%
原生 vLLM，eager       :   1703 token/s，为 CUDA graph 原生的 99%
从不跳                 :   1703 token/s，为 CUDA graph 原生的 99%
阈值 0.85             :   1692 token/s，为 CUDA graph 原生的 98%
总是跳                 :   1928 token/s，为 CUDA graph 原生的 112%
```

我们来读一下吞吐数字：
- **从不跳**（`1703 token/s`）是原生 vLLM CUDA graph（`1727 token/s`）的 `99%`。这说明 pLLM 为分段执行捕获的 graph 没有带来可见的开销。原生 vLLM 的 eager 模式（`1703 token/s`）在这里也有 `99%`：7B 在 32 个并发请求下一步 decode 约 19 ms，主要花在读权重上，逐算子发射的 CPU 开销藏得住；在一步只有几毫秒的小模型上，eager 会慢得多。
- **阈值 0.85**（约 `23%` 的 token 跳层）跑出了 `1692 token/s`（原生 CUDA graph 的 `98%`），和“从不跳”差不多；而**总是跳**跑出了 `1928 token/s`（原生 CUDA graph 的 `112%`）。
- 原因正如开头和 `explain` 所说：decode 时一层的时间主要花在读这一层的权重上，只要还有一行要算这 4 层，权重就要整层读一遍；部分行跳过只省下这些行的 attention 读取，还要给第一次跳过的行补算 QKV 写入 KV。只有当全部行都一起跳过时，这 4 层（28 层里的 4 层，约 14%）的权重读取才真正省掉，吞吐提升 12% 与此相符。

---

## 第五步：GPU 契约检查 {#zh-sec-6}

最后一步我们要回答的问题是：在真实的 GPU 模型和内核上，这个负载能不能通过运行期契约检查？

我们调用 `testing.check_on_gpu`，使用 `max_num_seqs=8`（8 个并发请求）的小配置，并把 `neutral` 设为永不跳的 `SkipMiddle(2.0)`。它会在 GPU 上验证：8 个稳定 decode 步里没有 GPU 到 CPU 的同步、请求被抢占后重算能复现结果，以及关闭跳层时与原生 vLLM 的 greedy 和固定种子采样逐位相同。

```python
print(testing.check_on_gpu(SkipMiddle(0.85), replace(CONFIG, max_num_seqs=8), neutral=SkipMiddle(2.0)))
```

```text
status  contract / reason
PASS    explain/probe
PASS    host syncs: eight steady decode steps
PASS    preempted: fixed per-token inputs, token IDs and terminal request state
PASS    neutral: greedy and seeded sampling bitwise
```

GPU 上的四项检查（`explain/probe`、`host syncs: eight steady decode steps`、`preempted`、`neutral`）全部 `PASS`。

### 小结 {#zh-sec-7}

- 在 `@pllm.custom_forward` 中，`active` 是一个在方法里算出来的 `[行, 1]` 布尔张量，用来按行控制某一段层是否执行。
- `active` 为 `False` 的行跳过该段层的计算，`hidden` 原样保留，被跳过层的 KV 由 pLLM 按规则补上。
- 跳不跳的决定留在 GPU 上，通过压紧到 2 的幂分组和 CUDA graph 条件节点执行，每步没有阻塞的 GPU 到 CPU 同步。
- `pllm.explain` 会直接列出这套执行机制以及它能省下计算的条件。
- 实际吞吐收益取决于跳过多少层、有多少行一起跳，以及模型本身是否为跳层专门训练过。
- 这一节是概念性演示：真实的跳层算法需要专门训练过的模型，否则质量会先崩。
- 当前限制：`active=` 只用于 `tokens="decode"` 的自定义 forward，且捕获自定义 forward 的 CUDA graph 可能超过默认的 64 MiB 捕获预算（本节在 7B、32 个并发请求下要 143 MiB 左右，设为 512 MiB）。

### 下一步 {#zh-sec-8}

接下来可以阅读 [13 让 agent 搜索算法](/pllm/docs/examples/13_algorithm_search/)：那一节会搭建一个“让 agent 改写负载、由 pLLM 检查并打分”的自动搜索循环，它的起点之一就是一个 early exit 负载。

</div>

</div>

<div class="zine-doc-footer">
  <a class="zine-pill-link" data-i18n-zh="← 返回 pLLM 项目与文档中心" data-i18n-en="← Back to pLLM Project &amp; Docs" href="/pllm/#docs">← Back to pLLM Project &amp; Docs</a>
  <a class="zine-pill-link" href="https://github.com/LinesHogan/pLLM/blob/main/doc/examples/12_adaptive_depth.ipynb" target="_blank" rel="noopener">GitHub &middot; doc/examples/12_adaptive_depth.ipynb ↗</a>
</div>

<script defer src="/assets/js/bilingual-post.js"></script>
