---
layout: page
title: "08 In-Place TTT: A Model That Updates Its Weights While Reading"
description: "pLLM Documentation · Algorithm Family Examples (06–13)"
permalink: /pllm/docs/examples/08_inplace_ttt/
nav: false
---

<link rel="stylesheet" href="{{ '/assets/css/pllm-hub.css' | relative_url | bust_file_cache }}">

<div class="zine-dispatch" data-title-zh="08 In-Place TTT：一边读一边改权重的模型" data-title-en="08 In-Place TTT: A Model That Updates Its Weights While Reading" data-desc-zh="pLLM 文档 · 算法族示例与实测 (06–13)" data-desc-en="pLLM Documentation · Algorithm Family Examples (06–13)">
  <div class="zine-dispatch-links">
    <a class="zine-pill-link primary" data-i18n-zh="← 返回 pLLM 文档中心" data-i18n-en="← pLLM Docs Hub" href="/pllm/#docs">← pLLM Docs Hub</a>
    <a class="zine-pill-link" data-i18n-zh="文档导航地图" data-i18n-en="Docs Roadmap" href="/pllm/docs/">Docs Roadmap</a>
    <span class="zine-lang-badge" data-i18n-zh="可执行教程 (.ipynb)" data-i18n-en="Notebook (.ipynb)">Notebook (.ipynb)</span>
  </div>
  <a class="zine-pill-link" data-i18n-zh="在 GitHub 打开 .ipynb ↗" data-i18n-en="Open .ipynb on GitHub ↗" href="https://github.com/LinesHogan/pLLM/blob/main/doc/examples/08_inplace_ttt.ipynb" target="_blank" rel="noopener">Open .ipynb on GitHub ↗</a>
</div>

<div id="markdown-content">

<div class="zine-lang-panel" data-lang-panel="en" lang="en" markdown="1">

<div id="en-sec-1"></div>

If you want to serve a Test-Time Training (TTT) model, you immediately run into a structural difference: it stores "memory" in a different place than a standard inference model.

A standard model keeps its weights frozen during inference and stores all context memory in the KV cache. TTT's idea is to treat part of the model's weights as "fast weights" that are continuously updated while reading inputs, memorizing context directly in the weights. ByteDance Seed's In-Place TTT (_In-Place Test-Time Training_, arXiv 2604.06169) is called "In-Place" because it adds no new layers—instead, it updates the existing MLP down-projection matrices directly as fast weights.

Putting such a model into an inference server raises three concrete challenges:

1. **Each request needs its own isolated fast weights**: in the 4B checkpoint we use below, fast weights span 6 matrices totaling 149,422,080 parameters—about 570 MiB per copy in FP32. Concurrent requests must never overwrite each other's weights.
2. **Updates happen mid-forward**: a long prompt triggers fast-weight updates partway through prefill; during token-by-token decode, as soon as a chunk fills up, an update must also trigger immediately.
3. **The official inference code supports only a single request**: built on Hugging Face `generate`, the official implementation runs one request at a time (`assert len(input_ids) == 1, "only support bs=1 for TTT inference"` in the source code; the authors note in issue #4 that this is an implementation limitation).

In this example, we walk through four steps:

1. Explain the In-Place TTT update rule in plain terms;
2. Load a community checkpoint with pLLM's built-in `InPlaceTTTWorkload`;
3. Decode across a chunk boundary to observe fast weights updating mid-generation and the very next token reading the new weights;
4. Benchmark throughput against the official implementation on the same RTX 4090, and discuss the trade-off between FP32 and BF16 precision.

## Method and Checkpoint Source {#en-method-and-checkpoint-source}

Before running code, let's clarify two things: which weights the model updates and how, and where our checkpoint comes from.

### Method {#en-method}

- **Which weights are fast weights**: in selected MLP layers, the down-projection matrix $$W$$ (shape `[d_model, d_ff] = [2560, 9728]` in this checkpoint) serves as fast weights. This 36-layer checkpoint selects layers `0, 6, 12, 18, 24, 30` (every 6th layer). Attention layers and the other two MLP matrices remain frozen during inference.
- **How chunked updates work**: input tokens are grouped into fixed-size chunks (`1024` tokens per chunk in this checkpoint). Whenever a full chunk is read, a weight update is computed:
  $$\Delta W = \eta \cdot V^\top Z$$
  where $$Z$$ holds each token's intermediate MLP activation (the input right before the down projection) across the chunk; $$V$$ is the "target", computed from the MLP input via a learned 1D convolution (kernel size 5, applied strictly within the current chunk) followed by a learned square projection; and $$\eta$$ is the learning rate (`3` in this checkpoint). The fast weights then update as $$W \leftarrow W + \Delta W$$.
- **Who sees the updated weights**: strict chunk-level isolation—tokens in chunk $$i$$ only read the updates accumulated from chunks $$1 \dots i-1$$. If a prompt is long, several updates fire during a single prefill; any trailing tokens at the end of the prompt that do not fill a chunk are buffered and continue accumulating during decode. Once a full chunk completes, an update triggers: the final token that completes the chunk still reads the old weights, and the very next token reads the new weights.
- **No cross-request interference**: every request starts from the checkpoint's base $$W$$ and maintains its own independent updates.
- **No external labels needed at inference**: the 1D convolution and square projection parameters above are learned during "outer-loop" continued pretraining (the paper trains on Qwen3-4B for ~20B tokens at 32K length, then ~15B tokens at 128K length). At inference time, the model simply updates $$W$$ via the formula above as it reads tokens, with no external labels.
- **Two differences between the paper and the official code**: first, the paper describes targets $$V$$ coming from token embeddings, whereas the official code default (and this checkpoint's config) uses the MLP input (`ttt_target="hidden_states"`); second, the paper's Qwen3-4B evaluation clips each update's norm to `1e-5`, whereas the official code applies no clipping. pLLM's `InPlaceTTTWorkload` follows the official code and checkpoint config: MLP inputs as targets, without clipping.

### Checkpoint Source {#en-checkpoint-source}

The paper authors have not released model weights (in issue #1 of the official repository, they state that internal data policies prevent releasing training data and weights).

This example uses the community-uploaded checkpoint `zhongweixie/inplace-ttt-qwen3-4b-32k-ttt` (pinned to a specific revision in the code below). Its model card states it was trained from Qwen3-4B for 6000 steps at 32K context length. We verified its config and tensor shapes (6 fast-weight layers, each with a conv and projection, learning rate $$\eta=3$$, chunk size 1024), though we did not audit its training run and do not evaluate generation quality here—we use it solely to inspect runtime mechanics and measure throughput.

First, let's read this checkpoint's configuration to see which layers hold fast weights along with its chunk size and learning rate.

```python
import os

os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"  # Show only vLLM errors and suppress startup logs

import argparse
from dataclasses import replace
from pathlib import Path

import torch
from vllm import SamplingParams

import pllm
from pllm.benchmark import throughput
from pllm.workloads.inplace_ttt import InPlaceTTTConfig, InPlaceTTTWorkload

CHECKPOINT = "zhongweixie/inplace-ttt-qwen3-4b-32k-ttt"  # Community-trained checkpoint, not released by the paper authors
REVISION = "1f6aa7ea4bac2149358c86afea614f9edbd73c54"
ttt = InPlaceTTTConfig.from_checkpoint(CHECKPOINT, revision=REVISION)
print("快权重所在的层:", [target.layer for target in ttt.targets])
print("块长:", ttt.chunk_size, " 学习率:", ttt.learning_rate)
```

```text
快权重所在的层: [0, 6, 12, 18, 24, 30]
块长: 1024  学习率: 3.0
```

In the output above, `InPlaceTTTConfig.from_checkpoint` reads the fast-weight layers `[0, 6, 12, 18, 24, 30]`, chunk size `1024`, and learning rate `3.0` directly from the checkpoint's `config.json`.

## What the Workload Looks Like and How Much VRAM It Uses {#en-what-the-workload-looks-like-and-how-much-vram-it-uses}

Next we answer two questions: how much code does an In-Place TTT workload take in pLLM, and before loading ~8 GB of model weights onto the GPU, how can we tell upfront how much VRAM those fast weights will consume?

`InPlaceTTTWorkload` is implemented in `pllm/workloads/inplace_ttt.py` in roughly 60 lines of body code. Its core is a single method decorated with `@pllm.train_in_forward`: given one chunk of MLP inputs (`targets`), intermediate MLP activations (`inputs`), and the accumulated fast-weight delta (`delta`), it computes $$V$$ via conv1d + projection, computes the chunk's weight update, and returns the updated `delta` along with the output correction for that chunk. Here is a simplified excerpt:

```python
@pllm.train_in_forward(
    visibility="exact",
    state="dense_fast_weights",
    targets=targets,                                 # def targets(self): return self.config.targets
    weight="delta",
    chunk=chunk_size,                                # def chunk_size(self): return self.config.chunk_size
    tokens="all",
)
def step(self, delta=pllm.parameter("dense_fast_weights", "delta"), inputs=pllm.operand("input"),
         targets=pllm.operand("target"), conv=pllm.parameter("target_parameters", "conv"),
         proj=pllm.parameter("target_parameters", "proj")):
    values = conv1d(targets, conv) @ proj            # V: intra-chunk conv1d followed by projection
    update = (values.T @ inputs) * learning_rate     # ΔW = η · Vᵀ Z
    return delta + update, inputs @ delta.T          # new delta; correction read by this chunk
```

The workload author writes only the tensor math above—zero lines of page-table management, continuous-batching plumbing, or custom CUDA kernels. From those ~25 lines of pure math, pLLM **automatically** delivers five capabilities that the official repository (`ByteDance-Seed/In-Place-TTT`) cannot:

1. **Breaking the `bs=1` wall with continuous batching and per-request isolation**: the official inference code mutates a single cloned `down_proj` in place and hardcodes `assert len(input_ids) == 1`; pLLM automatically lowers `RequestState` onto a paged state pool so concurrent requests of different lengths and chunk frontiers read and update their own `delta` pages in the same forward pass, reclaiming pages automatically upon completion or preemption.
2. **Arbitrarily sliced Chunked Prefill & exact cross-phase chunk boundaries**: the official loop requires feeding the entire prompt at once; pLLM automatically buffers partial-chunk tail rows across steps so that no matter how vLLM slices a long prompt—or whether the 1024th token lands mid-prefill or mid-decode—the update fires on the exact chunk boundary and subsequent tokens immediately read the updated weights (`visibility="exact"`).
3. **Full-model decode CUDA Graph capture with device-side conditional updates (zero host syncs)**: hand-written chunk-boundary branches easily trigger CPU–GPU syncs or break graph capture; pLLM captures steady-state decode inside full-model CUDA graphs and wraps the 1-in-1024 chunk update in a device-side conditional node with 0 host syncs per step.
4. **Zero-copy page sharing & paged streaming readouts on non-boundary steps**: 6 layers of FP32 fast weights weigh ~570 MiB per request, so naive per-token copy-on-write and gathering would waste nearly 3 GB of HBM traffic per step; pLLM statically analyzes `step`'s FX graph to share unchanged `delta` pages read-only across the 1023/1024 non-boundary decode steps (0 bytes copied), borrows read-only `ModelCheckpoint` conv/proj weights in place, and lowers `inputs @ delta.to(inputs.dtype).T` into a paged streaming kernel that skips idle padding lanes.
5. **Single frozen BF16 base + FP32 delta accumulation**: all requests share a single read-only BF16 base model (while vLLM automatically skips `ttt_conv` / `ttt_proj` claimed by `ModelCheckpoint`) and accumulate `delta` in FP32, avoiding both base-weight duplication and BF16 mantissa swamping of small updates.

Before launching the engine to load ~8 GB of weights, let's call `pllm.explain` (see [Seeing Costs While Writing Code](/pllm/docs/guides/explain/)) to inspect the VRAM budget and chosen execution mode at concurrency 1 (`max_num_seqs=1`).

```python
small = pllm.ModelConfig(CHECKPOINT, revision=REVISION, max_model_len=2048, max_num_seqs=1,
                         max_num_batched_tokens=2048, gpu_memory_utilization=0.4)
plan = pllm.explain(InPlaceTTTWorkload(ttt), small)
print("eager 原因:", plan.eager_reason)
for state in plan.states:
    print(f"{state.name:20s} {state.scope:8s} 每份 {state.bytes_per_root / 2**20:6.1f} MiB × {state.roots} 份")
```

```text
eager 原因: 
dense_fast_weights   request  每份  714.0 MiB × 4 份
target_parameters    instance 每份   75.1 MiB × 1 份
```

Let's read the `pllm.explain` output line by line:

- **`eager_reason` is empty**: under default settings, exact chunk-boundary updates mid-decode run inside CUDA graphs; all subsequent runs and throughput numbers in this notebook have CUDA graphs enabled. Exiting an engine context releases its model, request state, and graphs, allowing another engine to be created in the same process.
- **`dense_fast_weights` (`714.0 MiB × 4` copies)**: this is per-request exclusive state (`request` scope). Each `714.0 MiB` slot holds the 6 layers of FP32 `delta` matrices (~570 MiB) plus the buffer for partial-chunk input rows. Four copies are reserved for 1 request so the engine can hold both the newly written version and the version currently being read across concurrent/in-flight steps.
- **`target_parameters` (`75.1 MiB × 1` copy)**: these are the 6 layers of 1D conv and square projection weights, stored in `instance` scope and shared across all requests at ~75 MiB total.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, revision=REVISION)
text = Path(argparse.__file__).read_text()  # A long text available on every machine: source code of the argparse module
ids = tokenizer(text, add_special_tokens=False).input_ids
print("长文本共", len(ids), "个 token")

prompt = [{"prompt_token_ids": ids[:1023]}]  # One token short of a full chunk (1024)
five = SamplingParams(temperature=0.0, max_tokens=5, ignore_eos=True, logprobs=0)
runs = {}
for name, config in (("TTT", ttt), ("不更新", replace(ttt, chunk_size=2048))):
    with pllm.make_llm(small, workload=InPlaceTTTWorkload(config)) as llm:
        sample = llm.generate(prompt, five, use_tqdm=False)[0].outputs[0]
        runs[name] = [(token, step[token].logprob) for token, step in zip(sample.token_ids, sample.logprobs)]
        print(f"{name}: 快权重更新了 {llm.diagnostics().update_count} 次")

print("两组生成的 token 相同:", [t for t, _ in runs["TTT"]] == [t for t, _ in runs["不更新"]])
print("\n生成的 token   算出它的位置   TTT 的 logprob   不更新的 logprob   差")
for k, ((token, a), (_, b)) in enumerate(zip(runs["TTT"], runs["不更新"])):
    print(f"{tokenizer.decode(token)!r:14s} {1022 + k:6d}      {a:10.4f}       {b:10.4f}      {a - b:+.4f}")
```

```text
长文本共 19736 个 token
TTT: 快权重更新了 6 次
不更新: 快权重更新了 0 次
两组生成的 token 相同: True

生成的 token   算出它的位置   TTT 的 logprob   不更新的 logprob   差
' _'             1022         -0.4400          -0.4400      +0.0000
'get'            1023         -1.9058          -1.9058      +0.0000
'_default'       1024         -2.2565          -2.3191      +0.0626
'_action'        1025         -1.8538          -1.8214      -0.0324
'(action'        1026         -1.7855          -1.7124      -0.0732
```

The output makes the chunk-boundary update timing unmistakable:

- **Update counts**: the `TTT` run performed **6 updates**, corresponding to 6 fast-weight layers × 1 completed chunk; the control run (updates disabled by setting `chunk_size=2048`, labeled `不更新` in the output) performed **0 updates**.
- **First two generated tokens (logprob diff `+0.0000`)**:
  - The 1st generated token `' _'` is computed from position `1022` (the last token of the prompt, i.e., the 1023rd token). The 1024-token chunk is not full yet, so both runs report the exact same logprob `-0.4400`.
  - The 2nd generated token `'get'` is computed from position `1023` (the 1024th token overall). Arriving at position `1023` completes the first 1024-token chunk and triggers the fast-weight update, **yet the final token that completes the chunk still reads the old weights**, so both logprobs remain identical (`-1.9058`, diff `+0.0000`).
- **From the 3rd generated token onward (logprobs diverge)**:
  - The 3rd token `'_default'` is computed from position `1024` (the 1st position of the second chunk). It reads the newly updated fast weights, shifting its logprob from `-2.3191` (without updates) to `-2.2565` (diff `+0.0626`).
  - Positions `1025` and `1026` likewise read the updated weights, with logprob diffs of `-0.0324` and `-0.0732`. (Across these 5 greedy decode steps, the argmax tokens happen to match despite the logprob shifts: `True` in the output.)

This is the exact chunk-level visibility defined in the paper. It happens automatically during vLLM's token-by-token decode without any hand-written scheduling code in `InPlaceTTTWorkload` to buffer chunks or flip weight visibility.

Having verified correctness, let's check the next natural question: how fast does this run in pLLM compared to the official Hugging Face implementation?

```python
def measure(llm, length: int, batch: int) -> float:
    """Throughput of batch identical prompts (first length tokens), each generating 256 greedy tokens."""
    prompts = [{"prompt_token_ids": ids[:length]}] * batch
    return throughput(llm, prompts, SamplingParams(temperature=0.0, max_tokens=256, ignore_eos=True))


long = pllm.ModelConfig(CHECKPOINT, revision=REVISION, max_model_len=16640, max_num_seqs=1,
                        max_num_batched_tokens=8192, enable_chunked_prefill=True, gpu_memory_utilization=0.5)
for dtype in (torch.float32, torch.bfloat16):
    with pllm.make_llm(long, workload=InPlaceTTTWorkload(ttt, dtype=dtype)) as llm:
        for length in (4096, 16384):
            print(f"pllm {str(dtype)[6:]:8s} 1 个请求  prompt {length:5d}: {measure(llm, length, 1):6.2f} token/s")
```

```text
pllm float32  1 个请求  prompt  4096:  73.02 token/s
pllm float32  1 个请求  prompt 16384:  47.97 token/s
pllm bfloat16 1 个请求  prompt  4096:  61.02 token/s
pllm bfloat16 1 个请求  prompt 16384:  42.32 token/s
```

```python
# 4 requests with one fast-weight state each (10 copies total including in-flight versions), plus CUDA graph private pools; leave slightly less VRAM for vLLM on a 24 GiB GPU.
four = replace(long, max_model_len=4416, max_num_seqs=4, max_num_batched_tokens=4416, gpu_memory_utilization=0.48)
with pllm.make_llm(four, workload=InPlaceTTTWorkload(ttt)) as llm:
    print(f"pllm float32  4 个请求  prompt  4096: {measure(llm, 4096, 4):6.2f} token/s")
```

```text
pllm float32  4 个请求  prompt  4096: 171.56 token/s
```

## Throughput Comparison with the Official Implementation {#en-throughput-comparison-with-the-official-implementation}

In this step, we answer: on the same hardware and identical input/output lengths, how does pLLM's throughput compare to the official implementation, and how much does throughput scale when moving from 1 request to concurrent requests?

First, let's align the benchmarking protocol on both sides:

- **Hardware and input**: the same RTX 4090 GPU, using the exact same prompt tokens (the first 4096 or 16384 tokens of `argparse` above).
- **Decoding settings**: greedy decoding, `ignore_eos=True` (no early stop on EOS), generating exactly 256 tokens.
- **Timing method**: 1 warmup round followed by 3 timed rounds taking the median; throughput is `total generated tokens / wall-clock time of generate (including prefill)`. That is exactly what `pllm.benchmark.throughput` measures.

The official implementation comes from `ByteDance-Seed/In-Place-TTT` using Hugging Face `generate`, BF16 precision, and FlashAttention 2. Because it depends on a separate Python environment (`torch 2.8.0`, `transformers 4.57.3`), we ran it separately on the same machine and same RTX 4090:

Official benchmark snippet (run in its own environment):

```python
from inference_model.hf_qwen3.modeling_qwen3 import Qwen3ForCausalLM  # Model class from the official repository

model = Qwen3ForCausalLM.from_pretrained(checkpoint, dtype=torch.bfloat16,
                                         attn_implementation="flash_attention_2").eval().cuda()
model.generate(input_ids=prompt, attention_mask=torch.ones_like(prompt), max_new_tokens=256,
               min_new_tokens=256, do_sample=False, eos_token_id=None, pad_token_id=151643, use_cache=True)
```

Putting the numbers from the two cells above side by side with the official implementation (all in `token/s`; pLLM on vLLM 0.32.0 with CUDA graphs enabled; official measured on the same day, same GPU, same protocol):

| | 1 request, prompt 4096 | 1 request, prompt 16384 | 4 requests, prompt 4096 |
| --- | --- | --- | --- |
| Official HF implementation (BF16) | 54.79 | 34.89 | Unsupported (hardcoded `bs=1`) |
| pLLM, `float32` fast weights | 73.02 | 47.97 | 171.56 (4 requests total) |
| pLLM, `bfloat16` fast weights | 61.02 | 42.32 | Not measured |

Several conclusions jump straight out of the table:

- **Single-request comparison**: pLLM's default `float32` mode is ~33% faster than the official code at prompt 4096 (`73.02` vs. `54.79`) and ~37% faster at prompt 16384 (`47.97` vs. `34.89`); `bfloat16` mode (reproducing the official BF16 rounding behavior) is also ~11% (`61.02`) and ~21% (`42.32`) faster. Why is `float32` substantially faster than `bfloat16` here? Because in `float32` mode pLLM detects that `inputs @ delta.to(inputs.dtype).T` is a pure linear readout and streams the residual correction directly from shared read-only FP32 pages on non-boundary steps, whereas `bfloat16` mode combines and rounds base weights with `delta` into BF16 and runs two matmuls to emulate in-place BF16 rounding bit-for-bit.
- **Multi-request concurrency**: with 4 concurrent requests, pLLM (`float32` fast weights) reaches `171.56 token/s` aggregate—**`3.13×`** the official single-request implementation (`54.79 token/s`) and **`2.35×`** single-request pLLM (`73.02 token/s`)—whereas the official implementation's `bs=1` limitation cannot serve concurrent requests. Four requests hold 8 fast-weight versions total (714 MiB each) plus CUDA graph private pools, fitting on a 24 GiB card at `gpu_memory_utilization=0.48`.
- **Bit-for-bit output comparison**: pLLM's generated token sequence is not bit-for-bit identical to the official implementation. Matmul tiling and floating-point reduction order differ between the two stacks, and whenever greedy decoding encounters two candidate tokens with nearly tied scores, tiny numerical differences can steer subsequent steps down a different token path.

## Precision Choice: FP32 or BF16 for Fast Weights? {#en-precision-choice-fp32-or-bf16-for-fast-weights}

In the benchmarks above, we saw two fast-weight precision modes: `float32` and `bfloat16`. In this final section, let's clarify how they differ numerically and which one to pick.

The paper does not specify what floating-point precision should store fast weights at inference time. Inspecting the official repository shows that it clones the original down-projection matrix as fast weights and adds each chunk's $$\Delta W$$ directly into that tensor. Thus, when the model is loaded in BF16, the fast weights are also BF16, with no separate FP32 master copy.

This creates a numerical hazard: **adding a tiny update directly onto a much larger BF16 matrix loses most of the update to BF16 mantissa rounding (swamping).**

We ran a rounding test on this checkpoint's actual layer-0 down-projection matrix: generating an update matrix of norm `1e-5` (the per-update clipping threshold used in the paper's Qwen3-4B evaluation) and adding it in BF16 directly onto layer 0's matrix changed only `472` out of all `24,903,680` elements, retaining only ~`0.97%` of the intended update magnitude. (Note: this is a numerical demonstration of floating-point rounding, not a model-quality experiment; moreover, this community checkpoint does not enable clipping, so its single-step updates are much larger than `1e-5`.)

To prevent small updates from being swallowed by the base weights, pLLM offers two engineering choices:

- **Default: store deltas separately in FP32 (`dtype=torch.float32`)**: base weights remain untouched in BF16, while each request stores its accumulated `delta` in FP32 and adds `inputs @ delta` to the base output during forward. Even tiny updates accumulate cleanly in FP32 `delta` without rounding away.
- **Official-rounding BF16 mode (`InPlaceTTTWorkload(ttt, dtype=torch.bfloat16)`)**: if you want step-by-step rounding aligned with the official code, pass `dtype=torch.bfloat16` explicitly. After every update, the fast weights are rounded in BF16 just like the official implementation, while the difference between that rounded result and the base weight is still stored in FP32 without further loss.

Both are sound engineering choices; since the paper leaves precision unspecified, pLLM defaults to FP32 mode so small updates are never lost.

### Summary {#en-summary}

1. **Core mechanism of In-Place TTT**: add no new layers; treat the MLP down-projection in selected layers as fast weights, updating them every full chunk (e.g., 1024 tokens) using targets computed by a conv1d + projection so subsequent chunks immediately read the updated weights.
2. **Running and inspecting with pLLM**: the complete implementation lives in [`pllm/workloads/inplace_ttt.py`](https://github.com/LinesHogan/pLLM/blob/main/pllm/workloads/inplace_ttt.py). `pllm.explain` reports exact per-request fast-weight VRAM before loading weights; across chunk boundaries during decode, the framework handles fast-weight updates, visibility transitions, and multi-request concurrency automatically.
3. **Connection to earlier examples**: if you read [07 Custom Residual Connections](/pllm/docs/examples/07_custom_residual/), you will recognize that the stateful memory sublayer there and In-Place TTT here share the exact same "chunk-updated, read-old-within-chunk" computational structure.

</div>

<div class="zine-lang-panel is-hidden" data-lang-panel="zh" lang="zh-CN" markdown="1">

<div id="zh-sec-1"></div>

如果你想在服务里跑一个测试时训练（Test-Time Training，TTT）模型，先会遇到一个问题：它和普通推理模型存"记忆"的地方不一样。

普通模型在推理时权重固定不变，它对上文的记忆全部存在 KV cache 里。TTT 的想法是：模型在读输入的同时，把一部分权重当作"快权重"不断更新，直接用权重记住上文。ByteDance Seed 提出的 In-Place TTT（论文 _In-Place Test-Time Training_，arXiv 2604.06169）之所以叫 "In-Place"，是因为它不往模型里加新层，而是直接把已有 MLP 的最后一个投影矩阵（down projection）当成快权重来改。

把这种模型放进推理服务里，有三个具体的难点：

1. **每个请求都要有一份独立的快权重**：我们接下来要用的 4B 检查点里，快权重一共是 6 个矩阵、149,422,080 个数，按 FP32 存放每份约 570 MiB。不同请求之间不能串改。
2. **更新发生在 forward 中途**：长 prompt 在 prefill 阶段读到一半就要更新快权重；到了逐 token 生成的 decode 阶段，凑够一块也要立刻更新。
3. **官方推理代码只支持单请求**：官方实现基于 Hugging Face `generate`，一次只能跑一个请求（源码里写着 `assert len(input_ids) == 1, "only support bs=1 for TTT inference"`，作者在 issue #4 里说明这是实现上的限制）。

这一节我们一步步做四件事：

1. 用白话把 In-Place TTT 的更新规则讲清楚；
2. 用 pLLM 自带的 `InPlaceTTTWorkload` 加载一个社区检查点；
3. 让 decode 跨过块边界，亲眼看到快权重在生成中途更新、下一个 token 立刻读到新权重；
4. 在同一张 RTX 4090 上和官方实现对比吞吐，并讲清 FP32 与 BF16 两种精度的取舍。

## 方法与检查点来源 {#zh-sec-2}

在动手跑代码之前，我们先搞清楚两件事：模型到底在改哪块权重、怎么改，以及我们手头这个检查点是从哪来的。

### 方法 {#zh-sec-3}

- **哪部分是快权重**：模型在选定层的 MLP 里，把 down projection 矩阵 $$W$$（在这个检查点里形状是 `[d_model, d_ff] = [2560, 9728]`）当作快权重。这个 36 层的检查点选了第 0、6、12、18、24、30 层（每 6 层选一个）。Attention 层以及 MLP 的另外两个矩阵在推理时保持不变。
- **怎么按块更新**：输入的 token 按固定大小的块（这个检查点是 1024 个 token 一块）分组。每读完一整块，就算一次权重更新：
  $$\Delta W = \eta \cdot V^\top Z$$
  其中 $$Z$$ 是这一块里每个 token 在 MLP 中的中间激活（也就是进入 down projection 之前的输入）；$$V$$ 是"目标"，由 MLP 的输入经过一个学出来的一维卷积（卷积核长度为 5，只在当前块内部做卷积）和一个学出来的方阵投影算出来；$$\eta$$ 是学习率（这个检查点里是 3）。算完后，快权重更新为 $$W \leftarrow W + \Delta W$$。
- **谁能看到更新后的权重**：按块严格隔离——第 $$i$$ 块的 token 只能读到前面 $$i-1$$ 块累积的更新。如果 prompt 很长，一次 prefill 过程里就会触发好几次更新；prompt 末尾凑不满一整块的 token 会先存起来，等到了 decode 阶段生成新 token 时继续往里攒。一旦凑满一块就触发更新：凑满那一块的最后一个 token 读到的还是旧权重，从下一个 token 开始读到的就是新权重。
- **请求之间互不干扰**：每个请求都从检查点里原本的 $$W$$ 出发，各自维护自己的更新。
- **推理时不需要标签**：上面提到的一维卷积和方阵投影的参数，是在"外层"继续预训练阶段学出来的（论文里基于 Qwen3-4B 先在 32K 长度上训练约 200 亿 token，再在 128K 长度上训练约 150 亿 token）。到了推理阶段，模型只管一边读 token 一边按上面的公式改 $$W$$，不需要外部标签。
- **论文与官方代码的两处差别**：第一，论文里写目标 $$V$$ 来自 token embedding，而官方代码默认（以及这个检查点的配置）用的是 MLP 的输入（`ttt_target="hidden_states"`）；第二，论文在 Qwen3-4B 的评测里把每次更新按范数裁剪到 `1e-5`，而官方代码没有做裁剪。pLLM 的 `InPlaceTTTWorkload` 跟随官方代码和检查点的实际配置：用 MLP 输入作为目标，不做裁剪。

### 检查点来源 {#zh-sec-4}

论文作者没有发布模型权重（作者在官方仓库的 issue #1 中说明，由于内部数据政策，无法发布训练数据和模型权重）。

本示例使用的是社区用户上传的检查点 `zhongweixie/inplace-ttt-qwen3-4b-32k-ttt`（固定在一个 revision，见下面加载它的代码）。模型卡说明它基于 Qwen3-4B，在 32K 长度上训练了 6000 步。我们核对过它的配置文件与张量形状（共 6 层带快权重、每层包含一个卷积和一个投影、学习率 $$\eta=3$$、块长 1024），但没有核实它的训练过程，也不在这个示例里评测它的生成质量；下面只用它来观察运行机制和测量速度。

下面我们先读取这个检查点的配置，看看快权重分布在哪几层、块长和学习率分别是多少。

```python
import os

os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"  # 只显示 vLLM 的报错，省掉启动日志

import argparse
from dataclasses import replace
from pathlib import Path

import torch
from vllm import SamplingParams

import pllm
from pllm.benchmark import throughput
from pllm.workloads.inplace_ttt import InPlaceTTTConfig, InPlaceTTTWorkload

CHECKPOINT = "zhongweixie/inplace-ttt-qwen3-4b-32k-ttt"  # 社区训练的检查点，不是论文作者发布的
REVISION = "1f6aa7ea4bac2149358c86afea614f9edbd73c54"
ttt = InPlaceTTTConfig.from_checkpoint(CHECKPOINT, revision=REVISION)
print("快权重所在的层:", [target.layer for target in ttt.targets])
print("块长:", ttt.chunk_size, " 学习率:", ttt.learning_rate)
```

```text
快权重所在的层: [0, 6, 12, 18, 24, 30]
块长: 1024  学习率: 3.0
```

上面的输出里，`InPlaceTTTConfig.from_checkpoint` 直接从检查点的 `config.json` 中读出了带快权重的层号 `[0, 6, 12, 18, 24, 30]`、块长 `1024` 和学习率 `3.0`。

## 负载长什么样，以及它要占多少显存 {#zh-sec-5}

这一步我们要弄清楚两个问题：在 pLLM 里写一个 In-Place TTT 负载需要写多少代码？在真正把 8 GB 的模型权重加载进 GPU 之前，怎么先算清楚这些快权重会花掉多少显存？

`InPlaceTTTWorkload` 的实现在 `pllm/workloads/inplace_ttt.py`，正文大约 60 行。它的核心是一个带 `@pllm.train_in_forward` 装饰器的方法：每次拿到一块的 MLP 输入（`targets`）、MLP 中间激活（`inputs`）以及当前累积的快权重增量 `delta`，先算卷积和投影得到 $$V$$，再算出这一块的权重更新，最后返回更新后的 `delta` 和这一块输出所需的修正项。下面是简化后的摘录：

```python
@pllm.train_in_forward(
    visibility="exact",
    state="dense_fast_weights",
    targets=targets,                                 # def targets(self): return self.config.targets
    weight="delta",
    chunk=chunk_size,                                # def chunk_size(self): return self.config.chunk_size
    tokens="all",
)
def step(self, delta=pllm.parameter("dense_fast_weights", "delta"), inputs=pllm.operand("input"),
         targets=pllm.operand("target"), conv=pllm.parameter("target_parameters", "conv"),
         proj=pllm.parameter("target_parameters", "proj")):
    values = conv1d(targets, conv) @ proj            # V：块内卷积，再投影
    update = (values.T @ inputs) * learning_rate     # ΔW = η · Vᵀ Z
    return delta + update, inputs @ delta.T          # 新的 delta；这一块读到的修正
```

工作负载只负责写出上面的数学公式，不需要写一行分页管理、连续批处理调度或自定义 CUDA 算子。正是从这二十多行纯张量数学出发，pLLM **全自动**实现了五件官方仓库（`ByteDance-Seed/In-Place-TTT`）无法做到的事：

1. **打破 `bs=1` 限制，支持连续批处理与多请求隔离**：官方推理代码直接复制底座模型的 `down_proj` 原地改写，只能在入口处硬编码 `assert len(input_ids) == 1`；pLLM 自动把 `RequestState` 管成分页状态池，让不同长度、处于不同块进度的并发请求在同一次 forward 中各自读写自己的 `delta`，请求结束或抢占时自动回收显存页。
2. **任意切分的 Chunked Prefill 与跨阶段精确块边界**：官方代码要求整段 prompt 一次性送入并在单个循环内切块；pLLM 自动为未凑满 1024 个 token 的尾行维护跨步缓冲区，不论调度器把长 prompt 切成多大碎片、也不论第 1024 个 token 落在 prefill 还是 decode 中途，都会在凑满整块的瞬间触发更新，保证后续块严格读到前序完整块的更新（`visibility="exact"`）。
3. **Decode 整图捕获与设备端条件块更新（零主机同步）**：手写块边界分支极易引发 CPU–GPU 同步或打断 CUDA graph；pLLM 自动把稳态 decode 录进整模型 CUDA graph，并用设备端条件节点包裹 1/1024 步才触发的整块更新，稳态每步 0 次主机阻塞。
4. **非边界步零拷贝页共享与分页流式读取**：6 层 FP32 快权重每份重达约 570 MiB，若每步 decode 都做写时复制和逐请求收集，每步要白白多搬近 3 GB 显存；pLLM 静态分析 `step` 的计算图后，自动识别出 1023/1024 个非边界 decode 步不会修改 `delta`（直接共享只读页、0 字节复制），原地借用 `ModelCheckpoint` 里的只读卷积与投影权重，并把 `inputs @ delta.to(inputs.dtype).T` 自动降级为直接从分页状态池流式读取的内核（自动跳过未满批的空闲槽位）。
5. **单份冻结底座 + FP32 增量累积**：所有请求共享同一份只读 BF16 底座模型（vLLM 加载底座时自动跳过 `ModelCheckpoint` 接管的 `ttt_conv` / `ttt_proj`），每个请求单独用 FP32 累积 `delta`，既省掉多份底座副本，又避免微小更新被 BF16 尾数吞掉。

在启动引擎加载 8 GB 的模型权重之前，我们先调用 `pllm.explain`（详见[写代码时就看到成本](/pllm/docs/guides/explain/)），看看单并发（`max_num_seqs=1`）下这个负载需要多少显存、为什么会选择对应的执行模式。

```python
small = pllm.ModelConfig(CHECKPOINT, revision=REVISION, max_model_len=2048, max_num_seqs=1,
                         max_num_batched_tokens=2048, gpu_memory_utilization=0.4)
plan = pllm.explain(InPlaceTTTWorkload(ttt), small)
print("eager 原因:", plan.eager_reason)
for state in plan.states:
    print(f"{state.name:20s} {state.scope:8s} 每份 {state.bytes_per_root / 2**20:6.1f} MiB × {state.roots} 份")
```

```text
eager 原因: 
dense_fast_weights   request  每份  714.0 MiB × 4 份
target_parameters    instance 每份   75.1 MiB × 1 份
```

我们逐行看一下 `pllm.explain` 给出的结果：

- **`eager 原因` 为空**：默认设置下，decode 中途的精确块边界更新也在 CUDA graph 里运行；这篇后面的几次运行和吞吐数字都开着 CUDA graph。关闭引擎会释放它的模型、请求状态和 graph，可以在同一进程中继续创建引擎。
- **`dense_fast_weights`（每份 `714.0 MiB × 4 份`）**：这是每个请求独占的状态（`request` 作用域）。每份 `714.0 MiB` 里，包含 6 层 FP32 的 `delta` 矩阵（约 570 MiB），外加用来暂存还没凑满一整块的输入行的缓冲区。之所以给 1 个请求预留 4 份，是因为在更新时内存中需要同时保留正在写入的新版本和正在被读取的旧版本。
- **`target_parameters`（每份 `75.1 MiB × 1 份`）**：这是那 6 层的一维卷积和方阵投影参数，属于实例级状态（`instance` 作用域），所有请求共享这一份，占约 75 MiB。

## 验证 decode 中途的跨块更新 {#zh-sec-6}

看完了静态的显存账单，接下来我们验证一个关键行为：**当快权重的更新点刚好落在 decode 阶段中途时，到底哪一个 token 读旧权重、哪一个 token 开始读新权重？**

为了亲眼看到分界线，我们设计一个小实验：

- 文本取自 Python 标准库 `argparse` 模块的源码（每台装了 Python 的机器上都有这份文件）。
- 我们截取前 **1023 个 token** 作为 prompt（比块长 1024 刚好少 1 个 token），然后用贪心解码连续生成 5 个 token。
- 我们跑两组对比：第一组（`TTT`）使用正常的块长 1024；第二组（`不更新`）作为对照组，把块长改成 2048。因为整条序列加起来只有 $$1023 + 5 = 1028$$ 个 token，凑不满 2048，所以对照组的快权重从头到尾不会更新。

下面运行这组对比，打印两组的更新次数和每一步生成 token 的 logprob。

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, revision=REVISION)
text = Path(argparse.__file__).read_text()  # 一段每台机器上都有的长文本：argparse 模块的源码
ids = tokenizer(text, add_special_tokens=False).input_ids
print("长文本共", len(ids), "个 token")

prompt = [{"prompt_token_ids": ids[:1023]}]  # 比一块（1024）少一个 token
five = SamplingParams(temperature=0.0, max_tokens=5, ignore_eos=True, logprobs=0)
runs = {}
for name, config in (("TTT", ttt), ("不更新", replace(ttt, chunk_size=2048))):
    with pllm.make_llm(small, workload=InPlaceTTTWorkload(config)) as llm:
        sample = llm.generate(prompt, five, use_tqdm=False)[0].outputs[0]
        runs[name] = [(token, step[token].logprob) for token, step in zip(sample.token_ids, sample.logprobs)]
        print(f"{name}: 快权重更新了 {llm.diagnostics().update_count} 次")

print("两组生成的 token 相同:", [t for t, _ in runs["TTT"]] == [t for t, _ in runs["不更新"]])
print("\n生成的 token   算出它的位置   TTT 的 logprob   不更新的 logprob   差")
for k, ((token, a), (_, b)) in enumerate(zip(runs["TTT"], runs["不更新"])):
    print(f"{tokenizer.decode(token)!r:14s} {1022 + k:6d}      {a:10.4f}       {b:10.4f}      {a - b:+.4f}")
```

```text
长文本共 19736 个 token
TTT: 快权重更新了 6 次
不更新: 快权重更新了 0 次
两组生成的 token 相同: True

生成的 token   算出它的位置   TTT 的 logprob   不更新的 logprob   差
' _'             1022         -0.4400          -0.4400      +0.0000
'get'            1023         -1.9058          -1.9058      +0.0000
'_default'       1024         -2.2565          -2.3191      +0.0626
'_action'        1025         -1.8538          -1.8214      -0.0324
'(action'        1026         -1.7855          -1.7124      -0.0732
```

输出把跨块更新的时机展示得很清楚：

- **更新次数**：`TTT` 组一共更新了 **6 次**，对应 6 个快权重层 × 1 个凑满的块；对照组（`不更新`）因为块长是 2048，全程更新了 **0 次**。
- **前两个生成的 token（logprob 差为 `+0.0000`）**：
  - 第 1 个生成的 token `' _'` 是由位置 `1022`（也就是 prompt 的最后一个 token，第 1023 个 token）算出来的，此时还没凑满 1024，两边的 logprob 都是 `-0.4400`。
  - 第 2 个生成的 token `'get'` 是由位置 `1023`（全序列的第 1024 个 token）算出来的。位置 `1023` 的到来正好凑满了第一块（1024 个 token）并触发快权重更新，但**凑满这一块的最后一个 token 自己读到的仍然是旧权重**，所以两边的 logprob 依然完全一致（都是 `-1.9058`，差为 `+0.0000`）。
- **从第 3 个生成的 token 起（logprob 开始出现差异）**：
  - 第 3 个 token `'_default'` 由位置 `1024`（第二块的第 1 个位置）算出，它读到了刚刚更新过的快权重，logprob 从不更新时的 `-2.3191` 变成了 `-2.2565`（差 `+0.0626`）。
  - 随后的位置 `1025` 和 `1026` 同样读到了新权重，logprob 差分别为 `-0.0324` 和 `-0.0732`。（在这 5 步贪心解码里，虽然 logprob 变了，两边选出来的 token 相同：输出里的 `两组生成的 token 相同: True`。）

这就是论文里定义的块级可见性。它在 vLLM 的逐 token decode 过程中自动发生，而我们在编写 `InPlaceTTTWorkload` 时并没有手写"什么时候攒够一块、什么时候切换可见权重"的调度代码。

确认了行为是对的，接下来我们看大家关心的另一个问题：把这套机制放进 pLLM 里跑，和官方的 Hugging Face 实现相比速度怎么样？

```python
def measure(llm, length: int, batch: int) -> float:
    """batch 个同样的 prompt（前 length 个 token），各贪心生成 256 个 token 的吞吐。"""
    prompts = [{"prompt_token_ids": ids[:length]}] * batch
    return throughput(llm, prompts, SamplingParams(temperature=0.0, max_tokens=256, ignore_eos=True))


long = pllm.ModelConfig(CHECKPOINT, revision=REVISION, max_model_len=16640, max_num_seqs=1,
                        max_num_batched_tokens=8192, enable_chunked_prefill=True, gpu_memory_utilization=0.5)
for dtype in (torch.float32, torch.bfloat16):
    with pllm.make_llm(long, workload=InPlaceTTTWorkload(ttt, dtype=dtype)) as llm:
        for length in (4096, 16384):
            print(f"pllm {str(dtype)[6:]:8s} 1 个请求  prompt {length:5d}: {measure(llm, length, 1):6.2f} token/s")
```

```text
pllm float32  1 个请求  prompt  4096:  73.02 token/s
pllm float32  1 个请求  prompt 16384:  47.97 token/s
pllm bfloat16 1 个请求  prompt  4096:  61.02 token/s
pllm bfloat16 1 个请求  prompt 16384:  42.32 token/s
```

```python
# 4 个请求各一份快权重（含在途版本共 10 份），再加上 CUDA graph 的私有池，24 GiB 的卡要给 vLLM 少留一点显存。
four = replace(long, max_model_len=4416, max_num_seqs=4, max_num_batched_tokens=4416, gpu_memory_utilization=0.48)
with pllm.make_llm(four, workload=InPlaceTTTWorkload(ttt)) as llm:
    print(f"pllm float32  4 个请求  prompt  4096: {measure(llm, 4096, 4):6.2f} token/s")
```

```text
pllm float32  4 个请求  prompt  4096: 171.56 token/s
```

## 与官方实现对比吞吐 {#zh-sec-7}

这一步我们要回答的问题是：在相同的硬件和相同的输入输出长度下，pLLM 的吞吐和官方实现相比如何？当从单请求变成多请求并发时，吞吐能提升多少？

我们先对齐两边的测试协议：

- **硬件与输入**：同一张 RTX 4090；使用同一段 prompt token（上面 `argparse` 源码的前 4096 个或前 16384 个 token）。
- **解码设置**：贪心解码、不在 EOS 处提前停止、固定生成 256 个 token。
- **计时方式**：先跑 1 轮预热，再跑 3 轮计时取中位数；吞吐定义为 `生成的 token 总数 / 整次 generate 的耗时（含 prefill）`。pLLM 这边用的 `pllm.benchmark.throughput` 就是这样计时的。

官方实现来自仓库 `ByteDance-Seed/In-Place-TTT` 的推理代码，基于 Hugging Face `generate`，使用 BF16 精度和 FlashAttention 2。因为它依赖另一套 Python 环境（`torch 2.8.0`、`transformers 4.57.3`），我们在同一台机器上单独运行它（两边在同一张 RTX 4090 上分别运行）。

官方实现的测法（在它自己的环境里运行）：

```python
from inference_model.hf_qwen3.modeling_qwen3 import Qwen3ForCausalLM  # 官方仓库里的模型类

model = Qwen3ForCausalLM.from_pretrained(checkpoint, dtype=torch.bfloat16,
                                         attn_implementation="flash_attention_2").eval().cuda()
model.generate(input_ids=prompt, attention_mask=torch.ones_like(prompt), max_new_tokens=256,
               min_new_tokens=256, do_sample=False, eos_token_id=None, pad_token_id=151643, use_cache=True)
```

把刚才两格跑出来的数字和官方实现的测量结果放在一起对比（单位均为 `token/s`；pLLM 用 vLLM 0.32.0、开着 CUDA graph；官方实现在同一天、同一张卡上按同样协议测得）：

| | 1 个请求，prompt 4096 | 1 个请求，prompt 16384 | 4 个请求，prompt 4096 |
| --- | --- | --- | --- |
| 官方 HF 实现（BF16） | 54.79 | 34.89 | 不支持（硬编码 `bs=1`） |
| pLLM，float32 快权重 | 73.02 | 47.97 | 171.56（4 个请求合计） |
| pLLM，bfloat16 快权重 | 61.02 | 42.32 | 未测 |

我们可以从表格里读出几条直接的结论：

- **单请求下的对比**：pLLM 默认的 `float32` 模式在 prompt 4096 时比官方快约 33%（`73.02` 对 `54.79`），在 prompt 16384 时快约 37%（`47.97` 对 `34.89`）；严格复现官方 BF16 舍入行为的 `bfloat16` 模式在两档长度上也分别快约 11%（`61.02`）和 21%（`42.32`）。为什么 `float32` 反而比 `bfloat16` 明显更快？因为在 `float32` 模式下，pLLM 自动识别出 `inputs @ delta.to(inputs.dtype).T` 是纯线性读出，在非块边界步直接从只读共享的 FP32 页流式算出残差修正；而在 `bfloat16` 模式下，为了逐位模拟官方把更新直接加到底座上的 BF16 舍入效果，需要先将底座权重和 `delta` 合成并舍入成 BF16，再做两次矩阵乘。
- **多请求并发**：当 4 个请求一起跑时，pLLM（`float32` 快权重）的合计吞吐达到 `171.56 token/s`，是官方单请求实现（`54.79 token/s`）的 **3.13 倍**、pLLM 单请求模式（`73.02 token/s`）的 **2.35 倍**；而官方实现由于 `bs=1` 的架构限制无法并发服务多个请求。4 个请求各一份快权重（含在途版本共 8 份，每份 714 MiB），再加上 CUDA graph 的私有池，在 24 GiB 的卡上把 vLLM 的显存比例设到 0.48 即可装下。
- **关于生成结果的逐位对比**：pLLM 生成的 token 序列与官方实现并不逐位相同。这是因为两边的矩阵乘切分方式和浮点舍入顺序不同，当贪心解码遇到两个候选 token 的得分非常接近时，微小的数值差异就会让后续走向不同的 token。

## 精度选择：快权重该用 FP32 还是 BF16？ {#zh-sec-8}

在上面的测速里，我们看到了 `float32` 和 `bfloat16` 两种快权重模式。最后一节我们来搞清楚：这两种模式在数值上到底有什么区别？平时该选哪一种？

论文里并没有说明推理时快权重应该用什么浮点精度存放。看官方仓库的代码可以发现，它是把原本的 down projection 矩阵复制一份作为快权重，每算出一块的 $$\Delta W$$ 就直接加到这份矩阵上。因此，当模型以 BF16 加载时，这份快权重也是 BF16，并没有单独保留一份 FP32 的主副本。

这种做法有一个数值上的隐患：**把一个很小的更新量直接加到一个数值较大的 BF16 矩阵上，大部分更新会被 BF16 的尾数精度直接舍入掉（大数吃小数）。**

我们在该检查点第 0 层的真实 down projection 矩阵上做过一个舍入测试：如果生成一个范数为 `1e-5` 的更新矩阵（也就是论文评测 Qwen3-4B 时每次更新被裁剪到的大小），直接用 BF16 加到第 0 层的矩阵上，在全部 `24,903,680` 个元素里，只有 `472` 个元素的值发生了变化，实际生效的改变量只有预期值的约 `0.97%`。（说明一下：这只是一个用来展示浮点舍入效应的数值演示，不是模型质量实验；而且我们手头这个社区检查点没有开裁剪，单次更新的幅度比 `1e-5` 大得多。）

为了避免小更新被底座权重吃掉，pLLM 提供了两种工程选择：

- **默认用 FP32 单独存增量（`dtype=torch.float32`）**：底座权重保持 BF16 不动，每个请求用 FP32 单独存放累积的增量 `delta`，前向计算时在底座输出上加上 `inputs @ delta`。这样即使单次更新很小，也能在 FP32 的 `delta` 里累积起来，不会被舍入掉。
- **对齐官方舍入的 BF16 模式（`InPlaceTTTWorkload(ttt, dtype=torch.bfloat16)`）**：如果你想和官方代码逐步对齐舍入方式，可以显式传入 `dtype=torch.bfloat16`。在这个模式下，每次更新后快权重都会按照与官方一致的方式做 BF16 舍入；同时，这个舍入后的结果与底座权重之间的差值仍然用 FP32 存放，不会再额外多丢精度。

这两种方式都是合理的工程选择；由于论文没有规定精度，pLLM 默认选择保留小更新的 FP32 模式。

### 小结 {#zh-sec-9}

1. **In-Place TTT 的核心机制**：不加新层，把选定层 MLP 的 down projection 当作快权重，每读满一块（如 1024 个 token）就用卷积+投影算出的目标更新一次权重，后续块立刻读到更新后的权重。
2. **用 pLLM 运行与观测**：完整实现在 [`pllm/workloads/inplace_ttt.py`](https://github.com/LinesHogan/pLLM/blob/main/pllm/workloads/inplace_ttt.py)。借助 `pllm.explain` 可以在加载权重前看到每个请求的快权重要占多少显存；在 decode 跨过块边界时，框架会自动完成快权重更新与可见性切换，并支持多请求并发推理。
3. **与其他教程的联系**：如果你读过上一篇 [07 改写残差连接](/pllm/docs/examples/07_custom_residual/)
，会发现那里的记忆子层和这里的 In-Place TTT 是同一种"按块更新、块内只读旧状态"的计算结构。

</div>

</div>

<div class="zine-doc-footer">
  <a class="zine-pill-link" data-i18n-zh="← 返回 pLLM 项目与文档中心" data-i18n-en="← Back to pLLM Project &amp; Docs" href="/pllm/#docs">← Back to pLLM Project &amp; Docs</a>
  <a class="zine-pill-link" href="https://github.com/LinesHogan/pLLM/blob/main/doc/examples/08_inplace_ttt.ipynb" target="_blank" rel="noopener">GitHub &middot; doc/examples/08_inplace_ttt.ipynb ↗</a>
</div>

<script defer src="/assets/js/bilingual-post.js"></script>
