---
layout: page
title: "10 Custom Attention: Monitoring, Read-Only Top-K, and KV Layouts"
description: "pLLM Documentation · Algorithm Family Examples (06–13)"
permalink: /pllm/docs/examples/10_custom_attention/
nav: false
---

<link rel="stylesheet" href="{{ '/assets/css/pllm-hub.css' | relative_url | bust_file_cache }}">

<div class="zine-dispatch" data-title-zh="10 自定义 attention：监控、只读 top-K、改 KV 布局" data-title-en="10 Custom Attention: Monitoring, Read-Only Top-K, and KV Layouts" data-desc-zh="pLLM 文档 · 算法族示例与实测 (06–13)" data-desc-en="pLLM Documentation · Algorithm Family Examples (06–13)">
  <div class="zine-dispatch-links">
    <a class="zine-pill-link primary" data-i18n-zh="← 返回 pLLM 文档中心" data-i18n-en="← pLLM Docs Hub" href="/pllm/#docs">← pLLM Docs Hub</a>
    <a class="zine-pill-link" data-i18n-zh="文档导航地图" data-i18n-en="Docs Roadmap" href="/pllm/docs/">Docs Roadmap</a>
    <span class="zine-lang-badge" data-i18n-zh="可执行教程 (.ipynb)" data-i18n-en="Notebook (.ipynb)">Notebook (.ipynb)</span>
  </div>
  <a class="zine-pill-link" data-i18n-zh="在 GitHub 打开 .ipynb ↗" data-i18n-en="Open .ipynb on GitHub ↗" href="https://github.com/LinesHogan/pLLM/blob/main/doc/examples/10_custom_attention.ipynb" target="_blank" rel="noopener">Open .ipynb on GitHub ↗</a>
</div>

<div id="markdown-content">

<div class="zine-lang-panel" data-lang-panel="en" lang="en" markdown="1">

<div id="en-sec-1"></div>

After reading Examples 00, 06, and 07, if you are interested in attention-level inference-time algorithms—such as attention monitoring (measuring where attention mass lands), training-free sparse reads (such as Quest page-level selection), or altering KV storage layouts (sliding window, prompt sink plus recent window, or cross-layer KV sharing)—they share one property: none of them modify q/k/v projections; they only change **how a layer reads KV** or **how KV is stored**. In vLLM, however, attention is a fused kernel and KV is paged and managed by the scheduler; changing either usually requires editing the attention backend or model code.

`@pllm.custom_attention` replaces only the attention read at the specified layers: native projections and KV writes proceed as usual, while your method receives the current layer's (RoPE-applied) `q` and a read-only paged KV view `kv`. You invoke the native kernel via `kv.attend(...)` (optionally with a window or per-head position selection), return the attention output, and can optionally return an extra `aux` tensor for downstream methods to read. Meanwhile, `keep=`, `window=`, `sink=`, and `source=` on the same decorator declare how KV is stored, leaving actual allocation and reclamation to vLLM's own sliding-window, R-SWA, or KV-sharing mechanisms.

Before starting, keep these facts, limitations, and costs in mind:
- Qwen2.5-0.5B was not trained with these layouts. Cross-layer sharing and even-layer sliding windows corrupt its answers here (see Step 3); this example demonstrates wiring, checks, and costs, not generation quality.
- Read-only top-K (selective reading) saves data read volume, not KV VRAM. Here page scoring is written directly in PyTorch, reading every page's keys at every layer and step, which drops 7B throughput to 13% of native: it illustrates semantics, not a fast implementation. Real speedups require maintaining per-page min/max incrementally and writing a custom kernel (see Example 09).
- Score-based physical KV memory eviction (such as H2O or SnapKV) is unsupported in pLLM (`Score-based memory eviction is unsupported`). vLLM assumes each request's KV is a contiguous prefix starting from position 0, deriving positions and block tables from the number of computed tokens with append-only block tables; punching holes in the middle would break those underlying invariants.
- Execution constraints: requires single-GPU (`tensor_parallel_size=1`), the FlashAttention 2 backend with unquantized KV, and prefix caching disabled; static layouts cannot be combined with `PositionState` or `@pllm.custom_forward`.

This example walks through three steps (measured on RTX 4090, vLLM 0.32.0, bf16; Qwen2.5-0.5B-Instruct for behavior, and Qwen2.5-7B-Instruct at the end of Step 3 for KV accounting and throughput):
1. **Monitoring**: read out the fraction of attention mass falling on recent positions without altering model output.
2. **Read-only top-K**: implement Quest-style page selection and compare it against reading only recent positions under the same budget.
3. **KV layouts**: compare how much KV VRAM sliding window, prompt-sink plus window, and cross-layer sharing each save, along with their throughput.

To observe how different read patterns and layouts behave over long contexts, we first prepare a ~1500-token test prompt: bury `"The secret code for the blue door is 7361."` inside a museum passage and ask the model for the code. Let's run vanilla vLLM first as our baseline.

```python
import os

os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"  # Show only vLLM errors to suppress startup logs

from collections import defaultdict
from dataclasses import dataclass, replace

import torch
from transformers import AutoTokenizer
from vllm import SamplingParams

import pllm
from pllm import testing
from pllm.benchmark import throughput

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
# 2048-token context; chunked prefill up to 256 tokens/step; 1 GiB pLLM CUDA graph capture budget
# (ExecutionOptions apply only to engines with a workload; the same config also launches vanilla vLLM.)
CONFIG = pllm.ModelConfig(MODEL, max_model_len=2048, max_num_seqs=8, max_num_batched_tokens=256,
                          enable_chunked_prefill=True, gpu_memory_utilization=0.4, seed=0,
                          execution=pllm.ExecutionOptions(cuda_graph_pool_budget_bytes=1 << 30))
```

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL)
filler = ("The museum opens at nine in the morning and closes at six in the evening. Visitors can buy tickets at the "
          "front desk or online. The cafe on the second floor serves coffee, tea and sandwiches. ")
passage = filler * 6 + "The secret code for the blue door is 7361. " + filler * 30
question = "What is the secret code for the blue door? Answer with the number only."
prompt = tokenizer.apply_chat_template([{"role": "user", "content": passage + "\n\n" + question}], tokenize=False,
                                       add_generation_prompt=True)
greedy = SamplingParams(max_tokens=12, temperature=0.0)
print("prompt 长度:", len(tokenizer(prompt).input_ids), "token")


def answer(llm) -> str:
    return llm.generate([prompt], greedy, use_tqdm=False)[0].outputs[0].text


with pllm.make_llm(CONFIG) as plain:
    print("原生 vLLM 的回答:", repr(answer(plain)))
```

```text
prompt 长度: 1500 token
原生 vLLM 的回答: '7361'
```

Vanilla vLLM answers `'7361'` on this 1500-token prompt. Now let's move to Step 1: **How do we monitor what fraction of a layer's attention falls on recent positions without changing the model's output?**

Here we want to extract an intermediate statistic during attention computation and export it to the host. `kv.attend(q, *, window=0, topk=None)` returns `(output [rows, heads, head_dim], float32 LSE [rows, heads])`; without selection arguments it invokes the native causal kernel, where LSE is the log-sum-exp (the log of the softmax normalization constant).

The `RecentMass` workload below attaches the custom attention method `split` only at layer 12:
- First, call `kv.attend(q)` for the native read to get the normal `out` and full-sequence `lse`;
- Next, call `kv.attend(q, window=64)` to read only the most recent 64 positions (`window` counts backward from the current token inclusive), obtaining the local `near` (local LSE);
- Subtract and exponentiate `(near - lse).exp()` to compute the attention share falling on the most recent 64 positions, average across heads, and return it as the second output (`aux`);
- Return the native `out` as the first value, so model behavior stays unchanged.

Then, `report` decorated with `@pllm.export` reads this `aux` tensor from layer 12 via `pllm.attention(12, "aux")` (inside an export method it receives the sampled rows; passing `"lse"` would read the LSE of that layer's first `kv.attend`), exports it to the host every step, and `on_export` collects it per request (`prompt_index`).

We run two requests together for contrast: the 1500-token museum prompt and the short `"What is 2 + 3?"`.

```python
@dataclass(frozen=True)
class RecentMass(pllm.Workload):
    recent: int = 64

    @pllm.custom_attention(layers=(12,))
    def split(self, q: torch.Tensor, kv: pllm.model.PagedKV) -> tuple[torch.Tensor, torch.Tensor]:
        out, lse = kv.attend(q)                     # Native read, also capturing log-sum-exp
        _, near = kv.attend(q, window=self.recent)  # Read only the most recent `recent` positions
        return out, (near - lse).exp().mean(-1, keepdim=True)  # aux: fraction of attention on recent positions, averaged over heads

    @pllm.export(name="recent_mass", max_rows=8, max_cols=1)
    def report(self, share: torch.Tensor = pllm.attention(12, "aux")) -> torch.Tensor:
        return share
```

```python
shares: dict[int, list[float]] = defaultdict(list)


@dataclass(frozen=True)
class RecordedMass(RecentMass):
    def on_export(self, event: pllm.io.TraceEvent) -> None:
        for identity, row in zip(event.identities, event.tensor().tolist()):
            shares[identity.prompt_index].append(round(row[0], 3))


short = tokenizer.apply_chat_template([{"role": "user", "content": "What is 2 + 3?"}], tokenize=False,
                                      add_generation_prompt=True)
with pllm.make_llm(CONFIG, workload=RecordedMass()) as llm:
    outputs = llm.generate([prompt, short], greedy, use_tqdm=False)
    print("eager 原因:", repr(llm.diagnostics().eager_reason))
for index, output in enumerate(outputs):
    print(f"prompt {index}: {output.outputs[0].text!r}")
    print("    最近 64 个位置的注意力占比:", shares[index])
```

```text
eager 原因: ''
prompt 0: '7361'
    最近 64 个位置的注意力占比: [0.633, 0.588, 0.666, 0.757, 0.586]
prompt 1: '2 + 3 equals 5.'
    最近 64 个位置的注意力占比: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
```

Look at the output: `eager_reason` is `''`, confirming this monitoring workload runs inside CUDA graphs. The long prompt (`prompt 0`) still answers `'7361'`; at each generated token, the most recent 64 positions at layer 12 receive `0.586` to `0.757` of the attention mass, with the rest going to earlier positions. The short prompt (`prompt 1`) answers `'2 + 3 equals 5.'` with `1.0` at every step—because its prompt is only 37 tokens, even with the generated tokens the total length stays under 64 tokens. At the end, `check_on_gpu` compares this monitoring workload against the vanilla model's full-vocabulary distribution.

Next, Step 2: **When each layer and head may only read a fixed small budget of positions, how does page-based top-K selection (Quest) compare to simply reading the most recent positions?**

In plain terms, Quest (Tang et al., 2024) works like this: in vLLM, KV is stored in pages (16 tokens per page here). For each page, record the per-dimension minimum and maximum across all keys on that page; given the current query vector $$q$$, $$\sum_d \max(q_d \cdot \text{min}_d, q_d \cdot \text{max}_d)$$ upper-bounds the maximum possible value of $$q \cdot k$$ on that page. Each attention head picks the pages with the highest upper bounds and attends only to those pages.

In the code below:
- `page_scores` computes the per-row, per-head, per-page upper bound directly in PyTorch from `kv.key_cache` and `kv.block_table` (Qwen2.5-0.5B uses GQA with 14 query heads grouped over 2 KV heads).
- `QuestRead` attaches to layers 2 through 23 (layers 0 and 1 read all positions normally). It masks unwritten pages to `-inf`, selects the top `pages` per head, converts them to logical token positions, and passes them to `kv.attend(q, topk=...)` (`topk` accepts shape `[rows, k]` or `[rows, heads, k]`, padded with `-1` where fewer positions exist; budgets may differ across heads and layers).
- As a control, when `recent_only=True`, it calls `kv.attend(q, window=128)` under the same budget (8 pages = 128 tokens) to read only the most recent 128 positions. We also define `NativeRead`, which reads all positions at every layer as both a control here and the baseline for KV layouts in the next step.

The next cell runs `pllm.explain` and contract checks on CPU to inspect read cost estimates, followed by testing the secret-code question on GPU.

```python
def page_scores(q: torch.Tensor, kv: pllm.model.PagedKV) -> torch.Tensor:
    """Quest: upper-bound the maximum q·k on each page using per-dimension min/max of its keys."""
    keys = kv.key_cache[kv.block_table[kv.requests].long()].float()  # [rows, pages, pos_in_page, kv_heads, head_dim]
    low, high = keys.amin(2), keys.amax(2)                           # [rows, pages, kv_heads, head_dim]
    rows, heads, dim = q.shape
    g = q.float().reshape(rows, low.shape[2], heads // low.shape[2], dim)  # GQA: group by KV head
    bound = (torch.einsum("rkgd,rpkd->rkgp", g.clamp(min=0), high)
             + torch.einsum("rkgd,rpkd->rkgp", g.clamp(max=0), low))
    return bound.reshape(rows, heads, -1)                            # [rows, heads, pages]


@dataclass(frozen=True)
class NativeRead(pllm.Workload):
    """Control: every layer reads all positions and stores all KV as usual."""

    @pllm.custom_attention()
    def read(self, q: torch.Tensor, kv: pllm.model.PagedKV) -> torch.Tensor:
        return kv.attend(q)[0]


@dataclass(frozen=True)
class QuestRead(pllm.Workload):
    pages: int = 8          # Pages read per head (16 tokens per page)
    recent_only: bool = False  # Control: same budget, reading only recent positions
    depth: int = 24         # Model layers: 24 for 0.5B; pass 28 for the 7B throughput run at the end of Step 3

    def later_layers(self) -> tuple[int, ...]:
        return tuple(range(2, self.depth))

    @pllm.custom_attention(layers=later_layers)
    def read(self, q: torch.Tensor, kv: pllm.model.PagedKV) -> torch.Tensor:
        block = kv.key_cache.shape[1]
        if self.recent_only:
            return kv.attend(q, window=self.pages * block)[0]
        scores = page_scores(q, kv)
        page = torch.arange(scores.shape[-1], device=q.device)
        scores = scores.masked_fill(page * block > kv.positions[:, None, None], -float("inf"))  # Unwritten pages
        chosen = scores.topk(min(self.pages, scores.shape[-1]), dim=-1).indices            # [rows, heads, pages]
        tokens = (chosen[..., None] * block + torch.arange(block, device=q.device)).flatten(2)
        return kv.attend(q, topk=tokens)[0]  # Each head reads only its selected positions
```

```python
for workload in (NativeRead(), QuestRead()):
    for line in str(pllm.explain(workload, CONFIG)).splitlines():
        if line.startswith("attention layer 2:"):
            print(f"{type(workload).__name__:10s}", line)
print(testing.check(QuestRead(), CONFIG))
```

```text
NativeRead attention layer 2: custom `read`; all positions natively; 7340032 FLOP/token, <= 1048576 KV bytes/token [estimate at max_model_len; per-head budget upper bound]
QuestRead  attention layer 2: custom `read`; topk <= 128/head; 458752 FLOP/token, <= 458752 KV bytes/token [estimate at max_model_len; per-head budget upper bound]
status  contract / reason
PASS    tensor_parallel_size=1: admission
PASS    compile
PASS    custom-attention method math / CPU paged KV schedules
PASS    row permutation
PASS    request isolation
PASS    mixed prefill/decode
PASS    chunking / tokens=
PASS    preempted
SKIP    publish_lag: no row method reads versioned state
SKIP    neutral: pass neutral= with the author's disabled settings
```

```python
for workload in (QuestRead(pages=8), QuestRead(pages=8, recent_only=True), QuestRead(pages=2)):
    with pllm.make_llm(CONFIG, workload=workload) as llm:
        print(f"{workload}: {answer(llm)!r}")
```

```text
QuestRead(pages=8, recent_only=False, depth=24): '7361'
QuestRead(pages=8, recent_only=True, depth=24): ''
QuestRead(pages=2, recent_only=False, depth=24): '9'
```

First, examine the CPU estimates from `explain` (computed at `max_model_len=2048`):
- When `NativeRead` reads all positions, layer 2 reads up to `1048576` KV bytes/token at `7340032` FLOP/token;
- When `QuestRead` reads 128 positions per head (`topk <= 128/head`), compute drops to `458752` FLOP/token, yet each token still reads up to `458752` KV bytes/token.

Why is the KV read upper bound ~44% (`458752 / 1048576 ≈ 0.44`) when 128 positions is only 1/16 of 2048? Because this model's 14 query heads share 2 KV heads; when each query head chooses its own positions, the same KV head's data can be read separately by multiple query heads. `explain` surfaces that cost right at authoring time.

Next, `testing.check` on CPU: the custom attention logic remains invariant across paged KV scheduling—`row permutation`, `request isolation`, `mixed prefill/decode`, `chunking / tokens=`, and `preempted` all `PASS`.

Now look at the GPU answers:
- `QuestRead(pages=8, recent_only=False)` picking 8 pages (128 tokens) per head still answers `'7361'`;
- `QuestRead(pages=8, recent_only=True)` under the same 128-token budget outputs `''` (the first generated token is EOS) because the secret-code sentence lies outside the most recent 128 positions;
- `QuestRead(pages=2, recent_only=False)` cutting the budget to 2 pages per head answers `'9'`.

Note that this is a functional demo on a single prompt, not a quality benchmark. Also remember: read-only top-K reduces read traffic, not KV VRAM.

Now for Step 3: **If we want to actually reduce per-request KV VRAM, how do we declare KV storage layouts? How much VRAM does each layout save, and how does it affect model output?**

Below we compare four workloads whose method body is simply `return kv.attend(q)[0]`, differing only in the storage layout declared on `@pllm.custom_attention` (window trimming and sharing are handled by the storage layout itself):
1. `NativeRead`: keeps all KV by default.
2. `LocalGlobal`: even layers declare `keep="window", window=256`, retaining only the most recent 256 tokens of KV while older blocks are reclaimed automatically by vLLM's sliding-window mechanism; odd layers keep all KV.
3. `PromptPlusWindow`: even layers add `sink=True` to `keep="window", window=256`, retaining the entire prompt plus the most recent 256 generated tokens (corresponding to vLLM's R-SWA); odd layers stay unchanged.
4. `CrossLayerShared`: odd layers declare `keep="none", source=1`, allocating no KV of their own and reading directly from the layer 1 below (i.e., the preceding even layer).

Without starting the GPU engine, `pllm.explain` computes each request's maximum KV VRAM on CPU (an upper bound matching vLLM's own allocation cap, including tokens currently in flight).

```python
@dataclass(frozen=True)
class LocalGlobal(pllm.Workload):
    """Even layers keep only the most recent 256 tokens of KV (sliding window); odd layers unchanged."""

    depth: int = 24  # Model depth (24 for 0.5B, 28 for 7B)

    def even_layers(self) -> tuple[int, ...]:
        return tuple(range(0, self.depth, 2))

    @pllm.custom_attention(layers=even_layers, keep="window", window=256)
    def read(self, q: torch.Tensor, kv: pllm.model.PagedKV) -> torch.Tensor:
        return kv.attend(q)[0]


@dataclass(frozen=True)
class PromptPlusWindow(pllm.Workload):
    """Even layers keep the entire prompt plus the most recent 256 generated tokens."""

    depth: int = 24

    def even_layers(self) -> tuple[int, ...]:
        return tuple(range(0, self.depth, 2))

    @pllm.custom_attention(layers=even_layers, keep="window", window=256, sink=True)
    def read(self, q: torch.Tensor, kv: pllm.model.PagedKV) -> torch.Tensor:
        return kv.attend(q)[0]


@dataclass(frozen=True)
class CrossLayerShared(pllm.Workload):
    """Odd layers store no KV and read directly from the layer below (CLA-style cross-layer sharing)."""

    depth: int = 24

    def odd_layers(self) -> tuple[int, ...]:
        return tuple(range(1, self.depth, 2))

    @pllm.custom_attention(layers=odd_layers, keep="none", source=1)
    def read(self, q: torch.Tensor, kv: pllm.model.PagedKV) -> torch.Tensor:
        return kv.attend(q)[0]


LAYOUTS = (NativeRead(), LocalGlobal(), PromptPlusWindow(), CrossLayerShared())
for workload in LAYOUTS:
    per_request = sum(capacity.bytes for capacity in pllm.explain(workload, CONFIG).kv_capacity)
    print(f"{type(workload).__name__:17s}: 每个请求最多 {per_request / 2**20:5.1f} MiB KV")
```

```text
NativeRead       : 每个请求最多  24.0 MiB KV
LocalGlobal      : 每个请求最多  16.6 MiB KV
PromptPlusWindow : 每个请求最多  24.0 MiB KV
CrossLayerShared : 每个请求最多  12.0 MiB KV
```

```python
for workload in LAYOUTS:
    with pllm.make_llm(CONFIG, workload=workload) as llm:
        concurrency = llm.diagnostics().kv_cache_max_concurrency  # Computed by vLLM from the actually allocated KV pool
        print(f"{type(workload).__name__:17s}: 可同时容纳 {concurrency:6.1f} 个 2048-token 请求；"
              f"回答 {answer(llm)!r}")
```

```text
NativeRead       : 可同时容纳  358.1 个 2048-token 请求；回答 '7361'
LocalGlobal      : 可同时容纳  517.6 个 2048-token 请求；回答 '3'
PromptPlusWindow : 可同时容纳  358.1 个 2048-token 请求；回答 '7361'
CrossLayerShared : 可同时容纳  710.4 个 2048-token 请求；回答 'arga textDecorationugi-ingenSalesproj:oランキング étapelessKey'
```

Compare `explain`'s CPU estimates side by side with vLLM's actual pool allocation after startup:
- `explain` computes per-request max KV VRAM for the four workloads as: `NativeRead` `24.0 MiB`, `LocalGlobal` `16.6 MiB`, `PromptPlusWindow` `24.0 MiB`, `CrossLayerShared` `12.0 MiB`.
- After engine startup, `llm.diagnostics().kv_cache_max_concurrency` reports concurrency capacity from the actual allocated KV pool—how many 2048-token requests fit concurrently: `358.1`, `517.6`, `358.1`, and `710.4`.
- `LocalGlobal` reaches ~`1.45x` `NativeRead`'s concurrency, matching `24.0 / 16.6 ≈ 1.45`; `CrossLayerShared` reaches ~`1.98x`, matching `24.0 / 12.0 = 2.00`. Actual KV pool sizing also depends on compiler and workspace reservations, so concurrency does not scale purely from the per-request upper bound alone.
- `PromptPlusWindow`'s per-request upper bound stays `24.0 MiB`: in the worst case, a prompt fills all 2048 tokens and `sink=True` retains the whole prompt. It saves KV when post-prompt generation exceeds the window; actual concurrency vs. native can still differ slightly with workspace sizing.

Look at the answers on the secret-code question:
- `LocalGlobal` answers `'3'`, and `CrossLayerShared` emits gibberish `'arga textDecorationugi-ingenSalesproj:oランキング étapelessKey'`. As noted upfront, Qwen2.5-0.5B was not trained for local sliding windows or cross-layer KV sharing, so imposing those layouts breaks its answers.
- `PromptPlusWindow` still answers `'7361'`: generation is only 12 tokens and the full prompt is retained, so no KV is actually dropped.

Having inspected memory capacity, let's measure **the batched concurrent generation throughput of these custom reads and KV layouts**. 0.5B sufficed for inspecting behavior; for throughput we switch to `Qwen/Qwen2.5-7B-Instruct`: a 0.5B decode step takes only a few milliseconds, magnifying any fixed overhead, whereas 7B reflects typical deployment scale (bf16 weights ~15 GiB, requiring a 24 GiB GPU). Below we first compute the 7B per-request KV caps with `explain`, then benchmark throughput across 128 concurrent requests (8 prompts × 16 samples × 256 generated tokens) using `pllm.benchmark.throughput`: 1 warmup round followed by the median of 3 timed rounds, timing the full `generate` call (including prefill).

```python
many = [f"Explain the mathematical idea numbered {i}:" for i in range(8)]
sampling = SamplingParams(n=16, max_tokens=256, temperature=0.8, top_p=0.95, ignore_eos=True, seed=7)
# 7B has 28 layers, so declare layouts for 28 layers. Benchmark sequences are ~270 tokens, so cap context at 512 with up to 128 tokens/step:
# NativeRead, Quest, and CrossLayerShared use Inductor; window layouts preserve source ops. Memory utilization is 0.70 across all runs.
# QuestRead's PyTorch scorer materializes all pages' keys in FP32 per scheduled token; 7B's wider KV heads add >1 GiB of temporary VRAM.
BIG = pllm.ModelConfig("Qwen/Qwen2.5-7B-Instruct", max_model_len=512, max_num_seqs=128, max_num_batched_tokens=128,
                       enable_chunked_prefill=True, gpu_memory_utilization=0.70, seed=0,
                       execution=pllm.ExecutionOptions(cuda_graph_pool_budget_bytes=1 << 30))
for workload in (NativeRead(), LocalGlobal(depth=28), PromptPlusWindow(depth=28), CrossLayerShared(depth=28)):
    per_request = sum(capacity.bytes for capacity in pllm.explain(workload, replace(BIG, max_model_len=2048)).kv_capacity)
    print(f"7B {type(workload).__name__:17s}: 每个 2048-token 请求最多 {per_request / 2**20:5.1f} MiB KV")

rates = {}
with pllm.make_llm(BIG) as plain:
    rates["原生 vLLM"] = throughput(plain, many, sampling)
for workload in (NativeRead(), QuestRead(pages=8, depth=28), LocalGlobal(depth=28), CrossLayerShared(depth=28)):
    with pllm.make_llm(BIG, workload=workload) as llm:
        rates[type(workload).__name__] = throughput(llm, many, sampling)
for name, rate in rates.items():
    print(f"7B {name:17s}: {rate:6.0f} token/s，为原生 vLLM 的 {rate / rates['原生 vLLM']:.0%}")
```

```text
7B NativeRead       : 每个 2048-token 请求最多 112.0 MiB KV
7B LocalGlobal      : 每个 2048-token 请求最多  70.4 MiB KV
7B PromptPlusWindow : 每个 2048-token 请求最多 112.0 MiB KV
7B CrossLayerShared : 每个 2048-token 请求最多  56.0 MiB KV
7B 原生 vLLM          :   4855 token/s，为原生 vLLM 的 100%
7B NativeRead       :   4817 token/s，为原生 vLLM 的 99%
7B QuestRead        :    649 token/s，为原生 vLLM 的 13%
7B LocalGlobal      :   4804 token/s，为原生 vLLM 的 99%
7B CrossLayerShared :   4798 token/s，为原生 vLLM 的 99%
```

First, the 7B KV accounting (`explain` max KV VRAM per 2048-token request): `NativeRead` `112.0 MiB`, `LocalGlobal` drops to `70.4 MiB`, `PromptPlusWindow` stays `112.0 MiB`, and `CrossLayerShared` drops to `56.0 MiB`. 7B uses 4x the KV per layer per token of 0.5B (4 KV heads × head_dim 128 vs. 2 KV heads × head_dim 64) and has more layers, so the absolute savings are much larger at the same relative ratios.

Now look at throughput (RTX 4090, vLLM 0.32.0, Qwen2.5-7B-Instruct bf16; 8 prompts × n=16, 256 tokens each; `max_model_len=512`, max 128 tokens/step, `gpu_memory_utilization=0.70`; 1 warmup + median of 3 timed rounds, full `generate`): all runs have CUDA graphs enabled; `NativeRead`, `QuestRead`, and `CrossLayerShared` use Inductor, while `LocalGlobal` uses the eager compiler.
- Vanilla vLLM: `4855 token/s`.
- `NativeRead`: `4817 token/s`, `99%` of vanilla.
- `LocalGlobal`: `4804 token/s` (`99%`); `CrossLayerShared`: `4798 token/s` (`99%`). Total sequence length is ~270 tokens, so a 256-token window trims very little history; the KV accounting above shows the capacity savings directly.
- `QuestRead`: `649 token/s` (`13%`). Here `page_scores` reads all pages' keys in PyTorch every step as a didactic implementation—do not take this number as a claim about the Quest algorithm's intrinsic speed.

Configurations are timed sequentially, so ratios include graph structure, fused kernels, and run-to-run variance rather than representing an interleaved paired speedup study. Under Inductor, Quest's temporaries OOMed at the earlier 0.75 utilization, so all runs in this cell share 0.70 to leave room for temporary tensors.

Finally, we return to the initial monitoring workload `RecentMass` and run a full GPU contract check with `testing.check_on_gpu`. Here we pass `neutral=RecentMass()` itself: because a monitoring workload is designed never to alter model generation, it serves directly as its own no-op control against vanilla vLLM's full-vocabulary distribution. Inductor may select different fused kernels for different model graph structures, in which case exact bitwise token parity is not required: `check_on_gpu` teacher-forces vanilla's greedy and fixed-seed continuations and compares the full-vocabulary distribution, requiring bidirectional KL mean/P99 `<= 0.001/0.01` nats and TV mean/P99 `<= 0.01/0.05`. Paths using identical kernels still compare token IDs bitwise.

```python
print(testing.check_on_gpu(RecentMass(), CONFIG, neutral=RecentMass()))
```

```text
status  contract / reason
PASS    explain/probe
PASS    host syncs: eight steady decode steps
PASS    preempted: fixed per-token inputs, token IDs and terminal request state
PASS    neutral: teacher-forced full-vocabulary KL/TV vs vanilla; T=0: KL mean/P99 0/0, TV 0/0; T=0.8: KL mean/P99 0/0, TV 0/0
```

All four GPU contract checks `PASS`:
- `explain/probe` passes;
- `host syncs: eight steady decode steps`: no GPU-to-CPU synchronization occurs over 8 steady decode steps;
- `preempted`: after forced request preemption and recomputation, per-token inputs, generated token IDs, and terminal request state reproduce identically;
- `neutral: teacher-forced full-vocabulary KL/TV vs vanilla`: compares full-vocabulary distributions on vanilla's greedy and fixed-seed continuations, reporting mean and P99 against the thresholds above.

### Summary {#en-summary}

- `@pllm.custom_attention` replaces only the attention read at the specified layers; native q/k/v projections and KV writes proceed as usual.
- `kv.attend` returns the native kernel's output and float32 LSE, supporting local windows via `window=` or per-head logical position selection via `topk=`; the method can also return an `aux` tensor readable downstream via `pllm.attention(layer, "aux")` (or `"lse"` for the LSE of that layer's first `kv.attend`).
- KV storage layouts (`keep="window"`, `window=`, `sink=True`, `keep="none"`, `source=`) are declared directly in the decorator arguments, delegating underlying allocation and reclamation to vLLM; `pllm.explain` reports per-request KV VRAM caps on CPU before starting the engine.
- Know the boundary: read-only top-K (selective reading) saves read bandwidth, not KV VRAM; score-based dynamic eviction freeing KV VRAM (such as H2O or SnapKV) is unsupported in pLLM.
- Constraints: currently requires single-GPU (`tensor_parallel_size=1`), FlashAttention 2 backend with unquantized KV, and prefix caching disabled; static layouts cannot be combined with `PositionState` or `@pllm.custom_forward`.

### Next {#en-next}

In the next guide, [11 Lookup Memory](/pllm/docs/examples/11_lookup_memory/), we look at how to look up a large table by token ID and explicitly choose whether it lives in GPU VRAM or host memory.

</div>

<div class="zine-lang-panel is-hidden" data-lang-panel="zh" lang="zh-CN" markdown="1">

<div id="zh-sec-1"></div>

在阅读过 00、06 和 07 之后，如果你关心 attention 层面的推理算法——比如统计注意力落在哪些位置上的注意力监控、训练无关的稀疏读取（如 Quest 按页挑选、只看部分位置），或者改变 KV 的存储布局（滑动窗口、保留 prompt 加最近窗口、跨层共享 KV）——你会发现这些算法有一个共同点：它们都不改 q/k/v 的投影，只改“这一层读 KV 的方式”或“KV 怎么存”。但在 vLLM 里，attention 是一个融合好的 kernel，KV 分页存放、由调度器分配和回收；想要改动它，通常需要修改 attention 后端或模型代码。

`@pllm.custom_attention` 只替换指定层的 attention 读法：原生的投影和 KV 写入照常进行，方法拿到当前层（已加 RoPE 的）`q` 和一个只读的分页 KV 视图 `kv`，用 `kv.attend(...)` 调用原生 kernel（可加窗口或每个头各自的位置选择），返回 attention 输出，还可以多返回一个 `aux` 张量给后面的方法读。同时，同一个装饰器上的 `keep=`、`window=`、`sink=`、`source=` 参数用来声明 KV 怎么存，具体的存储由 vLLM 自己的滑动窗口、R-SWA 或 KV 共享机制负责分配和回收。

在开始之前，有几项事实、限制和代价需要先说明清楚：
- Qwen2.5-0.5B 没有按这些布局训练过。跨层共享和偶数层滑动窗口在这里会把答案弄坏（实测见第三步）；这个示例演示的是接法、检查和代价，不演示生成质量。
- 只读 top-K（选择读取）省下的是读取的数据量，不省 KV 显存。这里的页打分直接用 PyTorch 编写，每层每步都要把所有页的 key 读一遍，在 7B 上实测吞吐只有原生的 13%：它用来说明语义，不是提速的写法。真正提速需要把页的 min/max 增量维护，并写成自定义 kernel（见示例 09 的做法）。
- 按分数真正释放 KV 显存（例如 H2O、SnapKV 这类驱逐算法）在 pLLM 中不支持（`Score-based memory eviction is unsupported`）。原因在于：vLLM 假设每个请求的 KV 是从头开始连续的一段，位置和块表都由“已经算了多少 token”推出，块表只增不减；要在中间挖洞，需要修改 vLLM 的这些底层约定。
- 运行限制：要求单卡（`tensor_parallel_size=1`）、未量化 KV 的 FlashAttention 2 后端、关闭前缀缓存；静态布局不能和 `PositionState` 或 `@pllm.custom_forward` 一起使用。

本示例分三步带你做实验（实测环境：RTX 4090、vLLM 0.32.0、bf16；看行为用 Qwen2.5-0.5B-Instruct，第三步末尾的 KV 账单和吞吐用 Qwen2.5-7B-Instruct）：
1. **监控**：读出注意力落在最近位置上的比例，不改变模型输出。
2. **只读 top-K**：实现 Quest 式的按页挑选读取，并对照“同样预算下只读最近的位置”。
3. **KV 布局**：对比滑动窗口、保留 prompt 加窗口、跨层共享这几种布局各自省下多少 KV 显存，以及吞吐是多少。

为了方便观察不同读法和布局对长上下文的影响，我们先准备一段约 1500 token 的测试长 prompt：在一篇博物馆介绍中间埋入一句 `"The secret code for the blue door is 7361."`，然后向模型提问密码是多少。下面先用原生 vLLM 回答一次作为基准。

```python
import os

os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"  # 只显示 vLLM 的报错，省掉启动日志

from collections import defaultdict
from dataclasses import dataclass, replace

import torch
from transformers import AutoTokenizer
from vllm import SamplingParams

import pllm
from pllm import testing
from pllm.benchmark import throughput

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
# 2048 token 的上下文；分块 prefill，每步最多 256 个 token；pLLM 的 CUDA graph 捕获预算 1 GiB
# （执行选项只作用于带负载的引擎，同一个配置也可以起原生 vLLM）。
CONFIG = pllm.ModelConfig(MODEL, max_model_len=2048, max_num_seqs=8, max_num_batched_tokens=256,
                          enable_chunked_prefill=True, gpu_memory_utilization=0.4, seed=0,
                          execution=pllm.ExecutionOptions(cuda_graph_pool_budget_bytes=1 << 30))
```

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL)
filler = ("The museum opens at nine in the morning and closes at six in the evening. Visitors can buy tickets at the "
          "front desk or online. The cafe on the second floor serves coffee, tea and sandwiches. ")
passage = filler * 6 + "The secret code for the blue door is 7361. " + filler * 30
question = "What is the secret code for the blue door? Answer with the number only."
prompt = tokenizer.apply_chat_template([{"role": "user", "content": passage + "\n\n" + question}], tokenize=False,
                                       add_generation_prompt=True)
greedy = SamplingParams(max_tokens=12, temperature=0.0)
print("prompt 长度:", len(tokenizer(prompt).input_ids), "token")


def answer(llm) -> str:
    return llm.generate([prompt], greedy, use_tqdm=False)[0].outputs[0].text


with pllm.make_llm(CONFIG) as plain:
    print("原生 vLLM 的回答:", repr(answer(plain)))
```

```text
prompt 长度: 1500 token
原生 vLLM 的回答: '7361'
```

原生 vLLM 在这段 1500 token 的 prompt 上回答出了 `'7361'`。接下来进入第一步：**我们要如何在不改变模型输出的前提下，监控某一层的注意力有多大比例落在了最近的位置上？**

这里我们关心的是如何在 attention 计算过程中提取中间统计量并导出到主机。`kv.attend(q, *, window=0, topk=None)` 会返回 `(输出 [行, 头, 头维], float32 LSE [行, 头])`，不带选择参数时调用的就是原生的因果 kernel；其中的 LSE 是 log-sum-exp，即 softmax 归一化常数取对数。

下面的 `RecentMass` 负载把自定义 attention 方法 `split` 只装在第 12 层：
- 先调用一次 `kv.attend(q)` 完成原生的读取，拿到正常输出 `out` 和全序列的 `lse`；
- 再调用一次 `kv.attend(q, window=64)`，只读最近 64 个位置（`window` 从当前 token 往回数，含当前 token），拿到局部窗口的 `near`（即局部 LSE）；
- 将两个 LSE 相减再取指数 `(near - lse).exp()`，算出的就是“最近 64 个位置分到的注意力占比”，对注意力头取平均后作为第二个输出（`aux`）返回；
- 第一个返回值依然用第一次算出的原生结果 `out`，因此模型行为保持不变。

接着，`@pllm.export` 装饰的 `report` 方法通过 `pllm.attention(12, "aux")` 读取第 12 层自定义 attention 返回的这个 `aux` 张量（在导出方法里读到的是被采样的行；如果写 `"lse"` 则会读取该层第一次 `kv.attend` 的 LSE），每步将其导出到主机，最后由 `on_export` 回调按请求（`prompt_index`）收集起来。

我们同时跑两个请求来做对照：一个是刚才 1500 token 的博物馆长 prompt，另一个是很短的 `"What is 2 + 3?"`。

```python
@dataclass(frozen=True)
class RecentMass(pllm.Workload):
    recent: int = 64

    @pllm.custom_attention(layers=(12,))
    def split(self, q: torch.Tensor, kv: pllm.model.PagedKV) -> tuple[torch.Tensor, torch.Tensor]:
        out, lse = kv.attend(q)                     # 原生的读法，同时拿到 log-sum-exp
        _, near = kv.attend(q, window=self.recent)  # 只读最近 recent 个位置
        return out, (near - lse).exp().mean(-1, keepdim=True)  # aux：最近位置分到的注意力占比，对头取平均

    @pllm.export(name="recent_mass", max_rows=8, max_cols=1)
    def report(self, share: torch.Tensor = pllm.attention(12, "aux")) -> torch.Tensor:
        return share
```

```python
shares: dict[int, list[float]] = defaultdict(list)


@dataclass(frozen=True)
class RecordedMass(RecentMass):
    def on_export(self, event: pllm.io.TraceEvent) -> None:
        for identity, row in zip(event.identities, event.tensor().tolist()):
            shares[identity.prompt_index].append(round(row[0], 3))


short = tokenizer.apply_chat_template([{"role": "user", "content": "What is 2 + 3?"}], tokenize=False,
                                      add_generation_prompt=True)
with pllm.make_llm(CONFIG, workload=RecordedMass()) as llm:
    outputs = llm.generate([prompt, short], greedy, use_tqdm=False)
    print("eager 原因:", repr(llm.diagnostics().eager_reason))
for index, output in enumerate(outputs):
    print(f"prompt {index}: {output.outputs[0].text!r}")
    print("    最近 64 个位置的注意力占比:", shares[index])
```

```text
eager 原因: ''
prompt 0: '7361'
    最近 64 个位置的注意力占比: [0.633, 0.588, 0.666, 0.757, 0.586]
prompt 1: '2 + 3 equals 5.'
    最近 64 个位置的注意力占比: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
```

先看运行结果：`eager 原因` 为空字符串 `''`，说明这个监控负载在 CUDA graph 里运行。长 prompt（`prompt 0`）的回答依然是 `'7361'`；在它生成的每个 token 上，第 12 层最近 64 个位置分到了 `0.586` 到 `0.757` 的注意力，其余注意力落在更早的位置。短 prompt（`prompt 1`）回答了 `'2 + 3 equals 5.'`，每一步的占比都是 `1.0`——因为它的 prompt 只有 37 个 token，加上生成的内容总长也不到 64 个 token，全部位置都在“最近 64 个”的范围里。最后一步用 `check_on_gpu` 比较监控负载和原生模型的完整词表分布。

接下来进入第二步：**如果每层每个头只允许读固定预算的少量位置，按页挑选高分位置（Quest）和直接只读最近的位置有什么区别？**

Quest（Tang et al., 2024）的做法用白话概括就是：在 vLLM 中，KV 按页（这里每页 16 个 token）存放。对每一页，记下这一页里所有 key 在每个维度上的最小值和最大值；对于当前的查询向量 $$q$$，$$\sum_d \max(q_d \cdot \text{min}_d, q_d \cdot \text{max}_d)$$ 就是 $$q \cdot k$$ 在这一页里可能取到的最大值的上界。每个注意力头各自挑出上界最大的几页，只对这几页做 attention。

在下面的代码里：
- `page_scores` 用 PyTorch 直接从 `kv.key_cache` 和 `kv.block_table` 算出每一行、每个头、每一页的上界（Qwen2.5-0.5B 使用 GQA，14 个 query 头按 2 个 KV 头分组）。
- `QuestRead` 装在第 2 到 23 层上（第 0、1 层照常读全部位置）。它先把当前位置还没写到的页屏蔽成 `-inf`，然后每个头各自取上界最大的 `pages` 页，换算成 token 的逻辑位置交给 `kv.attend(q, topk=...)`（`topk` 支持形状 `[行, k]` 或 `[行, 头, k]`，不足处可用 `-1` 补齐，不同头、不同层可以有不同预算）。
- 为了对照，当 `recent_only=True` 时，我们在同样的预算（8 页 = 128 个 token）下改为调用 `kv.attend(q, window=128)`，即只读最近的 128 个位置。另外我们还定义了 `NativeRead`，每层读全部位置，既作为这里的对照，也作为下一步比较 KV 布局的基准。

下一个单元格先在 CPU 上运行 `pllm.explain` 查看读取开销估算和契约检查，随后在 GPU 上测试它们能否答出密码问题。

```python
def page_scores(q: torch.Tensor, kv: pllm.model.PagedKV) -> torch.Tensor:
    """Quest：用每页 key 在各维上的最小/最大值，给 q·k 在这一页里的最大可能值一个上界。"""
    keys = kv.key_cache[kv.block_table[kv.requests].long()].float()  # [行, 页, 页内位置, kv 头, 头维]
    low, high = keys.amin(2), keys.amax(2)                           # [行, 页, kv 头, 头维]
    rows, heads, dim = q.shape
    g = q.float().reshape(rows, low.shape[2], heads // low.shape[2], dim)  # GQA：按 kv 头分组
    bound = (torch.einsum("rkgd,rpkd->rkgp", g.clamp(min=0), high)
             + torch.einsum("rkgd,rpkd->rkgp", g.clamp(max=0), low))
    return bound.reshape(rows, heads, -1)                            # [行, 头, 页]


@dataclass(frozen=True)
class NativeRead(pllm.Workload):
    """对照组：每层照常读全部位置，KV 照常全部保存。"""

    @pllm.custom_attention()
    def read(self, q: torch.Tensor, kv: pllm.model.PagedKV) -> torch.Tensor:
        return kv.attend(q)[0]


@dataclass(frozen=True)
class QuestRead(pllm.Workload):
    pages: int = 8          # 每个头读几页（每页 16 个 token）
    recent_only: bool = False  # 对照组：同样的预算，只读最近的位置
    depth: int = 24         # 模型层数：0.5B 是 24，第三步末尾在 7B 上测吞吐时传 28

    def later_layers(self) -> tuple[int, ...]:
        return tuple(range(2, self.depth))

    @pllm.custom_attention(layers=later_layers)
    def read(self, q: torch.Tensor, kv: pllm.model.PagedKV) -> torch.Tensor:
        block = kv.key_cache.shape[1]
        if self.recent_only:
            return kv.attend(q, window=self.pages * block)[0]
        scores = page_scores(q, kv)
        page = torch.arange(scores.shape[-1], device=q.device)
        scores = scores.masked_fill(page * block > kv.positions[:, None, None], -float("inf"))  # 还没写到的页
        chosen = scores.topk(min(self.pages, scores.shape[-1]), dim=-1).indices            # [行, 头, pages]
        tokens = (chosen[..., None] * block + torch.arange(block, device=q.device)).flatten(2)
        return kv.attend(q, topk=tokens)[0]  # 每个头只读自己选中的位置
```

```python
for workload in (NativeRead(), QuestRead()):
    for line in str(pllm.explain(workload, CONFIG)).splitlines():
        if line.startswith("attention layer 2:"):
            print(f"{type(workload).__name__:10s}", line)
print(testing.check(QuestRead(), CONFIG))
```

```text
NativeRead attention layer 2: custom `read`; all positions natively; 7340032 FLOP/token, <= 1048576 KV bytes/token [estimate at max_model_len; per-head budget upper bound]
QuestRead  attention layer 2: custom `read`; topk <= 128/head; 458752 FLOP/token, <= 458752 KV bytes/token [estimate at max_model_len; per-head budget upper bound]
status  contract / reason
PASS    tensor_parallel_size=1: admission
PASS    compile
PASS    custom-attention method math / CPU paged KV schedules
PASS    row permutation
PASS    request isolation
PASS    mixed prefill/decode
PASS    chunking / tokens=
PASS    preempted
SKIP    publish_lag: no row method reads versioned state
SKIP    neutral: pass neutral= with the author's disabled settings
```

```python
for workload in (QuestRead(pages=8), QuestRead(pages=8, recent_only=True), QuestRead(pages=2)):
    with pllm.make_llm(CONFIG, workload=workload) as llm:
        print(f"{workload}: {answer(llm)!r}")
```

```text
QuestRead(pages=8, recent_only=False, depth=24): '7361'
QuestRead(pages=8, recent_only=True, depth=24): ''
QuestRead(pages=2, recent_only=False, depth=24): '9'
```

我们先看 `explain` 在 CPU 上给出的估算（上下文按 `max_model_len=2048` 计算）：
- `NativeRead` 读全部位置时，第 2 层每个 token 最多读 `1048576` 字节 KV，计算量为 `7340032` FLOP/token；
- `QuestRead` 每个头读 128 个位置（`topk <= 128/head`）时，计算量降到 `458752` FLOP/token，但每个 token 最多要读 `458752` 字节 KV。

为什么 128 个位置只是 2048 的 1/16，KV 读取量的上限却占到了约 44%（`458752 / 1048576 ≈ 0.44`）？因为这个模型的 14 个 query 头共享 2 个 KV 头，当每个 query 头各选各的位置时，同一个 KV 头的数据会被不同的 query 头各读一遍。`explain` 在写代码阶段就把这个代价直接摆了出来。

接着看 `testing.check` 的 CPU 契约检查：自定义 attention 方法的数学逻辑在各种分页 KV 调度下保持一致，行置换（`row permutation`）、请求隔离（`request isolation`）、prefill/decode 混合（`mixed prefill/decode`）、分块（`chunking / tokens=`）以及抢占重算（`preempted`）全部 `PASS`。

再看 GPU 上的回答结果：
- `QuestRead(pages=8, recent_only=False)` 每头挑 8 页（128 个 token）时，仍然答出了 `'7361'`；
- `QuestRead(pages=8, recent_only=True)` 在同样 128 个 token 的预算下只读最近位置，回答是空字符串 `''`（第一个生成的 token 就是结束符），因为包含密码的那句话在最近 128 个位置之外；
- `QuestRead(pages=2, recent_only=False)` 把预算缩减到每头只读 2 页时，回答成了 `'9'`。

请注意，这只是一个 prompt 上的功能演示，不是质量评测。此外还要再次提醒：只读 top-K 省下的是读取量，不会节省 KV 显存。

接下来进入第三步：**如果我们希望真正减少每个请求占用的 KV 显存，应该怎么声明 KV 存储布局？不同布局各自能省多少显存、对模型输出有什么影响？**

我们在下面的代码中对比四个负载，它们的方法体全都只是 `return kv.attend(q)[0]`，区别只在于 `@pllm.custom_attention` 装饰器里的存储布局声明（窗口裁剪和共享都由存储布局本身负责）：
1. `NativeRead`：默认全部保存。
2. `LocalGlobal`：偶数层声明 `keep="window", window=256`，只保留最近 256 个 token 的 KV，更早的块由 vLLM 的滑动窗口机制自动释放；奇数层照常保存全部 KV。
3. `PromptPlusWindow`：偶数层在 `keep="window", window=256` 的基础上加上 `sink=True`，保留整个 prompt 再加上最近 256 个生成的 token（对应 vLLM 的 R-SWA）；奇数层照常。
4. `CrossLayerShared`：奇数层声明 `keep="none", source=1`，这一层不分配自己的 KV，直接读取下面第 1 层（即前一层偶数层）的 KV。

不需要启动 GPU 引擎，`pllm.explain` 在 CPU 上就能算出每个请求最多需要多少 KV 显存（`explain` 算出的这个值是 vLLM 自己的分配上限，包含了正在调度中的 token）。

```python
@dataclass(frozen=True)
class LocalGlobal(pllm.Workload):
    """偶数层只保留最近 256 个 token 的 KV（滑动窗口），奇数层照常。"""

    depth: int = 24  # 模型层数（0.5B 是 24，7B 是 28）

    def even_layers(self) -> tuple[int, ...]:
        return tuple(range(0, self.depth, 2))

    @pllm.custom_attention(layers=even_layers, keep="window", window=256)
    def read(self, q: torch.Tensor, kv: pllm.model.PagedKV) -> torch.Tensor:
        return kv.attend(q)[0]


@dataclass(frozen=True)
class PromptPlusWindow(pllm.Workload):
    """偶数层保留整个 prompt，加上最近 256 个生成的 token。"""

    depth: int = 24

    def even_layers(self) -> tuple[int, ...]:
        return tuple(range(0, self.depth, 2))

    @pllm.custom_attention(layers=even_layers, keep="window", window=256, sink=True)
    def read(self, q: torch.Tensor, kv: pllm.model.PagedKV) -> torch.Tensor:
        return kv.attend(q)[0]


@dataclass(frozen=True)
class CrossLayerShared(pllm.Workload):
    """奇数层不存 KV，直接读下面一层的 KV（CLA 式跨层共享）。"""

    depth: int = 24

    def odd_layers(self) -> tuple[int, ...]:
        return tuple(range(1, self.depth, 2))

    @pllm.custom_attention(layers=odd_layers, keep="none", source=1)
    def read(self, q: torch.Tensor, kv: pllm.model.PagedKV) -> torch.Tensor:
        return kv.attend(q)[0]


LAYOUTS = (NativeRead(), LocalGlobal(), PromptPlusWindow(), CrossLayerShared())
for workload in LAYOUTS:
    per_request = sum(capacity.bytes for capacity in pllm.explain(workload, CONFIG).kv_capacity)
    print(f"{type(workload).__name__:17s}: 每个请求最多 {per_request / 2**20:5.1f} MiB KV")
```

```text
NativeRead       : 每个请求最多  24.0 MiB KV
LocalGlobal      : 每个请求最多  16.6 MiB KV
PromptPlusWindow : 每个请求最多  24.0 MiB KV
CrossLayerShared : 每个请求最多  12.0 MiB KV
```

```python
for workload in LAYOUTS:
    with pllm.make_llm(CONFIG, workload=workload) as llm:
        concurrency = llm.diagnostics().kv_cache_max_concurrency  # vLLM 按实际分配的 KV 池算出
        print(f"{type(workload).__name__:17s}: 可同时容纳 {concurrency:6.1f} 个 2048-token 请求；"
              f"回答 {answer(llm)!r}")
```

```text
NativeRead       : 可同时容纳  358.1 个 2048-token 请求；回答 '7361'
LocalGlobal      : 可同时容纳  517.6 个 2048-token 请求；回答 '3'
PromptPlusWindow : 可同时容纳  358.1 个 2048-token 请求；回答 '7361'
CrossLayerShared : 可同时容纳  710.4 个 2048-token 请求；回答 'arga textDecorationugi-ingenSalesproj:oランキング étapelessKey'
```

把 `explain` 在 CPU 上的预估和 vLLM 启动后的实际分配放在一起看：
- `explain` 算出四个负载每个请求最多占用的 KV 显存分别是：`NativeRead` `24.0 MiB`、`LocalGlobal` `16.6 MiB`、`PromptPlusWindow` `24.0 MiB`、`CrossLayerShared` `12.0 MiB`。
- 启动引擎后，`llm.diagnostics().kv_cache_max_concurrency` 给出按真实 KV 池分配算出的并发容量，即能同时装下几个 2048-token 请求：分别是 `358.1`、`517.6`、`358.1` 和 `710.4` 个。
- `LocalGlobal` 的实际并发容量约为 `NativeRead` 的 `1.45` 倍，与 `24.0 / 16.6 ≈ 1.45` 相近；`CrossLayerShared` 约为 `1.98` 倍，与 `24.0 / 12.0 = 2.00` 相近。实际 KV 池还受编译器和工作区定容影响，不能只由单请求上限推算出完全相同的并发容量。
- `PromptPlusWindow` 的单请求上限仍是 `24.0 MiB`：最坏情况下 prompt 本身占满 2048 token，而 `sink=True` 要保留整个 prompt。它真正节省 KV 的场景是 prompt 之后生成的内容超出窗口；与原生读法的实际并发容量可以因工作区大小而略有不同。

再看这四种布局在密码问题上的回答：
- `LocalGlobal` 回答成了 `'3'`，`CrossLayerShared` 输出了乱码 `'arga textDecorationugi-ingenSalesproj:oランキング étapelessKey'`。正如开头说明的，Qwen2.5-0.5B 没有按局部滑动窗口或跨层共享 KV 的方式训练过，直接改布局会把答案弄坏。
- `PromptPlusWindow` 仍然回答 `'7361'`：因为这次生成的长度只有 12 个 token，加上保留了完整 prompt，实际上没有任何 KV 被丢掉。

看完了显存容量，下一步我们测一下**这些自定义读法和 KV 布局在批量并发生成时的吞吐表现如何**。前面的观察用 0.5B 就够了；吞吐换到 `Qwen/Qwen2.5-7B-Instruct` 上测：0.5B 一步 decode 只有几毫秒，任何额外开销的占比都被放大，7B 才是常见的部署规模（bf16 权重约 15 GiB，需要一张 24 GiB 的卡）。下面先用 `explain` 算出 7B 上四种布局的单请求 KV 上限，再测吞吐。测试方式是：同时跑 128 个并发请求（8 个 prompt × 每个采样 16 个样本 × 每个生成 256 token），用 `pllm.benchmark.throughput` 测 token/s：预热一轮，再计时三轮取中位数，每轮计整次 `generate` 的时间（含 prefill）。

```python
many = [f"Explain the mathematical idea numbered {i}:" for i in range(8)]
sampling = SamplingParams(n=16, max_tokens=256, temperature=0.8, top_p=0.95, ignore_eos=True, seed=7)
# 7B 有 28 层，布局按 28 层声明。吞吐测试的序列只有约 270 个 token，上下文上限设为 512；每步最多 128 个 token：
# 原生读取、Quest 和跨层共享用 Inductor，窗口布局保留 source 算子；显存比例统一使用 0.70。
# QuestRead 的 PyTorch 打分为每个被调度的 token 把所有页的 key 取成 FP32，7B 的 KV 头更宽，临时显存会多出 1 GiB 以上。
BIG = pllm.ModelConfig("Qwen/Qwen2.5-7B-Instruct", max_model_len=512, max_num_seqs=128, max_num_batched_tokens=128,
                       enable_chunked_prefill=True, gpu_memory_utilization=0.70, seed=0,
                       execution=pllm.ExecutionOptions(cuda_graph_pool_budget_bytes=1 << 30))
for workload in (NativeRead(), LocalGlobal(depth=28), PromptPlusWindow(depth=28), CrossLayerShared(depth=28)):
    per_request = sum(capacity.bytes for capacity in pllm.explain(workload, replace(BIG, max_model_len=2048)).kv_capacity)
    print(f"7B {type(workload).__name__:17s}: 每个 2048-token 请求最多 {per_request / 2**20:5.1f} MiB KV")

rates = {}
with pllm.make_llm(BIG) as plain:
    rates["原生 vLLM"] = throughput(plain, many, sampling)
for workload in (NativeRead(), QuestRead(pages=8, depth=28), LocalGlobal(depth=28), CrossLayerShared(depth=28)):
    with pllm.make_llm(BIG, workload=workload) as llm:
        rates[type(workload).__name__] = throughput(llm, many, sampling)
for name, rate in rates.items():
    print(f"7B {name:17s}: {rate:6.0f} token/s，为原生 vLLM 的 {rate / rates['原生 vLLM']:.0%}")
```

```text
7B NativeRead       : 每个 2048-token 请求最多 112.0 MiB KV
7B LocalGlobal      : 每个 2048-token 请求最多  70.4 MiB KV
7B PromptPlusWindow : 每个 2048-token 请求最多 112.0 MiB KV
7B CrossLayerShared : 每个 2048-token 请求最多  56.0 MiB KV
7B 原生 vLLM          :   4855 token/s，为原生 vLLM 的 100%
7B NativeRead       :   4817 token/s，为原生 vLLM 的 99%
7B QuestRead        :    649 token/s，为原生 vLLM 的 13%
7B LocalGlobal      :   4804 token/s，为原生 vLLM 的 99%
7B CrossLayerShared :   4798 token/s，为原生 vLLM 的 99%
```

先看 7B 上的 KV 账单：`explain` 算出每个 2048-token 请求最多占用的 KV 显存，`NativeRead` 是 `112.0 MiB`，`LocalGlobal` 降到 `70.4 MiB`，`PromptPlusWindow` 仍是 `112.0 MiB`，`CrossLayerShared` 降到 `56.0 MiB`。7B 每层每个 token 的 KV 是 0.5B 的 4 倍（4 个 KV 头 × 头维 128，对 2 个 KV 头 × 头维 64），层数也多，所以同样的布局在 7B 上省下的绝对显存大得多，比例和 0.5B 一样。

再看吞吐（RTX 4090、vLLM 0.32.0、Qwen2.5-7B-Instruct bf16；8 个 prompt × n=16，各 256 token；max_model_len=512、每步最多 128 token、显存比例 0.70；预热 1 轮后 3 轮取中位数，计整次 generate）：所有组都开着 CUDA graph，`NativeRead`、`QuestRead`、`CrossLayerShared` 使用 Inductor，`LocalGlobal` 使用 eager 编译器。
- 原生 vLLM：`4855 token/s`。
- `NativeRead`：`4817 token/s`，为原生的 `99%`。
- `LocalGlobal`：`4804 token/s`（`99%`）；`CrossLayerShared`：`4798 token/s`（`99%`）。序列总长约 270 token，256 的窗口只截去很少的历史；上面的 KV 账单更直接地体现了布局节省的容量。
- `QuestRead`：`649 token/s`（`13%`）。这里的 `page_scores` 每步用 PyTorch 完整读出所有页的 key 来计算 min/max，仍是教学实现，不能据此宣称 Quest 算法本身比原生 attention 慢。

各组按顺序计时，比例同时包含模型图结构、融合 kernel 和运行波动的影响，不是交错配对的算法加速结论。Inductor 下 Quest 的临时张量在此前的显存比例 0.75 会 OOM；本格所有组统一改为 0.70，给临时张量留出空间。

最后一步，我们回到最初的监控负载 `RecentMass`，用 `testing.check_on_gpu` 在 GPU 上做一次完整的契约检查。这里把 `neutral` 参数传为 `RecentMass()` 自己：因为监控负载的设计目标就是完全不改变模型的生成结果，所以直接拿它自己当“关闭修改”的对照版本，与原生 vLLM 比较完整词表分布。Inductor 可能因模型图结构不同而选择不同的融合 kernel；此时不要求生成的 token 逐位相同。`check_on_gpu` 在原生模型的贪心和固定种子续写上 teacher-force 相同 token，比较完整词表分布：双向 KL 的均值/P99 不超过 `0.001/0.01` nats，TV 的均值/P99 不超过 `0.01/0.05`。使用相同 kernel 的路径仍逐位比较 token ID。。

```python
print(testing.check_on_gpu(RecentMass(), CONFIG, neutral=RecentMass()))
```

```text
status  contract / reason
PASS    explain/probe
PASS    host syncs: eight steady decode steps
PASS    preempted: fixed per-token inputs, token IDs and terminal request state
PASS    neutral: teacher-forced full-vocabulary KL/TV vs vanilla; T=0: KL mean/P99 0/0, TV 0/0; T=0.8: KL mean/P99 0/0, TV 0/0
```

GPU 契约检查的四项全部 `PASS`：
- `explain/probe` 通过；
- `host syncs: eight steady decode steps`：连续 8 个稳定的 decode 步内没有发生 GPU 到 CPU 的同步；
- `preempted`：请求被强制驱逐后再重算，每个 token 的输入、生成的 token ID 以及请求的终止状态都能复现；
- `neutral: teacher-forced full-vocabulary KL/TV vs vanilla`：比较原生贪心和固定种子续写上的完整词表分布，输出均值与 P99；阈值见上格说明。

### 小结 {#zh-sec-2}

- `@pllm.custom_attention` 只替换指定层的 attention 读法，原生的 q/k/v 投影和 KV 写入照常进行。
- `kv.attend` 返回原生 kernel 的输出和 float32 LSE，支持通过 `window=` 指定局部窗口，或通过 `topk=` 为每个头指定各自要读的逻辑位置；方法可以额外返回
一个 `aux` 张量，供后续方法通过 `pllm.attention(layer, "aux")` 读取（或用 `"lse"` 读取第一次 `kv.attend` 的 LSE）。
- KV 的存储布局（`keep="window"`、`window=`、`sink=True`、`keep="none"`、`source=`）直接写在装饰器参数里，底层存储的分配与回收交给 vLLM 自己的机制负责；`pllm.explain` 在启动引擎前就能在 CPU 上给出每个请求的 KV 显存上限。
- 区分两类做法的边界：只读 top-K（选择读取）省的是读取量，不省 KV 显存；而按分数动态驱逐并释放 KV 显存的做法（如 H2O、SnapKV）pLLM 不支持。
- 适用限制：当前要求单卡（`tensor_parallel_size=1`）、未量化 KV 的 FlashAttention 2 后端、关闭前缀缓存；静态布局不能与 `PositionState` 或 `@pllm.custom_forward` 同时使用。

### 下一步 {#zh-sec-3}

下一篇 [11 查表记忆](/pllm/docs/examples/11_lookup_memory/) 将介绍如何按 token id 查一张大表，并由你决定把这张表放在 GPU 显存还是主机内存里。

</div>

</div>

<div class="zine-doc-footer">
  <a class="zine-pill-link" data-i18n-zh="← 返回 pLLM 项目与文档中心" data-i18n-en="← Back to pLLM Project &amp; Docs" href="/pllm/#docs">← Back to pLLM Project &amp; Docs</a>
  <a class="zine-pill-link" href="https://github.com/LinesHogan/pLLM/blob/main/doc/examples/10_custom_attention.ipynb" target="_blank" rel="noopener">GitHub &middot; doc/examples/10_custom_attention.ipynb ↗</a>
</div>

<script defer src="/assets/js/bilingual-post.js"></script>
