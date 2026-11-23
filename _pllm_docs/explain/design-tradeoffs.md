---
layout: page
title: "Design Tradeoffs: Why Some Algorithms Cannot Be Written Today"
description: "pLLM Documentation · Architecture & Design Explanations"
permalink: /pllm/docs/explain/design-tradeoffs/
nav: false
---

<link rel="stylesheet" href="{{ '/assets/css/pllm-hub.css' | relative_url | bust_file_cache }}">

<div class="zine-dispatch" data-title-zh="设计取舍：为什么有些算法今天写不了" data-title-en="Design Tradeoffs: Why Some Algorithms Cannot Be Written Today" data-desc-zh="pLLM 文档 · 架构与机制解析" data-desc-en="pLLM Documentation · Architecture &amp; Design Explanations">
  <div class="zine-dispatch-links">
    <a class="zine-pill-link primary" data-i18n-zh="← 返回 pLLM 文档中心" data-i18n-en="← pLLM Docs Hub" href="/pllm/#docs">← pLLM Docs Hub</a>
    <a class="zine-pill-link" data-i18n-zh="文档导航地图" data-i18n-en="Docs Roadmap" href="/pllm/docs/">Docs Roadmap</a>
    <span class="zine-lang-badge" data-i18n-zh="架构与机制解析" data-i18n-en="Architecture &amp; Design Explanations">Architecture &amp; Design Explanations</span>
  </div>
  <a class="zine-pill-link" data-i18n-zh="GitHub 源码 ↗" data-i18n-en="View on GitHub ↗" href="https://github.com/LinesHogan/pLLM/blob/main/doc/explain/design-tradeoffs.md" target="_blank" rel="noopener">View on GitHub ↗</a>
</div>

<div id="markdown-content">

<div class="zine-lang-panel" data-lang-panel="en" lang="en" markdown="1">

<div id="en-design-tradeoffs-why-some-algorithms-cannot-be-written-today"></div>

In a standalone PyTorch script or Hugging Face `generate` loop, a researcher can prototype a new idea in minutes with a few lines of `register_forward_hook`, an inline conditional branch, or `.item()`. Move that same idea into a high-throughput serving engine built on **continuous batching** and **CUDA Graph replay**—such as vLLM—and you quickly run into hard physical and architectural constraints.

pLLM is not a black-box library of pre-packaged algorithms. It is a **programmable stateful substrate** bridging frontier inference-time modeling algorithms and modern serving engines. The substrate handles continuous-batch packing, static CUDA Graph capture, versioned snapshot publishing, and memory lifetime planning so that algorithms compatible with high-throughput execution run at near-native speed.

When an algorithm's assumptions fundamentally collide with those physical mechanisms (fixed tensor shapes, paged block tables, asynchronous stream execution, and strict cross-request isolation), pLLM chooses **not to silently compromise or degrade**; instead, it exposes the system boundary and mathematical reason clearly. This page documents 11 intentional design tradeoffs and limitations, each organized as **Problem to Solve $$\to$$ pLLM's Choice $$\to$$ Why $$\to$$ Alternatives**.

---

## 1. Score-Based KV Eviction & Contiguous KV Caches (H2O, SnapKV) {#en-sec-1}

### Problem to Solve {#en-problem-to-solve}

In long-context generation, physical KV cache memory grows linearly with sequence length. Score-based dynamic eviction methods—such as H2O, SnapKV, and PyramidKV—track cumulative attention scores across historical tokens during autoregressive generation and physically free KV entries for low-scoring "unimportant" tokens, keeping only initial sink tokens and key heavy hitters to slash VRAM usage.

### pLLM's Choice {#en-pllms-choice}

pLLM does not support dynamically freeing physical KV cache memory based on attention scores.

### Why {#en-why}

vLLM's underlying memory model (PagedAttention) rests on one core invariant: **each request's KV cache is a logically contiguous sequence starting at position 0**.

1. **Scalar length counter & slot mapping**: The engine tracks a single integer per request recording how many tokens have been computed. Which physical page and slot each layer writes to for a given token—as well as its absolute Rotary Position Embedding (RoPE) position—is derived directly from this monotonically increasing counter. Block tables only append new pages at the end; there is no mechanism to punch out slots in the middle.
2. **Holes break paged indexing**: Evicting scattered mid-sequence tokens by attention score creates "holes" in the logical sequence. Supporting holes would require rewriting PagedAttention's block-lookup and stride arithmetic from scratch, and because eviction decisions depend on runtime attention scores on the current step, slot layouts could no longer be planned ahead of time.
3. **Breaks CUDA Graph capture**: Reclaiming mid-sequence blocks and compacting block tables on every decode step requires blocking GPU-to-CPU metadata syncs or invalidates the static tensor pointers and memory layouts that CUDA Graphs depend on, destroying graph-mode throughput.

### Alternatives {#en-alternatives}

- **Sparse reads (saves compute and memory bandwidth, not VRAM capacity)**: Inside [`@pllm.custom_attention`](/pllm/docs/guides/forward-structure/), call `kv.attend(q, topk=...)` or write custom scoring logic (such as Quest). Within each layer and head, only the top-$$K$$ highest-scoring positions participate in attention while the rest are skipped. This cuts attention FLOPs and memory traffic, runs entirely on GPU, and stays inside CUDA Graphs; see [10 Custom Attention](/pllm/docs/examples/10_custom_attention/).
- **Rule-driven regular physical reclamation (actually saves VRAM)**: To lower per-request peak KV memory, use a static KV layout aligned with block-table boundaries: sliding window (`keep="window", window=N`), prompt-sink plus sliding window (`sink=True`, i.e., R-SWA), or cross-layer sharing where odd layers reuse even layers' KV (`keep="none", source=k`). These reclaim memory at clean block boundaries without punching holes inside blocks.

```python
from dataclasses import dataclass

import torch

import pllm
from pllm import testing


@dataclass(frozen=True)
class LocalGlobalAttention(pllm.Workload):
    """Even layers keep only the latest 256 tokens of KV (sliding window); odd layers keep full context."""

    window_size: int = 256

    def window(self) -> int:
        return self.window_size

    @pllm.custom_attention(layers=tuple(range(0, 24, 2)), keep="window", window=window)
    def read(self, q: torch.Tensor, kv: pllm.model.PagedKV) -> torch.Tensor:
        return kv.attend(q, window=self.window_size)[0]


CONFIG = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=4)
report = pllm.explain(LocalGlobalAttention(), CONFIG)
```

---

## 2. Compressed Remote Memory & Page Padding Waste (Activation Beacon) {#en-sec-2}

### Problem to Solve {#en-problem-to-solve-1}

Activation Beacon and related KV compression methods condense distant history (hundreds or thousands of past tokens) into a compact set of summary tokens or compressed entries to balance long-context recall against bounded VRAM.

### pLLM's Choice {#en-pllms-choice-1}

pLLM does not currently integrate compressed remote memory into its standard runtime pipeline.

### Why {#en-why-1}

Deeply integrating compressed remote memory into a continuous-batching engine faces three fundamental physical and algorithmic obstacles:

1. **Page padding waste**: vLLM allocates KV memory in fixed-size pages (blocks of typically 16 or 32 tokens). Storing heavily compressed entries in paged blocks (e.g., compressing 16 tokens down to 1 entry) still pads out whole pages, erasing most of the expected memory savings while introducing severe internal fragmentation.
2. **Chunk-dependent state non-determinism**: When compressed summaries depend on deep hidden states and `enable_chunked_prefill` is active, whether a given position's compressed entry is ready on the current step depends on how the scheduler happened to slice the prompt under current concurrency pressure. Varying chunk boundaries cause compressed features to drift across runs for the exact same input, violating inference determinism.
3. **Two-pass attention overhead**: Running one attention kernel over the full-resolution local window and a second pass over compressed remote entries, then merging them in Python/Triton, incurs kernel-launch and memory-reordering overhead that far exceeds a single native dense read (in a single-request single-layer microbenchmark, split two-pass read + merge took ~30.31 µs versus 11.88 µs for a native full read).

### Alternatives {#en-alternatives-1}

- **Fixed-layout reduction**: Use sliding windows (`keep="window"`) or cross-layer sharing (`keep="none", source=k`) to reclaim whole physical KV pages with zero two-pass scheduling overhead.
- **Lookup-prior injection**: For long-range background memory or cross-document priors, use token-history lookup memory (such as the Engram architecture in [11 Lookup Memory](/pllm/docs/examples/11_lookup_memory/)). Store large pretrained tables in host DRAM via `pllm.Pretrained(..., placement="host")`, prefetch them asynchronously over PCIe during forward, and inject them into the residual stream without perturbing the KV cache pipeline.

---

## 3. DSA Top-K Index Reuse & `sm90` Hardware Requirements {#en-sec-3}

### Problem to Solve {#en-problem-to-solve-2}

In DeepSeek Sparse Attention (DSA, e.g., DeepSeek V3/V4), a lightweight indexer scores query–key affinity ahead of attention and selects the top-$$k$$ blocks or tokens (`IndexCache`). Researchers often want to read that indexer's selection and reuse it across subsequent layers without recomputing scores.

### pLLM's Choice {#en-pllms-choice-2}

pLLM does not currently support reading and reusing native DSA top-$$k$$ index selections across layers.

### Why {#en-why-2}

DSA is baked into the model architecture itself, so readable indexer selections exist only on models trained with DSA (such as DeepSeek V3.2 / V4). vLLM's attention kernels for those models require `sm90` or newer GPUs (e.g., H100); on `sm80` (A100) and `sm89` (RTX 4090), those models cannot run at all, leaving no native top-$$k$$ selection to reuse. Today `pllm.attention(layer, ...)` exposes only `"aux"` and `"lse"`.

### Alternatives {#en-alternatives-2}

- On `sm80`/`sm89` hardware, implement custom attention filtering inside `@pllm.custom_attention` using PyTorch or Triton ops—combining [09 Custom Kernels](/pllm/docs/examples/09_custom_kernels/) with explicit selection via `kv.attend(q, topk=...)` from [10 Custom Attention](/pllm/docs/examples/10_custom_attention/).
- To reduce compute across layers, use cross-layer KV sharing (`keep="none", source=k`).

---

## 4. Batch-Competitive Routing (MoR Expert-Choice) & Vanilla MoD {#en-sec-4}

### Problem to Solve {#en-problem-to-solve-3}

Mixture-of-Depths (MoD) and Mixture-of-Recursions (MoR) allocate varying compute depth per token:
1. **MoR expert-choice routing**: Each layer ranks tokens across the entire batch and lets only the top-$$k$$ tokens execute that layer;
2. **Vanilla MoD**: Skipped tokens neither enter the layer's computation nor write any KV entries for that layer.

### pLLM's Choice {#en-pllms-choice-3}

pLLM explicitly rejects batch-competitive expert-choice routing and vanilla MoD's no-KV-write skipping. Instead, pLLM supports per-row independent **token-choice adaptive depth** and **early exit**.

### Why {#en-why-3}

1. **Request isolation must never be broken**: In continuous batching, a single 2D tensor packs completely unrelated requests from different clients. Competing for top-$$k$$ slots across the batch makes whether a token runs a layer depend on which other requests happen to be co-batched at that millisecond. As concurrency or queue order fluctuates, the same prompt generates different text, breaking idempotency and determinism (`pllm.testing.check` catches this immediately with `ContractError: method ... mixes rows`).
2. **KV sequences cannot have holes**: In vanilla MoD, if token $$t$$ skips layer $$L$$ without writing KV, layer $$L$$'s KV cache has a missing slot at position $$t$$. When token $$t+1$$ attends at layer $$L$$, causal masking and paged block tables expect a contiguous history; missing positions break causal alignment just like score-based KV eviction.

### Alternatives {#en-alternatives-3}

- **Token-choice adaptive routing**: Each token decides independently whether to run a layer from its own hidden state or local router (e.g., a sigmoid threshold), without competing against other requests.
- **GPU-side KV backfill & in-graph dispatch**: Pass a boolean mask `active=mask` (shape `[rows, 1]`) inside `@pllm.custom_forward`. Inactive rows skip the layer forward while pLLM automatically computes approximate K/V on GPU using the skipped layer's norm and projections so subsequent tokens always see a contiguous KV cache. Branching executes entirely on GPU via CUDA Graph conditional nodes with zero GPU-to-CPU syncs. See [12 Adaptive Depth](/pllm/docs/examples/12_adaptive_depth/).

```python
from dataclasses import dataclass

import torch

import pllm
from pllm import testing


@dataclass(frozen=True)
class SkipMiddle(pllm.Workload):
    """Token-choice adaptive depth: decide layer skipping per row during decode only."""

    threshold: float = 2.0

    @pllm.custom_forward(tokens="decode")
    def forward(self, layers: pllm.model.Layers) -> torch.Tensor:
        h12 = layers(0, 12)[11]
        h16 = layers(12, 16, hidden=h12)[15]
        moved = (
            torch.nn.functional.cosine_similarity(h16.float(), h12.float(), dim=-1).unsqueeze(-1)
            <= self.threshold
        )
        h20 = layers(16, 20, hidden=h16, active=moved)[19]
        return layers(20, hidden=h20)[-1]


check = pllm.testing.check(SkipMiddle(), CONFIG)
```

---

## 5. Early-Exit Layer Skipping Is Restricted to Decode Tokens {#en-sec-5}

### Problem to Solve {#en-problem-to-solve-4}

In adaptive-depth algorithms (such as LayerSkip or CALM), developers sometimes want to extend early exit to the prefill stage so certain prompt tokens exit at shallow layers to reduce Time to First Token (TTFT).

### pLLM's Choice {#en-pllms-choice-4}

The dynamic skip mask `active=` in `@pllm.custom_forward` is strictly restricted to `tokens="decode"`. Passing `active=` under `tokens="all"` or during prefill raises a build-time error:

```text
Use @pllm.custom_forward(tokens="decode") for active=: skipped prompt tokens would leave approximate K/V that every later token reads. Prompt rows run the full model by default; add an unmasked tokens="prefill" method only to change prompt computation (prefer over="span").
```

### Why {#en-why-4}

When a row skips later layers, pLLM backfills approximate K/V for the skipped layers so subsequent tokens still have entries to read at those positions:
- During decode, only that single newly generated position is skipped;
- Every token in the prompt forms the shared context read by all future generated tokens. Skipping layers during prefill replaces the entire context's upper-layer K/V with approximations, compounding error across every subsequent decode step.

Prompts therefore run the full model by default, and `active=` is allowed only with `tokens="decode"`.

### Alternatives {#en-alternatives-4}

- Keep prompts running through all layers to build an accurate KV cache foundation, and apply `@pllm.custom_forward(tokens="decode")` with `layers(..., active=mask)` during autoregressive decode.
- If you genuinely need to alter prefill topology (e.g., looped depth or span-level forward passes), declare an unmasked forward method or a span-specific forward (`over="span"`).

---

## 6. MTP, Speculative Decoding & Outer Decode-Loop Control {#en-sec-6}

### Problem to Solve {#en-problem-to-solve-5}

Multi-Token Prediction (MTP), speculative decoding (draft-model speculation or Medusa multi-head verification), and tree-based rollback speculation aim to predict multiple future tokens per step or roll back generation progress and KV cache positions when draft tokens are rejected.

### pLLM's Choice {#en-pllms-choice-5}

pLLM does not support taking over the outer decode scheduling loop (draft–verify loops, KV cache rollback, or dynamic tree-branch expansion).

### Why {#en-why-5}

pLLM is designed as a **programmable extension substrate inside and around a single-step forward pass**. Its lifecycle covers a single engine step: request admission, forward execution, hidden-state reads/writes, residual rewrites, attention interception, pre/post-sampling hooks, and online backward passes.

Speculative decoding and rollback algorithms fundamentally **alter the outer scheduler's control state machine**: managing draft-model streams, pruning candidate trees, decrementing token position counters, and transactionally rolling back physical KV block tables. Those responsibilities live in the serving scheduler and batching engine's outer loop, outside a single-step modeling substrate.

### Alternatives {#en-alternatives-5}

- To accelerate production throughput via speculative decoding, use vLLM's built-in native speculative decoding support.
- To reshape candidate probabilities or incorporate intermediate-layer predictions on the current step, use `@pllm.add_logits` for additive vocabulary biases or `@pllm.guide_sampling` for contrastive reweighting.
- For multi-round generation, candidate resampling, or verification feedback, drive batched generation and updates from a host feedback loop (see [05 Host Feedback Loops](/pllm/docs/tutorials/05_host_feedback/)).

---

## 7. Pretraining New Architectures from Scratch Is Out of Scope {#en-sec-7}

### Problem to Solve {#en-problem-to-solve-6}

A researcher wants to invent an entirely new neural architecture from scratch—for example, a non-attention token mixer, a redesigned MoE router dispatch system, or pretraining a randomly initialized model from zero.

### pLLM's Choice {#en-pllms-choice-6}

Pretraining new architectures from scratch is outside pLLM's scope.

### Why {#en-why-6}

pLLM exists to solve the engineering challenges of deploying **inference-time and test-time algorithms** (activation steering, online fine-tuning, adaptive compute, fast-weight memory, lookup residuals, etc.) inside a production serving engine.

Pretraining a base model from scratch requires distributed cluster orchestration (Megatron-LM, FSDP, DeepSpeed), streaming petabyte-scale datasets, and fault-tolerant checkpointing—a completely separate engineering stack. Every abstraction in pLLM centers on serving and adapting existing pretrained checkpoints.

### Alternatives {#en-alternatives-6}

- **Attach new sublayers onto a supported base**: Use [`@pllm.custom_residual`](/pllm/docs/guides/forward-structure/), `pllm.model.ActivationPatch`, and `pllm.model.Adapter` to mount fine-tuned or incrementally trained modules onto a supported base model.
- **Load newly admitted hybrid architectures**: Pretrained hybrid architectures from the community (such as Qwen3.5 mixing linear and full attention) work out of the box in pLLM once admitted to the support matrix.
- **Online test-time learning**: Test-time training (TTT / ESamp / TLM) is first-class in pLLM via `@pllm.train` and `@pllm.train_in_forward`; see [Online Training](/pllm/docs/guides/training/).

---

## 8. Admission by Exact Native Model Class {#en-sec-8}

### Problem to Solve {#en-problem-to-solve-7}

A developer passes a custom model wrapper, a derived subclass (such as a class inheriting from `Qwen2ForCausalLM`), or an unregistered class from a related family (such as Phi-3 or Mistral variants).

### pLLM's Choice {#en-pllms-choice-7}

pLLM validates admission strictly by exact native vLLM model class (`type(model) in architectures`), rejecting any unregistered subclass:

```text
choose a native vLLM model class; workload admission does not accept model subclasses
```

### Why {#en-why-7}

pLLM's efficiency and zero-copy interception rely on strict structural invariants of the admitted model classes:

1. **Residual interception contract**: Intercepting per-layer activations relies on admitted families following a uniform sublayer convention—specifically, $$2L+1$$ LayerNorm / RMSNorm modules that accept `(branch, residual)` and return `(normalized, sum)`. Any unadmitted subclass that changes norm signatures or call order breaks residual mounting.
2. **Projection naming and layout**: `pllm.model.Adapter` and weight overlays bind directly to exact projection attributes (`q_proj`, `k_proj`, `v_proj`, `gate_proj`, etc.). Custom subclasses frequently fuse projections (e.g., packing QKV into one matrix), alter tensor alignment, or change the LM Head. Without strict allowlist admission, mismatched tensor slicing would crash CUDA Graph capture or silently read/write the wrong memory offsets.

### Alternatives {#en-alternatives-7}

- Use an admitted native vLLM model class from the [Support Matrix](/pllm/docs/reference/support-matrix/) (e.g., `Qwen2ForCausalLM`, `LlamaForCausalLM`, `Qwen3ForCausalLM`, `Qwen3MoeForCausalLM`).
- If your checkpoint is structurally compatible with an admitted family, ensure the `architectures` field in `config.json` names the standard native vLLM class.

---

## 9. Fail-Closed Guarantees & Explicit Expensive Tradeoffs {#en-sec-9}

### Problem to Solve {#en-problem-to-solve-8}

In complex experiments or deployments, when memory runs tight, graph capture hits an obstacle, or component dependencies collide, traditional ML frameworks often resort to "silent fallbacks"—quietly disabling CUDA graphs in favor of slow eager execution, spilling tensors to CPU RAM, or silently dropping updates.

### pLLM's Choice {#en-pllms-choice-8}

pLLM enforces a **fail-closed** policy: any semantic ambiguity, undeclared budget overflow, or incompatible combination raises an explicit error at build time (`pllm.explain` or `pllm.make_llm`) instead of silently switching execution paths. The sole automatic fallback is CUDA graph capture: under the default `cuda_graphs="auto"`, certain patterns requiring custom sliced scheduling (such as `@pllm.custom_forward`, where pLLM captures per-layer-slice graphs itself) let vLLM skip the whole-model graph, with the reason reported on `explain`'s `forward` line and in `eager_reason`; setting `cuda_graphs=True` makes that a hard error too. Expensive system tradeoffs must be declared explicitly in code:
- **Storage placement**: large lookup tables in GPU VRAM (`placement="device"`) vs. host DRAM (`placement="host"`);
- **Physical KV layout**: sacrificing positions to save memory (`keep="window"`, `sink=True`, or `keep="none", source=k`);
- **Training visibility lag**: strict blocking visibility (`publish_lag=0`) vs. one-step delayed publishing for full async overlap with the next forward (`publish_lag=1`).

### Why {#en-why-8}

In high-concurrency serving, silent fallbacks are catastrophic:
- Silently spilling a large lookup table to host RAM over PCIe slows every step by an order of magnitude, cratering production throughput with no obvious root cause;
- Silently dropping CUDA Graphs turns small-model or low-concurrency decode throughput down by half (on an RTX 4090 at concurrency 128, 0.5B drops to 49% of graph throughput; 7B steps are longer and lose ~2%; see [Performance & Memory](/pllm/docs/explain/performance/));
- Silently delaying or dropping online weight updates degrades output quality irreproducibly, leaving researchers unable to tell whether their algorithm is flawed or the runtime cut corners.

### Alternatives {#en-alternatives-8}

- Run `pllm.explain(workload, config)` on CPU before launching the GPU engine to inspect the full execution plan and memory bill (see [Inspecting Costs: `pllm.explain`](/pllm/docs/guides/explain/)).
- Configure expensive tradeoffs explicitly for your workload: pass `ExecutionOptions(publish_lag=0)` when strict step-by-step update visibility is required, or set `placement="host"` / static KV layouts when VRAM is tight.

---

## 10. At Most One `@pllm.guide_sampling` Method {#en-sec-10}

### Problem to Solve {#en-problem-to-solve-9}

In a composed workload (for instance, combining two algorithm submodules), a developer attaches `@pllm.guide_sampling` to two different methods so both models or feature streams can guide the final sampling distribution.

### pLLM's Choice {#en-pllms-choice-9}

A workload (including all nested components) may declare at most one `@pllm.guide_sampling` method. Declaring two or more fails at compile time:

```text
... and ... both declare @pllm.guide_sampling; keep at most one guidance method
```

### Why {#en-why-9}

This differs fundamentally from `@pllm.add_logits`:
- `@pllm.add_logits` simply adds a bias vector to unnormalized pre-filter logits. Addition is commutative and associative ($$A + B = B + A$$), so applying multiple additive biases in any order yields the exact same result—pLLM allows any number of `@pllm.add_logits` methods to coexist.
- `@pllm.guide_sampling` performs **nonlinear distribution reweighting** inside the sampler after top-k / top-p truncation (via the contrastive guidance rule `(1 + beta) * logits - beta * pred_logits`). Given two distinct hidden-state predictions, should they be averaged in feature space, multiplied in probability space, or cascaded through two separate truncations? Each choice defines a different algorithm, and pLLM refuses to guess.

### Alternatives {#en-alternatives-9}

- Combine multiple guidance signals explicitly inside a single method—read the layers or modules you need, apply your weighting, interpolation, or gating math, and return one unified tensor;
- For simple additive vocabulary score adjustments, use stackable `@pllm.add_logits` methods instead.

```python
@dataclass(frozen=True)
class VocabularyBiases(pllm.Workload):
    """Multiple @pllm.add_logits methods can coexist because vocabulary bias addition is commutative."""

    @pllm.add_logits()
    def shallow_bias(self, h: torch.Tensor = pllm.hidden(0)) -> torch.Tensor:
        return torch.zeros((h.shape[0], 151936), device=h.device, dtype=torch.float32)

    @pllm.add_logits()
    def deep_bias(self, h: torch.Tensor = pllm.hidden(-1)) -> torch.Tensor:
        return torch.zeros((h.shape[0], 151936), device=h.device, dtype=torch.float32)


bias_report = pllm.explain(VocabularyBiases(), CONFIG)
```

---

## 11. Prefix Caching Must Be Disabled When Rewriting Forward Execution {#en-sec-11}

### Problem to Solve {#en-problem-to-solve-10}

In multi-turn chat or long system-prompt workloads, a developer wants to enable prefix caching (`enable_prefix_caching=True`) alongside `@pllm.custom_residual`, `@pllm.custom_attention`, parameter overlays (`pllm.model.Adapter`), or activation patches (`pllm.model.ActivationPatch`).

### pLLM's Choice {#en-pllms-choice-10}

Whenever a workload rewrites internal model computation sites, prefix caching must stay disabled (`enable_prefix_caching=False`, the default). Enabling prefix caching alongside forward rewrites is rejected at admission time:

```text
disable prefix caching (enable_prefix_caching=False) when a workload rewrites model sites; cached KV would not match
```
or
```text
disable prefix caching when using parameter overlays or activation patches
```

### Why {#en-why-10}

Prefix caching assumes that "identical input token sequences on the same model always produce bitwise-identical KV caches," allowing cached KV pages to be reused across requests solely by hashing token IDs.

When your workload dynamically modifies forward computation:
1. Activation patches or residual rewrites make the KV tensors written at each layer depend not only on token IDs, but also on the request's dynamic state, online-trained weight version, or injected steering vector;
2. With prefix caching enabled, a later request with the same prompt would silently reuse stale KV pages computed under a different request's state, corrupting the forward pass and breaking mathematical correctness.

### Alternatives {#en-alternatives-10}

- Keep `ModelConfig(enable_prefix_caching=False)` (pLLM's default).
- Pure read-only observation workloads (extracting probe features with `@pllm.export`) and pure vocabulary-bias workloads (`@pllm.add_logits`) do not alter in-layer forward activations or KV generation, so prefix caching may be safely enabled with them.

---

See the [Error Catalog](/pllm/docs/reference/errors/) for the exact error messages and fixes corresponding to these boundaries.

---

## Algorithms Are Not pLLM's Job: pLLM Provides the Substrate {#en-algorithms-are-not-pllms-job-pllm-provides-the-substrate}

When evaluating whether an algorithm can run efficiently inside a serving engine, the key mental model is separating **algorithm math** from **system substrate**:

- **Algorithm math belongs to you**: projection formulas, hidden-state interpolations, loss functions, and even custom Triton kernels are yours to define freely.
- **System substrate belongs to pLLM and vLLM**: tensor memory layouts, static CUDA Graph capture, continuous-batch packing, and strict cross-request memory isolation form the foundation that guarantees high throughput, low latency, and deterministic outputs.

When an algorithm hits a boundary today because it conflicts with physical or serving constraints, that boundary marks where the algorithm's assumptions need to align with high-throughput concurrent inference. Understanding these tradeoffs helps you write valid `pllm.Workload`s and design new algorithms with serving performance in mind from day one.

---

## Self-Check {#en-self-check}

1. **Why does pLLM allow multiple `@pllm.add_logits` methods in a workload, yet at most one `@pllm.guide_sampling` method?**
   <details markdown="1">
   <summary>Answer</summary>

   <code>@pllm.add_logits</code> simply adds bias vectors to pre-filter vocabulary logits; addition is commutative and associative ($$A + B = B + A$$), so the result is uniquely defined regardless of order. By contrast, <code>@pllm.guide_sampling</code> performs nonlinear contrastive reweighting ($$(1 + \beta)\text{logits} - \beta\,\text{pred\_logits}$$) on a distribution already truncated by top-k / top-p filtering. Combining two predictions—averaging features, summing log-probabilities, or cascading truncations—represents distinct algorithmic semantics, which pLLM refuses to guess.

   </details>

2. **If a researcher wants to drop unimportant prompt tokens to reduce Time to First Token (TTFT), why can't they pass an `active=` mask for prompt rows in `@pllm.custom_forward`?**
   <details markdown="1">
   <summary>Answer</summary>

   When a row skips a layer via early exit, pLLM backfills approximate K/V entries using that layer's projections on the row's current hidden state so the KV cache stays contiguous. Prompts span tens to thousands of tokens; skipping layers during prefill replaces whole stretches of upper-layer context KV with synthetic approximations. Over hundreds of subsequent decode steps, every generated token would attend over distorted prompt KV, rapidly degrading generation quality.

   </details>

3. **Why is enabling Prefix Caching mutually exclusive with `@pllm.custom_residual` or parameter overlays?**
   <details markdown="1">
   <summary>Answer</summary>

   Prefix caching assumes that identical prompt token sequences always produce identical KV caches and indexes reusable pages solely by token-ID hashes. When <code>@pllm.custom_residual</code> or parameter overlays alter forward computation, the resulting KV values depend on dynamic per-request state, online-trained weight versions, or injected steering vectors. Enabling prefix caching would cause new requests with matching prompts to reuse stale KV pages produced under another request's state, violating mathematical correctness.

   </details>

---

## Where to Go Next {#en-where-to-go-next}

- Browse the full feature feasibility checklist: [Capability Map](/pllm/docs/guides/capability-map/)
- Review core physical mechanisms and the step timeline: [Core Concepts](/pllm/docs/explain/core-concepts/)
- Inspect measured throughput and memory overhead across mechanisms: [Performance & Memory](/pllm/docs/explain/performance/)
- Learn how to write valid custom-attention reads: [Customizing Forward Structure](/pllm/docs/guides/forward-structure/) and [10 Custom Attention](/pllm/docs/examples/10_custom_attention/)
- Learn how to implement token-choice adaptive depth: [12 Adaptive Depth](/pllm/docs/examples/12_adaptive_depth/)
- Look up full error messages and fixes: [Error Catalog](/pllm/docs/reference/errors/)

</div>

<div class="zine-lang-panel is-hidden" data-lang-panel="zh" lang="zh-CN" markdown="1">

<div id="zh-sec-1"></div>

在独立 PyTorch 脚本或 Hugging Face `generate` 循环中，研究员只需几行 `register_forward_hook`、在循环体里插入一段条件分支，或者对张量调用 `.item()`，就能迅速验证一项新算法构想。然而，当尝试把同样的算法直接搬入以连续批处理（continuous batching）和 CUDA Graph 异步回放为基石的高吞吐服务引擎（如 vLLM）时，往往会撞上各种各样的物理与架构限制。

pLLM 的定位不是一个收录海量现成算法的黑盒算法库，而是连接前沿建模算法与现代服务引擎之间的**可编程状态化底座（substrate）**。底座的职责是把连续批处理打包、CUDA Graph 静态图捕获、版本化快照发布以及显存生命周期规划等底层工程脏活接管下来，让能够与现代高吞吐物理机制相容的算法以极致性能运行。

当某种算法的设计假设与底层物理机制（固定内存形状、分页块表、流式异步执行、请求间严格隔离）发生根本冲突时，pLLM 明确选择**不静默妥协、不隐式降级**，而是将系统边界和数学原因清晰地暴露给开发者。这一页梳理了目前不支持或受限的 11 项典型设计决策。每一节都按照统一的逻辑展开：**要解决的问题 $$\to$$ pLLM 的选择 $$\to$$ 为什么这样选 $$\to$$ 替代方案**。

---

## 1. 按分数驱逐 KV 显存与 KV Cache 的连续性（H2O、SnapKV） {#zh-sec-2}

### 要解决的问题 {#zh-sec-3}

在长序列生成中，KV 缓存的物理显存占用与上下文长度呈线性增长。以 H2O、SnapKV、PyramidKV 为代表的“按分数动态驱逐”算法，希望在自回归生成过程中持续统计每个历史 token 的累积注意力分数，并据此在物理显存中直接释放那些注意力权重低于阈值的“不重要”token，仅保留开头若干注意力和关键 heavy hitters，从而在理论上把显存占用压到极低水平。

### pLLM 的选择 {#zh-sec-4}

pLLM 不支持按注意力分数动态释放 KV 物理显存。

### 为什么这样选 {#zh-sec-5}

vLLM 的底层显存管理模型（PagedAttention）建立在一项核心约定之上：**每个请求的 KV 缓存是逻辑上从位置 0 开始的一整段连续序列**。

1. **逻辑计数与槽位绑定**：引擎仅用一个标量整数记录该请求“已经计算了多少个 token”。每一层每个 token 应该写在物理页块表的哪一页、哪一槽，以及旋转位置编码（RoPE）对应的绝对位置，都是由这个连续递增的计数直接推出的。每个请求在 vLLM 内部的块表只会向后追加分配新页，从不设计从中间抠除某一槽位的机制。
2. **中间留洞破坏分页索引**：如果根据注意力分数丢弃序列中间任意分散的 token，就会在这段连续内存中形成“空洞”。这不仅需要彻底重构底层 PagedAttention 算子的块查找与步长计算逻辑，而且每一步驱逐哪些 token 依赖于当前步运行时的数值计算，无法提前推断。
3. **破坏 CUDA Graph 捕获**：如果在解码的每一步动态释放中间页块、重新压缩排列块表，就必须在 GPU 与 CPU 之间触发阻塞式的元数据同步，或者破坏 CUDA Graph 所依赖的静态张量指针与内存布局，直接摧毁图模式的吞吐收益。

### 替代方案 {#zh-sec-6}

- **稀疏读取（省算力与访存带宽，不省显存容量）**：使用 [`@pllm.custom_attention`](/pllm/docs/guides/forward-structure/)，在方法体内部调用 `kv.attend(q, topk=...)` 或自定义打分逻辑（如 Quest 算法）。在每一层每个注意力头内，只挑选得分最高的前 $$K$$ 个位置参与注意力计算，其余位置直接跳过读取。这能成倍降低注意力内核的 FLOP 与访存带宽，完全在 GPU 上运行且兼容 CUDA Graph，详见 [10 自定义注意力](/pllm/docs/examples/10_custom_attention/)。
- **规则驱动的规整物理释放（真正释放物理显存）**：如果目标是真正降低单请求的显存占用上限，应使用与底层块管理模式对齐的静态 KV 布局。例如滑动窗口（`keep="window", window=N`）、保留 prompt 加最近生成窗口（`sink=True`，即 R-SWA），或者奇数层直接复用偶数层 KV 的跨层共享（`keep="none", source=k`）。这些规则在块的边界对齐释放，不需要在块内挖洞。

```python
from dataclasses import dataclass

import torch

import pllm
from pllm import testing


@dataclass(frozen=True)
class LocalGlobalAttention(pllm.Workload):
    """偶数层只保留最近 256 个 token 的 KV（滑动窗口），奇数层保留全部上下文。"""

    window_size: int = 256

    def window(self) -> int:
        return self.window_size

    @pllm.custom_attention(layers=tuple(range(0, 24, 2)), keep="window", window=window)
    def read(self, q: torch.Tensor, kv: pllm.model.PagedKV) -> torch.Tensor:
        return kv.attend(q, window=self.window_size)[0]


CONFIG = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=4)
report = pllm.explain(LocalGlobalAttention(), CONFIG)
```

---

## 2. 压缩远端记忆与分页填充损耗（Activation Beacon） {#zh-sec-7}

### 要解决的问题 {#zh-sec-8}

Activation Beacon 或各类型压缩 KV 算法试图将远端历史上下文（例如过去成百上千个 token）压缩为少量浓缩的表示（Summary Tokens 或压缩条目），以此兼顾长上下文检索与有限的显存容量。

### pLLM 的选择 {#zh-sec-9}

pLLM 目前未将压缩远端记忆（compressed remote memory）集成到底座的标准流水线中。

### 为什么这样选 {#zh-sec-10}

将压缩远端记忆深度集成到连续批处理引擎中存在三项物理与算法层面的根本障碍：

1. **分页格式的块填充损耗（Page padding）**：vLLM 的显存按固定大小的“页”（Block，通常为 16 或 32 个 token）分配。将远端 token 深度压缩后存入分页结构时，若压缩比很高（例如每 16 个 token 压缩为 1 个条目），剩余条目仍须补齐整页。整页补齐之后，实际物理显存并未得到实质性削减，反而引入了严重的内存碎片。
2. **分块调度的非确定性（Chunk-dependent state）**：若远端记忆的生成依赖较深层的隐藏状态，在开启分块预填充（`enable_chunked_prefill`）的并发场景下，某个位置的压缩条目在当前步是否已经计算完毕，取决于系统调度器在当前步把 prompt 划分为多大的分块。不同的并发压力会导致切块边界动态变化，进而导致同一输入在不同负载下的压缩特征发生漂移，破坏推理的确定性。
3. **双段注意力的延迟惩罚（Two-pass attention overhead）**：在解码阶段，将近处窗口的全量 KV 与远处的压缩条目分两次执行注意力内核再在 Python/Triton 中拼接合并，其核函数启动开销与访存重排成本远高于直接执行一次原生的全量注意力读（在单请求单层微基准测试中，分两次读合并耗时约 30.31 微秒，而原生全量读仅需 11.88 微秒）。

### 替代方案 {#zh-sec-11}

- **固定布局削减**：使用滑动窗口（`keep="window"`）或跨层共享（`keep="none", source=k`），它们能够直接按整页物理回收显存，且不引入双阶段注意力的额外调度开销。
- **查表先验注入**：如果需要模型具备长程背景记忆或跨文档先验知识，推荐使用按 token 历史检索的查表记忆机制（参见 [11 查表记忆](/pllm/docs/examples/11_lookup_memory/) 中的 Engram 结构）。通过 `pllm.Pretrained(..., placement="host")` 把大规模预存记忆放在主机内存或独立存储中，前向时跨 PCIe 异步预取并叠加在残差流上，避免干扰 KV 缓存流水线。

---

## 3. DSA 的 Top-K 索引复用与 sm90 硬件要求 {#zh-sec-12}

### 要解决的问题 {#zh-sec-13}

在 DeepSeek 稀疏注意力（DSA，如 DeepSeek V3/V4 系列）中，模型使用轻量索引器提前计算 query 与 key 的匹配程度并选出 top-k 个最重要的块或 token。研究员通常希望直接读取该索引器的选择结果（IndexCache），并在后续若干层中直接复用这一索引，避免重复计算打分。

### pLLM 的选择 {#zh-sec-14}

pLLM 目前不支持读取并跨层复用底层原生 DSA 的 top-k 索引结果。

### 为什么这样选 {#zh-sec-15}

DSA 是模型结构本身的一部分，要在训练出 DSA 的模型（DeepSeek V3.2、V4 一类）上才有可读的选择结果；而 vLLM 运行这些模型的 attention 路径只支持 sm90 及以上的 GPU（H100 一类）。在 sm80（A100）、sm89（RTX 4090）上根本跑不起来这类模型，也就没有可以复用的 top-k。`pllm.attention(layer, ...)` 今天只提供 `"aux"` 和 `"lse"` 两种读法。

### 替代方案 {#zh-sec-16}

- 在 sm80/sm89 硬件上，可以在 `@pllm.custom_attention` 中使用自定义 PyTorch 或 Triton 算子实现注意力过滤，例如通过 [09 自写算子](/pllm/docs/examples/09_custom_kernels/) 配合 [10 自定义注意力](/pllm/docs/examples/10_custom_attention/) 中的 `kv.attend(q, topk=...)` 显式计算并指定位置。
- 若只需跨层降低计算量，可直接在模型结构上采用跨层共享 KV 布局（`keep="none", source=k`）。

---

## 4. 批次竞争式路由（MoR Expert-Choice）与原版 MoD {#zh-sec-17}

### 要解决的问题 {#zh-sec-18}

Mixture-of-Depths (MoD) 与 Mixture-of-Recursions (MoR) 旨在为不同的 token 分配不同的计算深度。其中：
1. **MoR 的 expert-choice 路由**：让每一层在整个 batch 内部展开自顶向下的竞争，挑选分数排在前 $$k$$ 名的 token 执行计算；
2. **原版 MoD**：被跳过的 token 完全不进入该层计算，也不在该层写入任何 KV。

### pLLM 的选择 {#zh-sec-19}

pLLM 明确不支持批次竞争式的 expert-choice 路由，也不支持原版 MoD 这种完全不写 KV 的跳层模式；pLLM 支持的是基于每行独立掩码的 **token-choice 自适应深度** 与 **早退（Early Exit）**。

### 为什么这样选 {#zh-sec-20}

1. **绝对禁止破坏请求隔离（Request Isolation）**：在连续批处理中，同一个二维张量里打包着不同客户端发来的完全独立的请求行。如果在整个 batch 里做 top-k 竞争，那么某个 token 到底算不算这一层，将取决于这一瞬间刚好有哪些并发请求被打包在同一个 batch 里。当并发数增减或排队顺序发生毫秒级变化时，同一个 prompt 生成的文本内容就会发生改变，彻底破坏了系统的幂等性与确定性契约（`pllm.testing.check` 会直接捕获并抛出 `ContractError: method ... mixes rows`）。
2. **KV 序列不可留洞**：在原版 MoD 中，如果第 $$t$$ 个 token 跳过了第 $$L$$ 层且不写入任何 KV，第 $$L$$ 层的 KV 序列在位置 $$t$$ 就会出现缺失。当后续第 $$t+1$$ 个 token 在第 $$L$$ 层做自回归注意力时，因果掩码和分页块表预期能够回溯完整的历史序列，缺失的位置会导致注意力因果对齐失效，产生与按分数驱逐 KV 相同的系统性崩溃。

### 替代方案 {#zh-sec-21}

- **使用 token-choice 自适应路由**：每个 token 仅根据自身的隐藏特征或局部路由判定网络独立决定是否执行某一层（例如通过 sigmoid 阈值独立过滤），不与其他请求争抢配额。
- **GPU 端自动补齐 KV 并在图内调度**：在 `@pllm.custom_forward` 中传入动态布尔掩码 `active=mask`（形状为 `[行, 1]`）。未激活的行跳过该层前向，而 pLLM 会在 GPU 上通过被跳过层的归一化和投影自动计算近似的 K/V 并写入缓存，确保后续 token 永远读到结构完整的 KV 序列；同时，跳层判定完全留在 GPU 上通过 CUDA Graph 的分组分支节点执行，不产生任何 GPU 到 CPU 的同步阻断。详见 [12 自适应深度](/pllm/docs/examples/12_adaptive_depth/)。

```python
from dataclasses import dataclass

import torch

import pllm
from pllm import testing


@dataclass(frozen=True)
class SkipMiddle(pllm.Workload):
    """token-choice 自适应深度：仅在 decode 阶段根据自身状态决定跳层。"""

    threshold: float = 2.0

    @pllm.custom_forward(tokens="decode")
    def forward(self, layers: pllm.model.Layers) -> torch.Tensor:
        h12 = layers(0, 12)[11]
        h16 = layers(12, 16, hidden=h12)[15]
        moved = (
            torch.nn.functional.cosine_similarity(h16.float(), h12.float(), dim=-1).unsqueeze(-1)
            <= self.threshold
        )
        h20 = layers(16, 20, hidden=h16, active=moved)[19]
        return layers(20, hidden=h20)[-1]


check = pllm.testing.check(SkipMiddle(), CONFIG)
```

---

## 5. 早退跳层仅限 Decode Token {#zh-sec-22}

### 要解决的问题 {#zh-sec-23}

在自适应深度算法（如 LayerSkip、CALM 等）中，开发者有时希望将早退逻辑推广到 prefill（处理 prompt）阶段，让 prompt 中的部分 token 在浅层及早退出以缩短首字延迟（TTFT）。

### pLLM 的选择 {#zh-sec-24}

`@pllm.custom_forward` 中的动态跳层掩码 `active=` 严格限制为仅在 `tokens="decode"` 时生效；若在 `tokens="all"` 或 prefill 阶段尝试传入 `active=`，构建期将抛出异常：

```text
Use @pllm.custom_forward(tokens="decode") for active=: skipped prompt tokens would leave approximate K/V that every later token reads. Prompt rows run the full model by default; add an unmasked tokens="prefill" method only to change prompt computation (prefer over="span").
```

### 为什么这样选 {#zh-sec-25}

当某一行跳过后面的层时，pLLM 要为被跳过的层补上近似的 K/V，否则之后的 token 读不到这些位置。
- decode 时被跳过的只是这一个新位置；
- prompt 的每个 token 都是之后所有生成 token 要读的上下文。在 prefill 时跳层，会把整段上下文在高层的 K/V 换成近似值，误差会影响此后生成的每一步。

所以 prompt 默认跑完整模型，`active=` 只用于 `tokens="decode"`。

### 替代方案 {#zh-sec-26}

- 默认保持 prompt 跑完完整的模型层数，以建立高精度的上下文 KV 缓存基石；仅在解码生成阶段使用 `@pllm.custom_forward(tokens="decode")` 结合 `layers(..., active=mask)` 动态早退。
- 若确实需要在 prefill 阶段改写计算拓扑（例如循环深度或跨度级前向），应声明不带 `active=` 掩码的统一前向方法，或针对特定跨度定制前向（如 `over="span"`）。

---

## 6. MTP、投机解码与解码外环控制 {#zh-sec-27}

### 要解决的问题 {#zh-sec-28}

多 Token 预测（Multi-Token Prediction / MTP）、投机解码（Speculative Decoding，如草稿模型投机或 Medusa 多头投机）以及基于树状搜索的可回滚解码（Rollback / Tree Speculation），希望在单步推理中同时预测多个未来 token，或者根据验证结果将请求的生成进度与已写入的 KV 缓存向后回滚若干步。

### pLLM 的选择 {#zh-sec-29}

pLLM 目前不支持接管外层解码调度循环的算法（包括草稿生成与验证循环、KV 缓存回滚和动态树状分支展开）。

### 为什么这样选 {#zh-sec-30}

pLLM 的定位是**模型前向内部与单步推理周围的可编程扩展底座**。它介入的生命周期是单步推理时间线（从请求准入、前向计算、隐藏状态读写、残差流改写、注意力拦截到采样前后处理与在线梯度反向）。

投机解码与回滚算法的本质是**改变外层调度循环的控制状态机**：它们需要管理草稿模型的调度流、候选树的分支剪枝、生成位置计数器的减小、以及物理 KV 块表的事务性回滚操作。这些控制逻辑属于服务调度器（Scheduler）与批处理引擎的外环状态机，超出了单步前向建模底座的职责范围。

### 替代方案 {#zh-sec-31}

- 若生产环境需要通过投机解码提升整体吞吐，应使用 vLLM 原生内置的投机解码服务架构。
- 若需要在当前步改变候选概率或根据中间层预测修正分布，可以使用 `@pllm.add_logits` 在词表上注入修正偏置，或使用 `@pllm.guide_sampling` 进行对比重加权。
- 若算法包含多轮交互、候选重采样或验证反馈，可以在宿主侧构建反馈循环（参见 [05 反馈循环](/pllm/docs/tutorials/05_host_feedback/)），由外部宿主驱动批量生成与参数更新。

---

## 7. 从头预训练新结构不在职责范围内 {#zh-sec-32}

### 要解决的问题 {#zh-sec-33}

算法研究员可能希望从零开始发明一种全新的神经网络拓扑，例如一种完全摒弃自注意力机制的新型 token mixer、重新设计的底层 MoE 路由分发器，或者用随机初始化从头预训练一个全新架构的模型。

### pLLM 的选择 {#zh-sec-34}

从零开始预训练全新架构（Pretraining from scratch）不在 pLLM 的工作范围内。

### 为什么这样选 {#zh-sec-35}

pLLM 的核心使命在于解决**推理期与测试期算法**（如激活引导、在线微调、自适应计算、快权重记忆、查表残差等）在服务引擎上的落地难题。

从头预训练一个全新的基础模型，需要的是大规模分布式训练集群编排（如 Megatron-LM、FSDP、DeepSpeed）、海量数据集流式加载与容错检查点系统，这是一套完全不同的工程技术栈。pLLM 的全部抽象都围绕“现有预训练权重的服务与适配”展开。

### 替代方案 {#zh-sec-36}

- **在现有基座上接入新子层**：通过 [`@pllm.custom_residual`](/pllm/docs/guides/forward-structure/)、`pllm.model.ActivationPatch`、`pllm.model.Adapter` 等接口，把基于现有基座微调或增量训练好的权重挂载到模型中运行。
- **支持新型已准入混合架构**：对于社区中已经完成预训练的新型混合结构（例如线性注意力与全注意力混合的 Qwen3.5），只要其已被列入支持矩阵，即可直接在 pLLM 中加载使用。
- **测试期在线学习**：测试期训练（Test-Time Training / TTT / ESamp / TLM）完全受支持，可通过 `@pllm.train` 或 `@pllm.train_in_forward` 在推理过程中持续更新状态，详见 [在线训练](/pllm/docs/guides/training/)。

---

## 8. 按精确原生模型类准入 {#zh-sec-37}

### 要解决的问题 {#zh-sec-38}

开发者尝试传入自定义的模型包装类、派生子类（如继承自 `Qwen2ForCausalLM` 的子类），或者来自相似模型家族（如 Phi3、Mistral）的非登记类。

### pLLM 的选择 {#zh-sec-39}

pLLM 严格按照 vLLM 的具体原生模型类进行准入验证（`type(model) in architectures`），明确拒绝任何未登记的模型子类：

```text
choose a native vLLM model class; workload admission does not accept model subclasses
```

### 为什么这样选 {#zh-sec-40}

pLLM 的高效与零拷贝能力建立在底层模型精细结构的不变性契约之上：

1. **残差流拦截点契约**：pLLM 在模型内部各层拦截激活时，依赖于被准入家族统一遵循的子层规范——即 $$2L+1$$ 个 LayerNorm / RMSNorm 必须严格接受 `(branch, residual)` 二元组并返回 `(normalized, sum)`；任何未经准入的子类一旦修改了 Norm 签名或调用顺序，残差挂载将完全失效。
2. **权重投影的命名与排布**：`pllm.model.Adapter` 和模型权重重写直接绑定到模型具体的投影属性（如 `q_proj`、`k_proj`、`v_proj`、`gate_proj` 等）。许多衍生模型或自定义子类会合并投影（例如将 QKV 打包为单个矩阵）、改变张量对齐或改变 LM Head 的计算方式。如果不做白名单式的精确类准入，一旦底层张量切分不匹配，不仅会导致 CUDA Graph 录制崩溃，甚至可能在显存中读写错误偏移的数据。

### 替代方案 {#zh-sec-41}

- 使用 [支持矩阵](/pllm/docs/reference/support-matrix/) 中已准入的原生模型架构（如 `Qwen2ForCausalLM`、`LlamaForCausalLM`、`Qwen3ForCausalLM`、`Qwen3MoeForCausalLM` 等）。
- 若权重属于兼容家族，确保模型目录下的 `config.json` 中的 `architectures` 字段指向标准的 vLLM 原生类名称。

---

## 9. 拒绝静默回退（Fail-Closed）与显式昂贵取舍 {#zh-sec-42}

### 要解决的问题 {#zh-sec-43}

在复杂的实验环境或服务部署中，当出现显存不足、图捕获受阻、或者组件之间产生依赖竞争时，传统深度学习库往往倾向于“静默回退（Silent Fallback）”——例如悄悄关掉 CUDA Graph 退回极其缓慢的 eager 解释执行、悄悄把张量从 GPU 溢写到 CPU 内存、或者默默忽略某些更新。

### pLLM 的选择 {#zh-sec-44}

pLLM 坚持**闭合失败（Fail-Closed）**原则：任何语义歧义、未声明的超额资源占用或不相容的组合，在构建阶段（`pllm.explain` 或 `pllm.make_llm`）立刻抛出异常明确拒绝，不悄悄换一条路径执行。唯一的自动退让是 CUDA graph：默认 `cuda_graphs="auto"` 下，个别需要自定义切片调度的写法（例如 `@pllm.custom_forward` 改由 pLLM 按层切片独立捕获图）会让 vLLM 跳过整模型图，原因写在 `explain` 的 `forward` 行和 `eager_reason` 里，设 `cuda_graphs=True` 则直接报错。昂贵的系统取舍必须由开发者在代码中显式声明：
- **存储位置**：大查找表是放 GPU 显存还是 CPU 主机内存（`placement="device"` 还是 `"host"`）；
- **KV 物理布局**：是否牺牲某些位置释放显存（`keep="window"`、`sink=True` 还是 `keep="none", source=k`）；
- **训练结果可见滞后**：参数更新是严格无滞后阻塞等待（`publish_lag=0`），还是隔步发布以与下一步前向完全异步重叠（`publish_lag=1`）。

### 为什么这样选 {#zh-sec-45}

在现代高并发在线服务中，静默回退是灾难性的：
- 如果大查找表因为显存不足被系统静默搬移到主机内存，跨 PCIe 传输会把每一步的延迟拖慢数十倍，线上吞吐断崖式下跌，且排查时极难定位根因；
- 如果系统在遇到小算子时悄悄关闭 CUDA Graph，小模型或低并发下解码吞吐会跌掉一半（RTX 4090 上 0.5B、128 并发掉到 49%；7B 一步更长，只慢约 2%，见 [性能与显存](/pllm/docs/explain/performance/)）；
- 如果在线学习或快权重更新被框架悄悄推迟或丢弃，模型生成质量将出现难以复现的劣化，算法研究员根本无法区分“是自己的算法理论不对”还是“框架在底层偷工减料”。

### 替代方案 {#zh-sec-46}

- 在启动 GPU 服务之前，先在 CPU 上调用 `pllm.explain(workload, config)` 查看完整的执行计划与显存账单（参见 [看成本：pllm.explain](/pllm/docs/guides/explain/)）。
- 根据算法的物理代价主动显式配置：若需要严格的更新时序，显式传入 `ExecutionOptions(publish_lag=0)`；若显存吃紧，主动配置 `placement="host"` 或调整静态 KV 布局。

---

## 10. 采样引导方法仅限单一来源 {#zh-sec-47}

### 要解决的问题 {#zh-sec-48}

开发者在一个复合负载（例如组合了两个独立算法子模块）中，尝试在两个不同的方法上同时添加 `@pllm.guide_sampling` 装饰器，希望两个模型或两套特征同时引导最终的采样分布。

### pLLM 的选择 {#zh-sec-49}

一个负载（含所有嵌套子组件）中最多只能包含一个 `@pllm.guide_sampling` 方法。若声明了两个或更多，编译期将直接报错：

```text
... and ... both declare @pllm.guide_sampling; keep at most one guidance method
```

### 为什么这样选 {#zh-sec-50}

这与 `@pllm.add_logits` 的性质有着本质区别：
- `@pllm.add_logits` 只是在过滤前向未归一化的 logits 上累加一个偏置向量。加法运算满足交换律与结合律（$$A + B = B + A$$），多个修改先后相加，数学结果完全一致，因此 pLLM 允许任意多个 `@pllm.add_logits` 并存。
- `@pllm.guide_sampling` 则是运行在采样器内部、经过 top-k / top-p 截断过滤之后的**非线性分布重加权**（基于对比引导公式 `(1 + beta) * logits - beta * pred_logits`）。当存在两个不同的隐藏层预测向量时，它们应该在特征空间先求平均？还是在概率空间先做乘法？亦或是先后级联应用两次不同的截断？不同的组合方式代表着完全不同的算法语义。pLLM 坚决不替开发者做任何主观猜想。

### 替代方案 {#zh-sec-51}

- 将多个引导信号的融合逻辑显式写在同一个方法中：读取多个隐藏层或模块输出，按照你的算法设计完成加权、插值或门控计算后，返回一个统一的张量；
- 如果只需简单的词表得分增减，改用可多重叠加的 `@pllm.add_logits`。

```python
@dataclass(frozen=True)
class VocabularyBiases(pllm.Workload):
    """多个 @pllm.add_logits 可以并存，因为词表偏置相加满足交换律。"""

    @pllm.add_logits()
    def shallow_bias(self, h: torch.Tensor = pllm.hidden(0)) -> torch.Tensor:
        return torch.zeros((h.shape[0], 151936), device=h.device, dtype=torch.float32)

    @pllm.add_logits()
    def deep_bias(self, h: torch.Tensor = pllm.hidden(-1)) -> torch.Tensor:
        return torch.zeros((h.shape[0], 151936), device=h.device, dtype=torch.float32)


bias_report = pllm.explain(VocabularyBiases(), CONFIG)
```

---

## 11. 改写前向时关闭前缀缓存（Prefix Caching） {#zh-sec-52}

### 要解决的问题 {#zh-sec-53}

在多轮对话或系统提示词（System Prompt）较长的场景下，开发者希望开启前缀缓存（`enable_prefix_caching=True`），同时又在负载中使用了 `@pllm.custom_residual`、`@pllm.custom_attention`、参数覆盖（`pllm.model.Adapter`）或激活补丁（`pllm.model.ActivationPatch`）来改写模型的前向计算。

### pLLM 的选择 {#zh-sec-54}

当负载改写了模型内部计算节点时，必须保持前缀缓存关闭（`enable_prefix_caching=False`，默认值）；若在改写前向的同时开启了前缀缓存，系统在准入期直接报错拦截：

```text
disable prefix caching (enable_prefix_caching=False) when a workload rewrites model sites; cached KV would not match
```
或
```text
disable prefix caching when using parameter overlays or activation patches
```

### 为什么这样选 {#zh-sec-55}

前缀缓存（Prefix Caching）的核心原理是假定“相同的输入 token 序列，在同一个模型上必定产生逐位完全相同的 KV Cache”，因此只需对 token id 序列计算哈希值即可跨请求复用已有的 KV 页。

但是，当你的负载包含动态修改前向的逻辑时：
1. 激活补丁或残差改写使得每一层写入 KV 缓存的张量数值不仅取决于 token id，还取决于该请求当前的动态状态、在线更新的权重版本、或外部注入的引导向量；
2. 如果开启前缀缓存，当后续一个携带相同 prompt 的新请求进入时，引擎会直接复用前一个请求在不同状态下计算出的旧 KV 缓存，导致新请求的前向计算直接建立在脏数据之上，彻底破坏了算法的数学正确性。

### 替代方案 {#zh-sec-56}

- 保持 `ModelConfig(enable_prefix_caching=False)`（这是 pLLM 的默认配置）。
- 对于纯只读观测型负载（如通过 `@pllm.export` 提取探针特征）或纯词表偏置型负载（`@pllm.add_logits`），由于它们并不修改模型层内的前向激活或 KV 生成过程，前缀缓存可以安全开启。

---

这些限制对应的报错原文和改法见[报错目录](/pllm/docs/reference/errors/)。

---

## 算法不是 pLLM 的工作：pLLM 提供的是底座 {#zh-sec-57}

在评估一项新算法能否在服务引擎中高效运行时，一个核心心智模型是区分**算法数学**与**系统基质**：

- **算法数学属于开发者**：你的前向投影公式、隐藏状态插值、损失函数计算、甚至自定义 Triton 核函数，都由你自由定义与编写。
- **系统基质属于 pLLM 与 vLLM**：张量的显存排布、CUDA Graph 的静态拓扑录制、连续批处理的动态打包切分、跨请求的严格内存隔离，是系统保证高吞吐、低延迟与确定性输出的底层防线。

当一个算法因为触碰了物理法则或服务化边界而“今天写不了”时，说明该算法需要与高吞吐并发推理的现实条件做一次明确的理论对齐。理解这些设计取舍，不仅能帮助你写出正确可运行的 `pllm.Workload`，也能让你在设计新算法时，从第一天起就具备服务化与系统级性能的全局视野。

---

## 自测 {#zh-sec-58}

1. **为什么在 pLLM 中可以声明多个 `@pllm.add_logits`，却只能声明一个 `@pllm.guide_sampling`？**
   <details markdown="1">
   <summary>答案</summary>

   <code>@pllm.add_logits</code> 只是在采样前向词表 logits 累加偏置向量，加法运算满足交换律与结合律（$$A + B = B + A$$），无论添加顺序如何，数学结果完全唯一确定。而 <code>@pllm.guide_sampling</code> 是在经过 top-k / top-p 过滤后的分布上执行非线性的对比重加权（基于 $$(1 + \beta)\text{logits} - \beta\,\text{pred\_logits}$$）。如果存在多个预测特征，它们应该在特征空间加权平均、在对数概率空间相加、还是分层级联截断，代表着完全不同的算法意图，pLLM 拒绝替开发者做出未经证实的语义猜测。

   </details>

2. **如果一位研究员希望通过丢弃 prompt 中不重要的 token 来缩减首字延迟（TTFT），为什么不能在 `@pllm.custom_forward` 中直接给 prompt 传入 `active=` 掩码？**
   <details markdown="1">
   <summary>答案</summary>

   当某一行早退跳过某一层时，为了保证 KV 缓存的物理与结构完整，pLLM 会用当前隐藏状态经过该层的投影补齐近似的 K/V。Prompt 通常由数十到数千个连续 token 组成，如果在 prompt 阶段大面积早退，高层网络中整段上下文的 KV 将全被合成的近似值占据；在接下来的数百步自回归解码中，后续生成的每一个 decode token 在做因果注意力时都会读取这些严重失真的 prompt KV，导致生成文本迅速退化。

   </details>

3. **为什么开启前缀缓存（Prefix Caching）与使用 `@pllm.custom_residual` 或参数覆盖是互斥的？**
   <details markdown="1">
   <summary>答案</summary>

   前缀缓存的基本假设是“相同的 prompt token 序列必然产生逐位完全一致的 KV 缓存”，并仅以 token id 的哈希值作为复用依据。而当使用 <code>@pllm.custom_residual</code> 或参数覆盖改写前向计算时，模型层内产生的 KV 数值不仅取决于输入的 token id，还取决于该请求当前的动态状态、在线微调的权重版本或外部注入的引导向量。如果开启前缀缓存，后续相同 prompt 的新请求会错误地复用前一个请求在不同状态下产生的旧 KV 缓存，破坏推理的数学确定性。

   </details>

---

## 下一步去哪 {#zh-sec-59}

- 查阅完整的功能可行性清单：[能力地图](/pllm/docs/guides/capability-map/)
- 了解核心物理机制与执行时间线：[核心概念](/pllm/docs/explain/core-concepts/)
- 查看各项机制的实测吞吐与开销：[性能与显存](/pllm/docs/explain/performance/)
- 学习如何在自定义注意力中编写合法读取：[改 forward 结构](/pllm/docs/guides/forward-structure/) 与 [10 自定义注意力](/pllm/docs/examples/10_custom_attention/)
- 学习如何编写 token-choice 自适应深度：[12 自适应深度](/pllm/docs/examples/12_adaptive_depth/)
- 查看完整的报错信息与修复建议：[报错目录](/pllm/docs/reference/errors/)

</div>

</div>

<div class="zine-doc-footer">
  <a class="zine-pill-link" data-i18n-zh="← 返回 pLLM 项目与文档中心" data-i18n-en="← Back to pLLM Project &amp; Docs" href="/pllm/#docs">← Back to pLLM Project &amp; Docs</a>
  <a class="zine-pill-link" href="https://github.com/LinesHogan/pLLM/blob/main/doc/explain/design-tradeoffs.md" target="_blank" rel="noopener">GitHub &middot; doc/explain/design-tradeoffs.md ↗</a>
</div>

<script defer src="/assets/js/bilingual-post.js"></script>
