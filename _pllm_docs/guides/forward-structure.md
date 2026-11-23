---
layout: page
title: "Editing Forward Structure: Residuals, Attention, and Layer Execution Order"
description: "pLLM Documentation · User Guides"
permalink: /pllm/docs/guides/forward-structure/
nav: false
---

<link rel="stylesheet" href="{{ '/assets/css/pllm-hub.css' | relative_url | bust_file_cache }}">

<div class="zine-dispatch" data-title-zh="改 forward 结构：残差、attention 与层的执行顺序" data-title-en="Editing Forward Structure: Residuals, Attention, and Layer Execution Order" data-desc-zh="pLLM 文档 · 用户指南" data-desc-en="pLLM Documentation · User Guides">
  <div class="zine-dispatch-links">
    <a class="zine-pill-link primary" data-i18n-zh="← 返回 pLLM 文档中心" data-i18n-en="← pLLM Docs Hub" href="/pllm/#docs">← pLLM Docs Hub</a>
    <a class="zine-pill-link" data-i18n-zh="文档导航地图" data-i18n-en="Docs Roadmap" href="/pllm/docs/">Docs Roadmap</a>
    <span class="zine-lang-badge" data-i18n-zh="用户指南" data-i18n-en="User Guides">User Guides</span>
  </div>
  <a class="zine-pill-link" data-i18n-zh="GitHub 源码 ↗" data-i18n-en="View on GitHub ↗" href="https://github.com/LinesHogan/pLLM/blob/main/doc/guides/forward-structure.md" target="_blank" rel="noopener">View on GitHub ↗</a>
</div>

<div id="markdown-content">

<div class="zine-lang-panel" data-lang-panel="en" lang="en" markdown="1">

<div id="en-editing-forward-structure-residuals-attention-and-layer-execution-order"></div>

Modifying a model's computation structure in PyTorch or Hugging Face is straightforward: rewrite `forward()`, insert hooks, or call custom functions between layers. In a throughput-oriented serving engine like vLLM, however, dozens of concurrent requests are packed into a single tensor by continuous batching, and the whole model forward pass is captured into CUDA graphs. Forking model source files to insert Python logic breaks operator fusion, forces fallback to eager execution, and leaves per-request private state unmanaged.

pLLM provides three decorators specifically for restructuring the forward pass while preserving CUDA graph throughput and request isolation:
1. `@pllm.custom_residual`: overrides how sublayer outputs merge into the residual stream, replaces the values read by attention, or attaches a chunk-updated fast-weight memory sublayer.
2. `@pllm.custom_attention`: customizes how attention reads KV pages (top-K sparse reads, attention monitoring) or declares physical KV cache retention layouts (sliding window, prompt sink + window, cross-layer KV sharing).
3. `@pllm.custom_forward`: orchestrates layer execution order in ordinary Python (looped depth, token-wise early exit / layer skipping, cross-step continuous thought state).

---

## 1. Custom Residual Connections: `@pllm.custom_residual` {#en-sec-1}

### Use Cases and Motivation {#en-use-cases-and-motivation}
Use `@pllm.custom_residual` when an algorithm applies custom math at the boundaries where Transformer sublayers read from or write back to the residual stream. Common examples include:
- **Value Residual** (e.g., ResFormer): mixes shallow-layer (e.g., layer 0) values into subsequent layers' values.
- **Hyper-Connections**: expands the single residual stream into $$n$$ parallel streams, applying weighted reads before sublayers and linear mixing afterward.
- **LAuReL**: adds low-rank corrections or gated scaling when merging a sublayer output into the residual stream.
- **Stateful memory sublayers** (e.g., TTT, Titans): attaches a per-request fast-weight memory matrix alongside attention or MLP that reads and updates chunk by chunk.

In native vLLM, residual addition and LayerNorm / RMSNorm are typically compiled into fused kernels (Fused Add+Norm). Hacking model code to alter residuals forces the whole model back to eager mode. `@pllm.custom_residual` lets you take over residual updates at selected sites while native attention projections, MLP computations, and KV cache writes run unchanged—and the rewritten operators remain fully captured inside the model's CUDA graphs.

### Minimal Runnable Example (CPU) {#en-minimal-runnable-example-cpu}
The example below implements ResFormer-style Value Residual: layers 1 through 23 mix layer 0's values into their own values before writing to the KV cache and running attention.

```python
from dataclasses import dataclass
import torch
import torch.nn.functional as F

import pllm
from pllm import testing

CONFIG = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=8)


@dataclass(frozen=True)
class ValueResidual(pllm.Workload):
    mixing: float = 0.0

    @pllm.custom_residual(site="value", layers=tuple(range(1, 24)))
    def mix(self, v: torch.Tensor, first: torch.Tensor = pllm.value(0)) -> torch.Tensor:
        return (v.float() + self.mixing * (first.float() - v.float())).to(v.dtype)


print(repr(pllm.explain(ValueResidual(), CONFIG).eager_reason))
```

```text
''
```

`eager_reason` is the empty string `''`, confirming the workload stays completely inside the model's CUDA graphs.

### Sites (`site`) and Input Signatures {#en-sites-site-and-input-signatures}
`@pllm.custom_residual` uses `site` to select which residual boundary to own:

| `site` | Positional tensors (no defaults) | Return value | Native behavior replaced | Typical use |
| --- | --- | --- | --- | --- |
| `"attention"` | Pre-sublayer residual `x`, attention branch output `y` | Updated residual stream | `x + y` | Attention residual scaling, gated branches |
| `"mlp"` | Pre-sublayer residual `x`, MLP branch output `y` | Updated residual stream | `x + y` | MLP residual rewrites, attached fast-weight memory |
| `"both"` | Same as above, invoked in forward order on attention then MLP | Updated residual stream | `x + y` | LAuReL, Hyper-Connections sublayer mixing |
| `"value"` | Current full-attention layer's value rows `v` `[rows, kv_heads, head_dim]` | Replacement value | Native `v` | Value Residual (feeds attention and writes to KV cache) |
| `"final"` | Residual stream `x` before final norm | Single residual stream `[rows, hidden]` | Native residual stream | Merging multi-stream Hyper-Connections back to a single stream |

**Signature and input rules**:
1. **Positional tensors first**: `attention`, `mlp`, and `both` must take `x, y` as their first two parameters; `value` and `final` must take the site tensor as their first parameter, all without default values.
2. **Input markers on subsequent parameters**: after the positional tensors, you may read:
   - `pllm.parameter(state, field)` or `pllm.module(state)`: read carried fields from an instance or request state.
   - `pllm.depth()`: current layer index (Python `int`). With `site="both"`, attention gets `2 * layer` and MLP gets `2 * layer + 1`.
   - `pllm.value(layer)`: the value tensor produced at an earlier layer for the same token in this forward pass.
   - `pllm.residual(layer)`: the residual stream output at the end of an earlier layer for the same token in this forward pass.
3. **Precision**: incoming tensors have the model's native dtype (e.g., `bfloat16`). For numerically sensitive accumulations or divisions, cast to float32 (`.float()`) inside the method and cast back (`.to(v.dtype)`) before returning.
4. **Multiple residual streams (`streams=n`)**: setting `streams=n` ($$n > 1$$) expands the residual stream from embedding onward to `[rows, n, hidden]`. Pair this with `read="method"` naming an `(x, ...) -> branch_input` method that collapses the $$n$$ streams into one before each sublayer, and declare one `site="final"` method to merge back to `[rows, hidden]`. See [07 Custom Residuals](/pllm/docs/examples/07_custom_residual/).

### Identity Restoration and Numerical Equivalence {#en-identity-restoration-and-numerical-equivalence}
When a custom residual method returns the identity expression (such as `x + y` for `site="attention"` or `v` for `site="value"`), model outputs are **bitwise equal** to native vLLM in eager mode.

**Why is there a tiny difference from native vLLM in default graph mode?**
In default graph mode, native vLLM enables PyTorch Inductor fused kernels (Fused Add+Norm). When `@pllm.custom_residual` is loaded, vLLM instead captures the model's unfused operators into CUDA graphs. This preserves exact PyTorch arithmetic—matching native vLLM eager mode bitwise—while differing by normal bf16 rounding ulps from Inductor's fused kernels. Compare identity runs against eager native vLLM or use `testing.check_on_gpu`.

### Stateful Memory Sublayers (Fast Weights) {#en-stateful-memory-sublayers-fast-weights}
Algorithms like TTT or Titans maintain a per-request fast-weight state $$M$$ (e.g., a `[hidden, hidden]` matrix) that is read and updated as tokens flow.

To declare a stateful sublayer on `@pllm.custom_residual`:
- `state="name"`: points to a `pllm.RequestState` with a single field (its carried memory) and no optimizer.
- `chunk=N`: fixed chunk size $$N$$ (a positive integer, e.g., 16).
- Supported only on a single `"attention"` or `"mlp"` site at **one** layer.

```python
@dataclass(frozen=True)
class FastWeightMemory(pllm.Workload):
    gate: float = 0.0
    lr: float = 0.1

    @property
    def memory(self) -> pllm.RequestState:
        return pllm.RequestState({"M": lambda m: (m.hidden_size, m.hidden_size)})

    @pllm.custom_residual(site="mlp", layers=(12,), state="memory", chunk=16)
    def remember(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        M: torch.Tensor = pllm.parameter("memory", "M"),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k = F.normalize(x.float(), dim=-1)
        read = k @ M.T
        stream = (x.float() + y.float() + self.gate * read).to(x.dtype)
        next_M = M - self.lr / len(x) * (read - y.float()).T @ k
        return stream, next_M


print(repr(pllm.explain(FastWeightMemory(), CONFIG).eager_reason))
```

```text
''
```

**Update semantics and design rationale**:
Every row within the same chunk (16 rows) reads the **carried state from the start of that chunk**; only after the final row of the chunk completes does the returned `next_M` take effect as the starting state for the next chunk.
If a request's remaining tokens in a scheduling step do not fill 16 rows, the framework buffers the partial rows and triggers the state update once subsequent steps complete the chunk. Because vLLM's continuous-batching scheduler may arbitrarily split long prompts across steps (Chunked Prefill), anchoring updates strictly to fixed chunk boundaries guarantees that **outputs are completely independent of how the scheduler chunks prompts**.

### Overhead and Limitations {#en-overhead-and-limitations}
- **Overhead**: custom residual logic is captured directly into CUDA graphs, avoiding per-step Python sync overhead; throughput cost comes solely from the added tensor math. Measured on an RTX 4090 with Qwen2.5-7B (vLLM 0.32.0, 256 tokens, median of 3 rounds after 1 warmup; see [07 Custom Residuals](/pllm/docs/examples/07_custom_residual/)):
  - Value Residual (editing 27 value layers, 32 concurrent requests): **99%** of native CUDA graph throughput.
  - Hyper-Connections ($$n=4$$, editing 56 sublayers, 32 concurrent requests): **93%** of native CUDA graph throughput.
  - Stateful memory sublayer (reading/writing a `[hidden, hidden]` FP32 matrix per token, ~49 MiB per request on 7B, so run at 8 concurrent requests): **75%** of native CUDA graph throughput at 8 concurrent requests.
- **Limitations and reasons**:
  - `tensor_parallel_size=1`: residual rewrite operators run inside single-GPU layer forward graphs; cross-GPU residual tensor and state synchronization is not supported yet.
  - Prefix caching must be disabled (`enable_prefix_caching=False`): prefix caching assumes identical prompt prefixes yield identical KVs; residual edits and stateful updates produce different activations or cross-token state accumulation, invalidating prefix reuse.
  - Stateful memory sublayers require raising `ExecutionOptions(cuda_graph_pool_budget_bytes=...)`: per-request private state memory and chunk history buffers exceed the default 64 MiB graph capture budget (7B at 8 concurrent requests needs ~1.9 GiB); if the budget is too small, engine startup fails with the exact byte requirement.
  - Stateful memory sublayers **cannot** be combined with `@pllm.custom_forward`: a stateful sublayer advances its memory strictly once per token, whereas a custom forward may loop or skip layers within a token, leaving update timing undefined.

---

## 2. Custom Attention Reads and KV Layouts: `@pllm.custom_attention` {#en-sec-2}

### Use Cases and Motivation {#en-use-cases-and-motivation-1}
Use `@pllm.custom_attention` when an algorithm modifies how attention reads information or how the KV cache is laid out physically. Typical examples:
- **Attention monitoring**: extracts intermediate statistics (such as the fraction of attention mass falling on a local window) without altering model outputs.
- **Training-free sparse reads** (e.g., Quest): dynamically scores KV pages and selects only top-K logical positions per head for attention.
- **Physical KV cache shrinking**: uses local sliding windows, prompt sink + sliding window (R-SWA), or shares KV caches across adjacent layers.

In an inference engine, attention is a deeply fused kernel whose KV block allocation and reclamation are managed by a paged memory pool. Traditionally, trying sparse reads or changing KV geometry meant hacking C++/CUDA kernels. `@pllm.custom_attention` preserves native Q/K/V projections and KV writes, replacing only the FlashAttention read step and exposing declarative physical KV layouts.

### Minimal Runnable Example (CPU) {#en-minimal-runnable-example-cpu-1}
The two examples below demonstrate common patterns: one keeps only a 256-token sliding window on even layers (and full KV on odd layers); the other monitors layer 12's local attention mass and exports it to the host.

```python
@dataclass(frozen=True)
class LocalGlobal(pllm.Workload):
    """Keep the most recent 256 tokens of KV on even layers, full KV on odd layers."""

    @pllm.custom_attention(layers=tuple(range(0, 24, 2)), keep="window", window=256)
    def read(self, q: torch.Tensor, kv: pllm.model.PagedKV) -> torch.Tensor:
        return kv.attend(q)[0]


print(testing.check(LocalGlobal(), CONFIG).passed)
```

```text
('tensor_parallel_size=1: admission', 'compile', ...)
```

```python
@dataclass(frozen=True)
class RecentMass(pllm.Workload):
    """Monitor the fraction of layer-12 attention falling on the last 64 positions without altering outputs."""

    recent: int = 64

    @pllm.custom_attention(layers=(12,))
    def split(self, q: torch.Tensor, kv: pllm.model.PagedKV) -> tuple[torch.Tensor, torch.Tensor]:
        out, lse = kv.attend(q)
        _, near = kv.attend(q, window=self.recent)
        # aux returns the attention share on recent positions: exp(LSE_local - LSE_global)
        return out, (near - lse).exp().mean(-1, keepdim=True)

    @pllm.export(name="recent_mass", max_rows=8, max_cols=1)
    def report(self, share: torch.Tensor = pllm.attention(12, "aux")) -> torch.Tensor:
        return share


print(testing.check(RecentMass(), CONFIG).passed)
```

```text
('tensor_parallel_size=1: admission', 'compile', ...)
```

### Method Signature and `PagedKV` {#en-method-signature-and-pagedkv}
A custom attention method has signature `f(self, q: torch.Tensor, kv: pllm.model.PagedKV, ...markers)`:
- `q`: RoPE-applied Query tensor of shape `[rows, heads, head_dim]`.
- `kv`: read-only paged KV view `pllm.model.PagedKV`. The current token's K/V have already been written to the cache. It exposes `key_cache`, `value_cache`, `block_table`, `seq_lens`, and related attributes for custom inspection.
- `kv.attend(q, *, window=0, topk=None)`: runs attention and returns `(output, float32 LSE [rows, heads])`.
  - `window`: lookback window size from the current position (including the current token).
  - `topk`: logical token position tensor of shape `[rows, k]` or `[rows, heads, k]` (padded with `-1`).
- **Optional `aux` return value**: the method may return `(out, aux)`. Downstream layers or export methods can read this tensor via `pllm.attention(layer, "aux")`, while `"lse"` reads the float32 log-sum-exp from the first `kv.attend` call.

### Why Can Such a Minimal `kv.attend` Signature Express Rich Custom Attention? {#en-why-can-such-a-minimal-kvattend-signature-express-rich-custom-attention}

At first glance, `@pllm.custom_attention` exposes neither a `[rows, heads, seq_len]` attention-weight matrix nor an arbitrary 2D `attn_mask`, nor in-place mutation of cached K/V—its core primitive is just `kv.attend(q, *, window=0, topk=None) -> (out, lse)`. That minimalism is deliberate: it preserves three hardware and serving invariants simultaneously:
1. **Keeping online softmax fused in on-chip SRAM**: FlashAttention is fast precisely because it tiles $$Q K^\top$$, softmax, and $$\times V$$ inside SRAM without ever materializing `[rows, heads, seq_len]` matrices in HBM; exposing dense weight or mask tensors would immediately degrade long-context bandwidth and memory back to quadratic.
2. **Keeping continuous batching inside static-shape CUDA graphs**: requests in the same batch have different `seq_len` values, so any intermediate tensor carrying a `seq_len` axis introduces dynamic shapes or host syncs that break full-model graph capture.
3. **Portability across physical cache architectures (GQA and DeepSeek MLA)**: in DeepSeek V2/V3 Multi-head Latent Attention (MLA), the paged cache stores compressed latent vectors (`kv_lora_rank`) plus shared RoPE keys rather than expanded multi-head K/V. Abstracting reads behind `kv.attend` lets the exact same workload code run FlashAttention on GQA models and matrix-absorbed latent attention directly over compressed MLA pages without decompressing the cache.

Within those invariants, four orthogonal mechanisms compose to express nearly every hardware-friendly custom attention variant:
- **`(out, lse)` forms an online-softmax monoid (sufficient statistics for subset mass & disjoint union)**: the returned `float32` `lse` always has fixed shape `[rows, heads]` ($$\log \sum_{j \in S} e^{s_j}$$), independent of sequence length, yet retains the exact normalizer:
  - **Subset probability mass** (e.g., recent-window mass, Lookback Lens, sink ratio): call `kv.attend` on subset $$S$$ to obtain `lse_s`, then compute `(lse_s - lse_full).exp()` in two lines with zero `[seq_len]` allocations.
  - **Exact union of disjoint regions** (e.g., anchor/sink positions `topk` $$\sqcup$$ local `window`, or multi-tier sparse windows): for any two disjoint index sets $$A$$ and $$B$$, calling `kv.attend` for `(out_a, lse_a)` and `(out_b, lse_b)` reconstructs exact attention over $$A \sqcup B$$ via `m = torch.logaddexp(lse_a, lse_b)` and `out = (lse_a - m).exp()[..., None] * out_a + (lse_b - m).exp()[..., None] * out_b` (ensure $$A$$ and $$B$$ do not overlap, or shared positions would be double-counted).
  - **Query-space transforms and head-wise composition** (e.g., Differential Attention $$\text{Attn}(q_1) - \lambda \text{Attn}(q_2)$$, mixing retrieval heads and streaming heads via `torch.where`, per-head gating): transform fixed-shape `q` and `out` directly around `kv.attend`.
- **Fixed-width `-1`-padded `topk` expresses per-head dynamic budgets inside static graphs**: when different layers or heads select different numbers of positions, allocate `[rows, heads, K_max]` at the maximum budget and mask unused slots to `-1` on GPU (`masked_fill(..., -1)`). The kernel treats `-1` as $$-\infty$$ (returning zero output and `-inf` LSE when all entries are `-1`), enabling adaptive sparsity without dynamic shapes.
- **`aux` and `pllm.attention(...)` enable cross-layer index reuse**: index-sharing schemes (such as IndexCache) can score pages once in an earlier layer, return `(out, chosen)`, and accept `chosen: torch.Tensor = pllm.attention(earlier_layer, "aux")` in deeper layers to call `kv.attend(q, topk=chosen)` without re-scoring.
- **Orthogonal separation of pre-cache value rewrites, physical block reclamation, and read selection**: modifying values before cache write (e.g., Value Residual or LoRA) belongs to `@pllm.custom_residual(site="value")` or `pllm.model.Adapter`; freeing physical blocks outside a window belongs to `keep=` / `window=` / `sink=` / `source=` on the decorator; and `@pllm.custom_attention` focuses purely on reading the committed cache, preserving native KV writes and block reclamation.

### Static KV Layout Declarations {#en-static-kv-layout-declarations}
`@pllm.custom_attention` lets you declare physical KV cache retention directly on the decorator:
- `keep="window", window=N`: retains only the most recent $$N$$ tokens of KV; older physical blocks are reclaimed into the pool by vLLM's `SlidingWindowSpec`.
- `sink=True` (requires `keep="window", window=N`): retains the entire prompt plus the most recent $$N$$ generated tokens (mapping to vLLM's `RSWASpec`).
- `keep="none", source=k` ($$k \ge 1$$): allocates no KV memory for this layer and instead reads layer $$k$$'s KV cache directly (`source` must point to an earlier layer; to read a deeper layer's KV, use `@pllm.custom_forward` with `kv="peek"`).

### Why Top-K Saves Bandwidth Without Saving VRAM, and Why Score-Based Eviction Is Unsupported {#en-why-top-k-saves-bandwidth-without-saving-vram-and-why-score-based-eviction-is-unsupported}
- **Why Top-K saves reads, not VRAM**:
  Passing selected indices to `kv.attend(q, topk=...)` makes FlashAttention load only those keys and values during softmax and GEMM, saving FLOPs and HBM read bandwidth. However, all historical tokens' KVs must remain in physical VRAM pages so future tokens can re-score and select from them. Thus Top-K read selection does not free physical KV capacity.
- **Why true score-based eviction is unsupported**:
  Algorithms like H2O and SnapKV that evict physical blocks mid-sequence by attention score are not supported in the current architecture (see [Design Trade-offs](/pllm/docs/explain/design-tradeoffs/)).
  **Why**: vLLM's physical block manager relies on a hard invariant—each request's logical KV sequence is a contiguous 1D array starting at position 0 whose block table grows monotonically. Punching holes mid-sequence to free blocks breaks block-table addressing, positional indexing, and continuous-batching scheduling.

### Overhead and Limitations {#en-overhead-and-limitations-1}
- **Overhead**:
  - Static layouts substantially reduce physical VRAM usage. On Qwen2.5-7B (2048-token context), `explain`'s per-request KV memory ceiling drops from **112.0 MiB** (full) to **70.4 MiB** with a 256-token window on even layers, and **56.0 MiB** with cross-layer sharing—doubling concurrency capacity.
  - In the same example measured on RTX 4090 (7B, 128 concurrent requests): passthrough `NativeRead`, even-layer window `LocalGlobal`, and cross-layer `CrossLayerShared` all achieve **99%** of native vLLM throughput (windowed layers use vLLM's FlashInfer window kernel; see [Performance and Memory](/pllm/docs/explain/performance/)). Pure-PyTorch dynamic Top-K scoring (such as Quest without a custom kernel) runs at **13%** of native throughput; production speed requires a fused operator (see [Custom Operators](/pllm/docs/guides/custom-ops/) and [09 Custom Triton Operators](/pllm/docs/examples/09_custom_kernels/)).
- **Limitations and reasons**:
  - `tensor_parallel_size=1`: custom FlashAttention reads and paged KV layouts do not yet implement cross-TP head sharding and sync.
  - Requires FlashAttention 2 backend: pLLM reads page tables and sequence lengths from FA2 metadata to organize each read; plain sliding-window reads then dispatch to vLLM's built-in FlashInfer kernel because FA2's window path disables GQA packing (see [Performance and Memory](/pllm/docs/explain/performance/)).
  - Unquantized KV (`kv_cache_dtype="auto"`): quantized formats like FP8 pack storage bytes and would require dedicated dequantization kernels during sparse gather.
  - Standard RoPE attention: attention variants with ALiBi, logit soft-capping, or hardware-specific sink tokens are unsupported.
  - Prefix caching disabled: sliding windows and cross-layer sharing change the KV layout retained per request, breaking prefix caching's global hash-tree assumptions.
  - Fully paged KV models only: supported only on models whose KV state lives entirely in vLLM paged blocks (Qwen2/2.5, Llama, Qwen3, etc.); hybrid models with extra recurrent state (such as Qwen3.5 DeltaNet) are unsupported.
  - Static layouts cannot combine with `PositionState` or `@pllm.custom_forward`: static layouts partition vLLM into multiple KV groups, whereas custom forwards and position state require a single unified KV group.

---

## 3. Custom Layer Execution Order: `@pllm.custom_forward` {#en-sec-3}

### Use Cases and Motivation {#en-use-cases-and-motivation-2}
Use `@pllm.custom_forward` when an algorithm reorders or dynamically alters the layer execution trajectory. Typical examples:
- **Looped / Recurrent Depth**: models like Huginn or Ouro that repeat a block of layers multiple times during decode to trade extra compute for deeper reasoning.
- **Per-token early exit / Adaptive depth**: methods like CALM or LayerSkip that let easy tokens skip deep layers during generation.
- **Continuous thought cross-step state**: methods like Coconut that feed the hidden vector produced by one token directly into the next token via `pllm.PositionState`.
- **Speculative dry runs**: looking ahead one step without corrupting the main KV cache.

In standard inference, the forward pass runs sequentially from layer 0 to the last layer in a fixed topology. Writing loops or layer skipping externally incurs complex KV cache bookkeeping and heavy host-device blocking syncs. `@pllm.custom_forward` wraps the forward pass into a slice-callable `layers(...)` object so you can orchestrate execution in Python while the framework automatically splits and captures per-slice CUDA graphs under the hood.

### Minimal Runnable Example (CPU) {#en-minimal-runnable-example-cpu-2}
The example below implements Ouro-style looped depth with per-token early exit: during decode, layers 20–23 loop up to 2 extra passes; any row whose residual change falls below a threshold converges and exits the loop early.

```python
@dataclass(frozen=True)
class LatentLoop(pllm.Workload):
    start: int = 20
    loops: int = 2
    tol: float = 0.05

    @pllm.custom_forward(tokens="decode")  # active= is restricted to decode
    def think(self, layers: pllm.model.Layers) -> torch.Tensor:
        hidden = layers()[-1]
        active = torch.ones_like(hidden[:, :1], dtype=torch.bool)
        for _ in range(self.loops):
            again = layers(start=self.start, hidden=hidden, active=active)[-1]
            change = (again - hidden).float().norm(dim=-1, keepdim=True) / hidden.float().norm(dim=-1, keepdim=True)
            hidden = torch.where(active, again, hidden)
            active = active & (change > self.tol)  # converged rows stop computing in later loops
        return hidden


print(testing.check(LatentLoop(), CONFIG).passed)
```

```text
('tensor_parallel_size=1: admission', 'compile', ...)
```

### Calling `layers(...)`, `LayerOutputs` Indexing, and KV Semantics {#en-calling-layers-layeroutputs-indexing-and-kv-semantics}
The `layers` argument is a `pllm.model.Layers` instance with the following contract:
- **Call signature**: `layers(start=0, stop=None, hidden=None, *, kv="commit", active=None) -> LayerOutputs`
  - Runs layers `start` through `stop - 1`. If `stop` is omitted, runs through the final layer; if `hidden` is omitted, starts from token embeddings (valid only when `start=0`).
- **Absolute indexing on `LayerOutputs`**:
  `LayerOutputs` must be indexed by **absolute model layer index**! For example, after `out = layers(12, 16)` (which runs layers 12–15), accessing layer 15's output requires `out[15]`; negative index `out[-1]` always refers to the last layer run by that call. Indexing outside the slice (such as `out[0]`) raises `ValueError: index a tap inside the Layers slice that produced it`.
- **`kv="commit"` vs. `kv="peek"`**:
  - `kv="commit"` (default): computes each layer's K/V in the slice and **writes** it to the KV cache. In a loop, later passes overwrite earlier passes at the current token's slot, so subsequent generated tokens see the K/V written by the final pass.
  - `kv="peek"`: attends over historical committed KV plus the current pass's newly computed K/V, **without writing** to the KV cache (leaving the cache untouched as it was before the call). Useful for speculative lookahead or dry runs.
  - **Empirical results from the Ouro paper**:
    The Ouro paper (arXiv 2510.25741, §5.4.2 Table 14) compared KV caching schemes for decode-time looped depth on GSM8K:
    - Storing a separate KV copy for every loop pass: **78.92** accuracy.
    - Keeping only the last pass's KV (the default overwrite `kv="commit"`): **78.85** accuracy (virtually lossless vs. storing all passes).
    - Keeping only the first pass's KV (commit on pass 1, `kv="peek"` on later passes): accuracy collapses to **18.73**.
    That collapse occurs because subsequent tokens can only read coarse first-pass K/Vs, breaking a model trained on deep looped representations. Hence `kv="commit"` overwriting on each pass is the default.

### Why Per-Token Layer Skipping (`active=`) Is Restricted to `tokens="decode"` {#en-why-per-token-layer-skipping-active-is-restricted-to-tokensdecode}
When you pass a boolean mask (`[rows, 1]`) to `layers(..., active=mask)`, rows with `False` skip that slice's matrix math and pass their `hidden` through unchanged.
For skipped rows, the framework propagates state on the first skip at a layer by running that layer's input norm, QKV projection, and RoPE on the stopped hidden state to populate K/V, and keeps that value on subsequent skips of the same layer.

**Why is `active=` strictly limited to `tokens="decode"`?**
Propagated K/Vs on skipped layers are approximations. During decode, a skipped row affects only one new position; every prompt token, by contrast, is context read by every future generated token, so skipping layers during prefill would replace the entire context's deep K/Vs with approximations whose error compounds across every later step. Prompts therefore run the full model, and `active=` is restricted to decode.

### Cross-Step State: `pllm.PositionState` and `load` / `save` {#en-cross-step-state-pllmpositionstate-and-load--save}
For continuous-thought algorithms like Coconut, the model passes one token's output hidden vector directly as the next token's input via `pllm.PositionState`:

```python
@dataclass(frozen=True)
class StepLoopWorkload(pllm.Workload):
    """Save each token's pre-norm residual and feed it as layer-0 input for generated positions."""

    history = pllm.PositionState()

    @pllm.custom_forward(tokens="all")
    def loop(self, layers: pllm.model.Layers) -> torch.Tensor:
        previous = layers.load("history", offset=-1)
        taps = layers(hidden=torch.where(layers.generated, previous, layers.embeddings))
        layers.save("history", taps[-1])
        return taps[-1]


print(testing.check(StepLoopWorkload(), CONFIG).passed)
```

```text
('tensor_parallel_size=1: admission', 'compile')
```

- **Call rules**: `layers.load("history", offset=-1)` reads the vector saved at the previous position (even if `save` was already called in the current step, `load` still returns the previous token's snapshot); `layers.save("history", value)` commits the current position's snapshot (if called multiple times, the last write wins).
- **Full-prompt spans (`over="span"`)**:
  The default `over="token"` mode steps causally position by position. With `@pllm.custom_forward(over="span")`, a request's contiguous prompt span is passed together as a `[M, H]` tensor, suited for Jacobi-style parallel prompt iteration. Note that `over="span"` has no sequential predecessor token, so `layers.load` is disallowed.

### Control-Flow Rules: Bounded Loops, No Tensor Branching, and `pllm.scan` {#en-control-flow-rules-bounded-loops-no-tensor-branching-and-pllmscan}
1. **Bounded Python loops unroll automatically**: Python `for` loops over workload fields (e.g., `for _ in range(self.loops):`) unroll at trace time into sliced CUDA graphs.
2. **No data-dependent tensor branching**: method bodies must **never** contain `if tensor.norm() > 1:`, `while`, or `break`. Forward graph topology must be static; otherwise tracing raises `methods run inside CUDA graphs, so Python cannot branch on tensor values (if, while or break on a tensor); ...`. Loop a fixed number of times and express conditionals with `torch.where` or `active=` masks (`active=` requires `tokens="decode"`; see [12 Adaptive Depth](/pllm/docs/examples/12_adaptive_depth/)).
3. **Linear scans (`pllm.scan`)**: to express long-sequence linear recurrences (of the form $$S_t = A_t S_{t-1} + B_t$$) inside a custom forward, call `pllm.scan(step, carry=..., over=..., chunk=...)`. pLLM recognizes affine recurrences and lowers them to a GPU **parallel prefix scan** when applicable, which is much faster than sequential chunk loops (see [Performance and Memory](/pllm/docs/explain/performance/) and [04 Custom Forward](/pllm/docs/tutorials/04_custom_forward/)).

### Overhead, Limitations, and Composition {#en-overhead-limitations-and-composition}
- **Overhead**:
  With `@pllm.custom_forward`, vLLM's whole-model CUDA graph is replaced, and `eager_reason` explicitly reports `@pllm.custom_forward`. **This does not mean per-operator eager execution**: pLLM captures separate per-slice CUDA graphs for each `layers(...)` call.
  In [12 Adaptive Depth](/pllm/docs/examples/12_adaptive_depth/) (RTX 4090, Qwen2.5-7B, 32 concurrent requests, 256 tokens, vLLM 0.32.0), a sliced `custom_forward` without skipping hits **99%** of native vLLM CUDA graph throughput. (At 7B where a step is ~19 ms, native eager also reaches 99% because launch overhead is hidden; on smaller models with few-millisecond steps, sliced graphs clearly beat eager.)
- **Memory and hardware constraints**:
  - Per-slice graph capture uses its own pool space and can exceed the default 64 MiB model graph budget (Example 12's workload on 7B at 32 concurrency needs ~143 MiB). Raise `ExecutionOptions(cuda_graph_pool_budget_bytes=128 << 20)` when needed (this budget counts toward `explain`'s peak VRAM estimate, so give only a small margin above the reported byte requirement rather than allocating gigabytes). If the budget is too small, generation fails with the exact bytes required.
  - Multi-GPU tensor parallelism requires vLLM 0.32.0 or newer.
- **Mutually exclusive with `pllm.hidden`**:
  Within the same workload, `@pllm.custom_forward` **cannot** be combined with `pllm.hidden(layer)` or `pllm.final_hidden()`.
  **Why**: under looped or skipped layers, a given layer may execute multiple times or zero times, so "layer $$L$$'s hidden state" has no unique physical meaning. For observation, use `pllm.residual(layer)` or `pllm.layer_logits((layer,))`, which always read the residual stream from the **last visit** to that layer.
- **Interaction with `pllm.model.Adapter`**:
  If a workload also declares a `pllm.model.Adapter` (see [Editing Models](/pllm/docs/guides/model-edits/)), weight deltas $$\Delta W$$ apply **on every actual projection invocation**. If a slice runs $$N$$ times in a loop, deltas apply $$N$$ times; native QKV propagation triggered for skipped layers also applies the corresponding Q/V deltas.
- **Combining with training overlays (`apply_updates()`)**:
  When combined with `@pllm.train_in_forward`, calling `layers.apply_updates()` inside `custom_forward` flushes and concatenates projection captures accumulated since the last boundary and steps the optimizer, making the updated private weights visible to subsequent layer calls on the same token.

---

## Common Errors and Troubleshooting {#en-common-errors-and-troubleshooting}

For exact error messages, root causes, and fixes when declarations on this page are rejected, see [Model sites and custom forwards](/pllm/docs/reference/errors/#en-model-sites-and-custom-forwards) and [Memory budgets](/pllm/docs/reference/errors/#en-memory-budgets) in the Error Catalog.

---

## Self-Check {#en-self-check}

1. Why is an identity `@pllm.custom_residual` (returning `x + y` or `v`) bitwise equal to native vLLM in eager mode, yet slightly different in bf16 rounding from native vLLM's default graph mode?
<details markdown="1">
<summary>Answer</summary>

In default graph mode, native vLLM uses PyTorch Inductor fused kernels (such as Fused Add+Norm). When `@pllm.custom_residual` is attached, vLLM captures the model's unfused operators into CUDA graphs instead. The arithmetic therefore matches eager mode bitwise while exhibiting normal bf16 floating-point rounding differences relative to Inductor's fused kernels.

</details>

2. Why does `kv.attend(q, topk=...)` save read bandwidth and FLOPs without reducing per-request KV VRAM capacity?
<details markdown="1">
<summary>Answer</summary>

`kv.attend(q, topk=...)` reads only the selected positions when computing attention softmax and GEMM, saving memory read bandwidth. However, all historical tokens' KVs must remain resident in physical VRAM pages so future tokens can score and select from them, so no physical blocks can be freed.

</details>

3. In looped-depth models, how do `layers(...)`'s default `kv="commit"` and `kv="peek"` differ, and why did the Ouro paper find overwriting every pass (keeping the last pass) works well (78.85) while keeping only the first pass collapses (18.73)?
<details markdown="1">
<summary>Answer</summary>

`kv="commit"` writes each pass's computed K/V into the cache, with later passes overwriting earlier ones, so subsequent tokens see the refined high-quality K/V from that token's final loop pass (78.85 on GSM8K in the Ouro paper). Using `kv="peek"` after the first pass leaves only the coarse shallow K/V from pass 1 in the cache, starving later tokens of looped deep representations and collapsing models trained with looped depth (18.73 on GSM8K).

</details>

</div>

<div class="zine-lang-panel is-hidden" data-lang-panel="zh" lang="zh-CN" markdown="1">

<div id="zh-sec-1"></div>

在 PyTorch 或 Hugging Face 中修改模型的计算结构非常直接：改写 `forward()` 函数、插入 hook，或者在层与层之间调用自定义函数。但在以吞吐为第一目标的 vLLM 服务中，数十个并发请求会被连续批处理（continuous batching）拼进同一个大张量，整模型的前向传播又被捕获进 CUDA graph。此时如果直接 fork 模型源码插入 Python 逻辑，不仅会破坏算子融合退回 eager 执行，更无法妥善管理每个并发请求的私有状态。

pLLM 提供了三个专用于改写前向计算结构的装饰器，让你在保持 CUDA graph 高吞吐和请求隔离的前提下重构前向传播：
1. `@pllm.custom_residual`：改写残差流汇入方式、替换 attention 读入的 value，或挂载按块更新的快权重记忆子层。
2. `@pllm.custom_attention`：改写 attention 的读取方式（只读 top-K、监控注意力分布），或声明物理 KV 缓存保留布局（滑动窗口、保留 prompt 加窗口、跨层共享 KV）。
3. `@pllm.custom_forward`：用普通 Python 重新编排各层的执行顺序（循环深度、逐 token 早退跳层、连续思维跨步状态）。

---

## 1. 定制残差连接：`@pllm.custom_residual` {#zh-sec-2}

### 适用场景与动机 {#zh-sec-3}
当算法需要在 Transformer 子层进出残差流的切入点上做数学变换时，使用 `@pllm.custom_residual`。常见算法包括：
- **Value Residual**（如 ResFormer）：将浅层（例如第 0 层）计算出的 value 混入后续各层的 value。
- **超连接（Hyper-Connections）**：将单一残差流扩展为 $$n$$ 条并行流，在子层前后分别执行加权读取与线性混合。
- **LAuReL**：在子层输出并入残差流时加入低秩修正或门控缩放。
- **带状态的记忆子层**（如 TTT、Titans）：在 attention 或 MLP 旁挂载一个由当前请求独立拥有的快权重记忆矩阵，边读边按块更新。

在原生 vLLM 中，残差加法与 LayerNorm / RMSNorm 通常被编译为融合算子（Fused Add+Norm）。如果为了改动残差而修改模型代码，往往会迫使整个模型退回 eager 模式。`@pllm.custom_residual` 让你接管指定位置的残差更新，而原生的 attention 投影、MLP 计算与 KV cache 写入照常进行，且改写后的算子仍完整捕获在模型的 CUDA graph 中。

### 最小可运行示例（CPU） {#zh-sec-4}
下面的示例实现 ResFormer 风格的 Value Residual：第 1 到 23 层在将 value 写入 KV cache 和送入 attention 之前，按比例混入第 0 层的 value。

```python
from dataclasses import dataclass
import torch
import torch.nn.functional as F

import pllm
from pllm import testing

CONFIG = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=8)


@dataclass(frozen=True)
class ValueResidual(pllm.Workload):
    mixing: float = 0.0

    @pllm.custom_residual(site="value", layers=tuple(range(1, 24)))
    def mix(self, v: torch.Tensor, first: torch.Tensor = pllm.value(0)) -> torch.Tensor:
        return (v.float() + self.mixing * (first.float() - v.float())).to(v.dtype)


print(repr(pllm.explain(ValueResidual(), CONFIG).eager_reason))
```

```text
''
```

输出的 `eager_reason` 为空字符串 `''`，说明该负载完全运行在模型的 CUDA graph 中。

### 挂载位置表（`site`）与输入参数 {#zh-sec-5}
`@pllm.custom_residual` 通过 `site` 参数指定接管哪一个位置的残差逻辑：

| `site` | 方法接收的位置张量（无默认值） | 返回值 | 替换的原生行为 | 典型用途 |
| --- | --- | --- | --- | --- |
| `"attention"` | 子层前的残差流 `x`、attention 分支输出 `y` | 更新后的残差流 | `x + y` | 注意力残差缩放、门控分支 |
| `"mlp"` | 子层前的残差流 `x`、MLP 分支输出 `y` | 更新后的残差流 | `x + y` | MLP 残差改写、外挂快权重记忆 |
| `"both"` | 同上，按前向顺序依次作用于 attention 与 MLP | 更新后的残差流 | `x + y` | LAuReL、超连接子层混合 |
| `"value"` | 当前全注意力层的 value 行 `v` `[rows, kv_heads, head_dim]` | 替换后的 value | 原生 `v` | Value Residual（替换后供 attention 读取并写入 KV cache） |
| `"final"` | 进入最终 norm 前的残差流 `x` | 单条残差流 `[rows, hidden]` | 原生残差流 | 超连接多流合并回单流 |

**方法签名与输入规则**：
1. **位置张量在前**：`attention`、`mlp`、`both` 必须在前两个参数接收 `x, y`；`value` 与 `final` 必须在第一个参数接收位置张量，均不带默认值。
2. **后续参数使用标记默认值**：位置张量之后可以读取以下标记：
   - `pllm.parameter(state, field)` 或 `pllm.module(state)`：读取实例状态或请求状态的携带量。
   - `pllm.depth()`：当前层号（Python `int`）。若 `site="both"`，attention 处为 `2 * layer`，MLP 处为 `2 * layer + 1`。
   - `pllm.value(layer)`：同一 token 在本次前向中更早层算出的 value 张量。
   - `pllm.residual(layer)`：同一 token 在本次前向中更早层最后算出的残差流。
3. **精度处理**：传入的张量为模型原生类型（如 `bfloat16`）。涉及数值敏感的累加或除法时，在方法内部转为 float32 计算（`.float()`），返回前转回原类型（`.to(v.dtype)`）。
4. **多残差流（`streams=n`）**：设 `streams=n`（$$n > 1$$）时，残差流自 embedding 起扩展为 `[rows, n, hidden]`。配合 `read="method"` 命名一个 `(x, ...) -> branch_input` 方法在进入子层前将多流压为一条；最后必须声明一个 `site="final"` 的方法将多流合并回 `[rows, hidden]`。详见例程 [07 改写残差连接](/pllm/docs/examples/07_custom_residual/)。

### 恒等还原与数值等价性 {#zh-sec-6}
当自定义残差方法返回恒等结果（如 `site="attention"` 返回 `x + y`，或 `site="value"` 返回 `v`）时，模型输出与 eager 模式的原生 vLLM **逐位一致（bitwise equal）**。

**为什么它与默认图模式的原生 vLLM 存在微小差异？**
默认图模式的原生 vLLM 会启用 PyTorch Inductor 融合内核（Fused Add+Norm）。当加载 `@pllm.custom_residual` 时，vLLM 改为捕获模型原本的非融合算子到 CUDA graph 中。这种方式保留了纯正的 PyTorch 算术语义，因此与 eager 模式的原生 vLLM 逐位相同，但和 Inductor 融合内核之间会有极细微的 bf16 舍入差异。验证恒等行为时，请与 eager 模式的原生引擎对比，或直接调用 `testing.check_on_gpu`。

### 带状态的记忆子层（快权重） {#zh-sec-7}
在实现 TTT 或 Titans 等算法时，算法需要维护每个请求独立的快权重状态 $$M$$（例如 `[hidden, hidden]` 矩阵），并在生成过程中不断读写。

在 `@pllm.custom_residual` 中声明状态子层：
- `state="name"`：指向一个只有一个字段（携带量 carry）的 `pllm.RequestState`，不带优化器。
- `chunk=N`：固定块长 $$N$$（正整数，例如 16）。
- 仅支持挂载在某**一层**的单个 `"attention"` 或 `"mlp"` 位置。

```python
@dataclass(frozen=True)
class FastWeightMemory(pllm.Workload):
    gate: float = 0.0
    lr: float = 0.1

    @property
    def memory(self) -> pllm.RequestState:
        return pllm.RequestState({"M": lambda m: (m.hidden_size, m.hidden_size)})

    @pllm.custom_residual(site="mlp", layers=(12,), state="memory", chunk=16)
    def remember(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        M: torch.Tensor = pllm.parameter("memory", "M"),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k = F.normalize(x.float(), dim=-1)
        read = k @ M.T
        stream = (x.float() + y.float() + self.gate * read).to(x.dtype)
        next_M = M - self.lr / len(x) * (read - y.float()).T @ k
        return stream, next_M


print(repr(pllm.explain(FastWeightMemory(), CONFIG).eager_reason))
```

```text
''
```

**更新规则与设计原因**：
同一 chunk（16 行）内的每一行，在计算读取时读到的都是**该 chunk 起始时的携带量**；只有该 chunk 的最后一行执行完毕后，方法返回的 `next_M` 才会正式生效并更新为下一 chunk 的起始状态。
如果某次调度步中某请求剩下的 token 未凑满 16 行，未完成的行由框架暂存，待后续步骤凑满整块后再触发状态更新。这样设计的原因在于：vLLM 的连续批处理调度器可能会将长 prompt 任意切块（Chunked Prefill），通过将更新严格锚定在固定块长边界上，保证了**生成结果完全不依赖调度器如何分步切分 prompt**。

### 运行开销与限制 {#zh-sec-8}
- **开销**：改写逻辑直接录入 CUDA graph，避免了逐步 Python 同步开销。吞吐下降仅来自于算法多出的张量算术。在 RTX 4090 上对 Qwen2.5-7B 实测（vLLM 0.32.0，256 token，预热 1 轮后 3 轮中位数；详见 [07 改写残差连接](/pllm/docs/examples/07_custom_residual/)）：
  - Value Residual（改写 27 层 value，32 个并发请求）：吞吐为原生 CUDA graph 的 **99%**。
  - 超连接（$$n=4$$，改写 56 个子层，32 个并发请求）：吞吐为原生 CUDA graph 的 **93%**。
  - 带状态记忆子层（每 token 读写 `[hidden, hidden]` 的 FP32 矩阵，7B 上每个请求约 49 MiB，所以只开 8 个并发请求）：吞吐为同样 8 个并发请求的原生 CUDA graph 的 **75%**。
- **限制与原因**：
  - `tensor_parallel_size=1`：残差改写算子在单卡层前向图内执行，暂不支持跨多卡的残差张量与状态同步。
  - 关闭前缀缓存（`enable_prefix_caching=False`）：前缀缓存假定 prompt 前缀的 KV 恒定不变；残差改写或带状态更新会导致同一前缀产生不同的激活或跨 token 状态积累，使前缀缓存失效。
  - 带状态记忆子层需要调大 `ExecutionOptions(cuda_graph_pool_budget_bytes=...)`：为每个请求分配的私有状态显存与 chunk 历史缓冲会超出默认 64 MiB 的图捕获预算，7B、8 个并发请求时约需 1.9 GiB；若预算不足，引擎启动时会明确报错并给出所需字节数。
  - 带状态记忆子层**不能**与 `@pllm.custom_forward` 组合：状态子层的记忆按 token 严格推进，而自定义前向可能在同一 token 内多次循环或跳过某层，导致步进时机未定义。

---

## 2. 定制 attention 读法与 KV 布局：`@pllm.custom_attention` {#zh-sec-9}

### 适用场景与动机 {#zh-sec-10}
当算法聚焦于 attention 计算过程中的信息获取或 KV cache 的存储形态时，使用 `@pllm.custom_attention`。典型场景包括：
- **注意力监控**：提取中间统计量（如注意力落在局部窗口上的比例），且不影响模型输出。
- **训练无关的稀疏读取**（如 Quest）：动态计算每页键值的分数，每头仅选择 top-K 个逻辑位置送入 attention。
- **物理 KV 缓存瘦身**：使用局部滑动窗口、保留 prompt 加滑动窗口（R-SWA），或在相邻层之间共享 KV 缓存。

在推理引擎中，attention 通常是高度融合的底层算子，KV 块的分配与回收由分页内存池全权接管。传统实现若想尝试稀疏读取或改写 KV 几何分布，必须修改 C++/CUDA 内核。`@pllm.custom_attention` 保留原生的 Q/K/V 投影和 KV 写入过程，仅替换 FlashAttention 读取步骤，并提供声明物理 KV 布局的接口。

### 最小可运行示例（CPU） {#zh-sec-11}
下面的示例展示两种典型用法：一种声明偶数层仅保留 256 token 的滑动窗口（奇数层保留全量）；另一种监控第 12 层注意力分布并导出到主机。

```python
@dataclass(frozen=True)
class LocalGlobal(pllm.Workload):
    """偶数层保留最近 256 个 token 的 KV，奇数层保留全量 KV。"""

    @pllm.custom_attention(layers=tuple(range(0, 24, 2)), keep="window", window=256)
    def read(self, q: torch.Tensor, kv: pllm.model.PagedKV) -> torch.Tensor:
        return kv.attend(q)[0]


print(testing.check(LocalGlobal(), CONFIG).passed)
```

```text
('tensor_parallel_size=1: admission', 'compile', ...)
```

```python
@dataclass(frozen=True)
class RecentMass(pllm.Workload):
    """监控第 12 层落在最近 64 个位置上的注意力占比，输出保持原模型不变。"""

    recent: int = 64

    @pllm.custom_attention(layers=(12,))
    def split(self, q: torch.Tensor, kv: pllm.model.PagedKV) -> tuple[torch.Tensor, torch.Tensor]:
        out, lse = kv.attend(q)
        _, near = kv.attend(q, window=self.recent)
        # aux 返回最近位置分到的注意力比例：exp(LSE_local - LSE_global)
        return out, (near - lse).exp().mean(-1, keepdim=True)

    @pllm.export(name="recent_mass", max_rows=8, max_cols=1)
    def report(self, share: torch.Tensor = pllm.attention(12, "aux")) -> torch.Tensor:
        return share


print(testing.check(RecentMass(), CONFIG).passed)
```

```text
('tensor_parallel_size=1: admission', 'compile', ...)
```

### 方法签名与 `PagedKV` {#zh-sec-12}
自定义 attention 方法的签名为 `f(self, q: torch.Tensor, kv: pllm.model.PagedKV, ...markers)`：
- `q`：加过 RoPE 的 Query 张量，形状为 `[rows, heads, head_dim]`。
- `kv`：只读的分页 KV 视图 `pllm.model.PagedKV`。当前 token 的 K/V 已在此前写入缓存。该对象暴露 `key_cache`、`value_cache`、`block_table`、`seq_lens` 等属性供自定义算法分析。
- `kv.attend(q, *, window=0, topk=None)`：执行 attention 计算，返回元组 `(输出, float32 LSE [rows, heads])`。
  - `window`：从当前位置往前数的回溯窗口大小（含当前 token）。
  - `topk`：逻辑 token 位置张量，形状为 `[rows, k]` 或 `[rows, heads, k]`（不足处用 -1 补齐）。
- **可选返回值 `aux`**：方法可返回 `(out, aux)`。后续层或导出方法可通过 `pllm.attention(layer, "aux")` 读取该张量；`"lse"` 则读取第一次调用 `kv.attend` 返回的 float32 log-sum-exp。

### 为什么极简的 `kv.attend` 签名能表达丰富的自定义注意力？ {#zh-sec-13}

初看之下，`@pllm.custom_attention` 既不暴露形状为 `[rows, heads, seq_len]` 的注意力权重矩阵，也不接收任意二维 `attn_mask`，更不允许原地改写缓存里的 K/V——核心原语只有 `kv.attend(q, *, window=0, topk=None) -> (out, lse)`。这并非功能缺失，而是为了同时守住三条硬件与系统红线：
1. **保住在线 Softmax 片上融合**：FlashAttention 之所以快，正是因为它在片上 SRAM 内分块完成 $$Q K^\top$$、Softmax 与乘 $$V$$，从不把随序列长度增长的 `[rows, heads, seq_len]` 矩阵写回显存；一旦暴露整行权重或稠密掩码，长上下文的显存带宽和容量就会退回二次方。
2. **保住连续批处理与静态形状 CUDA graph**：同一批请求的 `seq_len` 各不相同，任何带 `seq_len` 维度的中间张量都会引入动态形状或主机同步，直接打碎整模型图捕获。
3. **跨物理缓存架构通用（GQA 与 DeepSeek MLA）**：在 DeepSeek V2/V3 的多头潜在注意力（MLA）中，分页缓存里存放的是压缩潜变量 `kv_lora_rank` 与共享旋转位置向量，根本没有展开的多头 K/V。把读取抽象为 `kv.attend` 后，同一份负载代码既能在普通 GQA 模型上运行 FlashAttention，也能在 MLA 模型上自动通过矩阵吸收直接在压缩潜变量上完成注意力计算，无需解压缓存。

在这个约束下，四项正交设计组合起来覆盖了绝大多数对硬件友好的自定义注意力变体：
- **`(out, lse)` 构成在线 Softmax 的完备统计量（精确合并与子集归约）**：返回的 `float32` `lse` 形状恒为 `[rows, heads]`（即 $$\log \sum_{j \in S} e^{s_j}$$），与序列长度无关，却保留了完整的归一化分母信息：
  - **注意力分布统计**（如局部窗口占比、Lookback Lens、开头位置占比）：对子集 $$S$$ 调一次 `kv.attend` 拿到 `lse_s`，用 `(lse_s - lse_full).exp()` 两行即可算出精确概率质量，零 `[seq_len]` 张量分配。
  - **不相交区域的精确拼接**（如“固定锚点位置 `topk` $$\sqcup$$ 最近局部窗口 `window`”，或多级稀疏窗口）：只要两个位置集合 $$A$$ 与 $$B$$ 互不重叠，分别调用 `kv.attend` 得到 `(out_a, lse_a)` 与 `(out_b, lse_b)` 后，可按在线 Softmax 结合律精确合并为 $$A \sqcup B$$ 上的注意力：`m = torch.logaddexp(lse_a, lse_b)`，`out = (lse_a - m).exp()[..., None] * out_a + (lse_b - m).exp()[..., None] * out_b`（注意两段集合不能有交集，否则重叠位置会被重复计入分母）。
  - **查询侧变换与头间组合**（如 Differential Attention 的双路查询差分 $$\text{Attn}(q_1) - \lambda \text{Attn}(q_2)$$、检索头与局部流式头按头掩码 `torch.where` 混合、逐头门控缩放）：在调用 `kv.attend` 前后对固定形状的 `q` 和 `out` 直接做张量运算即可。
- **定宽 `-1` 填充的 `topk` 在静态图内表达逐头可变预算**：当不同层、不同头需要根据打分保留不同数量的位置时，只需按最大预算 `K_max` 分配 `[rows, heads, K_max]`，在 GPU 上将超出各头预算的槽位用 `masked_fill(..., -1)` 填为 `-1`。内核会自动把 `-1` 屏蔽为 $$-\infty$$（某头全为 `-1` 时安全返回零输出与 `-inf` LSE），无需动态张量形状即可实现自适应稀疏度。
- **`aux` 与 `pllm.attention(...)` 实现跨层索引复用**：像跨层共享稀疏选择器（IndexCache 一类）只需在浅层打分选出一次 `chosen` 索引并返回 `(out, chosen)`，深层通过参数默认值 `chosen: torch.Tensor = pllm.attention(earlier_layer, "aux")` 接收后直接调用 `kv.attend(q, topk=chosen)`，省去逐层重复扫页打分。
- **把“写前改值”、“物理块回收”与“注意力读取”拆成三个正交原语**：需要在写入缓存前修改 value（如 Value Residual 或 LoRA）交给 `@pllm.custom_residual(site="value")` 或 `pllm.model.Adapter`；需要真正释放滑出窗口的物理块交给装饰器上的 `keep=` / `window=` / `sink=` / `source=`；`@pllm.custom_attention` 方法体只专注在已写入的缓存上组织读取，从而完整保留原生缓存写入路径与物理块回收机制。

### 静态 KV 布局声明 {#zh-sec-14}
`@pllm.custom_attention` 允许在装饰器中直接声明底层 KV 缓存的物理留存策略：
- `keep="window", window=N`：仅保留最近 $$N$$ 个 token 的 KV，更早的物理块由 vLLM 的 `SlidingWindowSpec` 自动回收至显存池。
- `sink=True`（需配合 `keep="window", window=N`）：保留整个 prompt 加上最近 $$N$$ 个生成的 token（对应 vLLM 的 `RSWASpec`）。
- `keep="none", source=k`（$$k \ge 1$$）：本层完全不分配独立的 KV 显存，直接读取下方第 $$k$$ 层的 KV 缓存（源层必须是更早的层；若需读取更深层的 KV，请改用 `@pllm.custom_forward` 的 `kv="peek"`）。

### Top-K 选择只省读取不省显存，以及按分数驱逐不支持 {#zh-sec-15}
- **Top-K 为什么只省读不省显存**：
  在 `kv.attend(q, topk=...)` 中传入选中的位置索引，FlashAttention 内核在执行 softmax 和 GEMM 时只加载这部分 key 和 value，因而节省了计算 FLOPs 与显存读取带宽。然而，所有历史 token 的 KV 仍然必须完整存留在物理显存页中，供后续新生成的 token 重新计算打分并挑选。因此，Top-K 选择无法释放物理显存容量。
- **为什么不支持按分数真正驱逐（Score-based Eviction）**：
  像 H2O、SnapKV 这类按注意力分数在序列中间剔除物理块的算法，在当前架构下不支持（详见[设计取舍](/pllm/docs/explain/design-tradeoffs/)）。
  **原因**：vLLM 的物理块管理器基于强假设构建——每个请求的逻辑 KV 序列是从位置 0 开始连续递增的一维数组，其块表只增不减。如果在中间任意“挖洞”释放显存块，会导致块表寻址、位置编码计算与连续批处理调度逻辑彻底失效。

### 运行开销与限制 {#zh-sec-16}
- **开销**：
  - 静态布局能实质减少物理显存占用。在 Qwen2.5-7B（上下文 2048 token）上，`explain` 算出的单请求 KV 显存上限从全量的 **112.0 MiB** 降至偶数层窗口 256 的 **70.4 MiB**，跨层共享则降至 **56.0 MiB**，并发容量翻倍。
  - 同一个示例在 RTX 4090、7B、128 并发下测得：原样读取的 `NativeRead`、偶数层窗口 `LocalGlobal`、跨层共享 `CrossLayerShared` 吞吐都为原生 vLLM 的 **99%**（窗口层走 vLLM 的 FlashInfer 窗口内核，见[性能与显存](/pllm/docs/explain/performance/)）。纯 PyTorch 实现的 Top-K 动态打分（如未写成专用内核的 Quest）吞吐仅为原生的 **13%**，生产环境加速需使用融合算子（详见[自写算子](/pllm/docs/guides/custom-ops/)与 [09 自写 Triton 算子](/pllm/docs/examples/09_custom_kernels/)）。
- **限制与原因**：
  - `tensor_parallel_size=1`：自定义 FlashAttention 读法与分页 KV 布局尚未实现跨 TP rank 的注意力头切分与通信同步。
  - 引擎需使用 FlashAttention 2 后端：pLLM 从 FA2 的元数据里取页表和序列长度来组织每一种读法；其中普通的窗口读取再交给 vLLM 自带的 FlashInfer 内核执行，因为 FA2 的窗口路径会关掉 GQA 打包（见[性能与显存](/pllm/docs/explain/performance/)）。
  - 未量化 KV（`kv_cache_dtype="auto"`）：FP8 等量化格式打包了存储字节，在稀疏读取 gather 时需要专用的反量化内核。
  - 普通 RoPE 注意力：不支持带有 ALiBi、Logit Soft-capping 或硬件特定 Sink Token 的注意力模型。
  - 关闭前缀缓存：滑动窗口与跨层共享会改变每个请求实际留存的 KV 结构，破坏前缀缓存的全局哈希树假设。
  - 仅限全部分页模型：仅支持 KV 全部存放在 vLLM 分页块中的模型（Qwen2/2.5、Llama、Qwen3 等）；带有额外递归状态的模型（如 Qwen3.5 DeltaNet）不支持。
  - 静态布局不能与 `PositionState` 或 `@pllm.custom_forward` 组合：静态布局会将 vLLM 划分为多个独立的 KV 组，而自定义前向与跨步状态要求单一全局的 KV 组管理。

---

## 3. 改写层的执行顺序：`@pllm.custom_forward` {#zh-sec-17}

### 适用场景与动机 {#zh-sec-18}
当算法需要完全重排甚至动态改变层的前向执行轨迹时，使用 `@pllm.custom_forward`。典型场景包括：
- **循环深度（Looped / Recurrent Depth）**：如 Huginn、Ouro 等模型，在 decode 阶段将一段层重复执行多遍，用额外的推理时间换取更强的推理能力。
- **逐 token 提前退出 / 自适应深度（Early Exit / Adaptive Depth）**：如 CALM、LayerSkip，在生成过程中让简单的 token 提前跳过深层。
- **连续思维跨步状态（Continuous Thought）**：如 Coconut，通过 `pllm.PositionState` 将上一个 token 产生的 hidden 向量直接喂给下一个 token 作为输入。
- **投机试算**：在不破坏主干 KV 缓存的前提下，向前“试算一步”。

在标准推理中，模型前向按固定拓扑从第 0 层顺序跑至最后一层。如果要在外部写循环或跳层，不仅需要处理极其复杂的 KV 缓存状态维护，还会带来密集的 host-device 阻塞同步。`@pllm.custom_forward` 将整个前向过程封装为可分段调用的 `layers(...)` 对象，由用户在 Python 中编排计算逻辑，框架在底层自动切分并捕获分段 CUDA graph。

### 最小可运行示例（CPU） {#zh-sec-19}
下面的示例实现 Ouro 风格的循环深度与逐 token 提前退出：在 decode 阶段将第 20 到 23 层重复执行最多 2 遍；若某行残差变化量小于阈值，则提前收敛停止循环。

```python
@dataclass(frozen=True)
class LatentLoop(pllm.Workload):
    start: int = 20
    loops: int = 2
    tol: float = 0.05

    @pllm.custom_forward(tokens="decode")  # active= 仅限 decode 阶段
    def think(self, layers: pllm.model.Layers) -> torch.Tensor:
        hidden = layers()[-1]
        active = torch.ones_like(hidden[:, :1], dtype=torch.bool)
        for _ in range(self.loops):
            again = layers(start=self.start, hidden=hidden, active=active)[-1]
            change = (again - hidden).float().norm(dim=-1, keepdim=True) / hidden.float().norm(dim=-1, keepdim=True)
            hidden = torch.where(active, again, hidden)
            active = active & (change > self.tol)  # 收敛的行在后续循环中停算
        return hidden


print(testing.check(LatentLoop(), CONFIG).passed)
```

```text
('tensor_parallel_size=1: admission', 'compile', ...)
```

### `layers(...)` 调用、`LayerOutputs` 索引与 KV 机制 {#zh-sec-20}
`layers` 参数是 `pllm.model.Layers` 实例，提供以下调用契约：
- **调用形式**：`layers(start=0, stop=None, hidden=None, *, kv="commit", active=None) -> LayerOutputs`
  - 运行第 `start` 到 `stop - 1` 层。若不传 `stop`，默认运行至最后一层；若不传 `hidden`，默认以 token embedding 作为起始输入（仅在 `start=0` 时合法）。
- **`LayerOutputs` 的绝对索引规则**：
  `LayerOutputs` 必须按**模型的绝对层号**索引！例如 `out = layers(12, 16)` 运行了第 12–15 层，访问第 15 层的输出必须写 `out[15]`；负数索引 `out[-1]` 恒定指向本次调用运行的最后一层。若使用了切片范围之外的索引（如 `out[0]`），会抛出 `ValueError: index a tap inside the Layers slice that produced it`。
- **`kv="commit"` 与 `kv="peek"` 的本质区别**：
  - `kv="commit"`（默认）：计算当前分段各层的 K/V 并**写入** KV cache。在循环调用中，后一遍计算出的 K/V 会直接覆盖前一遍在当前位置的槽位，后续生成的 token 将看到最后一遍循环写入的 K/V。
  - `kv="peek"`：结合历史已提交的 KV 缓存与本遍新计算的 K/V 参与 attention 计算，但**不写入** KV cache（缓存被完全保留在本次调用前的状态）。常用于投机预演或试算。
  - **Ouro 论文的对比实验数据**：
    Ouro 论文（arXiv 2510.25741，§5.4.2 表 14）深入对比了 decode 阶段层循环中不同 KV 处理机制在 GSM8K 上的表现：
    - 每一遍循环都独立存一份 KV：准确率 **78.92**。
    - 只留最后一遍的 KV（即默认的每遍覆盖写回 `kv="commit"`）：准确率 **78.85**（与全量保留几乎无损）。
    - 只留第一遍的 KV（第一遍 commit，后续循环改用 `kv="peek"`）：准确率暴跌至 **18.73**。
    这是因为如果后续 token 只能读到浅层第一遍产出的粗糙 K/V，整个基于深层特征训练的模型会彻底崩塌。因此循环中默认使用 `kv="commit"` 覆盖写回。

### 逐 Token 跳层（`active=`）为什么仅限 `tokens="decode"` {#zh-sec-21}
当向 `layers(..., active=mask)` 传入布尔掩码（形状 `[rows, 1]`）时，掩码为 `False` 的行不参与该分段的矩阵计算，其 `hidden` 原样穿透传递。
对于被跳过的行，框架会在其第一次跳过某层时，用该行停下的隐藏状态经过该层的输入 norm、QKV 投影与 RoPE 补写对应的 K/V（状态传播），之后再跳过该层时保留该值。

**为什么 `active=` 严格限制在 `tokens="decode"` 中？**
跳层补写的 K/V 是近似值。decode 时被跳过的只是一个新位置；prompt 的每个 token 都是之后所有生成 token 要读的上下文，在 prefill 时跳层会把整段上下文在高层的 K/V 换成近似值，误差影响此后生成的每一步。因此 prompt 默认跑完整模型，`active=` 只用于 decode。

### 跨步状态：`pllm.PositionState` 与 `load` / `save` {#zh-sec-22}
对于类似 Coconut 的“连续思维”算法，模型需要把上一个 token 产生的 hidden 向量直接带给下一个 token。这可以通过声明 `pllm.PositionState` 实现：

```python
@dataclass(frozen=True)
class StepLoopWorkload(pllm.Workload):
    """保存每个 token 的 pre-norm 残差，在生成位置作为第 0 层的起始输入。"""

    history = pllm.PositionState()

    @pllm.custom_forward(tokens="all")
    def loop(self, layers: pllm.model.Layers) -> torch.Tensor:
        previous = layers.load("history", offset=-1)
        taps = layers(hidden=torch.where(layers.generated, previous, layers.embeddings))
        layers.save("history", taps[-1])
        return taps[-1]


print(testing.check(StepLoopWorkload(), CONFIG).passed)
```

```text
('tensor_parallel_size=1: admission', 'compile')
```

- **调用规则**：`layers.load("history", offset=-1)` 读取前一个位置保存的向量（即使当前步骤已先调用过 `save`，它依然返回上一个 token 的快照）；`layers.save("history", value)` 提交当前位置的快照（多次调用以最后一次为准）。
- **整段 Prompt 跨度（`over="span"`）**：
  默认 `over="token"` 模式按因果顺序逐位置推进。设 `@pllm.custom_forward(over="span")` 时，一个请求的一整段连续 prompt 被打包为 `[M, H]` 张量统一送入，适用于 Jacobi 式全 prompt 并行迭代。注意 span 模式下没有串行前驱 token，不能调用 `layers.load`。

### 控制流规范：有界循环、禁止张量分支、`pllm.scan` {#zh-sec-23}
1. **有界 Python 循环自动展开**：基于负载字段的 Python `for` 循环（如 `for _ in range(self.loops):`）会在追踪期直接展开录入分段 CUDA graph。
2. **禁止基于张量值的控制流**：方法体内部绝对**不能**出现 `if tensor.norm() > 1:`、`while` 或 `break`。前向图的拓扑结构必须确定，否则追踪时报错 `methods run inside CUDA graphs, so Python cannot branch on tensor values (if, while or break on a tensor); ...`。循环固定次数，条件用 `torch.where` 或 `active=` 掩码表达（`active=` 需 `tokens="decode"`，见 [12 自适应深度](/pllm/docs/examples/12_adaptive_depth/)）。
3. **线性扫描（`pllm.scan`）**：若在自定义前向中需要实现长序列线性递推（形如 $$S_t = A_t S_{t-1} + B_t$$），应调用 `pllm.scan(step, carry=..., over=..., chunk=...)`。pLLM 会识别仿射递推，在条件允许时改用 GPU 上的**并行前缀扫描**，比逐块串行的循环快得多（见[性能与显存](/pllm/docs/explain/performance/)）。详见 [04 改写 forward](/pllm/docs/tutorials/04_custom_forward/)。

### 运行开销、限制与组合 {#zh-sec-24}
- **运行开销**：
  使用 `@pllm.custom_forward` 时，vLLM 原生的大模型整图不再生效，`eager_reason` 会明确标明 `@pllm.custom_forward`。**但这绝不意味着模型退回了逐算子 eager 运行**：pLLM 为每次调用的层分段分别捕获了独立的分段 CUDA graph。
  在 [12 自适应深度](/pllm/docs/examples/12_adaptive_depth/) 的实测中（RTX 4090，Qwen2.5-7B，32 个并发请求，256 token，vLLM 0.32.0），一个无跳层、分段调用的 `custom_forward` 吞吐为原生 vLLM CUDA graph 的 **99%**。这个设置下原生 eager 也有 99%（7B 一步约 19 ms，发射开销藏得住）；步长只有几毫秒的小模型上，分段图和 eager 的差别才明显。
- **显存与硬件限制**：
  - 分段图捕获需要自己的池空间，可能超过默认的 64 MiB 模型图预算（示例 12 的负载在 7B、32 个并发请求下要 143 MiB 左右），需要时在配置中调大捕获预算（例如 `ExecutionOptions(cuda_graph_pool_budget_bytes=128 << 20)`；这份预算会计入 `explain` 报的 pLLM 显存峰值，所以按报错给出的字节数留一点余量即可，不必给到几 GiB）；若预算不足，生成时会直接报错并指明所需的精确字节数。
  - 多卡张量并行要求使用 vLLM 0.32.0。
- **与状态读取器（`pllm.hidden`）互斥**：
  在同一个负载中，挂了 `@pllm.custom_forward` 后**不能**再声明读取 `pllm.hidden(layer)` 或 `pllm.final_hidden()`。
  **原因**：在循环或跳层前向中，某一层可能被运行了多次，也可能一次都没运行，“第 $$L$$ 层的隐藏状态”不再具备唯一的物理定义。如需观测，请改用语义明确的 `pllm.residual(layer)` 或 `pllm.layer_logits((layer,))`，它们稳定读取该层**最后一次被访问时**输出的残差流。
- **与 `pllm.model.Adapter` 的交互**：
  如果负载中同时包含 `pllm.model.Adapter`（详见[改模型](/pllm/docs/guides/model-edits/)），权重增量 $$\Delta W$$ 严格**跟随每一次实际发生的投影调用**生效。若某段层在循环中被执行了 $$N$$ 遍，增量就会被施加 $$N$$ 遍；跳层时为补齐 K/V 而触发的原生 QKV 计算也会施加对应的 Q/V 增量。
- **训练 Overlay 与当步生效（`apply_updates()`）**：
  当与 `@pllm.train_in_forward` 组合时，在 `custom_forward` 内部调用 `layers.apply_updates()` 会立即结算并拼接自上个边界以来的投影捕获，执行优化器更新。当前 token 后续的层调用将当场读到更新后的私有权重版本。

---

## 常见报错与排查 {#zh-sec-25}

这一页的写法被拒绝时的报错原文、原因和改法，见报错目录的[改模型与 forward 结构](/pllm/docs/reference/errors/#zh-sec-8)、[显存预算](/pllm/docs/reference/errors/#zh-sec-11)几节。

---

## 自测 {#zh-sec-26}

1. 为什么 `@pllm.custom_residual` 的恒等改写（返回 `x + y` 或 `v`）与 eager 模式的原生 vLLM 逐位相同，却与默认图模式的原生 vLLM 存在微小 bf16 舍入差异？
<details markdown="1">
<summary>答案</summary>

默认图模式下的原生 vLLM 会使用 PyTorch Inductor 融合算子（例如 Fused Add+Norm）。当挂载 `@pllm.custom_residual` 时，vLLM 改为捕获模型原本的非融合算子到 CUDA graph 中。因此数学运算与 eager 模式完全逐位一致，但和 Inductor 融合算子之间存在正常的 bf16 浮点舍入差异。

</details>

2. 为什么 `kv.attend(q, topk=...)` 能够节省读取带宽与 FLOPs，却无法降低每个请求占用的 KV 显存容量？
<details markdown="1">
<summary>答案</summary>

`kv.attend(q, topk=...)` 只是在计算 attention softmax 和 GEMM 时仅读取被选中的位置，从而节省了显存读取带宽。但所有历史 token 的 KV 仍然必须常驻在物理显存页中，供未来的 token 计算相关性分数并从中挑选，因此无法释放物理显存块。

</details>

3. 在循环深度模型中，`layers(...)` 默认 `kv="commit"` 与 `kv="peek"` 分别有什么影响？为什么 Ouro 论文发现每一遍都覆盖写回（保留最后一遍）效果优异（78.85），而只留第一遍（18.73）会导致模型崩掉？
<details markdown="1">
<summary>答案</summary>

`kv="commit"` 每一遍都把算出的 K/V 写进缓存，后一遍覆盖前一遍，因此后续 token 看到的是这个 token 最后一遍循环提炼后的高质量 K/V；Ouro 论文实测 GSM8K 达 78.85。若改用 `kv="peek"`（或只在第一遍 commit、后续 peek），缓存里留下的只是第一遍浅层计算的粗糙 K/V，使得后续 token 无法利用深层循环的特征，导致按循环训练过的模型崩掉（GSM8K 仅 18.73）。

</details>

</div>

</div>

<div class="zine-doc-footer">
  <a class="zine-pill-link" data-i18n-zh="← 返回 pLLM 项目与文档中心" data-i18n-en="← Back to pLLM Project &amp; Docs" href="/pllm/#docs">← Back to pLLM Project &amp; Docs</a>
  <a class="zine-pill-link" href="https://github.com/LinesHogan/pLLM/blob/main/doc/guides/forward-structure.md" target="_blank" rel="noopener">GitHub &middot; doc/guides/forward-structure.md ↗</a>
</div>

<script defer src="/assets/js/bilingual-post.js"></script>
