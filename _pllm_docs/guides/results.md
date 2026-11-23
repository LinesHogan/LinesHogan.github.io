---
layout: page
title: "Where Results Go: Export, Logits, Sampling Guidance, and State Writes"
description: "pLLM Documentation · User Guides"
permalink: /pllm/docs/guides/results/
nav: false
---

<link rel="stylesheet" href="{{ '/assets/css/pllm-hub.css' | relative_url | bust_file_cache }}">

<div class="zine-dispatch" data-title-zh="结果去哪：导出、logits、采样引导与写状态" data-title-en="Where Results Go: Export, Logits, Sampling Guidance, and State Writes" data-desc-zh="pLLM 文档 · 用户指南" data-desc-en="pLLM Documentation · User Guides">
  <div class="zine-dispatch-links">
    <a class="zine-pill-link primary" data-i18n-zh="← 返回 pLLM 文档中心" data-i18n-en="← pLLM Docs Hub" href="/pllm/#docs">← pLLM Docs Hub</a>
    <a class="zine-pill-link" data-i18n-zh="文档导航地图" data-i18n-en="Docs Roadmap" href="/pllm/docs/">Docs Roadmap</a>
    <span class="zine-lang-badge" data-i18n-zh="用户指南" data-i18n-en="User Guides">User Guides</span>
  </div>
  <a class="zine-pill-link" data-i18n-zh="GitHub 源码 ↗" data-i18n-en="View on GitHub ↗" href="https://github.com/LinesHogan/pLLM/blob/main/doc/guides/results.md" target="_blank" rel="noopener">View on GitHub ↗</a>
</div>

<div id="markdown-content">

<div class="zine-lang-panel" data-lang-panel="en" lang="en" markdown="1">

<div id="en-where-results-go-export-logits-sampling-guidance-and-state-writes"></div>

In pLLM, a method reads input tensors and runs tensor math. Under continuous batching and CUDA graph replay, however, the framework cannot know where to send a computed tensor, how long to keep it, or when to publish it to subsequent steps or external processes unless you declare that destination explicitly.

Alongside gradient-based online training via optimizers (see [Online Training](/pllm/docs/guides/training/)), pLLM provides four effect decorators for non-training outputs, each targeting a distinct consumer:

1. **`@pllm.export`**: Bounded, asynchronous device-to-host export to CPU for monitoring, feature logging, and analysis;
2. **`@pllm.add_logits`**: Injects per-token biases into model logits before sampler filtering (e.g., DoLa contrastive decoding);
3. **`@pllm.guide_sampling`**: Produces hidden-state predictions projected through the LM head to reweight sampling distributions after top-k / top-p filtering (e.g., ESamp online distillation);
4. **`@pllm.write`**: Directly computes new state field values with custom tensor math and atomically publishes the next state version without optimizers or backpropagation.

This guide examines each effect's semantics, shape contracts, runtime cost, and constraints.

```python
from collections import defaultdict
from dataclasses import dataclass

import torch
from torch import nn

import pllm
from pllm import testing

CONFIG = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=8)
```

---

## 1. Exporting Observations: `@pllm.export` and `on_export` {#en-sec-1}

### Concept and definition {#en-concept-and-definition}

`@pllm.export` performs bounded, asynchronous copies of tensors computed during the forward pass or side tasks to the host CPU callback `on_export`, enabling feature logging, probe analysis, or SAE activation monitoring without blocking or slowing down GPU inference.

### Minimal runnable example {#en-minimal-runnable-example}

In this example, we read the residual stream after layer 12, compute the L2 norm of each request's current token, and export it to the host:

```python
seen: dict[int, list[float]] = defaultdict(list)


class StreamNorm(pllm.Workload):
    # Declare export channel: at most 8 rows per step, capped at 1 column per row
    @pllm.export(name="stream_norm", max_rows=8, max_cols=1)
    def norm(self, r: torch.Tensor = pllm.residual(12)) -> torch.Tensor:
        return r.float().norm(dim=-1, keepdim=True)

    # Asynchronous host consumer callback: runs in the engine process
    def on_export(self, event: pllm.io.TraceEvent) -> None:
        for identity, row in zip(event.identities, event.tensor().tolist()):
            seen[identity.prompt_index].append(row[0])


print(testing.check(StreamNorm(), CONFIG).passed)
```

When running on GPU, open `make_llm` with a context manager:

```python
from vllm import SamplingParams

GPU_CONFIG = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=8, gpu_memory_utilization=0.3)
GREEDY = SamplingParams(max_tokens=8, temperature=0.0)

with pllm.make_llm(GPU_CONFIG, workload=StreamNorm()) as llm:
    llm.generate(["The capital of France is", "One plus one is"], GREEDY)
    assert llm.diagnostics().eager_reason == ""  # Reading the residual stream preserves CUDA graph capture
```

### Shape contract and event structure {#en-shape-contract-and-event-structure}

- **Input/output shape**: The method returns a `[rows, cols]` tensor, where dimension 0 indexes request rows under continuous batching.
- **Truncation and cadence**: Each step exports at most `max_rows` rows and at most the first `max_cols` columns, executing once every `every` steps.
- **Drop counter `event.dropped_rows`**: If active rows in a step exceed `max_rows`, excess rows are truncated and the dropped count is recorded in `event.dropped_rows`.
- **Row identities `event.identities`**: A `tuple[RequestIdentity, ...]` aligned one-to-one with the rows of the event tensor:
  - `prompt_index`: Index of the prompt within the `generate` call (or the explicit index passed via `generate(..., prompt_indices=...)`);
  - `sample_index`: Relative sample index under that prompt (`0` for the first sample);
  - `request_id`: pLLM's internal request identifier (distinct from vLLM's `RequestOutput.request_id`).
- **Token positions `event.positions`**: Integer tuple giving each row's token offset within its request.
- **Dynamic layer provenance `event.layers`**: When an export method reads a single [`pllm.model.LayerRoute`](/pllm/docs/guides/model-edits/#en-advanced-entry-point-runtime-dynamic-layer-routing-pllmmodellayerroute), carries a tuple of each row's resolved physical layer index (`-1` when disabled or unmatched); empty `()` for static layer reads.
- **CPU tensor `event.tensor()`**: Returns an independent CPU tensor copy.
- **All-token reads `tokens="all"`**: By default, methods read the latest row per request; to observe every fresh token computed in a step (including all prefill-chunk tokens), pass `tokens="all"` on the input marker (e.g., `pllm.residual(12, tokens="all")`). All-token reads are available only to `@pllm.export` methods.

To export full-width hidden representations (width 896 on Qwen2.5-0.5B) for the last token of each prompt as prefill completes, read `tokens="prefill"`:

```python
prompt_ends: dict[int, torch.Tensor] = {}


class PromptEnd(pllm.Workload):
    @pllm.export(name="prompt_end", max_rows=4, max_cols=896)
    def last_token(self, r: torch.Tensor = pllm.residual(12, tokens="prefill")) -> torch.Tensor:
        return r.float()  # [rows, 896]: last row of each prompt that just finished prefill

    def on_export(self, event: pllm.io.TraceEvent) -> None:
        assert event.dropped_rows == 0
        for identity, row in zip(event.identities, event.tensor()):
            prompt_ends[identity.prompt_index] = row


print(testing.check(PromptEnd(), CONFIG).passed)
```

### Cost and overhead {#en-cost-and-overhead}

- **Device-to-host (D2H) transfer**: Issues a non-blocking asynchronous GPU-to-pinned-CPU copy of $$\text{rows} \times \text{cols} \times \text{element\_size}$$ bytes per step (upper-bounded by $$\text{max\_rows} \times \text{max\_cols} \times 8$$ bytes). Measured on RTX 4090 with `Qwen/Qwen2.5-7B-Instruct` (0 dropped rows, reconstructing independent CPU tensors on the background daemon thread): streaming full 3584-dim residuals for a single request (7 KiB/step) yields a pure decode TPOT of **16.47 ms** vs. **16.50 ms** on native vLLM (**100%**), and exporting full residuals across 128 concurrent sequences (896 KiB/step) retains **99.5%** of native throughput (TPOT **22.06 ms** vs. native **21.97 ms**).
- **Static cost reporting**: `pllm.explain` prints the per-step row limit, max bytes transferred, and event split count for every export channel.

### Constraints and rationale {#en-constraints-and-rationale}

1. **Whole-row packing and 32 KiB per-event limit**:
   - **Rule**: Each export event contains whole rows up to 32 KiB (32,768 bytes), and a single row must fit in one event—enforced as `max_cols * 8 <= 32768` (at most 4,096 columns).
   - **Why**: Export events live in fixed-size ring buffers (256 events, up to 32 KiB each). A single row is never split across two events, so one row must fit; when the rows exported in one step exceed 32 KiB in total, they are automatically split into multiple events.
2. **256-event queue capacity and `overflow="drop"` (default)**:
   - **Rule**: The background queue holds up to 256 pending events. By default (`overflow="drop"`), a full queue drops the oldest events and increments `export_dropped`; generation blocks to wait for queue space only when `overflow="block"` is set explicitly.
   - **Why**: **Observation must never stall inference.** If `on_export` performs slow disk I/O, network calls, or logging and falls behind generation, blocking by default would stall the entire serving engine. Throughput is protected by default; use `overflow="block"` only in offline evaluations where sample completeness matters more than throughput.
3. **`on_export` execution context and no re-entrancy**:
   - **Rule**: `on_export` runs asynchronously on a dedicated daemon thread inside the engine process. **Never call engine methods** (such as `llm.generate`) from inside `on_export`.
   - **Why**: The engine core holds scheduling locks and pipeline state; calling back into the engine from the asynchronous export thread causes deadlocks or state corruption.
4. **Automatic drain on context exit**:
   - **Rule**: Exiting `with pllm.make_llm(...)` or calling `llm.close()` automatically calls `drain()`, waiting until every queued export event has been processed by `on_export`. Your collected data is complete as soon as the context exits, with no tail loss.

### Runtime trace selector (`select_trace`) {#en-runtime-trace-selector-select_trace}

To capture tensors, gradients, or kernel timings temporarily at runtime without editing the workload class, use the host-side trace selector:

```python
from pllm.io import TraceSelection

# 1. Enable selector: temporarily capture the tensor named "loss"
llm.select_trace(TraceSelection(names=("loss",), tensors=True, max_tensor_bytes=32768))

# 2. Cursor recipe: record the absolute event index before generation
before = llm.trace_snapshot()
cursor = before.dropped + len(before.events)

# 3. Run inference
llm.generate(prompts, params)

# 4. Extract fresh events belonging exclusively to this call
after = llm.trace_snapshot()
fresh = after.events[max(0, cursor - after.dropped) :]

# 5. Disable observation with an empty selection
llm.select_trace(TraceSelection())
```

- **How the cursor recipe works**: The trace ring persists across `generate` calls rather than clearing automatically. If the queue overflows, the oldest events are dropped and added to `dropped`. Computing `cursor = before.dropped + len(before.events)` yields a monotonic absolute index, and subtracting `after.dropped` isolates only the fresh events produced during that `generate` call.
- **Calling discipline**: Call `llm.select_trace()` and `llm.trace_snapshot()` **serially between `generate` calls**, never concurrently from another thread while `generate` is running.

---

## 2. Adding Vocabulary Biases: `@pllm.add_logits` {#en-sec-2}

### Concept and definition {#en-concept-and-definition-1}

`@pllm.add_logits` adds a user-computed tensor element-wise to the model's unnormalized logits before the sampler applies temperature, top-k, and top-p filtering—ideal for training-free decoding interventions such as lexical suppression, DoLa contrastive decoding, or prompt-level biases.

### Minimal runnable example {#en-minimal-runnable-example-1}

This example injects an externally imported global bias vector scaled by `strength` across the vocabulary:

```python
@dataclass(frozen=True)
class LogitBias(pllm.Workload):
    strength: float = 1.0
    bias = pllm.InstanceState({"b": lambda m: (m.vocab_size,)}, init=pllm.Imported())

    def factor(self) -> float:
        return self.strength

    @pllm.add_logits()
    def add(
        self,
        sample: torch.Tensor = pllm.sample_index(),
        b: torch.Tensor = pllm.parameter("bias", "b"),
    ) -> torch.Tensor:
        # [rows, vocab]: broadcast bias across rows, returning exact zeros when neutral=0
        return self.factor() * b.float().expand(sample.shape[0], -1)


print(testing.check(LogitBias(), CONFIG, neutral=LogitBias(strength=0.0)).passed)
```

In DoLa contrastive decoding, the algorithm compares predictions from a mature (final) layer against a candidate premature layer to suppress surface-level premature patterns and amplify deeper factual predictions:

```python
@dataclass(frozen=True)
class DoLaContrast(pllm.Workload):
    candidate: int = 12

    def lens(self) -> tuple[int, ...]:
        return (self.candidate, -1)

    @pllm.add_logits()
    def contrast(
        self,
        logits: torch.Tensor = pllm.layer_logits(lens),
    ) -> torch.Tensor:
        mature = logits[:, -1].log_softmax(-1)
        premature = logits[:, 0].log_softmax(-1)
        # Add the difference onto the final-layer logits (native sampler logits equal logits[:, -1])
        return (mature - premature) - logits[:, -1]


print(testing.check(DoLaContrast(), CONFIG).passed)
```

For multi-concurrency benchmarks of DoLa, see [Example 06: Logit Lens and DoLa Contrastive Decoding](/pllm/docs/examples/06_reading_inside/).

### Shape contract {#en-shape-contract}

- **Return shape**: Must return a floating-point tensor of shape `[rows, vocab]` (`float32`, `float16`, or `bfloat16`).
- **Row alignment**: Dimension 0 (`rows`) must equal the number of tokens drawn by the sampler in the current step (one per sampled token).
- **Column alignment**: Dimension 1 must equal the model's full `vocab_size` (e.g., 151,936 for Qwen2.5-0.5B).

### Cost and overhead {#en-cost-and-overhead-1}

- `@pllm.add_logits` additions are fused into the sampler's existing pass over the vocabulary prior to sampling.
- Compute is purely element-wise memory traffic and addition—typically microseconds of latency with negligible impact on forward time.

### Constraints and rationale {#en-constraints-and-rationale-1}

1. **Multiple `@pllm.add_logits` methods are allowed**:
   - **Rule**: A workload (including subcomponents) may declare multiple `@pllm.add_logits` methods.
   - **Why**: Tensor addition is associative and commutative. The framework adds all `@pllm.add_logits` biases onto the model logits in declaration order before sampling, with no conflict between biases.
2. **Read logits with `pllm.layer_logits((-1,))`, not `pllm.base_logits()`**:
   - **Rule**: If an `@pllm.add_logits` method needs the model's unscaled logits, it must read `pllm.layer_logits((-1,))`; reading `pllm.base_logits()` is rejected at graph build time.
   - **Why**: On the engine's step timeline, `pllm.base_logits()` is defined as the unmodified base logits consumed by the sampler and is only materialized after sampling (`AFTER_SAMPLE` or inside `@pllm.train`). `@pllm.add_logits` runs before sampling (`BEFORE_SAMPLE`) to shape the sampler's input, before `base_logits()` exists. Use `pllm.layer_logits((-1,))` to read final-layer logits prior to sampling.
3. **Neutral baseline must return exact zeros**:
   - **Rule**: When verifying neutrality with `testing.check(..., neutral=...)`, passing the neutral instance (e.g., `strength=0.0`) must cause the method to return exact zeros everywhere.
   - **Why**: The neutral contract verifies that disabling the algorithm yields the exact unmodified base-model distribution; even tiny non-zero residuals or NaNs shift probabilities.
4. **Return width is checked at build time**:
   - **Rule**: Returning hidden width (e.g., `[rows, 896]`) instead of vocabulary width (`[rows, 151936]`) is caught during shape tracing in `pllm.explain` and `make_llm`, with an error naming the expected shape and fix.

---

## 3. Sampling Guidance: `@pllm.guide_sampling` {#en-sec-3}

### Concept and definition {#en-concept-and-definition-2}

`@pllm.guide_sampling` takes a hidden-width state prediction from your method, projects it to vocabulary space through the model's native LM head, and contrastively reweights the base logits over the surviving candidate tokens after top-k / top-p truncation—steering generation away from token patterns already predicted by that vector.

### Minimal runnable example and the ESamp pattern {#en-minimal-runnable-example-and-the-esamp-pattern}

The canonical online sampling-guidance algorithm is ESamp ([Tutorial 03: Decode-Time Training](/pllm/docs/tutorials/03_decode_training/)): each request owns a private `distiller` network trained during decode to predict deep hidden states from shallow hidden states, and guidance activates once the network has completed at least one training step:

```python
@dataclass(frozen=True)
class ShallowPredictor(pllm.Workload):
    beta: float = 0.5

    # What to store: each request owns a private distiller and an independent AdamW optimizer
    distiller = pllm.RequestState(
        init=pllm.FromModule(lambda m: nn.Sequential(nn.Linear(m.hidden_size, 4), nn.Linear(4, m.hidden_size))),
        optimizer=pllm.AdamW(1e-3),
    )

    # Selector: activate guidance only after at least 1 weight update
    def trained(self, versions: torch.Tensor = pllm.version("distiller")) -> torch.Tensor:
        return versions > 0

    # Config passing: return the instance field via a method
    def strength(self) -> float:
        return self.beta

    # Where results go: sampling guidance (with strength and row selector)
    @pllm.guide_sampling(beta=strength, rows="trained")
    def predict(
        self,
        shallow: torch.Tensor = pllm.hidden(0),
        distiller: nn.Module = pllm.module("distiller"),
    ) -> torch.Tensor:
        return distiller(shallow)  # Returns hidden-width prediction vector [rows, hidden]

    # Online training: predict deep from shallow with MSE loss
    @pllm.train(state="distiller")
    def loss(
        self,
        shallow: torch.Tensor = pllm.hidden(0),
        deep: torch.Tensor = pllm.hidden(-1),
        distiller: nn.Module = pllm.module("distiller"),
    ) -> torch.Tensor:
        return (distiller(shallow) - deep.float()).square().mean(dim=-1)


print(testing.check(ShallowPredictor(), CONFIG, neutral=ShallowPredictor(beta=0.0)).passed)
```

### Shape contract and guidance formula {#en-shape-contract-and-guidance-formula}

- **Return shape**: Must return `[rows, hidden]`, where column count matches the model's `hidden_size` (e.g., 896 on Qwen2.5-0.5B).
- **Reweighting formula**: Given method output $$\hat{h}$$, the framework projects through the LM head to obtain $$\text{pred\_logits} = \hat{h} \cdot W_{\text{lm\_head}}^T$$. Over the surviving candidate set after top-k and top-p truncation, guided logits are computed as:
  $$\text{logits}_{\text{guided}} = (1 + \beta) \cdot \text{logits} - \beta \cdot \text{pred\_logits}$$
  For $$\beta > 0$$, tokens easily predicted from shallow features are downweighted, encouraging tokens that carry genuinely deeper information.
- **Semantics of $$\beta = 0$$**: Declares no sampling guidance on this call, falling back to standard sampling.
- **Row selector `rows=`**: Bind a method returning a `[rows]` boolean mask via `rows="method_name"`. Unselected (`False`) rows keep unmodified sampling. See [Workload Anatomy: Selectors](/pllm/docs/guides/workload-anatomy/#en-selectors-acting-on-a-subset-of-rows).

### Cost and overhead {#en-cost-and-overhead-2}

- **One extra LM-head projection**: Performs one $$\hat{h} \cdot W_{\text{lm\_head}}^T$$ GEMM per decode step.
- `pllm.explain` reports this projection's FLOPs and memory budget explicitly under the sampler task.

### Constraints and rationale {#en-constraints-and-rationale-2}

1. **At most one `@pllm.guide_sampling` across the entire workload**:
   - **Rule**: A workload (including all nested components) can declare at most one `@pllm.guide_sampling` method. Declaring two raises `ValueError` at compile time.
   - **Why**: Guidance reweights the post–top-k/top-p distribution once per step using a single prediction. Whether two predictions should be summed, averaged, or applied sequentially is an algorithmic choice pLLM refuses to guess—combine them into one method, and use `@pllm.add_logits` for any additional vocabulary-space biases.
2. **Return width must equal `hidden_size`**:
   - **Rule**: Return `[rows, hidden]`. This width is not currently validated on CPU and will fail only on GPU, so verify the trailing dimension yourself.

---

## 4. Custom State Writes: `@pllm.write` {#en-sec-4}

### Concept and definition {#en-concept-and-definition-3}

`@pllm.write` computes new state field values using ordinary PyTorch tensor math and atomically publishes them as a tuple representing the next state version—without losses, autograd, or optimizers.

### Minimal runnable example {#en-minimal-runnable-example-2}

This example maintains a per-request running mean of the final hidden state at every decode step:

```python
class RunningMean(pllm.Workload):
    # What to store: each request owns a hidden-width mean vector
    stats = pllm.RequestState({"mean": pllm.Field(lambda width, _: (width,))})

    # Where results go: custom state write updating the mean field of stats
    @pllm.write(state="stats", fields=("mean",))
    def update(
        self,
        h: torch.Tensor = pllm.final_hidden(),
        mean: torch.Tensor = pllm.parameter("stats", "mean"),
    ) -> tuple[torch.Tensor]:
        # Return a tuple containing the new tensor version for each field
        return (0.9 * mean + 0.1 * h.float(),)

    # Export the L2 norm of the running mean
    @pllm.export(name="mean_norm", max_rows=8, max_cols=1)
    def report(
        self,
        mean: torch.Tensor = pllm.parameter("stats", "mean"),
    ) -> torch.Tensor:
        return mean.norm(dim=-1, keepdim=True)


print(testing.check(RunningMean(), CONFIG).passed)
```

### Shape contract and return values {#en-shape-contract-and-return-values}

- **Must return a tuple**: The return value must be a Python `tuple` whose length and order match `fields` strictly. **Even when updating a single field, return a 1-tuple `(new_tensor,)`**—returning a bare tensor raises an error.
- **Leading row axis for `RequestState`**: Each returned tensor for a `RequestState` must carry the leading request row axis `[rows, *field_shape]`.
- **Shape for `InstanceState`**: Each returned tensor for an `InstanceState` matches the field's declared shape (or carries `[rows, *field_shape]` if `per_row=True`).

### Cost and overhead {#en-cost-and-overhead-3}

- Forward-only tensor math: no autograd graph construction, no backward pass, and no optimizer momentum/variance memory.
- Writes directly into private state slots inside the CUDA graph and atomically flips version pointers at step end with zero CPU involvement.

### Constraints and rationale {#en-constraints-and-rationale-3}

1. **No `rows=` parameter; whole-batch writes and conditional updates**:
   - **Rule**: `@pllm.write` does not accept `rows=`. On every execution it writes for all active request rows in the batch.
   - **Why**: Each write publishes a complete new version of the state. "Updating only some rows" is equivalent to copying old values back for the remaining rows, which is expressed directly with `torch.where` without needing a separate mechanism.
   - **Conditional update pattern**: Read the old state into the method and select element-wise with `torch.where`:
     ```python
     class ConditionalUpdate(pllm.Workload):
         stats = pllm.RequestState({"mean": pllm.Field(lambda width, _: (width,))})

         @pllm.write(state="stats", fields=("mean",))
         def update_selected(
             self,
             h: torch.Tensor = pllm.final_hidden(),
             mean: torch.Tensor = pllm.parameter("stats", "mean"),
         ) -> tuple[torch.Tensor]:
             # Update only rows whose norm exceeds the threshold; keep others unchanged
             mask = h.float().norm(dim=-1, keepdim=True) > 1.0
             new_mean = 0.9 * mean + 0.1 * h.float()
             return (torch.where(mask, new_mean, mean),)

         @pllm.export(name="mean_norm", max_rows=8, max_cols=1)
         def report(self, mean: torch.Tensor = pllm.parameter("stats", "mean")) -> torch.Tensor:
             return mean.norm(dim=-1, keepdim=True)


     print(testing.check(ConditionalUpdate(), CONFIG).passed)
     ```
2. **Accumulator fields automatically use `float32`**:
   - **Rule**: When a write method reads a field that it also writes (a self-referential accumulator such as a running mean or step counter), `dtype="auto"` automatically promotes that field to `float32`.
   - **Why**: Over dozens or hundreds of autoregressive steps, small incremental updates added to larger values in `bfloat16` (7 mantissa bits) suffer underflow and truncate to zero (swallowed updates), stalling accumulation. `float32` preserves accuracy over long horizons.
3. **Reads see the version published before the current step began**:
   - **Rule**: A method executing on step $$n$$ always reads the version published at the end of step $$n-1$$ (or the initial value on step 0).
   - **Why**: pLLM's versioned publishing isolates reads from concurrent writes within a step, eliminating read-after-write hazards so the forward pass, custom writes, and exports can all be captured in CUDA graphs and run safely in parallel across CUDA streams.

---

## 5. Boundary Between Selectors (`rows=`) and Training (`@pllm.train`) {#en-sec-5}

When designing dataflow in a workload, keep the responsibilities of **row selectors** and **training effects** clear:

### Where row selectors (`rows=`) apply {#en-where-row-selectors-rows-apply}

- On `@pllm.add_logits` and `@pllm.guide_sampling`, `rows="selector_name"` restricts the effect to matching rows in the batch; unselected rows retain unmodified logits or standard sampling.
- In standalone per-request training without in-model intervention, training runs unconditionally overlapped with sampling and therefore rejects `rows=`. For complete selector rules and availability, see [Workload Anatomy: Selectors](/pllm/docs/guides/workload-anatomy/#en-selectors-acting-on-a-subset-of-rows).

### Non-training effects vs. `@pllm.train` {#en-non-training-effects-vs-pllmtrain}

| Mechanism | Driven By | Autograd? | Memory & Optimizer | Typical Uses |
| :--- | :--- | :--- | :--- | :--- |
| **`@pllm.export`** | Forward tensor copy | No | 32 KiB host ring buffer only | Probe analysis, SAE monitoring, activation capture |
| **`@pllm.add_logits`** | Pre-sample logit addition | No | No persistent extra VRAM | Vocabulary biases, DoLa contrastive decoding, constrained decoding |
| **`@pllm.guide_sampling`** | Post-filter LM-head reweighting | No | LM-head GEMM workspace | ESamp online distillation guidance |
| **`@pllm.write`** | Custom tensor math | No | State pages only | Running means, state machines, cross-step counters |
| **`@pllm.train`** | Optimizer gradient step | **Yes** | Depends on optimizer (e.g., AdamW moments) | Online fine-tuning, per-request LoRA, TTT adapters |

When your algorithm needs loss backpropagation and parameter updates, see [Online Training](/pllm/docs/guides/training/).

---

## 6. Common Errors and Troubleshooting {#en-sec-6}

When patterns on this page are rejected, exact error messages, causes, and fixes are listed in the [Effects and results](/pllm/docs/reference/errors/#en-effects-and-results) and [Phases: when a value exists](/pllm/docs/reference/errors/#en-phases-when-a-value-exists) sections of the error catalog.

---

## 7. Self-Check {#en-sec-7}

<details markdown="1">
<summary>1. Why does <code>@pllm.export</code> default to <code>overflow="drop"</code> rather than <code>"block"</code>, and why cannot a single row exceed 32 KiB?</summary>

Observation must never slow down generation: if `on_export` hits slow disk I/O, blocking would stall the entire inference engine, so the default drops excess events and increments a counter. Export events sit in fixed-size buffers of at most 32 KiB each, and a single row is never split across two events, so one row must fit (`max_cols * 8 <= 32768`).

</details>

<details markdown="1">
<summary>2. Why does reading <code>pllm.base_logits()</code> inside <code>@pllm.add_logits</code> fail, and how should you read the current step's model output instead?</summary>

`pllm.base_logits()` represents the unmodified base logits consumed by the sampler and is materialized only after sampling (`Phase.AFTER_SAMPLE`) for downstream training or feedback. `@pllm.add_logits` must run before sampling (`BEFORE_SAMPLE`) to supply a logit bias, when `base_logits()` does not yet exist. Read `pllm.layer_logits((-1,))` instead to obtain the current unnormalized final-layer logits.

</details>

<details markdown="1">
<summary>3. Why does <code>@pllm.write</code> not support <code>rows=</code>, and how do you update state for only a subset of requests?</summary>

Each write publishes one complete new version of the state. To update only selected requests, read the old state into the method and use `torch.where(mask, new_value, old_value)` to compute per-row conditional updates while writing unselected rows back unchanged.

</details>

<details markdown="1">
<summary>4. Why are multiple <code>@pllm.add_logits</code> methods allowed in one workload, whereas multiple <code>@pllm.guide_sampling</code> methods are disallowed?</summary>

Bias addition is associative and commutative, so outputs from multiple `@pllm.add_logits` methods can be summed onto the model logits in declaration order before entering the sampler. By contrast, `@pllm.guide_sampling` reweights the post–top-k/top-p candidate set once per step using one prediction; how two predictions should be combined is an algorithmic decision that belongs inside a single method.

</details>

</div>

<div class="zine-lang-panel is-hidden" data-lang-panel="zh" lang="zh-CN" markdown="1">

<div id="zh-sec-1"></div>

在 pLLM 中，一个方法负责读取张量并完成数学运算。但大模型推理处于连续批处理（continuous batching）与 CUDA Graph 回放的高并发执行环境中：方法计算出的张量如果不做显式声明，框架无法获知应当将其发往何处、以何种生命周期保存、何时发布给后续步骤或外部进程。

除通过梯度与优化器更新权重的在线训练（详见 [在线训练](/pllm/docs/guides/training/)）外，pLLM 提供了四种面向非训练结果的效果装饰器，分别对应四种不同的消费去向：

1. **`@pllm.export`**：将张量安全、有界地异步导出到宿主机 CPU，供监控、特征落盘与分析；
2. **`@pllm.add_logits`**：在采样过滤前向模型 logits 注入逐 token 偏置（如 DoLa 对比解码）；
3. **`@pllm.guide_sampling`**：输出隐藏状态预测，由 LM head 投影后引导 top-k / top-p 过滤后的采样分布（如 ESamp 在线蒸馏）；
4. **`@pllm.write`**：使用自定义张量数学直接覆写状态字段并原子发布新版本，无需优化器与反向传播。

本文逐一解析这四种效果的语义、契约、运行成本与边界限制。

```python
from collections import defaultdict
from dataclasses import dataclass

import torch
from torch import nn

import pllm
from pllm import testing

CONFIG = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=8)
```

---

## 1. 导出观测：`@pllm.export` 与 `on_export` {#zh-sec-2}

### 概念与定义 {#zh-sec-3}

`@pllm.export` 将模型前向或计算过程中产生的张量以有界、异步的方式拷贝并送到宿主机 CPU 的 `on_export` 回调中，供特征记录、探针分析或 SAE 激活监控，且绝不阻塞或拖慢 GPU 上的主干推理。

### 最小可运行示例 {#zh-sec-4}

在下面的示例中，我们读取第 12 层之后的残差流，计算每个请求当前 token 的 L2 范数，并将其导出到宿主机：

```python
seen: dict[int, list[float]] = defaultdict(list)


class StreamNorm(pllm.Workload):
    # 声明导出通道：每步至多导出 8 行、每行截取前 1 列
    @pllm.export(name="stream_norm", max_rows=8, max_cols=1)
    def norm(self, r: torch.Tensor = pllm.residual(12)) -> torch.Tensor:
        return r.float().norm(dim=-1, keepdim=True)

    # 宿主机异步消费回调：在运行模型的进程中异步执行
    def on_export(self, event: pllm.io.TraceEvent) -> None:
        for identity, row in zip(event.identities, event.tensor().tolist()):
            seen[identity.prompt_index].append(row[0])


print(testing.check(StreamNorm(), CONFIG).passed)
```

在 GPU 引擎上实际运行时，通过上下文管理器启动 `make_llm`：

```python
from vllm import SamplingParams

GPU_CONFIG = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=8, gpu_memory_utilization=0.3)
GREEDY = SamplingParams(max_tokens=8, temperature=0.0)

with pllm.make_llm(GPU_CONFIG, workload=StreamNorm()) as llm:
    llm.generate(["The capital of France is", "One plus one is"], GREEDY)
    assert llm.diagnostics().eager_reason == ""  # 读取残差流保持 CUDA Graph 捕获
```

### 形状契约与事件结构 {#zh-sec-5}

- **输入与输出形状**：方法返回形状为 `[rows, cols]` 的张量。第 0 维为请求行（连续批处理中的行）。
- **截断与门控**：每一步前向至多导出 `max_rows` 行，列宽截取前 `max_cols` 列，每 `every` 步执行一次。
- **丢弃计数 `event.dropped_rows`**：若当前步的活跃行数超过 `max_rows`，超出部分的行会被截断丢弃，并在该事件的 `event.dropped_rows` 字段中记录丢弃的行数。
- **行身份 `event.identities`**：类型为 `tuple[RequestIdentity, ...]`，与事件张量的每一行一一对应：
  - `prompt_index`：该行所属 Prompt 在单次 `generate` 调用中的序号（若调用时指定了 `generate(..., prompt_indices=...)`，则为显式指定的索引）；
  - `sample_index`：每个 Prompt 下采样的相对序号（首个采样为 0）；
  - `request_id`：pLLM 内部的请求标识（不同于 vLLM 的 `RequestOutput.request_id`）。
- **位置索引 `event.positions`**：整型元组，指示各行在其请求内的 token 偏移量。
- **动态层溯源 `event.layers`**：当导出方法读取单个 [`pllm.model.LayerRoute`](/pllm/docs/guides/model-edits/#zh-sec-18) 时，附带每行实际命中的物理层号元组（未命中或关闭时为 `-1`）；静态层读取时为空元组 `()`。
- **获取 CPU 张量 `event.tensor()`**：调用该方法返回一个独立的 CPU 张量副本。
- **全量 token 读取 `tokens="all"`**：默认情况下，方法读取当前步的最新行；若需要观测一个步骤内计算出的全部 token（包括 prompt 的全部切片分块），可在输入标记中指定 `tokens="all"`（例如 `pllm.residual(12, tokens="all")`）。全量 token 读取仅对 `@pllm.export` 方法开放。

如果需要在 Prefill 结束时导出每个 Prompt 最后一个 token 的全宽隐藏特征（Qwen2.5-0.5B 隐藏层宽度为 896），可以读取 `tokens="prefill"`：

```python
prompt_ends: dict[int, torch.Tensor] = {}


class PromptEnd(pllm.Workload):
    @pllm.export(name="prompt_end", max_rows=4, max_cols=896)
    def last_token(self, r: torch.Tensor = pllm.residual(12, tokens="prefill")) -> torch.Tensor:
        return r.float()  # [rows, 896]：刚完成 prefill 的 prompt 的最后一行

    def on_export(self, event: pllm.io.TraceEvent) -> None:
        assert event.dropped_rows == 0
        for identity, row in zip(event.identities, event.tensor()):
            prompt_ends[identity.prompt_index] = row


print(testing.check(PromptEnd(), CONFIG).passed)
```

### 成本与开销 {#zh-sec-6}

- **设备到主机（D2H）传输**：每步产生非阻塞的 GPU 到 CPU 锁页内存异步拷贝，传输字节数为 $$\text{rows} \times \text{cols} \times \text{element\_size}$$（上界为 $$\text{max\_rows} \times \text{max\_cols} \times 8$$ 字节）。在 RTX 4090 上实测 `Qwen/Qwen2.5-7B-Instruct`（0 丢行并在后台守护线程逐行还原为 CPU 张量）：单请求流式导出整层 3584 维残差（每步 7 KiB）的纯 decode TPOT 为 **16.47 ms**（原生 vLLM 为 **16.50 ms**，**100%**）；128 并发序列同时全宽导出（每步 896 KiB）时吞吐仍达原生的 **99.5%**（TPOT **22.06 ms** 对原生 **21.97 ms**）。
- **静态成本展示**：`pllm.explain` 在计划阶段会直接输出每个导出任务每步传输的行数、上限字节数和拆分的事件数。

### 限制与设计原因 {#zh-sec-7}

1. **整行打包且单事件不超过 32 KiB**：
   - 规则：每个导出事件由若干完整行组成，单个事件的尺寸上限为 32 KiB（32768 字节），且单行数据必须能完整装进一个事件中，即强制校验 `max_cols * 8 <= 32768`（至多 4096 列）。
   - 原因：导出事件放在固定大小的缓冲里（256 个事件，每个至多 32 KiB），一行不会被拆到两个事件里，所以一行必须装得下。一步导出的行加起来超过 32 KiB 时，会自动拆成多个事件递送。
2. **256 事件队列容量与 `overflow="drop"`（默认）**：
   - 规则：后台队列最多容纳 256 个待消费事件。默认策略为 `overflow="drop"`，队列满时丢弃最旧的事件并计入 `export_dropped`；仅当显式指定 `overflow="block"` 时，才会阻塞生成等待队列腾出空间。
   - 原因：**观测绝不能阻塞前向与生成**。如果 `on_export` 回调中包含耗时的文件 I/O、网络传输或日志打印，导致消费速度赶不上生成速度，若默认阻塞将直接拖垮整个高并发推理引擎。因此系统优先保障模型推理吞吐；仅在对样本完整性有严格要求且能承受吞吐下降的离线评估场景中，才显式配置 `overflow="block"`。
3. **`on_export` 执行环境与禁止重入**：
   - 规则：`on_export` 在运行模型的同一个系统进程内的独立守护线程中异步调用。**不要在 `on_export` 内部调用引擎方法**（如 `llm.generate`）。
   - 原因：推理引擎核心持有调度锁与请求管线状态，在异步回调线程中调用引擎会引发线程死锁或状态损坏。
4. **退出上下文自动排空（drain）**：
   - 规则：当代码退出 `with pllm.make_llm(...)` 代码块或调用 `llm.close()` 时，系统会自动调用 `drain()`，挂起等待直到所有排队的导出事件完全被 `on_export` 处理完毕。这确保了离开上下文后采集的数据完全就绪，不会发生尾部丢失。

### 运行时 Trace 选择器（`select_trace`） {#zh-sec-8}

如果不想修改负载类代码，仅希望在运行时临时截取某次生成的张量、梯度或内核耗时，可以使用宿主端提供的运行时 Trace 选择器：

```python
from pllm.io import TraceSelection

# 1. 开启选择器：临时捕获名为 "loss" 的张量数据
llm.select_trace(TraceSelection(names=("loss",), tensors=True, max_tensor_bytes=32768))

# 2. 游标配方：记录生成前的绝对事件序号
before = llm.trace_snapshot()
cursor = before.dropped + len(before.events)

# 3. 执行推理
llm.generate(prompts, params)

# 4. 提取本次生成独有的新鲜事件
after = llm.trace_snapshot()
fresh = after.events[max(0, cursor - after.dropped) :]

# 5. 用空选择器关闭观测
llm.select_trace(TraceSelection())
```

- **游标配方原理**：Trace 事件队列在跨 `generate` 调用时是全局保留的，不会自动清空。如果队列发生溢出，最旧的事件会被丢弃并累加到 `dropped` 计数中。因此，使用 `cursor = before.dropped + len(before.events)` 构造绝对单调递增索引，并在截取时通过 `cursor - after.dropped` 抵消丢弃量，才能精准切出“仅属于本次调用”的新鲜事件。
- **调用时机约束**：`llm.select_trace()` 与 `llm.trace_snapshot()` **只能在两次 `generate` 调用之间串行调用**，不要在 `generate` 正在执行时跨线程并发调用。

---

## 2. 注入词表偏置：`@pllm.add_logits` {#zh-sec-9}

### 概念与定义 {#zh-sec-10}

`@pllm.add_logits` 在采样器执行 temperature、top-k 和 top-p 概率截断之前，将用户计算出的张量按元素直接累加到模型的未归一化 logits 上，用于实现无训练的解码干预（如抑制特定词、对比解码 DoLa、提示词偏置等）。

### 最小可运行示例 {#zh-sec-11}

下面的示例演示如何根据每个 Prompt 的采样序号给词表注入一个外部导入的全局偏置向量：

```python
@dataclass(frozen=True)
class LogitBias(pllm.Workload):
    strength: float = 1.0
    bias = pllm.InstanceState({"b": lambda m: (m.vocab_size,)}, init=pllm.Imported())

    def factor(self) -> float:
        return self.strength

    @pllm.add_logits()
    def add(
        self,
        sample: torch.Tensor = pllm.sample_index(),
        b: torch.Tensor = pllm.parameter("bias", "b"),
    ) -> torch.Tensor:
        # [rows, vocab]：将偏置按行广播，并在 neutral=0 时输出全零
        return self.factor() * b.float().expand(sample.shape[0], -1)


print(testing.check(LogitBias(), CONFIG, neutral=LogitBias(strength=0.0)).passed)
```

在对比解码（DoLa）算法中，算法需要对比成熟层（最后一层）与候选浅层的预测差异，抑制过早收敛的浅层表面模式并抬高深层新知识：

```python
@dataclass(frozen=True)
class DoLaContrast(pllm.Workload):
    candidate: int = 12

    def lens(self) -> tuple[int, ...]:
        return (self.candidate, -1)

    @pllm.add_logits()
    def contrast(
        self,
        logits: torch.Tensor = pllm.layer_logits(lens),
    ) -> torch.Tensor:
        mature = logits[:, -1].log_softmax(-1)
        premature = logits[:, 0].log_softmax(-1)
        # 将差值加在最后一层 logits 上（采样器原生 logits 等价于 logits[:, -1]）
        return (mature - premature) - logits[:, -1]


print(testing.check(DoLaContrast(), CONFIG).passed)
```

关于 DoLa 在多并发下的实测表现，详见示例教程 [06 logit lens 与 DoLa 对比解码](/pllm/docs/examples/06_reading_inside/)。

### 形状契约 {#zh-sec-12}

- **返回值形状**：必须返回形状为 `[rows, vocab]` 的浮点张量（float32 或 float16/bfloat16）。
- **行对齐**：第 0 维的 `rows` 必须严格等于当前步采样器抽取的 token 行数（每行抽取一个 token）。
- **列对齐**：第 1 维必须严格等于模型的完整词表大小（`vocab_size`，例如 Qwen2.5-0.5B 为 151936）。

### 成本与开销 {#zh-sec-13}

- `@pllm.add_logits` 的累加由框架在采样前完成，和采样器已有的对词表的遍历合在一起。
- 计算量为纯粹的逐元素内存搬运与加法，耗时通常在微秒级别，几乎不增加前向时延。

### 限制与设计原因 {#zh-sec-14}

1. **允许多个 `@pllm.add_logits` 并存**：
   - 规则：同一个负载（包括子组件）中可以声明多个 `@pllm.add_logits` 方法。
   - 原因：张量加法满足交换律与结合律。框架会按照声明顺序，将所有 `@pllm.add_logits` 返回的偏置张量依次叠加在模型 logits 上，最后统一进入采样器，各个偏置之间不存在互斥冲突。
2. **读 logits 必须用 `pllm.layer_logits((-1,))`，不要使用 `pllm.base_logits()`**：
   - 规则：在 `@pllm.add_logits` 方法内部若需要参考模型当前的未缩放 logits，必须通过 `pllm.layer_logits((-1,))` 读取；若尝试读取 `pllm.base_logits()`，会在图构建时被直接拒绝并报错。
   - 原因：在引擎的时间步划分中，`pllm.base_logits()` 的物理定义是“模型产出、送入采样器前的未经修改的原始基准 logits”。在流水线调度中，该张量是在采样阶段完成之后才物化并暴露给外部的（仅在 `AFTER_SAMPLE` 阶段或 `@pllm.train` 中可读）。而 `@pllm.add_logits` 必须在采样前（`BEFORE_SAMPLE`）执行并影响采样输入，此时 `base_logits()` 根本尚未物化生成。读取最后一层流产生的 logits 应使用 `pllm.layer_logits((-1,))`。
3. **基准中性（neutral）实例必须返回精确为零**：
   - 规则：在通过 `testing.check(..., neutral=...)` 校验中性契约时，传入中性实例后（如 `strength=0.0`），该方法必须返回所有元素绝对为 0 的张量。
   - 原因：中性检查用于验证“当算法强度关闭时，模型输出分布与未经干涉的原生模型完全一致”。若中性参数下输出微小的非零数值或 NaN，会导致基准概率分布发生漂移。
4. **返回宽度在构建时检查**：
   - 规则：返回了隐藏层宽度（如 `[rows, 896]`）而不是词表宽度（`[rows, 151936]`）时，`pllm.explain` 和 `make_llm` 在追踪形状时就报错，报错给出期望的形状和改法。

---

## 3. 采样引导：`@pllm.guide_sampling` {#zh-sec-15}

### 概念与定义 {#zh-sec-16}

`@pllm.guide_sampling` 接收方法返回的一个隐藏层宽度的状态预测，由框架调用模型原生的 LM head 将其投影至词表空间，并在 top-k / top-p 概率截断后的幸存候选 token 集合上，对模型原始 logits 进行对比重加权采样，将生成推离预测向量所代表的语义模式。

### 最小可运行示例与 ESamp 范式 {#zh-sec-17}

在线采样引导最典型的算法是 ESamp（教程 [03 decode 时训练](/pllm/docs/tutorials/03_decode_training/)）：每个请求持有一个私有小网络 `distiller`，在 decode 阶段一边生成一边用浅层隐藏状态预测深层隐藏状态。只有当小网络学习至少一步后，才开启引导：

```python
@dataclass(frozen=True)
class ShallowPredictor(pllm.Workload):
    beta: float = 0.5

    # 存什么：每个请求拥有一份私有蒸馏网络与独立的 AdamW 优化器
    distiller = pllm.RequestState(
        init=pllm.FromModule(lambda m: nn.Sequential(nn.Linear(m.hidden_size, 4), nn.Linear(4, m.hidden_size))),
        optimizer=pllm.AdamW(1e-3),
    )

    # 选择器：仅对已经发生过至少 1 次权重更新的状态激活引导
    def trained(self, versions: torch.Tensor = pllm.version("distiller")) -> torch.Tensor:
        return versions > 0

    # 配置传递：通过方法返回实例字段值
    def strength(self) -> float:
        return self.beta

    # 结果去哪：采样引导（指定强度与行选择器）
    @pllm.guide_sampling(beta=strength, rows="trained")
    def predict(
        self,
        shallow: torch.Tensor = pllm.hidden(0),
        distiller: nn.Module = pllm.module("distiller"),
    ) -> torch.Tensor:
        return distiller(shallow)  # 返回隐藏层宽度的预测向量 [rows, hidden]

    # 在线训练：用浅层预测深层，计算均方误差损失
    @pllm.train(state="distiller")
    def loss(
        self,
        shallow: torch.Tensor = pllm.hidden(0),
        deep: torch.Tensor = pllm.hidden(-1),
        distiller: nn.Module = pllm.module("distiller"),
    ) -> torch.Tensor:
        return (distiller(shallow) - deep.float()).square().mean(dim=-1)


print(testing.check(ShallowPredictor(), CONFIG, neutral=ShallowPredictor(beta=0.0)).passed)
```

### 形状契约与引导公式 {#zh-sec-18}

- **返回值形状**：必须返回形状为 `[rows, hidden]` 的张量，列数严格等于模型的隐藏层维度（`hidden_size`，如 Qwen2.5-0.5B 为 896）。
- **重加权公式**：设方法输出为 $$\hat{h}$$，框架通过 LM head 得到预测 logits $$\text{pred\_logits} = \hat{h} \cdot W_{\text{lm\_head}}^T$$。在经过 top-k 与 top-p 截断保留的幸存 token 集合上，重加权后的采样 logits 为：
  $$\text{logits}_{\text{guided}} = (1 + \beta) \cdot \text{logits} - \beta \cdot \text{pred\_logits}$$
  当 $$\beta > 0$$ 时，模型会主动避开浅层已能轻易预测出的 token，鼓励生成包含深层全新信息的 token。
- **$$\beta = 0$$ 的语义**：声明本次调用不产生采样引导，采样器直接回退为标准采样。
- **行选择器 `rows=`**：通过 `rows="method_name"` 绑定一个返回 `[rows]` 布尔掩码的方法。未选中的行（False）保持原生采样，不受引导干扰。关于选择器的声明方式，参见 [负载的组成：选择器](/pllm/docs/guides/workload-anatomy/#zh-sec-13)。

### 成本与开销 {#zh-sec-19}

- **额外的一次 LM-Head 投影**：在每个解码步中，框架需要执行一次 $$\hat{h} \cdot W_{\text{lm\_head}}^T$$ 的矩阵乘法投影。
- 在 `pllm.explain` 报告中，该成本会显式计入采样器任务的投影 FLOP 与显存预算中。

### 限制与设计原因 {#zh-sec-20}

1. **全负载至多只能有一个 `@pllm.guide_sampling`**：
   - 规则：一个工作负载（包括其嵌套的所有子组件）中至多声明一个 `@pllm.guide_sampling` 方法。若声明了两个，框架在编译时会直接抛出 `ValueError`。
   - 原因：引导要在 top-k / top-p 过滤之后的分布上按一个预测重新加权，采样器每步只做这一次。两个预测该相加、取平均还是先后应用，是算法本身的选择，pLLM 不替你猜：在一个方法里把它们合成一个预测返回；其余的词表修正用 `@pllm.add_logits`。
2. **返回宽度要等于 hidden 宽度**：
   - 规则：返回 `[rows, hidden]`。这个宽度目前不在 CPU 上检查，写错时要到 GPU 上才会出错，所以写完先自己核对最后一维。

---

## 4. 自定义写状态：`@pllm.write` {#zh-sec-21}

### 概念与定义 {#zh-sec-22}

`@pllm.write` 允许你使用普通的 PyTorch 张量数学直接计算状态字段的新数值，并以元组形式原子发布为该状态的下一个版本，无需经过损失函数、自动求导或优化器。

### 最小可运行示例 {#zh-sec-23}

下面的示例在每个 decode 步维护一个记录最终隐藏状态滑动均值的统计量：

```python
class RunningMean(pllm.Workload):
    # 存什么：每个请求拥有一份宽度为 hidden 的均值向量
    stats = pllm.RequestState({"mean": pllm.Field(lambda width, _: (width,))})

    # 结果去哪：自定义写状态，更新 stats 状态的 mean 字段
    @pllm.write(state="stats", fields=("mean",))
    def update(
        self,
        h: torch.Tensor = pllm.final_hidden(),
        mean: torch.Tensor = pllm.parameter("stats", "mean"),
    ) -> tuple[torch.Tensor]:
        # 返回元组，包含对应字段的新版本张量
        return (0.9 * mean + 0.1 * h.float(),)

    # 导出统计量的 L2 范数
    @pllm.export(name="mean_norm", max_rows=8, max_cols=1)
    def report(
        self,
        mean: torch.Tensor = pllm.parameter("stats", "mean"),
    ) -> torch.Tensor:
        return mean.norm(dim=-1, keepdim=True)


print(testing.check(RunningMean(), CONFIG).passed)
```

### 形状契约与返回值 {#zh-sec-24}

- **必须返回元组（tuple）**：方法的返回值必须是一个 Python 元组，元组内的张量数量与顺序必须与 `fields` 声明的字段严格匹配。**即使只更新一个字段，也必须返回单元素元组 `(new_tensor,)`**，直接返回裸张量会报错。
- **请求状态的领先行轴**：对于 `RequestState`，返回的每个字段张量必须具有领先的请求行轴：`[rows, *field_shape]`。
- **实例状态的形状**：对于 `InstanceState`，返回的张量形状与字段本身的声明形状一致（若配置了 `per_row=True` 则同样具有 `[rows]` 行轴）。

### 成本与开销 {#zh-sec-25}

- 纯前向张量运算，无需构建计算图，无反向求导计算，无优化器动量与方差显存开销。
- 状态写入由框架在 CUDA Graph 内直接向分配的私有显存槽位写入，步末自动原子切换版本指针，零 CPU 交互开销。

### 限制与设计原因 {#zh-sec-26}

1. **没有 `rows=` 参数，全量写入与条件更新**：
   - 规则：`@pllm.write` 不接受 `rows=` 参数。每一轮执行时，它会对当前批次中的所有活跃请求行进行统一写入。
   - 原因：一次写入发布这份状态的一个完整新版本。"只更新部分行"等价于让其余行原样写回旧值，用 `torch.where` 直接表达，不需要另一套机制。
   - 条件更新做法：如果算法只需要对满足特定条件的请求行更新状态、其余行保持旧值，应在方法体中读出旧状态，并使用 `torch.where` 进行按元素选择：
     ```python
     class ConditionalUpdate(pllm.Workload):
         stats = pllm.RequestState({"mean": pllm.Field(lambda width, _: (width,))})

         @pllm.write(state="stats", fields=("mean",))
         def update_selected(
             self,
             h: torch.Tensor = pllm.final_hidden(),
             mean: torch.Tensor = pllm.parameter("stats", "mean"),
         ) -> tuple[torch.Tensor]:
             # 仅对范数大于阈值的请求更新，其余请求保留原值
             mask = h.float().norm(dim=-1, keepdim=True) > 1.0
             new_mean = 0.9 * mean + 0.1 * h.float()
             return (torch.where(mask, new_mean, mean),)

         @pllm.export(name="mean_norm", max_rows=8, max_cols=1)
         def report(self, mean: torch.Tensor = pllm.parameter("stats", "mean")) -> torch.Tensor:
             return mean.norm(dim=-1, keepdim=True)


     print(testing.check(ConditionalUpdate(), CONFIG).passed)
     ```
2. **累加器字段自动以 `float32` 存储**：
   - 规则：当一个写入方法同时读取它要写入的字段时（即形成自引用累加器，如滑动均值、步数计数器），在 `dtype="auto"` 模式下，框架会自动将该字段存储为 `float32` 格式。
   - 原因：在连续数十或上百步的自回归生成中，累加更新通常具有较小的变化步长。如果采用仅有 7 位尾数的 `bfloat16`，微小的更新量加到较大的现有数值上时会发生严重的浮点精度下溢（更新被直接截断归零，即更新吞噬），导致累加停滞；`float32` 保证了跨步长时间累加过程中的数值稳定性。
3. **读取看到的是当前步开始前已发布的版本**：
   - 规则：在第 $$n$$ 步执行的方法读取状态时，看到的永远是第 $$n-1$$ 步结束时发布的版本（第 0 步看到的是初始值）。
   - 原因：pLLM 采用版本化发布模型。这种步间隔离机制完全消除了同一时间步内多个方法之间的读写冒险（read-after-write hazard），使得模型前向、自定义写入与观测方法能够被完全捕获进 CUDA Graph 并在不同的 CUDA stream 上安全并行执行。

---

## 5. 选择器（`rows=`）与训练（`@pllm.train`）的边界 {#zh-sec-27}

在设计算法的数据流向时，需要清晰区分**行选择器**与**训练效果**的职责边界：

### 行选择器（`rows=`）的适用范围 {#zh-sec-28}

- 在 `@pllm.add_logits` 和 `@pllm.guide_sampling` 上，可以通过 `rows="selector_name"` 绑定一个方法，只让当前批次中满足条件的请求行生效；未选中的行保持原生 logits 或原生采样。
- 在无模型干预的独立请求训练中，训练是与采样重叠的无条件步进更新，因此会拒绝 `rows=`。选择器的完整声明语法与可用性规则详见 [负载的组成：选择器](/pllm/docs/guides/workload-anatomy/#zh-sec-13)。

### 非训练效果 vs `@pllm.train` {#zh-sec-29}

| 机制 | 驱动方式 | 是否反向求导 | 显存与优化器 | 典型场景 |
| :--- | :--- | :--- | :--- | :--- |
| **`@pllm.export`** | 前向张量传递 | 否 | 仅 32 KiB 宿主缓冲池 | 探针分析、SAE 特征监控、激活落盘 |
| **`@pllm.add_logits`** | 采样前张量累加 | 否 | 无额外持久显存 | 词表偏置、DoLa 对比解码、语法受限解码 |
| **`@pllm.guide_sampling`** | 采样过滤后 LM head 投影加权 | 否 | LM head GEMM 显存 | ESamp 在线蒸馏引导 |
| **`@pllm.write`** | 自定义张量数学计算 | 否 | 仅状态本身页面显存 | 运行均值、状态机流转、跨步计数器 |
| **`@pllm.train`** | 优化器梯度步进更新 | **是** | 依赖优化器（如 AdamW 动量与方差） | 在线微调、每请求 LoRA、TTT 适配器 |

若算法需要通过损失函数（Loss）反向传播梯度并更新参数，请参阅 [在线训练](/pllm/docs/guides/training/)。

---

## 6. 常见错误与排查 {#zh-sec-30}

这一页的写法被拒绝时的报错原文、原因和改法，见报错目录的[效果与结果](/pllm/docs/reference/errors/#zh-sec-5)、[阶段](/pllm/docs/reference/errors/#zh-sec-3)几节。

---

## 7. 自测小问题 {#zh-sec-31}

<details markdown="1">
<summary>1. 为什么 <code>@pllm.export</code> 默认采用 <code>overflow="drop"</code> 而非 <code>"block"</code>？为什么单行张量不能超过 32 KiB？</summary>

观测不能拖慢生成：<code>on_export</code> 若遇到缓慢的磁盘 I/O，阻塞等待会让整个引擎停下来，所以默认丢弃事件、只计数。导出事件放在固定大小的缓冲里，每个事件至多 32 KiB，一行不会被拆到两个事件里，所以一行必须装得下（<code>max_cols * 8 <= 32768</code>）。

</details>

<details markdown="1">
<summary>2. 为什么在 <code>@pllm.add_logits</code> 中读取 <code>pllm.base_logits()</code> 会报错？应当如何正确读取当前步的模型输出？</summary>

<code>pllm.base_logits()</code> 在流水线中的语义是“采样器执行时所消费的未经干预的原生基准 logits”。在调度时序上，该张量是在采样动作执行完成后才正式物化并对外暴露的（位于 <code>Phase.AFTER_SAMPLE</code>），供后续训练或反馈读取。而 <code>@pllm.add_logits</code> 必须在采样前（<code>BEFORE_SAMPLE</code>）运行以提供偏置输入，此时 <code>base_logits()</code> 尚未存在，因此读取会报错。若需要读取最后一层流产生的当前未归一化 logits，应当使用 <code>pllm.layer_logits((-1,))</code>。

</details>

<details markdown="1">
<summary>3. 为什么 <code>@pllm.write</code> 不支持 <code>rows=</code> 参数？如何实现仅对部分请求更新状态？</summary>

一次写入发布这份状态的一个完整新版本。若算法只需对部分请求更新，应当在方法内部同时读取旧状态参数，并利用 <code>torch.where(mask, new_value, old_value)</code> 实现按行条件更新，将未选中的请求原样写回。

</details>

<details markdown="1">
<summary>4. 为什么一个负载中允许并存多个 <code>@pllm.add_logits</code>，却不要声明多个 <code>@pllm.guide_sampling</code>？</summary>

偏置累加运算满足数学上的结合律与交换律，多个 <code>@pllm.add_logits</code> 的输出可以按声明顺序直接累加到模型 logits 上，最后统一进入采样器；而 <code>@pllm.guide_sampling</code> 在 top-k / top-p 截断后的候选集合上按一个预测重新加权，采样器每步只做这一次；两个预测怎样合成是算法的选择，应在一个方法里合成后返回。

</details>

</div>

</div>

<div class="zine-doc-footer">
  <a class="zine-pill-link" data-i18n-zh="← 返回 pLLM 项目与文档中心" data-i18n-en="← Back to pLLM Project &amp; Docs" href="/pllm/#docs">← Back to pLLM Project &amp; Docs</a>
  <a class="zine-pill-link" href="https://github.com/LinesHogan/pLLM/blob/main/doc/guides/results.md" target="_blank" rel="noopener">GitHub &middot; doc/guides/results.md ↗</a>
</div>

<script defer src="/assets/js/bilingual-post.js"></script>
