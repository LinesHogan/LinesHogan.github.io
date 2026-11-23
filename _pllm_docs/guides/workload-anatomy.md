---
layout: page
title: "Workload Anatomy"
description: "pLLM Documentation · User Guides"
permalink: /pllm/docs/guides/workload-anatomy/
nav: false
---

<link rel="stylesheet" href="{{ '/assets/css/pllm-hub.css' | relative_url | bust_file_cache }}">

<div class="zine-dispatch" data-title-zh="负载的组成" data-title-en="Workload Anatomy" data-desc-zh="pLLM 文档 · 用户指南" data-desc-en="pLLM Documentation · User Guides">
  <div class="zine-dispatch-links">
    <a class="zine-pill-link primary" data-i18n-zh="← 返回 pLLM 文档中心" data-i18n-en="← pLLM Docs Hub" href="/pllm/#docs">← pLLM Docs Hub</a>
    <a class="zine-pill-link" data-i18n-zh="文档导航地图" data-i18n-en="Docs Roadmap" href="/pllm/docs/">Docs Roadmap</a>
    <span class="zine-lang-badge" data-i18n-zh="用户指南" data-i18n-en="User Guides">User Guides</span>
  </div>
  <a class="zine-pill-link" data-i18n-zh="GitHub 源码 ↗" data-i18n-en="View on GitHub ↗" href="https://github.com/LinesHogan/pLLM/blob/main/doc/guides/workload-anatomy.md" target="_blank" rel="noopener">View on GitHub ↗</a>
</div>

<div id="markdown-content">

<div class="zine-lang-panel" data-lang-panel="en" lang="en" markdown="1">

<div id="en-workload-anatomy"></div>

pLLM separates **algorithm declarations** from **engine scheduling** (for why it does not use hooks, see [Core Concepts](/pllm/docs/explain/core-concepts/)): you write a `pllm.Workload` subclass whose declarations answer three questions—**what to read**, **what to store**, and **where results go**—and write standard PyTorch tensor math inside its methods. When building the engine, pLLM reads these declarations once, automatically schedules when and on which CUDA stream each method runs, manages per-request state memory, and captures the computation into CUDA graphs.

Start with a minimal runnable workload: it stores no state, reads the residual stream after layer 12, computes the L2 norm of each request's current token, and exports it to the CPU.

```python
from dataclasses import dataclass, field

import torch
from torch import nn

import pllm
from pllm import testing

CONFIG = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=8)


class StreamNorm(pllm.Workload):
    @pllm.export(name="stream_norm", max_rows=8, max_cols=1)  # Where results go: export to host on_export
    def norm(
        self,
        r: torch.Tensor = pllm.residual(12),                  # What to read: residual stream after layer 12
    ) -> torch.Tensor:
        return r.float().norm(dim=-1, keepdim=True)           # [rows, 1]: norm along the feature dim (dim=-1)


print(pllm.explain(StreamNorm(), CONFIG))
```

There are no registration handles or runtime callbacks in this code. If the algorithm also needs each request to maintain its own trainable state (for example, online distillation in ESamp, where each request predicts deep activations from shallow activations as it generates and uses that shallow prediction to guide sampling), simply add a **what to store** declaration to the same class:

```python
@dataclass(frozen=True)
class Distill(pllm.Workload):
    beta: float = 0.5

    # What to store: each request owns an independent small network and AdamW optimizer state
    distiller = pllm.RequestState(
        init=pllm.FromModule(
            lambda m: nn.Sequential(nn.Linear(m.hidden_size, 4), nn.Linear(4, m.hidden_size))
        ),
        optimizer=pllm.AdamW(1e-3),
    )

    # What to read: default values declare input sources; where results go: @pllm.train minimizes the per-row loss
    @pllm.train(state="distiller")
    def loss(
        self,
        shallow: torch.Tensor = pllm.hidden(0),
        deep: torch.Tensor = pllm.hidden(-1),
        f: nn.Module = pllm.module("distiller"),
    ) -> torch.Tensor:
        return (f(shallow.float()) - deep.float()).square().mean(-1)  # [rows]

    # Method that reads configuration from the instance, referenced by the decorator below
    def strength(self) -> float:
        return self.beta

    # Where results go: guide sampling using the predicted hidden vector at strength beta
    @pllm.guide_sampling(beta=strength)
    def predict(
        self,
        shallow: torch.Tensor = pllm.hidden(0),
        f: nn.Module = pllm.module("distiller"),
    ) -> torch.Tensor:
        return f(shallow.float())  # [rows, hidden]
```

Below we break down the structure of a workload class, the three declaration categories, rules for method bodies, and common configuration and composition patterns. If this is your first time running an engine with a workload, you can also follow along with [Tutorial 00: Quickstart](/pllm/docs/tutorials/00_quickstart/).

---

## The Workload Class and Three Declaration Categories {#en-the-workload-class-and-three-declaration-categories}

A workload is a subclass of `pllm.Workload`. Whenever a workload carries hyperparameters or switches (such as `beta: float = 0.5` above), define it as `@dataclass(frozen=True)`.

Why freeze it (`frozen=True`)? Because pLLM reads the configuration on the workload instance once when building the engine and generates a static execution plan; mutating attributes on the Python instance after the engine starts will not alter the execution graph already compiled on the GPU. `make_llm` deep-copies the workload; mutating module globals or objects captured by methods afterward produces undefined behavior (as is always true under `torch.compile`). Using a frozen dataclass prevents accidental runtime mutation and lets you construct a zero-strength baseline with `Distill(beta=0.0)` or `dataclasses.replace(workload, beta=0.0)` for `pllm.testing.check` to verify that disabling the algorithm matches the base model strictly.

Every declaration in a workload falls into one of three categories:

| Question Answered | Syntax | Public Names | Detailed Guides |
| --- | --- | --- | --- |
| **What to read** (method data sources) | Default values on method parameters | Model activations & logits: `pllm.hidden`, `pllm.residual`, `pllm.final_hidden`, `pllm.layer_logits`, `pllm.attention`<br>State & metadata: `pllm.parameter`, `pllm.module`, `pllm.version`, `pllm.index`, `pllm.carry`<br>Tokens & sampling feedback: `pllm.input_tokens`, `pllm.sample_index`, `pllm.base_logits`, `pllm.sampled_tokens`<br>Outputs from other methods in the same step: `pllm.result`<br>In-layer site & projection operands: `pllm.depth`, `pllm.value`, `pllm.operand`<br>Host training batches & differentiable model: `pllm.batch`, `pllm.causal_model` | [Reading Model Internals](/pllm/docs/guides/reading/)<br>[State](/pllm/docs/guides/state/) |
| **What to store** (tensors held across steps or requests) | Class attributes, or `@property` with a return type annotation | State scopes: `pllm.RequestState` (one private copy per request), `pllm.InstanceState` (one shared copy across the engine), `pllm.PositionState`<br>Field specs: `pllm.Field`, `pllm.Dimensions`<br>Initializers `init=`: `pllm.Zeros` (default), `pllm.Computed`, `pllm.Pretrained`, `pllm.ModelCheckpoint`, `pllm.FromModule`, `pllm.Fed`, `pllm.Imported`<br>Optimizers `optimizer=`: `pllm.AdamW`, `pllm.SGD` | [State](/pllm/docs/guides/state/)<br>[Online Training](/pllm/docs/guides/training/) |
| **Where results go** (how return values or state take effect) | Method decorators, or model-edit class attributes | Observation & sampling: `@pllm.export`, `@pllm.add_logits`, `@pllm.guide_sampling`<br>State updates & training: `@pllm.write`, `@pllm.train`, `@pllm.differentiate`<br>In-model edits: `pllm.model.ActivationPatch`, `pllm.model.Adapter`, `@pllm.train_in_forward`<br>Forward structure rewrites: `@pllm.custom_residual`, `@pllm.custom_attention`, `@pllm.custom_forward` | [Where Results Go](/pllm/docs/guides/results/)<br>[Model Edits](/pllm/docs/guides/model-edits/)<br>[Forward Structure](/pllm/docs/guides/forward-structure/)<br>[Host Loops and Batch Training](/pllm/docs/guides/host-loops/) |

See the [API Reference](/pllm/docs/reference/api/) for full signatures and descriptions of all public interfaces. Three basic rules apply when writing these declarations:

1. **Every input parameter except `self` must have an input marker as its default value.** Python parameter names (like `shallow`, `deep`) are just local variable names inside the method body—pLLM never guesses which layer you want from a parameter name. The default value (such as `= pllm.hidden(0)`) is the declaration read by the planner, and the type annotation (`: torch.Tensor`) tells your IDE and type checker what type the method receives. (The only exceptions are the stream tensors `x, y` in `@pllm.custom_residual`, `q, kv` in `@pllm.custom_attention`, and `layers` in `@pllm.custom_forward`, which are passed as leading positional arguments while any trailing parameters still use input-marker defaults.)
2. **Methods that produce effects must carry an effect decorator.** A plain method that merely returns a tensor without `@pllm.export` / `@pllm.train` or another effect decorator gives pLLM no target (sampler, export queue, or optimizer), so it will not be scheduled on its own unless another method references it via `pllm.result("method_name")` or `rows="method_name"`, or it is explicitly decorated with `@pllm.on`. If the workload declares no effects at all, engine build fails with an error.
3. **At most one pLLM effect decorator of the same kind per method.** Do not stack `@pllm.export` and `@pllm.train` on a single method—if allowed, the outer decorator would overwrite the inner decorator's effect metadata and silently drop one effect. pLLM rejects stacked effect decorators during declaration checking; when you need two effects from the same computation, split them into two methods and reuse the output with `pllm.result("method_name")`.

---

## What Goes in a Method Body {#en-what-goes-in-a-method-body}

Once inputs and effects are declared, the method body is standard PyTorch tensor math (if you need a custom Triton kernel, call it via `torch.library.custom_op`; see [Custom Operators](/pllm/docs/guides/custom-ops/)). Because methods run under continuous batching and CUDA graphs, method bodies must obey four boundaries:

### 1. Dimension 0 is the request row: reduce along features (`dim=-1`), never across rows (`dim=0`) {#en-sec-1}

For activation tensors, request-state parameters, and sampled-row tensors passed into a method, dimension 0 (`[rows, ...]`) always indexes the **request rows in the current step** (one row per active request, or one row per token drawn by the sampler).

- **Why**: Under continuous batching, which requests run together in a step and which row index each request occupies are decided dynamically by the scheduler as requests arrive, finish, or get preempted. If you write `h - h.mean(dim=0)`, `softmax(dim=0)`, BatchNorm, or sort across rows, one request's result mixes in activations from unrelated concurrent requests—changing the batch size or row order changes its output.
- **What to do**: Always reduce along feature dimensions (e.g., `.mean(dim=-1, keepdim=True)`, `.norm(dim=-1)`), so each row's output depends only on that row's own inputs and its own request state. `pllm.testing.check` explicitly tests this via row permutation and neighbor-request perturbation (`request isolation`).

### 2. Inputs and state parameters are borrowed read-only views: do not mutate in-place {#en-sec-2}

Never perform in-place writes on method parameters, such as `h.mul_(2)`, `w.add_(1)`, or `torch.matmul(..., out=h)`.

- **Why**: The activation and state tensors passed to your method are read-only views borrowed from the framework. Other methods in the same step may read the same activations, and older pages in the state pool may still be read by concurrent tasks. Mutating them in-place corrupts shared memory and bypasses versioned state publishing.
- **What to do**: Use out-of-place tensor operations that return new tensors (e.g., `h * 2`, `w + 1`); if you genuinely need slice assignment on a local intermediate, `.clone()` it first. Any persistent state update must be published as a new version through `@pllm.write` or `@pllm.train` (see [State](/pllm/docs/guides/state/) and [Where Results Go](/pllm/docs/guides/results/)).

### 3. No GPU→CPU synchronization and no Python branches on tensor values {#en-sec-3}

Do not call `.item()`, `.tolist()`, `.cpu()`, or `print(tensor)` inside a method body, and do not write data-dependent Python control flow such as `if h.norm() > 1:` or `while not converged:`.

- **Why**: For two reasons. First, during engine build (and `pllm.explain`), pLLM traces every method with PyTorch `meta` tensors (carrying shape and dtype with no backing storage) to infer shapes and memory budgets; calling `.item()` or branching on a `meta` tensor immediately raises `RuntimeError: Tensor.item() cannot be called on meta tensors`. Second, on GPU at runtime, methods are captured into CUDA graphs and often run on a side CUDA stream overlapped with the model forward pass; CUDA graphs record a fixed operator sequence and cannot capture dynamic Python branches, while GPU→CPU syncs stall the host thread and break stream capture.
- **What to do**:
  - Express conditional logic with boolean masks and `torch.where(mask, a, b)`;
  - To inspect tensor values at runtime, export them asynchronously with `@pllm.export` to `on_export`;
  - **Allowed Python loops**: Bounded `for` loops whose trip count comes from a workload config field (such as `for _ in range(self.loops):`) are completely fine. Because `self.loops` is already a fixed Python integer at build time, tracing unrolls the loop into a fixed sequence of operators.

### 4. Watch tensor dtypes and return shapes {#en-sec-4}

- **Dtype**: Model activations (`pllm.hidden`, `pllm.residual`, `pllm.final_hidden`) are captured in the model's compute dtype (typically `bfloat16`), whereas `pllm.layer_logits` is `float32`. Cast with `.float()` before squaring, taking norms, or accumulating to avoid `bfloat16` underflow or precision loss.
- **Return shapes**:
  - Per-request `@pllm.train` methods must return a 1D `[rows]` loss tensor (one scalar per row)—do **not** call `.mean()` at the end to collapse it to a 0D scalar. Different rows in the same batch may belong to different request states (and padding or unselected rows in the capture table must be masked out first); pLLM reduces valid rows per state (`reduction="mean"` by default, or `"sum"`). By contrast, host batch-training methods reading `pllm.batch(...)` take one step over an external batch and return a single scalar loss (see [Online Training](/pllm/docs/guides/training/) and [Host Loops and Batch Training](/pllm/docs/guides/host-loops/)).
  - `@pllm.add_logits` must return `[rows, vocab_size]`; `@pllm.guide_sampling` must return `[rows, hidden_size]` (projected into vocabulary space by the model's LM head).

---

## Reading Configuration from the Instance {#en-reading-configuration-from-the-instance}

When writing reusable workloads, hyperparameters such as layer index, guidance strength `beta`, or low-rank `rank` are usually dataclass fields. Passing those fields into decorators, input markers, or state declarations has two common pitfalls.

### Why you cannot write `beta=beta` directly in a decorator {#en-why-you-cannot-write-betabeta-directly-in-a-decorator}

Look at this broken pattern:

```text
@dataclass(frozen=True)
class BadDistill(pllm.Workload):
    beta: float = 0.5
    ...
    @pllm.guide_sampling(beta=beta)   # Here beta evaluates to the class-level default 0.5 at definition time
    def predict(self, ...): ...
```

- **Why**: Python evaluates `@pllm.guide_sampling(beta=beta)` while executing the `class` body, before any `self` instance exists. The name `beta` binds the class attribute's default `0.5`. Later, when you instantiate `BadDistill(beta=0.0)` and pass it to `testing.check(..., neutral=BadDistill(beta=0.0))`, the decorator still holds `0.5`, and the neutral check fails with: `[neutral] guide_sampling still declares a sampler stage; set its beta to zero.`.
- **What to do**: Whenever a decorator argument or input marker argument (such as `@pllm.guide_sampling(beta=...)`, `pllm.hidden(layer)`, `pllm.layer_logits(layers)`, `@pllm.export(every=...)`, or `@pllm.train(steps=...)`) depends on an instance field, pass a **method taking the workload instance and returning that config value** (or `lambda self: self.beta`). pLLM calls it once when binding the concrete instance. Writing a typed method (e.g., `def strength(self) -> float:`) also gives you IDE go-to-definition and static type checking.

### Why a `@property` state must have a return type annotation {#en-why-a-property-state-must-have-a-return-type-annotation}

When a state's shape depends on an instance field (such as `self.rank`), a class attribute cannot see `self`, so you write the state as a `@property`. In that case, you **must** add a return type annotation to the `@property` (such as `-> pllm.RequestState` or `-> pllm.InstanceState`; for model-edit properties, `-> pllm.model.Adapter`, `-> pllm.model.ActivationPatch`, or `-> pllm.PositionState`). If the class defines any ordinary helper `@property` methods too, annotate their return types as well (e.g., `-> int`).

- **Why**: When analyzing a workload class, pLLM must identify which members are state or model-edit declarations and snapshot method/function dependencies without blindly executing arbitrary user getters, and only then evaluate state declarations in order. Inspecting `@property` return type annotations (including stringified annotations under `from __future__ import annotations`) lets the framework recognize declaration properties without running unknown property code.

The example below combines passing `self.layer` to an input marker via a method and declaring a `self.rank`-dependent state via a typed `@property`:

```python
@dataclass(frozen=True)
class ConfigurableProbe(pllm.Workload):
    layer: int = 12
    rank: int = 4

    @property
    def head(self) -> pllm.RequestState:
        return pllm.RequestState(
            init=pllm.FromModule(lambda m: nn.Linear(m.hidden_size, self.rank))
        )

    def source_layer(self) -> int:
        return self.layer

    @pllm.export(name="probe", max_rows=8, max_cols=4)
    def score(
        self,
        r: torch.Tensor = pllm.residual(source_layer),
        head: nn.Module = pllm.module("head"),
    ) -> torch.Tensor:
        return head(r.float())


print(testing.check(ConfigurableProbe(layer=6, rank=4), CONFIG).passed)
```

---

## When Methods Run {#en-when-methods-run}

You rarely need to specify when a method runs manually, because every effect decorator carries a default phase (`pllm.Phase`) matching its semantics:

| Effect | Default Phase | Window Within a Step |
| --- | --- | --- |
| `@pllm.train_in_forward`, `@pllm.custom_residual`, `@pllm.custom_attention` | `Phase.IN_FORWARD` | During the model forward pass |
| `@pllm.export` | `Phase.AFTER_FORWARD` | After the model forward pass and state publish, before sampling |
| `@pllm.add_logits`, `@pllm.guide_sampling` | `Phase.BEFORE_SAMPLE` | After the model forward pass, before the sampler draws tokens |
| `@pllm.write`, `@pllm.train` | `Phase.AUTOMATIC` | Placed by the planner based on what the method reads: runs as soon as its inputs are ready; per-request training runs after sampling on a side stream, overlapping with the next step's forward pass |

### `pllm.Phase` is a window within a step, not a runtime hook {#en-pllmphase-is-a-window-within-a-step-not-a-runtime-hook}

`pllm.Phase` has six enum values: `BEFORE_FORWARD`, `IN_FORWARD`, `AFTER_FORWARD`, `BEFORE_SAMPLE`, `AFTER_SAMPLE`, `AUTOMATIC`.

They represent **windows** on the engine's single-step timeline, not callbacks invoked at runtime. At engine build time, pLLM places each method into its assigned window and topologically orders methods according to their data dependencies (for example, method B reading method A via `pllm.result("A")`, or reading a published version of state).

This is why **a value can only be read in the phase that produces it or a later phase**:
- Current-step `pllm.hidden`, `pllm.residual`, and `pllm.final_hidden` exist only after the model forward pass and cannot be read in `BEFORE_FORWARD`;
- `pllm.base_logits()` (the unmodified base-model logits) and `pllm.sampled_tokens()` (tokens drawn in the current step) exist only after sampling and can only be read in `@pllm.train` or methods explicitly marked `@pllm.on(pllm.Phase.AFTER_SAMPLE)`. If you need model logits before sampling (for example inside `@pllm.add_logits`), use `pllm.layer_logits((-1,))`.

When you need to move a method to a non-default window, or schedule an undecorated helper method in a specific window (so other methods can read it via `pllm.result`), decorate it with `@pllm.on(phase)` (taking either a `pllm.Phase` value or a config method returning a `pllm.Phase`):

```python
class RunningMean(pllm.Workload):
    stats = pllm.RequestState({"mean": pllm.Field(lambda width, _: (width,))})

    @pllm.export(name="mean_norm", max_rows=8, max_cols=1)
    def report(self, mean: torch.Tensor = pllm.parameter("stats", "mean")) -> torch.Tensor:
        return mean.norm(dim=-1, keepdim=True)

    @pllm.on(pllm.Phase.AFTER_SAMPLE)
    @pllm.write(state="stats", fields=("mean",))
    def update(
        self,
        h: torch.Tensor = pllm.final_hidden(),
        mean: torch.Tensor = pllm.parameter("stats", "mean"),
    ) -> tuple[torch.Tensor]:
        return (0.9 * mean + 0.1 * h.float(),)


print(testing.check(RunningMean(), CONFIG).passed)
```

---

## Selectors: Acting on a Subset of Rows {#en-selectors-acting-on-a-subset-of-rows}

Returning to the `Distill` workload from the introduction: on decode step 0 of a newly admitted request, its `distiller` network has not trained even once (`version == 0`), so guiding sampling with its random initial weights only injects noise. We want to enable sampling guidance only for requests whose state has been updated at least once.

Because method bodies execute inside CUDA graphs, you cannot write `if version > 0:`, nor can you dynamically drop rows from a tensor (which would change tensor shapes). pLLM solves this with **row selectors (`rows="method_name"`)**: define a method that reads marked inputs and returns a 1D boolean mask `bool [rows]`, and pass its name to the effect decorator's `rows=` argument.

```python
@dataclass(frozen=True)
class GatedDistill(pllm.Workload):
    beta: float = 0.5

    distiller = pllm.RequestState(
        init=pllm.FromModule(
            lambda m: nn.Sequential(nn.Linear(m.hidden_size, 4), nn.Linear(4, m.hidden_size))
        ),
        optimizer=pllm.AdamW(1e-3),
    )

    # Selector: returns a bool [rows] mask selecting only requests updated at least once
    def trained(self, versions: torch.Tensor = pllm.version("distiller")) -> torch.Tensor:
        return versions > 0

    @pllm.train(state="distiller")
    def loss(
        self,
        shallow: torch.Tensor = pllm.hidden(0),
        deep: torch.Tensor = pllm.hidden(-1),
        f: nn.Module = pllm.module("distiller"),
    ) -> torch.Tensor:
        return (f(shallow.float()) - deep.float()).square().mean(-1)

    def strength(self) -> float:
        return self.beta

    @pllm.guide_sampling(beta=strength, rows="trained")
    def predict(
        self,
        shallow: torch.Tensor = pllm.hidden(0),
        f: nn.Module = pllm.module("distiller"),
    ) -> torch.Tensor:
        return f(shallow.float())
```

The selector method (`trained`) is compiled as a regular tensor task in the graph, keeping the mask entirely on the GPU: in `@pllm.guide_sampling` and `@pllm.add_logits`, rows whose mask is `False` receive zero weight and keep their original logits untouched with no GPU→CPU sync.

### Where `rows=` is supported and where it is rejected {#en-where-rows-is-supported-and-where-it-is-rejected}

Not all decorators accept `rows=`. The table below summarizes support and rationale by effect:

| Effect | Supports `rows=`? | Behavior and Rationale |
| --- | --- | --- |
| `@pllm.add_logits(rows="...")`<br>`@pllm.guide_sampling(beta=..., rows="...")` | Supported | Unselected rows keep the sampler's original logits unchanged. |
| Per-request `@pllm.train` paired with in-model effects (e.g., `Adapter`, `ActivationPatch`, without `@pllm.guide_sampling`) or paired with `@pllm.add_logits` | Supports `rows=` and `update_rows=` | The method still evaluates on all rows; `rows=` masks which rows contribute to the reduced loss, and `update_rows=` determines which states run an optimizer step and publish a new version. |
| **Standalone** per-request `@pllm.train` (no in-model effects and no `@pllm.add_logits`), or `@pllm.train` paired with **`@pllm.guide_sampling`** | **Unsupported** (rejected at declaration time) | Both paths take the unconditional post-sample update path (running on a side stream overlapped with the next model forward): every step unconditionally advances the optimizer and publishes a new version for each active request's state, with no per-row conditional skipping. |
| Host batch training reading `pllm.batch(...)` | **Unsupported** | Host batch training performs one update over the whole batch; to skip samples, set `weights` to `0` for those rows in `WeightedRowBatch` or `WeightedCEBatch`. |
| `@pllm.write` | **Unsupported** | `@pllm.write` writes all rows of the state on every step; to keep selected rows unchanged, read the old value and return `torch.where(mask, new_val, old_val)`. |

> [!TIP]
> **How do you suppress certain rows in standalone training or training paired with `@pllm.guide_sampling`?**
> 1. If your goal is to prevent untrained state from affecting generation, attach `rows="trained"` to `@pllm.guide_sampling` as in `GatedDistill` above;
> 2. If your goal is to produce zero gradient for certain rows on a given step, set `@pllm.train(state=..., reduction="sum")` and multiply those rows' losses by `0` inside `loss` (do not use the default `reduction="mean"`, because zeroed rows still count toward the denominator and dilute the gradient of other rows sharing the same state).

---

## Composing Multiple Workloads into One {#en-composing-multiple-workloads-into-one}

If you already have an online distillation workload `GatedDistill` and an activation-monitoring workload `StreamNorm` and want to run both in the same engine, you do not need to copy-paste methods into a single class—simply declare them as fields on an outer dataclass workload:

```python
@dataclass(frozen=True)
class Combined(pllm.Workload):
    distill: GatedDistill = field(default_factory=GatedDistill)
    monitor: StreamNorm = field(default_factory=StreamNorm)
```

Each field acts as an isolated component: states, methods, and export names are prefixed with the field name (shown in `explain` as `distill.loss`, `monitor.norm`), components cannot read each other's state, and genuinely conflicting combinations (such as two components both guiding sampling) are rejected at build time with both sides named. For why composition is designed this way and which combinations are rejected, see [Composing Workloads](/pllm/docs/guides/composition/).

---

## From Code to GPU: The Development Loop {#en-from-code-to-gpu-the-development-loop}

The fastest workflow is not to spin up the full GPU engine on every edit, but to spend a few seconds inspecting the execution plan and running contract checks on CPU first before moving to GPU:

1. **Author the workload**: Define it in a script, notebook cell, or `.py` module. When using the multiprocessing engine (`ModelConfig(multiprocessing=True)`), workload fields must be picklable (no locks, open file handles, etc.) so worker subprocesses can reconstruct and plan the workload independently (see [Production Deployment](/pllm/docs/howto/production/)).
2. **Inspect costs on CPU (`pllm.explain`)**: Reads only the model's `config.json` metadata without loading weights or allocating GPU memory, and prints whether the model stays in CUDA graph, when and on which stream each method runs, how many state bytes are reserved, and whether any method contains a GPU→CPU sync. See [Inspecting Costs: pllm.explain](/pllm/docs/guides/explain/).
3. **Check contracts on CPU (`pllm.testing.check`)**: Runs a fixed three-request schedule through the real planner and executor on CPU to verify row permutation invariance, request isolation, mixed prefill/decode, chunked prefill, preemption recomputation, version visibility (`publish_lag`), component isolation, and zero-strength equivalence (`neutral=`). See [Checking Correctness](/pllm/docs/guides/checking/).
4. **Run on GPU (`pllm.make_llm`)**: Load the model, generate outputs, and inspect runtime metrics with `llm.diagnostics()`.
5. **Run full GPU contract checks (`pllm.testing.check_on_gpu`)**: Verifies zero synchronization across consecutive decode steps, KV-eviction recomputation, and bitwise equivalence of `neutral=` against native vLLM (in-model edits such as `ActivationPatch`, `Adapter`, and `@pllm.custom_residual` are marked `SKIP` on CPU and require this step for GPU verification).

Run steps 2 and 3 on CPU for `GatedDistill` and `Combined`:

```python
print(pllm.explain(GatedDistill(), CONFIG))
print(testing.check(GatedDistill(), CONFIG, neutral=GatedDistill(beta=0.0)))
print(testing.check(Combined(), CONFIG, neutral=Combined(distill=GatedDistill(beta=0.0))).passed)
```

Once CPU checks pass, run steps 4 and 5 on GPU:

```python
from vllm import SamplingParams

GPU_CONFIG = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=8, gpu_memory_utilization=0.3)

with pllm.make_llm(GPU_CONFIG, workload=GatedDistill()) as llm:
    outputs = llm.generate(["Explain gradient descent:"], SamplingParams(n=4, max_tokens=32, seed=0))
    report = llm.diagnostics()
print(repr(report.eager_reason), report.update_count, report.trained_rows)

print(testing.check_on_gpu(GatedDistill(), GPU_CONFIG, neutral=GatedDistill(beta=0.0)))
```

---

## Common Errors {#en-common-errors}

When patterns on this page are rejected, exact error messages, causes, and fixes are listed in the [Method inputs and declarations](/pllm/docs/reference/errors/#en-method-inputs-and-declarations), [Method bodies](/pllm/docs/reference/errors/#en-method-bodies), and [Contract check errors](/pllm/docs/reference/errors/#en-contract-check-errors) sections of the error catalog.

---

## Self-Check {#en-self-check}

1. Why does `hidden - hidden.mean(dim=0, keepdim=True)` pass Python syntax checks yet fail `pllm.testing.check` with `mixes rows`?
<details markdown="1">
<summary>Answer</summary>

Dimension 0 indexes the distinct request rows packed together in the current step (`[rows, hidden]`). Taking a mean over `dim=0` mixes activations across unrelated requests in the same batch, making one request's output depend on whichever other requests happen to be batched alongside it. Reduce along the feature dimension instead: `hidden - hidden.mean(dim=-1, keepdim=True)`.

</details>

2. In a `@dataclass(frozen=True)` workload, why does `@pllm.guide_sampling(beta=beta)` break `neutral=MyWorkload(beta=0.0)`, and how do you fix it?
<details markdown="1">
<summary>Answer</summary>

When the `class` body executes, no `self` instance exists yet, so `beta=beta` captures the class attribute's default value (e.g., `0.5`) inside the decorator; constructing `MyWorkload(beta=0.0)` later does not change it. Define a method `def strength(self) -> float: return self.beta` and write `@pllm.guide_sampling(beta=strength)` so pLLM reads `self.beta` when binding the instance.

</details>

3. To save a new tensor into each request's own state on every decode step, can a plain method read `w: torch.Tensor = pllm.parameter("state", "w")` and call `w.copy_(new_val)`? Why or why not?
<details markdown="1">
<summary>Answer</summary>

No. `pllm.parameter` returns a read-only view, and the same state version may be read concurrently by other methods in the same step. Calling `w.copy_()` in-place breaks version isolation and triggers a `mutates borrowed input` failure in `testing.check`. Decorate the method with `@pllm.write(state="state", fields=("w",))` and return `(new_val,)` so the framework atomically publishes the next version of the state.

</details>

---

## Next Steps {#en-next-steps}

- Learn which activations and logits you can read at every layer → [Reading Model Internals](/pllm/docs/guides/reading/)
- Declare per-request state, instance-wide state, initializers, and `key_fields` sharing → [State](/pllm/docs/guides/state/)
- Export observations, bias logits, guide sampling, and write back state → [Where Results Go](/pllm/docs/guides/results/)
- Train state during decoding and control gradient edges and `publish_lag` → [Online Training](/pllm/docs/guides/training/)
- Modify layer activations or linear projection weights directly from state → [Model Edits](/pllm/docs/guides/model-edits/)
- Rewrite residual connections, attention reads, or looped layer depth → [Forward Structure](/pllm/docs/guides/forward-structure/)

</div>

<div class="zine-lang-panel is-hidden" data-lang-panel="zh" lang="zh-CN" markdown="1">

<div id="zh-sec-1"></div>

pLLM 把**算法声明**与**引擎调度**拆开（为什么不用 hook，见[核心概念](/pllm/docs/explain/core-concepts/)）：你写一个 `pllm.Workload` 子类，用声明回答三个问题——**读什么**、**存什么**、**结果去哪**——再在方法体里写普通的 PyTorch 张量数学。pLLM 在构建引擎时只读一次这些声明，自动规划每个方法在什么时机、哪条 CUDA stream 上运行，替你管理每个请求的状态显存并将计算捕获进 CUDA graph。

先看一个最小的可运行负载：不存任何状态，只读取模型第 12 层之后的残差流，算出每个请求当前 token 的 L2 范数并导出到 CPU。

```python
from dataclasses import dataclass, field

import torch
from torch import nn

import pllm
from pllm import testing

CONFIG = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=8)


class StreamNorm(pllm.Workload):
    @pllm.export(name="stream_norm", max_rows=8, max_cols=1)  # 结果去哪：导出到宿主机的 on_export
    def norm(
        self,
        r: torch.Tensor = pllm.residual(12),                  # 读什么：第 12 层之后的残差流
    ) -> torch.Tensor:
        return r.float().norm(dim=-1, keepdim=True)           # [rows, 1]：按特征维（dim=-1）求范数


print(pllm.explain(StreamNorm(), CONFIG))
```

这段代码里没有注册句柄，也没有运行时回调。如果算法还需要每个请求独立维护一份可训练的状态（例如在线蒸馏 ESamp：每个请求边生成边用浅层激活预测深层激活，并用浅层预测引导采样），只需在同一个类里加上**存什么**的声明：

```python
@dataclass(frozen=True)
class Distill(pllm.Workload):
    beta: float = 0.5

    # 存什么：每个请求独立拥有一份小网络和 AdamW 优化器状态
    distiller = pllm.RequestState(
        init=pllm.FromModule(
            lambda m: nn.Sequential(nn.Linear(m.hidden_size, 4), nn.Linear(4, m.hidden_size))
        ),
        optimizer=pllm.AdamW(1e-3),
    )

    # 读什么：默认值声明输入来源；结果去哪：@pllm.train 最小化返回的逐行 loss
    @pllm.train(state="distiller")
    def loss(
        self,
        shallow: torch.Tensor = pllm.hidden(0),
        deep: torch.Tensor = pllm.hidden(-1),
        f: nn.Module = pllm.module("distiller"),
    ) -> torch.Tensor:
        return (f(shallow.float()) - deep.float()).square().mean(-1)  # [rows]

    # 从实例读取配置的方法，供装饰器引用
    def strength(self) -> float:
        return self.beta

    # 结果去哪：用预测的 hidden 向量按强度 beta 引导采样
    @pllm.guide_sampling(beta=strength)
    def predict(
        self,
        shallow: torch.Tensor = pllm.hidden(0),
        f: nn.Module = pllm.module("distiller"),
    ) -> torch.Tensor:
        return f(shallow.float())  # [rows, hidden]
```

下面逐一拆解负载类的结构、三类声明、方法体的编写规则以及常见的配置与组合方式。如果你是第一次运行带负载的引擎，也可以先配合 [教程 00：快速上手](/pllm/docs/tutorials/00_quickstart/) 边跑边看。

---

## 负载类与三类声明 {#zh-sec-2}

一个负载是 `pllm.Workload` 的子类。当负载带有超参数或开关（如上面的 `beta: float = 0.5`）时，把它写成 `@dataclass(frozen=True)`。

为什么要冻结（`frozen=True`）？因为 pLLM 在构建引擎时一次性读取负载实例上的配置并生成执行计划；引擎启动后再去修改 Python 实例的属性，不会改变已经在 GPU 上编译好的执行图。`make_llm` 会深拷贝负载；之后再修改模块全局变量或方法捕获的对象，结果未定义（在 `torch.compile` 下本来就是如此）。使用冻结的 dataclass 既能防止运行期误改属性，又能随时用 `Distill(beta=0.0)` 或 `dataclasses.replace(workload, beta=0.0)` 构造一个零强度的对照实例，交给 `pllm.testing.check` 检查“关闭算法时是否严格等价于原模型”。

负载里的所有声明都归为三类：

| 声明回答的问题 | 写法 | 可用的公开名字 | 详细指南 |
| --- | --- | --- | --- |
| **读什么**（方法的数据来源） | 方法参数的默认值 | 模型激活与 logits：`pllm.hidden`、`pllm.residual`、`pllm.final_hidden`、`pllm.layer_logits`、`pllm.attention`<br>状态与元数据：`pllm.parameter`、`pllm.module`、`pllm.version`、`pllm.index`、`pllm.carry`<br>Token 与采样反馈：`pllm.input_tokens`、`pllm.sample_index`、`pllm.base_logits`、`pllm.sampled_tokens`<br>同一步其他方法的结果：`pllm.result`<br>模型层内位置与投影操作数：`pllm.depth`、`pllm.value`、`pllm.operand`<br>宿主训练批次与可微模型：`pllm.batch`、`pllm.causal_model` | [读取模型内部](/pllm/docs/guides/reading/)<br>[状态](/pllm/docs/guides/state/) |
| **存什么**（跨步或跨请求持有的张量） | 类属性，或标注了返回类型的 `@property` | 状态作用域：`pllm.RequestState`（每个请求一份独立副本）、`pllm.InstanceState`（全引擎共享一份副本）、`pllm.PositionState`<br>字段描述：`pllm.Field`、`pllm.Dimensions`<br>初值 `init=`：`pllm.Zeros`（默认）、`pllm.Computed`、`pllm.Pretrained`、`pllm.ModelCheckpoint`、`pllm.FromModule`、`pllm.Fed`、`pllm.Imported`<br>优化器 `optimizer=`：`pllm.AdamW`、`pllm.SGD` | [状态](/pllm/docs/guides/state/)<br>[在线训练](/pllm/docs/guides/training/) |
| **结果去哪**（返回值或状态如何生效） | 方法装饰器，或模型改动类属性 | 观测与采样：`@pllm.export`、`@pllm.add_logits`、`@pllm.guide_sampling`<br>状态更新与训练：`@pllm.write`、`@pllm.train`、`@pllm.differentiate`<br>模型层内改动：`pllm.model.ActivationPatch`、`pllm.model.Adapter`、`@pllm.train_in_forward`<br>前向结构改写：`@pllm.custom_residual`、`@pllm.custom_attention`、`@pllm.custom_forward` | [结果去哪](/pllm/docs/guides/results/)<br>[改模型](/pllm/docs/guides/model-edits/)<br>[改 forward 结构](/pllm/docs/guides/forward-structure/)<br>[宿主循环与批训练](/pllm/docs/guides/host-loops/) |

全部公开接口的签名与说明见 [API 参考](/pllm/docs/reference/api/)。写这三类声明时有三条基本规则：

1. **除 `self` 外的每个输入参数都必须写标记默认值。** Python 参数名（如 `shallow`、`deep`）只是方法体内部使用的局部变量名，pLLM 不会根据参数名猜测你想读哪一层；默认值（如 `= pllm.hidden(0)`）才是给规划器看的声明，类型注解（`: torch.Tensor`）则告诉 IDE 和类型检查器方法体拿到的值是什么类型。（唯一的例外是 `@pllm.custom_residual` 的流张量 `x, y`、`@pllm.custom_attention` 的 `q, kv` 以及 `@pllm.custom_forward` 的 `layers`，它们作为打头的位置参数传入，后面的参数依然使用标记默认值。）
2. **产出结果的方法必须挂效果装饰器。** 一个普通方法只返回张量、不挂 `@pllm.export` / `@pllm.train` 等装饰器时，pLLM 不知道返回值该送往采样器、导出队列还是优化器，因此不会单独调度它（除非它被另一个方法通过 `pllm.result("方法名")`、`rows="方法名"` 引用，或显式挂了 `@pllm.on`）。若整个负载没有任何效果声明，构建时会直接报错提醒。
3. **同一个方法上同一种类的 pLLM 装饰器最多挂一个。** 不要在一个方法上同时叠放 `@pllm.export` 和 `@pllm.train`——如果允许叠放，外层装饰器会覆盖内层装饰器的效果属性，导致其中一个效果静默丢失。因此 pLLM 在声明检查时直接拒绝叠放效果；需要把同一份计算结果用于两个效果时，拆成两个方法并用 `pllm.result("方法名")` 复用结果即可。

---

## 方法体里写什么 {#zh-sec-3}

声明好输入和效果后，方法体内部就是普通的 PyTorch 张量运算（如果需要接入自写的 Triton 算子，可通过 `torch.library.custom_op` 调入，见[自写算子](/pllm/docs/guides/custom-ops/)）。由于方法在连续批处理和 CUDA graph 下运行，方法体需要遵守四条边界：

### 1. 第 0 维是请求行：只在特征维（`dim=-1`）上归约，不要跨行（`dim=0`）计算 {#zh-sec-4}

方法拿到的激活张量、请求状态参数和采样行张量，第 0 维（`[rows, ...]`）都是**当前步的请求行**（每个运行中的请求一行，或采样器每抽取一个 token 一行）。

- **原因**：在连续批处理下，每一步有哪些请求同时在跑、哪个请求排在第几行，完全由引擎调度器根据请求的到达、完成和抢占动态决定。如果你在方法里写了 `h - h.mean(dim=0)`、`softmax(dim=0)`、BatchNorm 或按行排序，一个请求的计算结果就会混入同一批次里其他无关请求的激活值，批次大小或排列顺序一变，输出就会跟着变。
- **做法**：归约一律沿着特征维度写（如 `.mean(dim=-1, keepdim=True)`、`.norm(dim=-1)`），让每一行的输出只依赖该行自己的输入和自己的请求状态。`pllm.testing.check` 会通过打乱行顺序（`row permutation`）和扰动邻居请求（`request isolation`）专门检查这一条。

### 2. 输入与状态参数是借用的只读视图：不要原地修改 {#zh-sec-5}

不要对方法参数做原地（in-place）写入，比如 `h.mul_(2)`、`w.add_(1)` 或 `torch.matmul(..., out=h)`。

- **原因**：方法收到的激活张量和状态张量是框架借给当前方法的只读视图，同一步里的其他方法可能也要读同一块激活，而状态池里的旧版本页面还被其他正在读取该版本的任务引用。原地修改会直接篡改共享内存，绕过状态的版本发布机制。
- **做法**：方法体里只写返回新张量的非原地运算（如 `h * 2`、`w + 1`）；如果确实需要在局部张量上做切片赋值，先 `.clone()` 一份副本。对状态的任何持久化修改都通过 `@pllm.write` 或 `@pllm.train` 发布成新版本（见[状态](/pllm/docs/guides/state/)与[结果去哪](/pllm/docs/guides/results/)）。

### 3. 不要做 GPU→CPU 同步，也不要根据张量的值写 Python 条件分支 {#zh-sec-6}

方法体里不能出现 `.item()`、`.tolist()`、`.cpu()`、`print(tensor)`，也不能写 `if h.norm() > 1:` 或 `while not converged:` 这类依赖张量值的 Python 分支与循环。

- **原因**：有两个层面的原因。第一，在构建引擎（以及运行 `pllm.explain`）时，pLLM 会先用 PyTorch 的 `meta` 张量（只有形状和 dtype、没有实际数据存储）追踪一遍方法体来推导形状与显存预算；在 `meta` 张量上调用 `.item()` 或做布尔判断会立刻触发 `RuntimeError: Tensor.item() cannot be called on meta tensors`。第二，在 GPU 运行期，方法会被捕获进 CUDA graph，并且常常放在旁路 CUDA stream 上与模型前向重叠执行；CUDA graph 只能记录固定的算子序列，无法录制 Python 的动态分支，而 GPU→CPU 同步则会阻塞主机线程并破坏流捕获。
- **做法**：
  - 条件逻辑用布尔掩码和 `torch.where(mask, a, b)` 表达；
  - 想在运行期查看张量数值，用 `@pllm.export` 异步导出到 CPU 的 `on_export` 回调；
  - **允许的 Python 循环**：以负载配置字段为上界的有界 `for` 循环（例如 `for _ in range(self.loops):`）完全可以使用。因为 `self.loops` 在构建时已经是固定的 Python 整数，追踪时循环会直接展开成固定步数的算子序列。

### 4. 注意张量精度与返回形状 {#zh-sec-7}

- **精度（dtype）**：模型内部的激活输入（`pllm.hidden`、`pllm.residual`、`pllm.final_hidden`）默认以模型的计算精度（通常为 `bfloat16`）存储和捕获，而 `pllm.layer_logits` 为 `float32`。做平方误差、范数或累加时，先调用 `.float()` 转成 `float32`，可以避免 `bfloat16` 下溢或精度损失。
- **返回形状**：
  - 按请求行训练的 `@pllm.train` 方法必须返回一维的 `[rows]` loss 张量（每行一个标量），**不要**在方法末尾自己做 `.mean()` 归约成 0 维标量。原因是同一个批次里的不同行可能属于不同请求的状态（且捕获表中的填充行或未选中行需要由框架先用掩码剔除），pLLM 会按状态对有效行做归约（默认 `reduction="mean"`，可选 `"sum"`）。相反，读取 `pllm.batch(...)` 的宿主批训练方法针对整个外部批次做一步更新，由方法自己返回单个标量 loss（见[在线训练](/pllm/docs/guides/training/)与[宿主循环与批训练](/pllm/docs/guides/host-loops/)）。
  - `@pllm.add_logits` 必须返回 `[rows, vocab_size]`；`@pllm.guide_sampling` 必须返回 `[rows, hidden_size]`（框架再用模型自己的 LM head 把它投影到词表空间）。

---

## 从实例读取配置 {#zh-sec-8}

写可复用的负载时，我们常把层号、引导强度 `beta`、低秩维度 `rank` 等设为 dataclass 字段。把这些字段传给装饰器、输入标记或状态声明时，有两个高频易错点。

### 为什么不能在装饰器里直接写 `beta=beta`？ {#zh-sec-9}

观察下面这种错误写法：

```text
@dataclass(frozen=True)
class BadDistill(pllm.Workload):
    beta: float = 0.5
    ...
    @pllm.guide_sampling(beta=beta)   # 这里的 beta 是类定义时的默认值 0.5
    def predict(self, ...): ...
```

- **原因**：Python 在执行 `class` 定义体时就会求值 `@pllm.guide_sampling(beta=beta)` 的参数，此时还没有任何 `self` 实例存在。名字 `beta` 拿到的是类属性默认值 `0.5`。当你之后实例化 `BadDistill(beta=0.0)` 传给 `testing.check(..., neutral=BadDistill(beta=0.0))` 时，装饰器里绑定的仍然是 `0.5`，导致零强度检查报错：`[neutral] guide_sampling still declares a sampler stage; set its beta to zero.`。
- **做法**：任何装饰器参数或输入标记参数（如 `@pllm.guide_sampling(beta=...)`、`pllm.hidden(layer)`、`pllm.layer_logits(layers)`、`@pllm.export(every=...)`、`@pllm.train(steps=...)`）只要依赖实例字段，就传一个**接收负载实例并返回该配置值的方法**（或者 `lambda self: self.beta`）。pLLM 会在绑定具体实例时调用一次这个方法。写成带返回类型注解的方法（如 `def strength(self) -> float:`）还能让 IDE 支持跳转定义，并让类型检查器校验返回类型。

### 为什么状态写成 `@property` 时必须标注返回类型？ {#zh-sec-10}

当状态的形状依赖实例字段（如 `self.rank`）时，类属性拿不到 `self`，需要把状态写成 `@property`。此时**必须**给 `@property` 加上返回类型注解（如 `-> pllm.RequestState` 或 `-> pllm.InstanceState`；模型改动属性则标 `-> pllm.model.Adapter`、`-> pllm.model.ActivationPatch` 或 `-> pllm.PositionState`）。如果类里还定义了其他普通辅助 `@property`，同样要标上它们各自的返回类型（如 `-> int`）。

- **原因**：pLLM 在分析负载类时，需要在不盲目执行任意用户 getter 代码的前提下，先识别出哪些成员是状态或模型改动声明，并对方法与函数做依赖快照，随后才按顺序求值状态声明。通过检查 `@property` 的返回类型注解（包含 `from __future__ import annotations` 下的字符串注解），框架无需运行未知的 property 代码就能准确认出声明属性。

下面的例子同时展示了“用方法向输入标记传递层号 `self.layer`”和“用带返回类型注解的 `@property` 声明依赖 `self.rank` 的状态”：

```python
@dataclass(frozen=True)
class ConfigurableProbe(pllm.Workload):
    layer: int = 12
    rank: int = 4

    @property
    def head(self) -> pllm.RequestState:
        return pllm.RequestState(
            init=pllm.FromModule(lambda m: nn.Linear(m.hidden_size, self.rank))
        )

    def source_layer(self) -> int:
        return self.layer

    @pllm.export(name="probe", max_rows=8, max_cols=4)
    def score(
        self,
        r: torch.Tensor = pllm.residual(source_layer),
        head: nn.Module = pllm.module("head"),
    ) -> torch.Tensor:
        return head(r.float())


print(testing.check(ConfigurableProbe(layer=6, rank=4), CONFIG).passed)
```

---

## 方法什么时候运行 {#zh-sec-11}

你通常不需要手动指定方法的运行时间，因为每个效果装饰器都有与它语义匹配的默认阶段（`pllm.Phase`）：

| 效果 | 默认阶段 | 在单步中的时间窗口 |
| --- | --- | --- |
| `@pllm.train_in_forward`、`@pllm.custom_residual`、`@pllm.custom_attention` | `Phase.IN_FORWARD` | 模型前向传播执行期间 |
| `@pllm.export` | `Phase.AFTER_FORWARD` | 模型前向完成、状态发布之后，采样之前 |
| `@pllm.add_logits`、`@pllm.guide_sampling` | `Phase.BEFORE_SAMPLE` | 模型前向完成之后、采样器抽取 token 之前 |
| `@pllm.write`、`@pllm.train` | `Phase.AUTOMATIC` | 由规划器按方法读的输入安排：输入一就绪就运行；按请求行训练在采样之后、于旁路 stream 上运行，与下一步前向重叠 |

### `pllm.Phase` 是步内的时间窗口，不是运行时钩子 {#zh-sec-12}

`pllm.Phase` 共有六个枚举值：`BEFORE_FORWARD`、`IN_FORWARD`、`AFTER_FORWARD`、`BEFORE_SAMPLE`、`AFTER_SAMPLE`、`AUTOMATIC`。

它们表示引擎单步时间轴上的**窗口**，而不是运行时触发的回调函数。构建引擎时，pLLM 把每个方法放进所属窗口，并根据方法之间的数据依赖（例如方法 B 通过 `pllm.result("A")` 读取方法 A 的输出，或读取某块状态的发布版本）自动排出拓扑顺序。

这也是为什么**一个值只能在产出它的阶段或更晚的阶段被读取**：
- 当前步的 `pllm.hidden`、`pllm.residual`、`pllm.final_hidden` 只有在模型前向完成后才存在，不能在 `BEFORE_FORWARD` 中读取；
- `pllm.base_logits()`（未经负载修改的模型原始 logits）和 `pllm.sampled_tokens()`（本步采样抽出的 token）只有在采样完成后才存在，只能在 `@pllm.train` 或显式标了 `@pllm.on(pllm.Phase.AFTER_SAMPLE)` 的方法中读取。若要在采样前读取模型 logits（例如在 `@pllm.add_logits` 里），应使用 `pllm.layer_logits((-1,))`。

当你需要把一个方法移到非默认窗口，或者让一个不带效果装饰器的方法在指定窗口运行（供其他方法通过 `pllm.result` 读取）时，给方法加上 `@pllm.on(phase)`（只接收一个 `pllm.Phase` 参数，或返回 `pllm.Phase` 的设置方法）：

```python
class RunningMean(pllm.Workload):
    stats = pllm.RequestState({"mean": pllm.Field(lambda width, _: (width,))})

    @pllm.export(name="mean_norm", max_rows=8, max_cols=1)
    def report(self, mean: torch.Tensor = pllm.parameter("stats", "mean")) -> torch.Tensor:
        return mean.norm(dim=-1, keepdim=True)

    @pllm.on(pllm.Phase.AFTER_SAMPLE)
    @pllm.write(state="stats", fields=("mean",))
    def update(
        self,
        h: torch.Tensor = pllm.final_hidden(),
        mean: torch.Tensor = pllm.parameter("stats", "mean"),
    ) -> tuple[torch.Tensor]:
        return (0.9 * mean + 0.1 * h.float(),)


print(testing.check(RunningMean(), CONFIG).passed)
```

---

## 选择器：只让一部分行生效 {#zh-sec-13}

回到开头的 `Distill` 负载：新请求刚进入解码的第 0 步时，它的 `distiller` 小网络还没有训练过任何一步（更新次数 `version == 0`），此时用随机初值去引导采样只会引入噪声。我们希望“只有当一个请求的状态至少更新过 1 次之后，才对它开启采样引导”。

因为方法体在 CUDA graph 里运行，不能写 `if version > 0:`，也不能动态删减张量的行（那会改变张量形状）。pLLM 的做法是**行选择器（`rows="方法名"`）**：定义一个读取标记输入、返回一维布尔掩码 `bool [rows]` 的方法，再把方法名传给效果装饰器的 `rows=`。

```python
@dataclass(frozen=True)
class GatedDistill(pllm.Workload):
    beta: float = 0.5

    distiller = pllm.RequestState(
        init=pllm.FromModule(
            lambda m: nn.Sequential(nn.Linear(m.hidden_size, 4), nn.Linear(4, m.hidden_size))
        ),
        optimizer=pllm.AdamW(1e-3),
    )

    # 选择器：返回 bool [rows] 掩码，只选中已经更新过至少一次的请求行
    def trained(self, versions: torch.Tensor = pllm.version("distiller")) -> torch.Tensor:
        return versions > 0

    @pllm.train(state="distiller")
    def loss(
        self,
        shallow: torch.Tensor = pllm.hidden(0),
        deep: torch.Tensor = pllm.hidden(-1),
        f: nn.Module = pllm.module("distiller"),
    ) -> torch.Tensor:
        return (f(shallow.float()) - deep.float()).square().mean(-1)

    def strength(self) -> float:
        return self.beta

    @pllm.guide_sampling(beta=strength, rows="trained")
    def predict(
        self,
        shallow: torch.Tensor = pllm.hidden(0),
        f: nn.Module = pllm.module("distiller"),
    ) -> torch.Tensor:
        return f(shallow.float())
```

选择器方法（`trained`）同样被编译成图里的一个普通张量任务，掩码全程留在 GPU 上：在 `@pllm.guide_sampling` 和 `@pllm.add_logits` 中，掩码为 `False` 的行权重为 0，采样器保持该行原有的 logits 不变，无需任何 GPU→CPU 同步。

### `rows=` 在哪里可用，在哪里会被拒绝？ {#zh-sec-14}

并非所有装饰器都支持 `rows=`。下表列出各效果对行选择器的支持情况与原因：

| 效果 | 是否支持 `rows=` | 行为与原因 |
| --- | --- | --- |
| `@pllm.add_logits(rows="...")`<br>`@pllm.guide_sampling(beta=..., rows="...")` | 支持 | 未选中的行在采样器端保持原始 logits 不变。 |
| 配合模型内效果（如 `Adapter`、`ActivationPatch`，且无 `@pllm.guide_sampling`）或配合 `@pllm.add_logits` 的按请求行 `@pllm.train` | 支持 `rows=` 与 `update_rows=` | 方法仍在所有行上求值，`rows=` 掩码决定哪些行的 loss 计入归约；`update_rows=` 决定哪些状态执行优化器更新并发布新版本。 |
| **独立**的按请求行 `@pllm.train`（无模型内效果且无 `@pllm.add_logits`），或**与 `@pllm.guide_sampling` 并存**的 `@pllm.train` | **不支持**（声明期报错） | 这两条路径走的是采样后的无条件更新路径（在旁路 stream 上与下一步模型前向重叠）：每一步都会无条件推进每个运行中请求的状态优化器并发布新版本，不实现按行掩码的条件跳过。 |
| 读取 `pllm.batch(...)` 的宿主批训练 | **不支持** | 宿主批训练在批次整体上做一步更新；需要跳过某些样本时，将 `WeightedRowBatch` 或 `WeightedCEBatch` 中对应行的 `weights` 设为 `0`。 |
| `@pllm.write` | **不支持** | `@pllm.write` 每步写入该状态的全部行；若想让部分行保持原值，可在方法内读入旧值并用 `torch.where(mask, new_val, old_val)` 返回。 |

> [!TIP]
> **在独立训练或配 `@pllm.guide_sampling` 的训练里，如何屏蔽某些行的影响？**
> 1. 如果目的是“不让未训练好的状态影响生成”，像上面的 `GatedDistill` 一样，把 `rows="trained"` 挂在 `@pllm.guide_sampling` 上；
> 2. 如果目的是“让某些行在当前步不产生梯度”，将训练装饰器设为 `@pllm.train(state=..., reduction="sum")`，并在 `loss` 方法内把这些行的 loss 乘以 `0`（注意不要用默认的 `reduction="mean"`，因为乘以 `0` 的行仍会被算进均值分母，稀释同一状态其余行的梯度）。

---

## 把多个负载组合成一个 {#zh-sec-15}

如果你已经写好了在线蒸馏负载 `GatedDistill` 和激活监控负载 `StreamNorm`，想在同一个引擎里同时运行它们，不需要把两个类的方法复制粘贴到一起——只需在一个外层 dataclass 负载里把它们声明成字段：

```python
@dataclass(frozen=True)
class Combined(pllm.Workload):
    distill: GatedDistill = field(default_factory=GatedDistill)
    monitor: StreamNorm = field(default_factory=StreamNorm)
```

每个字段是一个独立的组件：状态、方法和导出名带上字段名前缀（`explain` 里显示为 `distill.loss`、`monitor.norm`），组件之间不能互读状态，真正冲突的组合（例如两个组件都引导采样）在构建时被拒绝并点名双方。为什么这样设计、哪些组合被拒绝，见[组合负载](/pllm/docs/guides/composition/)。

---

## 从写完到上卡：开发循环 {#zh-sec-16}

写负载最快的方式不是每次都启动完整的 GPU 引擎，而是先在 CPU 上花几秒钟看完执行计划、跑完调度契约检查，最后再上 GPU：

1. **写负载**：脚本、notebook 单元格或 `.py` 模块里定义都可以；开启多进程引擎（`ModelConfig(multiprocessing=True)`）时，负载字段要能 pickle（不放锁、打开的文件等句柄），worker 子进程据此重建负载并自行规划（详见[上生产](/pllm/docs/howto/production/)）。
2. **在 CPU 上看成本（`pllm.explain`）**：只读取模型的 `config.json` 元数据，不加载模型权重、不分配 GPU 显存，直接打印模型是否保持 CUDA graph、每个方法在哪个时机和哪条 stream 上运行、状态预留多少字节、方法内是否有 GPU→CPU 同步。详见[看成本：pllm.explain](/pllm/docs/guides/explain/)。
3. **在 CPU 上查契约（`pllm.testing.check`）**：用真实的规划器和执行器在 CPU 上跑三个请求的固定调度，验证行重排、请求隔离、混合预填充与解码、分块预填充、抢占重算、版本可见性（`publish_lag`）、组件独立性以及零强度等价（`neutral=`）。详见[检查正确性](/pllm/docs/guides/checking/)。
4. **在 GPU 上运行（`pllm.make_llm`）**：加载模型生成，并通过 `llm.diagnostics()` 检查实际运行指标。
5. **在 GPU 上查完整契约（`pllm.testing.check_on_gpu`）**：验证连续解码步零同步、KV 驱逐重算以及 `neutral=` 与原生 vLLM 逐位一致（模型层内改动如 `ActivationPatch`、`Adapter`、`@pllm.custom_residual` 在 CPU 检查中会标记为 `SKIP`，需要这一步在 GPU 上验证）。

下面在 CPU 上对 `GatedDistill` 和组合后的 `Combined` 运行第 2、3 步：

```python
print(pllm.explain(GatedDistill(), CONFIG))
print(testing.check(GatedDistill(), CONFIG, neutral=GatedDistill(beta=0.0)))
print(testing.check(Combined(), CONFIG, neutral=Combined(distill=GatedDistill(beta=0.0))).passed)
```

CPU 检查通过后，在 GPU 上运行第 4、5 步：

```python
from vllm import SamplingParams

GPU_CONFIG = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=8, gpu_memory_utilization=0.3)

with pllm.make_llm(GPU_CONFIG, workload=GatedDistill()) as llm:
    outputs = llm.generate(["Explain gradient descent:"], SamplingParams(n=4, max_tokens=32, seed=0))
    report = llm.diagnostics()
print(repr(report.eager_reason), report.update_count, report.trained_rows)

print(testing.check_on_gpu(GatedDistill(), GPU_CONFIG, neutral=GatedDistill(beta=0.0)))
```

---

## 常见报错 {#zh-sec-17}

这一页的写法被拒绝时的报错原文、原因和改法，见报错目录的[声明与输入](/pllm/docs/reference/errors/#zh-sec-2)、[方法体](/pllm/docs/reference/errors/#zh-sec-4)、[testing.check 的 ContractError](/pllm/docs/reference/errors/#zh-sec-13)几节。

---

## 自测 {#zh-sec-18}

1. 为什么在方法体里写 `hidden - hidden.mean(dim=0, keepdim=True)` 能通过 Python 语法检查，却会在 `pllm.testing.check` 时报 `mixes rows`？
<details markdown="1">
<summary>答案</summary>

张量的第 0 维是当前步打包在一起的不同请求行（`[rows, hidden]`）。`dim=0` 求均值会把同一批次里不同请求的激活混在一起，导致一个请求的输出取决于此时有哪些其他请求恰好和它同批运行。应改为沿着特征维归约：`hidden - hidden.mean(dim=-1, keepdim=True)`。

</details>

2. 在 `@dataclass(frozen=True)` 负载类中，为什么 `@pllm.guide_sampling(beta=beta)` 会让 `neutral=MyWorkload(beta=0.0)` 失效，应该怎么改？
<details markdown="1">
<summary>答案</summary>

`class` 定义体执行时还没有实例 `self`，直接写 `beta=beta` 会把类属性的默认值（如 `0.5`）固定进装饰器里，之后构造 `MyWorkload(beta=0.0)` 也不会改变它。应在类里定义方法 `def strength(self) -> float: return self.beta`，并在装饰器里写 `@pllm.guide_sampling(beta=strength)`，让 pLLM 在绑定该实例时读取 `self.beta`。

</details>

3. 想在每步解码时把一个新张量存入每个请求自己的状态里，能不能在普通方法里直接对 `w: torch.Tensor = pllm.parameter("state", "w")` 调用 `w.copy_(new_val)`？为什么？
<details markdown="1">
<summary>答案</summary>

不能。`pllm.parameter` 拿到的是只读视图，同一状态版本可能正被同一步的其他方法读取，原地调用 `w.copy_()` 会破坏版本隔离并在 `testing.check` 中触发 `mutates borrowed input` 报错。应当使用 `@pllm.write(state="state", fields=("w",))` 装饰方法并返回 `(new_val,)`，由框架原子地发布为状态的下一个版本。

</details>

---

## 下一步 {#zh-sec-19}

- 了解模型每一层能读到哪些激活与 logits → [读取模型内部](/pllm/docs/guides/reading/)
- 声明每请求状态、全实例状态、初值来源与 `key_fields` 共享 → [状态](/pllm/docs/guides/state/)
- 导出观测、修改 logits、引导采样与写回状态 → [结果去哪](/pllm/docs/guides/results/)
- 在解码途中训练状态、控制梯度边与 `publish_lag` → [在线训练](/pllm/docs/guides/training/)
- 用状态直接改动层激活或线性投影权重 → [改模型](/pllm/docs/guides/model-edits/)
- 改写残差连接、注意力读法或循环层深度 → [改 forward 结构](/pllm/docs/guides/forward-structure/)

</div>

</div>

<div class="zine-doc-footer">
  <a class="zine-pill-link" data-i18n-zh="← 返回 pLLM 项目与文档中心" data-i18n-en="← Back to pLLM Project &amp; Docs" href="/pllm/#docs">← Back to pLLM Project &amp; Docs</a>
  <a class="zine-pill-link" href="https://github.com/LinesHogan/pLLM/blob/main/doc/guides/workload-anatomy.md" target="_blank" rel="noopener">GitHub &middot; doc/guides/workload-anatomy.md ↗</a>
</div>

<script defer src="/assets/js/bilingual-post.js"></script>
