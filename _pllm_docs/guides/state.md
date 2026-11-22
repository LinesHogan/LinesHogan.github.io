---
layout: page
title: "State"
description: "pLLM Documentation · User Guides"
permalink: /pllm/docs/guides/state/
nav: false
---

<link rel="stylesheet" href="{{ '/assets/css/pllm-hub.css' | relative_url | bust_file_cache }}">

<div class="zine-dispatch" data-title-zh="状态" data-title-en="State" data-desc-zh="pLLM 文档 · 用户指南" data-desc-en="pLLM Documentation · User Guides">
  <div class="zine-dispatch-links">
    <a class="zine-pill-link primary" data-i18n-zh="← 返回 pLLM 文档中心" data-i18n-en="← pLLM Docs Hub" href="/pllm/#docs">← pLLM Docs Hub</a>
    <a class="zine-pill-link" data-i18n-zh="文档导航地图" data-i18n-en="Docs Roadmap" href="/pllm/docs/">Docs Roadmap</a>
    <span class="zine-lang-badge" data-i18n-zh="用户指南" data-i18n-en="User Guides">User Guides</span>
  </div>
  <a class="zine-pill-link" data-i18n-zh="GitHub 源码 ↗" data-i18n-en="View on GitHub ↗" href="https://github.com/LinesHogan/pLLM/blob/main/doc/guides/state.md" target="_blank" rel="noopener">View on GitHub ↗</a>
</div>

<div id="markdown-content">

<div class="zine-lang-panel" data-lang-panel="en" lang="en" markdown="1">

<div id="en-state"></div>

Probe weights, per-request adapters, large tables shared across requests, and running statistics during generation: anything an inference-time algorithm keeps across steps or across requests must be declared as state in pLLM.

**State is the only mutable thing in pLLM.** Whenever a value needs to survive from step to step or across requests, declare it explicitly as state. Once declared, pLLM handles GPU memory allocation, version-safe publishing, CUDA graph capture, and overlapping background training with forward passes.

---

## 1. Why State Is the Only Mutable Thing {#en-sec-1}

Method bodies replay inside CUDA graphs, so Python assignments (`self.mean += 1`) do not execute on each step; input tensors and parameters are shared read-only views that cannot be mutated in-place; and row ordering changes on every step. Therefore, any value that persists across steps must be declared as state and only published as a new version via `@pllm.train` or `@pllm.write`. For the complete set of rules and rationale, see [What Goes in a Method Body](/pllm/docs/guides/workload-anatomy/#en-what-goes-in-a-method-body).

---

## 2. Two Scopes and Routing Keys {#en-sec-2}

When declaring state, the first decision is its **lifetime and sharing scope**. pLLM provides two scopes:

| Scope | Declaration | Semantics & Lifetime | Typical Uses |
| :--- | :--- | :--- | :--- |
| **Request scope** | `pllm.RequestState(...)` | Each request (or group of requests) owns a private copy of the state. Allocated upon request admission and freed when the request finishes. | Per-request LoRA, per-request online distillation networks, per-request running token statistics. |
| **Instance scope** | `pllm.InstanceState(...)` | A single copy of the state shared across the entire inference engine instance. All concurrent requests read and operate on the same physical tensors. | Frozen probe weights, shared embedding/lookup tables, large global tables, engine-wide statistics. |

### Grouping request state with `key_fields` {#en-grouping-request-state-with-key_fields}

By default, `RequestState` is strictly isolated per request—each request gets its own private state copy. When a group of requests should share one copy (for example, multiple candidate samples for the same prompt sharing one adapter, or requests from the same tenant sharing trained updates), specify routing keys via `key_fields`:

- `("request_id",)` (default): One isolated state copy per request.
- `("prompt_index",)`: All samples (`sample`) drawn for the same prompt within a single `generate` call share one state copy.
- `("sample_index",)`: Shares state across prompts aligned by relative sample index.
- `("route",)`: Shares state by an external application routing key. The host passes routing keys via `generate(prompts, routes=...)`, and requests with matching route values join the same state copy. Even after a request finishes, that state copy remains retained for the session so subsequent requests with the same route reuse it. For advanced routing patterns and capacity settings, see [Share Training Across Requests](/pllm/docs/howto/share-training-across-requests/).

---

## 3. Declaring Fields: Shapes, `Dimensions`, and `Field` {#en-sec-3}

State consists of one or more named fields, declared either with a single shape or with a field dictionary:

### Single-field and multi-field state {#en-single-field-and-multi-field-state}

- **Single field**: Pass `shape=(...)` directly; the state gets a single default field named `"value"`:
  ```python
  from dataclasses import dataclass

  import torch
  from torch import nn

  import pllm

  bias = pllm.InstanceState(shape=(896,))
  ```
- **Multiple fields**: Pass `fields={"name": spec, ...}` (or pass the dictionary as the first positional argument) to declare named fields:
  ```python
  linear = pllm.RequestState({
      "weight": lambda m: (m.hidden_size, m.hidden_size),
      "bias": (896,),
  })
  ```

### Forms of field shape specs {#en-forms-of-field-shape-specs}

Each field's shape spec accepts three forms:
1. **Static fixed tuple**: such as `(896,)` or `(4, 896)`.
2. **Callable taking `pllm.Dimensions`**: such as `lambda m: (m.vocab_size,)`. The argument `m` exposes static model metadata:
   - `m.hidden_size`: hidden dimension (e.g., 896 for Qwen2.5-0.5B);
   - `m.vocab_size`: vocabulary size (e.g., 151,936);
   - `m.layer_count`: number of layers (e.g., 24);
   - `m.target_shapes`: `(in_features, out_features)` of target projections when the state is read by `pllm.model.Adapter`;
   - `m.dtype`: the model's compute dtype.
3. **`pllm.Field`**: Used when a field's shape depends dynamically on each target projection layer (for instance, with `pllm.model.Adapter`):
   ```python
   lora_a = pllm.Field(lambda in_features, out_features: (4, in_features))
   ```
   `pllm.Field` also lets you override the storage dtype per field (such as `pllm.Field((896,), dtype=torch.float32)`).
   Plain shape tuples and `pllm.Field` entries can be mixed freely within the same state (for example, `{"w": pllm.Field(lambda width, _: (width,)), "b": (1,)}`).

### Class attribute vs. typed `@property` {#en-class-attribute-vs-typed-property}

- **Class attribute**: If shapes and config are known when writing the class, declare the state directly as a class attribute.
- **Typed `@property`**: If a state's shape depends on instance parameters (such as a user-supplied `rank` on a `@dataclass(frozen=True)`), declare it as a `@property` **and annotate its return type** (`-> pllm.RequestState` or `-> pllm.InstanceState`):

```python
@dataclass(frozen=True)
class DynamicProbe(pllm.Workload):
    rank: int = 4

    @property
    def head(self) -> pllm.RequestState:
        return pllm.RequestState(
            init=pllm.FromModule(lambda m: nn.Linear(m.hidden_size, self.rank))
        )
```

**Why is an explicit return type annotation required?**
When building the engine and running [`pllm.explain`](/pllm/docs/guides/explain/), pLLM inspects class attributes and methods via static reflection. Type annotations let the planner identify which members are state declarations and their scopes without instantiating unknown code or triggering side effects.

---

## 4. Initial Values: `init=` {#en-sec-4}

Every state must have a well-defined initial value at admission (request arrival or engine startup). That initializer is declared via `init=` (each new request's initial state copy is copied from this source):

| `init=` Option | Scope | Initializer Source & Behavior |
| :--- | :--- | :--- |
| `pllm.Zeros()` | Request / Instance | Initializes all fields to zero (the default). |
| `pllm.Computed(function)` | Request / Instance | Calls `function(fields, generator)` once in-place with a seeded generator at engine build time; subsequent admissions copy that initial buffer. |
| `pllm.Pretrained(path, placement="device")` | Request / Instance | Loads tensors matching field names from a `.safetensors` file (whose hash enters workload identity). `placement="host"` pins large read-only tables in host pinned memory; the default `"device"` never silently falls back to host if it does not fit. |
| `pllm.ModelCheckpoint(names)` | `InstanceState` only | Reads extra tensors from the served model checkpoint not used by vLLM's model class (`names` maps `{field: tensor_name}` and may include `{layer}` placeholders); those tensors are excluded from vLLM's own weight load. Read-only, no optimizer. Do not use this to read weights that vLLM itself consumes. |
| `pllm.FromModule(factory, pretrained="")` | Request / Instance | Uses the parameters of the `nn.Module` returned by `factory(dimensions)` as the state's sole field, initialized by the module itself or loaded from `pretrained`. |
| `pllm.Fed(order="ordered" \| "latest")` | `InstanceState` only | Dynamically pushed by an external producer process via `llm.feed(state)`; `"ordered"` consumes every published update strictly in order, while `"latest"` keeps only the newest. At most one per workload. |
| `pllm.Imported()` | Request / Instance | Supplied per inference call via `generate(batch=pllm.io.StateImport(...))` before the request's first forward pass; `RequestState` supplies one row per prompt, while `InstanceState` supplies the whole tensor. |

For loading pretrained weights or external `.safetensors` files and troubleshooting them, see [Load Trained Weights](/pllm/docs/howto/load-trained-weights/).

---

## 5. Frozen State (Read-Only State) {#en-sec-5}

If a state does not need to be updated by `@pllm.train` or `@pllm.write`, omit `optimizer=` (or set `optimizer=None`). This is called **frozen state**.

Frozen state has three key properties:
1. **Zero extra overhead**: No gradient slots or Adam momentum slots are allocated; physical memory equals only the field tensors themselves.
2. **Never modified by an optimizer**: Its value comes solely from its initializer, `pllm.io.StateImport`, or new versions written by `@pllm.write`.
3. **Freely readable anywhere**: Can be read via `pllm.parameter` from any method (including `@pllm.export`, `@pllm.guide_sampling`, `@pllm.custom_residual`, etc.).

---

## 6. Automatic Storage Dtype (`dtype="auto"`) and Explicit Overrides {#en-sec-6}

State parameters default to `dtype="auto"`. At build time, pLLM statically analyzes each field's read/write roles across methods and balances numerical stability against memory cost:

| Role of Field in Methods | Dtype Chosen by `dtype="auto"` | Rationale & Measured Impact |
| :--- | :--- | :--- |
| **Parameters updated by an optimizer** (`@pllm.train`) | `torch.float32` | Prevents small gradients and momentum updates from underflowing in low precision. Long-sequence gradient accumulation in BF16 easily diverges. |
| **Accumulators read and written back** (`@pllm.write`) | `torch.float32` | Cross-step accumulations (such as running means) make small updates each step whose truncation error compounds in low precision. |
| **Cross-chunk / cross-step carry fields** (`carry`) | `torch.float32` | Sequential cross-step recurrence tests show relative error up to ~ $$1.4 \times 10^{-1}$$ in BF16 vs. ~ $$3.7 \times 10^{-4}$$ in FP32. |
| **Read-only fields** (frozen probes, steering vectors) or **write-only fields** | Matches model dtype (e.g., `torch.bfloat16`) | Cuts state VRAM and memory bandwidth in half and matches the backbone model's GEMM compute dtype directly. |

### Explicit dtype override {#en-explicit-dtype-override}

If you need to run FP32-only operators directly on a read-only field, set `dtype=torch.float32` on `InstanceState` / `RequestState`, or override a single field in `pllm.Field`:

```python
state = pllm.InstanceState({
    "w_fp32": pllm.Field(lambda width, _: (width,), dtype=torch.float32),  # Explicit FP32
    "w_model": pllm.Field(lambda width, _: (width,)),                     # Auto-follows model dtype (BF16)
})
```

---

## 7. Reading State and Visible Versions {#en-sec-7}

Inside workload methods, state and state metadata are read via four input markers (declared as parameter defaults):

| Input Marker | Return Type & Shape | Semantics |
| :--- | :--- | :--- |
| `pllm.parameter(state, field, *, per_row=False)` | `torch.Tensor` | Reads a named field of the given state.<br>• **`RequestState`**: Automatically carries a leading row axis `[rows, *S]` aligned to the current op's scheduled rows (each row sees its own request's state).<br>• **`InstanceState`**: Keeps its declared shape `[*S]`; setting `per_row=True` expands a broadcast row axis `[rows, *S]` for batched per-row math. |
| `pllm.module(state)` | `nn.Module` | Reads a module initialized via `init=pllm.FromModule(...)`.<br>For `RequestState`, each request row runs using that request's private parameter copy. |
| `pllm.version(state)` | `torch.Tensor` (`int64 [rows]`) | Returns how many updates have been published for each request's state (0 upon admission). Commonly used to gate on "has trained yet" (e.g., `torch.where(version > 0, ...)`). |
| `pllm.index(state)` | `torch.Tensor` (`int64 [rows]`) | Returns each request's slot index in the underlying state pool, for per-row indexing into external pooled tensors. |

### Which version does a read see? {#en-which-version-does-a-read-see}

Whenever any method reads state during a step, it **always sees the version already published before the current step began** (the pre-publish version).
- Even if a side stream is training during the current step and preparing a new version, the main stream's forward pass still reads the version published on the previous step;
- Running requests always observe a consistent snapshot and never read half-updated intermediate data. For the version-publishing timeline and stream overlap model, see [Core Concepts](/pllm/docs/explain/core-concepts/).

---

## 8. Cross-Step Memory: `carry` Fields and `pllm.carry` {#en-sec-8}

When you need to carry non-optimizer continuous memory across decode steps (such as the previous step's prediction error or shallow feature), use **carry fields**:

1. **Declare a carry field**: Configure `carry={"field_name": shape}` on the state. Carry fields allocate no gradient or momentum slots and are not updated by the optimizer.
2. **Write during training**: Wrap an input marker on `@pllm.train` with `pllm.carry(source, state=..., field=...)`. When the training update publishes, the framework automatically writes each request's last row (latest token) of that input into the corresponding carry field.
3. **Read on the next step**: Read the previously published value via `pllm.parameter(state, "field_name")`.

**Why is the carry field always zero on step 0?**
Carry fields are zero-initialized at request admission. Under the rule that reads observe the version published before the current step starts, at the start of step 0 no update has ever been published for that request, so step 0 necessarily sees zeros—a direct consequence of versioned publishing semantics.

---

## 9. `PositionState` and Host Read-Only Tables {#en-sec-9}

Beyond ordinary tensor state, pLLM supports two specialized state layouts:

### `PositionState` {#en-positionstate}

`pllm.PositionState(initial=0.0)` holds one hidden-width vector per KV-cache block, accessible only inside `@pllm.custom_forward` via `layers.load` / `layers.save` (see [Forward Structure](/pllm/docs/guides/forward-structure/)).

### Large host read-only tables (`placement="host"`) {#en-large-host-read-only-tables-placementhost}

When an algorithm uses a massive read-only table (such as a million-entry n-gram prior table or retrieval memory), keeping it in GPU VRAM severely shrinks available KV cache capacity. With `pllm.InstanceState(..., init=pllm.Pretrained(path, placement="host"))`, the table stays in pinned CPU memory:
- **Overlapped side-stream transfer**: Lookup methods that depend only on token IDs are scheduled during `request.prepare` at the start of each step on a dedicated side CUDA stream (`side` lane). They consume only roughly a quarter of SM capacity to asynchronously stream needed rows over PCIe, hiding transfer latency behind the model forward pass.
- **Allowed ops**: To guarantee bounded static transfers, host tables allow only row-gathering operations: slice indexing (`table[idx]`, including sliced views such as `table[:, ::2]` before indexing), `torch.index_select`, `torch.nn.functional.embedding`, and `torch.vmap`. Attempting any other op on a host table is rejected at plan time. See [Performance and Memory](/pllm/docs/explain/performance/) for the bandwidth model.

---

## 10. Memory Reservation and Budget Controls {#en-sec-10}

Although state slots are assigned on demand, their total memory must be reserved before engine startup:

- **Why are more pages reserved than `max_num_seqs`?**
  In [`pllm.explain`](/pllm/docs/guides/explain/) output, the page count per field (`x N pages`) is typically noticeably larger than `max_num_seqs`. Under the default cross-step overlap mode (`publish_lag=1`), the side stream writes step $$n+1$$'s new version while the main stream concurrently reads step $$n$$'s old version, so multiple versions must coexist in memory simultaneously. Old versions retain only fields that readers read: AdamW step counts and first/second moments are read only by the optimizer, so they are reserved only for the latest and currently-being-written versions.
- **Budget knobs**:
  `ExecutionOptions.state_page_budget_bytes` caps total state pool memory and raises `ValueError` at compile time if exceeded. `ExecutionOptions.execution_peak_budget_bytes` caps total execution peak memory (including state, method workspaces, and CUDA graph reservations), automatically falling back side-stream tasks to inline main-stream execution under memory pressure to avoid OOM. Inspect runtime memory usage and numerical health via `llm.diagnostics().state_usage` and `numeric_health`; see [Configuration](/pllm/docs/reference/config/) and [explain and diagnostics Fields](/pllm/docs/reference/diagnostics/).

---

## 11. Code Examples {#en-sec-11}

All examples below run and pass verification directly on CPU (no GPU required).

### Example 1: Frozen probe with `Computed` initialization (`InstanceState`) {#en-example-1-frozen-probe-with-computed-initialization-instancestate}

This example declares a frozen linear probe shared across the engine, populates its weights in-place via a seeded generator at build time, and exports per-token scores via `@pllm.export`.

```python
from dataclasses import dataclass

import torch
from torch import nn

import pllm
from pllm import testing

CONFIG = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=8)


def init_probe(fields: dict[str, torch.Tensor], generator: torch.Generator) -> None:
    fields["weight"].normal_(generator=generator)
    fields["bias"].zero_()


class FrozenProbe(pllm.Workload):
    # Instance scope + frozen state: initialized once at startup, never updated by an optimizer
    classifier = pllm.InstanceState(
        {"weight": lambda m: (m.hidden_size,), "bias": (1,)},
        init=pllm.Computed(init_probe),
    )

    @pllm.export(name="probe_score", max_rows=8, max_cols=1)
    def score(
        self,
        h: torch.Tensor = pllm.final_hidden(),
        w: torch.Tensor = pllm.parameter("classifier", "weight"),
        b: torch.Tensor = pllm.parameter("classifier", "bias"),
    ) -> torch.Tensor:
        # [rows, 896] * [896] -> [rows, 1]
        return torch.sigmoid((h.float() * w).sum(dim=-1, keepdim=True) + b)


print(testing.check(FrozenProbe(), CONFIG).passed)
```

### Example 2: Per-request neural network module (`RequestState`) {#en-example-2-per-request-neural-network-module-requeststate}

This example declares a small neural module dynamically allocated per request so each request owns independent weights and runs its own private module row-by-row.

```python
@dataclass(frozen=True)
class PerRequestHead(pllm.Workload):
    rank: int = 4

    # Shape depends on instance attribute rank, so declare as a typed property
    @property
    def head(self) -> pllm.RequestState:
        return pllm.RequestState(
            init=pllm.FromModule(lambda m: nn.Linear(m.hidden_size, self.rank))
        )

    @pllm.export(name="probe", max_rows=8, max_cols=4)
    def score(
        self,
        h: torch.Tensor = pllm.final_hidden(),
        head: nn.Module = pllm.module("head"),
    ) -> torch.Tensor:
        # head automatically binds each request row to that request's private weights
        return head(h.float())


print(testing.check(PerRequestHead(), CONFIG).passed)
```

### Example 3: Updating a per-request running mean with `@pllm.write` {#en-example-3-updating-a-per-request-running-mean-with-pllmwrite}

This example updates state with custom tensor math: on each step it reads the previous running mean and current token representation, computes a weighted update, and atomically publishes it via `@pllm.write`.

```python
class RunningMean(pllm.Workload):
    stats = pllm.RequestState({"mean": pllm.Field(lambda width, _: (width,))})

    @pllm.write(state="stats", fields=("mean",))
    def update(
        self,
        h: torch.Tensor = pllm.final_hidden(),
        mean: torch.Tensor = pllm.parameter("stats", "mean"),
    ) -> tuple[torch.Tensor]:
        # Accumulator reads and rewrites itself, so dtype="auto" resolves to float32
        return (0.9 * mean + 0.1 * h.float(),)

    @pllm.export(name="mean_norm", max_rows=8, max_cols=1)
    def report(
        self,
        mean: torch.Tensor = pllm.parameter("stats", "mean"),
    ) -> torch.Tensor:
        return mean.norm(dim=-1, keepdim=True)


print(testing.check(RunningMean(), CONFIG).passed)
```

### Example 4: Cross-step memory with a `carry` field {#en-example-4-cross-step-memory-with-a-carry-field}

This example passes information across steps during online training: the training method computes a prediction loss while saving shallow activations into a carry field via `pllm.carry`, then reads that carried vector back on the next step.

```python
@dataclass(frozen=True)
class StepMemory(pllm.Workload):
    beta: float = 0.5

    head = pllm.RequestState(
        {"w": lambda m: (m.hidden_size, m.hidden_size)},
        optimizer=pllm.AdamW(lr=1e-3),
        carry={"prev_act": lambda m: (m.hidden_size,)},
    )

    def strength(self) -> float:
        return self.beta

    @pllm.guide_sampling(beta=strength)
    def predict(
        self,
        shallow: torch.Tensor = pllm.hidden(0),
        w: torch.Tensor = pllm.parameter("head", "w"),
    ) -> torch.Tensor:
        return torch.bmm(shallow.float().unsqueeze(1), w).squeeze(1)

    @pllm.train(state="head")
    def loss(
        self,
        # On publish, writes each request's latest shallow row into prev_act
        shallow: torch.Tensor = pllm.carry(pllm.hidden(0), state="head", field="prev_act"),
        deep: torch.Tensor = pllm.hidden(-1),
        w: torch.Tensor = pllm.parameter("head", "w"),
        # Reads the value published on the previous step (all zeros on step 0)
        prev: torch.Tensor = pllm.parameter("head", "prev_act"),
    ) -> torch.Tensor:
        pred = torch.bmm(shallow.float().unsqueeze(1), w).squeeze(1)
        return (pred + prev - deep.float()).square().mean(-1)


print(pllm.explain(StepMemory(), CONFIG))
```

---

## 12. Common Errors {#en-sec-12}

When patterns on this page are rejected, exact error messages, causes, and fixes are listed in the [State and training](/pllm/docs/reference/errors/#en-state-and-training) and [Loading and placing state](/pllm/docs/reference/errors/#en-loading-and-placing-state) sections of the error catalog.

---

## 13. Self-Check {#en-sec-13}

<details>
<summary>1. Why does <code>dtype="auto"</code> pick <code>float32</code> rather than the more memory-efficient <code>bfloat16</code> for optimizer-updated state or accumulator fields?</summary>

During autoregressive decoding, per-step online training gradients and running averages are typically small. In `bfloat16` (only 7 mantissa bits), tiny gradient steps or weighted increments added to existing weights underflow and round straight to zero (update swallowing), preventing learning or causing divergence. Measurements show carry recurrence has ~ $$1.4 \times 10^{-1}$$ relative error in `bfloat16` versus ~ $$3.7 \times 10^{-4}$$ in `float32`. pLLM therefore uses `float32` automatically for fields with small accumulations while keeping read-only and write-only fields in the model dtype (e.g., `bfloat16`) to save memory.
</details>

<details>
<summary>2. Why does reading a <code>carry</code> field on step 0 always return all zeros?</summary>

Because of pLLM's version-publishing semantics: any method reading state during step $$n$$ always sees the version already published before step $$n$$ started (i.e., the output published on step $$n-1$$). At the start of step 0 upon request admission, no training update has run yet, and carry fields are zero-initialized at admission—so step 0 necessarily reads zeros.
</details>

<details>
<summary>3. Why do fields read from <code>RequestState</code> automatically carry a leading row axis (<code>[rows, *S]</code>), whereas fields read from <code>InstanceState</code> do not by default?</summary>

Under continuous batching, multiple concurrent requests are packed into the same batch. `RequestState` keeps a private parameter copy per request, so on every read pLLM gathers and aligns each scheduled row to its own request's slice, yielding `[rows, *S]`. `InstanceState` has a single global copy shared across the entire engine that does not scale with concurrent request count, so it keeps its declared shape `[*S]` by default (pass `per_row=True` if you want a broadcast row axis for batched per-row math).
</details>

</div>

<div class="zine-lang-panel is-hidden" data-lang-panel="zh" lang="zh-CN" markdown="1">

<div id="zh-sec-1"></div>

探针的权重、每个请求自己的适配器、跨请求共享的大表、生成过程的滑动统计量：推理期算法里要跨步或跨请求保留的值，在 pLLM 里都声明成状态。

**状态（State）是 pLLM 中唯一可变的东西**。只要某个值需要在步与步之间、请求与请求之间存活，就必须显式声明为状态。声明状态后，pLLM 负责分配显存、按版本安全发布、处理 CUDA Graph 捕获，并在后台重叠训练与前向。

---

## 1. 为什么状态是唯一可变的东西 {#zh-sec-2}

方法体在 CUDA graph 里回放，Python 赋值（`self.mean += 1`）不会每步执行；输入和参数是共享的只读视图，不能原地改；
行的排列每步都变。所以要跨步存活的值都声明成状态，只经 `@pllm.train` 或 `@pllm.write` 发布新版本。完整的规则与原因见
[方法体里写什么](/pllm/docs/guides/workload-anatomy/#zh-sec-3)。

---

## 2. 两种作用域与路由键 {#zh-sec-3}

声明状态时，首要决策是选择其**生命周期与共享范围**。pLLM 提供两种作用域：

| 作用域 | 声明方式 | 语义与生命周期 | 适用场景 |
| :--- | :--- | :--- | :--- |
| **请求作用域** | `pllm.RequestState(...)` | 每个请求（或每组请求）拥有一份独立的私有状态副本。随请求准入（admission）分配，随请求完成而释放。 | 每请求 LoRA、每个请求的在线蒸馏小网络、每个请求的 token 累加统计。 |
| **实例作用域** | `pllm.InstanceState(...)` | 整个推理引擎实例共享唯一一份状态。所有并发请求读取并操作同一组物理张量。 | 冻结探针权重、跨请求共享的嵌入词表、全局大表、模型级统计量。 |

### 用 `key_fields` 对请求状态分组 {#zh-sec-4}

`RequestState` 默认按请求严格隔离，即每个请求独立拥有自己的状态副本。如果需要让一组请求共享同一份状态（例如同一 Prompt 的多个候选采样共享同一个适配器，或者同一租户的请求共享训练成果），可以通过 `key_fields` 声明路由键：

- `("request_id",)`（默认）：每个请求独立一份状态副本，互不干扰。
- `("prompt_index",)`：同一次 `generate` 中同一 Prompt 抽取的多个采样（`sample`）共享同一份状态副本。
- `("sample_index",)`：跨 Prompt 按采样的相对序号对齐共享。
- `("route",)`：按外部业务路由键共享。宿主调用 `generate(prompts, routes=...)` 传入路由值，路由值相同的请求加入同一状态副本。即使当前请求结束，该状态副本仍会在会话中保留，供后续具有相同路由值的请求复用。关于路由键的进阶用法与容量配置，详见 [让一组请求共享训练成果](/pllm/docs/howto/share-training-across-requests/)。

---

## 3. 声明字段：形状、Dimensions 与 Field {#zh-sec-5}

状态由一个或多个具名字段组成。字段可以通过单个形状或字典方式声明：

### 单字段与多字段 {#zh-sec-6}

- **单个字段**：直接传 `shape=(...)`，该状态默认拥有一个名为 `"value"` 的字段。
  ```python
  from dataclasses import dataclass

  import torch
  from torch import nn

  import pllm

  bias = pllm.InstanceState(shape=(896,))
  ```
- **多字段**：通过 `fields={"name": spec, ...}`（或第一个位置参数传入字典）声明具名字段：
  ```python
  linear = pllm.RequestState({
      "weight": lambda m: (m.hidden_size, m.hidden_size),
      "bias": (896,),
  })
  ```

### 字段形状的声明形式 {#zh-sec-7}

每个字段的形状规范支持三种形式：
1. **静态固定元组**：如 `(896,)`、`(4, 896)`。
2. **接收 `pllm.Dimensions` 的 Callable**：如 `lambda m: (m.vocab_size,)`。参数 `m` 暴露模型的静态事实：
   - `m.hidden_size`：隐藏层维度（例如 Qwen2.5-0.5B 为 896）；
   - `m.vocab_size`：词表大小（例如 151936）；
   - `m.layer_count`：层数（例如 24）；
   - `m.target_shapes`：状态被 `pllm.model.Adapter` 读取时，各目标投影的 `(in_features, out_features)`；
   - `m.dtype`：模型的精度。
3. **`pllm.Field`**：当字段需要按所附加的目标投影层动态确定形状时使用（例如配合 `pllm.model.Adapter`）：
   ```python
   lora_a = pllm.Field(lambda in_features, out_features: (4, in_features))
   ```
   同时，`pllm.Field` 还支持为单个字段显式指定独立的存储精度（如 `pllm.Field((896,), dtype=torch.float32)`）。
   同一个状态中可以自由混合普通形状元组和 `pllm.Field`（例如 `{"w": pllm.Field(lambda width, _: (width,)), "b": (1,)}`）。

### 类属性 vs 带类型的 property {#zh-sec-8}

- **类属性**：如果状态的形状和配置在类编写时即可确定，直接声明为类属性。
- **带类型的 property**：如果状态形状依赖于算法的初始化参数（例如在 `@dataclass(frozen=True)` 中由用户传入的 `rank`），必须声明为 `@property`，并且**必须标注返回类型**（`-> pllm.RequestState` 或 `-> pllm.InstanceState`）：

```python
@dataclass(frozen=True)
class DynamicProbe(pllm.Workload):
    rank: int = 4

    @property
    def head(self) -> pllm.RequestState:
        return pllm.RequestState(
            init=pllm.FromModule(lambda m: nn.Linear(m.hidden_size, self.rank))
        )
```

**为什么必须显式标注返回类型？**
pLLM 在构建引擎和执行 [`pllm.explain`](/pllm/docs/guides/explain/) 规划时，通过静态反射检查类属性与方法。类型注解让规划器在不实例化未知代码或触发潜在副作用的情况下，快速准确地识别出哪些成员是状态以及它们的作用域。

---

## 4. 初始值来源：`init=` {#zh-sec-9}

每个状态在准入时（请求开始或引擎启动）都必须有明确的初值。初值由 `init=` 参数声明（每条请求的初始副本都拷贝自该来源）：

| `init=` 选项 | 作用域 | 初始值来源与行为 |
| :--- | :--- | :--- |
| `pllm.Zeros()` | Request / Instance | 所有字段初始化为 0（默认行为）。 |
| `pllm.Computed(function)` | Request / Instance | 引擎构建时用带种子的生成器调用 `function(fields, generator)` 原地填充一次，后续每次准入直接拷贝该初始值。 |
| `pllm.Pretrained(path, placement="device")` | Request / Instance | 从 `.safetensors` 文件加载与字段同名的张量（文件哈希计入负载身份）。`placement="host"` 可将只读大表固定在锁页主机内存中，默认 `"device"` 放不下时绝不隐式降级。 |
| `pllm.ModelCheckpoint(names)` | 仅限 InstanceState | 读服务模型检查点里 vLLM 模型类用不到的额外张量（`names` 映射 `{field: tensor_name}`，可含 `{layer}` 占位符），这些张量从 vLLM 的加载中排除；只读，无优化器。不要用它读 vLLM 自己要用的权重。 |
| `pllm.FromModule(factory, pretrained="")` | Request / Instance | 将 `factory(dimensions)` 构建的 `nn.Module` 的参数作为状态的唯一字段。按模块自身的机制初始化，或从 `pretrained` 路径加载权重。 |
| `pllm.Fed(order="ordered" \| "latest")` | 仅限 InstanceState | 由外部生产者进程通过 `llm.feed(state)` 动态推送更新；`"ordered"` 严格消费每次发布，`"latest"` 仅保留最新；每个负载至多声明一个。 |
| `pllm.Imported()` | Request / Instance | 每次发起推理时通过 `generate(batch=pllm.io.StateImport(...))` 在请求首个前向之前传入；RequestState 为每个 prompt 提供一行，InstanceState 提供整个张量。 |

关于从磁盘载入预训练权重或外部 safetensors 的具体操作与排查，详见 [加载训练好的权重](/pllm/docs/howto/load-trained-weights/)。

---

## 5. 冻结状态（只读状态） {#zh-sec-10}

如果一个状态不需要被 `@pllm.train` 或 `@pllm.write` 更新，只需省略 `optimizer=`（或显式写 `optimizer=None`）。这种状态称为**冻结状态（Frozen State）**。

冻结状态具有以下性质：
1. **零额外开销**：不分配梯度槽位与 Adam 动量槽位，物理显存仅等于字段张量本身的大小。
2. **不被优化器改变**：它的值只来自初值，或者来自 `pllm.io.StateImport` 导入和 `@pllm.write` 写入的新版本。
3. **全局自由可读**：可以在任何方法（包括观测导出 `@pllm.export`、采样引导 `@pllm.guide_sampling`、残差改写 `@pllm.custom_residual` 等）中通过 `pllm.parameter` 读取。

---

## 6. 存储精度自动规则（`dtype="auto"`）与显式覆盖 {#zh-sec-11}

状态参数默认采用 `dtype="auto"`。pLLM 会在构建期静态分析每个字段在各个方法中的读写角色，自动权衡数值稳定性与显存开销：

| 字段在方法中的角色 | `dtype="auto"` 推导的精度 | 原因与实测影响 |
| :--- | :--- | :--- |
| **被优化器更新的参数**（`@pllm.train`） | `torch.float32` | 防止微小梯度与动量更新在低精度下下溢被吞噬。在 BF16 下长序列梯度的累加极易发散。 |
| **读出再写回的累加器**（`@pllm.write`） | `torch.float32` | 跨步累加（如滑动均值）每次更新幅度小，在低精度下误差会持续累积放大。 |
| **跨块/跨步进位字段**（carry 字段） | `torch.float32` | 跨步连续递推测试表明，carry 采用 BF16 相对误差可高达约 $$1.4 \times 10^{-1}$$，而采用 FP32 仅约 $$3.7 \times 10^{-4}$$。 |
| **只读字段**（如冻结探针、引导向量）<br>或 **只写字段** | 跟随模型精度（如 `torch.bfloat16`） | 省下一半状态显存与访存带宽，且与主干模型的 GEMM 计算精度自然对齐。 |

### 显式覆盖精度 {#zh-sec-12}

如果需要在只读字段上直接执行仅支持单精度的算子，可以在 `InstanceState` / `RequestState` 上指定 `dtype=torch.float32`，或在 `pllm.Field` 中为单个字段覆盖：

```python
state = pllm.InstanceState({
    "w_fp32": pllm.Field(lambda width, _: (width,), dtype=torch.float32),  # 显式 FP32
    "w_model": pllm.Field(lambda width, _: (width,)),                     # 自动跟随模型精度（BF16）
})
```

---

## 7. 读取状态与可见版本 {#zh-sec-13}

在算法方法中，通过以下四种输入标记读取状态及其元数据（作为参数默认值声明）：

| 读取标记 | 返回类型与形状 | 语义说明 |
| :--- | :--- | :--- |
| `pllm.parameter(state, field, *, per_row=False)` | `torch.Tensor` | 读取指定状态的某个具名字段。<br>• **RequestState**：自动带有领先行轴 `[rows, *S]`，对齐当前操作调度的行（每行对应自己请求的状态）。<br>• **InstanceState**：保持声明时的原始形状 `[*S]`；若指定 `per_row=True`，会自动展开一个广播行轴 `[rows, *S]` 以便进行按行矩阵运算。 |
| `pllm.module(state)` | `nn.Module` | 读取以 `init=pllm.FromModule(...)` 初始化的模块。<br>对 `RequestState`，每个请求行在计算时使用该请求私有的参数副本。 |
| `pllm.version(state)` | `torch.Tensor`（int64 `[rows]`） | 返回当前每条请求的状态已完成更新发布的次数（准入时为 0）。常用作“是否已训练过”的条件门控（例如 `torch.where(version > 0, ...)`）。 |
| `pllm.index(state)` | `torch.Tensor`（int64 `[rows]`） | 返回当前每条请求在底层状态池中的槽位编号，用于对外部池化张量做按行查表。 |

### 读取看到的是哪个版本？ {#zh-sec-14}

任何方法在某一步中读取状态时，看到的**永远是当前步骤开始前已发布的版本**（发布前版本）。
- 即使当前步骤的旁路流正在训练并准备写入新版本，当前主干流的前向计算依然读取上一步发布的旧版本；
- 正在运行的请求始终保持自己的版本快照，绝不会读到只更新了一半的中间脏数据。关于版本发布的时序与流重叠模型，详见 [核心概念](/pllm/docs/explain/core-concepts/)。

---

## 8. 跨步记忆：carry 字段与 `pllm.carry` {#zh-sec-15}

如果需要在解码步骤之间传递非优化器参数的连续记忆（例如上一步的预测误差、上一步记录的浅层特征），可以使用 **carry 字段**：

1. **声明 carry 字段**：在状态中配置 `carry={"field_name": shape}`。carry 字段不分配梯度与动量槽位，不会被优化器更新。
2. **在训练中写入**：把 `@pllm.train` 方法的某个输入默认值用 `pllm.carry(source, state=..., field=...)` 包裹。在训练更新发布时，框架自动将该输入中每条请求最后一行（最新 token）的张量值写入对应的 carry 字段。
3. **在下一步读取**：使用 `pllm.parameter(state, "field_name")` 读取上一步发布的值。

**为什么第一步读取到的始终是零？**
状态的 carry 字段在请求准入时进行零初始化。根据“读取看到当前步开始前已发布版本”的时序规则，在第 0 步开始时，该状态尚未发生过任何更新发布，因此第 0 步读取到的初始版本只能是零。这正是版本发布语义的自然体现。

---

## 9. PositionState 与主机只读表 {#zh-sec-16}

除了常规的张量状态外，pLLM 还支持两种特定拓扑的状态存储：

### PositionState {#zh-positionstate}

`pllm.PositionState(initial=0.0)` 每个 KV cache 块持有一个 hidden 宽度的向量，仅供 `@pllm.custom_forward` 内通过 `layers.load` / `layers.save` 读写（详见 [改写 forward 结构](/pllm/docs/guides/forward-structure/)）。

### 主机只读大表（`placement="host"`） {#zh-sec-17}

当算法需要使用超大规模的只读表（如百万级 n-gram 先验表、检索记忆库）时，将其全部常驻 GPU 显存会导致 KV Cache 空间严重缩减。通过 `pllm.InstanceState(..., init=pllm.Pretrained(path, placement="host"))`，可将只读表保留在 CPU 锁页内存中：
- **旁路重叠搬运**：仅依赖 token ID 的查表方法会被规划器调度在每步最初的 `request.prepare` 阶段、于独立的旁路 CUDA stream（`side` lane）上执行。它只占用约四分之一的 SM 算力通过 PCIe 异步搬运所需行，将传输时间完全隐藏在模型前向计算的背后。
- **允许的算子**：为了保证静态内存与传输边界，主机表仅支持按行抽取操作：切片索引（`table[idx]`，包含先做切片视图如 `table[:, ::2]` 再索引）、`torch.index_select`、`torch.nn.functional.embedding` 以及 `torch.vmap`。在规划期尝试对主机表执行其他算子会被直接拒绝。更详细的带宽模型见 [性能与显存](/pllm/docs/explain/performance/)。

---

## 10. 显存预留与预算控制 {#zh-sec-18}

状态虽然按需分配，但必须在引擎启动前完成显存预留规划：

- **为什么预留的页数大于并发请求数？**
  在 [`pllm.explain`](/pllm/docs/guides/explain/) 的报告中，状态每个字段的页数（`x N pages`）通常显著大于 `max_num_seqs`。这是因为在默认的跨步重叠模式（`publish_lag=1`）下，旁路流正在写出第 $$n+1$$ 步的新版本，而主干流还在并发读取第 $$n$$ 步的旧版本，新旧多个版本必须在显存中同时并存。旧版本只保留读者会读的字段：AdamW 的步数和一、二阶矩只有优化器读，它们只为最新版本和正在写的版本预留。
- **预算控制旋钮**：
  `ExecutionOptions.state_page_budget_bytes` 约束状态池的总显存上限，超出可用空间时会在编译期抛出 `ValueError`。`ExecutionOptions.execution_peak_budget_bytes` 约束包括状态、方法工作区和 CUDA Graph 预留在内的整体执行峰值，并在显存吃紧时自动将旁路任务退回主流内联执行以防止 OOM。运行时的具体物理开销与数值健康状态可通过 `llm.diagnostics().state_usage` 和 `numeric_health` 查询，详见 [配置项](/pllm/docs/reference/config/) 与 [explain 与 diagnostics 字段](/pllm/docs/reference/diagnostics/)。

---

## 11. 代码示例 {#zh-sec-19}

下面的代码示例均可在 CPU 上直接运行校验（无需 GPU）。

### 示例 1：带 `Computed` 初值的冻结探针（`InstanceState`） {#zh-sec-20}

该示例声明一个整个引擎共享的冻结探针分类器，在初始化时通过生成器原地填充权重，通过 `@pllm.export` 导出每个 token 的判别分数。

```python
from dataclasses import dataclass

import torch
from torch import nn

import pllm
from pllm import testing

CONFIG = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=8)


def init_probe(fields: dict[str, torch.Tensor], generator: torch.Generator) -> None:
    fields["weight"].normal_(generator=generator)
    fields["bias"].zero_()


class FrozenProbe(pllm.Workload):
    # 实例作用域 + 冻结状态：准入时初始化一次，永不被优化器更新
    classifier = pllm.InstanceState(
        {"weight": lambda m: (m.hidden_size,), "bias": (1,)},
        init=pllm.Computed(init_probe),
    )

    @pllm.export(name="probe_score", max_rows=8, max_cols=1)
    def score(
        self,
        h: torch.Tensor = pllm.final_hidden(),
        w: torch.Tensor = pllm.parameter("classifier", "weight"),
        b: torch.Tensor = pllm.parameter("classifier", "bias"),
    ) -> torch.Tensor:
        # [rows, 896] * [896] -> [rows, 1]
        return torch.sigmoid((h.float() * w).sum(dim=-1, keepdim=True) + b)


print(testing.check(FrozenProbe(), CONFIG).passed)
```

### 示例 2：每请求独立的神经网络模块（`RequestState`） {#zh-sec-21}

该示例声明一个随请求动态分配的小型神经网络模块，每条请求拥有独立的参数，并在前向计算时逐行调用各自的私有网络。

```python
@dataclass(frozen=True)
class PerRequestHead(pllm.Workload):
    rank: int = 4

    # 形状依赖实例属性 rank，声明为带类型注解的 property
    @property
    def head(self) -> pllm.RequestState:
        return pllm.RequestState(
            init=pllm.FromModule(lambda m: nn.Linear(m.hidden_size, self.rank))
        )

    @pllm.export(name="probe", max_rows=8, max_cols=4)
    def score(
        self,
        h: torch.Tensor = pllm.final_hidden(),
        head: nn.Module = pllm.module("head"),
    ) -> torch.Tensor:
        # head 内部自动为每条请求行绑定该请求独立的权重副本
        return head(h.float())


print(testing.check(PerRequestHead(), CONFIG).passed)
```

### 示例 3：用 `@pllm.write` 更新请求滑动均值 {#zh-sec-22}

该示例演示如何使用自定义张量数学更新状态：每一步读取上一步的滑动均值与当前 token 的特征向量，计算加权更新后通过 `@pllm.write` 原子发布。

```python
class RunningMean(pllm.Workload):
    stats = pllm.RequestState({"mean": pllm.Field(lambda width, _: (width,))})

    @pllm.write(state="stats", fields=("mean",))
    def update(
        self,
        h: torch.Tensor = pllm.final_hidden(),
        mean: torch.Tensor = pllm.parameter("stats", "mean"),
    ) -> tuple[torch.Tensor]:
        # 累加器读取并重写自身，自动推导存储精度为 float32
        return (0.9 * mean + 0.1 * h.float(),)

    @pllm.export(name="mean_norm", max_rows=8, max_cols=1)
    def report(
        self,
        mean: torch.Tensor = pllm.parameter("stats", "mean"),
    ) -> torch.Tensor:
        return mean.norm(dim=-1, keepdim=True)


print(testing.check(RunningMean(), CONFIG).passed)
```

### 示例 4：带 `carry` 进位字段的跨步记忆 {#zh-sec-23}

该示例演示在在线训练时跨步记录信息：训练方法计算预测误差，同时通过 `pllm.carry` 将浅层激活记录进 carry 字段，供下一步训练时作为递推输入读出。

```python
@dataclass(frozen=True)
class StepMemory(pllm.Workload):
    beta: float = 0.5

    head = pllm.RequestState(
        {"w": lambda m: (m.hidden_size, m.hidden_size)},
        optimizer=pllm.AdamW(lr=1e-3),
        carry={"prev_act": lambda m: (m.hidden_size,)},
    )

    def strength(self) -> float:
        return self.beta

    @pllm.guide_sampling(beta=strength)
    def predict(
        self,
        shallow: torch.Tensor = pllm.hidden(0),
        w: torch.Tensor = pllm.parameter("head", "w"),
    ) -> torch.Tensor:
        return torch.bmm(shallow.float().unsqueeze(1), w).squeeze(1)

    @pllm.train(state="head")
    def loss(
        self,
        # 发布更新时，将 shallow 中每个请求最新行的值存入 prev_act 字段
        shallow: torch.Tensor = pllm.carry(pllm.hidden(0), state="head", field="prev_act"),
        deep: torch.Tensor = pllm.hidden(-1),
        w: torch.Tensor = pllm.parameter("head", "w"),
        # 读取上一步发布的值（第一步读取为全 0）
        prev: torch.Tensor = pllm.parameter("head", "prev_act"),
    ) -> torch.Tensor:
        pred = torch.bmm(shallow.float().unsqueeze(1), w).squeeze(1)
        return (pred + prev - deep.float()).square().mean(-1)


print(pllm.explain(StepMemory(), CONFIG))
```

---

## 12. 常见错误 {#zh-sec-24}

这一页的写法被拒绝时的报错原文、原因和改法，见报错目录的[状态与训练](/pllm/docs/reference/errors/#zh-sec-6)、[加载与放置状态](/pllm/docs/reference/errors/#zh-sec-7)几节。

---

## 13. 自测 {#zh-sec-25}

<details>
<summary>1. 为什么在带优化器的状态或累加器字段上，<code>dtype="auto"</code> 会自动选择 <code>float32</code> 而非更省显存的 <code>bfloat16</code>？</summary>

在自回归解码中，在线训练每步的梯度更新和滑动平均通常幅度微小。在 `bfloat16` 精度下，尾数仅有 7 位，微小的梯度增量或加权量在加到现有权重上时会因精度下溢而被直接截断归零（更新吞噬），导致模型无法学到任何变化甚至数值发散。实测表明，Carry 进位使用 `bfloat16` 的相对误差高达约 $$1.4 \times 10^{-1}$$，而使用 `float32` 仅为约 $$3.7 \times 10^{-4}$$。因此框架自动为需要微小累加的字段采用 `float32`，而对只读权重和只写前向字段保留模型精度（如 `bfloat16`）以节省显存。
</details>

<details>
<summary>2. 为什么在第 0 步读取 <code>carry</code> 字段时拿到的张量值始终是全零？</summary>

这是由 pLLM 的版本发布时序规则决定的：任何方法在某一步中读取状态时，看到的都是“当前步开始前已发布的版本”（即第 $$n$$ 步读取第 $$n-1$$ 步发布的结果）。而在请求准入的第 0 步开始时，该请求尚未执行过任何训练方法，不存在之前发布的版本，且 carry 字段在准入时默认进行零初始化，因此第 0 步读取到的数值必然是零。
</details>

<details>
<summary>3. 为什么读取 <code>RequestState</code> 的字段时张量自动带有领先的行轴（<code>[rows, *S]</code>），而读取 <code>InstanceState</code> 默认没有？</summary>

在连续批处理中，多个请求被拼接在同一个批次内并发执行。<code>RequestState</code> 为每个请求维护私有的参数副本，因此读取时框架自动根据当前操作所调度的行，为每一行请求对齐并切出其私有的参数切片，形成 <code>[rows, *S]</code> 的张量；而 <code>InstanceState</code> 在全实例范围内只有一份全局唯一的张量副本，不随并发请求数扩缩，因此默认保持声明时的原始维度 <code>[*S]</code>（若需参与按行批处理运算，可显式指定 <code>per_row=True</code> 展开广播行轴）。
</details>

</div>

</div>

<div class="zine-doc-footer">
  <a class="zine-pill-link" data-i18n-zh="← 返回 pLLM 项目与文档中心" data-i18n-en="← Back to pLLM Project &amp; Docs" href="/pllm/#docs">← Back to pLLM Project &amp; Docs</a>
  <a class="zine-pill-link" href="https://github.com/LinesHogan/pLLM/blob/main/doc/guides/state.md" target="_blank" rel="noopener">GitHub &middot; doc/guides/state.md ↗</a>
</div>

<script defer src="/assets/js/bilingual-post.js"></script>
