---
layout: page
title: "Editing Models: ActivationPatch, Adapter, and In-Forward Training"
description: "pLLM Documentation · User Guides"
permalink: /pllm/docs/guides/model-edits/
nav: false
---

<link rel="stylesheet" href="{{ '/assets/css/pllm-hub.css' | relative_url | bust_file_cache }}">

<div class="zine-dispatch" data-title-zh="改模型：激活补丁、Adapter 与前向内训练" data-title-en="Editing Models: ActivationPatch, Adapter, and In-Forward Training">
  <div class="zine-dispatch-links">
    <a class="zine-pill-link primary" data-i18n-zh="← 返回 pLLM 文档中心" data-i18n-en="← pLLM Docs Hub" href="/pllm/#docs">← pLLM Docs Hub</a>
    <a class="zine-pill-link" data-i18n-zh="文档导航地图" data-i18n-en="Docs Roadmap" href="/pllm/docs/">Docs Roadmap</a>
    <span class="zine-lang-badge" data-i18n-zh="用户指南" data-i18n-en="User Guides">User Guides</span>
  </div>
  <a class="zine-pill-link" data-i18n-zh="GitHub 源码 ↗" data-i18n-en="View on GitHub ↗" href="https://github.com/LinesHogan/pLLM/blob/main/doc/guides/model-edits.md" target="_blank" rel="noopener">View on GitHub ↗</a>
</div>

<div id="markdown-content">

<div class="zine-lang-panel" data-lang-panel="en" lang="en" markdown="1">

<div id="en-editing-models-activationpatch-adapter-and-in-forward-training"></div>

When prototyping algorithms in PyTorch or Hugging Face, modifying a model's activations or effective weights usually means registering a `forward_hook` or rewriting `linear.weight` directly. In a high-concurrency inference engine like vLLM, however, tokens from dozens of requests are flattened into a single 2D tensor by continuous batching, and the model's forward pass is captured into CUDA graphs—Python hooks cannot tell which row belongs to which request and do not even fire during CUDA graph replay, while mutating shared weight matrices breaks request isolation and causes memory hazards.

pLLM provides three declarative mechanisms for editing models so each request or instance can customize computation without mutating shared base weights:

1. [`pllm.model.ActivationPatch`](/pllm/docs/reference/api/#en-pllmmodelactivationpatch): adds activation corrections onto the residual stream after specified layers (e.g., activation steering, LoReFT), running entirely inside the model's CUDA graph.
2. [`pllm.model.Adapter`](/pllm/docs/reference/api/#en-pllmmodeladapter): applies weight deltas $$\Delta W$$ onto specified projection matrices (such as attention Q/K/V or MLP projections, e.g., per-request LoRA, dense deltas, per-head deltas), automatically isolated per request and addressed inside CUDA graphs.
3. [`@pllm.train_in_forward`](/pllm/docs/reference/api/#en-pllmtrain_in_forward): updates private state directly from a projection layer's inputs and targets *while the forward pass is running* (e.g., In-Place TTT fast weights), closing the train-during-forward loop.

---

## `ActivationPatch`: Patching the Residual Stream {#en-activationpatch-patching-the-residual-stream}

**Problem solved**: Applying an additive intervention or low-rank affine transform to the residual stream output immediately after certain layers—for representation steering or representation fine-tuning (LoReFT)—without slowing down inference.

Start with a minimal example: adding an externally supplied steering vector to layer 12's residual stream on every decode step:

```python
from dataclasses import dataclass

import torch

import pllm
from pllm import testing

CONFIG = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=8)


class Steering(pllm.Workload):
    steer = pllm.InstanceState({"vector": pllm.Field(lambda width, _: (width,))}, init=pllm.Imported())
    patch = pllm.model.ActivationPatch("steer", layers=(12,), operator="add", decode=True)


print(pllm.explain(Steering(), CONFIG).eager_reason)
```

`pllm.explain(Steering(), CONFIG).eager_reason` prints the empty string `""`, meaning `ActivationPatch` stays completely inside the model's CUDA graph and does not fall back to eager mode.

### Constructor Parameters and Operators {#en-constructor-parameters-and-operators}

The signature of `pllm.model.ActivationPatch` is:

```text
pllm.model.ActivationPatch(
    state: str,
    layers: tuple[int, ...],
    first_tokens: int = 0,
    last_tokens: int = 2,
    operator: Literal["low_rank_affine", "add"] = "low_rank_affine",
    *,
    decode: bool = False,
    refresh: Literal["admission", "step"] = "admission",
    on_missing: Literal["stop", "hold"] = "stop",
)
```

- **`state`**: name of the state holding intervention parameters (supports `InstanceState` or `RequestState`).
- **`layers`**: tuple of layer indices to intervene on (strictly ascending and unique); the intervention site is the residual stream after that layer (matching `hidden_states[layer + 1]` in Hugging Face outputs).
- **`operator`**:
  - `"add"`: pure additive intervention; requires one `(hidden_size,)` vector field per layer in the state and computes $$h \leftarrow h + v$$.
  - `"low_rank_affine"`: low-rank affine transform (LoReFT style); requires three fields per layer—weight matrix `(rank, hidden_size)`, bias `(rank,)`, and projection/rotation matrix `(hidden_size, rank)`—and computes $$h \leftarrow h + R^T (W h + b - R h)$$.
- **Token scope**:
  - `first_tokens` and `last_tokens`: apply only to the first $$k$$ and last $$m$$ tokens of the prompt (prefill phase).
  - `decode=True`: keyword-only flag specifying whether the patch remains active on every generated token during decode. If `decode=False`, specifying a non-default `refresh` or `on_missing` fails at build time.
- **Dynamic update policy**:
  - `refresh="admission"` (default): a request keeps reading the state version bound when it was admitted for its entire lifetime.
  - `refresh="step"`: switches to the latest published version on every decode step.
  - `on_missing="stop"` (default) vs. `"hold"`: used with external streaming input `pllm.io.Feed`. If no new input arrives on a step, `"hold"` reuses the last successfully applied version, whereas `"stop"` stops the request.

### Forward-Only KV: Past Tokens Are Never Rewritten {#en-forward-only-kv-past-tokens-are-never-rewritten}

`ActivationPatch` follows **Forward-only KV** semantics:
An activation edit injected at layer $$L$$ affects the current token in all subsequent layers ($$> L$$) and enters the K/V cache written by later attention layers, influencing future generated tokens; however, it **never retroactively modifies the K/V cache already written by past tokens at layer $$L$$ or earlier**.

Why not recompute past tokens' KV? Autoregressive decoding's core throughput advantage comes from reusing cached prefix KVs. If every activation tweak required updating past tokens' KV entries, per-step decode cost would degrade from $$O(1)$$ to $$O(T)$$ over the entire context, collapsing continuous-batching throughput.

---

## `Adapter`: Weight Corrections on Projections {#en-adapter-weight-corrections-on-projections}

**Problem solved**: Attaching per-request LoRA or parameter patches to specific linear projections (such as attention Q/K/V or MLP Gate/Up/Down projections) without duplicating the base model, and computing different requests' deltas concurrently within the same batch.

Consider `PromptLoRA`: on admission, each request loads its own rank-4 low-rank factors $$A$$ and $$B$$ via `pllm.io.StateImport`, attached to `q_proj` and `v_proj` at layer 12:

```python
@dataclass(frozen=True)
class PromptLoRA(pllm.Workload):
    rank: int = 4

    @property
    def lora(self) -> pllm.RequestState:
        return pllm.RequestState(
            {
                "a": pllm.Field(lambda inputs, outputs: (self.rank, inputs)),
                "b": pllm.Field(lambda inputs, outputs: (outputs, self.rank)),
            },
            init=pllm.Imported(),
        )

    @property
    def apply(self) -> pllm.model.Adapter:
        targets = (pllm.model.Projection("q_proj", 12), pllm.model.Projection("v_proj", 12))
        return pllm.model.Adapter("lora", targets, operator="low_rank")


print(pllm.explain(PromptLoRA(), CONFIG).eager_reason)
```

This also prints `""`: an `Adapter` applied to `RequestState` stays completely inside the CUDA graph.

```python
from vllm import SamplingParams

with pllm.make_llm(CONFIG, workload=PromptLoRA()) as llm:
    # Prepare LoRA factors externally per prompt: RequestState imports carry a leading prompt axis
    a_q = torch.randn(1, 4, 896) * 0.02  # q_proj: 896 in, 896 out
    b_q = torch.randn(1, 896, 4) * 0.02
    a_v = torch.randn(1, 4, 896) * 0.02  # v_proj: Qwen2.5-0.5B uses GQA, so output is 2 KV heads * 64 = 128
    b_v = torch.randn(1, 128, 4) * 0.02
    # Field order matches the two targets: a_0, b_0, a_1, b_1
    batch = pllm.io.StateImport((a_q, b_q, a_v, b_v))
    outputs = llm.generate(["Hello world!"], SamplingParams(max_tokens=16), batch=batch)
```

### Constructor Parameters and Target Naming {#en-constructor-parameters-and-target-naming}

`pllm.model.Adapter` has the following signature:

```text
pllm.model.Adapter(
    state: str,
    targets: tuple[pllm.model.Projection, ...],
    weight: str = "value",
    operator: Literal["dense", "low_rank", "per_head"] = "dense",
    factors: tuple[str, str] = ("a", "b"),
    scale: float = 1.0,
)
```

- **`targets`**: tuple of target projections. Each is declared via `pllm.model.Projection(name, layer=None)` using standard Hugging Face module names (`"q_proj"`, `"k_proj"`, `"v_proj"`, `"gate_proj"`, `"up_proj"`, `"down_proj"`); `layer=None` targets every layer of the model.
- **`operator`**:
  - `"low_rank"`: low-rank product $$\text{scale} \cdot (x A^T) B^T$$, reading the two fields named by `factors` (default `("a", "b")`).
  - `"dense"`: dense matrix delta $$\text{scale} \cdot x W_{\text{delta}}^T$$, reading the field named by `weight` (default `"value"`).
  - `"per_head"`: independent `(head_dim, head_dim)` block transform per attention head, reading the `weight` field.
- **Multi-target field expansion rule**:
  When one `Adapter` targets multiple `targets`, each `pllm.Field` shape callback `lambda in_features, out_features: (...)` in the state is invoked in turn with each target projection's input and output widths. Underlying private fields are automatically expanded and named in target order as `a_0`, `b_0`, `a_1`, `b_1`, ...

### Both `RequestState` and `InstanceState` Stay in CUDA Graphs {#en-both-requeststate-and-instancestate-stay-in-cuda-graphs}

- **`RequestState`**: each request owns a dedicated state page. Inside the CUDA graph, `Adapter` maps each batch row's request index to its private factor address and computes the delta without re-capturing graphs, **staying inside the CUDA graph**.
- **`InstanceState`**: all requests share one state (or snapshots pinned at request admission version). Inside the CUDA graph, `Adapter` indexes the bound state page via a unified version-slot table, also **staying inside the CUDA graph** without falling back to eager mode.
  (If `cuda_graphs="auto"` is active alongside another non-whole-graph feature such as `@pllm.custom_forward`, both `explain` and `llm.diagnostics().eager_reason` report that cause.)

### Why Trainable Low-Rank States Must Have Non-Zero Initialization {#en-why-trainable-low-rank-states-must-have-non-zero-initialization}

If a state has an optimizer (for example, online LoRA training) and uses `operator="low_rank"`, pLLM forbids the default `init=pllm.Zeros()` and requires explicit non-zero initialization (such as `init=pllm.Computed(...)`):

**Why**: The low-rank output is $$\Delta y = \text{scale} \cdot (x A^T) B^T$$. The gradient with respect to factor $$A$$ is proportional to $$B^T$$, and the gradient with respect to $$B$$ is proportional to $$A^T$$. If both factors start at 0, both gradients computed on the first backward pass are identically zero, and the optimizer can never move the parameters. The standard practice in `pllm.Computed` is to initialize $$A$$ with random Gaussian noise (e.g., std 0.02) and $$B$$ with zeros: this keeps the initial effective delta at 0 (preserving base model behavior) while giving $$B$$ non-zero gradients on the very first step.

### Behavior on Quantized Base Models {#en-behavior-on-quantized-base-models}

When using `Adapter` on AWQ, GPTQ, or FP8 quantized models:
- The base linear transformation still runs through its high-performance quantized kernel.
- The `Adapter`'s low-rank or dense delta is computed separately in high precision (FP16/BF16/FP32) and added as a **high-precision residual correction** onto the quantized kernel's output: $$y = \text{quant\_linear}(x) + \Delta W \cdot x$$.
- The base model's quantized weights are not unpacked. Only when a training method explicitly requests `pllm.operand("weight")` does the engine dequantize and materialize a high-precision dense copy of that projection's base weight, incurring extra VRAM usage.

### Tensor Parallelism (TP) Sharding Rules {#en-tensor-parallelism-tp-sharding-rules}

Under `tensor_parallel_size > 1` (vLLM 0.32+), `"dense"` and `"low_rank"` operators follow the base projection's exact communication sharding:
- **Column-parallel projections (`q_proj`, `k_proj`, `v_proj`, `gate_proj`, `up_proj`)**: sharded along output features. Dense weights are row-sharded; in low-rank deltas, $$B$$ is sharded along output features while $$A$$ is replicated across ranks.
- **Row-parallel projections (`down_proj`)**: sharded along input features. Dense weights are column-sharded; in low-rank deltas, $$A$$ is sharded along input features while $$B$$ is replicated across ranks.
- State must be initialized with `pllm.Zeros` or `pllm.Computed` and may only be updated shard-locally via `@pllm.train_in_forward` (training a replicated $$A$$ from column-sharded outputs or a replicated $$B$$ from row-sharded inputs across GPUs is disallowed).
- The `"per_head"` operator and states with optimizers or external imports (`Imported` / checkpoints) currently require `tensor_parallel_size=1`.

---

## Decoupled State Producers: Multiple Sources for One `Adapter` {#en-decoupled-state-producers-multiple-sources-for-one-adapter}

`pllm.model.Adapter` is purely a **consumer declaration**: it specifies only "which state to read and which projections to apply it to," completely agnostic to where the state values come from. This orthogonal decoupling of producer and consumer lets the exact same `Adapter` structure work seamlessly across different algorithmic patterns:

1. **External import (`StateImport`)**: each request writes hypernetwork-generated factors or preloaded LoRA weights at admission (see Tutorial [02 Per-Request LoRA](/pllm/docs/tutorials/02_request_adapters/)).
2. **Post-forward write (`@pllm.write`)**: a shared lightweight network in `InstanceState` reads the final prompt token's representation and writes the request's factors after the forward pass (e.g., Text-to-LoRA).
3. **Online distillation & training (`@pllm.train`)**: updates factors on the side stream using AdamW and downstream hidden states as supervision (e.g., TLM).
4. **In-forward training (`@pllm.train_in_forward`)**: updates factors in place inside a single model forward pass using projection operands.

The consumer side remains identical: regardless of which producer drives the state, as long as the state name and shapes match, `pllm.model.Adapter("lora", targets, operator="low_rank")` applies it automatically.

---

## `@pllm.train_in_forward`: Inline Training During the Forward Pass {#en-pllmtrain_in_forward-inline-training-during-the-forward-pass}

**Problem solved**: Online Test-Time Training (e.g., In-Place TTT) requires fast weights to update **inside the forward pass** as tokens stream through, so subsequent tokens or layers immediately execute under the updated weights.

Consider `PromptAdapters`: during prompt prefill, every 16 tokens (one chunk), the model uses the target projection's input to reconstruct the layer's MLP input and updates fast weights via SGD, verified on CPU with the contract checker:

```python
TARGETS = tuple(pllm.model.Projection("down_proj", layer) for layer in (0, 2))


def start_small(fields: dict[str, torch.Tensor], generator: torch.Generator) -> None:
    fields["a"].normal_(std=0.02, generator=generator)
    fields["b"].zero_()


@dataclass(frozen=True)
class PromptAdapters(pllm.Workload):
    rank: int = 4

    @property
    def adapters(self) -> pllm.RequestState:
        return pllm.RequestState(
            {
                "a": pllm.Field(lambda inputs, outputs: (self.rank, inputs)),
                "b": pllm.Field(lambda inputs, outputs: (outputs, self.rank)),
            },
            init=pllm.Computed(start_small),
            optimizer=pllm.SGD(1e-3),
        )

    @pllm.train_in_forward(state="adapters", targets=TARGETS, operator="low_rank", chunk=16, flush="prompt_end")
    def learn(
        self,
        x: torch.Tensor = pllm.operand("input"),
        target: torch.Tensor = pllm.operand("target"),
        base: torch.Tensor = pllm.operand("weight"),
        a: torch.Tensor = pllm.parameter("adapters", "a"),
        b: torch.Tensor = pllm.parameter("adapters", "b"),
    ) -> torch.Tensor:
        prediction = x @ base.T + (x @ a.to(x.dtype).T) @ b.to(x.dtype).T
        return (prediction.float() - target.float()).square().mean(-1)


print(testing.check(PromptAdapters(), CONFIG).passed)
```

`testing.check(PromptAdapters(), CONFIG).passed` simulates row permutation, request isolation, chunked scheduling, and preemption replay on CPU, returning the tuple of verified contracts.

### Operands and Parameter Constraints {#en-operands-and-parameter-constraints}

Inside a `@pllm.train_in_forward` method, input parameters use marker defaults:
- `pllm.operand("input")`: the input rows currently arriving at the target projection.
- `pllm.operand("target")`: the MLP input activations at that projection's layer (the standard target feature for In-Place TTT).
- `pllm.operand("weight")`: the projection's base weight matrix (materializes a dense copy on quantized models if requested).
- `pllm.operand("prompt_end")`: a 0-d boolean scalar tensor indicating whether the current chunk has reached the end of the prompt.
- `pllm.parameter(state, field)`: references a request-private state buffer (such as $$a, b$$, or $$\Delta W$$).

**Mutation rules**: tensors captured via `pllm.operand` are intermediate activations of the base forward pass and are **strictly read-only**; the only buffers permitted to be mutated in place are private parameter buffers referenced via `pllm.parameter`.

### Chunked SGD and Tail Flushing {#en-chunked-sgd-and-tail-flushing}

- **`chunk=`**: sets the token granularity of updates. When the state binds `optimizer=pllm.SGD(lr)`, the method returns a per-row loss tensor of shape `[rows]`, and the framework accumulates gradients until a full chunk of rows completes before stepping SGD.
- **`flush="prompt_end"`**: when a prompt's token length is not divisible by `chunk` (e.g., length 50 with chunk 16), the trailing 2 tokens would otherwise remain buffered. Setting `flush="prompt_end"` flushes an update at the prompt boundary so every prompt token participates in adaptation.
- **`tokens=`**: selects which tokens train the state; defaults to `tokens="prefill"` (train on prompt tokens, freeze during decode), and can also be `"decode"` or `"all"` (see [08 In-Place TTT](/pllm/docs/examples/08_inplace_ttt/)).

### Visibility Policies: `forward_pinned` vs. the Cost of `exact` {#en-visibility-policies-forward_pinned-vs-the-cost-of-exact}

- **`visibility="forward_pinned"` (default)**: the entire forward pass reads the state version pinned on entry; updates produced inside the forward are published for subsequent forward steps.
- **`visibility="exact"`**: within the *same* forward pass, the next chunk's projection computation **immediately sees the parameters updated by the preceding chunk**.
- **Cost and trade-off**: `"exact"` breaks full-batch parallelism across chunks, inserting chunk-level synchronization and parameter dependencies inside the forward stream and requiring private intermediate state buffers; unless your algorithm relies on intra-forward chunk recurrence (such as TTT), keep the default `"forward_pinned"`.

### Manual Updates: Returning `UpdateOutput` {#en-manual-updates-returning-updateoutput}

If you derive a closed-form update or custom rule instead of using framework SGD, omit the optimizer on the state and return `pllm.UpdateOutput` directly:

```python
targets = (pllm.model.Projection("down_proj", 0),)


class ManualFastWeights(pllm.Workload):
    factors = pllm.RequestState(
        {
            "a": pllm.Field(lambda in_features, out_features: (2, in_features)),
            "b": pllm.Field(lambda in_features, out_features: (out_features, 2)),
        }
    )

    @pllm.train_in_forward(state="factors", targets=targets, operator="low_rank", factors=("a", "b"), scale=1.0)
    def update(
        self,
        x: torch.Tensor = pllm.operand("input"),
        y: torch.Tensor = pllm.operand("target"),
        a: torch.Tensor = pllm.parameter("factors", "a"),
        b: torch.Tensor = pllm.parameter("factors", "b"),
    ) -> pllm.UpdateOutput:
        residual = (x @ a.to(x.dtype).T) @ b.to(x.dtype).T  # return only the delta added onto base output
        a.add_((0.1 * y[:, : a.shape[0]].T @ x).to(a))      # update private parameters in place
        b.add_((0.1 * (y.T @ y)[:, : b.shape[1]]).to(b))
        return pllm.UpdateOutput(
            residual,
            residual.new_ones((), dtype=torch.bool),              # changed: whether state changed
            residual.new_ones((), dtype=torch.int64),             # update_count: updates applied
            residual.new_full((), x.shape[0], dtype=torch.int64), # trained_rows: rows trained
        )


print(pllm.explain(ManualFastWeights(), CONFIG).eager_reason)
```

`UpdateOutput` holds four fields: `residual` is the delta tensor added to the base projection output, and the remaining three are 0-d device tensors on the same device as the output, ensuring zero GPU-to-CPU synchronization during the forward pass.

---

## Using `Adapter` Inside `@pllm.custom_forward`: Deltas Follow Every Projection Call {#en-using-adapter-inside-pllmcustom_forward-deltas-follow-every-projection-call}

When overriding the model's layer loop with [`@pllm.custom_forward`](/pllm/docs/tutorials/04_custom_forward/) (e.g., looped depth, dynamic layer skipping, or layer replay), the standard forward assumption that "every layer runs once per token" no longer holds. An `Adapter` attached to the model follows this core rule:

> **Core rule: weight corrections follow every actual invocation of the target projection, using that invocation's actual inputs and each row's request state.**

1. **Triggered per invocation**: whenever a target projection actually executes—whether in a normal layer pass, repeated iterations in looped depth, or replay—`Adapter` applies a weight delta using that invocation's actual activations. If a slice of layers loops 3 times in looped depth, its target projections receive the delta on all 3 passes, matching the physical semantics of running 3 passes with the adapted weights.
2. **State ownership pinned at forward entry**: which request state each row maps to is anchored on forward entry. Even if `@pllm.custom_forward` compacts rows with `active=` or reorders requests into waves, underlying device indices preserve each row's request ownership so every row addresses its own private state.
3. **Skipped projections get nothing; propagated projections get theirs**: projections that are skipped never run and receive no delta; however, if layer skipping triggers native QKV propagation to populate K/V caches, the Q/V projections that actually run still receive their weight deltas.
4. **Model adaptation decoupled from depth schedule**: the exact same `Adapter`-equipped model can run under the standard forward pass or be reused verbatim across different `@pllm.custom_forward` schedules without rewriting adapter logic.
5. **Limitation**: `Adapter` represents a physical weight change to that projection and has no per-loop-iteration on/off switch. If an algorithm needs a correction "only on the first loop pass and not subsequent passes," that is a hidden-state intervention and should be applied explicitly in Python to the first pass's hidden tensor inside `@pllm.custom_forward`, rather than using `Adapter`.

---

## Common Errors and Runtime Constraints: Preemption, Recomputation, and Prefix Caching {#en-common-errors-and-runtime-constraints-preemption-recomputation-and-prefix-caching}

When deploying model-editing algorithms, concurrent scheduling introduces a few boundary constraints.

### Recomputation After Preemption {#en-recomputation-after-preemption}

When vLLM runs out of GPU memory for new KV blocks, the scheduler triggers **preemption**: it evicts low-priority requests' KV caches and later recomputes those requests' histories from the start of their prompts once memory frees up.

- **Workloads that recompute seamlessly**:
  If request state is keyed by `request_id` (`key_fields` includes `request_id`), and the workload has **no mutable instance state** and **no external feeds, imports, or host inputs**, replaying the prompt history reproduces the exact state, so the engine recomputes directly. The `preempt` line of `pllm.explain` states which category your workload falls into.
- **Workloads that fail with a runtime error on preemption**:
  Any overlay or activation-editing workload that depends on external per-step inputs, mutable `InstanceState`, or cross-step state dependencies **cannot guarantee identical state upon recomputation** when preempted; the engine raises a runtime error instead. See [Requests and run time](/pllm/docs/reference/errors/#en-requests-and-run-time) in the Error Catalog for exact messages and fixes.

> **Historical KV retention**: in causal autoregressive models, already-written KV cache entries reflect the effective weights at the time they were computed; later updates to `Adapter` factors do not retroactively recompute prefix KV entries.

### Why Prefix Caching Is Mutually Exclusive with Overlays and Activation Patches {#en-why-prefix-caching-is-mutually-exclusive-with-overlays-and-activation-patches}

In vLLM, `enable_prefix_caching=True` shares KV blocks across requests that share identical prompt prefixes.
**In pLLM, overlays (`Adapter`, `train_in_forward`) and `ActivationPatch` are strictly mutually exclusive with Prefix Caching (except immutable `InstanceState` patches with `last_tokens=0`) and are rejected at build time if combined.**

**Why**:
Prefix Caching assumes "identical token sequences produce identical KV tensors." However, per-request LoRA weights, step-updated fast weights, and layer-wise activation steering all change the mathematical computation of the forward pass:
Two requests with the exact same prompt text but different LoRA factors produce completely different KV caches; sharing prefix blocks would feed the second request activations computed under the first request's weights.
Therefore, pLLM enforces `enable_prefix_caching=False` whenever a workload alters model activations or projection weights in a request-dependent or mutable way.

---

## Self-Check {#en-self-check}

<details>
<summary>1. Why cannot a trainable low-rank LoRA state use the default init=pllm.Zeros()?</summary>

The low-rank delta is $$\Delta y = \text{scale} \cdot (x A^T) B^T$$. The partial derivative of the loss with respect to $$A$$ is multiplied by $$B^T$$, and the derivative with respect to $$B$$ is multiplied by $$A^T$$. If both $$A$$ and $$B$$ start at zero, both first-step gradients are mathematically zero, so the optimizer can never move the parameters. Non-zero initialization (such as random Gaussian $$A$$ and zero $$B$$) is required.
</details>

<details>
<summary>2. Why does an ActivationPatch at layer 12 not alter the layer-12 KV cache of past tokens?</summary>

Autoregressive decoding follows Forward-only KV semantics: past tokens' KV entries were already computed and committed during the forward pass that processed those tokens. Updating past KV caches to reflect new activation edits would require re-running the forward pass over the entire prefix on every generation step, turning $$O(1)$$ per-step decode into $$O(T)$$ and destroying concurrent throughput.
</details>

<details>
<summary>3. Does attaching pllm.model.Adapter to an InstanceState force eager mode?</summary>

No. Whether bound to `RequestState` or a shared `InstanceState`, `pllm.model.Adapter` resolves deltas via a graph-capturable overlay table indexed by row or version slot, staying inside vLLM's CUDA graphs. Whole-model CUDA graphs are omitted only when explicitly disabled (`cuda_graphs=False`) or when combined with a sliced-graph feature such as `@pllm.custom_forward`.
</details>

</div>

<div class="zine-lang-panel is-hidden" data-lang-panel="zh" lang="zh-CN" markdown="1">

<div id="zh-sec-1"></div>

在 PyTorch 或 Hugging Face 里调试算法时，修改模型的激活值或有效权重通常只需要注册一个 `forward_hook`，或者直接改写 `linear.weight`。但在 vLLM 这类高并发推理引擎中，几十个请求的 token 被连续批处理（continuous batching）打平在同一个二维张量中，模型的前向传播也被捕获进了 CUDA graph——Python 层的 hook 既无法区分哪一行张量属于哪个请求，在 CUDA graph 重放时更不会被触发；直接修改共享的权重矩阵则会破坏请求隔离并导致显存冲突。

pLLM 提供了三种声明式的改模型机制，在不改动共享底座权重的前提下，让每个请求或全局实例按需定制计算：

1. [`pllm.model.ActivationPatch`](/pllm/docs/reference/api/#zh-pllmmodelactivationpatch)：在指定层之后的残差流上叠加激活修正（例如激活引导 Steering、LoReFT），完全运行在模型的 CUDA graph 内。
2. [`pllm.model.Adapter`](/pllm/docs/reference/api/#zh-pllmmodeladapter)：在指定的投影矩阵（如注意力 Q/K/V 或 MLP 投影）上叠加权重增量 $$\Delta W$$（例如每请求 LoRA、稠密增量、多头增量），自动按请求隔离并在 CUDA graph 内寻址。
3. [`@pllm.train_in_forward`](/pllm/docs/reference/api/#zh-pllmtrain_in_forward)：在前向传播运行的过程中，直接利用投影层的输入与目标更新私有状态（例如 In-Place TTT 快权重），实现边前向边训练的闭环。

---

## ActivationPatch：在残差流上打补丁 {#zh-sec-2}

**要解决的问题**：在某些层计算完成后，直接对其输出的残差流（residual stream）做加性干预或低秩仿射变换，用于表征引导（Steering）或表征微调（LoReFT），且不能拖慢推理速度。

先看一个最小示例：在第 12 层的残差流上叠加一个外部传入的引导向量，并在解码阶段的每个生成 step 持续生效：

```python
from dataclasses import dataclass

import torch

import pllm
from pllm import testing

CONFIG = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=8)


class Steering(pllm.Workload):
    steer = pllm.InstanceState({"vector": pllm.Field(lambda width, _: (width,))}, init=pllm.Imported())
    patch = pllm.model.ActivationPatch("steer", layers=(12,), operator="add", decode=True)


print(pllm.explain(Steering(), CONFIG).eager_reason)
```

调用 `pllm.explain(Steering(), CONFIG).eager_reason` 输出为空字符串 `""`，这表明 `ActivationPatch` 完全留在模型的 CUDA graph 中执行，不会让前向传播退回 eager 模式。

### 声明参数与算子 {#zh-sec-3}

`pllm.model.ActivationPatch` 的构造参数如下：

```text
pllm.model.ActivationPatch(
    state: str,
    layers: tuple[int, ...],
    first_tokens: int = 0,
    last_tokens: int = 2,
    operator: Literal["low_rank_affine", "add"] = "low_rank_affine",
    *,
    decode: bool = False,
    refresh: Literal["admission", "step"] = "admission",
    on_missing: Literal["stop", "hold"] = "stop",
)
```

- **`state`**：存放干预参数的状态名称（支持 `InstanceState` 或 `RequestState`）。
- **`layers`**：要干预的层号元组（必须升序且唯一），干预点位于该层之后的残差流（对应 Hugging Face 输出中的 `hidden_states[layer + 1]`）。
- **`operator`**：
  - `"add"`：纯加法干预，状态中每层需要一个形状为 `(hidden_size,)` 的向量字段，执行 $$h \leftarrow h + v$$。
  - `"low_rank_affine"`：低秩仿射变换（LoReFT 风格），状态中每层需要三个字段：权重矩阵 `(rank, hidden_size)`、偏置 `(rank,)` 与旋转矩阵 `(hidden_size, rank)`，执行 $$h \leftarrow h + R^T (W h + b - R h)$$。
- **Token 作用范围**：
  - `first_tokens` 与 `last_tokens`：仅对 prompt（prefill 阶段）的前 $$k$$ 个 token 和后 $$m$$ 个 token 生效。
  - `decode=True`：关键字参数，声明该补丁是否对生成阶段（decode）的每一个 token 持续生效。若 `decode=False`，声明非默认的 `refresh` 或 `on_missing` 会在构建时直接报错。
- **动态更新策略**：
  - `refresh="admission"`（默认）：请求在整个生命周期内固定读取它准入时绑定的状态版本。
  - `refresh="step"`：每个 decode 步都切换读取当步发布的最新版本。
  - `on_missing="stop"`（默认）与 `"hold"`：配合外部流式输入 `pllm.io.Feed` 使用。若某一 step 没有新的输入到达，`"hold"` 沿用上一步成功使用的版本，`"stop"` 则停下该请求。

### Forward-only KV：旧 token 不会回改 {#zh-sec-4}

`ActivationPatch` 遵循 **Forward-only KV** 语义：
在第 $$L$$ 层注入的激活变化，会影响当前 token 在第 $$L$$ 层之后的后续网络层（$$> L$$），并伴随后续注意力层写入 K/V 缓存，从而对未来生成的 token 产生影响；但它**绝不会回改已计算完成的旧 token 在第 $$L$$ 层及之前已固化的 KV 缓存**。

为什么不重新计算旧 token 的 KV？因为自回归解码的核心吞吐优势在于重用前缀缓存。若每次激活微调都要求旧 token 的 KV 随之更新，解码单步的开销就会从 $$O(1)$$ 退化为重新遍历整个上下文的 $$O(T)$$，使连续批处理的吞吐彻底崩溃。

---

## Adapter：投影矩阵上的权重修正 {#zh-sec-5}

**要解决的问题**：在不复制整个底座模型的前提下，为特定投影层（例如注意力的 Q/K/V 投影或 MLP 的 Gate/Up/Down 投影）挂载每请求独立的 LoRA 或参数补丁，并在同一个 batch 中并发计算不同请求的增量。

下面以 `PromptLoRA` 为例：每个请求在准入时通过 `pllm.io.StateImport` 载入属于自己的秩为 4 的低秩矩阵 $$A$$ 与 $$B$$，挂载在第 12 层的 `q_proj` 与 `v_proj` 上：

```python
@dataclass(frozen=True)
class PromptLoRA(pllm.Workload):
    rank: int = 4

    @property
    def lora(self) -> pllm.RequestState:
        return pllm.RequestState(
            {
                "a": pllm.Field(lambda inputs, outputs: (self.rank, inputs)),
                "b": pllm.Field(lambda inputs, outputs: (outputs, self.rank)),
            },
            init=pllm.Imported(),
        )

    @property
    def apply(self) -> pllm.model.Adapter:
        targets = (pllm.model.Projection("q_proj", 12), pllm.model.Projection("v_proj", 12))
        return pllm.model.Adapter("lora", targets, operator="low_rank")


print(pllm.explain(PromptLoRA(), CONFIG).eager_reason)
```

输出同样为空字符串 `""`：对 `RequestState` 施加的 `Adapter` 完全驻留在 CUDA graph 内。

```python
from vllm import SamplingParams

with pllm.make_llm(CONFIG, workload=PromptLoRA()) as llm:
    # 外部为每个 prompt 准备一份 LoRA 因子：RequestState 的导入值带一个前导的 prompt 轴
    a_q = torch.randn(1, 4, 896) * 0.02  # q_proj：输入 896，输出 896
    b_q = torch.randn(1, 896, 4) * 0.02
    a_v = torch.randn(1, 4, 896) * 0.02  # v_proj：Qwen2.5-0.5B 用 GQA，输出只有 2 个 KV 头 × 64 = 128
    b_v = torch.randn(1, 128, 4) * 0.02
    # 字段顺序对应两个 target：a_0, b_0, a_1, b_1
    batch = pllm.io.StateImport((a_q, b_q, a_v, b_v))
    outputs = llm.generate(["Hello world!"], SamplingParams(max_tokens=16), batch=batch)
```

### 声明参数与目标命名 {#zh-sec-6}

`pllm.model.Adapter` 的构造签名如下：

```text
pllm.model.Adapter(
    state: str,
    targets: tuple[pllm.model.Projection, ...],
    weight: str = "value",
    operator: Literal["dense", "low_rank", "per_head"] = "dense",
    factors: tuple[str, str] = ("a", "b"),
    scale: float = 1.0,
)
```

- **`targets`**：目标投影元组。投影名称由 `pllm.model.Projection(name, layer=None)` 声明，可选名称为 Hugging Face 标准的 `"q_proj"`、`"k_proj"`、`"v_proj"`、`"gate_proj"`、`"up_proj"`、`"down_proj"`；`layer=None` 表示对模型所有层生效。
- **`operator`**：
  - `"low_rank"`：低秩乘积 $$\text{scale} \cdot (x A^T) B^T$$，读取 `factors`（默认 `("a", "b")`）命名的两个字段。
  - `"dense"`：稠密矩阵加法 $$\text{scale} \cdot x W_{\text{delta}}^T$$，读取 `weight`（默认 `"value"`）命名的字段。
  - `"per_head"`：按注意力头独立施加形状为 `(head_dim, head_dim)` 的块变换，读取 `weight` 字段。
- **多目标字段展开规则**：
  当一个 `Adapter` 作用于多个 `targets` 时，状态中的 `pllm.Field` 形状回调 `lambda in_features, out_features: (...)` 会按目标投影的输入与输出宽度被依次调用。底层分配的私有字段将按照 targets 列表的顺序自动展开并命名为 `a_0`, `b_0`, `a_1`, `b_1`, ...。

### `RequestState` 与 `InstanceState` 都能进图 {#zh-sec-7}

- **`RequestState`**：每个请求独享一份状态页。`Adapter` 在 CUDA graph 内部通过 batch 中每一行所属的请求索引找到对应的私有因子地址并计算增量，全程无需重新捕获图，**留在 CUDA graph 内**。
- **`InstanceState`**：所有请求共享同一份状态（或按请求准入版本隔离的状态快照）。`Adapter` 通过统一的版本槽表在 CUDA graph 内部直接索引当前步骤绑定的状态页面，同样**留在 CUDA graph 内**，不会触发回退到 eager 模式。
  （在开启 `cuda_graphs="auto"` 且存在其他不可进图写法如 `@pllm.custom_forward` 时，`explain` 与 `llm.diagnostics().eager_reason` 会给出对应原因。）

### 为什么可训练的低秩状态必须非零初始化 {#zh-sec-8}

如果状态带有优化器（例如在线训练 LoRA），且算子为 `"low_rank"`，pLLM 禁止使用默认的 `init=pllm.Zeros()`，必须显式传入非零初值（例如 `init=pllm.Computed(...)`）：

**原因**：低秩输出为 $$\Delta y = \text{scale} \cdot (x A^T) B^T$$。损失函数对因子 $$A$$ 的梯度正比于 $$B^T$$，对 $$B$$ 的梯度正比于 $$A^T$$。如果两个因子初始全为 0，那么第一步反向传播计算出的两个梯度的数值严格恒等于 0，优化器永远无法更新参数。常见的做法是在 `pllm.Computed` 中将 $$A$$ 随机高斯初始化（例如标准差 0.02），将 $$B$$ 置零：这既保证了初始时的有效增量为 0（不破坏基座表现），又保证了反向传播时 $$B$$ 能够获得非零梯度。

### 量化基座模型上的行为 {#zh-sec-9}

在 AWQ、GPTQ、FP8 等量化模型上使用 `Adapter` 时：
- 底座的线性变换依然使用原本的高性能量化算子计算。
- `Adapter` 的低秩或稠密增量以高精度（FP16/BF16/FP32）独立计算后，作为**高精度残差修正**加在量化算子的输出上：$$y = \text{quant\_linear}(x) + \Delta W \cdot x$$。
- 底座模型本身的量化权重不会解包。只有当你的训练方法显式声明了 `pllm.operand("weight")` 时，引擎才会为该投影解量化并常驻一份高精度的稠密底座权重副本，带来额外的显存占用。

### 张量并行（TP）分片规则 {#zh-sec-10}

在 `tensor_parallel_size > 1`（vLLM 0.32）下，`"dense"` 与 `"low_rank"` 算子严格遵循底座投影的通信切分：
- **列并行投影（q_proj, k_proj, v_proj, gate_proj, up_proj）**：按输出特征切分。稠密权重按行切分；低秩补丁中 $$B$$ 按输出维度分片，$$A$$ 在各卡间复制。
- **行并行投影（down_proj）**：按输入特征切分。稠密权重按列切分；低秩补丁中 $$A$$ 按输入维度分片，$$B$$ 在各卡间复制。
- 状态必须由 `pllm.Zeros` 或 `pllm.Computed` 初始化，且只能由 `@pllm.train_in_forward` 在单卡本地分片上更新（禁止跨卡从列切分输出反向训练复制的 $$A$$，或从行切分输入反向训练复制的 $$B$$）。
- `"per_head"` 算子以及携带优化器或其他外部导入（`Imported` / Checkpoint）的状态目前要求 `tensor_parallel_size=1`。

---

## 独立声明状态生产者：同一份 Adapter 的多种来源 {#zh-sec-11}

`pllm.model.Adapter` 只是一个**消费者声明**，它只负责回答“从哪个状态读取数据并施加给哪些投影”，完全不关心状态内部的数据从何而来。这种生产者与消费者的正交解耦，使得同一份 Adapter 结构可以无缝适配多种算法演进：

1. **外部导入（StateImport）**：每个请求在准入时直接写入外部超网络产出的因子或预加载的 LoRA 权重（参见教程 [02 每请求 LoRA](/pllm/docs/tutorials/02_request_adapters/)）。
2. **前向后写入（@pllm.write）**：由驻留在 `InstanceState` 中的共享轻量网络读取当前请求 prompt 最后一个 token 的表征，在前向结束后写入请求的因子（例如 Text-to-LoRA）。
3. **在线蒸馏与训练（@pllm.train）**：在旁路 stream 上利用后层隐藏状态作为监督信号，通过 AdamW 优化器更新因子（例如 TLM）。
4. **前向内训练（@pllm.train_in_forward）**：在模型推理的单次前向内直接利用操作数就地更新因子。

消费端代码保持完全一致：无论由哪种生产者驱动，只要状态名称与形状匹配，`pllm.model.Adapter("lora", targets, operator="low_rank")` 就能在底层自动应用。

---

## @pllm.train_in_forward：前向过程中的内联训练 {#zh-sec-12}

**要解决的问题**：在线测试期训练（Test-Time Training / In-Place TTT）要求快权重在**前向传播的内部**根据当前已流过的 token 立即更新，使得后续的 token 或后续层能立即在修正后的权重下执行。

下面以 `PromptAdapters` 为例：在 prompt prefill 阶段，模型每经过 16 个 token（一个 chunk），就利用当前层的输入重建该层 MLP 的输入，并通过 SGD 更新快权重，最后在 CPU 上用契约检查器验证正确性：

```python
TARGETS = tuple(pllm.model.Projection("down_proj", layer) for layer in (0, 2))


def start_small(fields: dict[str, torch.Tensor], generator: torch.Generator) -> None:
    fields["a"].normal_(std=0.02, generator=generator)
    fields["b"].zero_()


@dataclass(frozen=True)
class PromptAdapters(pllm.Workload):
    rank: int = 4

    @property
    def adapters(self) -> pllm.RequestState:
        return pllm.RequestState(
            {
                "a": pllm.Field(lambda inputs, outputs: (self.rank, inputs)),
                "b": pllm.Field(lambda inputs, outputs: (outputs, self.rank)),
            },
            init=pllm.Computed(start_small),
            optimizer=pllm.SGD(1e-3),
        )

    @pllm.train_in_forward(state="adapters", targets=TARGETS, operator="low_rank", chunk=16, flush="prompt_end")
    def learn(
        self,
        x: torch.Tensor = pllm.operand("input"),
        target: torch.Tensor = pllm.operand("target"),
        base: torch.Tensor = pllm.operand("weight"),
        a: torch.Tensor = pllm.parameter("adapters", "a"),
        b: torch.Tensor = pllm.parameter("adapters", "b"),
    ) -> torch.Tensor:
        prediction = x @ base.T + (x @ a.to(x.dtype).T) @ b.to(x.dtype).T
        return (prediction.float() - target.float()).square().mean(-1)


print(testing.check(PromptAdapters(), CONFIG).passed)
```

`testing.check(PromptAdapters(), CONFIG).passed` 在 CPU 上完整模拟了行置换、请求隔离、分块调度和抢占重放，返回验证通过的契约列表。

### 操作数（Operands）与参数约束 {#zh-sec-13}

在 `@pllm.train_in_forward` 方法体内，参数通过默认值声明输入来源：
- `pllm.operand("input")`：目标投影算子当前收到的输入行。
- `pllm.operand("target")`：当前投影所属层 MLP 的输入激活值（标准 In-Place TTT 的目标特征）。
- `pllm.operand("weight")`：该投影的底座权重矩阵（量化模型上会按需物化稠密副本）。
- `pllm.operand("prompt_end")`：一个 0 维布尔标量张量，指示当前 chunk 是否已经到达 prompt 的末尾。
- `pllm.parameter(state, field)`：引用请求私有的状态缓冲区（如 $$a, b$$ 或 $$\Delta W$$）。

**修改权限**：`pllm.operand` 捕获的张量是底座前向的中间激活，**严格只读**；代码中唯一允许原地修改（in-place）的只有 `pllm.parameter` 授权的私有参数缓冲区。

### 分块优化（Chunked SGD）与尾部刷新 {#zh-sec-14}

- **`chunk=`**：设置更新的 token 粒度。当状态绑定了 `optimizer=pllm.SGD(lr)` 时，方法只需返回一个逐行 loss 张量（形状为 `[rows]`），框架会在累积满一个 chunk 的行数后自动执行一次 SGD 步进。
- **`flush="prompt_end"`**：当 prompt 的总 token 数不能被 `chunk` 整除时（例如长度为 50，chunk 为 16），末尾余下的 2 个 token 通常会被丢弃。设置 `flush="prompt_end"` 后，框架会在遇到 prompt 边界时就地触发一次更新，确保所有 token 都参与适应。
- **`tokens=`**：指定哪些阶段参与更新，默认为 `tokens="prefill"`（只在 prompt 阶段训练，生成阶段冻结使用），亦可设为 `"decode"` 或 `"all"`（参见 [08 In-Place TTT](/pllm/docs/examples/08_inplace_ttt/)）。

### 可见性策略：forward_pinned 与 exact 的代价 {#zh-sec-15}

- **`visibility="forward_pinned"`（默认）**：整趟前向传播自始至终读取进入该步时固定的状态版本；前向内产生的更新只发布给后续的前向步骤读取。
- **`visibility="exact"`**：同一个前向传播内部，后一个 chunk 的投影计算能够**立即看到前一个 chunk 刚更新出来的最新参数**。
- **代价与权衡**：`"exact"` 打破了整批并行的独立性，在前向计算流内部插入了 chunk 级别的同步与参数依赖，需要框架维护私有的中间状态缓冲区；除非算法显式依赖块间递推（如 TTT），否则建议使用默认的 `"forward_pinned"`。

### 手动计算更新：返回 UpdateOutput {#zh-sec-16}

如果不使用框架的 SGD 优化器，而是自己推导了封闭解或特定算子，可以不给状态指定优化器，方法直接返回 `pllm.UpdateOutput`：

```python
targets = (pllm.model.Projection("down_proj", 0),)


class ManualFastWeights(pllm.Workload):
    factors = pllm.RequestState(
        {
            "a": pllm.Field(lambda in_features, out_features: (2, in_features)),
            "b": pllm.Field(lambda in_features, out_features: (out_features, 2)),
        }
    )

    @pllm.train_in_forward(state="factors", targets=targets, operator="low_rank", factors=("a", "b"), scale=1.0)
    def update(
        self,
        x: torch.Tensor = pllm.operand("input"),
        y: torch.Tensor = pllm.operand("target"),
        a: torch.Tensor = pllm.parameter("factors", "a"),
        b: torch.Tensor = pllm.parameter("factors", "b"),
    ) -> pllm.UpdateOutput:
        residual = (x @ a.to(x.dtype).T) @ b.to(x.dtype).T  # 只返回叠加在底座上的残差
        a.add_((0.1 * y[:, : a.shape[0]].T @ x).to(a))      # 原地更新私有参数
        b.add_((0.1 * (y.T @ y)[:, : b.shape[1]]).to(b))
        return pllm.UpdateOutput(
            residual,
            residual.new_ones((), dtype=torch.bool),              # changed: 是否产生更新
            residual.new_ones((), dtype=torch.int64),             # update_count: 更新步数计数
            residual.new_full((), x.shape[0], dtype=torch.int64), # trained_rows: 参与训练的行数
        )


print(pllm.explain(ManualFastWeights(), CONFIG).eager_reason)
```

`UpdateOutput` 包含四个字段：`residual` 为加到底座输出上的增量张量；其余三个均为与输出同设备的 0 维张量计数器，确保整个前向过程不需要任何 GPU 到 CPU 的主机同步。

---

## 在 @pllm.custom_forward 里使用 Adapter：修正跟着每一次投影调用走 {#zh-sec-17}

在使用 [`@pllm.custom_forward`](/pllm/docs/tutorials/04_custom_forward/) 重写模型的层循环逻辑时（例如循环深度 looped depth、动态跳层或层重放），标准前向中“每一层每个 token 恰好执行一次”的假设被打破了。此时挂载在模型上的 `Adapter` 遵循以下核心规则：

> **核心原则：权重修正跟着每一次实际的投影算子调用走，按当次调用的实际输入计算，并按每一行所属请求的状态施加。**

1. **按次触发，有调用就有修正**：只要目标投影被实际执行——无论是常规层调用、循环深度中的重复调用，还是重放——`Adapter` 都会基于当次调用的实际激活值施加一次权重增量。如果某几层在循环深度中被循环执行了 3 遍，那么这几层的目标投影就会被修正 3 次。这在物理语义上等价于“带着被修正后的权重跑了 3 圈”。
2. **状态归属进入前向时固定**：每一行对应哪个请求的状态，在进入 forward 时就已经锚定。即便在 `@pllm.custom_forward` 里通过 `active=` 压缩了活跃行，或者将请求重排成不同的 wave，底层索引始终保留每行的请求归属，确保每行计算时准确索引到自己请求的私有状态。
3. **跳过的不加，补写的照加**：完全跳过的投影不会发生调用，自然不施加修正；但在跳层执行时，如果触发了原生 QKV 传播来补齐 K/V 缓存，实际运行的 Q/V 投影依然会加上各自的权重增量。
4. **模型与深度调度解耦**：同一个带有 `Adapter` 的适配模型，既可以直接跑原生前向，也可以原封不动地交给不同的 `@pllm.custom_forward` 调度逻辑复用，无需为自定义控制流专门重写适配代码。
5. **限制**：`Adapter` 表达的是“该投影的物理权重改变了”，没有按循环圈数开关的选项。如果算法要求“仅在第一圈施加修正，后续各圈不再生效”，这属于隐藏层激活干预，应当在 `@pllm.custom_forward` 中通过 Python 代码显式加在第一圈的 hidden 张量上，而非使用 `Adapter`。

---

## 常见错误与运行约束：抢占、重算与前缀缓存 {#zh-sec-18}

在将模型修改类算法推向实际部署时，底层的并发调度会引入一些边界约束。

### 抢占（Preemption）后的重算行为 {#zh-sec-19}

当 vLLM 的 GPU 显存用尽且无法容纳新的 KV block 时，调度器会触发**抢占（preemption）**：驱逐某些低优先级请求的 KV cache，待显存空闲后再重新计算这些请求从 prompt 起始的历史。

- **能够无缝重算的负载**：
  若请求状态以 `request_id` 为键（`key_fields` 包含 `request_id`），负载中**没有可变的实例状态**，也**没有外部 feed、导入或宿主输入**，重算历史就能完全复现原来的状态，引擎直接重算。`pllm.explain` 的 `preempt` 行会写明你的负载属于哪一种。
- **会直接报错中断的负载**：
  任何依赖外部逐步输入、含有可变 `InstanceState`、或存在跨步状态依赖的 overlay / 激活修改负载，在发生抢占时**无法保证重算状态与历史严格一致**，此时引擎会直接抛出运行时错误，原文和改法见报错目录的[请求与运行时](/pllm/docs/reference/errors/#zh-sec-12)一节。

> **历史 KV 的因子保留**：在因果自回归模型中，已写入的旧 KV 缓存保留的是计算当时所使用的有效权重；后续 step 更新 Adapter 因子不会回溯重算历史前缀的 KV。

### 为什么 Prefix Caching 与 Overlay / 激活补丁互斥 {#zh-sec-20}

在 vLLM 配置中，`enable_prefix_caching=True` 会在不同的请求之间共享具有相同前缀文本的 KV 块。
**在 pLLM 中，所有 Overlay（Adapter、train_in_forward）与 ActivationPatch 均与 Prefix Caching 严格互斥，组合时会在构建期直接拒绝。**

**原因**：
Prefix Caching 的基本假设是“相同的 token 序列产生完全相同的 KV 张量”。但无论是每请求的 LoRA 权重，还是逐步更新的快权重，亦或是逐层注入的激活引导，都会改变前向传播的实际数学计算：
1. 具有相同 prompt 文本的两个请求，若挂载了不同的 LoRA 因子，其产生的 KV 缓存是完全不同的；若强行共享前缀缓存，后一个请求就会直接读取前一个请求在错误权重下生成的激活值。
因此，pLLM 采取安全关闭的原则：凡是改写模型激活或投影权重的负载，必须配置 `enable_prefix_caching=False`。

---

## 自测 {#zh-sec-21}

<details>
<summary>1. 为什么在带优化器的低秩 LoRA 状态中，不能使用默认的 init=pllm.Zeros()？</summary>

低秩矩阵增量计算公式为 $$\Delta y = \text{scale} \cdot (x A^T) B^T$$。损失对 $$A$$ 的偏导数乘有因子 $$B^T$$，对 $$B$$ 的偏导数乘有因子 $$A^T$$。若 $$A$$ 和 $$B$$ 均初始化为全 0，则两个矩阵的第一步梯度数学上严格恒为 0，优化器永远无法学动参数。必须使用非零初始化（如 $$A$$ 高斯随机初始化，$$B$$ 置 0）。
</details>

<details>
<summary>2. 为什么 ActivationPatch 对第 12 层的修改不会改变旧 token 在第 12 层的 KV cache？</summary>

自回归解码遵循 Forward-only KV 机制：旧 token 的 KV 缓存在处理该 token 的当步前向时就已经计算并落盘。若要让过去的 KV 缓存也反映新的激活干预，引擎必须在每个生成步骤重新执行一遍整个历史前缀的前向传播，这会导致生成延迟从 $$O(1)$$ 骤增为 $$O(T)$$，严重破坏并发吞吐。
</details>

<details>
<summary>3. 挂载到 InstanceState 上的 pllm.model.Adapter 会退回 eager 模式吗？</summary>

不会。无论绑定 `RequestState` 还是共享 `InstanceState`，`pllm.model.Adapter` 都通过统一的可进图覆盖表按每行或版本槽地址解析增量，保留在 vLLM 的 CUDA graph 内。只有在主动关闭 CUDA graph（`cuda_graphs=False`）或使用了切片捕获图写法（如 `@pllm.custom_forward`）时才不会保留整模型 CUDA graph。
</details>

</div>

</div>

<div class="zine-doc-footer">
  <a class="zine-pill-link" data-i18n-zh="← 返回 pLLM 项目与文档中心" data-i18n-en="← Back to pLLM Project &amp; Docs" href="/pllm/#docs">← Back to pLLM Project &amp; Docs</a>
  <a class="zine-pill-link" href="https://github.com/LinesHogan/pLLM/blob/main/doc/guides/model-edits.md" target="_blank" rel="noopener">GitHub &middot; doc/guides/model-edits.md ↗</a>
</div>

<script defer src="/assets/js/bilingual-post.js"></script>
