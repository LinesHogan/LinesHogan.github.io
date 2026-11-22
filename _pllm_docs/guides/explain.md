---
layout: page
title: "Seeing Costs Early: pllm.explain"
description: "pLLM Documentation · User Guides"
permalink: /pllm/docs/guides/explain/
nav: false
---

<link rel="stylesheet" href="{{ '/assets/css/pllm-hub.css' | relative_url | bust_file_cache }}">

<div class="zine-dispatch" data-title-zh="写代码时就看到成本" data-title-en="Seeing Costs Early: pllm.explain" data-desc-zh="pLLM 文档 · 用户指南" data-desc-en="pLLM Documentation · User Guides">
  <div class="zine-dispatch-links">
    <a class="zine-pill-link primary" data-i18n-zh="← 返回 pLLM 文档中心" data-i18n-en="← pLLM Docs Hub" href="/pllm/#docs">← pLLM Docs Hub</a>
    <a class="zine-pill-link" data-i18n-zh="文档导航地图" data-i18n-en="Docs Roadmap" href="/pllm/docs/">Docs Roadmap</a>
    <span class="zine-lang-badge" data-i18n-zh="用户指南" data-i18n-en="User Guides">User Guides</span>
  </div>
  <a class="zine-pill-link" data-i18n-zh="GitHub 源码 ↗" data-i18n-en="View on GitHub ↗" href="https://github.com/LinesHogan/pLLM/blob/main/doc/guides/explain.md" target="_blank" rel="noopener">View on GitHub ↗</a>
</div>

<div id="markdown-content">

<div class="zine-lang-panel" data-lang-panel="en" lang="en" markdown="1">

<div id="en-seeing-costs-early-pllmexplain"></div>

Written for readers who have already gone through Tutorial 03 (ESamp, per-request online training): with a single call to `pllm.explain(workload, config)`—without loading weights or touching a GPU—you can see in seconds whether your workload enters CUDA graphs, where each computation is scheduled, how much memory pLLM will reserve, and which setting to adjust if a budget is too tight.

## Why You Need It {#en-why-you-need-it}

The brute-force way to find out whether a workload falls back to eager mode or how much memory it uses is to spin up the full engine: load weights, allocate the KV cache, capture CUDA graphs, and then inspect `llm.diagnostics().eager_reason` or wait for an error.

`pllm.explain` runs the exact same declarations, admission rules, and planner on CPU to show you the plan up front. It only reads the model config (`config.json`) and traces tensor shapes on CPU—no model weights loaded, no CUDA initialized, and no GPU memory allocated. Using Tutorial 03's `MiniESamp` with a 7B config as an example, the first call takes ~0.2 s on CPU (including reading the config, excluding `import pllm`), and a second call in the same process takes ~0.1 s.

## A CUDA-Graph-Captured Workload: MiniESamp from Tutorial 03 {#en-a-cuda-graph-captured-workload-miniesamp-from-tutorial-03}

To check whether your online-training workload fits inside CUDA graphs and how its per-step compute overlaps, call `pllm.explain` directly on `MiniESamp` from Tutorial 03:

```python
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
import pllm

class Distiller(nn.Module):
    def __init__(self, width: int, rank: int) -> None:
        super().__init__()
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + self.up(F.silu(self.down(x)))
        return F.layer_norm(y, y.shape[-1:])

@dataclass(frozen=True)
class MiniESamp(pllm.Workload):
    beta: float
    rank: int = 4

    @property
    def distiller(self) -> pllm.RequestState:
        return pllm.RequestState(
            init=pllm.FromModule(lambda m: Distiller(m.hidden_size, self.rank)), optimizer=pllm.AdamW(1e-3)
        )

    def trained(self, versions: torch.Tensor = pllm.version("distiller")) -> torch.Tensor:
        return versions > 0

    @pllm.train(state="distiller")
    def loss(self, shallow: torch.Tensor = pllm.hidden(0), deep: torch.Tensor = pllm.hidden(-1),
             f: Distiller = pllm.module("distiller")) -> torch.Tensor:
        return (f(shallow) - deep).square().mean(-1)

    def strength(self) -> float:
        return self.beta

    @pllm.guide_sampling(beta=strength, rows="trained")
    def predict(self, shallow: torch.Tensor = pllm.hidden(0), f: Distiller = pllm.module("distiller")) -> torch.Tensor:
        return f(shallow)

config = pllm.ModelConfig("Qwen/Qwen2.5-7B-Instruct", max_model_len=512, max_num_seqs=128,
                          max_num_batched_tokens=8192)
print(pllm.explain(MiniESamp(beta=0.5), config))
```

Output (excerpted; `kernels`, `note`, and a few `timing` details omitted):

```text
model    Qwen2ForCausalLM, hidden 3584, 28 layers, vocab 152064, bfloat16 [exact, config.json]
parallel tensor_parallel_size=1; reservations are per rank
forward  CUDA graphs: no effect needs eager
base     14140571648 FLOP/token [estimate: dense projections and LM head; excludes attention over KV history]
member  effect  anchor / lane / graph  state reads (origin, distance)
loss  train  token.commit / side / captured  distiller (latest, 0); one AdamW step per state per engine step; prompt rows train once per prefill chunk
  CUDA-graph pool reservation: 240756768 bytes [planned capture residency; cuda_graph_pool_budget_bytes caps the pool]
  estimate / step (128 rows): 18350080 matmul FLOP + 0 projection FLOP; state read/write 44040704/44040704 bytes, optimizer 88081408 bytes (included), sampler 0 bytes
predict  guide_sampling  model.execute / main / captured  distiller (boundary, 1)
  CUDA-graph pool reservation: 2150432 bytes [planned capture residency; cuda_graph_pool_budget_bytes caps the pool]
  estimate / step (128 rows): 0 matmul FLOP + 0 projection FLOP; state read/write 0/0 bytes, optimizer 0 bytes (included), sampler 0 bytes
predict  guide_sampling  state.publish / main / captured  distiller (boundary, 1)
  CUDA-graph pool reservation: 98902048 bytes [planned capture residency; cuda_graph_pool_budget_bytes caps the pool]
  estimate / step (128 rows): 7340032 matmul FLOP + 3758096384 projection FLOP; state read/write 14680064/0 bytes, optimizer 0 bytes (included), sampler 4080928768 bytes
tap      model.layers.0[0], decode: 1837056 bytes shared by loss, predict
tap      model.layers.27[0], decode: 1837056 bytes shared by loss
state  scope  bytes/root  reserved bytes [exact; state_page_budget_bytes]
distiller  request  344080  102764592  field 0: float32 x 384 pages, field 1: float32 x 256 pages, field 2: float32 x 256 pages, field 3: float32 x 256 pages; source: module init; placement: device; host bytes: 0; logical transfer bytes/row: 0; decode bytes/step: 0
memory   peak 1330848404 bytes funded by pLLM [per-rank tensor reservations; excludes vLLM model-graph/activation pools and GPU-dependent BLAS workspace; execution_peak_budget_bytes]
         below 1330848404 bytes tasks run inline; below 1168476308 bytes planning fails
preempt  state keep; replay rejection: not applicable to kept state
host     0 syncs in traced workload math
timing   publish_lag=1 (default): updates at step n become readable at n + 2; training may overlap later forwards
```

Here is how to read the plan line by line:

- **`forward`**: The entire model forward can run inside CUDA graphs; no effect forces eager execution.
- **`base`**: Estimated FLOPs per token for the base model's dense projections and LM head, excluding attention over KV history.
- **`member`**: Where each of your methods is scheduled. A single method may span multiple lines, one per execution stage (for example, `predict` appears twice):
  - `anchor` (when it runs): `token.commit` runs after tokens for the step are committed; `model.execute` runs during model forward; `state.publish` runs when new state versions are published.
  - `lane` (which CUDA stream): `side` is the side stream overlapping with the next forward; `main` is the main stream.
  - `graph`: `captured` means the task is captured into a CUDA graph.
  - `state reads (origin, distance)`: Which state it reads and how many steps behind the latest version that read sits. `loss` reads the latest version (distance 0), while `predict` reads a version 1 step behind (which is why Tutorial 03 notes that "updates from step N become readable at step N+2", enabling overlap). So for `MiniESamp`, `loss` trains on the side stream inside a CUDA graph, and `predict` runs on the main stream inside a CUDA graph.
  - `CUDA-graph pool reservation`: Private GPU memory pool reserved to capture this task's CUDA graph. Because this comes out of the memory available to the KV cache, it is called out explicitly.
  - `estimate / step`: Estimated per-step FLOPs (sized at `max_num_seqs` rows), state read/write bytes, and sampler bytes. `predict`'s `projection FLOP` and `sampler` bytes come from projecting predictions through the LM head to guide sampling.
- **`tap`**: Hidden activations extracted from the model (layer 0 and layer 27) and their reserved buffer sizes (1,837,056 bytes each); layer 0's buffer is shared between `loss` and `predict`.
- **`state`**: `distiller` is a request-scoped state with a page pool per field, where `x N pages` shows the pages reserved for that field. One version across all fields totals 344,080 bytes, with 102,764,592 bytes (~98 MiB) reserved overall; `[exact]` indicates an exact calculation. Field 0 holds the parameters and reserves 384 = 3 × 128 pages: each concurrent request needs to retain the version read in the current step, the older version read one step behind by `predict`, and the new version currently being written by training. Fields 1–3 hold AdamW's step count and first/second moments, which only training reads—old versions never need to keep them—so each request only retains the latest version and the new version being written, totaling 256 pages each (see [Performance & Memory, Section 3.1](/pllm/docs/explain/performance/)).
- **`memory`**: Peak memory funded by pLLM under `execution_peak_budget_bytes` (including state pages, task workspaces and results, graph pools, and anchor reservations): 1,330,848,404 bytes (~1,269 MiB). The second line explains the thresholds: below 1,330,848,404 bytes, side-stream tasks fall back to running inline on the main stream (no longer overlapping); below 1,168,476,308 bytes, planning fails.
- **`host`**: No GPU-to-CPU synchronizations appear in the traced workload math (`0 syncs`).
- **`timing`**: Timing settings and their consequences; `(default)` means you kept the default value.
- **`note`**: Things this report cannot know statically (see the limitations section below).

## A Sliced-Graph Workload: Conditional Early Exit (`custom_forward`) {#en-a-sliced-graph-workload-conditional-early-exit-custom_forward}

`Adapter` (whether bound to `RequestState` or a shared `InstanceState`) and `ActivationPatch` stay inside vLLM's whole-model CUDA graph. What actually splits the whole-model CUDA graph into per-slice wave graphs (and explains why in `eager_reason`) is `@pllm.custom_forward`. Running `pllm.explain` before starting the engine shows the pool reservations and slicing notes up front:

```python
@dataclass(frozen=True)
class LoopTail(pllm.Workload):
    threshold: float = 0.0

    @pllm.custom_forward(tokens="decode")
    def forward(self, layers: pllm.model.Layers) -> torch.Tensor:
        h = layers(stop=26)[25]
        active = torch.ones_like(h[:, :1], dtype=torch.bool)
        for layer in range(26, 28):
            active = active & (h.float().square().mean(-1, keepdim=True) > self.threshold)
            h = layers(layer, layer + 1, hidden=h, active=active)[layer]
        return h


print(pllm.explain(LoopTail(), config))
```

The leading lines of output look like this (`timing` and `note` lines omitted):

```text
model    Qwen2ForCausalLM, hidden 3584, 28 layers, vocab 152064, bfloat16 [exact, config.json]
parallel tensor_parallel_size=1; reservations are per rank
forward  eager: @pllm.custom_forward schedules model slices and owns their wave graphs
base     14140571648 FLOP/token [estimate: dense projections and LM head; excludes attention over KV history]
member  effect  anchor / lane / graph  state reads (origin, distance)
kernels  forward: ~11 (eager) / step
forward  custom_forward  model.execute / main / eager  - (entry, 0)
  CUDA-graph pool reservation: 4244635648 bytes [planned capture residency; cuda_graph_pool_budget_bytes caps the pool]
  matmul/byte estimate / step: unavailable; profile this model-site or host-batch body with a loaded model
  forward eager because forward: @pllm.custom_forward schedules model slices and owns their wave graphs
kernels  total: ~11 (eager) / step [estimate: source operators; excludes base model]
state  scope  bytes/root  reserved bytes [exact; state_page_budget_bytes]
memory   peak 4273995778 bytes funded by pLLM [per-rank tensor reservations; excludes vLLM model-graph/activation pools and GPU-dependent BLAS workspace; execution_peak_budget_bytes]
preempt  state reset; replay rejection: None
host     0 syncs in traced workload math
```

Two key differences stand out here:

- **`forward eager`**: `@pllm.custom_forward` conditionally controls the execution order of layer spans, so the outer wrapper runs eagerly while individual layer spans are still captured as wave CUDA graphs (budgeted in `CUDA-graph pool reservation`). By contrast, simply adding or subtracting weight overlays with `Adapter` on a shared `InstanceState` or per-request `RequestState` never triggers this line.
- **`model.execute / main / eager`**: Sliced execution replaces the model's main forward itself, so it attaches to `model.execute` on the main lane.

Comparing the two workloads side by side:

| Workload | Model Forward & Execution Mode |
| --- | --- |
| `MiniESamp` | vLLM whole-model CUDA graph; training overlaps subsequent forwards on the side stream |
| `LoopTail` | Outer conditional slice scheduling; layer spans run inside wave CUDA graphs |

You can weigh the trade-off before running anything: when you want whole-model static graph overhead and minimal dispatch latency, prefer standard layer sites with `Adapter` / `ActivationPatch`; when you need dynamic cross-layer routing or early exit, `pllm.explain` sizes the wave-graph pool for you up front.

## When a Budget Is Too Tight, It Tells You What to Change {#en-when-a-budget-is-too-tight-it-tells-you-what-to-change}

If you configure a memory budget and want to check whether the current config fits—or which parameter to tune if it does not—inspect `budget_errors` on the report:

```python
small = pllm.ModelConfig("Qwen/Qwen2.5-7B-Instruct", max_model_len=512, max_num_seqs=128,
                         max_num_batched_tokens=8192,
                         execution=pllm.ExecutionOptions(execution_peak_budget_bytes=64 << 20))
print(pllm.explain(MiniESamp(beta=0.5), small).budget_errors)
```

Output:

```text
('pLLM execution needs 1168476308 bytes at its peak but only 67108864 fit; lower max_num_batched_tokens or raise ExecutionOptions.execution_peak_budget_bytes; reduce device tables or explicitly declare placement="host" for read-only lookup tables',)
```

An empty `budget_errors` tuple means the workload fits within the declared budget. When non-empty, the report still shows the unconstrained plan and attaches the exact error message `make_llm` would raise. Other admission failures (such as unsupported compositions) cause `pllm.explain` to raise directly with the same error message as engine creation.

## What It Does Not Know {#en-what-it-does-not-know}

Because `pllm.explain` runs on CPU, it has the following boundaries:

- Whether everything fits on your GPU is only decided at `make_llm`: reported memory excludes model weights, the KV cache, and vLLM's own activations and fragmentation.
- BLAS workspace sizes used by projections and scans depend on the GPU and are excluded; affected peaks are lower bounds.
- FLOPs and logical memory traffic bytes are shape-based estimates (lower bounds for custom operators), with no throughput or latency predictions—measure throughput empirically (as in Tutorial 03).
- Sync counts cover only CPU shape-traced workload math; computations inside model sites (`ActivationPatch`, `custom_residual`, etc.) are marked `unverified`.
- Config-only inspection currently derives projection shapes for dense Qwen2, Qwen3, and Llama configs; projection admission for other model families is checked at `make_llm`.

## For Scripts and Coding Agents {#en-for-scripts-and-coding-agents}

If you are writing automation scripts or having an agent check workloads automatically, there is no need to parse the text output. The object returned by `pllm.explain` converts via `dataclasses.asdict(report)` into a JSON-serializable dictionary containing `eager_reason`, `members` (per-method tasks), `states`, `peak_memory_bytes`, `budget_errors`, `exports`, and more; `flops_per_token_estimate` gives the combined per-token FLOP estimate across the base model and all tasks:

```python
import json
from dataclasses import asdict

report = pllm.explain(MiniESamp(beta=0.5), config)
print(report.eager_reason == "", [state.reserved_bytes for state in report.states])
print(len(json.dumps(asdict(report))) > 0)
```

This prints `True [102764592]` and `True`.

</div>

<div class="zine-lang-panel is-hidden" data-lang-panel="zh" lang="zh-CN" markdown="1">

<div id="zh-sec-1"></div>

本文面向已经读过教程 03（ESamp，每请求在线训练）的读者：只需一行 `pllm.explain(workload, config)`，不用加载权重、不用占用 GPU，几秒钟就能在 CPU 上看到负载会不会进 CUDA graph、计算排在哪、pLLM 要预留多少显存，以及预算不够时该改哪个参数。

## 为什么需要它 {#zh-sec-2}

要知道一个负载会不会退回 eager 模式、要占多少显存，最直接的办法是真正启动引擎：加载权重、分配 KV cache、捕获 CUDA graph，然后看 `llm.diagnostics().eager_reason` 或者等报错。

`pllm.explain` 用同一套声明、同一套准入规则和规划器，在 CPU 上把计划算出来给你看。它只读取模型配置（`config.json`）并在 CPU 上追踪张量形状，不加载模型权重，不初始化 CUDA，也不分配 GPU 显存。以教程 03 的 `MiniESamp` 和 7B 的配置为例，在本机 CPU 上首次调用约 0.2 s（包含读取配置，不含 `import pllm`），同一进程里第二次调用约 0.1 s。

## 一个能进 CUDA graph 的负载：教程 03 的 MiniESamp {#zh-sec-3}

如果你想知道写好的在线训练负载能不能跑进 CUDA graph、每步计算怎么重叠，可以直接对教程 03 里的 `MiniESamp` 类调用 `pllm.explain`（类的定义见教程 03）：

```python
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
import pllm

class Distiller(nn.Module):
    def __init__(self, width: int, rank: int) -> None:
        super().__init__()
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + self.up(F.silu(self.down(x)))
        return F.layer_norm(y, y.shape[-1:])

@dataclass(frozen=True)
class MiniESamp(pllm.Workload):
    beta: float
    rank: int = 4

    @property
    def distiller(self) -> pllm.RequestState:
        return pllm.RequestState(
            init=pllm.FromModule(lambda m: Distiller(m.hidden_size, self.rank)), optimizer=pllm.AdamW(1e-3)
        )

    def trained(self, versions: torch.Tensor = pllm.version("distiller")) -> torch.Tensor:
        return versions > 0

    @pllm.train(state="distiller")
    def loss(self, shallow: torch.Tensor = pllm.hidden(0), deep: torch.Tensor = pllm.hidden(-1),
             f: Distiller = pllm.module("distiller")) -> torch.Tensor:
        return (f(shallow) - deep).square().mean(-1)

    def strength(self) -> float:
        return self.beta

    @pllm.guide_sampling(beta=strength, rows="trained")
    def predict(self, shallow: torch.Tensor = pllm.hidden(0), f: Distiller = pllm.module("distiller")) -> torch.Tensor:
        return f(shallow)

config = pllm.ModelConfig("Qwen/Qwen2.5-7B-Instruct", max_model_len=512, max_num_seqs=128,
                          max_num_batched_tokens=8192)
print(pllm.explain(MiniESamp(beta=0.5), config))
```

输出如下（节选，省略了 `kernels`、`note` 行和几行 `timing` 细节）：

```text
model    Qwen2ForCausalLM, hidden 3584, 28 layers, vocab 152064, bfloat16 [exact, config.json]
parallel tensor_parallel_size=1; reservations are per rank
forward  CUDA graphs: no effect needs eager
base     14140571648 FLOP/token [estimate: dense projections and LM head; excludes attention over KV history]
member  effect  anchor / lane / graph  state reads (origin, distance)
loss  train  token.commit / side / captured  distiller (latest, 0); one AdamW step per state per engine step; prompt rows train once per prefill chunk
  CUDA-graph pool reservation: 240756768 bytes [planned capture residency; cuda_graph_pool_budget_bytes caps the pool]
  estimate / step (128 rows): 18350080 matmul FLOP + 0 projection FLOP; state read/write 44040704/44040704 bytes, optimizer 88081408 bytes (included), sampler 0 bytes
predict  guide_sampling  model.execute / main / captured  distiller (boundary, 1)
  CUDA-graph pool reservation: 2150432 bytes [planned capture residency; cuda_graph_pool_budget_bytes caps the pool]
  estimate / step (128 rows): 0 matmul FLOP + 0 projection FLOP; state read/write 0/0 bytes, optimizer 0 bytes (included), sampler 0 bytes
predict  guide_sampling  state.publish / main / captured  distiller (boundary, 1)
  CUDA-graph pool reservation: 98902048 bytes [planned capture residency; cuda_graph_pool_budget_bytes caps the pool]
  estimate / step (128 rows): 7340032 matmul FLOP + 3758096384 projection FLOP; state read/write 14680064/0 bytes, optimizer 0 bytes (included), sampler 4080928768 bytes
tap      model.layers.0[0], decode: 1837056 bytes shared by loss, predict
tap      model.layers.27[0], decode: 1837056 bytes shared by loss
state  scope  bytes/root  reserved bytes [exact; state_page_budget_bytes]
distiller  request  344080  102764592  field 0: float32 x 384 pages, field 1: float32 x 256 pages, field 2: float32 x 256 pages, field 3: float32 x 256 pages; source: module init; placement: device; host bytes: 0; logical transfer bytes/row: 0; decode bytes/step: 0
memory   peak 1330848404 bytes funded by pLLM [per-rank tensor reservations; excludes vLLM model-graph/activation pools and GPU-dependent BLAS workspace; execution_peak_budget_bytes]
         below 1330848404 bytes tasks run inline; below 1168476308 bytes planning fails
preempt  state keep; replay rejection: not applicable to kept state
host     0 syncs in traced workload math
timing   publish_lag=1 (default): updates at step n become readable at n + 2; training may overlap later forwards
```

按行对照就能读懂这份执行计划：

- **`forward`**：模型前向能整段走 CUDA graph，没有哪个效果要求 eager。
- **`base`**：基座模型每个 token 的稠密投影与 LM head FLOP 估计，不含对 KV 历史的 attention。
- **`member`**：你写的每个方法被安排在哪。一个方法可能对应多行，每行是它的一段计算（比如 `predict` 出现了两行）：
  - `anchor`（什么时候跑）：`token.commit` 表示这一步的 token 确定之后；`model.execute` 表示模型前向时；`state.publish` 表示状态发布新版本时。
  - `lane`（在哪条 CUDA stream）：`side` 是旁路 stream，和下一步 forward 重叠；`main` 是主 stream。
  - `graph`：`captured` 表示进 CUDA graph。
  - `state reads (origin, distance)`：读哪份状态、读到的版本离最新差几步。`loss` 读最新的（距离 0），`predict` 读的版本落后 1（这就是教程 03 里说的“第 N 步的更新在第 N+2 步被读到”带来的重叠）。所以对 `MiniESamp` 来说，`loss` 训练在旁路 stream、进 graph；`predict` 在主 stream、进 graph。
  - `CUDA-graph pool reservation`：为这段计算捕获 CUDA graph 要预留的私有显存池；这部分显存要从 KV cache 那里分走，所以列出来。
  - `estimate / step`：这段计算每步（按 `max_num_seqs` 行算）的 FLOP、状态读写字节和采样器字节估计。`predict` 的 `projection FLOP` 与 `sampler` 字节来自用 LM head 投影预测并引导采样。
- **`tap`**：从模型里取出的 hidden（第 0 层、第 27 层）和预留的缓冲大小（各 1837056 字节）；第 0 层的那份由 `loss` 和 `predict` 共用。
- **`state`**：`distiller` 是请求级状态，每个字段各有一个页池，`x N pages` 是这个字段预留的页数；一个版本的全部字段共 344080 字节，总预留 102764592 字节（约 98 MiB）；`[exact]` 表示这是精确值。字段 0 是参数，预留 384 = 3 × 128 页：每个并发请求要同时留住这一步读的版本、`predict` 落后一步要读的旧版本和训练正在写的新版本。字段 1–3 是 AdamW 的步数和一、二阶矩，只有训练自己读，旧版本不必保留它们，所以每个请求只留最新版本和正在写的新版本，共 256 页（见[性能说明 3.1 节](/pllm/docs/explain/performance/#zh-sec-10)）。
- **`memory`**：由 `execution_peak_budget_bytes` 约束的 pLLM 执行峰值（含状态页、任务工作区与结果、图池和 anchor 预留），为 1330848404 字节（约 1269 MiB）。第二行说明：预算低于 1330848404 字节时，旁路任务改成在主 stream 上内联运行（不再重叠）；低于 1168476308 字节时规划失败。
- **`host`**：负载的张量数学里没有 GPU 到 CPU 的同步（`0 syncs`）。
- **`timing`**：时机设置及其后果；`(default)` 表示你没改过默认值。
- **`note`**：这份报告不知道的事（见下文的限制说明）。

## 一个改用切片图的负载：条件早退层（`custom_forward`） {#zh-sec-4}

`Adapter`（无论绑定在 `RequestState` 还是共享 `InstanceState` 上）与 `ActivationPatch` 都已经留在 vLLM 的整模型 CUDA graph 中；真正会把整模型 CUDA graph 拆成切片捕获图（并在 `eager_reason` 里说明）的写法是 `@pllm.custom_forward`。启动引擎前用 `pllm.explain` 即可提前看到池大小预留和切片说明：

```python
@dataclass(frozen=True)
class LoopTail(pllm.Workload):
    threshold: float = 0.0

    @pllm.custom_forward(tokens="decode")
    def forward(self, layers: pllm.model.Layers) -> torch.Tensor:
        h = layers(stop=26)[25]
        active = torch.ones_like(h[:, :1], dtype=torch.bool)
        for layer in range(26, 28):
            active = active & (h.float().square().mean(-1, keepdim=True) > self.threshold)
            h = layers(layer, layer + 1, hidden=h, active=active)[layer]
        return h


print(pllm.explain(LoopTail(), config))
```

输出的前几行如下（`timing` 与 `note` 行略）：

```text
model    Qwen2ForCausalLM, hidden 3584, 28 layers, vocab 152064, bfloat16 [exact, config.json]
parallel tensor_parallel_size=1; reservations are per rank
forward  eager: @pllm.custom_forward schedules model slices and owns their wave graphs
base     14140571648 FLOP/token [estimate: dense projections and LM head; excludes attention over KV history]
member  effect  anchor / lane / graph  state reads (origin, distance)
kernels  forward: ~11 (eager) / step
forward  custom_forward  model.execute / main / eager  - (entry, 0)
  CUDA-graph pool reservation: 4244635648 bytes [planned capture residency; cuda_graph_pool_budget_bytes caps the pool]
  matmul/byte estimate / step: unavailable; profile this model-site or host-batch body with a loaded model
  forward eager because forward: @pllm.custom_forward schedules model slices and owns their wave graphs
kernels  total: ~11 (eager) / step [estimate: source operators; excludes base model]
state  scope  bytes/root  reserved bytes [exact; state_page_budget_bytes]
memory   peak 4273995778 bytes funded by pLLM [per-rank tensor reservations; excludes vLLM model-graph/activation pools and GPU-dependent BLAS workspace; execution_peak_budget_bytes]
preempt  state reset; replay rejection: None
host     0 syncs in traced workload math
```

这里重点看两处变化：

- **`forward eager`**：`@pllm.custom_forward` 需要按条件控制各层段的执行顺序，因此外层包装函数走 eager 调度，而各层段内部依然捕获成波次 CUDA graph（显存预算计入 `CUDA-graph pool reservation`）；如果只是加减权重覆盖层，共享 `InstanceState` 或每请求 `RequestState` 上的 `Adapter` 都不会触发这一行。
- **`model.execute / main / eager`**：切片执行替换的是模型主体前向本身，所以它挂在主路径的 `model.execute` 上。

把两个负载放在一起对比：

| 负载 | 模型前向与执行方式 |
| --- | --- |
| `MiniESamp` | vLLM 整模型 CUDA graph，训练在旁路 stream 上与后续前向重叠 |
| `LoopTail` | 外层按切片条件调度，层段内部使用波次 CUDA graph |

写之前就能权衡：要整模型图的静态开销与最短调度，就优先用标准层站点和 `Adapter` / `ActivationPatch`；需要跨层动态路由或早退时，`pllm.explain` 会先把波次图池大小算清楚。

## 预算不够时，它告诉你改什么 {#zh-sec-5}

如果你给负载设了显存预算，想知道当前配置能不能装下、装不下该调哪个参数，可以直接查看报告的 `budget_errors`：

```python
small = pllm.ModelConfig("Qwen/Qwen2.5-7B-Instruct", max_model_len=512, max_num_seqs=128,
                         max_num_batched_tokens=8192,
                         execution=pllm.ExecutionOptions(execution_peak_budget_bytes=64 << 20))
print(pllm.explain(MiniESamp(beta=0.5), small).budget_errors)
```

输出如下：

```text
('pLLM execution needs 1168476308 bytes at its peak but only 67108864 fit; lower max_num_batched_tokens or raise ExecutionOptions.execution_peak_budget_bytes; reduce device tables or explicitly declare placement="host" for read-only lookup tables',)
```

`budget_errors` 为空表示声明的预算装得下；不为空时，报告仍会给出不带这个限制的计划，并附上和 `make_llm` 相同的报错信息。如果是其他准入错误（例如不支持的组合），`pllm.explain` 会直接抛出异常，报错信息与创建引擎时相同。

## 它不知道的事 {#zh-sec-6}

`pllm.explain` 在 CPU 上运行，有以下边界：

- 能不能装进你的 GPU，要到 `make_llm` 时才确定：报告里的显存不含模型权重、KV cache、vLLM 自己的激活和碎片。
- 投影和 scan 用到的 BLAS 工作区大小取决于 GPU，报告不计入；受影响的峰值是下界。
- FLOP 和逻辑访存字节是按形状算出的估计（自写算子给出下界）；没有吞吐或延迟估计——吞吐还是要实测（教程 03 的做法）。
- 同步次数只覆盖在 CPU 上追踪过形状的负载数学；作用在模型内部的计算（激活补丁、`custom_residual` 等）标为 `unverified`。
- 配置读取目前支持稠密 Qwen2、Qwen3 和 Llama 的投影形状；其他家族的投影准入在 `make_llm` 时判断。

## 给程序/代理用 {#zh-sec-7}

如果你在写自动化脚本或让代理自动检查负载，不需要解析文本。`pllm.explain` 返回的对象可以用 `dataclasses.asdict(report)` 转成能够 `json.dumps` 的字典，包含 `eager_reason`、`members`（每个方法的任务）、`states`、`peak_memory_bytes`、`budget_errors`、`exports` 等字段；`flops_per_token_estimate` 是基座与各任务合计的每 token FLOP 估计：

```python
import json
from dataclasses import asdict

report = pllm.explain(MiniESamp(beta=0.5), config)
print(report.eager_reason == "", [state.reserved_bytes for state in report.states])
print(len(json.dumps(asdict(report))) > 0)
```

输出为 `True [102764592]` 和 `True`。

</div>

</div>

<div class="zine-doc-footer">
  <a class="zine-pill-link" data-i18n-zh="← 返回 pLLM 项目与文档中心" data-i18n-en="← Back to pLLM Project &amp; Docs" href="/pllm/#docs">← Back to pLLM Project &amp; Docs</a>
  <a class="zine-pill-link" href="https://github.com/LinesHogan/pLLM/blob/main/doc/guides/explain.md" target="_blank" rel="noopener">GitHub &middot; doc/guides/explain.md ↗</a>
</div>

<script defer src="/assets/js/bilingual-post.js"></script>
