---
layout: page
title: "02 Per-Request LoRA: Generating Adapters On the Fly from Prompts"
description: "pLLM Documentation · Step-by-Step Tutorials (00–05)"
permalink: /pllm/docs/tutorials/02_request_adapters/
nav: false
---

<link rel="stylesheet" href="{{ '/assets/css/pllm-hub.css' | relative_url | bust_file_cache }}">

<div class="zine-dispatch" data-title-zh="02 每请求 LoRA：从 prompt 现场生成适配器" data-title-en="02 Per-Request LoRA: Generating Adapters On the Fly from Prompts" data-desc-zh="pLLM 文档 · 交互式入门教程 (00–05)" data-desc-en="pLLM Documentation · Step-by-Step Tutorials (00–05)">
  <div class="zine-dispatch-links">
    <a class="zine-pill-link primary" data-i18n-zh="← 返回 pLLM 文档中心" data-i18n-en="← pLLM Docs Hub" href="/pllm/#docs">← pLLM Docs Hub</a>
    <a class="zine-pill-link" data-i18n-zh="文档导航地图" data-i18n-en="Docs Roadmap" href="/pllm/docs/">Docs Roadmap</a>
    <span class="zine-lang-badge" data-i18n-zh="可执行教程 (.ipynb)" data-i18n-en="Notebook (.ipynb)">Notebook (.ipynb)</span>
  </div>
  <a class="zine-pill-link" data-i18n-zh="在 GitHub 打开 .ipynb ↗" data-i18n-en="Open .ipynb on GitHub ↗" href="https://github.com/LinesHogan/pLLM/blob/main/doc/tutorials/02_request_adapters.ipynb" target="_blank" rel="noopener">Open .ipynb on GitHub ↗</a>
</div>

<div id="markdown-content">

<div class="zine-lang-panel" data-lang-panel="en" lang="en" markdown="1">

<div id="en-sec-1"></div>

**Scenario**: A line of recent work uses a hypernetwork to turn a task description directly into LoRA weights—such as Text-to-LoRA (Charakorn et al. 2025) and Drag-and-Drop LLMs (Liang et al. 2025). At serving time, each incoming request brings its own task, so each request needs **its own LoRA**, and that LoRA can only be generated after reading its prompt.

vLLM's built-in multi-LoRA serving only selects among pre-trained adapters loaded from disk by name; there is nowhere in vLLM to store and apply an adapter synthesized on the fly for a single request.

In this tutorial, we build a Text-to-LoRA-style architecture:

1. Declare per-request LoRA factors and an adapter applying them to layer 12's `q` and `v` projections;
2. Declare an engine-wide hypernetwork shared across requests that reads the last prompt token's hidden state and writes that request's factors;
3. Verify that when the hypernetwork outputs zeros, generation matches plain vLLM identically;
4. Switch to a nonzero hypernetwork and observe how each request's output changes.

The hypernetwork here is randomly initialized and **untrained**: this tutorial demonstrates the structure and per-request isolation rather than Text-to-LoRA task quality.

```python
import os

os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"  # Show only vLLM errors to suppress startup logs

from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoTokenizer
from vllm import SamplingParams

import pllm
from pllm import testing

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
CONFIG = pllm.ModelConfig(MODEL, max_model_len=512, max_num_seqs=4, max_num_batched_tokens=512,
                          gpu_memory_utilization=0.3, seed=0)
```

## Step 1: The Hypernetwork Is a Standard `nn.Module` {#en-step-1-the-hypernetwork-is-a-standard-nnmodule}

It maps a hidden vector to one flat vector concatenating the LoRA factors for all target projections. `sizes` holds `(input_width, output_width)` for each target projection, supplied by pLLM after model loading; `scale` controls the final layer's initialization, where `0` produces all-zero outputs.

```python
class HyperNet(nn.Module):
    def __init__(self, width: int, sizes: tuple[tuple[int, int], ...], rank: int, scale: float) -> None:
        super().__init__()
        out = sum(rank * inputs + outputs * rank for inputs, outputs in sizes)
        self.body = nn.Sequential(nn.Linear(width, 256), nn.Tanh(), nn.Linear(256, out))
        nn.init.normal_(self.body[2].weight, std=scale)
        nn.init.zeros_(self.body[2].bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.body(h)
```

## Step 2: The Workload {#en-step-2-the-workload}

Four declarations:

- `lora`: A `pllm.RequestState`, **one copy per request**. The shapes of fields `a` (`rank × input_width`) and `b` (`output_width × rank`) depend on the target projections; because there are two targets, each field has two slices named in order: `a_0, b_0` (`q_proj`) and `a_1, b_1` (`v_proj`). There is no optimizer—the state is written once rather than trained.
- `apply`: A `pllm.model.Adapter` applying `lora` as a low-rank delta to the target projections: `y = W x + B A x`.
- `hyper`: A `pllm.InstanceState(init=pllm.FromModule(...))` for the shared hypernetwork, initialized via the module's own constructor.
- `write_lora`: Decorated with `@pllm.write(state="lora", fields=...)`. It reads `pllm.hidden(-1, tokens="prefill")`—the final-layer output at each request's **last prompt token**—runs the hypernetwork, and returns that request's four factors. `pllm.parameter("lora", ...)` is passed here purely to obtain each field's shape and dtype.

In timeline order: the prompt itself runs through the unmodified base model; once the prompt finishes, `write_lora` populates the factors; starting from the very first generated token, that request's `q` and `v` projections carry its own LoRA.

```python
TARGETS = (pllm.model.Projection("q_proj", 12), pllm.model.Projection("v_proj", 12))


@dataclass(frozen=True)
class TextToLoRA(pllm.Workload):
    rank: int = 4
    scale: float = 0.0

    @property
    def lora(self) -> pllm.RequestState:
        return pllm.RequestState({"a": pllm.Field(lambda inputs, outputs: (self.rank, inputs)),
                                  "b": pllm.Field(lambda inputs, outputs: (outputs, self.rank))})

    @property
    def apply(self) -> pllm.model.Adapter:
        return pllm.model.Adapter("lora", TARGETS, operator="low_rank")

    @property
    def hyper(self) -> pllm.InstanceState:
        return pllm.InstanceState(
            init=pllm.FromModule(lambda m: HyperNet(m.hidden_size, m.target_shapes, self.rank, self.scale))
        )

    @pllm.write(state="lora", fields=("a_0", "b_0", "a_1", "b_1"))
    def write_lora(self, prompt: torch.Tensor = pllm.hidden(-1, tokens="prefill"),
                   hyper: HyperNet = pllm.module("hyper"),
                   a_q: torch.Tensor = pllm.parameter("lora", "a_0"), b_q: torch.Tensor = pllm.parameter("lora", "b_0"),
                   a_v: torch.Tensor = pllm.parameter("lora", "a_1"), b_v: torch.Tensor = pllm.parameter("lora", "b_1"),
                   ) -> tuple[torch.Tensor, ...]:
        fields = (a_q, b_q, a_v, b_v)  # Each has shape [num_requests, ...]
        flat = hyper(prompt.float())
        pieces = torch.split(flat, [f[0].numel() for f in fields], dim=-1)
        return tuple(p.reshape_as(f).to(f.dtype) for p, f in zip(pieces, fields))
```

## Step 3: With Zero Hypernetwork Output, Outputs Match Plain vLLM {#en-step-3-with-zero-hypernetwork-output-outputs-match-plain-vllm}

Two requests, compared against a plain vLLM baseline.

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL)
questions = ["What is the capital of France? Answer in one sentence.", "Name three primary colors."]
prompts = [tokenizer.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                         add_generation_prompt=True) for q in questions]
greedy = SamplingParams(max_tokens=24, temperature=0.0)

with pllm.make_llm(CONFIG) as plain:
    baseline = [output.outputs[0].text for output in plain.generate(prompts, greedy, use_tqdm=False)]

with pllm.make_llm(CONFIG, workload=TextToLoRA(scale=0.0)) as llm:
    texts = [output.outputs[0].text for output in llm.generate(prompts, greedy)]
print("与原生 vLLM 相同:", texts == baseline)
for text in texts:
    print("   ", repr(text))
```

```text
与原生 vLLM 相同: True
    'The capital of France is Paris.'
    'Three primary colors are red, blue, and yellow.'
```

## Step 4: Nonzero Hypernetwork, Different LoRA per Request {#en-step-4-nonzero-hypernetwork-different-lora-per-request}

Setting the initialization standard deviation of the hypernetwork's final layer to `0.02` causes it to emit nonzero LoRA weights for each prompt.

Below we send both prompts in a single `generate` call; each request applies the factors generated from its own prompt.

```python
with pllm.make_llm(CONFIG, workload=TextToLoRA(scale=0.02)) as llm:
    texts = [output.outputs[0].text for output in llm.generate(prompts, greedy)]
for before, after in zip(baseline, texts):
    print("原生:", repr(before))
    print("LoRA:", repr(after))
```

```text
原生: 'The capital of France is Paris.'
LoRA: 'The capital of France is Paris, and it is also known as the city of which country? Q. (tài'
原生: 'Three primary colors are red, blue, and yellow.'
LoRA: 'Three primary colors are black, white, which is:\n1. White\n2. white: 1: 5'
```

Both answers change and drift off-topic: a randomly initialized hypernetwork hasn't learned anything, so its generated LoRA pushes the model in a random direction. What matters here is the mechanism: the two prompts have different final-layer hidden states, the hypernetwork writes distinct factors for each, and each request generates using the LoRA derived from its own prompt. Plugging in a trained hypernetwork gives you the serving architecture of Text-to-LoRA.

## Swapping in a Trained Hypernetwork {#en-swapping-in-a-trained-hypernetwork}

`pllm.FromModule` accepts a factory function that constructs the module when the engine boots. To use trained weights, load them inside the constructor—for example, add a `checkpoint: str` parameter to `HyperNet`, call `self.load_state_dict(torch.load(checkpoint))` at the end of `__init__`, and pass `init=pllm.FromModule(lambda m: HyperNet(..., checkpoint="t2l.pt"))`—or save weights as safetensors and pass `pllm.FromModule(..., pretrained="t2l.safetensors")`. Note that this factory may only capture plain values (strings, numbers), not an already-instantiated module object.

## Summary {#en-summary}

- `pllm.RequestState` gives every request its own parameters; `pllm.model.Adapter` applies them to specified projections;
- `@pllm.write` writes state from your tensor math, and reading `tokens="prefill"` hidden states runs it as soon as the prompt finishes;
- Zero hypernetwork output matches plain vLLM bit-for-bit, confirming the adapter path introduces zero numerical bias.

## Common Mistakes {#en-common-mistakes}

In this workload we wrote the states as properties (since their shapes depend on `rank`). Two common mistakes here: forgetting the return type annotation on a property, and mutating a borrowed state tensor in place inside `@pllm.write` instead of returning new tensors.

```python
@dataclass(frozen=True)
class Unannotated(pllm.Workload):
    rank: int = 4

    @property
    def lora(self):  # Missing -> pllm.RequestState
        return pllm.RequestState({"a": pllm.Field(lambda inputs, outputs: (self.rank, inputs)),
                                  "b": pllm.Field(lambda inputs, outputs: (outputs, self.rank))})

    @property
    def apply(self) -> pllm.model.Adapter:
        return pllm.model.Adapter("lora", TARGETS, operator="low_rank")


class InPlaceWrite(pllm.Workload):
    memo = pllm.RequestState({"last": pllm.Field(lambda width, _: (width,))})

    @pllm.write(state="memo", fields=("last",))
    def remember(self, h: torch.Tensor = pllm.hidden(-1, tokens="prefill"),
                 last: torch.Tensor = pllm.parameter("memo", "last")) -> tuple[torch.Tensor]:
        last.copy_(h)  # Trying to write directly into the state in place
        return (last,)

    @pllm.export(name="memo_norm", max_rows=4, max_cols=1)
    def show(self, last: torch.Tensor = pllm.parameter("memo", "last")) -> torch.Tensor:
        return last.float().norm(dim=-1, keepdim=True)


try:
    pllm.explain(Unannotated(), CONFIG)
except TypeError as error:
    print("Unannotated:", error)
try:
    testing.check(InPlaceWrite(), CONFIG)
except testing.ContractError as error:
    print("InPlaceWrite:", error.failures[0])
```

```text
Unannotated: property 'lora' needs a return annotation; annotate state declarations with -> pllm.RequestState or -> pllm.InstanceState, and other properties with their return type
InPlaceWrite: [CPU execution] ValueError: method `remember` mutates borrowed input `last` in place; inputs and parameters are read-only views. Use out-of-place tensor math or clone() before writing; publish state changes through @pllm.write or @pllm.train.
```

**Why properties need return annotations.** When building a workload, pLLM needs to identify which attributes are state declarations without executing arbitrary getter code on every property. Annotating `-> pllm.RequestState` lets it recognize state declarations statically.

**Why in-place writes are forbidden.** The `last` tensor passed to your method is a read-only view of the state's current version, which other methods or requests may be reading concurrently. Mutating it in place would corrupt data mid-read. `@pllm.write`'s contract is to return new tensors so pLLM can publish them atomically as a new version. `testing.check` catches in-place writes right on CPU.

## Self-Check {#en-self-check}

1. Why doesn't the prompt itself see the request's LoRA, whereas generation does starting from the first decoded token?

<details><summary>Answer</summary>

Computing the LoRA requires the final-layer hidden state of the last prompt token, which only exists at the end of the prompt forward pass. `write_lora` writes the factors immediately afterward, so the next forward pass (the first decode token) reads them.
</details>

2. If two requests generate together in the same batch, can their LoRAs interfere with each other?

<details><summary>Answer</summary>

No. `lora` is a `pllm.RequestState`, so each request owns an isolated copy; `Adapter` applies to each row only the factors belonging to that row's request.
</details>

3. What does matching plain vLLM identically in Step 3 (when the hypernetwork outputs zeros) verify?

<details><summary>Answer</summary>

It verifies that the adapter application path itself introduces no discrepancy: any subsequent change in output comes solely from the adapter factors rather than framework overhead. When authoring your own workload, building a "zero-strength" check against plain vLLM is the cheapest correctness test available.
</details>

That same state can also be **trained** token-by-token during generation rather than written once—which is the subject of [03 Decode-Time Training](/pllm/docs/tutorials/03_decode_training/).

</div>

<div class="zine-lang-panel is-hidden" data-lang-panel="zh" lang="zh-CN" markdown="1">

<div id="zh-sec-1"></div>

**场景**：一类新工作用超网络（hypernetwork）把任务描述直接变成 LoRA 权重，比如 Text-to-LoRA
（Charakorn et al. 2025）和 Drag-and-Drop LLMs（Liang et al. 2025）。部署时每个请求都带着自己的任务，
所以每个请求要有**自己的一份 LoRA**，而且这份 LoRA 要在读完它的 prompt 之后才生成出来。

vLLM 的多 LoRA 服务只能按名字使用事先从磁盘加载好的适配器；一个请求现场生成、只属于它自己的
适配器，在 vLLM 里没有地方放。

这篇教程做一个 Text-to-LoRA 式的结构：

1. 声明每个请求一份的 LoRA 因子，以及把它们加到第 12 层 q、v 投影上的适配器；
2. 声明一个所有请求共享的超网络，它读 prompt 最后一个 token 的 hidden，写出这个请求的因子；
3. 超网络输出为零时，确认输出和原生 vLLM 完全一样；
4. 换一个非零的超网络，看每个请求的输出怎么变。

这里的超网络是随机初始化的，**不是训练好的**：这一篇演示的是结构和隔离性，不是 Text-to-LoRA 的效果。

```python
import os

os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"  # 只显示 vLLM 的报错，省掉启动日志

from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoTokenizer
from vllm import SamplingParams

import pllm
from pllm import testing

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
CONFIG = pllm.ModelConfig(MODEL, max_model_len=512, max_num_seqs=4, max_num_batched_tokens=512,
                          gpu_memory_utilization=0.3, seed=0)
```

## 第一步：超网络是普通的 `nn.Module` {#zh-sec-2}

它把一个 hidden 向量映射成所有目标投影的 LoRA 因子拼起来的一长条。`sizes` 是每个目标投影的
（输入宽度，输出宽度），由 pLLM 在加载模型后提供；`scale` 控制最后一层的初始化，0 表示输出全零。

```python
class HyperNet(nn.Module):
    def __init__(self, width: int, sizes: tuple[tuple[int, int], ...], rank: int, scale: float) -> None:
        super().__init__()
        out = sum(rank * inputs + outputs * rank for inputs, outputs in sizes)
        self.body = nn.Sequential(nn.Linear(width, 256), nn.Tanh(), nn.Linear(256, out))
        nn.init.normal_(self.body[2].weight, std=scale)
        nn.init.zeros_(self.body[2].bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.body(h)
```

## 第二步：负载 {#zh-sec-3}

四个声明：

- `lora`：`pllm.RequestState`，**每个请求一份**。字段 `a`（rank × 输入宽度）和 `b`（输出宽度 × rank）
  的形状由目标投影决定；因为有两个目标，每个字段有两份，按顺序叫 `a_0, b_0`（q 投影）和 `a_1, b_1`
  （v 投影）。没有优化器：它不是训练出来的，而是被写进去的。
- `apply`：`pllm.model.Adapter`，把 `lora` 作为低秩增量加到目标投影上：`y = W x + B A x`。
- `hyper`：`pllm.InstanceState(init=pllm.FromModule(...))`，所有请求共享的超网络，参数按模块自己的初始化方式生成。
- `write_lora`：带 `@pllm.write(state="lora", fields=...)`。它读 `pllm.hidden(-1, tokens="prefill")`，
  即每个请求 **prompt 最后一个 token** 在最后一层的输出，跑超网络，返回这个请求的四个因子。
  `pllm.parameter("lora", ...)` 在这里只用来拿到每个字段的形状和 dtype。

时间上：prompt 本身用原模型处理；prompt 读完时 `write_lora` 写入因子；从第一个生成的 token 起，
这个请求的 q、v 投影就带上了它自己的 LoRA。

```python
TARGETS = (pllm.model.Projection("q_proj", 12), pllm.model.Projection("v_proj", 12))


@dataclass(frozen=True)
class TextToLoRA(pllm.Workload):
    rank: int = 4
    scale: float = 0.0

    @property
    def lora(self) -> pllm.RequestState:
        return pllm.RequestState({"a": pllm.Field(lambda inputs, outputs: (self.rank, inputs)),
                                  "b": pllm.Field(lambda inputs, outputs: (outputs, self.rank))})

    @property
    def apply(self) -> pllm.model.Adapter:
        return pllm.model.Adapter("lora", TARGETS, operator="low_rank")

    @property
    def hyper(self) -> pllm.InstanceState:
        return pllm.InstanceState(
            init=pllm.FromModule(lambda m: HyperNet(m.hidden_size, m.target_shapes, self.rank, self.scale))
        )

    @pllm.write(state="lora", fields=("a_0", "b_0", "a_1", "b_1"))
    def write_lora(self, prompt: torch.Tensor = pllm.hidden(-1, tokens="prefill"),
                   hyper: HyperNet = pllm.module("hyper"),
                   a_q: torch.Tensor = pllm.parameter("lora", "a_0"), b_q: torch.Tensor = pllm.parameter("lora", "b_0"),
                   a_v: torch.Tensor = pllm.parameter("lora", "a_1"), b_v: torch.Tensor = pllm.parameter("lora", "b_1"),
                   ) -> tuple[torch.Tensor, ...]:
        fields = (a_q, b_q, a_v, b_v)  # 每个都是 [请求数, ...]
        flat = hyper(prompt.float())
        pieces = torch.split(flat, [f[0].numel() for f in fields], dim=-1)
        return tuple(p.reshape_as(f).to(f.dtype) for p, f in zip(pieces, fields))
```

## 第三步：超网络输出为零，输出应该和原生 vLLM 一样 {#zh-sec-4}

两个请求，先用原生 vLLM 生成作对照。

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL)
questions = ["What is the capital of France? Answer in one sentence.", "Name three primary colors."]
prompts = [tokenizer.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                         add_generation_prompt=True) for q in questions]
greedy = SamplingParams(max_tokens=24, temperature=0.0)

with pllm.make_llm(CONFIG) as plain:
    baseline = [output.outputs[0].text for output in plain.generate(prompts, greedy, use_tqdm=False)]

with pllm.make_llm(CONFIG, workload=TextToLoRA(scale=0.0)) as llm:
    texts = [output.outputs[0].text for output in llm.generate(prompts, greedy)]
print("与原生 vLLM 相同:", texts == baseline)
for text in texts:
    print("   ", repr(text))
```

```text
与原生 vLLM 相同: True
    'The capital of France is Paris.'
    'Three primary colors are red, blue, and yellow.'
```

## 第四步：非零的超网络，每个请求一份不同的 LoRA {#zh-sec-5}

把超网络最后一层的初始化标准差设为 0.02，它就会为每个 prompt 写出一份非零的 LoRA。

下面把两个 prompt 一起传给 `generate`；每个请求使用由自己的 prompt 生成的因子。

```python
with pllm.make_llm(CONFIG, workload=TextToLoRA(scale=0.02)) as llm:
    texts = [output.outputs[0].text for output in llm.generate(prompts, greedy)]
for before, after in zip(baseline, texts):
    print("原生:", repr(before))
    print("LoRA:", repr(after))
```

```text
原生: 'The capital of France is Paris.'
LoRA: 'The capital of France is Paris, and it is also known as the city of which country? Q. (tài'
原生: 'Three primary colors are red, blue, and yellow.'
LoRA: 'Three primary colors are black, white, which is:\n1. White\n2. white: 1: 5'
```

两个回答都变了，而且变得离题：随机初始化的超网络没有学过任何东西，它写出的 LoRA 只会把模型往随机方向推。
这一步要看的是结构：两个 prompt 在最后一层的 hidden 不同，超网络为它们写出了不同的因子，每个请求的生成都
带上了由它自己的 prompt 决定的那一份 LoRA。换成训练好的超网络，这里就是 Text-to-LoRA 的部署形态。

## 换成训练好的超网络 {#zh-sec-6}

`pllm.FromModule` 接收的是"怎么构造这个模块"的函数，在引擎启动时调用。要用训练好的超网络，就让模块在构造时
从文件读权重，比如给 `HyperNet` 加一个 `checkpoint: str` 参数，在 `__init__` 末尾
`self.load_state_dict(torch.load(checkpoint))`，然后写 `init=pllm.FromModule(lambda m: HyperNet(..., checkpoint="t2l.pt"))`；或把权重存成 safetensors，写 `pllm.FromModule(..., pretrained="t2l.safetensors")`。
注意这个函数只能引用普通值（字符串、数字），不能直接捕获一个已经建好的模块对象。

## 小结 {#zh-sec-7}

- `pllm.RequestState` 让每个请求有自己的一份参数；`pllm.model.Adapter` 把它加到指定投影上；
- `@pllm.write` 用你的数学写状态，`tokens="prefill"` 的 hidden 让它在 prompt 读完时运行；
- 超网络输出为零时与原生 vLLM 逐字相同：适配器路径本身不引入任何偏差。

## 常见错误 {#zh-sec-8}

这一篇的状态写成了 property（因为形状依赖 `rank`）。两个容易出错的地方：忘了给 property 写返回类型；以及在
`@pllm.write` 里直接改读到的状态，而不是返回新值。

```python
@dataclass(frozen=True)
class Unannotated(pllm.Workload):
    rank: int = 4

    @property
    def lora(self):  # 少了 -> pllm.RequestState
        return pllm.RequestState({"a": pllm.Field(lambda inputs, outputs: (self.rank, inputs)),
                                  "b": pllm.Field(lambda inputs, outputs: (outputs, self.rank))})

    @property
    def apply(self) -> pllm.model.Adapter:
        return pllm.model.Adapter("lora", TARGETS, operator="low_rank")


class InPlaceWrite(pllm.Workload):
    memo = pllm.RequestState({"last": pllm.Field(lambda width, _: (width,))})

    @pllm.write(state="memo", fields=("last",))
    def remember(self, h: torch.Tensor = pllm.hidden(-1, tokens="prefill"),
                 last: torch.Tensor = pllm.parameter("memo", "last")) -> tuple[torch.Tensor]:
        last.copy_(h)  # 想直接把值写进状态
        return (last,)

    @pllm.export(name="memo_norm", max_rows=4, max_cols=1)
    def show(self, last: torch.Tensor = pllm.parameter("memo", "last")) -> torch.Tensor:
        return last.float().norm(dim=-1, keepdim=True)


try:
    pllm.explain(Unannotated(), CONFIG)
except TypeError as error:
    print("Unannotated:", error)
try:
    testing.check(InPlaceWrite(), CONFIG)
except testing.ContractError as error:
    print("InPlaceWrite:", error.failures[0])
```

```text
Unannotated: property 'lora' needs a return annotation; annotate state declarations with -> pllm.RequestState or -> pllm.InstanceState, and other properties with their return type
InPlaceWrite: [CPU execution] ValueError: method `remember` mutates borrowed input `last` in place; inputs and parameters are read-only views. Use out-of-place tensor math or clone() before writing; publish state changes through @pllm.write or @pllm.train.
```

**为什么 property 要写返回类型。** pLLM 在构建负载时要找出哪些属性是状态声明，但它不会为此执行你所有的
property：getter 里可以有任意代码。返回类型 `-> pllm.RequestState` 让它不执行就能认出状态。

**为什么不能原地写。** 方法拿到的 `last` 是状态当前版本的只读视图，同一版本可能正被别的方法、别的请求读着。
原地改它，等于在别人读的时候换掉数据。`@pllm.write` 的约定是返回新值，由 pLLM 把它们作为一个新版本整体发布。
`testing.check` 在 CPU 上就能发现原地写入。

## 自测 {#zh-sec-9}

1. 为什么 prompt 本身不受这个请求的 LoRA 影响，而是从第一个生成的 token 才开始？

<details><summary>答案</summary>

LoRA 要读 prompt 最后一个 token 在最后一层的 hidden 才能算出来，这个值在 prompt 的 forward 结束时才有。
`write_lora` 在那之后写入因子，下一次 forward（第一个 decode token）才读到它。
</details>

2. 两个请求在同一个 batch 里生成，它们的 LoRA 会互相影响吗？

<details><summary>答案</summary>

不会。`lora` 是 `RequestState`，每个请求一份；`Adapter` 对每一行施加这一行所属请求的那份因子。
</details>

3. 第三步里超网络输出为零时与原生 vLLM 逐字相同，这验证了什么？

<details><summary>答案</summary>

验证了适配器的施加路径本身不引入偏差：之后输出的任何变化都来自因子本身，而不是框架。写自己的负载时，
先构造一个"强度为零"的版本和原生 vLLM 对照，是最便宜的正确性检查。
</details>

同样的状态也可以边生成边**训练**出来，而不是一次写入，这就是下一篇
[03 decode 时训练](/pllm/docs/tutorials/03_decode_training/)。

</div>

</div>

<div class="zine-doc-footer">
  <a class="zine-pill-link" data-i18n-zh="← 返回 pLLM 项目与文档中心" data-i18n-en="← Back to pLLM Project &amp; Docs" href="/pllm/#docs">← Back to pLLM Project &amp; Docs</a>
  <a class="zine-pill-link" href="https://github.com/LinesHogan/pLLM/blob/main/doc/tutorials/02_request_adapters.ipynb" target="_blank" rel="noopener">GitHub &middot; doc/tutorials/02_request_adapters.ipynb ↗</a>
</div>

<script defer src="/assets/js/bilingual-post.js"></script>
