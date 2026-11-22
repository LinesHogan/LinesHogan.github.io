---
layout: page
title: "04 Custom Forward: Looping Upper Layers Multiple Times"
description: "pLLM Documentation · Step-by-Step Tutorials (00–05)"
permalink: /pllm/docs/tutorials/04_custom_forward/
nav: false
---

<link rel="stylesheet" href="{{ '/assets/css/pllm-hub.css' | relative_url | bust_file_cache }}">

<div class="zine-dispatch" data-title-zh="04 改写 forward：让上半部分的层多跑几遍" data-title-en="04 Custom Forward: Looping Upper Layers Multiple Times" data-desc-zh="pLLM 文档 · 交互式入门教程 (00–05)" data-desc-en="pLLM Documentation · Step-by-Step Tutorials (00–05)">
  <div class="zine-dispatch-links">
    <a class="zine-pill-link primary" data-i18n-zh="← 返回 pLLM 文档中心" data-i18n-en="← pLLM Docs Hub" href="/pllm/#docs">← pLLM Docs Hub</a>
    <a class="zine-pill-link" data-i18n-zh="文档导航地图" data-i18n-en="Docs Roadmap" href="/pllm/docs/">Docs Roadmap</a>
    <span class="zine-lang-badge" data-i18n-zh="可执行教程 (.ipynb)" data-i18n-en="Notebook (.ipynb)">Notebook (.ipynb)</span>
  </div>
  <a class="zine-pill-link" data-i18n-zh="在 GitHub 打开 .ipynb ↗" data-i18n-en="Open .ipynb on GitHub ↗" href="https://github.com/LinesHogan/pLLM/blob/main/doc/tutorials/04_custom_forward.ipynb" target="_blank" rel="noopener">Open .ipynb on GitHub ↗</a>
</div>

<div id="markdown-content">

<div class="zine-lang-panel" data-lang-panel="en" lang="en" markdown="1">

<div id="en-sec-1"></div>

**Scenario**: Looped / recurrent-depth models run a block of layers multiple times per token, trading extra compute for stronger reasoning without adding parameters—for example, Huginn (Geiping et al. 2025) and Ouro (2025). Ouro also lets each token decide when to stop looping. Testing such architectures in a real serving engine normally requires rewriting vLLM's model forward pass and figuring out what happens to the KV cache on each iteration: write it? overwrite it? or read-only?

`@pllm.custom_forward` lets you orchestrate a token's forward pass in plain Python: model layers are exposed as range-callable functions, you write tensor math between layer spans, and a single parameter specifies KV cache semantics.

**To be clear upfront**: Qwen2.5 was not trained for looping, so repeating its layers won't make it smarter. This tutorial demonstrates the **mechanism**—the exact same code works unchanged on a recurrent-depth checkpoint from the same model family.

```python
import os

os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"  # Show only vLLM errors to suppress startup logs

from dataclasses import dataclass

import torch
from transformers import AutoTokenizer
from vllm import SamplingParams

import pllm

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
CONFIG = pllm.ModelConfig(MODEL, max_model_len=512, max_num_seqs=4, max_num_batched_tokens=512,
                          gpu_memory_utilization=0.3, seed=0)
```

## Step 1: A Looping Workload {#en-step-1-a-looping-workload}

The `think` method receives `layers` (`pllm.model.Layers`):

- `layers()` runs the full model and returns the residual stream `taps` after every layer, where `taps[-1]` is the final layer's output;
- `layers(start=s, hidden=h)` runs from layer `s` through the last layer starting from `h`;
- The method's return tensor is what gets sent to the final norm and LM head.

Every `layers(...)` call computes K and V for those layers. The default `kv="commit"` writes them into the KV cache. Writing on every iteration of a loop overwrites the previous iteration, so later tokens see the K and V from this token's final iteration. The Ouro paper (arXiv 2510.25741, §5.4.2 Table 14) compared decode-time KV strategies for looping: storing every iteration separately scored 78.92 on GSM8K; keeping only the last iteration scored 78.85; keeping only the first iteration collapsed to 18.73. Committing on every pass corresponds to "last iteration only," whereas writing on the first pass and reading without writing afterward corresponds to "first iteration only" and breaks a loop-trained model. `kv="peek"` reads prior cached positions alongside the current pass's freshly computed K and V without writing back; it is meant for lookahead trials that leave no trace (such as speculative rollouts), not for recurrent loops here.

`loops` sets the iteration count, and `start` sets the first layer of the repeated span. To keep a non-loop-trained model from collapsing immediately, each iteration blends its output at fraction `mix` (`mix=1` is a full replacement loop). `tol` implements Ouro-style per-token early stopping: once an iteration's relative change on a token falls below `tol`, later iterations leave that token unchanged. Notice that the loop uses a fixed Python `for` bound and expresses early stopping via `torch.where` rather than `if`: control flow in forward cannot depend on tensor values, ensuring bounded per-token compute.

```python
@dataclass(frozen=True)
class LatentLoop(pllm.Workload):
    start: int = 20
    loops: int = 0
    mix: float = 1.0
    tol: float = 0.0

    @pllm.custom_forward(tokens="all")
    def think(self, layers: pllm.model.Layers) -> torch.Tensor:
        hidden = layers()[-1]
        done = torch.zeros_like(hidden[:, :1], dtype=torch.bool)
        for _ in range(self.loops):
            again = layers(start=self.start, hidden=hidden)[-1]  # kv="commit": write back to KV on every pass
            step = torch.lerp(hidden, again, self.mix)
            change = (step - hidden).float().norm(dim=-1, keepdim=True) / hidden.float().norm(dim=-1, keepdim=True)
            hidden = torch.where(done, hidden, step)
            done = done | (change < self.tol)
        return hidden
```

## Step 2: `loops=0` Is the Original Model {#en-step-2-loops0-is-the-original-model}

With zero extra loops, `think` simply runs the whole model once. First verify that it matches plain vLLM word-for-word, confirming `custom_forward` itself alters no computation.

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL)
questions = ["What is the capital of France? Answer in one sentence.", "Name three primary colors."]
prompts = [tokenizer.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                         add_generation_prompt=True) for q in questions]
greedy = SamplingParams(max_tokens=32, temperature=0.0)

with pllm.make_llm(CONFIG) as plain:
    baseline = [output.outputs[0].text for output in plain.generate(prompts, greedy, use_tqdm=False)]


def run(workload: LatentLoop) -> list[str]:
    with pllm.make_llm(CONFIG, workload=workload) as llm:
        return [output.outputs[0].text for output in llm.generate(prompts, greedy)]


texts = run(LatentLoop(loops=0))
print("与原生 vLLM 相同:", texts == baseline)
for text in texts:
    print("   ", repr(text))
```

```text
与原生 vLLM 相同: True
    'The capital of France is Paris.'
    'Three primary colors are red, blue, and yellow.'
```

## Step 3: Turning On the Loop {#en-step-3-turning-on-the-loop}

Three configurations: one pure loop from layer 20 (`mix=1`); one loop from layer 20 mixed at 10%; and three loops from layer 20 mixed at 10% with early stopping enabled (`tol=0.05`) (this 0.5B model has 24 layers, indexed 0–23).

```python
for workload in (LatentLoop(loops=1, mix=1.0), LatentLoop(loops=1, mix=0.1), LatentLoop(loops=3, mix=0.1, tol=0.05)):
    print(workload)
    for text in run(workload):
        print("   ", repr(text))
```

```text
LatentLoop(start=20, loops=1, mix=1.0, tol=0.0)
    'The\xa0capital of the\xa0, the\xa0, and the 1,116 other ordinary, ordinary, ordinary, ordinary, and, and, and'
    '«I, 🙂\\, would indeed indeed indeed indeed indeed indeed indeed indeed indeed indeed indeed indeed indeed indeed indeed indeed indeed,\xa0‘‘‘‘‘‘'
LatentLoop(start=20, loops=1, mix=0.1, tol=0.0)
    'The capital of France is Paris. It is the largest city in the country and the administrative center of the country. The city is known for its historical landmarks,'
    'The three primary colors are:\n\n1. Red\n2. Blue\n3. Blue\n\nThese three colors are the most common and commonly used in the field of'
LatentLoop(start=20, loops=3, mix=0.1, tol=0.05)
    'The capital of the country is indeed, and is indeed, the city of the same name, which is the capital of the country, the Republic of the Republic'
    'The three primary colors are: **Red, Blue, and Yellow.»»»»»»»»»»»»»»»»»»'
```

Three outcomes:

- **Pure loop** (`mix=1`) breaks a model never trained for recurrence: layers 20–23 have never seen their own outputs fed back as inputs, and subsequent tokens attend to the K and V written by that pass;
- **One loop at 10% mix**: Answers remain coherent; the first answer gets slightly longer (adding extra sentences about Paris), while the second turns yellow into a second Blue;
- **Three loops at 10% mix**: The first answer drops "France / Paris" and the second ends in repeated symbols—starting to diverge.

On a model not trained for recurrence, stronger looping causes more drift; comparing loop counts, start layers, and stopping thresholds meaningfully requires a loop-trained checkpoint, with no code changes needed here.

## Runtime Cost {#en-runtime-cost}

Because `custom_forward` re-invokes layer spans from Python every token, vLLM's monolithic whole-model CUDA graph is no longer used, and `diagnostics().eager_reason` explains why; instead, pLLM schedules the model in segments and captures its own CUDA graphs per segment ([Example 12](/pllm/docs/examples/12_adaptive_depth/) measures an unskipped custom forward on Qwen2.5-7B at 32 concurrent requests at 99% of native CUDA graph throughput). On multi-GPU setups, it supports tensor parallelism on vLLM 0.32.0. It only supports models whose entire KV state resides in the paged cache (dense Qwen2/Llama/Qwen3 and Qwen3-MoE; Qwen3.5's DeltaNet layers carry recurrent state outside the paged cache and are unsupported).

## What Else You Can Build {#en-what-else-you-can-build}

- **Feed the previous step's hidden state as the next step's input** (Coconut-style continuous thought): Declare a `pllm.PositionState`, save each position's last-layer output with `layers.save(...)`, and load it on the next token with `layers.load(...)` as `layers(hidden=...)`. See `pllm.workloads.StepLoopWorkload`;
- **Cross-layer recirculation**: Norm-match deep outputs, blend them back into shallow layers, and rerun upper layers—see `pllm.workloads.RecirculationWorkload`;
- `@pllm.custom_forward(tokens="prefill", over="span")` hands you the whole prompt span at once for Jacobi-style parallel iterations—see `pllm.workloads.JacobiRecirculationWorkload`.

## Common Mistakes {#en-common-mistakes}

Taking full control of the forward pass comes with two easy-to-hit constraints: per-token early exit (`active=`) is restricted to decode tokens, and `pllm.hidden` cannot be read in the same workload.

```python
@dataclass(frozen=True)
class ExitEverywhere(pllm.Workload):
    @pllm.custom_forward(tokens="all")  # Skip layers with active= on prompt tokens too
    def think(self, layers: pllm.model.Layers) -> torch.Tensor:
        hidden = layers()[-1]
        active = hidden.float().norm(dim=-1, keepdim=True) > 1.0
        return layers(start=20, hidden=hidden, active=active)[-1]


class LoopAndProbe(LatentLoop):
    @pllm.export(name="mlp_out", max_rows=4, max_cols=1)
    def probe(self, h: torch.Tensor = pllm.hidden(21)) -> torch.Tensor:  # Which iteration does layer 21 output belong to?
        return h.float().norm(dim=-1, keepdim=True)


for workload in (ExitEverywhere(), LoopAndProbe(loops=2)):
    try:
        pllm.explain(workload, CONFIG)
    except ValueError as error:
        print(f"{type(workload).__name__}: {error}\n")
```

```text
ExitEverywhere: Use @pllm.custom_forward(tokens="decode") for active=: skipped prompt tokens would leave approximate K/V that every later token reads. Prompt rows run the full model by default; add an unmasked tokens="prefill" method only to change prompt computation (prefer over="span").

LoopAndProbe: probe reads block or model outputs, which a @pllm.custom_forward does not produce; use pllm.residual or pllm.layer_logits to read each layer's last visited stream
```

**Why `active=` is decode-only.** Frozen rows skip later layers, so K and V for those skipped layers can only be approximated. Skipping layers on a single generated token affects only a few later positions; skipping them on prompt tokens corrupts K and V that every subsequent token will attend to. Prompts therefore run the full model by default.

**Why `pllm.hidden` is disallowed.** Inside a loop, layer 21 may execute multiple times per token—or zero times when skipping—so "layer 21's output" no longer has a single well-defined meaning. `pllm.residual(layer)` and `pllm.layer_logits(...)` read the residual stream from the **last visited** pass at that layer, which is unambiguous and works alongside `custom_forward`.

In addition, loop counts must be static Python `for` loops, and early stopping must use `torch.where` and boolean masks rather than `if`, `break`, or `while`: the forward pass is captured into CUDA graphs, so the sequence of executed ops at each step must be fixed ahead of time.

## Self-Check {#en-self-check}

1. `layers(...)` defaults to `kv="commit"`. If pass 1 uses `commit` and subsequent passes use `kv="peek"`, which pass's K and V do later tokens read?

<details><summary>Answer</summary>

Pass 1's. `peek` reads without writing, leaving pass 1's K and V in the cache—corresponding to the "first iteration only" strategy in the Ouro paper, which severely degrades loop-trained models.
</details>

2. What does matching plain vLLM identically at `loops=0` demonstrate?

<details><summary>Answer</summary>

That `custom_forward` itself—wrapping layers into callable spans and scheduling them in segments via pLLM—alters no computation; any change when looping comes purely from your loop logic.
</details>

3. With `custom_forward`, `eager_reason` is non-empty. Does that mean the model falls back to ungraphed op-by-op execution at half speed?

<details><summary>Answer</summary>

No. vLLM's own whole-model CUDA graph is bypassed, but pLLM captures its own graphs for each layer segment. In [12 Adaptive Depth](/pllm/docs/examples/12_adaptive_depth/), a non-skipping custom forward on Qwen2.5-7B with 32 concurrent requests achieves ~99% of native CUDA graph throughput.
</details>

Next up, [05 Feedback Loop](/pllm/docs/tutorials/05_host_feedback/) returns to training: closing the loop of generation, majority voting, and weight updates inside a single engine.

</div>

<div class="zine-lang-panel is-hidden" data-lang-panel="zh" lang="zh-CN" markdown="1">

<div id="zh-sec-1"></div>

**场景**：循环深度（looped / recurrent-depth）模型在每个 token 上把一段层重复跑好几遍，用更多计算
换更强的推理，而不增加参数，比如 Huginn（Geiping et al. 2025）和 Ouro（2025）。Ouro 还让每个 token
自己决定什么时候停。想在真实服务里验证这类结构，你得重写 vLLM 的模型 forward，还得想清楚每一遍
循环的 KV 缓存怎么处理：写进去？覆盖？还是只读不写？

`@pllm.custom_forward` 让你用普通 Python 重写一个 token 的 forward：模型的层被包装成可以按区间调用的
函数，层与层之间写你自己的张量数学；KV 怎么处理用一个参数说清楚。

**先说清楚**：Qwen2.5 没有为循环训练过，把它的层多跑几遍不会让它更聪明。这一篇演示的是**结构**：
同样的代码可以原封不动地用在一个为循环训练过的同家族检查点上。

```python
import os

os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"  # 只显示 vLLM 的报错，省掉启动日志

from dataclasses import dataclass

import torch
from transformers import AutoTokenizer
from vllm import SamplingParams

import pllm

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
CONFIG = pllm.ModelConfig(MODEL, max_model_len=512, max_num_seqs=4, max_num_batched_tokens=512,
                          gpu_memory_utilization=0.3, seed=0)
```

## 第一步：一个循环负载 {#zh-sec-2}

`think` 方法收到 `layers`（`pllm.model.Layers`）：

- `layers()` 跑完整个模型，返回每一层之后的残差流 `taps`，`taps[-1]` 是最后一层的输出；
- `layers(start=s, hidden=h)` 从第 s 层开始、以 `h` 为输入跑到最后一层；
- 方法的返回值就是送去最终 norm 和 LM head 的那个向量。

每一次 `layers(...)` 调用都会算出这些层的 K、V。默认 `kv="commit"`：把它们写进 KV 缓存。循环里每一遍都写，后一遍覆盖前一遍，所以后面的 token 看到的是这个 token 最后一遍循环的 K、V。Ouro 论文（arXiv 2510.25741，§5.4.2 表 14）比较了 decode 时循环的几种 KV 用法：每一遍各存一份时 GSM8K 78.92；只留最后一遍 78.85；只留第一遍只有 18.73。每遍都写回就是“只留最后一遍”；如果第一遍写、之后只读不写，相当于“只留第一遍”，会让一个按循环训练过的模型崩掉。`kv="peek"` 读之前位置的缓存加上自己这一遍新算的 K、V，但不写回；它适合“往前试算一步、不留痕迹”的场景（例如投机式的预演），不用于这里的循环。

`loops` 是循环次数，`start` 是被循环的那段层的起点。为了让没训练过循环的模型不至于崩掉，每一遍的结果只按比例 `mix` 混进去（`mix=1` 就是纯粹的循环）。`tol` 是 Ouro 式的逐 token 早停：某个 token 一遍循环带来的相对变化小于 `tol`，之后的循环对它就不再生效。注意循环次数是固定的 Python `for`，早停用 `torch.where` 表达，而不是 `if`：forward 里的控制流不能依赖张量的值，每个 token 的计算量有上界。

```python
@dataclass(frozen=True)
class LatentLoop(pllm.Workload):
    start: int = 20
    loops: int = 0
    mix: float = 1.0
    tol: float = 0.0

    @pllm.custom_forward(tokens="all")
    def think(self, layers: pllm.model.Layers) -> torch.Tensor:
        hidden = layers()[-1]
        done = torch.zeros_like(hidden[:, :1], dtype=torch.bool)
        for _ in range(self.loops):
            again = layers(start=self.start, hidden=hidden)[-1]  # kv="commit"：每一遍都写回 KV
            step = torch.lerp(hidden, again, self.mix)
            change = (step - hidden).float().norm(dim=-1, keepdim=True) / hidden.float().norm(dim=-1, keepdim=True)
            hidden = torch.where(done, hidden, step)
            done = done | (change < self.tol)
        return hidden
```

## 第二步：`loops=0` 就是原来的模型 {#zh-sec-3}

不循环时，`think` 只是把整个模型跑一遍。先确认它和原生 vLLM 逐字一致：这说明 `custom_forward` 本身
没有改变任何计算。

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL)
questions = ["What is the capital of France? Answer in one sentence.", "Name three primary colors."]
prompts = [tokenizer.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                         add_generation_prompt=True) for q in questions]
greedy = SamplingParams(max_tokens=32, temperature=0.0)

with pllm.make_llm(CONFIG) as plain:
    baseline = [output.outputs[0].text for output in plain.generate(prompts, greedy, use_tqdm=False)]


def run(workload: LatentLoop) -> list[str]:
    with pllm.make_llm(CONFIG, workload=workload) as llm:
        return [output.outputs[0].text for output in llm.generate(prompts, greedy)]


texts = run(LatentLoop(loops=0))
print("与原生 vLLM 相同:", texts == baseline)
for text in texts:
    print("   ", repr(text))
```

```text
与原生 vLLM 相同: True
    'The capital of France is Paris.'
    'Three primary colors are red, blue, and yellow.'
```

## 第三步：循环起来 {#zh-sec-4}

三组设置：从第 20 层起纯循环一遍（`mix=1`）；从第 20 层起循环一遍、只混入 10%；从第 20 层起循环三遍、每遍混入 10%，并打开早停（`tol=0.05`）（这个 0.5B 模型共 24 层，编号 0–23）。

```python
for workload in (LatentLoop(loops=1, mix=1.0), LatentLoop(loops=1, mix=0.1), LatentLoop(loops=3, mix=0.1, tol=0.05)):
    print(workload)
    for text in run(workload):
        print("   ", repr(text))
```

```text
LatentLoop(start=20, loops=1, mix=1.0, tol=0.0)
    'The\xa0capital of the\xa0, the\xa0, and the 1,116 other ordinary, ordinary, ordinary, ordinary, and, and, and'
    '«I, 🙂\\, would indeed indeed indeed indeed indeed indeed indeed indeed indeed indeed indeed indeed indeed indeed indeed indeed indeed,\xa0‘‘‘‘‘‘'
LatentLoop(start=20, loops=1, mix=0.1, tol=0.0)
    'The capital of France is Paris. It is the largest city in the country and the administrative center of the country. The city is known for its historical landmarks,'
    'The three primary colors are:\n\n1. Red\n2. Blue\n3. Blue\n\nThese three colors are the most common and commonly used in the field of'
LatentLoop(start=20, loops=3, mix=0.1, tol=0.05)
    'The capital of the country is indeed, and is indeed, the city of the same name, which is the capital of the country, the Republic of the Republic'
    'The three primary colors are: **Red, Blue, and Yellow.»»»»»»»»»»»»»»»»»»'
```

三组结果：

- **纯循环**（`mix=1`）让没训练过循环的模型崩掉：第 20–23 层从没见过自己的输出作为输入，而且后面的 token 读到的是这一遍写下的 K、V；
- **只混入 10% 的一遍循环**：回答仍通顺，第一个回答比原来长了（多说了几句巴黎），第二个回答把黄色说成了第二个 Blue；
- **循环三遍、每遍 10%**：第一个回答丢了 "France / Paris"，第二个回答末尾出现重复符号；开始偏离。

在这个没为循环训练过的模型上，循环越强，回答偏离越多；要比较循环次数、起点和早停阈值，得换上为循环训练过的检查点，代码不用改。

## 运行时的代价 {#zh-sec-5}

`custom_forward` 每个 token 都在 Python 描述的程序里重跑层，所以 vLLM 自己的整模型 CUDA graph 不再使用，
`diagnostics().eager_reason` 会写明原因；模型的层由 pLLM 按段调度，pLLM 为这些段捕获自己的 CUDA graph
（[示例 12](/pllm/docs/examples/12_adaptive_depth/) 里在 Qwen2.5-7B、32 个并发请求上实测，一个不跳层的 custom forward 是原生 CUDA graph 吞吐的 99%）。
多卡时，它可以在 vLLM 0.32.0 上用张量并行。它只支持 KV 全部在分页缓存里的模型（稠密 Qwen2/Llama/Qwen3 和 Qwen3-MoE；Qwen3.5 的 DeltaNet 层有缓存之外的
递归状态，不支持）。

## 还能写什么 {#zh-sec-6}

- **把上一步的 hidden 当这一步的输入**（Coconut 式连续思维）：声明一个 `pllm.PositionState`，用
  `layers.save(...)` 存下每个位置的最后一层输出，下一个 token 用 `layers.load(...)` 读回来，作为
  `layers(hidden=...)` 的输入。现成例子是 `pllm.workloads.StepLoopWorkload`；
- **层间再循环**：把深层的输出按范数匹配后混回浅层，再跑一遍上面的层，见 `pllm.workloads.RecirculationWorkload`；
- `@pllm.custom_forward(tokens="prefill", over="span")` 把整段 prompt 一起交给你，可以写 Jacobi 式的并行迭代，
  见 `pllm.workloads.JacobiRecirculationWorkload`。

## 常见错误 {#zh-sec-7}

`custom_forward` 把整个 forward 交给你，有两个约束容易踩到：逐 token 提前停止（`active=`）只能用于 decode token；
同一个负载里不能再用 `pllm.hidden` 读某一层的输出。

```python
@dataclass(frozen=True)
class ExitEverywhere(pllm.Workload):
    @pllm.custom_forward(tokens="all")  # prompt token 也按 active= 跳层
    def think(self, layers: pllm.model.Layers) -> torch.Tensor:
        hidden = layers()[-1]
        active = hidden.float().norm(dim=-1, keepdim=True) > 1.0
        return layers(start=20, hidden=hidden, active=active)[-1]


class LoopAndProbe(LatentLoop):
    @pllm.export(name="mlp_out", max_rows=4, max_cols=1)
    def probe(self, h: torch.Tensor = pllm.hidden(21)) -> torch.Tensor:  # 第 21 层的输出是哪一遍的？
        return h.float().norm(dim=-1, keepdim=True)


for workload in (ExitEverywhere(), LoopAndProbe(loops=2)):
    try:
        pllm.explain(workload, CONFIG)
    except ValueError as error:
        print(f"{type(workload).__name__}: {error}\n")
```

```text
ExitEverywhere: Use @pllm.custom_forward(tokens="decode") for active=: skipped prompt tokens would leave approximate K/V that every later token reads. Prompt rows run the full model by default; add an unmasked tokens="prefill" method only to change prompt computation (prefer over="span").

LoopAndProbe: probe reads block or model outputs, which a @pllm.custom_forward does not produce; use pllm.residual or pllm.layer_logits to read each layer's last visited stream
```

**为什么 `active=` 只用于 decode。** 被冻结的行跳过了后面的层，这些层的 K、V 只能近似地补上。一个生成的
token 跳层只影响它自己之后的少数位置；prompt token 的 K、V 却会被之后每一个 token 读到，近似值会污染整个生成。
所以 prompt 默认跑完整模型。

**为什么不能读 `pllm.hidden`。** 在循环里第 21 层每个 token 跑了好几遍，跳层时又可能一遍都不跑，"第 21 层的
输出"不再是唯一的值。`pllm.residual(layer)` 和 `pllm.layer_logits(...)` 读的是这一层**最后一次**算出的残差流，
含义明确，可以和 `custom_forward` 一起用。

另外，循环次数要写成固定的 Python `for`，提前停止用 `torch.where` 和掩码表达，不能写 `if`、`break` 或
`while`：forward 被录进 CUDA graph，每一步执行的算子必须事先确定。

## 自测 {#zh-sec-8}

1. `layers(...)` 默认 `kv="commit"`。如果循环的第一遍用 `commit`、之后几遍改成 `kv="peek"`，后面的 token 会读到哪一遍的 K、V？

<details><summary>答案</summary>

第一遍的。`peek` 只读不写，所以缓存里留下的是第一遍写入的 K、V，对应 Ouro 论文里"只留第一遍"的做法，
它会让为循环训练过的模型明显变差。
</details>

2. `loops=0` 时输出与原生 vLLM 逐字相同，说明了什么？

<details><summary>答案</summary>

`custom_forward` 本身（把层包装成可调用的区间、由 pLLM 按段调度）不改变计算；循环带来的任何差异都来自你的
循环逻辑。
</details>

3. 用了 `custom_forward` 之后，`eager_reason` 不再为空。这是否意味着模型退回逐算子执行、慢了一倍？

<details><summary>答案</summary>

不是。vLLM 自己的整模型 CUDA graph 不再使用，但 pLLM 为各段层捕获了自己的 graph。示例库中
[12 自适应深度](/pllm/docs/examples/12_adaptive_depth/) 在 Qwen2.5-7B、32 个并发请求上实测，一个不跳层的 custom forward 约为原生 CUDA graph 吞吐的 99%。
</details>

下一篇 [05 反馈循环](/pllm/docs/tutorials/05_host_feedback/) 回到训练：生成、投票、更新在同一个引擎里闭环。

</div>

</div>

<div class="zine-doc-footer">
  <a class="zine-pill-link" data-i18n-zh="← 返回 pLLM 项目与文档中心" data-i18n-en="← Back to pLLM Project &amp; Docs" href="/pllm/#docs">← Back to pLLM Project &amp; Docs</a>
  <a class="zine-pill-link" href="https://github.com/LinesHogan/pLLM/blob/main/doc/tutorials/04_custom_forward.ipynb" target="_blank" rel="noopener">GitHub &middot; doc/tutorials/04_custom_forward.ipynb ↗</a>
</div>

<script defer src="/assets/js/bilingual-post.js"></script>
