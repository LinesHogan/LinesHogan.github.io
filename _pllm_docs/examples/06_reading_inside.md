---
layout: page
title: "06 Reading Inside the Model: Residual Streams, Per-Layer Logits, and DoLa"
description: "pLLM Documentation · Algorithm Family Examples (06–13)"
permalink: /pllm/docs/examples/06_reading_inside/
nav: false
---

<link rel="stylesheet" href="{{ '/assets/css/pllm-hub.css' | relative_url | bust_file_cache }}">

<div class="zine-dispatch" data-title-zh="06 读模型内部：残差流、逐层 logits 与 DoLa" data-title-en="06 Reading Inside the Model: Residual Streams, Per-Layer Logits, and DoLa" data-desc-zh="pLLM 文档 · 算法族示例与实测 (06–13)" data-desc-en="pLLM Documentation · Algorithm Family Examples (06–13)">
  <div class="zine-dispatch-links">
    <a class="zine-pill-link primary" data-i18n-zh="← 返回 pLLM 文档中心" data-i18n-en="← pLLM Docs Hub" href="/pllm/#docs">← pLLM Docs Hub</a>
    <a class="zine-pill-link" data-i18n-zh="文档导航地图" data-i18n-en="Docs Roadmap" href="/pllm/docs/">Docs Roadmap</a>
    <span class="zine-lang-badge" data-i18n-zh="可执行教程 (.ipynb)" data-i18n-en="Notebook (.ipynb)">Notebook (.ipynb)</span>
  </div>
  <a class="zine-pill-link" data-i18n-zh="在 GitHub 打开 .ipynb ↗" data-i18n-en="Open .ipynb on GitHub ↗" href="https://github.com/LinesHogan/pLLM/blob/main/doc/examples/06_reading_inside.ipynb" target="_blank" rel="noopener">Open .ipynb on GitHub ↗</a>
</div>

<div id="markdown-content">

<div class="zine-lang-panel" data-lang-panel="en" lang="en" markdown="1">

<div id="en-sec-1"></div>

If you have already read Tutorial 00 (activation steering) and Tutorial 01 (probes and `@pllm.export`), you might wonder what actually happens at each layer inside the model. In Hugging Face, passing `output_hidden_states=True` gives you every layer's residual stream, which you can multiply by the LM head for a logit lens. In a vLLM serving engine, however, intermediate tensors from concurrent requests are packed together and execution is captured into CUDA graphs, making those intermediate states hard to inspect directly. As a result, you cannot observe layer-by-layer token predictions during serving, nor run decoding algorithms like DoLa that rely on per-layer logits.

In pLLM, doing so takes only two input markers: `pllm.residual(layer)` reads the residual stream, and `pllm.layer_logits(layers)` computes the per-layer logit lens—and both run inside CUDA graphs (with an empty `eager_reason`). Moreover, DoLa is already included as a built-in workload in pLLM in just a few dozen lines of code.

In this example, we walk through three steps:
1. Read the residual stream to answer the question left open in Tutorial 00: "How large should a steering vector be?"
2. Run a logit lens to see at which layer the correct answer emerges.
3. Run DoLa to alter decoding using per-layer logits, and measure both its output changes and its throughput cost.

Let's load the model and tokenizer first.

```python
import os

os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"  # Show only vLLM errors and suppress startup logs

from collections import defaultdict

import torch
from transformers import AutoTokenizer
from vllm import SamplingParams

import pllm
from pllm.benchmark import throughput
from pllm.workloads import DoLaWorkload

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
CONFIG = pllm.ModelConfig(MODEL, max_model_len=512, max_num_seqs=32, max_num_batched_tokens=4096,
                          gpu_memory_utilization=0.5, seed=7)
```

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL)


def chat(question: str) -> str:
    return tokenizer.apply_chat_template([{"role": "user", "content": question}], tokenize=False,
                                         add_generation_prompt=True)


greedy = SamplingParams(max_tokens=32, temperature=0.0)
```

## Step 1: Residual Stream — How Large Should a Steering Vector Be? {#en-step-1-residual-stream--how-large-should-a-steering-vector-be}

In this step, we answer a practical question left open in Tutorial 00: when doing activation steering, is the steering vector we add actually large or small?

First, let's clarify what the residual stream is in plain terms: after each layer's attention and MLP compute their outputs, they _add_ those outputs onto a vector running through the entire model—that vector is the residual stream. The next layer reads from it as input, and when we steer activations, that same stream is what we add our vector into.

In pLLM, `pllm.residual(L)` denotes the residual stream _after_ layer `L` (that is, the initial embedding plus the outputs of all sublayers from layer `0` through layer `L`). Note how this differs from `pllm.hidden(L)` used in Tutorial 01: `pllm.hidden(L)` is only the portion newly computed by layer `L`'s MLP before it is added back into the residual stream.

Recall from Tutorial 00 that we added `strength × steering_vector` onto the output of layer 12. That steering vector had norm `7.09`, and we chose strengths of `±2`. The residual stream after layer 12's output corresponds directly to `pllm.residual(12)`. To see how large the added vector is relative to the residual stream itself, let's write a workload that only exports norms: in a single method, we read the residual streams after layers 6, 12, 18, and 23, compute the vector norm at every token position, and export them.

```python
norms: dict[int, list[list[float]]] = defaultdict(list)
WATCHED = (6, 12, 18, 23)


class StreamSize(pllm.Workload):
    @pllm.export(name="stream_norm", max_rows=4, max_cols=4)
    def size(self, r6: torch.Tensor = pllm.residual(6), r12: torch.Tensor = pllm.residual(12),
             r18: torch.Tensor = pllm.residual(18), r23: torch.Tensor = pllm.residual(23)) -> torch.Tensor:
        return torch.stack([r.float().norm(dim=-1) for r in (r6, r12, r18, r23)], dim=-1)

    def on_export(self, event: pllm.io.TraceEvent) -> None:
        for identity, row in zip(event.identities, event.tensor().tolist()):
            norms[identity.prompt_index].append(row)
```

```python
prompts = [chat("Write one sentence about my new job."), chat("Describe your weekend in one sentence.")]
with pllm.make_llm(CONFIG, workload=StreamSize()) as llm:
    llm.generate(prompts, greedy, use_tqdm=False)
    print("eager 原因:", repr(llm.diagnostics().eager_reason))

rows = torch.tensor([row for index in sorted(norms) for row in norms[index]])
for layer, mean in zip(WATCHED, rows.mean(0).tolist()):
    print(f"第 {layer:2d} 层之后的残差流：平均范数 {mean:5.1f}")
print(f"教程 00 在第 12 层加的向量：范数 7.09 × 强度 2 = {7.09 * 2:.1f}")
```

```text
eager 原因: ''
第  6 层之后的残差流：平均范数  13.3
第 12 层之后的残差流：平均范数  15.9
第 18 层之后的残差流：平均范数  35.5
第 23 层之后的残差流：平均范数  55.2
教程 00 在第 12 层加的向量：范数 7.09 × 强度 2 = 14.2
```

From the output, the residual stream norm grows steadily in deeper layers: the mean norm is `13.3` after layer 6, `15.9` after layer 12, `35.5` after layer 18, and `55.2` by layer 23. In Tutorial 00, adding our steering vector at layer 12 with strength `2` added a vector of norm `7.09 × 2 = 14.2`—almost as large as the mean norm of the layer-12 residual stream itself (`15.9`). That gives a concrete reference point: strength `2` roughly overhauls the layer-12 residual stream (and as Tutorial 00 noted, pushing a small model too hard causes repetition). Adding a vector of the same magnitude at a deeper layer would be much smaller relative to the residual stream there. Also note that `eager_reason` is `''` (printed as `eager 原因: ''` above), confirming that reading the residual stream does not require disabling CUDA graphs.

```python
LAYERS = tuple(range(24))
lens: dict[int, list[list[int]]] = defaultdict(list)


class LogitLens(pllm.Workload):
    @pllm.export(name="lens", max_rows=4, max_cols=len(LAYERS))
    def top1(self, logits: torch.Tensor = pllm.layer_logits(LAYERS)) -> torch.Tensor:
        return logits.argmax(dim=-1).float()  # Most likely token at each layer, [rows, layers]

    def on_export(self, event: pllm.io.TraceEvent) -> None:
        for identity, row in zip(event.identities, event.tensor().long().tolist()):
            lens[identity.prompt_index].append(row)
```

```python
with pllm.make_llm(CONFIG, workload=LogitLens()) as llm:
    seven = SamplingParams(max_tokens=7, temperature=0.0)  # Answer is exactly 7 tokens
    output = llm.generate([chat("What is the capital of France? Answer in one sentence.")], seven, use_tqdm=False)[0]
    print("eager 原因:", repr(llm.diagnostics().eager_reason))

generated = list(output.outputs[0].token_ids)
print("回答:", repr(output.outputs[0].text))
# Row 0 comes from the last prompt token; row k comes from the step taking the k-th generated token as input;
# the final layer in row k predicts the k-th generated token (0-indexed).
print("最后一层的预测与实际生成一致:", all(row[-1] == generated[k] for k, row in enumerate(lens[0])))

step = next(k for k, row in enumerate(lens[0]) if tokenizer.decode(row[-1]) == " Paris")
print(f"预测 ' Paris' 的那一步，每一层最可能的 token：")
for layer, token in zip(LAYERS, lens[0][step]):
    print(f"  第 {layer:2d} 层: {tokenizer.decode(token)!r}")
```

```text
eager 原因: ''
回答: 'The capital of France is Paris.'
最后一层的预测与实际生成一致: True
预测 ' Paris' 的那一步，每一层最可能的 token：
  第  0 层: '\';";\n'
  第  1 层: '\';";\n'
  第  2 层: ".','"
  第  3 层: ".','"
  第  4 层: ".','"
  第  5 层: '\';";\n'
  第  6 层: "',...\n"
  第  7 层: "',...\n"
  第  8 层: '","#'
  第  9 层: 'announcement'
  第 10 层: 'announcement'
  第 11 层: "',...\n"
  第 12 层: 'уницип'
  第 13 层: '那'
  第 14 层: '\';";\n'
  第 15 层: ';break'
  第 16 层: 'MAND'
  第 17 层: 'MAND'
  第 18 层: ';break'
  第 19 层: '家都知道'
  第 20 层: ' capitals'
  第 21 层: ' Paris'
  第 22 层: ' Paris'
  第 23 层: ' Paris'
```

The model's generated answer is `'The capital of France is Paris.'`. In the output, `True` for matching the final layer's predictions against the actual generated tokens confirms the alignment described above: the final layer's prediction in each row indeed corresponds to the next token actually generated.

Looking closely at the step that predicts `' Paris'`: from layer 0 through layer 19, the top-1 token at each layer consists of fragments unrelated to the question, such as `'\';";\n'`, `'announcement'`, and `';break'`; at layer 20 it shifts to the topically relevant `' capitals'`; and starting at layer 21, the top-1 token becomes `' Paris'` and stays there through layer 23 (the final layer). In this example, the answer takes shape only in the last few layers. Once again, `eager_reason` is empty.

## Step 3: DoLa — Altering Decoding with Per-Layer Logits {#en-step-3-dola--altering-decoding-with-per-layer-logits}

Since shallow layers only carry surface fragments while factual answers emerge in the final layers, can we exploit this contrast between shallow and deep layers to improve decoding? That is what we test next with DoLa (Chuang et al., 2024, _"Decoding by Contrasting Layers"_).

In plain terms, DoLa's idea is straightforward: at each generated token, pick from a set of candidate shallow layers the layer whose prediction diverges most from the final layer (measured by Jensen–Shannon divergence, where larger divergence means greater disagreement). Then re-score the vocabulary by subtracting that shallow layer's log-probabilities from the final layer's log-probabilities. To avoid picking tokens that were implausible to begin with, selection is restricted to tokens that the final layer considers sufficiently probable (probability at least `0.1×` the final layer's maximum probability), masking out all others. The intuition is to suppress surface patterns already guessable in shallow layers while boosting information newly formed in deep layers.

pLLM ships this algorithm as `pllm.workloads.DoLaWorkload` (implemented in `pllm/workloads/dola.py`, with about a dozen lines in its core method): it uses a `@pllm.add_logits` method reading `pllm.layer_logits`, with default candidate shallow layers `0, 2, ..., 14`. To use it, simply pass `DoLaWorkload()` to `make_llm`.

Below we first compare greedy answers from native decoding and DoLa on 4 factual questions, then measure the throughput cost. Here we focus only on output differences and runtime cost, without evaluating whether DoLa improves model accuracy.

Before launching the engine, let's inspect DoLa's plan and check its correctness on CPU. The `forward` line in `explain` tells us whether the model stays in CUDA graphs, `flops_per_token_estimate` combines DoLa's extra per-token compute with the base model, and `testing.check` verifies that DoLa's row-wise math does not mix rows across concurrent requests. DoLa has no strength toggle, so we omit `neutral=` here.

```python
from pllm import testing

report = pllm.explain(DoLaWorkload(), CONFIG)
print("CUDA graph:", "保留" if report.eager_reason == "" else report.eager_reason)
print(f"每 token FLOP 估计：{report.flops_per_token_estimate:.3g}（底座模型约 9.88e+08）")
print(testing.check(DoLaWorkload(), CONFIG))
```

```text
CUDA graph: 保留
每 token FLOP 估计：3.44e+09（底座模型约 9.88e+08）
status  contract / reason
PASS    tensor_parallel_size=1: admission
PASS    compile
PASS    row permutation
PASS    request isolation
PASS    mixed prefill/decode
PASS    chunking / tokens=
PASS    preempted
SKIP    publish_lag: no row method reads versioned state
SKIP    neutral: pass neutral= with the author's disabled settings
```

```python
questions = ["What is the capital of Australia?", "Who wrote the novel 1984?", "How many legs does a spider have?",
             "What is the boiling point of water at sea level in Fahrenheit?"]
answer = SamplingParams(max_tokens=40, temperature=0.0)


def answers(llm) -> list[str]:
    return [output.outputs[0].text for output in llm.generate([chat(q) for q in questions], answer, use_tqdm=False)]


with pllm.make_llm(CONFIG) as plain:
    plain_texts = answers(plain)
with pllm.make_llm(CONFIG, workload=DoLaWorkload()) as llm:
    dola_texts = answers(llm)

for question, a, b in zip(questions, plain_texts, dola_texts):
    print(question)
    print("  原生:", repr(a))
    print("  DoLa:", repr(b))
```

```text
What is the capital of Australia?
  原生: 'The capital of Australia is Canberra.'
  DoLa: 'The capital of Australia is Canberra. Canberra is the capital city of Australia.'
Who wrote the novel 1984?
  原生: 'The novel "1984" was written by George Orwell.'
  DoLa: '1984 is a work of fiction, created by George Orwell. The novel is part of Orwell\'s complete series of books titled "1984" and is set in a dystopian'
How many legs does a spider have?
  原生: 'A spider has eight legs.'
  DoLa: 'A spider has eight legs.'
What is the boiling point of water at sea level in Fahrenheit?
  原生: 'The boiling point of water at sea level in Fahrenheit is 212°F.'
  DoLa: 'The boiling point of water at sea level is 212 degrees Fahrenheit.'
```

Now let's measure the throughput cost. The 0.5B model is sufficient for the inspections above, whereas for throughput we switch to `Qwen/Qwen2.5-7B-Instruct`: a single decode step on 0.5B takes only a few milliseconds, exaggerating the relative share of the LM head and sampler, while 7B is a more typical deployment size (~15 GiB in bf16 weights, requiring a 24 GiB GPU). We test 32 concurrent requests (8 prompts × 4 samples each × 256 generated tokens) using `pllm.benchmark.throughput`: 1 warmup round followed by 3 timed rounds taking the median, where each round times the full `generate` call (including prefill). `DoLaWorkload`'s default candidate shallow layers `0, 2, ..., 14` also apply to the 7B model (28 layers).

```python
BIG = pllm.ModelConfig("Qwen/Qwen2.5-7B-Instruct", max_model_len=512, max_num_seqs=32,
                       max_num_batched_tokens=4096, gpu_memory_utilization=0.85, seed=7)
many = [f"Explain the mathematical idea numbered {i}:" for i in range(8)]
sampling = SamplingParams(n=4, max_tokens=256, temperature=0.8, top_p=0.95, ignore_eos=True, seed=7)

with pllm.make_llm(BIG) as plain:
    plain_rate = throughput(plain, many, sampling)
with pllm.make_llm(BIG, workload=DoLaWorkload()) as llm:
    dola_rate = throughput(llm, many, sampling)
    eager_reason = llm.diagnostics().eager_reason
print(f"7B 吞吐（32 个并发请求，各 256 token）：原生 {plain_rate:.0f} token/s，DoLa {dola_rate:.0f} token/s，"
      f"为原生的 {dola_rate / plain_rate:.0%}；eager 原因 {eager_reason!r}")
```

```text
7B 吞吐（32 个并发请求，各 256 token）：原生 1727 token/s，DoLa 1140 token/s，为原生的 66%；eager 原因 ''
```

First, looking at the output changes: out of the 4 questions, DoLa changed the answer on 3 of them. On the capital of Australia, DoLa appended a redundant sentence after answering Canberra; on the author of _1984_, DoLa's answer grew longer and introduced an inaccurate claim (saying the book belongs to a series by Orwell called "1984"); on how many legs a spider has, both answers were identical; and on the boiling point of water, DoLa rephrased `212°F` as `212 degrees Fahrenheit`. Of course, 4 examples cannot tell us whether DoLa makes this 0.5B model more accurate or worse overall.

Next, looking at the throughput cost (7B, 32 concurrent requests, 256 tokens each; RTX 4090, vLLM 0.32.0, bf16, CUDA graphs, median of 3 timed rounds after 1 warmup): in this run, native throughput is `1727 token/s`, and DoLa reaches `1140 token/s`, or `66%` of native. `eager_reason` remains `''`, meaning DoLa runs inside CUDA graphs throughout. The throughput drop comes from genuine compute: for every generated token, FP32 logits and divergences over the 152,064-token vocabulary must be computed across 9 layers (8 candidate shallow layers plus the final layer). The `explain` estimate reflects this directly: on 7B, DoLa increases per-token compute from ~14.1B FLOPs to ~24.0B FLOPs (`1.7×` the base model). On 0.5B, the same DoLa workload is `3.5×` the base model (cell 11 output: ~0.99B to ~3.44B FLOPs), because the vocabulary projection size does not shrink with model depth—the smaller the model, the larger fraction of a step it occupies, so DoLa's relative overhead looks much heavier on small models.

## What the Framework Does Under the Hood {#en-what-the-framework-does-under-the-hood}

Why can reading residual streams and computing per-layer logits both stay inside CUDA graphs without perturbing the model's own outputs?

1. **Residual stream reads**: The framework attaches reads onto the norm operations already invoked at every layer, copying out the requested rows directly inside the CUDA graph.
2. **Per-layer logits computation**: At the LM head, the framework runs a separate matrix multiplication to compute logits for the selected layers, completely decoupled from the model's own final-logits path—so the model's own logits remain bit-for-bit identical to native vLLM.

## Summary {#en-summary}

- `pllm.residual(layer)` reads the residual stream after a specified layer; comparing norms directly shows how large an activation-steering vector is relative to the residual stream itself.
- `pllm.layer_logits(layers)` provides intermediate-layer logits without breaking CUDA graphs, enabling both logit-lens inspection of where answers form and decoding algorithms that depend on per-layer logits.
- `pllm.workloads.DoLaWorkload` implements DoLa's shallow-vs-deep logit contrast decoding in about a dozen lines of core code; it executes inside CUDA graphs, achieving `66%` of native throughput on 7B due to computing FP32 full-vocabulary logits and divergences across multiple layers on every token.

In the next example, [07 Custom Residual Connections](/pllm/docs/examples/07_custom_residual/), we move from merely reading the residual stream to rewriting how residual updates themselves are computed (`@pllm.custom_residual`).

</div>

<div class="zine-lang-panel is-hidden" data-lang-panel="zh" lang="zh-CN" markdown="1">

<div id="zh-sec-1"></div>

如果你已经读过教程 00（激活引导）和教程 01（探针与 `@pllm.export`），你可能会好奇模型内部每一层到底发生了什么。在 Hugging Face 里，传一个 `output_hidden_states=True` 就能拿到每层的残差流，再乘上 LM head 就能做 logit lens。但在 vLLM 服务里，不同请求的中间结果被拼在一起处理，计算过程还被捕获进了 CUDA graph，你很难直接拿到这些中间状态。这导致你没法在服务里逐 token 观察每一层在预测什么，也没法运行像 DoLa 这样依赖逐层 logits 的解码算法。

在 pLLM 中，这只需要两个输入标记：`pllm.residual(layer)` 用来读取残差流，`pllm.layer_logits(layers)` 用来读取每层的 logit lens，而且它们都能在 CUDA graph 里运行（`eager_reason` 为空）。此外，DoLa 已经作为自带负载包含在 pLLM 里，只有几十行代码。

这个示例我们一步步做三件事：
1. 读取残差流，回答教程 00 留下的问题：“引导向量该多大？”
2. 做一次 logit lens，看看正确答案是在第几层成形的。
3. 运行 DoLa，用逐层 logits 改变解码过程，并测量它的输出变化与吞吐代价。

我们先加载模型和分词器。

```python
import os

os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"  # 只显示 vLLM 的报错，省掉启动日志

from collections import defaultdict

import torch
from transformers import AutoTokenizer
from vllm import SamplingParams

import pllm
from pllm.benchmark import throughput
from pllm.workloads import DoLaWorkload

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
CONFIG = pllm.ModelConfig(MODEL, max_model_len=512, max_num_seqs=32, max_num_batched_tokens=4096,
                          gpu_memory_utilization=0.5, seed=7)
```

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL)


def chat(question: str) -> str:
    return tokenizer.apply_chat_template([{"role": "user", "content": question}], tokenize=False,
                                         add_generation_prompt=True)


greedy = SamplingParams(max_tokens=32, temperature=0.0)
```

## 第一步：残差流 —— 引导向量该多大 {#zh-sec-2}

这一步我们要回答教程 00 留下的一个实际问题：做激活引导时，加进去的引导向量到底算大还是算小？

先用白话明确一下什么是残差流：模型每一层的 attention 和 MLP 算完之后，都会把自己的输出“加”到一条贯穿全模型的向量上，这条向量就是残差流。下一层读取的输入是它，我们在做激活引导时往模型里加向量，加的也是它。

在 pLLM 里，`pllm.residual(L)` 表示第 `L` 层之后的残差流（也就是最初的 embedding 加上从第 0 层到第 `L` 层所有子层的输出）。要注意它和教程 01 用过的 `pllm.hidden(L)` 的区别：`pllm.hidden(L)` 只是第 `L` 层 MLP 新算出来的那一部分，还没有加回残差流。

回顾教程 00，当时我们在第 12 层的输出上加了 `强度 × 引导向量`。那个引导向量的范数是 7.09，我们选用的强度是 ±2。第 12 层输出之后的残差流，正好对应 `pllm.residual(12)`。为了知道加上去的向量相对残差流本身有多大，我们写一个只导出范数的负载：在一个方法里同时读取第 6、12、18、23 层之后的残差流，算出每个 token 位置的向量范数并导出。

```python
norms: dict[int, list[list[float]]] = defaultdict(list)
WATCHED = (6, 12, 18, 23)


class StreamSize(pllm.Workload):
    @pllm.export(name="stream_norm", max_rows=4, max_cols=4)
    def size(self, r6: torch.Tensor = pllm.residual(6), r12: torch.Tensor = pllm.residual(12),
             r18: torch.Tensor = pllm.residual(18), r23: torch.Tensor = pllm.residual(23)) -> torch.Tensor:
        return torch.stack([r.float().norm(dim=-1) for r in (r6, r12, r18, r23)], dim=-1)

    def on_export(self, event: pllm.io.TraceEvent) -> None:
        for identity, row in zip(event.identities, event.tensor().tolist()):
            norms[identity.prompt_index].append(row)
```

```python
prompts = [chat("Write one sentence about my new job."), chat("Describe your weekend in one sentence.")]
with pllm.make_llm(CONFIG, workload=StreamSize()) as llm:
    llm.generate(prompts, greedy, use_tqdm=False)
    print("eager 原因:", repr(llm.diagnostics().eager_reason))

rows = torch.tensor([row for index in sorted(norms) for row in norms[index]])
for layer, mean in zip(WATCHED, rows.mean(0).tolist()):
    print(f"第 {layer:2d} 层之后的残差流：平均范数 {mean:5.1f}")
print(f"教程 00 在第 12 层加的向量：范数 7.09 × 强度 2 = {7.09 * 2:.1f}")
```

```text
eager 原因: ''
第  6 层之后的残差流：平均范数  13.3
第 12 层之后的残差流：平均范数  15.9
第 18 层之后的残差流：平均范数  35.5
第 23 层之后的残差流：平均范数  55.2
教程 00 在第 12 层加的向量：范数 7.09 × 强度 2 = 14.2
```

从输出可以看到，残差流的范数越往后的层越大：第 6 层之后平均范数是 13.3，第 12 层是 15.9，第 18 层是 35.5，到了第 23 层达到 55.2。教程 00 在第 12 层加的向量，在强度为 2 时范数是 `7.09 × 2 = 14.2`，几乎和第 12 层残差流本身的平均范数（15.9）一样大。这给了我们一个直观的参照：强度 2 差不多把第 12 层的残差流改头换面了（教程 00 里也提到，小模型被推得太猛会开始重复）。如果把同样大小的向量加在更深的层，相对残差流本身来说影响就会小得多。同时注意 `eager 原因: ''`，说明读取残差流不需要关掉 CUDA graph。

## 第二步：logit lens —— 答案在第几层成形 {#zh-sec-3}

看完了残差流的大小，我们来看看残差流里的内容是怎么变成最终答案的。这一步我们要回答的问题是：当我们问模型一个简单事实时，正确的 token 是在第几层才浮现出来的？

常用的观察方法是 logit lens：把模型最后那一层才用的 norm 和 LM head，直接套在中间某一层之后的残差流上，看看“如果模型在这一层就停下，它会预测什么词”。

在 pLLM 里，`pllm.layer_logits(layers)` 会一次给出你列出的每一层的 logits（数据类型为 FP32，形状为 `[行, 层数, 词表]`），其中最后一层的 logits 就是模型自己最终输出的 logits。这里顺便说明一下导出数据的对齐方式：导出的结果里每个 decode 步占一行；输入第 `k` 个生成 token 的那一步，最后一层预测的是第 `k + 1` 个生成 token（处理 prompt 最后一个 token 的那一步不在导出里）。

下面我们写一个负载，导出全部 24 层（编号 0 到 23）每一层概率最高的 token（`argmax`）。我们问模型法国的首都是哪里，然后找到预测出 `' Paris'` 的那一步，打印每一层的最可能 token。

```python
LAYERS = tuple(range(24))
lens: dict[int, list[list[int]]] = defaultdict(list)


class LogitLens(pllm.Workload):
    @pllm.export(name="lens", max_rows=4, max_cols=len(LAYERS))
    def top1(self, logits: torch.Tensor = pllm.layer_logits(LAYERS)) -> torch.Tensor:
        return logits.argmax(dim=-1).float()  # 每一层最可能的 token，[行, 层]

    def on_export(self, event: pllm.io.TraceEvent) -> None:
        for identity, row in zip(event.identities, event.tensor().long().tolist()):
            lens[identity.prompt_index].append(row)
```

```python
with pllm.make_llm(CONFIG, workload=LogitLens()) as llm:
    seven = SamplingParams(max_tokens=7, temperature=0.0)  # 回答正好 7 个 token
    output = llm.generate([chat("What is the capital of France? Answer in one sentence.")], seven, use_tqdm=False)[0]
    print("eager 原因:", repr(llm.diagnostics().eager_reason))

generated = list(output.outputs[0].token_ids)
print("回答:", repr(output.outputs[0].text))
# 第 0 行来自 prompt 的最后一个 token，第 k 行来自输入第 k 个生成 token 的那一步；
# 第 k 行的最后一层预测的就是第 k 个生成 token（从 0 数起）。
print("最后一层的预测与实际生成一致:", all(row[-1] == generated[k] for k, row in enumerate(lens[0])))

step = next(k for k, row in enumerate(lens[0]) if tokenizer.decode(row[-1]) == " Paris")
print(f"预测 ' Paris' 的那一步，每一层最可能的 token：")
for layer, token in zip(LAYERS, lens[0][step]):
    print(f"  第 {layer:2d} 层: {tokenizer.decode(token)!r}")
```

```text
eager 原因: ''
回答: 'The capital of France is Paris.'
最后一层的预测与实际生成一致: True
预测 ' Paris' 的那一步，每一层最可能的 token：
  第  0 层: '\';";\n'
  第  1 层: '\';";\n'
  第  2 层: ".','"
  第  3 层: ".','"
  第  4 层: ".','"
  第  5 层: '\';";\n'
  第  6 层: "',...\n"
  第  7 层: "',...\n"
  第  8 层: '","#'
  第  9 层: 'announcement'
  第 10 层: 'announcement'
  第 11 层: "',...\n"
  第 12 层: 'уницип'
  第 13 层: '那'
  第 14 层: '\';";\n'
  第 15 层: ';break'
  第 16 层: 'MAND'
  第 17 层: 'MAND'
  第 18 层: ';break'
  第 19 层: '家都知道'
  第 20 层: ' capitals'
  第 21 层: ' Paris'
  第 22 层: ' Paris'
  第 23 层: ' Paris'
```

模型生成的回答是 `'The capital of France is Paris.'`。输出里 `最后一层的预测与实际生成一致: True` 验证了我们刚才说的对齐关系：每一行最后一层的预测，确实对应着下一步实际生成的 token。

再看预测 `' Paris'` 的那一步：从第 0 层到第 19 层，每一层最可能的 token 都是像 `'\';";\n'`、`'announcement'`、`';break'` 这样和问题无关的碎片；到了第 20 层，变成了和主题相关的 `' capitals'`；从第 21 层开始，最可能的 token 才真正变成 `' Paris'`，并一直保持到第 23 层（最后一层）。也就是说，在这个例子里，答案是在最后几层才成形的。同样地，这里的 `eager 原因` 也是空的。

## 第三步：DoLa —— 用逐层 logits 改变解码 {#zh-sec-4}

既然浅层只有零碎的模式，而事实答案在最后几层才成形，我们能不能利用这种深浅层的差异来改进解码？这就是第三步要试的 DoLa（Chuang et al., 2024, "Decoding by Contrasting Layers"）。

DoLa 的思路用白话讲很简单：每生成一个 token 时，先从一组候选的浅层里，挑出一个和最后一层预测分歧最大的层（用 Jensen-Shannon 散度衡量，散度最大就是分歧最大）；然后用“最后一层的 log 概率减去这个浅层的 log 概率”来给词表里的词重新打分。为了避免选出本来极不可能的词，只在最后一层认为足够可能的 token（概率至少达到最后一层最大概率的 0.1 倍）里做选择，其余词直接排除。它的直觉是：浅层就能猜出来的表面模式会被压低，而深层新浮现出来的信息会被抬高。

pLLM 自带了这个算法的实现 `pllm.workloads.DoLaWorkload`（代码在 `pllm/workloads/dola.py`，核心方法只有十几行）：它用一个 `@pllm.add_logits` 方法读取 `pllm.layer_logits`，默认的候选浅层是 `0, 2, ..., 14`。用起来只需要把 `DoLaWorkload()` 传给 `make_llm`。

下面我们先在 4 个事实问题上对比原生解码和 DoLa 的贪心回答，再测吞吐代价。这里我们只看输出变化和性能代价，不评测 DoLa 是否让模型更准确。

在启动引擎之前，先在 CPU 上看 DoLa 的计划、检查它的正确性。`explain` 的 `forward` 行告诉我们模型是否还在
CUDA graph 里，`flops_per_token_estimate` 把 DoLa 每个 token 多出的计算和底座放在一起；`testing.check` 确认
DoLa 的逐行数学不会把不同请求的行混在一起。DoLa 没有强度开关，所以这里不传 `neutral=`。

```python
from pllm import testing

report = pllm.explain(DoLaWorkload(), CONFIG)
print("CUDA graph:", "保留" if report.eager_reason == "" else report.eager_reason)
print(f"每 token FLOP 估计：{report.flops_per_token_estimate:.3g}（底座模型约 9.88e+08）")
print(testing.check(DoLaWorkload(), CONFIG))
```

```text
CUDA graph: 保留
每 token FLOP 估计：3.44e+09（底座模型约 9.88e+08）
status  contract / reason
PASS    tensor_parallel_size=1: admission
PASS    compile
PASS    row permutation
PASS    request isolation
PASS    mixed prefill/decode
PASS    chunking / tokens=
PASS    preempted
SKIP    publish_lag: no row method reads versioned state
SKIP    neutral: pass neutral= with the author's disabled settings
```

```python
questions = ["What is the capital of Australia?", "Who wrote the novel 1984?", "How many legs does a spider have?",
             "What is the boiling point of water at sea level in Fahrenheit?"]
answer = SamplingParams(max_tokens=40, temperature=0.0)


def answers(llm) -> list[str]:
    return [output.outputs[0].text for output in llm.generate([chat(q) for q in questions], answer, use_tqdm=False)]


with pllm.make_llm(CONFIG) as plain:
    plain_texts = answers(plain)
with pllm.make_llm(CONFIG, workload=DoLaWorkload()) as llm:
    dola_texts = answers(llm)

for question, a, b in zip(questions, plain_texts, dola_texts):
    print(question)
    print("  原生:", repr(a))
    print("  DoLa:", repr(b))
```

```text
What is the capital of Australia?
  原生: 'The capital of Australia is Canberra.'
  DoLa: 'The capital of Australia is Canberra. Canberra is the capital city of Australia.'
Who wrote the novel 1984?
  原生: 'The novel "1984" was written by George Orwell.'
  DoLa: '1984 is a work of fiction, created by George Orwell. The novel is part of Orwell\'s complete series of books titled "1984" and is set in a dystopian'
How many legs does a spider have?
  原生: 'A spider has eight legs.'
  DoLa: 'A spider has eight legs.'
What is the boiling point of water at sea level in Fahrenheit?
  原生: 'The boiling point of water at sea level in Fahrenheit is 212°F.'
  DoLa: 'The boiling point of water at sea level is 212 degrees Fahrenheit.'
```

再测吞吐代价。前面的观察用 0.5B 就够了；吞吐换到 `Qwen/Qwen2.5-7B-Instruct` 上测：0.5B 一步 decode 只有几毫秒，LM head 和采样器在整步里的占比被放大，7B 才是这类算法常见的部署规模（bf16 权重约 15 GiB，需要一张 24 GiB 的卡）。设置是 32 个并发请求（8 个 prompt × 每个采样 4 个样本 × 生成 256 个 token）；吞吐用 `pllm.benchmark.throughput` 测：预热一轮，再计时三轮取中位数，每轮计整次 `generate` 的时间（含 prefill）。DoLaWorkload 默认的候选浅层 `0, 2, ..., 14` 在 7B（28 层）上同样适用。

```python
BIG = pllm.ModelConfig("Qwen/Qwen2.5-7B-Instruct", max_model_len=512, max_num_seqs=32,
                       max_num_batched_tokens=4096, gpu_memory_utilization=0.85, seed=7)
many = [f"Explain the mathematical idea numbered {i}:" for i in range(8)]
sampling = SamplingParams(n=4, max_tokens=256, temperature=0.8, top_p=0.95, ignore_eos=True, seed=7)

with pllm.make_llm(BIG) as plain:
    plain_rate = throughput(plain, many, sampling)
with pllm.make_llm(BIG, workload=DoLaWorkload()) as llm:
    dola_rate = throughput(llm, many, sampling)
    eager_reason = llm.diagnostics().eager_reason
print(f"7B 吞吐（32 个并发请求，各 256 token）：原生 {plain_rate:.0f} token/s，DoLa {dola_rate:.0f} token/s，"
      f"为原生的 {dola_rate / plain_rate:.0%}；eager 原因 {eager_reason!r}")
```

```text
7B 吞吐（32 个并发请求，各 256 token）：原生 1727 token/s，DoLa 1140 token/s，为原生的 66%；eager 原因 ''
```

先看回答的变化。在 4 个问题里，DoLa 改变了其中 3 个的回答：澳大利亚首都那题，DoLa 在答出堪培拉后多加了一句重复的话；《1984》作者那题，DoLa 的回答变长了，但里面多出了一句不准确的内容（说这本书属于 Orwell 一个叫“1984”的系列丛书）；蜘蛛有几条腿那题，两边回答一样；水的沸点那题，DoLa 换了一种说法（把 `212°F` 写成了 `212 degrees Fahrenheit`）。当然，这只是 4 个例子，不能用来说明 DoLa 让这个 0.5B 模型变得更准还是更差。

再看吞吐代价（7B，32 个并发请求、各生成 256 个 token；RTX 4090、vLLM 0.32.0、bf16、CUDA graph，预热 1 轮后 3 轮取中位数）：这次运行的原生吞吐是 1727 token/s，开启 DoLa 后是 1140 token/s，为原生的 66%。`eager 原因` 依然是 `''`，说明 DoLa 全程仍在 CUDA graph 里运行。吞吐下降的代价来自实打实的计算量：每生成一个 token，都要对 9 个层（8 个候选浅层加上最后一层）在大小为 152064 的词表上计算 FP32 的 logits 和散度。`explain` 的估计说明了这一点：在 7B 上，DoLa 让每个 token 的计算从约 141 亿 FLOP 增加到约 240 亿 FLOP，是底座的 1.7 倍。同样的 DoLa 在 0.5B 上是底座的 3.5 倍（第 11 格的输出：约 9.9 亿到约 34 亿），因为词表投影的大小和模型深浅无关，模型越小，它在整步里占得越多；所以小模型上测出的 DoLa 代价会比 7B 重得多。

## 框架在背后做了什么 {#zh-sec-5}

为什么读取残差流和逐层 logits 都能留在 CUDA graph 里，而且不会干扰模型原本的输出？

1. **残差流读取**：框架把读取操作挂在了模型每一层都会调用的 norm 上，直接在 CUDA graph 内部把需要的行拷贝出来。
2. **逐层 logits 计算**：框架在 LM head 处用一次独立的矩阵乘来计算所选层的 logits，和模型自己算最终 logits 的过程完全分开，因此模型自身的 logits 与原生 vLLM 保持逐位相同。

## 小结 {#zh-sec-6}

- `pllm.residual(layer)` 读取指定层之后的残差流；通过对比范数，我们可以直观地看出激活引导所加的向量相对于残差流本身有多大。
- `pllm.layer_logits(layers)` 在不破坏 CUDA graph 的前提下给出多个中间层的 logits，既可以用来做 logit lens 观察答案在哪一层成形，也可以用来编写依赖逐层 logits 的解码算法。
- `pllm.workloads.DoLaWorkload` 用十几行核心代码实现了对比深浅层 logits 的 DoLa 解码；它在 CUDA graph 内运行，但由于每个 token 都要在全词表上对多个层计算 FP32 logits 和散度，在 7B 上吞吐为原生的 66%。

在下一篇 [07 改写残差连接](/pllm/docs/examples/07_custom_residual/) 中，我们将不再只是读取残差流
，而是直接修改残差流的更新方式（`@pllm.custom_residual`）。

</div>

</div>

<div class="zine-doc-footer">
  <a class="zine-pill-link" data-i18n-zh="← 返回 pLLM 项目与文档中心" data-i18n-en="← Back to pLLM Project &amp; Docs" href="/pllm/#docs">← Back to pLLM Project &amp; Docs</a>
  <a class="zine-pill-link" href="https://github.com/LinesHogan/pLLM/blob/main/doc/examples/06_reading_inside.ipynb" target="_blank" rel="noopener">GitHub &middot; doc/examples/06_reading_inside.ipynb ↗</a>
</div>

<script defer src="/assets/js/bilingual-post.js"></script>
