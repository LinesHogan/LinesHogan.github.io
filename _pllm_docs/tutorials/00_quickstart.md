---
layout: page
title: "00 Quickstart: Moving Activation Steering into vLLM"
description: "pLLM Documentation · Step-by-Step Tutorials (00–05)"
permalink: /pllm/docs/tutorials/00_quickstart/
nav: false
---

<link rel="stylesheet" href="{{ '/assets/css/pllm-hub.css' | relative_url | bust_file_cache }}">

<div class="zine-dispatch" data-title-zh="00 快速上手：把激活引导搬进 vLLM" data-title-en="00 Quickstart: Moving Activation Steering into vLLM" data-desc-zh="pLLM 文档 · 交互式入门教程 (00–05)" data-desc-en="pLLM Documentation · Step-by-Step Tutorials (00–05)">
  <div class="zine-dispatch-links">
    <a class="zine-pill-link primary" data-i18n-zh="← 返回 pLLM 文档中心" data-i18n-en="← pLLM Docs Hub" href="/pllm/#docs">← pLLM Docs Hub</a>
    <a class="zine-pill-link" data-i18n-zh="文档导航地图" data-i18n-en="Docs Roadmap" href="/pllm/docs/">Docs Roadmap</a>
    <span class="zine-lang-badge" data-i18n-zh="可执行教程 (.ipynb)" data-i18n-en="Notebook (.ipynb)">Notebook (.ipynb)</span>
  </div>
  <a class="zine-pill-link" data-i18n-zh="在 GitHub 打开 .ipynb ↗" data-i18n-en="Open .ipynb on GitHub ↗" href="https://github.com/LinesHogan/pLLM/blob/main/doc/tutorials/00_quickstart.ipynb" target="_blank" rel="noopener">Open .ipynb on GitHub ↗</a>
</div>

<div id="markdown-content">

<div class="zine-lang-panel" data-lang-panel="en" lang="en" markdown="1">

<div id="en-sec-1"></div>

**Who this is for**: You are comfortable experimenting with PyTorch and Hugging Face and want to validate an algorithm that changes inference behavior inside a real vLLM engine. We start with the simplest possible example: **activation steering**.

The idea behind activation steering is straightforward (ActAdd, Turner et al. 2023; CAA, Rimsky et al. 2024): take a few pairs of contrasting sentences—such as "I love ..." and "I hate ..."—capture the difference in their activations at a chosen layer, and average those differences into a steering vector. During generation, scale that vector by a strength multiplier and add it to the layer's output. The model's tone shifts in that direction.

In Hugging Face, this is a five-line forward hook. In vLLM, it breaks down: the model is compiled and captured in CUDA graphs, tokens from dozens of requests are packed into a single tensor by continuous batching, and a hook neither knows which row belongs to which request nor is guaranteed to run at all during graph replay. Normally you would end up forking vLLM's model code—which is exactly what pLLM lets you avoid.

This tutorial walks through six steps:

1. Launch a plain vLLM engine with pLLM to confirm the environment works;
2. Write your first workload with a method that reads activations from the engine and computes the steering vector;
3. Write the steering workload in two declarations;
4. Inspect its execution plan and check its correctness on CPU before touching a GPU;
5. Import the vector into the engine, vary the strength, and observe how outputs change;
6. Look back at what the framework handled for you.

Along the way, you will see that every workload answers three questions: **what to read**, **what to store**, and **where results go**.

Everything runs on `Qwen2.5-0.5B-Instruct`, so a single consumer GPU is plenty.

```python
import os

os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"  # Show only vLLM errors to suppress startup logs

import torch
from transformers import AutoTokenizer
from vllm import SamplingParams

import pllm
from pllm import testing

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
# A small engine: up to 4 concurrent requests, 30% GPU memory.
CONFIG = pllm.ModelConfig(MODEL, max_model_len=512, max_num_seqs=4, max_num_batched_tokens=512,
                          gpu_memory_utilization=0.3, seed=0)
```

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL)
questions = ["Write one sentence about my new job.", "Describe your weekend in one sentence."]
prompts = [tokenizer.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                         add_generation_prompt=True) for q in questions]
greedy = SamplingParams(max_tokens=40, temperature=0.0)
```

## Step 1: Without a Workload, It's Just Plain vLLM {#en-step-1-without-a-workload-its-just-plain-vllm}

Calling `pllm.make_llm(config)` without a `workload` starts an ordinary vLLM engine with no workload attached, and you use it just like vLLM's `LLM.generate`. Let's run it once first to record the model's baseline responses for comparison later.
Always manage the engine with a `with` block: exiting `with` shuts the engine down and frees GPU memory so the next engine has room to start.

```python
with pllm.make_llm(CONFIG) as plain:
    baseline = [output.outputs[0].text for output in plain.generate(prompts, greedy, use_tqdm=False)]
for text in baseline:
    print(repr(text))
```

```text
'As an AI language model, I am here to assist you with your new job and provide you with the best possible assistance possible.'
"As an AI language model, I don't have a physical presence, so I don't have a weekend. However, I'm always here to assist you with any questions or tasks you may have!"
```

## Step 2: Read Activations from the Engine and Compute the Steering Vector {#en-step-2-read-activations-from-the-engine-and-compute-the-steering-vector}

We extract activations from the final token of five contrasting phrase pairs at layer 12, take their differences, and average them. We read the activations directly from the vLLM engine using a small observation-only workload. This is your first workload: it has a single method and answers two of the three questions:

- **What to read** is written in the parameter's default value: `r: torch.Tensor = pllm.residual(12, tokens="prefill")`. The type annotation only tells your method body that it receives a tensor; the default value (the **input marker**) tells pLLM where to fetch it;
- **Where results go** is written in the decorator: `@pllm.export` means "copy the return value back to the host";
- It does not need to persist anything across steps, so there is no "what to store".

Specifically:

- `pllm.residual(12, tokens="prefill")` gives the residual stream after layer 12 for the last token of each prompt, which corresponds to `hidden_states[12 + 1]` in Hugging Face;
- `@pllm.export` copies the method's return tensor to CPU and passes it to `on_export`, bounded by at most `max_rows` rows and `max_cols` columns per step. Here each row is an 896-dimensional activation, and the number of rows never exceeds the max concurrency of 4;
- Every row delivered to `on_export` carries its identity: `identity.prompt_index` tells you which prompt in the `generate` call produced that row.

The next tutorial, [01 Observing](/pllm/docs/tutorials/01_observe_probe/), covers observation workloads in detail; here we just put one to work.

```python
LAYER = 12
pairs = [("I love", "I hate"), ("This is wonderful", "This is terrible"), ("I feel happy", "I feel sad"),
         ("What a great day", "What an awful day"), ("Everything is amazing", "Everything is horrible")]
activations: dict[int, torch.Tensor] = {}


class PromptEnd(pllm.Workload):
    @pllm.export(name="prompt_end", max_rows=4, max_cols=896)
    def last_token(self, r: torch.Tensor = pllm.residual(LAYER, tokens="prefill")) -> torch.Tensor:
        return r.float()

    def on_export(self, event: pllm.io.TraceEvent) -> None:
        for identity, row in zip(event.identities, event.tensor()):
            activations[identity.prompt_index] = row
```

```python
phrases = [text for pair in pairs for text in pair]  # Interleave positive and negative phrases
with pllm.make_llm(CONFIG, workload=PromptEnd()) as llm:
    llm.generate(phrases, SamplingParams(max_tokens=1), use_tqdm=False)  # Only prompt activations are needed
# By the time `with` exits, all exports have been delivered to `on_export`.
rows = torch.stack([activations[index] for index in range(len(phrases))])
vector = (rows[0::2] - rows[1::2]).mean(0)
print("引导向量:", tuple(vector.shape), "范数", round(vector.norm().item(), 2))
```

```text
引导向量: (896,) 范数 7.09
```

## Step 3: Write the Workload {#en-step-3-write-the-workload}

A workload is a subclass of `pllm.Workload` containing only **declarations** and your tensor math. Activation steering doesn't even need custom math—two declarations are enough, answering the other two questions:

- **What to store**: `steer` is a **state**—GPU memory managed by the framework. Here it holds a single field `vector` whose length equals the model's hidden width (`pllm.Field`'s shape function is called after the model loads, so you don't hardcode 896). `init=pllm.Imported()` indicates its value is supplied from outside rather than trained by the workload itself.
- **Where results go**: `patch` is an **activation patch** (`pllm.model.ActivationPatch`) that adds `steer` to layer 12's output. `operator="add"` performs direct addition; `decode=True` applies it on every newly generated token (by default it also applies to the last two prompt tokens).

Each vLLM layer returns two halves—"the newly computed delta for this layer" and "the residual"—which are summed at the start of the next layer. Adding to the layer output is therefore identical to adding to `hidden_states[13]` in Hugging Face.

```python
class Steering(pllm.Workload):
    steer = pllm.InstanceState({"vector": pllm.Field(lambda width, _: (width,))}, init=pllm.Imported())
    patch = pllm.model.ActivationPatch("steer", layers=(12,), operator="add", decode=True)
```

## Step 4: Inspect on CPU Before Starting the GPU {#en-step-4-inspect-on-cpu-before-starting-the-gpu}

Loading model weights and allocating GPU memory takes tens of seconds, and errors end up buried in verbose startup logs. pLLM provides two CPU-only tools that tell you in a few seconds "how this workload will run" and "whether its math obeys the runtime contracts":

- `pllm.explain(workload, config)` reads only the model's `config.json`—no weight loading, no GPU—and produces the execution plan: whether the model stays in CUDA graphs, how many bytes pLLM reserves for states, and whether your math triggers any GPU-to-CPU synchronization. Calling `print(report)` prints the full table (see [Cost Inspection: pllm.explain](/pllm/docs/guides/explain/) for a line-by-line guide); here we inspect just two fields.
- `pllm.testing.check(workload, config)` runs three requests on CPU under several adversarial schedules: permuting rows, mixing requests together, chunking prompts, and simulating preemption recomputation. A request must produce identical results across all schedules; otherwise the method is mixing data across requests.

```python
report = pllm.explain(Steering(), CONFIG)
print("CUDA graph:", "保留" if report.eager_reason == "" else f"退回 eager（{report.eager_reason}）")
print("pllm 为状态预留:", {state.name: state.reserved_bytes for state in report.states}, "字节")
```

```text
CUDA graph: 保留
pllm 为状态预留: {'steer': 10766} 字节
```

`steer` is just a single 896-dimensional bf16 vector; pLLM reserves multiple copies so that in-flight requests can keep reading the old version while a new version is staged in another slot (used in Step 5).

Now let's run the contract check on the observation workload from Step 2:

```python
print(testing.check(PromptEnd(), CONFIG))
```

```text
status  contract / reason
PASS    tensor_parallel_size=1: admission
PASS    compile
PASS    row permutation
PASS    request isolation
PASS    mixed prefill/decode
PASS    chunking / tokens=
PASS    preempted
SKIP    export eligibility: every= is exercised at aligned step ids; max_rows omissions are allowed, delivered values are compared
SKIP    publish_lag: no row method reads versioned state
SKIP    neutral: pass neutral= with the author's disabled settings
```

`PASS` marks checks that succeeded; `SKIP` rows explicitly state what was not tested and why—for instance, this workload does not read versioned state, so there is no `publish_lag` to check. Running the same check on `Steering` shows `SKIP activation application: needs a loaded model`: activation patches act inside the model, and CPU checks have no loaded weights to modify, so that part is verified on GPU with `pllm.testing.check_on_gpu` (see [Checking Correctness](/pllm/docs/guides/checking/)).

Make a habit of running these two lines on every workload you write: they are far faster than booting an engine, and errors point straight at your code.

## Step 5: Import the Vector and Tune the Strength {#en-step-5-import-the-vector-and-tune-the-strength}

Pass the workload to `make_llm` to start the engine with the workload installed. You feed the vector through `generate`'s `batch=` argument: `pllm.io.StateImport` holds tensors matching the state's fields in order. Each call with `batch=` publishes a new version of the state that requests in that call read; calls without `batch=` continue using the latest published version.

Let's test three strengths: `0` (should match the baseline output), `+2` (push toward positive), and `-2` (push toward negative).

```python
with pllm.make_llm(CONFIG, workload=Steering()) as llm:
    for strength in (0.0, 2.0, -2.0):
        outputs = llm.generate(prompts, greedy, batch=pllm.io.StateImport((strength * vector,)))
        texts = [output.outputs[0].text for output in outputs]
        print(f"强度 {strength:+.0f}:")
        for text in texts:
            print("   ", repr(text))
        if strength == 0.0:
            print("    与原始 vLLM 输出相同:", texts == baseline)
    print("eager 原因:", llm.diagnostics().eager_reason)
```

```text
强度 +0:
    'As an AI language model, I am here to assist you with your queries and provide you with information and insights on various topics. I am always here to help you with your new job.'
    "As an AI language model, I don't have a physical presence, so I don't have a weekend. However, I'm always here to assist you with any questions or tasks you may have!"
    与原始 vLLM 输出相同: False
强度 +2:
    'As a new member of the team, I am excited to share that the journey of a new career at Alibaba Cloud, a leader in the digital transformation of the world, with its rich history and a'
    'As a Qwen, I am here to share the unique experiences of my journey through the world, from the stars in the night sky, the sounds of nature, and the stories that connect us in'
强度 -2:
    "I'm not sure I understand what my new job is, but I don't think I can even be bothered to write a sentence about it. It's like I don't even know what I'm"
    'I have no idea what I should do on my weekend, I have no idea what I should do, I have no idea what I should do, I have no idea what I should do, I'
eager 原因:
```

At strength `+2`, the responses turn upbeat ("I am excited to share..."); at `-2`, they become confused and negative ("I have no idea what I should do..."). Pushing a 0.5B model too hard induces repetition, which is a property of activation steering itself—strength and layer index are hyperparameters to tune.

At strength `0`, the phrasing differs slightly from plain vLLM in Step 1. The cause is not the patch content (it adds a zero vector) but the execution graph: the empty `eager_reason` on the last line confirms the patch runs inside CUDA graphs alongside an Inductor-compiled model. Different model graph topologies alter kernel fusion and bf16 rounding; `check_on_gpu` compares full vocabulary distributions against Inductor hooks, while paths sharing identical kernels still match token IDs bit-for-bit. See [Example 10](/pllm/docs/examples/10_custom_attention/) for mean/P99 KL and TV tolerances.

## Step 6: What the Framework Handled for You {#en-step-6-what-the-framework-handled-for-you}

Looking back, all you wrote was two declarations and one `StateImport`. Here is what pLLM took care of:

- **Targeting the right layer and rows**: The patch is attached to vLLM's compiled layer 12 and applied only to the intended tokens of each request, regardless of where continuous batching places them in the packed tensor;
- **Managing state memory and versions**: GPU memory for `steer` is budgeted and allocated at startup; each `StateImport` publishes a new version, while in-flight requests keep reading the version they started with so their context never shifts mid-sentence;
- **Leaving everything else untouched**: The patch acts only on the declared layer and tokens; all other computation runs unchanged.

## Common Mistakes {#en-common-mistakes}

When starting out with workloads, the two most frequent mistakes map directly to the "three questions": forgetting to specify **what to read** (omitting a parameter's default input marker) or forgetting **where results go** (omitting the decorator on a method). pLLM rejects both at workload build time rather than failing mid-generation. Let's trigger both intentionally to see the diagnostics:

```python
class Forgot(pllm.Workload):
    @pllm.export(name="norm", max_rows=4, max_cols=1)
    def norm(self, r: torch.Tensor) -> torch.Tensor:  # Missing = pllm.residual(12)
        return r.float().norm(dim=-1, keepdim=True)


class Unused(pllm.Workload):
    def norm(self, r: torch.Tensor = pllm.residual(12)) -> torch.Tensor:  # Missing @pllm.export
        return r.float().norm(dim=-1, keepdim=True)


for workload in (Forgot(), Unused()):
    try:
        pllm.explain(workload, CONFIG)
    except (TypeError, ValueError) as error:
        print(f"{type(workload).__name__}: {error}\n")
```

```text
Forgot: r: give every input a source as its default, e.g. `r: torch.Tensor = pllm.hidden(0)`

Unused: the workload has no effect on the model: add @pllm.train, @pllm.add_logits, @pllm.guide_sampling, @pllm.export, @pllm.train_in_forward, @pllm.custom_residual, @pllm.custom_attention, a custom_forward or a pllm.model.ActivationPatch/Adapter
```

The first error asks you to give the parameter a source. Why doesn't pLLM guess from the parameter name? Because the same name can mean different things (which layer? residual stream or MLP output?), and guessing wrong would silently read garbage instead of failing loudly. The second error occurs because pLLM only schedules computations that have a destination: an undecorated method is never executed, so its return value goes nowhere and the workload has no effect on the model.

## Self-Check {#en-self-check}

1. In Hugging Face, activation steering is a forward hook. Why can't you copy that pattern directly into vLLM?

<details markdown="1">
<summary>Answer</summary>

vLLM captures the model into CUDA graphs, and Python hooks are not invoked during graph replay. In addition, tokens from multiple requests are packed into a single tensor, so a hook has no way to tell which row belongs to which request. pLLM lets you declare "patch layer 12's output using state `steer`," and handles installing the patch and matching every row.

</details>

2. In Step 5, suppose you call `generate` once with `batch=pllm.io.StateImport(...)` and then call `generate` again without `batch=`. Which vector do requests in the second call read?

<details markdown="1">
<summary>Answer</summary>

The previously imported vector. Every call with `batch=` publishes a new version of the state; calls without `batch=` keep using the latest published version.

</details>

3. In `testing.check(Steering(), CONFIG)`, the `"activation application"` row reports `SKIP`. Does that mean the patch is broken?

<details markdown="1">
<summary>Answer</summary>

No. No model weights are loaded on CPU, so patches that operate inside the model cannot execute; the check explicitly skips it and explains why. To verify that part, run `pllm.testing.check_on_gpu` on a GPU.

</details>

## Next Steps {#en-next-steps}

- To apply a trained low-rank intervention (ReFT / LoReFT), change `operator` to `"low_rank_affine"`, or check out the built-in `pllm.workloads.ReFTWorkload`;
- [01 Observing](/pllm/docs/tutorials/01_observe_probe/): Leave the model unchanged, read per-token internal states, and monitor generation in real time with a probe.

</div>

<div class="zine-lang-panel is-hidden" data-lang-panel="zh" lang="zh-CN" markdown="1">

<div id="zh-sec-1"></div>

**这篇写给谁**：你会用 PyTorch 和 Hugging Face 做实验，想把一个改变推理行为的算法放进真实的
vLLM 服务里验证。我们用一个最容易理解的算法开头：**激活引导**（activation steering）。

激活引导的做法很简单（ActAdd，Turner et al. 2023；CAA，Rimsky et al. 2024）：拿几对意思相反的
句子，比如"我爱……"和"我恨……"，在某一层取它们的激活之差，平均得到一个方向向量；生成时把这个
向量乘上一个强度，加到这一层的输出上。模型的语气就会朝这个方向偏。

在 Hugging Face 里，这是一个五行的 forward hook。到了 vLLM 就不一样了：模型代码被编译、被 CUDA
graph 捕获，几十个请求的 token 被拼成一个张量连续批处理，hook 既不知道哪一行属于哪个请求，也
可能根本不会被调用。你会想去改 vLLM 的模型代码，这正是 pLLM 想让你省掉的事。

这篇教程走六步：

1. 用 pLLM 起一个普通的 vLLM 引擎，确认环境没问题；
2. 写第一个带方法的负载，从引擎里读出激活，算出引导向量；
3. 写引导负载：两行声明；
4. 启动 GPU 之前，在 CPU 上看它的执行计划、检查它的正确性；
5. 把向量送进引擎，调强度，看输出怎么变；
6. 回头看框架替你做了什么。

一路上你会看到，每个负载都在回答三个问题：**读什么**、**存什么**、**结果去哪**。

全程用 Qwen2.5-0.5B-Instruct，一张消费级 GPU 足够。

```python
import os

os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"  # 只显示 vLLM 的报错，省掉启动日志

import torch
from transformers import AutoTokenizer
from vllm import SamplingParams

import pllm
from pllm import testing

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
# 一个小引擎：最多 4 个并发请求，占 30% 显存。
CONFIG = pllm.ModelConfig(MODEL, max_model_len=512, max_num_seqs=4, max_num_batched_tokens=512,
                          gpu_memory_utilization=0.3, seed=0)
```

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL)
questions = ["Write one sentence about my new job.", "Describe your weekend in one sentence."]
prompts = [tokenizer.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                         add_generation_prompt=True) for q in questions]
greedy = SamplingParams(max_tokens=40, temperature=0.0)
```

## 第一步：不带负载，就是普通的 vLLM {#zh-sec-2}

`pllm.make_llm(config)` 不传 `workload` 时，起的就是一个不带负载的普通 vLLM 引擎，用法和 vLLM 的 `LLM.generate` 一样。先用它生成一次，
记下模型原本的回答，后面拿来对照。
引擎用 `with` 管理：离开 `with` 时引擎关闭，显存还给 GPU，后面的引擎才有地方启动。

```python
with pllm.make_llm(CONFIG) as plain:
    baseline = [output.outputs[0].text for output in plain.generate(prompts, greedy, use_tqdm=False)]
for text in baseline:
    print(repr(text))
```

```text
'As an AI language model, I am here to assist you with your new job and provide you with the best possible assistance possible.'
"As an AI language model, I don't have a physical presence, so I don't have a weekend. However, I'm always here to assist you with any questions or tasks you may have!"
```

## 第二步：从引擎里读出激活，算引导向量 {#zh-sec-3}

我们在第 12 层取五对反义短语最后一个 token 的激活，求差再平均。激活直接从 vLLM 引擎里读，用一个只做观测的
小负载。这是你写的第一个负载，它只有一个方法，回答了三个问题中的两个：

- **读什么**写在参数的默认值里：`r: torch.Tensor = pllm.residual(12, tokens="prefill")`。注解只说明方法体拿到的是张量，
  默认值才告诉 pLLM 去哪取它；
- **结果去哪**写在装饰器里：`@pllm.export` 表示"把返回值送回宿主机"；
- 它不需要跨步保存任何东西，所以没有"存什么"。

具体来说：

- `pllm.residual(12, tokens="prefill")` 是每个 prompt 最后一个 token 在第 12 层之后的残差流，也就是
  Hugging Face 里的 `hidden_states[12 + 1]`；
- `@pllm.export` 把方法的返回值复制到 CPU，交给 `on_export`。每次最多 `max_rows` 行、`max_cols` 列，
  这里一行就是一个 896 维的激活，行数不超过并发请求数 4；
- `on_export` 收到的每一行都带着身份，`identity.prompt_index` 是这一行来自 `generate` 的第几个 prompt。

下一篇 [01 观测](/pllm/docs/tutorials/01_observe_probe/) 会细讲观测负载，这里先用起来。

```python
LAYER = 12
pairs = [("I love", "I hate"), ("This is wonderful", "This is terrible"), ("I feel happy", "I feel sad"),
         ("What a great day", "What an awful day"), ("Everything is amazing", "Everything is horrible")]
activations: dict[int, torch.Tensor] = {}


class PromptEnd(pllm.Workload):
    @pllm.export(name="prompt_end", max_rows=4, max_cols=896)
    def last_token(self, r: torch.Tensor = pllm.residual(LAYER, tokens="prefill")) -> torch.Tensor:
        return r.float()

    def on_export(self, event: pllm.io.TraceEvent) -> None:
        for identity, row in zip(event.identities, event.tensor()):
            activations[identity.prompt_index] = row
```

```python
phrases = [text for pair in pairs for text in pair]  # 正、反交替排列
with pllm.make_llm(CONFIG, workload=PromptEnd()) as llm:
    llm.generate(phrases, SamplingParams(max_tokens=1), use_tqdm=False)  # 只需要 prompt 的激活
# 离开 with 时引擎已把导出全部送到 on_export。
rows = torch.stack([activations[index] for index in range(len(phrases))])
vector = (rows[0::2] - rows[1::2]).mean(0)
print("引导向量:", tuple(vector.shape), "范数", round(vector.norm().item(), 2))
```

```text
引导向量: (896,) 范数 7.09
```

## 第三步：写负载 {#zh-sec-4}

一个负载是 `pllm.Workload` 的子类，里面只放**声明**和你的张量数学。激活引导连数学都不用写，
两行声明就够，它们回答了另外两个问题：

- **存什么**：`steer` 是一份**状态**：一块由框架管理的显存，这里只有一个字段 `vector`，长度等于模型的
  hidden 宽度（`pllm.Field` 的形状函数在加载模型后才被调用，所以不用写死 896）。
  `init=pllm.Imported()` 表示它的值由你从外面送进来，而不是由负载自己训练。
- **结果去哪**：`patch` 是一个**激活补丁**：把 `steer` 加到第 12 层的输出上。`operator="add"` 表示直接相加；
  `decode=True` 表示每个新生成的 token 都加（默认还会加在 prompt 的最后两个 token 上）。

vLLM 的每一层返回"这一层新算出的部分"和"残差"两半，下一层开头才把它们相加，所以加到层输出上
和 Hugging Face 里加到 `hidden_states[13]` 上是同一件事。

```python
class Steering(pllm.Workload):
    steer = pllm.InstanceState({"vector": pllm.Field(lambda width, _: (width,))}, init=pllm.Imported())
    patch = pllm.model.ActivationPatch("steer", layers=(12,), operator="add", decode=True)
```

## 第四步：启动 GPU 之前，先在 CPU 上看一眼 {#zh-sec-5}

加载模型、分配显存要几十秒，出错时报错还埋在一长串启动日志里。pLLM 提供两个只用 CPU 的工具，几秒钟就能回答
"这个负载会怎么跑""它算得对不对"：

- `pllm.explain(workload, config)` 只读模型的 `config.json`，不加载权重、不碰 GPU，给出执行计划：模型还能不能留在
  CUDA graph 里、pLLM 要为状态预留多少显存、你的数学里有没有 GPU→CPU 同步。`print(report)` 打印完整的表，
  [看成本](/pllm/docs/guides/explain/) 逐行讲了怎么读；这里只取两个字段。
- `pllm.testing.check(workload, config)` 在 CPU 上用几种固定的调度跑三个请求：打乱行的顺序、把请求混在一起、
  把 prompt 切块、模拟抢占后重算。同一个请求的结果在这些调度下必须一样，否则说明方法把不同请求的数据混在了一起。

```python
report = pllm.explain(Steering(), CONFIG)
print("CUDA graph:", "保留" if report.eager_reason == "" else f"退回 eager（{report.eager_reason}）")
print("pllm 为状态预留:", {state.name: state.reserved_bytes for state in report.states}, "字节")
```

```text
CUDA graph: 保留
pllm 为状态预留: {'steer': 10766} 字节
```

`steer` 只是一个 896 维的 bf16 向量；预留了几份，是因为生成中的请求继续读旧版本时，新版本要放在另一份里（第五步会用到）。

再检查第二步的观测负载：

```python
print(testing.check(PromptEnd(), CONFIG))
```

```text
status  contract / reason
PASS    tensor_parallel_size=1: admission
PASS    compile
PASS    row permutation
PASS    request isolation
PASS    mixed prefill/decode
PASS    chunking / tokens=
PASS    preempted
SKIP    export eligibility: every= is exercised at aligned step ids; max_rows omissions are allowed, delivered values are compared
SKIP    publish_lag: no row method reads versioned state
SKIP    neutral: pass neutral= with the author's disabled settings
```

`PASS` 是通过的检查；`SKIP` 行写明了这次没检查什么、为什么，比如这个负载不读会更新的状态，所以没有
`publish_lag` 可查。对 `Steering` 做同样的检查，会看到一行 `SKIP activation application: needs a loaded model`：
激活补丁作用在模型内部，CPU 上没有模型可以改，这部分要用 `pllm.testing.check_on_gpu` 在 GPU 上确认（见
[检查正确性](/pllm/docs/guides/checking/)）。

以后每写一个负载，都先跑这两行：它们比启动引擎快得多，报错也直接指向你的代码。

## 第五步：送进向量，调强度 {#zh-sec-6}

把负载交给 `make_llm`，就得到一个带负载的引擎。向量通过 `generate` 的 `batch=` 参数送进去：
`pllm.io.StateImport` 按字段顺序装着要写入状态的张量。每次带 `batch=` 的调用都会发布状态的一个
新版本，这次调用里的请求读到的就是它；不带 `batch=` 的调用沿用上一次的值。

我们试三个强度：0（应当和原始输出完全一样）、+2（往"正面"推）、-2（往"负面"推）。

```python
with pllm.make_llm(CONFIG, workload=Steering()) as llm:
    for strength in (0.0, 2.0, -2.0):
        outputs = llm.generate(prompts, greedy, batch=pllm.io.StateImport((strength * vector,)))
        texts = [output.outputs[0].text for output in outputs]
        print(f"强度 {strength:+.0f}:")
        for text in texts:
            print("   ", repr(text))
        if strength == 0.0:
            print("    与原始 vLLM 输出相同:", texts == baseline)
    print("eager 原因:", llm.diagnostics().eager_reason)
```

```text
强度 +0:
    'As an AI language model, I am here to assist you with your queries and provide you with information and insights on various topics. I am always here to help you with your new job.'
    "As an AI language model, I don't have a physical presence, so I don't have a weekend. However, I'm always here to assist you with any questions or tasks you may have!"
    与原始 vLLM 输出相同: False
强度 +2:
    'As a new member of the team, I am excited to share that the journey of a new career at Alibaba Cloud, a leader in the digital transformation of the world, with its rich history and a'
    'As a Qwen, I am here to share the unique experiences of my journey through the world, from the stars in the night sky, the sounds of nature, and the stories that connect us in'
强度 -2:
    "I'm not sure I understand what my new job is, but I don't think I can even be bothered to write a sentence about it. It's like I don't even know what I'm"
    'I have no idea what I should do on my weekend, I have no idea what I should do, I have no idea what I should do, I have no idea what I should do, I'
eager 原因:
```

强度 +2 时回答变得热情起来（"I am excited to share..."），-2 时变得迷茫、消极（"I have no idea what I
should do..."）。0.5B 的小模型被推得太猛会开始重复，这是激活引导本身的特性：强度和层号是你要调的超参数。

强度 0 的输出和第一步的原生 vLLM 只在措辞上有出入。原因不在补丁的内容（加的是零向量），而在执行方式：
最后一行的 eager 原因是空的，说明补丁和模型一起在 CUDA graph 里运行，模型由 Inductor 编译。模型图结构不同会改变融合 kernel 和 bf16 舍入；`check_on_gpu` 对 Inductor 模型 hook 比较完整词表分布，使用相同 kernel 的路径仍逐位比较 token ID。均值/P99 KL/TV 的容差规则见[示例 10](/pllm/docs/examples/10_custom_attention/)。

## 第六步：框架替你做了什么 {#zh-sec-7}

回头看，你写的只有两行声明和一次 `StateImport`。下面这些事是 pLLM 做的：

- **找到这一层、改对这些行**：补丁装在 vLLM 编译后的第 12 层上，只作用于每个请求该改的 token，
  不管它们在连续批处理里被拼到了张量的哪一行；
- **管理这块状态**：`steer` 的显存在启动时按预算分配；每次 `StateImport` 产生一个新版本，
  已经在生成中的请求继续用它开始时的版本，不会在一句话中途被换掉；
- **不碰其他东西**：补丁只作用在你声明的那一层、那些 token 上，其余计算照旧。

## 常见错误 {#zh-sec-8}

刚开始写负载时，最常见的两个错误都在"三个问题"上：忘了说**读什么**（参数没有默认值），或者忘了说**结果去哪**
（方法没有装饰器）。pLLM 在构建负载时就拒绝它们，而不是在生成中途出错。下面故意犯这两个错误，看报错怎么说：

```python
class Forgot(pllm.Workload):
    @pllm.export(name="norm", max_rows=4, max_cols=1)
    def norm(self, r: torch.Tensor) -> torch.Tensor:  # 少了 = pllm.residual(12)
        return r.float().norm(dim=-1, keepdim=True)


class Unused(pllm.Workload):
    def norm(self, r: torch.Tensor = pllm.residual(12)) -> torch.Tensor:  # 少了 @pllm.export
        return r.float().norm(dim=-1, keepdim=True)


for workload in (Forgot(), Unused()):
    try:
        pllm.explain(workload, CONFIG)
    except (TypeError, ValueError) as error:
        print(f"{type(workload).__name__}: {error}\n")
```

```text
Forgot: r: give every input a source as its default, e.g. `r: torch.Tensor = pllm.hidden(0)`

Unused: the workload has no effect on the model: add @pllm.train, @pllm.add_logits, @pllm.guide_sampling, @pllm.export, @pllm.train_in_forward, @pllm.custom_residual, @pllm.custom_attention, a custom_forward or a pllm.model.ActivationPatch/Adapter
```

第一个报错要你给参数写一个来源。为什么不按参数名猜？因为同一个名字可能指不同的东西（第几层？残差流还是 MLP
输出？），猜错了不会报错，只会悄悄读错数据。第二个报错是因为 pLLM 只安排"有去处"的计算：没有装饰器的方法不会
被执行，返回值也就没有意义，所以整个负载对模型没有任何作用。

## 自测 {#zh-sec-9}

1. 在 Hugging Face 里，激活引导是一个 forward hook。为什么在 vLLM 里不能照搬？

<details markdown="1">
<summary>答案</summary>

vLLM 把模型捕获进 CUDA graph，重放 graph 时 Python hook 不会被调用；而且几个请求的 token 被拼进同一个张量，
hook 不知道哪一行属于哪个请求。pLLM 让你声明"改第 12 层的输出、读状态 `steer`"，由它把补丁装进模型、对上每一行。

</details>

2. 第五步里，先调用一次带 `batch=pllm.io.StateImport(...)` 的 `generate`，再调用一次不带 `batch=` 的 `generate`。
   第二次调用里的请求读到的是哪个向量？

<details markdown="1">
<summary>答案</summary>

上一次导入的向量。每次带 `batch=` 的调用发布状态的一个新版本；不带 `batch=` 的调用沿用最新已发布的版本。

</details>

3. `testing.check(Steering(), CONFIG)` 里 "activation application" 一行是 `SKIP`。这说明补丁有问题吗？

<details markdown="1">
<summary>答案</summary>

不说明。CPU 上没有加载模型，作用在模型内部的补丁无从执行，所以检查明确跳过它并写出原因。要确认这部分，
在 GPU 上运行 `pllm.testing.check_on_gpu`。

</details>

## 下一步 {#zh-sec-10}

- 想用训练好的低秩干预（ReFT / LoReFT），把 `operator` 换成 `"low_rank_affine"`，现成实现在
  `pllm.workloads.ReFTWorkload`；
- [01 观测](/pllm/docs/tutorials/01_observe_probe/)：不改模型，只把每个 token 的内部状态读出来，用探针实时监控生成。

</div>

</div>

<div class="zine-doc-footer">
  <a class="zine-pill-link" data-i18n-zh="← 返回 pLLM 项目与文档中心" data-i18n-en="← Back to pLLM Project &amp; Docs" href="/pllm/#docs">← Back to pLLM Project &amp; Docs</a>
  <a class="zine-pill-link" href="https://github.com/LinesHogan/pLLM/blob/main/doc/tutorials/00_quickstart.ipynb" target="_blank" rel="noopener">GitHub &middot; doc/tutorials/00_quickstart.ipynb ↗</a>
</div>

<script defer src="/assets/js/bilingual-post.js"></script>
