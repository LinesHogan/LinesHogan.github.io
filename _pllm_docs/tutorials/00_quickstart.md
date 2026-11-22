---
layout: page
title: "00 快速上手：把激活引导搬进 vLLM"
description: "pLLM · Step-by-Step Tutorials (00–05) (交互式入门教程 (00–05))"
permalink: /pllm/docs/tutorials/00_quickstart/
nav: false
---

<div class="zine-dispatch">
  <div class="zine-dispatch-links">
    <a class="zine-pill-link primary" href="/pllm/#docs">← pLLM Project &amp; Docs</a>
    <a class="zine-pill-link" href="/pllm/docs/">All Docs Index</a>
    <span class="zine-lang-badge">Notebook (.ipynb)</span>
  </div>
  <a class="zine-pill-link" href="https://github.com/LinesHogan/pLLM/blob/main/doc/tutorials/00_quickstart.ipynb" target="_blank" rel="noopener">Open .ipynb on GitHub ↗</a>
</div>

<div id="markdown-content" markdown="1">

<div id="sec-1"></div>

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

## 第一步：不带负载，就是普通的 vLLM {#sec-2}

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

## 第二步：从引擎里读出激活，算引导向量 {#sec-3}

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

## 第三步：写负载 {#sec-4}

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

## 第四步：启动 GPU 之前，先在 CPU 上看一眼 {#sec-5}

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

## 第五步：送进向量，调强度 {#sec-6}

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

## 第六步：框架替你做了什么 {#sec-7}

回头看，你写的只有两行声明和一次 `StateImport`。下面这些事是 pLLM 做的：

- **找到这一层、改对这些行**：补丁装在 vLLM 编译后的第 12 层上，只作用于每个请求该改的 token，
  不管它们在连续批处理里被拼到了张量的哪一行；
- **管理这块状态**：`steer` 的显存在启动时按预算分配；每次 `StateImport` 产生一个新版本，
  已经在生成中的请求继续用它开始时的版本，不会在一句话中途被换掉；
- **不碰其他东西**：补丁只作用在你声明的那一层、那些 token 上，其余计算照旧。

## 常见错误 {#sec-8}

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

## 自测 {#sec-9}

1. 在 Hugging Face 里，激活引导是一个 forward hook。为什么在 vLLM 里不能照搬？

<details><summary>答案</summary>

vLLM 把模型捕获进 CUDA graph，重放 graph 时 Python hook 不会被调用；而且几个请求的 token 被拼进同一个张量，
hook 不知道哪一行属于哪个请求。pLLM 让你声明"改第 12 层的输出、读状态 `steer`"，由它把补丁装进模型、对上每一行。
</details>

2. 第五步里，先调用一次带 `batch=pllm.io.StateImport(...)` 的 `generate`，再调用一次不带 `batch=` 的 `generate`。
   第二次调用里的请求读到的是哪个向量？

<details><summary>答案</summary>

上一次导入的向量。每次带 `batch=` 的调用发布状态的一个新版本；不带 `batch=` 的调用沿用最新已发布的版本。
</details>

3. `testing.check(Steering(), CONFIG)` 里 "activation application" 一行是 `SKIP`。这说明补丁有问题吗？

<details><summary>答案</summary>

不说明。CPU 上没有加载模型，作用在模型内部的补丁无从执行，所以检查明确跳过它并写出原因。要确认这部分，
在 GPU 上运行 `pllm.testing.check_on_gpu`。
</details>

## 下一步 {#sec-10}

- 想用训练好的低秩干预（ReFT / LoReFT），把 `operator` 换成 `"low_rank_affine"`，现成实现在
  `pllm.workloads.ReFTWorkload`；
- [01 观测](/pllm/docs/tutorials/01_observe_probe/)：不改模型，只把每个 token 的内部状态读出来，用探针实时监控生成。

</div>

<div class="zine-doc-footer">
  <a class="zine-pill-link" href="/pllm/#docs">← Back to pLLM Documentation Hub</a>
  <a class="zine-pill-link" href="https://github.com/LinesHogan/pLLM/blob/main/doc/tutorials/00_quickstart.ipynb" target="_blank" rel="noopener">Edit / View on GitHub ↗</a>
</div>
