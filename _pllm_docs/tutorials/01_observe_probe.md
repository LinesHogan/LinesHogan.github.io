---
layout: page
title: "01 观测：边生成边读模型内部"
description: "pLLM · Step-by-Step Tutorials (00–05) (交互式入门教程 (00–05))"
permalink: /pllm/docs/tutorials/01_observe_probe/
nav: false
---

<div class="zine-dispatch">
  <div class="zine-dispatch-links">
    <a class="zine-pill-link primary" href="/pllm/#docs">← pLLM Project &amp; Docs</a>
    <a class="zine-pill-link" href="/pllm/docs/">All Docs Index</a>
    <span class="zine-lang-badge">Notebook (.ipynb)</span>
  </div>
  <a class="zine-pill-link" href="https://github.com/LinesHogan/pLLM/blob/main/doc/tutorials/01_observe_probe.ipynb" target="_blank" rel="noopener">Open .ipynb on GitHub ↗</a>
</div>

<div id="markdown-content" markdown="1">

<div id="sec-1"></div>

**场景**：你训练了一个线性探针（或者一个稀疏自编码器 SAE），它能从 hidden state 里读出某种
信号，比如情绪、拒答倾向、模型是否在编造。离线验证做完了，你想知道：**在真实服务里逐 token
地跑它，能不能实时看到这个信号？** 这是很多可解释性和安全监控工作（探针监控、SAE 特征监控）
走向部署时要回答的问题。

vLLM 只把 token 交还给你，中间的 hidden state 留在 GPU 上、和别的请求拼在一起，还被 CUDA
graph 捕获着。用 hook 去拿，要么拿不到，要么得关掉 CUDA graph，还要自己把行和请求对上。

这篇教程：

1. 从引擎里读出每个 token 的最终 hidden，训练一个情绪探针；
2. 写一个负载，在每个生成的 token 上跑探针，把分数导出到宿主机；
3. 同时跑两个请求，看分数随 token 变化，并确认**观测不改变输出、不关闭 CUDA graph**；
4. 把探针换成 SAE 编码器要改什么。

```python
import os

os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"  # 只显示 vLLM 的报错，省掉启动日志

import tempfile
from collections import defaultdict

import torch
from safetensors.torch import save_file
from torch import nn
from transformers import AutoTokenizer
from vllm import SamplingParams

import pllm
from pllm import testing

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
CONFIG = pllm.ModelConfig(MODEL, max_model_len=512, max_num_seqs=4, max_num_batched_tokens=512,
                          gpu_memory_utilization=0.3, seed=0)
PROBE = os.path.join(tempfile.gettempdir(), "sentiment_probe.safetensors")  # 训练好的探针权重
```

## 第一步：读出训练数据，训练探针 {#sec-2}

探针读的是**最终 hidden**：最后一层之后、经过最终 norm、LM head 直接读的那个向量，也就是 Hugging Face
里的 `hidden_states[-1]`。如果你已经有一个离线训练好的探针，它读的是同一个量，可以直接搬过来。

这里我们现场训练一个。训练数据是 20 句带情绪标签的短句，每个 token 都算一个样本，所以要读出这些句子
**每个 token** 的最终 hidden。直接从 vLLM 引擎里读，用一个只做观测的小负载：

- `pllm.final_hidden(tokens="all")` 给出这一步计算的每个 token 一行，prompt 里的 token 也算在内；
- 一步最多计算 `max_num_batched_tokens`（这里 512）个 token，所以导出的 `max_rows` 也设成 512；
- `overflow="block"` 让导出在投递队列满时等待而不是丢弃：训练要用到每一行；
- 每一行除了身份 `identity`，还带着它在序列里的位置 `position`，靠这两个数把行放回原句。

导出的细节在第二步再讲。读完之后，一个 `nn.Linear(896, 1)` 加逻辑回归，几秒钟训完；训好的权重存成
safetensors 文件，第二步的负载从这个文件读入。

```python
positive = ["I love this movie, it is wonderful.", "What a fantastic and joyful day.",
            "The food was delicious and the staff were kind.", "I am so happy with my new phone.",
            "This book is a delight to read.", "Our trip was amazing and fun.", "She gave a brilliant, inspiring talk.",
            "The concert was beautiful.", "Thank you, this made my week!", "We won the game and everyone cheered."]
negative = ["I hate this movie, it is awful.", "What a terrible and miserable day.",
            "The food was disgusting and the staff were rude.", "I am so angry with my new phone.",
            "This book is a chore to read.", "Our trip was a disaster.", "She gave a boring, painful talk.",
            "The concert was horrible.", "This ruined my whole week.", "We lost the game and everyone cried."]
hidden: dict[tuple[int, int], torch.Tensor] = {}  # (第几句, 第几个 token) -> 最终 hidden


class EveryToken(pllm.Workload):
    @pllm.export(name="final_hidden", max_rows=512, max_cols=896, overflow="block")
    def rows(self, h: torch.Tensor = pllm.final_hidden(tokens="all")) -> torch.Tensor:
        return h.float()

    def on_export(self, event: pllm.io.TraceEvent) -> None:
        for identity, position, row in zip(event.identities, event.positions, event.tensor()):
            hidden[identity.prompt_index, position] = row


with pllm.make_llm(CONFIG, workload=EveryToken()) as llm:
    outputs = llm.generate(positive + negative, SamplingParams(max_tokens=1), use_tqdm=False)
# 只取 prompt 的 token；引擎可能还多算了一个生成的 token，不用它。
rows = [torch.stack([hidden[index, position] for position in range(len(output.prompt_token_ids))])
        for index, output in enumerate(outputs)]
features = torch.cat(rows).cuda()
labels = torch.cat([torch.full((len(r),), float(i < len(positive))) for i, r in enumerate(rows)]).cuda()

torch.manual_seed(0)
probe = nn.Linear(features.shape[1], 1).cuda()
optimizer = torch.optim.Adam(probe.parameters(), lr=1e-2)
for _ in range(300):
    loss = nn.functional.binary_cross_entropy_with_logits(probe(features).squeeze(-1), labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
accuracy = ((probe(features).squeeze(-1) > 0) == (labels > 0.5)).float().mean().item()
print(f"{len(features)} 个 token 样本，训练集准确率 {accuracy:.2f}")

save_file({"w": probe.weight.detach()[0].cpu(), "b": probe.bias.detach().cpu()}, PROBE)
```

```text
155 个 token 样本，训练集准确率 0.87
```

## 第二步：写观测负载 {#sec-3}

负载里有三样东西：

- `probe`：一份状态，存探针的 `w`（长度 = hidden 宽度）和 `b`。它没有优化器，所以是只读的；
  `init=pllm.Pretrained(...)` 指向刚才保存的文件，引擎启动时按字段名把 `w`、`b` 读进状态。状态按模型精度（bf16）
  存放，所以数学里先转成 float32 再算。
- `score`：你的数学。参数的**默认值**说明数据从哪来：`pllm.final_hidden()` 是每个请求当前
  token 的最终 hidden，一行一个请求；`pllm.parameter("probe", "w")` 是状态里的字段。
- `@pllm.export(...)`：说明结果去哪。返回值被复制到 CPU，按顺序交给 `on_export`。导出是有界的：
  每步最多 `max_rows` 行、`max_cols` 列，投递队列满时默认丢掉事件而不是等待，所以观测不会拖住 GPU，
  也不会撑爆内存。

`on_export` 收到的事件带着步号、这一步每一行属于哪个请求（`identities`），以及张量本身。我们把
分数按请求收集起来。

```python
scores: dict[int, list[float]] = defaultdict(list)


class SentimentMonitor(pllm.Workload):
    probe = pllm.InstanceState(
        {"w": pllm.Field(lambda width, _: (width,)), "b": (1,)}, init=pllm.Pretrained(PROBE)
    )

    @pllm.export(name="sentiment", max_rows=4, max_cols=1)
    def score(self, h: torch.Tensor = pllm.final_hidden(), w: torch.Tensor = pllm.parameter("probe", "w"),
              b: torch.Tensor = pllm.parameter("probe", "b")) -> torch.Tensor:
        return torch.sigmoid(h.float() @ w.float()[:, None] + b.float())

    def on_export(self, event: pllm.io.TraceEvent) -> None:
        for identity, value in zip(event.identities, event.tensor()[:, 0].tolist()):
            scores[identity.prompt_index].append(value)
```

## 第三步：同时跑两个请求 {#sec-4}

一个请求写海边美好的一天，一个写诸事不顺的一天。它们在同一个 batch 里并肩生成，探针分数
应该一个接近 1、一个接近 0。我们先用原生 vLLM 生成一遍作对照，再挂上监控负载生成同样的内容。

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL)
questions = ["Write two sentences about a wonderful day at the beach.",
             "Write two sentences about a terrible day when everything went wrong."]
prompts = [tokenizer.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                         add_generation_prompt=True) for q in questions]
greedy = SamplingParams(max_tokens=48, temperature=0.0)

with pllm.make_llm(CONFIG) as plain:
    baseline = [output.outputs[0].text for output in plain.generate(prompts, greedy, use_tqdm=False)]

with pllm.make_llm(CONFIG, workload=SentimentMonitor()) as llm:
    texts = [output.outputs[0].text for output in llm.generate(prompts, greedy)]
    eager_reason = llm.diagnostics().eager_reason

for index, text in enumerate(texts):
    values = scores[index]
    strip = "".join("+" if v > 0.5 else "-" for v in values)
    print(f"请求 {index}：{len(values)} 个 token，平均分 {sum(values) / len(values):.2f}")
    print(f"    {strip}")
    print(f"    {text!r}")
print("输出与原生 vLLM 相同:", texts == baseline)
print("eager 原因:", repr(eager_reason), "（空字符串表示模型仍在 CUDA graph 里运行）")
```

```text
请求 0：47 个 token，平均分 0.98
    +++++++++++++++++++++++-+++++++++++++++++++++++
    'On a wonderful day at the beach, the warm sun rays caress the sand, and the sound of waves crashing against the shore fills the air. The salty air invigorates the senses, and the feeling of being surrounded by nature and'
请求 1：38 个 token，平均分 0.05
    ----------------+--------------------+
    'A terrible day for me was when I had to cancel a major project due to unforeseen circumstances, leaving me with no choice but to focus on my personal life and prioritize my health.'
输出与原生 vLLM 相同: True
eager 原因: '' （空字符串表示模型仍在 CUDA graph 里运行）
```

每个 `+`/`-` 是一个生成的 token 上探针的判断。两个请求在同一个 batch 里，分数却清清楚楚地按请求分开：
写美好一天的请求几乎全是 `+`，另一个几乎全是 `-`。最后两行是观测的两条承诺：输出和原生 vLLM
逐字相同；`eager_reason` 为空，说明模型照常在 CUDA graph 里运行，探针计算也被一起捕获了。

## 第四步：换成 SAE 编码器 {#sec-5}

SAE 监控是同一个模式，只是探针变宽了：状态里放编码器矩阵 `W_enc`（字典大小 × hidden 宽度）和偏置，
数学换成 `ReLU(h @ W_enc.T + b)`。字典通常有几万个特征，超出导出上限，所以导出每个 token 激活最强的
k 个特征编号，而不是整个特征向量：

```python
@pllm.export(name="sae_top8", max_rows=4, max_cols=8)
def encode(self, h: torch.Tensor = pllm.final_hidden(), w_enc: torch.Tensor = pllm.parameter("sae", "w_enc"),
           b_enc: torch.Tensor = pllm.parameter("sae", "b_enc")) -> torch.Tensor:
    return torch.relu(h.float() @ w_enc.T + b_enc).topk(8, dim=-1).indices.float()
```

读哪一层也可以换：`pllm.hidden(L)` 是第 L 层新算出的那一半，即 MLP 的输出（加回残差之前），对应
Llama Scope 等工作里的 "MLP out" 位置；`pllm.final_hidden()` 是最终 norm 之后的向量。

## 小结 {#sec-6}

- 观测负载 = 一份只读状态 + 一个带 `@pllm.export` 的方法；参数默认值说明读什么。
- 导出有界、按请求标好身份、按顺序送到 `on_export`。
- 观测不改变输出，也不需要关掉 CUDA graph。

## 常见错误 {#sec-7}

观测负载最容易犯的两个错误，都来自把方法当成普通的 PyTorch 函数：在方法里打印数值，以及在行的维度上求平均。

```python
class Chatty(pllm.Workload):
    @pllm.export(name="norm", max_rows=4, max_cols=1)
    def norm(self, h: torch.Tensor = pllm.final_hidden()) -> torch.Tensor:
        print("范数", h.float().norm().item())  # 想在方法里看一眼数值
        return h.float().norm(dim=-1, keepdim=True)


class Centered(pllm.Workload):
    @pllm.export(name="norm", max_rows=4, max_cols=1)
    def norm(self, h: torch.Tensor = pllm.final_hidden()) -> torch.Tensor:
        centered = h.float() - h.float().mean(0)  # 想减去"平均激活"，却在第 0 维（请求）上求了平均
        return centered.norm(dim=-1, keepdim=True)


try:
    pllm.explain(Chatty(), CONFIG)
except RuntimeError as error:
    print("Chatty:", error)
try:
    testing.check(Centered(), CONFIG)
except testing.ContractError as error:
    print("Centered:", error.failures[0])
```

```text
Chatty: Tensor.item() cannot be called on meta tensors
Centered: [row permutation] request 'r2', export norm at token 5: max |diff| 23.652. method `norm` mixes rows: equal inputs for this request produced different outputs. Dimension 0 holds request rows; reduce features (dim=-1), not requests (dim=0).
```

**为什么不能 `.item()` 或 `print`。** 构建引擎时，pLLM 只用张量的形状把方法追踪一遍（这时的张量没有数值，
所以报 "meta tensors"），然后把它和模型一起录进 CUDA graph。生成时 graph 被直接重放，你的 Python 代码并不会
再逐行执行；即使能执行，每步把数值从 GPU 拷回 CPU 也会让整个引擎停下来等。想看数值，就把它作为导出的结果返回，
这正是 `@pllm.export` 的用途。同样的道理，`if h.norm() > 1:` 这样依赖张量数值的 Python 分支也不行，改用
`torch.where`。

**为什么不能在第 0 维上求平均。** 第 0 维的每一行是一个不同的请求，顺序由调度器决定。`h.mean(0)` 把别人的请求
混进了你的结果，而且同一个请求的分数会随着"这一步恰好和谁一起跑"而变。`testing.check` 正是通过打乱行的顺序、
换掉别的请求来发现这一点。需要一个固定的"平均激活"时，离线算好，像探针权重一样放进状态里。

## 自测 {#sec-8}

1. 你离线训练探针时用的是 Hugging Face 的 `hidden_states[13]`（第 12 层之后的残差流）。在 pLLM 里该读哪个输入？

<details><summary>答案</summary>

`pllm.residual(12)`。`pllm.hidden(12)` 只是第 12 层 MLP 的输出（加回残差之前），`pllm.final_hidden()` 是最终
norm 之后的向量，三者是不同的量。
</details>

2. 为什么挂上监控负载之后，输出仍和原生 vLLM 逐字相同？

<details><summary>答案</summary>

导出只读不写：它不改 logits、不改状态，模型也仍在 CUDA graph 里运行（`eager_reason` 为空），所以模型的计算和
不挂负载时完全一样。
</details>

3. 一次 `generate` 传入 8 个 prompt，用 `max_rows=4` 的导出读 `tokens="prefill"` 的行，会发生什么？

<details><summary>答案</summary>

同一步里结束 prefill 的 prompt 共用这 4 行，多出来的行被丢弃，丢了多少记在事件的 `dropped_rows` 里，不会阻塞
生成。要拿到每个 prompt 的一行，每次 `generate` 最多传 `max_rows` 个 prompt，或者调大 `max_rows`。本篇的
`CONFIG` 最多同时跑 4 个请求，所以一步最多 4 行。
</details>

下一篇 [02 每请求 LoRA](/pllm/docs/tutorials/02_request_adapters/) 开始改模型：每个请求读完 prompt 之后，由一个超网络
为它现场生成一份 LoRA。

</div>

<div class="zine-doc-footer">
  <a class="zine-pill-link" href="/pllm/#docs">← Back to pLLM Documentation Hub</a>
  <a class="zine-pill-link" href="https://github.com/LinesHogan/pLLM/blob/main/doc/tutorials/01_observe_probe.ipynb" target="_blank" rel="noopener">Edit / View on GitHub ↗</a>
</div>
