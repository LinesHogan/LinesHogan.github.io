---
layout: page
title: "05 Feedback Loop: TTRL"
description: "pLLM Documentation · Step-by-Step Tutorials (00–05)"
permalink: /pllm/docs/tutorials/05_host_feedback/
nav: false
---

<link rel="stylesheet" href="{{ '/assets/css/pllm-hub.css' | relative_url | bust_file_cache }}">

<div class="zine-dispatch" data-title-zh="05 反馈循环：TTRL" data-title-en="05 Feedback Loop: TTRL" data-desc-zh="pLLM 文档 · 交互式入门教程 (00–05)" data-desc-en="pLLM Documentation · Step-by-Step Tutorials (00–05)">
  <div class="zine-dispatch-links">
    <a class="zine-pill-link primary" data-i18n-zh="← 返回 pLLM 文档中心" data-i18n-en="← pLLM Docs Hub" href="/pllm/#docs">← pLLM Docs Hub</a>
    <a class="zine-pill-link" data-i18n-zh="文档导航地图" data-i18n-en="Docs Roadmap" href="/pllm/docs/">Docs Roadmap</a>
    <span class="zine-lang-badge" data-i18n-zh="可执行教程 (.ipynb)" data-i18n-en="Notebook (.ipynb)">Notebook (.ipynb)</span>
  </div>
  <a class="zine-pill-link" data-i18n-zh="在 GitHub 打开 .ipynb ↗" data-i18n-en="Open .ipynb on GitHub ↗" href="https://github.com/LinesHogan/pLLM/blob/main/doc/tutorials/05_host_feedback.ipynb" target="_blank" rel="noopener">Open .ipynb on GitHub ↗</a>
</div>

<div id="markdown-content">

<div class="zine-lang-panel" data-lang-panel="en" lang="en" markdown="1">

<div id="en-sec-1"></div>

**Scenario**: Test-Time Reinforcement Learning (TTRL, Zuo et al. 2025) trains a model on **unlabeled** test questions: sample many candidate answers per question, treat the majority-vote answer as a pseudo-label, reward answers that agree with the majority, update the model, and repeat. It is a closed loop: "generate → score → update → generate again."

Normally this requires two separate systems—vLLM for sampling and a separate training framework for updates, syncing weights from trainer to inference engine every round. In pLLM, sampling and training share a single engine and a single set of weights: the workload declares a LoRA and its training method, while a plain Python loop on the host runs majority voting, packs training batches, and feeds them back into the next `generate` call.

This tutorial implements the simplest form of TTRL: a supervised update step on samples matching the majority answer (equivalent to policy gradient with 0/1 rewards and no baseline). We run a few rounds on 16 GSM8K questions and watch majority accuracy and inter-sample agreement evolve. Finally, we swap the same loop to GRPO: keeping the same reward, but weighting each sample by its within-group advantage.

```python
import os

os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"  # Show only vLLM errors to suppress startup logs

import json
import math
import re
import time
from collections import Counter
from dataclasses import dataclass, replace

import torch
from transformers import AutoTokenizer
from vllm import SamplingParams

import pllm
from pllm.host import CausalTrainBatch

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
```

## Step 1: Workload—One LoRA and One Training Method {#en-step-1-workloadone-lora-and-one-training-method}

- `lora` is an engine-wide `pllm.InstanceState` shared across requests, with low-rank factor fields `a` and `b`. `pllm.Field`'s shape function receives each target projection's input and output widths, so one declaration allocates a pair of factors for `q` and `v` at every layer.
- `adapter` applies `lora` across every layer's `q` and `v` projections: `y = W x + scale · B A x`. `lora_init` initializes `b` to zero so the model starts identical to the base checkpoint, while initializing `a` randomly—if both factors start at zero, every gradient is zero and the LoRA never learns. pLLM therefore requires optimizer-backed low-rank adapters to provide nonzero initialization (`init=pllm.Computed(...)`).
- `learn` is decorated with `@pllm.train(state="lora")`. It reads a host batch (`pllm.batch(...)`) plus `pllm.causal_model("adapter")`: a full causal LM carrying the current LoRA where gradients flow only into the LoRA parameters. `model.token_logprobs(ids, labels, ...)` returns the log-probability of the next label token at each position (and `0` where `labels == -100`), chunking the LM head so the full batch's logits never need to materialize at once. The method returns a scalar loss; the framework handles backward and AdamW.

```python
def lora_init(fields: dict[str, torch.Tensor], generator: torch.Generator) -> None:
    for name, value in fields.items():
        if name.startswith("a_"):
            torch.nn.init.kaiming_uniform_(value, a=math.sqrt(5), generator=generator)
        else:
            value.zero_()


@dataclass(frozen=True)
class TTRL(pllm.Workload):
    rank: int = 8
    learning_rate: float = 1e-4

    @property
    def lora(self) -> pllm.InstanceState:
        return pllm.InstanceState(
            {"a": pllm.Field(lambda inputs, outputs: (self.rank, inputs)),
             "b": pllm.Field(lambda inputs, outputs: (outputs, self.rank))},
            optimizer=pllm.AdamW(self.learning_rate), init=pllm.Computed(lora_init))

    @property
    def adapter(self) -> pllm.model.Adapter:
        targets = (pllm.model.Projection("q_proj"), pllm.model.Projection("v_proj"))  # Every layer
        return pllm.model.Adapter("lora", targets, operator="low_rank", scale=2.0)

    @pllm.train(state="lora")
    def learn(self, input_ids: torch.Tensor = pllm.batch("input_ids"), labels: torch.Tensor = pllm.batch("labels"),
              attention_mask: torch.Tensor = pllm.batch("attention_mask"),
              model: pllm.model.CausalModel = pllm.causal_model("adapter")) -> torch.Tensor:
        logprobs = model.token_logprobs(input_ids, labels, attention_mask=attention_mask)  # [batch, seq_len - 1]
        return -logprobs.sum() / labels[:, 1:].ne(-100).sum()  # Mean negative log-prob over answer tokens
```

In `lora_init`, field names look like `a_0, b_0, a_1, ...`: one slice per target projection, numbered in model order.

## Step 2: Majority Voting on the Host {#en-step-2-majority-voting-on-the-host}

This part is plain Python: extract answers from `\boxed{}`, tally the majority vote, select samples matching the majority, and pack them into a `CausalTrainBatch` (setting prompt label positions to `-100` so loss is computed only on answer tokens). To bound training VRAM per step, each round selects only the 8 questions with the strongest majority consensus, taking one matching sample per question.

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL)
with open("gsm8k_test_16.json") as file:  # First 16 questions from GSM8K test split
    tests = json.load(file)


def normalize(text: str) -> str | None:
    try:
        return f"{float(text.replace(',', '').replace('$', '').strip()):g}"
    except ValueError:
        return None


def boxed(text: str) -> str | None:
    found = re.findall(r"\\boxed\{([^{}]*)\}", text)
    return normalize(found[-1]) if found else None


gold = [normalize(row["a"]) for row in tests]
prompts = [tokenizer.apply_chat_template(
    [{"role": "user", "content": row["q"] + "\nPut the final answer in \\boxed{}."}],
    tokenize=False, add_generation_prompt=True) for row in tests]


def vote(outputs) -> tuple[float, float, list[tuple[list[int], list[int]]]]:
    # Returns: majority-vote accuracy, mean majority vote share, and (prompt, answer) token sequences for training.
    correct, agreement, candidates = 0, 0.0, []
    for prompt, output, answer in zip(prompts, outputs, gold):
        answers = [boxed(sample.text) for sample in output.outputs]
        counts = Counter(a for a in answers if a is not None)
        if not counts:
            continue
        majority, count = counts.most_common(1)[0]
        correct += majority == answer
        agreement += count / len(answers)
        sample = next(s for s, a in zip(output.outputs, answers) if a == majority)
        candidates.append((count, tokenizer(prompt, add_special_tokens=False).input_ids, list(sample.token_ids)))
    # Top 8 questions with highest majority agreement, one matching sample each.
    candidates.sort(key=lambda item: -item[0])
    return correct / len(prompts), agreement / len(prompts), [(p, a) for _, p, a in candidates[:8]]


def training_batch(sequences: list[tuple[list[int], list[int]]]) -> CausalTrainBatch:
    width = max(len(p) + len(a) for p, a in sequences)
    ids = torch.full((len(sequences), width), tokenizer.pad_token_id)
    labels = torch.full_like(ids, -100)
    mask = torch.zeros_like(ids)
    for row, (p, a) in enumerate(sequences):
        ids[row, :len(p) + len(a)] = torch.tensor(p + a)
        labels[row, len(p):len(p) + len(a)] = torch.tensor(a)
        mask[row, :len(p) + len(a)] = 1
    return CausalTrainBatch(ids.cuda(), labels.cuda(), mask.cuda())
```

## Step 3: Closing the Loop {#en-step-3-closing-the-loop}

Each round samples 8 completions for each of the 16 questions (128 concurrent requests), runs majority voting, and passes the selected samples from the previous round via `batch=` into the next `generate` call: the framework runs one AdamW step at the start of that call, and the updated LoRA takes effect immediately for subsequent tokens.
`max_host_training_rows` sets the maximum sequence count accepted per training step and must be declared upfront in `ExecutionOptions` so the framework can budget GPU memory.

```python
# Give vLLM only 25% GPU memory, leaving the rest for full-model forward/backward during training.
config = pllm.ModelConfig(MODEL, max_model_len=640, max_num_seqs=128, max_num_batched_tokens=2048,
                          gpu_memory_utilization=0.25, seed=0,
                          execution=pllm.ExecutionOptions(max_host_training_rows=8))
sampling = SamplingParams(n=8, temperature=0.8, top_p=0.95, max_tokens=256, seed=0)

with pllm.make_llm(config, workload=TTRL()) as llm:
    batch = None
    for round_index in range(5):
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling, batch=batch)
        accuracy, agreement, chosen = vote(outputs)
        print(f"第 {round_index} 轮：多数答案准确率 {accuracy:.2f}，多数票占比 {agreement:.2f}，"
              f"挑出 {len(chosen)} 个样本，用时 {time.perf_counter() - start:.0f}s，"
              f"已更新 {llm.diagnostics().update_count} 次")
        batch = training_batch(chosen)
```

```text
第 0 轮：多数答案准确率 0.31，多数票占比 0.16，挑出 8 个样本，用时 16s，已更新 0 次
第 1 轮：多数答案准确率 0.25，多数票占比 0.16，挑出 8 个样本，用时 54s，已更新 1 次
第 2 轮：多数答案准确率 0.38，多数票占比 0.19，挑出 8 个样本，用时 43s，已更新 2 次
第 3 轮：多数答案准确率 0.38，多数票占比 0.20，挑出 8 个样本，用时 11s，已更新 3 次
第 4 轮：多数答案准确率 0.38，多数票占比 0.24，挑出 8 个样本，用时 44s，已更新 4 次
```

Across five rounds, majority accuracy fluctuates between 0.25 and 0.38 while majority agreement rises from 0.16 to 0.24. On just 16 questions that is only a one- or two-question difference and **not evidence of algorithmic efficacy**: this tutorial demonstrates how to wire the feedback loop, not a reproduction of TTRL results. Agreement is low mainly because this 0.5B model frequently omits `\boxed{}` formatting, and responses without a parsed answer count as 0.

Look at the elapsed times as well: some rounds are noticeably slower (40–55s vs. 11s) because pLLM recompiles the training graph whenever the host training batch takes on a new shape.

## Step 4: Switching to GRPO {#en-step-4-switching-to-grpo}

TTRL uses only samples matching the majority, all with weight `1`. GRPO (Shao et al. 2024) scores each question's group of samples jointly: subtract the group mean reward and divide by the group standard deviation to get each sample's **advantage**; positive-advantage samples are reinforced while negative-advantage samples are suppressed. The reward remains "matches the majority answer."

Only two changes compared to TTRL:

- The training method reads an extra `pllm.batch("weights")`: one scalar weight per sequence—its advantage. The loss computes each sample's mean negative log-probability over answer tokens, weighted by advantage and averaged;
- The host computes advantages and passes them in `CausalTrainBatch.weights`.

Because `max_host_training_rows=8` fits all 8 samples of one question, each round picks the question with the highest non-unanimous majority count (if all 8 agree, rewards are identical and advantages are all zero, yielding zero signal). Here we perform a single online step without importance-ratio clipping or a KL penalty.

```python
@dataclass(frozen=True)
class GRPO(TTRL):
    @pllm.train(state="lora")
    def learn(self, input_ids: torch.Tensor = pllm.batch("input_ids"), labels: torch.Tensor = pllm.batch("labels"),
              attention_mask: torch.Tensor = pllm.batch("attention_mask"),
              weights: torch.Tensor = pllm.batch("weights"),
              model: pllm.model.CausalModel = pllm.causal_model("adapter")) -> torch.Tensor:
        logprobs = model.token_logprobs(input_ids, labels, attention_mask=attention_mask)
        per_sample = logprobs.sum(-1) / labels[:, 1:].ne(-100).sum(-1)  # Mean log-prob over answer tokens per sample
        return -(weights * per_sample).mean()


def grpo_group(outputs) -> CausalTrainBatch:
    """Pick the question with highest non-unanimous majority count: all its samples weighted by within-group normalized advantage."""
    groups = []
    for prompt, output in zip(prompts, outputs):
        answers = [boxed(sample.text) for sample in output.outputs]
        counts = Counter(a for a in answers if a is not None)
        if not counts:
            continue
        majority, count = counts.most_common(1)[0]
        if count == len(answers):  # Unanimous: identical rewards in the group give zero signal
            continue
        rewards = torch.tensor([float(a == majority) for a in answers])
        ids = tokenizer(prompt, add_special_tokens=False).input_ids
        groups.append((count, [(ids, list(sample.token_ids)) for sample in output.outputs],
                       (rewards - rewards.mean()) / rewards.std(correction=0)))
    _, sequences, advantages = max(groups, key=lambda group: group[0])
    return replace(training_batch(sequences), weights=advantages.cuda())
```

```python
with pllm.make_llm(config, workload=GRPO()) as llm:
    batch = None
    for round_index in range(5):
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling, batch=batch)
        accuracy, agreement, _ = vote(outputs)
        print(f"第 {round_index} 轮：多数答案准确率 {accuracy:.2f}，多数票占比 {agreement:.2f}，"
              f"用时 {time.perf_counter() - start:.0f}s，已更新 {llm.diagnostics().update_count} 次")
        batch = grpo_group(outputs)
```

```text
第 0 轮：多数答案准确率 0.31，多数票占比 0.16，用时 11s，已更新 0 次
第 1 轮：多数答案准确率 0.38，多数票占比 0.16，用时 43s，已更新 1 次
第 2 轮：多数答案准确率 0.38，多数票占比 0.17，用时 12s，已更新 2 次
第 3 轮：多数答案准确率 0.31，多数票占比 0.17，用时 44s，已更新 3 次
第 4 轮：多数答案准确率 0.38，多数票占比 0.16，用时 46s，已更新 4 次
```

Across the five GRPO rounds, majority accuracy stays between 0.31 and 0.38 and majority agreement between 0.16 and 0.17. Updating on just 8 samples from a single question per round, as in the TTRL section, only shows the loop executing end-to-end rather than benchmark performance.

## Summary {#en-summary}

- Host-batch training methods read a host batch (`pllm.batch(...)`) and the LoRA-equipped causal model (`pllm.causal_model(...)`), returning a scalar loss;
- Voting, scoring, and sample selection stay in plain host Python, fed back into the engine via the next `generate(batch=...)`;
- Advantage-weighted updates (GRPO) simply add a per-sequence `weights` field to the batch and multiply it in the loss;
- Sampling and training share one engine and one set of weights—no weight synchronization needed.

pLLM also provides a higher-level host feedback helper: `llm.generate_feedback(...)` invokes your feedback callback after each round, retries selected requests according to its decision, and manages voting groups. We didn't use it here because this loop regenerates all questions every round and feeds training batches into the LoRA via `generate(batch=...)`, whereas `generate_feedback` retries only flagged requests and passes `StateImport` between rounds. The built-in `pllm.workloads.RewardHeadFeedback` is one such callback: it runs majority voting on `\boxed{}` answers and fits a reward head (`pllm.workloads.RewardHeadWorkload`) to those 0/1 rewards to predict reward from each response's final hidden state. A reward head trains purely on the host without running the base model: `pllm.host.make_host(config, workload)` returns a `HostTrainer` whose `train(batch)` runs one step.

## Common Mistakes {#en-common-mistakes}

Two common mistakes in host-batch training: passing `reduction=` out of habit from per-row training, and leaving both LoRA factors zero-initialized.

```python
@dataclass(frozen=True)
class SummedTTRL(TTRL):
    @pllm.train(state="lora", reduction="sum")  # Host-batch loss is already a scalar
    def learn(self, input_ids: torch.Tensor = pllm.batch("input_ids"), labels: torch.Tensor = pllm.batch("labels"),
              attention_mask: torch.Tensor = pllm.batch("attention_mask"),
              model: pllm.model.CausalModel = pllm.causal_model("adapter")) -> torch.Tensor:
        return -model.token_logprobs(input_ids, labels, attention_mask=attention_mask).sum()


@dataclass(frozen=True)
class ZeroStart(TTRL):
    @property
    def lora(self) -> pllm.InstanceState:  # No init=: both factors start at zero
        return pllm.InstanceState({"a": pllm.Field(lambda inputs, outputs: (self.rank, inputs)),
                                   "b": pllm.Field(lambda inputs, outputs: (outputs, self.rank))},
                                  optimizer=pllm.AdamW(self.learning_rate))


CPU_CONFIG = pllm.ModelConfig(MODEL, max_num_seqs=8)
for workload in (SummedTTRL(), ZeroStart()):
    try:
        pllm.explain(workload, CPU_CONFIG)
    except ValueError as error:
        print(f"{type(workload).__name__}: {error}\n")
```

```text
SummedTTRL: host-batch training returns one scalar and owns its reduction; remove reduction=

ZeroStart: lora: add init=pllm.Computed(...) (e.g. random A, zero B); with both low_rank factors at zero every gradient is zero and the adapter never learns
```

**Why host-batch training rejects `reduction=`.** In per-row training, the scheduler determines how many rows belong to each state copy in a step, so pLLM must perform the reduction. A host batch consists of complete sequences you assembled yourself—only you know sequence lengths, weights, and whether to average per token or per sequence (precisely what distinguishes GRPO, DAPO, and Dr.GRPO)—so the method returns a scalar directly and owns its reduction.

**Why zero-initialized LoRA never learns.** In `B A x`, the gradient with respect to `A` contains `B`, and the gradient with respect to `B` contains `A`. When both are zero, both gradients are identically zero and the optimizer can never move them away from zero. The standard initialization is random `A` and zero `B`: zero delta before training starts, nonzero gradients on step 1.

## Self-Check {#en-self-check}

1. Why must `max_host_training_rows` be declared upfront in `ExecutionOptions`?

<details><summary>Answer</summary>

Full-model training needs substantial temporary GPU memory. pLLM reserves that memory upfront at engine startup according to your declared maximum batch size instead of gambling on dynamic allocations mid-generation; batches exceeding the declared row limit are rejected.
</details>

2. Why were a few rounds much slower than others?

<details><summary>Answer</summary>

Whenever the training batch changes shape (number of sequences or sequence length), pLLM recompiles the training step for the new shape. Once shapes stabilize, recompilation stops.
</details>

3. What changed in the workload and host code when switching from TTRL to GRPO?

<details><summary>Answer</summary>

The workload's training method added one `pllm.batch("weights")` input and multiplied each sample's loss by its weight; the host code computed within-group advantages and passed them in `CausalTrainBatch.weights`. The engine, LoRA, and loop structure stayed unchanged.
</details>

This concludes the core tutorial track. From here, explore as needed: [Workload Anatomy](/pllm/docs/guides/workload-anatomy/) systematically organizes every pattern introduced across these six notebooks; [Core Concepts](/pllm/docs/explain/core-concepts/) explains the design behind the rules; and the [Examples](/pllm/docs/#en-examples) cover one algorithm family per notebook, such as [06 Reading Inside](/pllm/docs/examples/06_reading_inside/) for reading the residual stream and per-layer logits to implement DoLa.

</div>

<div class="zine-lang-panel is-hidden" data-lang-panel="zh" lang="zh-CN" markdown="1">

<div id="zh-sec-1"></div>

**场景**：测试时强化学习（TTRL，Zuo et al. 2025）在**没有标签**的测试题上训练模型：每道题采样很多个
答案，多数投票的结果当作伪标签，和多数一致的答案得到奖励，然后更新模型，再去做下一轮。它是一个
"生成 → 打分 → 更新 → 再生成"的循环。

通常这要两套系统：vLLM 负责采样，另一个训练框架负责更新，每一轮把权重从训练端同步回推理端。
在 pLLM 里，采样和训练用的是同一个引擎、同一份模型：负载声明一个 LoRA 和它的训练方法，
宿主机上的 Python 循环负责投票和组织训练数据，下一次 `generate` 时一起送进去。

这篇实现 TTRL 最简单的形式：对和多数答案一致的样本做一步监督更新（等价于奖励取 0/1、不减基线的
策略梯度）。在 GSM8K 的 16 道题上跑几轮，观察多数答案的准确率和样本之间的一致程度。最后把同一个循环换成
GRPO：奖励不变，按组内优势给每个样本加权。

```python
import os

os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"  # 只显示 vLLM 的报错，省掉启动日志

import json
import math
import re
import time
from collections import Counter
from dataclasses import dataclass, replace

import torch
from transformers import AutoTokenizer
from vllm import SamplingParams

import pllm
from pllm.host import CausalTrainBatch

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
```

## 第一步：负载，一个 LoRA 加一个训练方法 {#zh-sec-2}

- `lora` 是整个实例共享的一份状态（`pllm.InstanceState`），字段 `a`、`b` 是低秩因子。`pllm.Field` 的形状
  函数拿到每个目标投影的输入、输出宽度，所以一个声明就为每层的 q、v 投影各生成一对因子。
- `adapter` 把它作为 LoRA 加到每一层的 q、v 投影上：`y = W x + scale · B A x`。`lora_init` 把 `b` 初始化为零，
  所以训练前模型和原来完全一样；`a` 随机初始化，因为两个因子都是零时梯度也恒为零，LoRA 永远学不动。所以
  pLLM 要求带优化器的低秩适配器给出非零的初值（`init=pllm.Computed(...)`）。
- `learn` 带 `@pllm.train(state="lora")`。它的参数来自宿主机送来的一个批次（`pllm.batch(...)`），以及
  `pllm.causal_model("adapter")`：一个装着当前 LoRA 的完整因果语言模型，梯度只流向 LoRA。
  `model.token_logprobs(ids, labels, ...)` 给出每个位置上下一个标签 token 的对数概率，标签为 -100 的位置给 0；
  它分块算 LM head，不需要把整个批次的 logits 一次摊开。方法返回一个标量 loss，框架负责反向和 AdamW。

```python
def lora_init(fields: dict[str, torch.Tensor], generator: torch.Generator) -> None:
    for name, value in fields.items():
        if name.startswith("a_"):
            torch.nn.init.kaiming_uniform_(value, a=math.sqrt(5), generator=generator)
        else:
            value.zero_()


@dataclass(frozen=True)
class TTRL(pllm.Workload):
    rank: int = 8
    learning_rate: float = 1e-4

    @property
    def lora(self) -> pllm.InstanceState:
        return pllm.InstanceState(
            {"a": pllm.Field(lambda inputs, outputs: (self.rank, inputs)),
             "b": pllm.Field(lambda inputs, outputs: (outputs, self.rank))},
            optimizer=pllm.AdamW(self.learning_rate), init=pllm.Computed(lora_init))

    @property
    def adapter(self) -> pllm.model.Adapter:
        targets = (pllm.model.Projection("q_proj"), pllm.model.Projection("v_proj"))  # 每一层
        return pllm.model.Adapter("lora", targets, operator="low_rank", scale=2.0)

    @pllm.train(state="lora")
    def learn(self, input_ids: torch.Tensor = pllm.batch("input_ids"), labels: torch.Tensor = pllm.batch("labels"),
              attention_mask: torch.Tensor = pllm.batch("attention_mask"),
              model: pllm.model.CausalModel = pllm.causal_model("adapter")) -> torch.Tensor:
        logprobs = model.token_logprobs(input_ids, labels, attention_mask=attention_mask)  # [序列, token 数 - 1]
        return -logprobs.sum() / labels[:, 1:].ne(-100).sum()  # 答案 token 上的平均负对数概率
```

`lora_init` 里的字段名是 `a_0, b_0, a_1, ...`：一个字段在每个目标投影上各有一份，按模型里的顺序编号。

## 第二步：宿主机上的投票 {#zh-sec-3}

这部分是普通 Python：从 `\boxed{}` 里取答案，统计多数，挑出和多数一致的样本，拼成一个
`CausalTrainBatch`（prompt 部分的标签设为 -100，只在答案上算 loss）。为了控制一次训练的显存，
每轮只取多数票最集中的 8 道题，各取一个和多数一致的样本。

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL)
with open("gsm8k_test_16.json") as file:  # GSM8K 测试集的前 16 题
    tests = json.load(file)


def normalize(text: str) -> str | None:
    try:
        return f"{float(text.replace(',', '').replace('$', '').strip()):g}"
    except ValueError:
        return None


def boxed(text: str) -> str | None:
    found = re.findall(r"\\boxed\{([^{}]*)\}", text)
    return normalize(found[-1]) if found else None


gold = [normalize(row["a"]) for row in tests]
prompts = [tokenizer.apply_chat_template(
    [{"role": "user", "content": row["q"] + "\nPut the final answer in \\boxed{}."}],
    tokenize=False, add_generation_prompt=True) for row in tests]


def vote(outputs) -> tuple[float, float, list[tuple[list[int], list[int]]]]:
    # 返回：多数答案的准确率、多数票占比的平均值、用于训练的 (prompt, 答案) token 序列。
    correct, agreement, candidates = 0, 0.0, []
    for prompt, output, answer in zip(prompts, outputs, gold):
        answers = [boxed(sample.text) for sample in output.outputs]
        counts = Counter(a for a in answers if a is not None)
        if not counts:
            continue
        majority, count = counts.most_common(1)[0]
        correct += majority == answer
        agreement += count / len(answers)
        sample = next(s for s, a in zip(output.outputs, answers) if a == majority)
        candidates.append((count, tokenizer(prompt, add_special_tokens=False).input_ids, list(sample.token_ids)))
    # 多数票最集中的 8 道题，各取一个和多数一致的样本。
    candidates.sort(key=lambda item: -item[0])
    return correct / len(prompts), agreement / len(prompts), [(p, a) for _, p, a in candidates[:8]]


def training_batch(sequences: list[tuple[list[int], list[int]]]) -> CausalTrainBatch:
    width = max(len(p) + len(a) for p, a in sequences)
    ids = torch.full((len(sequences), width), tokenizer.pad_token_id)
    labels = torch.full_like(ids, -100)
    mask = torch.zeros_like(ids)
    for row, (p, a) in enumerate(sequences):
        ids[row, :len(p) + len(a)] = torch.tensor(p + a)
        labels[row, len(p):len(p) + len(a)] = torch.tensor(a)
        mask[row, :len(p) + len(a)] = 1
    return CausalTrainBatch(ids.cuda(), labels.cuda(), mask.cuda())
```

## 第三步：闭环 {#zh-sec-4}

每轮对 16 道题各采样 8 个答案（128 个并发请求），投票，把上一轮挑出的样本作为 `batch=` 送进这一轮的
`generate`：框架在这次调用开始时用它做一步 AdamW 更新，更新后的 LoRA 立刻被后续 token 使用。
`max_host_training_rows` 是一次训练最多接受的序列数，要在 `ExecutionOptions` 里事先声明，框架据此预算显存。

```python
# vLLM 只拿 25% 显存，剩下的留给训练时整模型的前向和反向。
config = pllm.ModelConfig(MODEL, max_model_len=640, max_num_seqs=128, max_num_batched_tokens=2048,
                          gpu_memory_utilization=0.25, seed=0,
                          execution=pllm.ExecutionOptions(max_host_training_rows=8))
sampling = SamplingParams(n=8, temperature=0.8, top_p=0.95, max_tokens=256, seed=0)

with pllm.make_llm(config, workload=TTRL()) as llm:
    batch = None
    for round_index in range(5):
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling, batch=batch)
        accuracy, agreement, chosen = vote(outputs)
        print(f"第 {round_index} 轮：多数答案准确率 {accuracy:.2f}，多数票占比 {agreement:.2f}，"
              f"挑出 {len(chosen)} 个样本，用时 {time.perf_counter() - start:.0f}s，"
              f"已更新 {llm.diagnostics().update_count} 次")
        batch = training_batch(chosen)
```

```text
第 0 轮：多数答案准确率 0.31，多数票占比 0.16，挑出 8 个样本，用时 16s，已更新 0 次
第 1 轮：多数答案准确率 0.25，多数票占比 0.16，挑出 8 个样本，用时 54s，已更新 1 次
第 2 轮：多数答案准确率 0.38，多数票占比 0.19，挑出 8 个样本，用时 43s，已更新 2 次
第 3 轮：多数答案准确率 0.38，多数票占比 0.20，挑出 8 个样本，用时 11s，已更新 3 次
第 4 轮：多数答案准确率 0.38，多数票占比 0.24，挑出 8 个样本，用时 44s，已更新 4 次
```

五轮下来，多数答案的准确率在 0.25 到 0.38 之间起伏，多数票占比从 0.16 升到 0.24。16 道题上这只是一两道题的差别，
**不足以说明效果**：这一篇演示的是闭环怎么搭，不是复现 TTRL 的结论。多数票占比偏低，是因为这个 0.5B 模型
经常不按要求把答案写进 `\boxed{}`，没有答案的题按 0 计。

用时那一列也值得看：有几轮明显更慢（40 到 55 秒对 11 秒），多出来的时间花在训练步的编译上，pLLM 遇到
新的批次形状时要重新编译训练图。

## 第四步：换成 GRPO {#zh-sec-5}

TTRL 只用和多数一致的样本，权重都是 1。GRPO（Shao et al. 2024）把一道题的一组样本放在一起看：每个样本的
奖励减去组内平均、除以组内标准差，得到它的**优势**；优势为正的样本被推高概率，为负的被压低。奖励仍然是
"和多数答案一致"。

和 TTRL 相比只改两处：

- 训练方法多读一个 `pllm.batch("weights")`：每个序列一个权重，这里就是优势。loss 是每个样本答案 token 的
  平均负对数概率，按优势加权后取平均；
- 宿主机上算出优势，放进 `CausalTrainBatch` 的 `weights` 字段。

`max_host_training_rows=8` 正好放下一道题的 8 个样本，所以每轮挑多数票最集中、但不是全票的那道题（全票时组内
奖励都一样，优势全为零，没有信号）。这里只做最简单的一步在线更新，没有重要性比率的裁剪，也没有 KL 惩罚。

```python
@dataclass(frozen=True)
class GRPO(TTRL):
    @pllm.train(state="lora")
    def learn(self, input_ids: torch.Tensor = pllm.batch("input_ids"), labels: torch.Tensor = pllm.batch("labels"),
              attention_mask: torch.Tensor = pllm.batch("attention_mask"),
              weights: torch.Tensor = pllm.batch("weights"),
              model: pllm.model.CausalModel = pllm.causal_model("adapter")) -> torch.Tensor:
        logprobs = model.token_logprobs(input_ids, labels, attention_mask=attention_mask)
        per_sample = logprobs.sum(-1) / labels[:, 1:].ne(-100).sum(-1)  # 每个样本答案 token 的平均对数概率
        return -(weights * per_sample).mean()


def grpo_group(outputs) -> CausalTrainBatch:
    """多数票最集中、但不是全票的那道题：它的全部样本，权重是组内归一化的优势。"""
    groups = []
    for prompt, output in zip(prompts, outputs):
        answers = [boxed(sample.text) for sample in output.outputs]
        counts = Counter(a for a in answers if a is not None)
        if not counts:
            continue
        majority, count = counts.most_common(1)[0]
        if count == len(answers):  # 全票：组内奖励都一样，没有信号
            continue
        rewards = torch.tensor([float(a == majority) for a in answers])
        ids = tokenizer(prompt, add_special_tokens=False).input_ids
        groups.append((count, [(ids, list(sample.token_ids)) for sample in output.outputs],
                       (rewards - rewards.mean()) / rewards.std(correction=0)))
    _, sequences, advantages = max(groups, key=lambda group: group[0])
    return replace(training_batch(sequences), weights=advantages.cuda())
```

```python
with pllm.make_llm(config, workload=GRPO()) as llm:
    batch = None
    for round_index in range(5):
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling, batch=batch)
        accuracy, agreement, _ = vote(outputs)
        print(f"第 {round_index} 轮：多数答案准确率 {accuracy:.2f}，多数票占比 {agreement:.2f}，"
              f"用时 {time.perf_counter() - start:.0f}s，已更新 {llm.diagnostics().update_count} 次")
        batch = grpo_group(outputs)
```

```text
第 0 轮：多数答案准确率 0.31，多数票占比 0.16，用时 11s，已更新 0 次
第 1 轮：多数答案准确率 0.38，多数票占比 0.16，用时 43s，已更新 1 次
第 2 轮：多数答案准确率 0.38，多数票占比 0.17，用时 12s，已更新 2 次
第 3 轮：多数答案准确率 0.31，多数票占比 0.17，用时 44s，已更新 3 次
第 4 轮：多数答案准确率 0.38，多数票占比 0.16，用时 46s，已更新 4 次
```

GRPO 的五轮里，多数答案准确率在 0.31 和 0.38 之间，多数票占比在 0.16 到 0.17 之间。每轮只用一道题的 8 个样本
更新一次，和 TTRL 那一节一样，这些数字只说明循环跑通了，不说明效果。

## 小结 {#zh-sec-6}

- 训练方法读宿主机批次（`pllm.batch(...)`）和带 LoRA 的整模型（`pllm.causal_model(...)`），返回一个标量 loss；
- 投票、奖励、挑样本都是宿主机上的普通 Python，结果通过下一次 `generate(batch=...)` 送进引擎；
- 带优势的更新（GRPO）只是批次多一个逐序列的 `weights` 字段、loss 里多乘一个权重；
- 采样和训练共用一个引擎、一份权重，没有权重同步。

pLLM 也有更完整的宿主反馈接口：`llm.generate_feedback(...)` 每轮生成后调用你的反馈函数，按它的决定重试部分
请求，并管理投票组。这一篇没有用它，因为这里每轮都要重新生成全部题目，并把训练批次通过 `generate(batch=...)`
交给 LoRA；`generate_feedback` 只重试被标记的请求，两轮之间也只能送入 `StateImport`。内置的 `pllm.workloads.RewardHeadFeedback` 就是这样一个反馈函数：它对每道题的
`\boxed{}` 答案做多数投票，再把一个奖励头（`pllm.workloads.RewardHeadWorkload`）拟合到投票得出的 0/1 奖励上，
让它从每个回答的最终 hidden 预测奖励。奖励头只在宿主上训练，不需要模型：`pllm.host.make_host(config, workload)`
返回一个 `HostTrainer`，每次 `train(batch)` 走一步。

## 常见错误 {#zh-sec-7}

宿主批训练有两个常见错误：照着逐行训练的习惯加 `reduction=`；以及让 LoRA 的两个因子都从零开始。

```python
@dataclass(frozen=True)
class SummedTTRL(TTRL):
    @pllm.train(state="lora", reduction="sum")  # 宿主批次上的 loss 已经是一个数
    def learn(self, input_ids: torch.Tensor = pllm.batch("input_ids"), labels: torch.Tensor = pllm.batch("labels"),
              attention_mask: torch.Tensor = pllm.batch("attention_mask"),
              model: pllm.model.CausalModel = pllm.causal_model("adapter")) -> torch.Tensor:
        return -model.token_logprobs(input_ids, labels, attention_mask=attention_mask).sum()


@dataclass(frozen=True)
class ZeroStart(TTRL):
    @property
    def lora(self) -> pllm.InstanceState:  # 没有 init=：两个因子都是零
        return pllm.InstanceState({"a": pllm.Field(lambda inputs, outputs: (self.rank, inputs)),
                                   "b": pllm.Field(lambda inputs, outputs: (outputs, self.rank))},
                                  optimizer=pllm.AdamW(self.learning_rate))


CPU_CONFIG = pllm.ModelConfig(MODEL, max_num_seqs=8)
for workload in (SummedTTRL(), ZeroStart()):
    try:
        pllm.explain(workload, CPU_CONFIG)
    except ValueError as error:
        print(f"{type(workload).__name__}: {error}\n")
```

```text
SummedTTRL: host-batch training returns one scalar and owns its reduction; remove reduction=

ZeroStart: lora: add init=pllm.Computed(...) (e.g. random A, zero B); with both low_rank factors at zero every gradient is zero and the adapter never learns
```

**为什么宿主批训练不接受 `reduction=`。** 逐行训练时，一步里每份状态有多少行由调度决定，所以要由 pLLM 归约。
宿主批次是你自己拼好的完整序列，长度、权重、按 token 还是按序列平均（GRPO、DAPO、Dr.GRPO 的区别正在这里）
都只有你知道，所以方法直接返回一个标量，归约写在你的数学里。

**为什么零初始化的 LoRA 学不动。** `B A x` 对 A 的梯度含 B、对 B 的梯度含 A，两个都是零时梯度恒为零，
优化器永远不会让它们离开零。常规做法是 A 随机、B 为零：训练前增量为零，第一步就有梯度。

## 自测 {#zh-sec-8}

1. 为什么 `max_host_training_rows` 要在 `ExecutionOptions` 里事先声明？

<details><summary>答案</summary>

整模型的训练要临时占用不少显存。pLLM 在启动引擎时按你声明的最大批次预留这部分显存，而不是在生成途中临时
申请、碰运气；超过声明行数的批次会被拒绝。
</details>

2. 为什么有几轮比其他轮慢很多？

<details><summary>答案</summary>

训练批次的形状（序列数、序列长度）变了，pLLM 要为新形状重新编译训练步。形状稳定后就不再重编。
</details>

3. 从 TTRL 改成 GRPO，负载和宿主代码各改了什么？

<details><summary>答案</summary>

负载的训练方法多读一个 `pllm.batch("weights")`，把每个样本的 loss 乘上权重；宿主代码算出组内优势，放进
`CausalTrainBatch` 的 `weights` 字段。引擎、LoRA 和循环结构都不变。
</details>

主线教程到这里结束。接下来按需要选读：[负载的组成](/pllm/docs/guides/workload-anatomy/) 把六篇里出现的写法系统地整理了一遍，[核心概念](/pllm/docs/explain/core-concepts/) 解释这些规则背后的原因；[示例库](/pllm/docs/#zh-sec-8) 里每一篇讲一个算法族，比如 [06 读模型内部](/pllm/docs/examples/06_reading_inside/) 读残差流和逐层 logits、实现 DoLa。

</div>

</div>

<div class="zine-doc-footer">
  <a class="zine-pill-link" data-i18n-zh="← 返回 pLLM 项目与文档中心" data-i18n-en="← Back to pLLM Project &amp; Docs" href="/pllm/#docs">← Back to pLLM Project &amp; Docs</a>
  <a class="zine-pill-link" href="https://github.com/LinesHogan/pLLM/blob/main/doc/tutorials/05_host_feedback.ipynb" target="_blank" rel="noopener">GitHub &middot; doc/tutorials/05_host_feedback.ipynb ↗</a>
</div>

<script defer src="/assets/js/bilingual-post.js"></script>
