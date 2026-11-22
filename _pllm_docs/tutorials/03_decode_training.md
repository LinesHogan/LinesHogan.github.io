---
layout: page
title: "03 decode 时训练：ESamp"
description: "pLLM · Step-by-Step Tutorials (00–05) (交互式入门教程 (00–05))"
permalink: /pllm/docs/tutorials/03_decode_training/
nav: false
---

<div class="zine-dispatch">
  <div class="zine-dispatch-links">
    <a class="zine-pill-link primary" href="/pllm/#docs">← pLLM Project &amp; Docs</a>
    <a class="zine-pill-link" href="/pllm/docs/">All Docs Index</a>
    <span class="zine-lang-badge">Notebook (.ipynb)</span>
  </div>
  <a class="zine-pill-link" href="https://github.com/LinesHogan/pLLM/blob/main/doc/tutorials/03_decode_training.ipynb" target="_blank" rel="noopener">Open .ipynb on GitHub ↗</a>
</div>

<div id="markdown-content" markdown="1">

<div id="sec-1"></div>

**场景**：有一类算法要在**生成的同时学习**：每个请求带着自己的一小份参数，每生成一个 token
就用这个 token 上的信号更新一次，更新后的参数再影响后面的生成。测试时训练（test-time training）
的很多变体都是这个形状。这篇用 ESamp 做例子：

> 每个请求训练一个小网络（distiller），用第 0 层的 hidden 预测最后一层的 hidden。采样时，把这个
> 浅层预测对应的 logits 按强度 beta 从模型的 logits 里对减：浅层就能猜到的 token 被压低，深层额外
> 贡献的 token 被抬高。

在推理引擎里手写这件事很费劲：autograd 要在推理模式里打开；每个请求要有自己的参数和 AdamW 状态，
请求来来去去，显存要分配和回收；反向和优化器步要和下一步 forward 重叠，否则吞吐直接腰斩；最后
还想让它们都进 CUDA graph。

这篇你只写：小网络本身（普通 `nn.Module`）、一个 loss、一个预测。然后在 128 个并发请求上测两件事：
采样多样性有没有变化，吞吐还剩多少。

```python
import os

os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"  # 只显示 vLLM 的报错，省掉启动日志

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from vllm import SamplingParams

import pllm
from pllm import testing
from pllm.benchmark import throughput

# 这一篇要测吞吐，用 7B：在 0.5B 上一步 decode 只有几毫秒，任何固定开销都会被放大，7B 才是这类算法
# 常见的部署规模。bf16 的 7B 权重约 15 GiB，需要一张 24 GiB 的卡。
MODEL = "Qwen/Qwen2.5-7B-Instruct"
# 128 个并发请求（8 个 prompt × 每个 16 个样本）。
CONFIG = pllm.ModelConfig(MODEL, max_model_len=512, max_num_seqs=128, max_num_batched_tokens=8192,
                          gpu_memory_utilization=0.85, seed=7)
```

## 第一步：小网络就是普通的 PyTorch {#sec-2}

一个低秩残差块加 layer norm。`up` 初始化为零，所以训练开始前它只是把输入做一次 layer norm。

```python
class Distiller(nn.Module):
    def __init__(self, width: int, rank: int) -> None:
        super().__init__()
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + self.up(F.silu(self.down(x)))
        return F.layer_norm(y, y.shape[-1:])
```

## 第二步：负载 {#sec-3}

- `distiller` 是一份 `pllm.RequestState`：**每个请求一份**。`init=pllm.FromModule(...)` 给出怎么构造这个网络
  （`m` 是加载模型后才知道的维度，`m.hidden_size` 对这里的 7B 模型是 3584），它的参数就是状态，
  `optimizer=` 说明用什么更新它。请求结束时这份状态连同 AdamW 的动量一起回收。
- `loss` 带 `@pllm.train(state="distiller")`：返回**每一行**的 loss（一行是一个请求当前的 token）。
  框架负责屏蔽填充行、求平均（`reduction="mean"` 表示先在每个请求内平均）、反向、应用 AdamW、
  发布新版本。`pllm.module("distiller")` 拿到的是"每一行用自己那个请求的参数"的网络。
- `predict` 带 `@pllm.guide_sampling(beta=...)`：返回的预测会被送进 LM head，采样分布变成
  `(1 + beta) * logits - beta * 预测的 logits`。`rows="trained"` 让还没训练过的请求不受影响；
  `trained` 方法用 `pllm.version` 读每个请求的状态被更新过几次。

```python
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
```

负载写成 frozen dataclass，`beta` 和 `rank` 是它的字段；`beta=strength` 把装饰器的参数交给上面定义的方法，
pLLM 在绑定负载时调用它一次，读到这个实例的 `beta`，这样同一个类可以用不同的 `beta` 实例化。IDE 能从
`strength` 跳到方法定义，类型检查器也会核对它返回 `float`。注意不要直接写 `beta=beta`：类体里的字段名
指向的是类默认值，不是实例的值。

## 第三步：128 个请求，看吞吐和输出 {#sec-4}

8 个 prompt，每个采样 16 个续写，各 256 个 token。三组对照：

- **原生 vLLM**：不带负载；
- **ESamp，beta = 0**：每个请求照常训练小网络，但不引导采样。训练不应该改变任何一个采样结果；
- **ESamp，beta = 0.5**：训练并引导采样。

每组先生成一次，记下输出；三组用同一个随机种子，所以可以逐个样本比较。吞吐用 `pllm.benchmark.throughput`
测：先预热一轮，再计时三轮取中位数，每轮计整次 `generate` 的时间（含 prefill），按实际生成的 token 数算。

```python
prompts = [f"Explain the mathematical idea numbered {i}:" for i in range(8)]
sampling = SamplingParams(n=16, max_tokens=256, temperature=0.8, top_p=0.95, ignore_eos=True, seed=7)


def samples(llm) -> list[str]:
    return [sample.text for output in llm.generate(prompts, sampling, use_tqdm=False) for sample in output.outputs]


with pllm.make_llm(CONFIG) as plain:
    plain_texts = samples(plain)
    plain_rate = throughput(plain, prompts, sampling)
print(f"原生 vLLM        : {plain_rate:6.0f} token/s")

for beta in (0.0, 0.5):
    with pllm.make_llm(CONFIG, workload=MiniESamp(beta)) as llm:
        texts = samples(llm)
        rate = throughput(llm, prompts, sampling)
        report = llm.diagnostics()
    same = sum(a == b for a, b in zip(texts, plain_texts))
    print(f"ESamp beta={beta:<4}: {rate:6.0f} token/s，为原生的 {rate / plain_rate:.0%}；"
          f"{same}/{len(texts)} 个样本与原生相同；累计训练 {report.trained_rows} 行；"
          f"eager 原因 {report.eager_reason!r}")
```

```text
原生 vLLM        :   5896 token/s
ESamp beta=0.0 :   5786 token/s，为原生的 98%；128/128 个样本与原生相同；累计训练 163200 行；eager 原因 ''
ESamp beta=0.5 :   5729 token/s，为原生的 97%；3/128 个样本与原生相同；累计训练 163200 行；eager 原因 ''
```

三行结果说明三件事：

- **训练不打扰生成**：beta = 0 时每个请求都在训练（"累计训练"是这个引擎上五次生成里所有请求、所有 token 上的
  训练行数），128 个样本和原生 vLLM 逐字相同。偶尔也会有一两个样本在中途分叉，那是浮点舍入层面的差异：带负载的
  引擎要从 forward 里取出 hidden，编译出的 graph 和原生不完全相同，256 个 token 的随机采样可能在几乎打平的地方
  走向另一边；
- **代价很小**：每个 token、每个请求都做一次前向、反向和 AdamW，这次运行的吞吐只比原生低两三个百分点，因为训练
  在旁路 stream 上和下一步 forward 重叠；
- **引导确实生效**：beta = 0.5 时 128 个样本里有 125 个变了，多出来的代价是每步对预测再做一次 LM head
  和对采样分布的修改。

这些数字来自本地 RTX 4090 上的一次运行（Qwen2.5-7B-Instruct、bf16、vLLM 0.32.0、CUDA graph；8 个 prompt × 16 个样本
= 128 个并发请求、各 256 token；预热 1 轮后 3 轮取中位数），不同机器、不同轮次会有一两个百分点的浮动。同样设置下对完整 ESamp 的基准测量见
[实测记录](https://github.com/LinesHogan/pLLM/blob/main/doc/reference/measurements.toml)：只训练为原生的 97.4%，训练加引导为 96.3%。

## 框架替你做了什么 {#sec-5}

- **每个请求的参数和优化器状态**：128 份小网络加各自的 AdamW 动量，按页分配在启动时就预算好的显存里，
  请求结束即回收；
- **CUDA graph**：`eager 原因` 为空，说明模型 forward、你的 loss、反向和 AdamW 更新都在 CUDA graph 里；
- **重叠**：训练放在旁路 stream 上，和下一步 forward 同时跑。代价是一步的延迟：第 N 步的更新在第 N+2 步
  才被预测读到。需要严格顺序时用 `pllm.ExecutionOptions(publish_lag=0)`；
- **调度**：你没有说训练在哪个 stream、什么时候同步。pLLM 的规划器从声明里看出 `loss` 只写
  `distiller`、`predict` 只读它，据此安排顺序和重叠。

完整的 ESamp 在 `pllm.workloads.ESampWorkload`，和这里的 `MiniESamp` 是同一个结构，只是小网络多了门控。

## 常见错误 {#sec-6}

训练负载有三个常见错误：自己把 loss 平均成一个数；在类体里直接写 `beta=beta`；以及给训练方法加 `rows=`。前两个
`testing.check` 能在 CPU 上抓到，第三个在构建时就被拒绝。

```python
class ScalarLoss(MiniESamp):
    @pllm.train(state="distiller")
    def loss(self, shallow: torch.Tensor = pllm.hidden(0), deep: torch.Tensor = pllm.hidden(-1),
             f: Distiller = pllm.module("distiller")) -> torch.Tensor:
        return (f(shallow) - deep).square().mean()  # 把所有行平均成了一个数


@dataclass(frozen=True)
class ClassDefault(MiniESamp):
    beta: float = 0.5

    @pllm.guide_sampling(beta=beta, rows="trained")  # 读到的是类默认值 0.5，不是实例的 beta
    def predict(self, shallow: torch.Tensor = pllm.hidden(0), f: Distiller = pllm.module("distiller")) -> torch.Tensor:
        return f(shallow)


class SelectedTraining(MiniESamp):
    @pllm.train(state="distiller", rows="trained")  # 想只训练已经训练过的请求
    def loss(self, shallow: torch.Tensor = pllm.hidden(0), deep: torch.Tensor = pllm.hidden(-1),
             f: Distiller = pllm.module("distiller")) -> torch.Tensor:
        return (f(shallow) - deep).square().mean(-1)


SMALL = pllm.ModelConfig(MODEL, max_num_seqs=8)
for name, run in (("ScalarLoss", lambda: testing.check(ScalarLoss(0.5), SMALL)),
                  ("ClassDefault", lambda: testing.check(ClassDefault(), SMALL, neutral=ClassDefault(beta=0.0))),
                  ("SelectedTraining", lambda: pllm.explain(SelectedTraining(0.5), SMALL))):
    try:
        run()
    except testing.ContractError as error:
        print(f"{name}: {error.failures[0]}\n")
    except ValueError as error:
        print(f"{name}: {error}\n")
```

```text
ScalarLoss: [compile] ValueError: training methods return one loss per row; pLLM reduces the selected rows per state
while shape tracing method 'loss'; methods run inside CUDA graphs: use tensor operations such as torch.where, not .item() or Python branches on tensor values

ClassDefault: [neutral] guide_sampling still declares a sampler stage; set its beta to zero.

SelectedTraining: rows= and update_rows= on request training require a model effect without sampler guidance, or @pllm.add_logits; standalone training and @pllm.guide_sampling update every selected state, so remove rows= and update_rows=
```

**为什么每行返回一个 loss。** 一步里有多少行属于同一个请求的状态，取决于调度：decode 时一行，prompt 被切块时
可能很多行，共享状态时还会有别的请求的行。只有 pLLM 知道这些行属于哪份状态，所以由它在每份状态内部求平均
（`reduction="mean"`）或求和（`"sum"`）。你自己 `.mean()` 会把不同请求的 loss 搅在一起。

**为什么 `beta=beta` 不对。** 类体执行时还没有实例，`beta` 这个名字指向的是类的默认值 0.5，所以 `beta=0` 的实例
仍在引导采样。`neutral=` 检查（"关闭开关时不能改变采样"）正好抓住它。把设置写成方法（`def strength(self)`）
再交给装饰器，pLLM 会对每个实例调用一次。

**为什么单独的训练不接受 `rows=`。** 只训练、或训练与 `@pllm.guide_sampling` 同在时，训练走的是为"每个请求每步
都学"设计的路径：每一步给每个在跑的请求的状态做一次优化器更新、发布一个新版本，并和下一步 forward 重叠。按行
跳过更新需要"有条件地发布"，这条路径目前没有实现，所以构建时直接拒绝，而不是悄悄忽略你的掩码。想让一部分行
不计入：在采样方法上用 `rows=`（本篇的 `predict` 就是这样），或者改用 `reduction="sum"` 并把这些行的 loss
乘以 0（用默认的 `"mean"` 时，乘 0 的行仍算在分母里）。与 `@pllm.add_logits` 或模型改写（`Adapter` 等）一起
时，训练是有条件的更新，`rows=` 可以用。

## 自测 {#sec-7}

1. 默认设置下，第 10 步算出的训练更新，`predict` 最早在第几步读到？为什么要晚这一步？

<details><summary>答案</summary>

第 12 步。默认 `publish_lag=1`：训练放在旁路 stream 上和第 11 步的 forward 同时跑，所以第 11 步还读不到它。
这一步的延迟换来了训练和生成的重叠；需要严格顺序时设 `pllm.ExecutionOptions(publish_lag=0)`，代价是训练不再
重叠。
</details>

2. 为什么 beta = 0 时，每个请求都在训练，采样结果却应该和原生 vLLM 一样？

<details><summary>答案</summary>

训练只更新 `distiller` 这份状态，不碰 logits；`beta=0` 的 `guide_sampling` 不声明任何采样器工作。模型的
forward 和采样都和原生一样，所以样本逐字相同（个别样本的分叉来自编译出的 graph 不同带来的舍入差异）。
</details>

3. `rows="trained"` 去掉之后，刚开始生成的请求会怎样？

<details><summary>答案</summary>

还没训练过的请求，小网络只是一个 layer norm，它的"预测"没有意义，却会按 beta 引导采样。`rows="trained"`
让版本号为 0 的请求照常采样，等它们的状态被更新过一次再开始引导。
</details>

下一篇 [04 改写 forward](/pllm/docs/tutorials/04_custom_forward/) 换一个方向：不加旁路计算，而是改变模型本身跑哪些层。

</div>

<div class="zine-doc-footer">
  <a class="zine-pill-link" href="/pllm/#docs">← Back to pLLM Documentation Hub</a>
  <a class="zine-pill-link" href="https://github.com/LinesHogan/pLLM/blob/main/doc/tutorials/03_decode_training.ipynb" target="_blank" rel="noopener">Edit / View on GitHub ↗</a>
</div>
