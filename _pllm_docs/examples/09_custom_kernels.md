---
layout: page
title: "09 在负载里用自己写的算子"
description: "pLLM · Algorithm Family Examples (06–13) (算法族示例与实测 (06–13))"
permalink: /pllm/docs/examples/09_custom_kernels/
nav: false
---

<div class="zine-dispatch">
  <div class="zine-dispatch-links">
    <a class="zine-pill-link primary" href="/pllm/#docs">← pLLM Project &amp; Docs</a>
    <a class="zine-pill-link" href="/pllm/docs/">All Docs Index</a>
    <span class="zine-lang-badge">Notebook (.ipynb)</span>
  </div>
  <a class="zine-pill-link" href="https://github.com/LinesHogan/pLLM/blob/main/doc/examples/09_custom_kernels.ipynb" target="_blank" rel="noopener">Open .ipynb on GitHub ↗</a>
</div>

<div id="markdown-content" markdown="1">

<div id="sec-1"></div>

这一节要回答的问题是：当你写了一个 Triton kernel，怎样把它放进 pLLM 的负载里运行，并且用 pLLM 的检查工具确认它没有破坏 CUDA graph、也没有和 CPU 参考实现算得不一样？

推理时的新算法越来越多地落到“PyTorch 一行写不出来，或者写出来太慢”的步骤上，比如查表的哈希、稀疏 attention 的页打分、分块扫描。遇到这些步骤时，研究者会自己写一个 Triton kernel。问题是，放进推理引擎以后，这个 kernel 要进 CUDA graph、不能偷偷做 GPU 到 CPU 的同步、还要有 CPU 上能对照的参考实现，否则出了错很难知道是 kernel 错了还是别处错了。

在 pLLM 里不需要学新的算子包装 API：用 PyTorch 自己的 `torch.library.custom_op` 包装 kernel，再在负载方法里像普通函数一样调用它。接入之后，pLLM 会做三件事：
- 在启动前的 `pllm.explain` 里列出这个算子（每行多少 FLOP、多少字节，有没有 CPU 参考）。
- `pllm.testing.check` 在 CPU 上用你的 CPU 参考跑调度契约。
- `pllm.testing.check_on_gpu` 把 CPU 上的输入搬到 GPU 上重放，比较 kernel 和 CPU 参考的输出（设备等价）。

**本篇的例子算法**

示例 06 看到残差流的范数随层数变大，同样一个引导向量在不同层、不同 token 上“分量”不一样。这里我们写一个“相对引导”：在第 12 层 MLP 之后，把残差流 $$h$$ 改成 $$h + \text{strength} \times |h| \times v$$，其中 $$v$$ 是单位方向向量，加的量和这一行自己的范数成正比。这一步要对每一行求范数再加回去，正好写成一个“一行一个 program”的 Triton kernel。

在开始之前，先说明两点事实：
- 这个 kernel 在这里并不比 PyTorch 快：在 7B 上实测吞吐都是原生 vLLM 的 98%（见第八步），两者差别在噪声范围内；第 12 层的一次加法不是瓶颈。写自定义算子的理由是你的算法真的需要一个 PyTorch 表达不了或太慢的步骤；这篇只演示接法和检查。
- 引导方向是用 5 对短语算出来的 CAA 向量，非零强度只说明修改生效，不说明模型变好。

**本篇步骤**

我们按以下 10 个步骤来做实验：
1. 写 Triton kernel 并用 `custom_op` 包装；
2. 先在 GPU 上和 CPU 参考对一下；
3. 准备引导方向；
4. 写负载；
5. 用 `explain` 查看算子信息；
6. 跑 CPU 契约检查；
7. 跑生成看引导效果；
8. 对比吞吐；
9. 跑 GPU 契约检查（设备等价）；
10. 故意写错 CPU 参考，看它被抓住。

先导入这节要用到的库，并准备好 `Qwen/Qwen2.5-0.5B-Instruct` 的 `ModelConfig`：

```python
import os

os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"  # 只显示 vLLM 的报错，省掉启动日志

import math
import tempfile
from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from safetensors.torch import save_file
from torch.utils.flop_counter import register_flop_formula
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import SamplingParams

import pllm
from pllm import testing
from pllm.benchmark import throughput

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
CONFIG = pllm.ModelConfig(MODEL, max_model_len=512, max_num_seqs=4, max_num_batched_tokens=512,
                          gpu_memory_utilization=0.3, seed=0)
```

## 第一步：写 kernel 并用 `custom_op` 包装 {#sec-2}

这一步要解决的问题是：怎样把 Triton kernel 包装成 PyTorch 和 pLLM 都能识别的算子？

下面的 `_relative_steer_kernel` 中，一个 program 处理一行：读入 `x` 和 `y` 相加得到 `h`（转成 `float32`），读入方向向量 `v`，求出当前行的范数 $$|h|$$，最后写回 $$h + \text{strength} \cdot |h| \cdot v$$。

接着我们用 `@torch.library.custom_op("tutorial::relative_steer", mutates_args=(), device_types="cuda")` 把它包装起来。这是 PyTorch 自己的机制，让 `torch.compile` 认识自定义算子用的也是这一套写法。
在它下面还有三个注册函数，各自有不同的用途：
- `register_fake`：只返回输出形状，不做实际计算。pLLM 用它在形状层面规划显存和检查同步，这一项是必需的。
- `register_kernel("cpu")`：提供 CPU 参考实现。CPU 契约检查和 GPU 上的设备等价检查都会用到它。
- `register_flop_formula`：告诉 `explain` 每次调用算多少 FLOP，这里按每个元素 5 次浮点运算来算。

在 pLLM 里使用 custom op 有几条规则：
- kernel 必须返回新张量（`mutates_args=()`），不能原地修改输入；改状态只能通过 pLLM 的效果。
- 必须有 `register_fake`。
- 在负载方法里要直接调用你定义的算子对象（这里的 `relative_steer`），不能写 `torch.ops.tutorial.relative_steer`，因为 pLLM 需要读到它的源码。
- 如果训练要穿过这个算子，需要 `register_autograd`；如果要在逐 token 的 custom forward 单元里使用，需要 `register_vmap`。`register_kernel("cpu")` 和 `register_flop_formula` 是可选的，但没有 CPU 实现的话，CPU 契约检查就跑不了这段数学。

```python
@triton.jit
def _relative_steer_kernel(x_ptr, y_ptr, v_ptr, out_ptr, strength, width, BLOCK: tl.constexpr):
    row = tl.program_id(0)  # 一个 program 处理一行
    cols = tl.arange(0, BLOCK)
    mask = cols < width
    h = tl.load(x_ptr + row * width + cols, mask, other=0.0).to(tl.float32)
    h += tl.load(y_ptr + row * width + cols, mask, other=0.0).to(tl.float32)
    v = tl.load(v_ptr + cols, mask, other=0.0).to(tl.float32)
    norm = tl.sqrt(tl.sum(h * h, axis=0))
    tl.store(out_ptr + row * width + cols, (h + strength * norm * v).to(out_ptr.dtype.element_ty), mask)


@torch.library.custom_op("tutorial::relative_steer", mutates_args=(), device_types="cuda")
def relative_steer(x: torch.Tensor, y: torch.Tensor, v: torch.Tensor, strength: float) -> torch.Tensor:
    """h = x + y，返回 h + strength * |h| * v。"""
    out = torch.empty_like(x)
    _relative_steer_kernel[(x.shape[0],)](x, y, v, out, strength, x.shape[1],
                                          BLOCK=triton.next_power_of_2(x.shape[1]))
    return out


@relative_steer.register_fake
def _(x, y, v, strength):
    return torch.empty_like(x)


@relative_steer.register_kernel("cpu")
def _(x, y, v, strength):
    h = x.float() + y.float()
    return (h + strength * h.norm(dim=-1, keepdim=True) * v.float()).to(x.dtype)


@register_flop_formula(torch.ops.tutorial.relative_steer)
def _(x_shape, *args, **kwargs) -> int:
    return 5 * math.prod(x_shape)
```

## 第二步：先在 GPU 上和 CPU 参考对一下 {#sec-3}

在把算子放进推理引擎之前，我们先确认一件事：在普通的 PyTorch 里，这个 GPU kernel 和它的 CPU 参考算出来的结果一致吗？

这一步与 pLLM 无关。我们生成随机的 8 行、宽 896（`Qwen2.5-0.5B` 的 hidden size）的 `bfloat16` 张量，`strength` 设为 0.3，分别在 GPU 和 CPU 上调用 `relative_steer`：

```python
x = torch.randn(8, 896, device="cuda", dtype=torch.bfloat16)
y = torch.randn_like(x)
v = torch.nn.functional.normalize(torch.randn(896, device="cuda"), dim=0).bfloat16()
on_gpu = relative_steer(x, y, v, 0.3)
on_cpu = relative_steer(x.cpu(), y.cpu(), v.cpu(), 0.3)
print("GPU 与 CPU 参考的最大差:", (on_gpu.cpu().float() - on_cpu.float()).abs().max().item())
```

```text
GPU 与 CPU 参考的最大差: 0.0
```

输出显示 `GPU 与 CPU 参考的最大差: 0.0`，说明在 `bfloat16` 下两边的计算结果一致。

## 第三步：准备引导方向 {#sec-4}

这一步我们要准备第 12 层的引导方向向量，并把它存成文件供负载读取。

做法和教程 00 一样：在第 12 层取 5 对反义短语最后一个 token 的激活之差再求平均。教程 00 从 vLLM 引擎里读激活；这里改在 CPU 上用 Hugging Face 跑，只要几秒钟。这样第五、六步的 `explain` 和 CPU 契约检查连同它们要读的方向文件，在没有 GPU 的机器上也能跑通。算出来之后除以它的范数，只保留单位方向向量 `direction`，把引导的强度交给残差流自己的范数 $$|h|$$。

最后我们把方向存成 safetensors 文件，下一步在负载里用 `init=pllm.Pretrained(...)` 从文件读进一份 `InstanceState`。为什么用文件而不是教程 00 的 `StateImport`？因为契约检查会用这份真实的方向去跑；如果方向是调用 `generate` 时才从外部送进来的，在检查脚本里它就是零，kernel 里的错误就看不出来了（第十步会用到这一点）。用 `pllm.Pretrained` 文件，检查时用的就是这份真实的方向。

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL)
LAYER = 12
pairs = [("I love", "I hate"), ("This is wonderful", "This is terrible"), ("I feel happy", "I feel sad"),
         ("What a great day", "What an awful day"), ("Everything is amazing", "Everything is horrible")]
hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)  # 在 CPU 上跑，几秒钟


@torch.no_grad()
def last_token_activation(text: str) -> torch.Tensor:
    ids = tokenizer(text, return_tensors="pt")
    return hf(**ids, output_hidden_states=True).hidden_states[LAYER + 1][0, -1].float()


vector = torch.stack([last_token_activation(a) - last_token_activation(b) for a, b in pairs]).mean(0)
direction = vector / vector.norm()  # 单位向量；强度由残差流自己的范数决定
print("CAA 向量范数", round(vector.norm().item(), 2),
      "；第 12 层输出的残差流范数（正面短语的最后一个 token，平均）",
      round(torch.stack([last_token_activation(a).norm() for a, _ in pairs]).mean().item(), 2))
DIRECTION = os.path.join(tempfile.gettempdir(), "relative_steer_direction.safetensors")
save_file({"direction": direction.bfloat16()}, DIRECTION)
del hf
```

```text
CAA 向量范数 7.08 ；第 12 层输出的残差流范数（正面短语的最后一个 token，平均） 18.18
```

从输出可以看到，算出来的 CAA 向量范数是 7.08；第 12 层输出的残差流范数（正面短语的最后一个 token，平均）是 18.18。所以当 `strength` 设为 0.3 时，大约相当于加上一个范数约 5.5 的向量（由 $$0.3 \times 18.18$$ 估算）。

## 第四步：写负载 {#sec-5}

这一步要回答的问题是：写好的 `relative_steer` 算子怎么放进 `pllm.Workload` 里？

我们在示例 07 介绍过 `@pllm.custom_residual(site="mlp", layers=(12,))`：这个方法接管第 12 层 MLP 之后的残差更新，参数 `x` 是进入 MLP 前的残差流，`y` 是 MLP 的输出，方法返回新的残差流。这里 `add` 的方法体就是一次 `relative_steer(x, y, v, self.strength)` 调用，和调用普通函数一样。

`strength` 是 frozen dataclass 的字段，`strength=0.0` 就是“不引导”：

```python
@dataclass(frozen=True)
class RelativeSteering(pllm.Workload):
    strength: float = 0.0
    path: str = DIRECTION

    @property
    def steer(self) -> pllm.InstanceState:
        return pllm.InstanceState({"direction": lambda m: (m.hidden_size,)}, init=pllm.Pretrained(self.path))

    @pllm.custom_residual(site="mlp", layers=(12,))
    def add(self, x: torch.Tensor, y: torch.Tensor,
            v: torch.Tensor = pllm.parameter("steer", "direction")) -> torch.Tensor:
        return relative_steer(x, y, v, self.strength)
```

## 第五步：启动前先看 `explain` {#sec-6}

在真正启动引擎和加载模型权重之前，我们先在 CPU 上跑一下 `pllm.explain`，看看 pLLM 是怎么分析这个自定义算子的。`explain` 列出的行很多，这里我们只打印和算子有关的几行：

```python
report = pllm.explain(RelativeSteering(0.3), CONFIG)
for line in str(report).splitlines():
    if line.startswith(("forward", "member", "add ", "op ")):
        print(line)
```

```text
forward  CUDA graphs: no effect needs eager
member  effect  anchor / lane / graph  state reads (origin, distance)
add  custom_residual  model site or inactive method
op       tutorial::relative_steer: add; 4480 [declared] FLOP/row; 5824 logical bytes/row [lower bound]; CPU reference=True, autograd=False; host syncs unverified (fake only)
```

我们来逐行读一下 `explain` 的输出：
- `forward  CUDA graphs: no effect needs eager`：说明模型照常在 CUDA graph 里跑，我们的 kernel 会被一起捕获进去。
- `add  custom_residual  model site or inactive method`：`add` 是一个挂在模型位置的 `custom_residual` 方法。
- `op` 行列出了 `tutorial::relative_steer` 的详细信息：
  - 调用位置在 `add`；
  - `4480 [declared] FLOP/row`：每行 4480 FLOP，标着 `[declared]`，来自我们注册的公式（$$896 \times 5 = 4480$$）；
  - `5824 logical bytes/row [lower bound]`：每行至少 5824 字节的逻辑读写；
  - `CPU reference=True, autograd=False`：有 CPU 参考实现，没有注册反向传播公式（这里不训练，不需要）；
  - `host syncs unverified (fake only)`：在 CPU 上运行 `explain` 时只看得到 fake 实现，看不到 kernel 里面有没有 GPU 到 CPU 的同步；这项检查要到第九步在 GPU 上实测。

## 第六步：CPU 契约检查 {#sec-7}

接下来我们在 CPU 上运行 `testing.check`。它会用几种固定调度跑三个请求并比较结果；在 CPU 上它跑的是我们注册的 CPU 参考实现。`neutral=` 参数传入 `strength=0.0` 的 `RelativeSteering(0.0)`，用来检查强度为 0 时是否等价于不修改：

```python
print(testing.check(RelativeSteering(0.3), CONFIG, neutral=RelativeSteering(0.0)))
```

```text
status  contract / reason
PASS    tensor_parallel_size=1: admission
PASS    compile
PASS    custom-residual method math / CPU schedules
PASS    neutral
SKIP    model-site execution: CPU decoder does not execute residual hooks, overlays, spans or PositionState
```

检查结果里，`admission`、`compile`、`custom-residual method math / CPU schedules`（方法数学在各种 CPU 调度下一致）和 `neutral`（`strength=0.0` 时不改变采样）都是 `PASS`。最后一行是 `SKIP`，原因写得很清楚：CPU 上的小解码器不执行残差钩子（`residual hooks`），真实模型上的执行要到 GPU 上去检查。

## 第七步：生成 {#sec-8}

现在我们启动真正的引擎，用两个问题、贪心解码 40 个 token，分别测试强度 `0.0`、`+0.3` 和 `-0.3` 下的生成结果，同时打印每次的 `eager_reason`：

```python
questions = ["Write one sentence about my new job.", "Describe your weekend in one sentence."]
prompts = [tokenizer.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                         add_generation_prompt=True) for q in questions]
greedy = SamplingParams(max_tokens=40, temperature=0.0)


def answers(llm) -> list[str]:
    return [output.outputs[0].text for output in llm.generate(prompts, greedy, use_tqdm=False)]


for strength in (0.0, 0.3, -0.3):
    with pllm.make_llm(CONFIG, workload=RelativeSteering(strength)) as llm:
        texts = answers(llm)
        print(f"强度 {strength:+.1f}，eager 原因 {llm.diagnostics().eager_reason!r}")
        for text in texts:
            print("   ", repr(text))
```

```text
强度 +0.0，eager 原因 ''
    'As an AI language model, I am here to assist you with your queries and provide you with information and insights on various topics. I am always here to help you with your new job.'
    "As an AI language model, I don't have a physical presence, so I don't have a weekend. However, I'm always here to assist you with any questions or tasks you may have!"
强度 +0.3，eager 原因 ''
    'In just a few years, I am excited to bring together the diverse and passionate team of creators and partners to create and share incredible experiences that inspire and inspire. Together, we are creating a world where'
    'My weekend is filled with a wide range of activities that bring me joy and excitement. From exploring the world of science and technology, to exploring the world of art and design, to exploring the world of'
强度 -0.3，eager 原因 ''
    "I'm sorry, but I don't know what kind of job you're trying to describe. I'm a language model and don't have access to your personal information or the specifics of your job."
    "I'm unable to describe my weekend in a one-sentence manner as I am a language model and do not have personal experiences or a subjective description of my behavior. My primary function is to provide informative"
```

从输出可以看到两件事：
1. 三次运行的 `eager 原因` 都是空字符串 `''`，说明模型和我们的 Triton kernel 都在 CUDA graph 里运行。
2. 强度 `+0.3` 时，两个回答都变得热情（`"I am excited to bring together..."`、`"My weekend is filled with a wide range of activities that bring me joy and excitement..."`）；强度 `-0.3` 时则变成推辞（`"I'm sorry, but I don't know what kind of job..."`、`"I'm unable to describe my weekend in a one-sentence manner..."`）。再次提醒，这里的引导方向只是 5 对短语算出的 CAA 向量，非零强度下的变化只说明修改生效，不说明模型变好。

另外，强度 `+0.0` 的输出读起来接近原始回答，但不一定与默认配置的原生 vLLM 逐字相同：因为 `custom_residual` 会让 vLLM 在 graph 里捕获模型原本的算子（不做 inductor 融合），浮点舍入和默认的原生 vLLM 略有不同。逐位对照交给第九步的 `check_on_gpu`，它会用同样的 graph 设置起一个原生 vLLM 来做比较。

## 第八步：吞吐 {#sec-9}

这一步我们要回答的问题是：把这个步骤写成 Triton kernel，在端到端推理里有没有比直接写 PyTorch 算子更快？

前面几步用 0.5B 就够了；吞吐换到 `Qwen/Qwen2.5-7B-Instruct` 上测：0.5B 一步 decode 只有几毫秒，任何额外开销的占比都被放大，7B 才是常见的部署规模（bf16 权重约 15 GiB，需要一张 24 GiB 的卡）。测速只看计算量，所以 7B 的引导方向用一个随机单位向量（宽度是 7B 的 hidden size 3584），强度仍是 0.3。我们用 128 个并发请求（8 个 prompt × 每个 16 个样本 × 256 token），用 `pllm.benchmark.throughput` 测：预热一轮，再计时三轮取中位数，每轮计整次 `generate` 的时间（含 prefill）。一共对比三组：原生 vLLM、用 Triton 算子的 `RelativeSteering(0.3)`、以及把同样的数学用普通 PyTorch 算子写的 `RelativeSteeringTorch(0.3)`：

```python
@dataclass(frozen=True)
class RelativeSteeringTorch(RelativeSteering):
    """同样的数学，用普通 PyTorch 算子写，作为对照。"""

    @pllm.custom_residual(site="mlp", layers=(12,))
    def add(self, x: torch.Tensor, y: torch.Tensor,
            v: torch.Tensor = pllm.parameter("steer", "direction")) -> torch.Tensor:
        h = x.float() + y.float()
        return (h + self.strength * h.norm(dim=-1, keepdim=True) * v.float()).to(x.dtype)


# 7B 上改写后的模型图约需 84 MiB 捕获池，超过默认的 64 MiB（不够时 pLLM 报错并给出需要的字节数）。
BIG = pllm.ModelConfig("Qwen/Qwen2.5-7B-Instruct", max_model_len=512, max_num_seqs=128, max_num_batched_tokens=512,
                       gpu_memory_utilization=0.85, seed=0,
                       execution=pllm.ExecutionOptions(cuda_graph_pool_budget_bytes=128 << 20))
BIG_DIRECTION = os.path.join(tempfile.gettempdir(), "relative_steer_direction_7b.safetensors")
save_file({"direction": torch.nn.functional.normalize(torch.randn(3584), dim=0).bfloat16()}, BIG_DIRECTION)
many = [f"Explain the mathematical idea numbered {i}:" for i in range(8)]
sampling = SamplingParams(n=16, max_tokens=256, temperature=0.8, top_p=0.95, ignore_eos=True, seed=7)

rates = {}
with pllm.make_llm(BIG) as plain:
    rates["原生 vLLM"] = throughput(plain, many, sampling)
for name, workload in (("Triton 算子", RelativeSteering(0.3, BIG_DIRECTION)),
                       ("PyTorch 算子", RelativeSteeringTorch(0.3, BIG_DIRECTION))):
    with pllm.make_llm(BIG, workload=workload) as llm:
        rates[name] = throughput(llm, many, sampling)
for name, rate in rates.items():
    print(f"{name:10s}: {rate:6.0f} token/s，为原生 vLLM 的 {rate / rates['原生 vLLM']:.0%}")
```

```text
原生 vLLM   :   5872 token/s，为原生 vLLM 的 100%
Triton 算子 :   5760 token/s，为原生 vLLM 的 98%
PyTorch 算子:   5755 token/s，为原生 vLLM 的 98%
```

这次运行（RTX 4090、vLLM 0.32.0、`Qwen2.5-7B-Instruct` bf16、CUDA graph；128 个并发请求 = 8 × 16，各 256 token；预热 1 轮后 3 轮取中位数）的吞吐如下：
- 原生 vLLM：5872 token/s（100%）
- Triton 算子：5760 token/s（为原生 vLLM 的 98%）
- PyTorch 算子：5755 token/s（为原生 vLLM 的 98%）

Triton 版和 PyTorch 版几乎一样，差别在测量噪声范围内：这个 kernel 在这里没有带来加速，因为第 12 层的一次加法并不是整网推理的瓶颈。就像开头说的，写自定义算子的理由是你的算法真的需要一个 PyTorch 表达不了或太慢的步骤。

## 第九步：GPU 契约检查 {#sec-10}

前面第五步的 `explain` 留了一个 `host syncs unverified (fake only)`，第六步的 CPU 检查也跳过了真实模型位置的执行。现在我们用 `testing.check_on_gpu` 在 GPU 上把完整的契约检查跑一遍。

`check_on_gpu` 会加载真实引擎，检查以下内容：
1. `device equivalence`（设备等价）：把 CPU 契约检查里每个方法的输入搬到 GPU 上重放，用 Triton kernel 算一遍，和 CPU 参考的结果做比较（浮点数按对应 dtype 的默认容差比较）。
2. `host syncs`：在连续 8 个稳定 decode 步里检查有没有发生 GPU 到 CPU 的同步。
3. `preempted`：强制触发 KV 驱逐后重算，检查结果能否复现。
4. `neutral`：检查 `strength=0.0` 时生成的 token 是否与同样 graph 设置的原生 vLLM 逐位相同。

因为 `check_on_gpu` 会启动好几个引擎实例，pLLM 建议在单独的进程里运行它；在这篇 notebook 里我们把它放在最后：

```python
print(testing.check_on_gpu(RelativeSteering(0.3), CONFIG, neutral=RelativeSteering(0.0)))
```

```text
status  contract / reason
PASS    device equivalence: CPU check script method outputs replayed on CUDA
PASS    explain/probe
PASS    host syncs: eight steady decode steps
PASS    preempted: fixed per-token inputs, token IDs and terminal request state
PASS    neutral: greedy and seeded sampling bitwise
```

五项检查全部 `PASS`：设备等价通过、连续 8 个 decode 步没有主机同步、驱逐重算一致、`neutral` 逐位相同。

## 第十步：故意写错 CPU 参考，看它被抓住 {#sec-11}

这一步要回答的问题是：如果我们的 GPU kernel 和 CPU 参考实现写得不一致，会发生什么？

当 CPU 参考和 GPU kernel 不一致时，只跑 CPU 的 `testing.check` 只会执行 CPU 参考，看不出问题；这正是 `check_on_gpu` 的设备等价检查要抓的错误。为了演示这一点，我们复用同一个 Triton kernel，再包装一个新算子 `relative_steer_wrong`，但在它的 CPU 参考实现里故意漏乘了 `|h|`：

```python
@torch.library.custom_op("tutorial::relative_steer_wrong", mutates_args=(), device_types="cuda")
def relative_steer_wrong(x: torch.Tensor, y: torch.Tensor, v: torch.Tensor, strength: float) -> torch.Tensor:
    out = torch.empty_like(x)
    _relative_steer_kernel[(x.shape[0],)](x, y, v, out, strength, x.shape[1],
                                          BLOCK=triton.next_power_of_2(x.shape[1]))
    return out


@relative_steer_wrong.register_fake
def _(x, y, v, strength):
    return torch.empty_like(x)


@relative_steer_wrong.register_kernel("cpu")
def _(x, y, v, strength):
    h = x.float() + y.float()
    return (h + strength * v.float()).to(x.dtype)  # 错误：漏乘了 |h|


@dataclass(frozen=True)
class WrongReference(RelativeSteering):
    @pllm.custom_residual(site="mlp", layers=(12,))
    def add(self, x: torch.Tensor, y: torch.Tensor,
            v: torch.Tensor = pllm.parameter("steer", "direction")) -> torch.Tensor:
        return relative_steer_wrong(x, y, v, self.strength)


try:
    print(testing.check_on_gpu(WrongReference(0.3), CONFIG))
except testing.ContractError as error:
    print(error)
```

```text
1 contract failure(s)
[device equivalence] [device equivalence] add: Tensor-likes are not close!

Mismatched elements: 12715 / 13440 (94.6%)
Greatest absolute difference: 2.53125 at index (5, 581) (up to 1e-05 allowed)
Greatest relative difference: 5536.0 at index (8, 688) (up to 0.016 allowed)
```

`check_on_gpu` 抛出了 `pllm.testing.ContractError`，报告 1 个契约失败：`[device equivalence] add: Tensor-likes are not close!`。在 13440 个元素里有 12715 个（94.6%）超出了 `bfloat16` 的容差，最大绝对差达到了 2.53125（允许上限为 `1e-05`），最大相对差达到了 5536.0（允许上限为 `0.016`）。这里之所以能直接抓出漏乘 `|h|` 的差异，正是因为我们在第三步把引导方向通过 `pllm.Pretrained` 放进了状态里，检查脚本拿到的是非零的真实方向。

## 小结与下一步 {#sec-12}

回顾这一节的内容：
- 在 pLLM 里接入自定义 Triton kernel 只需要用 PyTorch 的 `torch.library.custom_op` 包装，并在负载方法里直接调用该算子对象，不需要学 pLLM 专用的算子 API。
- `register_fake` 是必需的（供显存规划和同步检查使用）；`register_kernel("cpu")` 建议提供，没有它 CPU 契约检查跑不了这段数学，也做不了设备等价对照。
- 启动前的 `pllm.explain` 会列出算子的 FLOP、逻辑读写字节数以及由注册函数和 kernel 源码算出的指纹。
- `testing.check_on_gpu` 的设备等价检查能在真实输入重放中抓出 GPU kernel 与 CPU 参考实现的不一致；算子依赖的状态要用 `pllm.Pretrained` 这类在检查时也存在真实值的加载方式，检查才有意义。

下一步：在 [10 自定义 attention](/pllm/docs/examples/10_custom_attention/) 里，我们会用 PyTorch 写一个 Quest 式的页打分，并说明为什么真正提速需要一个像这节这样的自定义 kernel。

</div>

<div class="zine-doc-footer">
  <a class="zine-pill-link" href="/pllm/#docs">← Back to pLLM Documentation Hub</a>
  <a class="zine-pill-link" href="https://github.com/LinesHogan/pLLM/blob/main/doc/examples/09_custom_kernels.ipynb" target="_blank" rel="noopener">Edit / View on GitHub ↗</a>
</div>
