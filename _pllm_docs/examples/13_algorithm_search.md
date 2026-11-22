---
layout: page
title: "13 让 agent 搜索算法：一个可复现的打分循环"
description: "pLLM · Algorithm Family Examples (06–13) (算法族示例与实测 (06–13))"
permalink: /pllm/docs/examples/13_algorithm_search/
nav: false
---

<div class="zine-dispatch">
  <div class="zine-dispatch-links">
    <a class="zine-pill-link primary" href="/pllm/#docs">← pLLM Project &amp; Docs</a>
    <a class="zine-pill-link" href="/pllm/docs/">All Docs Index</a>
    <span class="zine-lang-badge">Notebook (.ipynb)</span>
  </div>
  <a class="zine-pill-link" href="https://github.com/LinesHogan/pLLM/blob/main/doc/examples/13_algorithm_search.ipynb" target="_blank" rel="noopener">Open .ipynb on GitHub ↗</a>
</div>

<div id="markdown-content" markdown="1">

<div id="sec-1"></div>

前面几篇教程展示了怎样在 pLLM 里手工编写、检查和评测推理算法。如果你希望让编码 agent 帮自己改写和尝试推理算法，这一节回答一个实际问题：怎样给 agent 搭一个安全、可复现的打分循环？

让 agent 改写推理算法，每个候选都需要便宜、可靠、能复现的反馈：成本报告、正确性检查、质量分数。pLLM 提供前两样（`pllm.explain`、`pllm.testing.check` 与 `check_on_gpu`），仓库里的 `examples/algorithm_search` 把它们和一个评测串成循环。这个例子不往 pLLM 里加搜索 API。

结果先说在前面：这次的分数差别在噪声范围内，后面会说明为什么。这个演示说明的是“循环能跑通、门槛起作用、分数能复现”，不说明 agent 找到了更好的算法。

在这个例子里，循环按下面的规则运行：
- **任务**：在 GSM8K 测试集的前 200 题上提高 `Qwen/Qwen2.5-0.5B-Instruct` 的准确率，同时保持至少原生 vLLM 90% 的吞吐。
- **三个起点（seed）**：原样的模型（一个普通的残差相加，等于不改）、内置的 DoLa、一个用 `active=` 的 early exit。
- **每一轮**：agent（默认使用 `codex` 命令行）在一个看不到仓库、没有 GPU 的沙箱里，读文档快照、任务、起点、当前最好的候选和完整的排行榜（每个候选的分数、`explain` 和 `check` 的文字），只改 `candidate.py`；agent 退出后，打分在新进程里进行。
- **打分（`score.py`）**：CPU 上先 `explain`，再 `pllm.testing.check`（不通过直接淘汰）；然后在 GPU 上跑 `check_on_gpu`、贪心生成（最多 512 token）算准确率（取回答里最后一个数字，和标准答案精确比较），再用 `pllm.benchmark.compare` 按统一设置测吞吐（batch 8、256 token、每个 prompt 16 个样本、预热 1 轮、计时 3 轮），和原生 vLLM 的 CUDA graph 吞吐比。
- **规则**：通过全部契约且吞吐不低于 90% 的候选里准确率最高者胜；平局看 `explain` 的 FLOP 估算。同一指纹（pLLM 按负载源码算出的）只打一次分。
- **记账与重放**：每个候选记一行 `ledger.jsonl`，包含源码、指纹、git 版本、模型和数据集的固定版本、题号、随机种子、`explain` 和 `check` 文字、GPU 型号、每题答案、三轮吞吐。`--replay` 不用 agent，把 ledger 里的源码在新进程里重新打分：答案必须逐题相同，吞吐给出和原记录的比较。agent 的提议不可复现，分数可复现。

这次运行在 RTX 4090、vLLM 0.27.1 环境下完成，共跑 3 轮，含重放在内 notebook 执行用时 2671 秒（约 45 分钟）。

我们先看看 agent 在沙箱里读到的任务说明（`task.md`）。

```python
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path.cwd().resolve().parents[1]          # 仓库根目录（本 notebook 位于 doc/examples/）
EXAMPLE = ROOT / "examples" / "algorithm_search"
print((EXAMPLE / "task.md").read_text())
```

```text
Submit one small, concrete candidate.py edit for this round, then finish. The
external scorer evaluates it after you exit and returns feedback to the next round.
This sandbox contains only docs and seed files: torch/pllm packages, repository
source and GPUs are unavailable. Use the supplied API docs; do not search for
packages, install dependencies or run inference here.

Improve Qwen/Qwen2.5-0.5B-Instruct accuracy on the pinned first 200 GSM8K test
questions while retaining at least 90% of vanilla throughput. Edit candidate.py
and define WORKLOAD and NEUTRAL as pllm.Workload instances. NEUTRAL must disable
every effect. Use only the documented pLLM API and torch math; do not read files,
fetch data, alter scoring, or depend on other workspace files from candidate.py.

Seeds: candidate.py is the current best; dola.py uses the DoLa built-in;
early_exit.py uses an active= loop over the last two layers. You may replace the
algorithm or tune it. The leaderboard contains previous accuracy, throughput,
explain and check feedback. A failed contract or throughput below 90% disqualifies
a candidate. Highest accuracy wins; ties go to lower explain FLOPs. Vanilla is a
valid candidate. Each proposal is scored once per source.

Quality uses greedy generation, up to 512 tokens, a fixed zero-shot chat prompt,
and exact numeric match of the last number in the response. Throughput uses the
aligned batch=8, tokens=256, sampling_n=16, warmup=1, rounds=3 settings. Model
graphs are enabled when supported; early exits require eager model execution.
The throughput floor always uses the vanilla graph reference. Submit only
candidate.py; execution and scoring happen outside this sandbox.
```

这一步我们看任务说明的要点，并检查第三个起点 `early_exit.py` 的代码是怎么写的。

从上面的 `task.md` 可以看到给 agent 定下的边界：
- 在固定的前 200 道 GSM8K 测试题上评测，吞吐不低于原生 vLLM 的 90%。
- 只改 `candidate.py`，必须定义 `WORKLOAD` 和 `NEUTRAL` 两个 `pllm.Workload` 实例，其中 `NEUTRAL` 关闭全部效果，用于 `neutral` 检查。
- 只能用文档里的 pLLM API 和 `torch` 数学运算，不能读文件、拉取数据、修改打分逻辑或依赖工作区里的其他文件。

三个起点里，第一个是原样模型，第二个是内置的 DoLa，第三个是下面要打印的 `early_exit.py`：在最后两层（第 22、23 层），每进一层前先计算隐藏状态元素平方的均值，小于等于阈值的行就跳过这一层（用的就是示例 12 讲过的 `active=` 写法）。

```python
print((EXAMPLE / "early_exit.py").read_text())
```

```text
"""Bounded early-exit loop seed for algorithm search on Qwen2.5-0.5B."""

from dataclasses import dataclass

import torch

import pllm


@dataclass(frozen=True)
class EarlyExit(pllm.Workload):
    """Skip the last two layers for rows below the hidden-energy threshold."""

    threshold: float = 0.0

    @pllm.custom_forward(tokens="decode")
    def forward(self, layers: pllm.model.Layers) -> torch.Tensor:
        """Visit two tail layers while the row's active mask remains true."""
        h = layers(stop=22)[21]
        active = torch.ones_like(h[:, :1], dtype=torch.bool)
        for layer in range(22, 24):
            active = active & (h.float().square().mean(-1, keepdim=True) > self.threshold)
            h = layers(layer, layer + 1, hidden=h, active=active)[layer]
        return h


WORKLOAD = EarlyExit()
NEUTRAL = EarlyExit(threshold=-1.0)
```

这一步我们先看清 `early_exit.py` 里的默认参数，然后启动 3 轮搜索。

注意 `EarlyExit` 的默认参数是 `threshold: float = 0.0`。因为隐藏状态元素平方的均值永远大于 0，所以 `threshold = 0.0` 这个默认设置实际上从不跳层（这一点后面分析结果时会用到）。而在 `NEUTRAL = EarlyExit(threshold=-1.0)` 里，阈值设为 `-1.0`，同样保留全部行通过，用来满足 `neutral` 检查。

下面运行 `search.py --rounds 3`。脚本要求输出目录是新的；它会先在独立进程里给三个起点逐一打分，再跑 3 轮 agent。每轮 agent 结束并打完分后，会打印一行 JSON。这一格在 GPU 上要跑几十分钟。

```python
OUTPUT = ROOT / "outputs" / "tutorial13-search"
shutil.rmtree(OUTPUT, ignore_errors=True)  # search.py 要求输出目录是新的
run = subprocess.run([sys.executable, str(EXAMPLE / "search.py"), "--rounds", "3", "--output", str(OUTPUT)],
                     cwd=ROOT, capture_output=True, text=True)
print("退出码:", run.returncode)
print(run.stdout)
```

```text
退出码: 0
{"fingerprint": "54bb2b3ece37d060e870f3aa15e8acadfc3659b991fbc08a4247044661018797", "category": null, "eligible": false, "accuracy": 0.465, "throughput_ratio": 0.895217258023644, "duplicate_of": null}
{"fingerprint": "db6a716ff4c77012a285c16e8ec9dcdfec093c6b2b7ad1ef610ae05301ca1c5c", "category": null, "eligible": true, "accuracy": 0.465, "throughput_ratio": 0.9033448027755293, "duplicate_of": null}
{"fingerprint": "18648cbd26a770accd7cceb5364b9805e1013245d7b47298d80a57a0729191cc", "category": null, "eligible": true, "accuracy": 0.465, "throughput_ratio": 0.9008080168905768, "duplicate_of": null}
```

这一步我们看上面 3 轮 agent 搜索的返回结果，并把账本（`ledger.jsonl`）里的全部记录列成排行榜。

从上面的输出可以看到，退出码为 0，三轮 agent 都产出了能打分的候选，且都没有和已有候选撞指纹（`duplicate_of` 为 `null`）：
- 第 1 轮候选的准确率为 0.465，但吞吐比是 `0.895217258023644`（约 0.895），没过 90% 的门槛（`eligible: false`）。
- 第 2 轮和第 3 轮候选都通过了全部门槛（`eligible: true`），吞吐比分别是 `0.9033448027755293`（约 0.903）和 `0.9008080168905768`（约 0.901），准确率都是 0.465。

下一格读取 `ledger.jsonl`，把三个起点（`seed`）和三个 agent 候选（`codex`）放在同一张表里对比。

```python
ledger = [json.loads(line) for line in (OUTPUT / "ledger.jsonl").read_text().splitlines()]


def title(record: dict) -> str:
    source = record.get("source", "")
    return source.split('"""')[1].strip().splitlines()[0] if '"""' in source else "(没有提交代码)"


print(f"{'来源':8s} {'通过全部门槛':8s} {'准确率':>8s} {'吞吐/原生':>9s}  说明")
for record in ledger:
    accuracy = record.get("accuracy")
    ratio = record.get("throughput_ratio")
    print(f"{record['agent']:8s} {str(record['eligible']):8s} "
          f"{'-' if accuracy is None else f'{accuracy:.3f}':>8s} {'-' if ratio is None else f'{ratio:.3f}':>9s}  "
          f"{title(record)}")
```

```text
来源       通过全部门槛        准确率     吞吐/原生  说明
seed     True        0.445     0.977  Vanilla seed edited by the algorithm-search author; scored in a separate process.
seed     False       0.300     0.152  DoLa seed for algorithm search, using the built-in layer contrast.
seed     False       0.465     0.894  Bounded early-exit loop seed for algorithm search on Qwen2.5-0.5B.
codex    False       0.465     0.895  Fused early-exit tail for Qwen2.5-0.5B, replacing per-layer loops with a single slice.
codex    True        0.465     0.903  Fused early-exit tail for Qwen2.5-0.5B with tuned energy threshold.
codex    True        0.465     0.901  Fused early-exit tail for Qwen2.5-0.5B with tuned energy threshold.
```

这一步我们解读上面的排行榜，并看看下一轮 agent 看到的文字反馈长什么样。

先看排行榜里的六条记录：
- **原样模型（Vanilla seed）**：通过全部门槛，准确率 0.445，吞吐是原生 vLLM 的 0.977。
- **DoLa 起点（DoLa seed）**：准确率 0.300，吞吐只有原生的 0.152，因为吞吐低于 90% 被门槛淘汰（示例 06 讲过 DoLa 的吞吐代价）。
- **early exit 起点（Bounded early-exit loop seed）**：准确率 0.465，但吞吐是 0.894，差一点没过 90% 的门槛，也被淘汰。
- **agent 的三个候选（`codex`）**：从文档字符串说明可以看出，agent 做的改动都是“把最后两层的单层循环合并成一次切片调用”（`Fused early-exit tail`）；第 1 轮吞吐 0.895 仍差一点，第 2、3 轮还调了能量阈值（分别为 1.0 和 1.4），吞吐达到 0.903 和 0.901，跨过了 90% 门槛。

除了准确率和吞吐数字，排行榜还会把每个候选的 `explain` 和 `check` 输出传给下一轮 agent。下一格打印第一条记录里这两段文字的开头。

```python
first = ledger[0]
print("explain（节选）:")
print("\n".join(first["explain"].splitlines()[:6]))
print()
print("check（节选）:")
print("\n".join(str(first["check"]).splitlines()[:8]))
```

```text
explain（节选）:
model    Qwen2ForCausalLM, hidden 896, 24 layers, vocab 151936, bfloat16 [exact, config.json]
parallel tensor_parallel_size=1; reservations are per rank
forward  CUDA graphs: no effect needs eager
base     987922432 FLOP/token [estimate: dense projections and LM head; excludes attention over KV history]
member  effect  anchor / lane / graph  state reads (origin, distance)
residual  custom_residual  model site or inactive method

check（节选）:
status  contract / reason
PASS    tensor_parallel_size=1: admission
PASS    compile
PASS    neutral
SKIP    model-site execution: CPU decoder does not execute residual hooks, overlays, spans or PositionState
```

这一步我们先看 `explain` 和 `check` 给 agent 提供了什么信息，再去检验 0.465 对 0.445 的准确率差别是不是真的提升。

在上面的输出里，`explain` 告诉 agent：模型配置是 24 层的 `Qwen2ForCausalLM`，前向仍在 CUDA graphs 里执行（`no effect needs eager`），基础开销约 `987922432 FLOP/token`，当前负载是一个 `custom_residual`；`check` 则列出了 `admission`、`compile`、`neutral` 等契约的通过情况。对在没有 GPU 的沙箱里工作的 agent 来说，这些文字就是不用跑 GPU 就能知道的代价和问题。

现在看核心问题：最佳候选的准确率 0.465 相比原样模型的 0.445，是不是真的提升？下一格拉取标准答案，逐题比较最佳候选和原样模型的回答，并做统计检验。

```python
import re
import urllib.request
from decimal import Decimal


def number(text: str) -> str:
    """与 score.py 相同的规则：取最后一个数字并规范化。"""
    numbers = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    return str(Decimal(numbers[-1].replace(",", "")).normalize()) if numbers else ""


url = ("https://raw.githubusercontent.com/openai/grade-school-math/" + ledger[0]["dataset_revision"]
       + "/grade_school_math/data/test.jsonl")
gold = [number(json.loads(line)["answer"]) for line in urllib.request.urlopen(url, timeout=60).read().splitlines()[:200]]

vanilla = ledger[0]
best = max((r for r in ledger if r["eligible"]), key=lambda r: (r["accuracy"], -r["flops"]))
right_v = [a == g for a, g in zip(vanilla["answers"], gold)]
right_b = [a == g for a, g in zip(best["answers"], gold)]
gained = sum(b and not v for v, b in zip(right_v, right_b))
lost = sum(v and not b for v, b in zip(right_v, right_b))
n, p = len(gold), vanilla["accuracy"]
print(f"最佳候选: {title(best)}")
print(f"准确率 {best['accuracy']:.3f}，原生 {p:.3f}：新答对 {gained} 题，新答错 {lost} 题，净 {gained - lost:+d} 题（共 {n} 题）")
print(f"{n} 道题上，准确率 {p:.3f} 的标准误约为 {math.sqrt(p * (1 - p) / n):.3f}")
k, m = min(gained, lost), gained + lost
p_value = min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2**m) if m else 1.0
print(f"配对符号检验（只看结果不同的 {m} 题）的双侧 p 值: {p_value:.2f}")
```

```text
最佳候选: Fused early-exit tail for Qwen2.5-0.5B with tuned energy threshold.
准确率 0.465，原生 0.445：新答对 12 题，新答错 8 题，净 +4 题（共 200 题）
200 道题上，准确率 0.445 的标准误约为 0.035
配对符号检验（只看结果不同的 20 题）的双侧 p 值: 0.50
```

这一步我们如实拆解上面的统计结果和账本证据，然后用 `--replay` 验证分数能不能在新进程里复现。

先看为什么说这次的“提升”在噪声范围内：
1. **统计检验**：最佳候选准确率 0.465 对原样模型 0.445，在 200 道题里新答对 12 题、新答错 8 题，净 +4 题。准确率 0.445 在 200 道题上的标准误约为 0.035；只看结果不同的 20 道题做配对符号检验，双侧 p 值为 0.50。
2. **更直接的账本证据**：回头看 `early_exit.py` 起点，它的阈值是 `0.0`，而隐藏状态元素平方的均值永远大于 0，所以它从不跳层；但它在排行榜上的准确率已经是 0.465，只是因为吞吐 0.894 没过 90% 的门槛而被淘汰。对照 `ledger.jsonl` 里三个 agent 候选的 `answers` 字段可以发现，它们（包括把阈值调到 1.0 和 1.4 的第 2、3 轮）在 200 道题上的答案彼此完全相同；early exit 起点的答案和它们略有不同，准确率同为 0.465。agent 实际做到的是把最后两层的两个单层调用合并成一次调用（`Fused early-exit tail`），把吞吐比从 0.894 提到 0.903 和 0.901，从而跨过了 90% 的门槛。0.465 和原样模型 0.445 的准确率差别，来自同一个模型换了一条执行路径（`custom_forward` 分段执行对比 vLLM 的整模型 CUDA graph）带来的数值差异，不来自跳层。

所以，这个演示说明的是“循环能跑通、门槛起作用、分数能复现”，不说明 agent 找到了更好的算法。

最后我们验证“分数能复现”。运行 `search.py --replay` 会读取 `ledger.jsonl` 里的源码，不用 agent，在新进程里把每条记录重新打一遍分；答案、门槛或指纹只要有一项和原记录不同，脚本就会以非零退出码报错退出。

```python
REPLAY = ROOT / "outputs" / "tutorial13-replay"
shutil.rmtree(REPLAY, ignore_errors=True)
run = subprocess.run([sys.executable, str(EXAMPLE / "search.py"), "--replay", str(OUTPUT / "ledger.jsonl"),
                      "--output", str(REPLAY)], cwd=ROOT, capture_output=True, text=True)
print("退出码:", run.returncode)  # 非零表示有候选的答案、门槛或指纹与原记录不同
for line in run.stdout.splitlines():
    if line.startswith("{"):
        row = json.loads(line)
        print(f"准确率 {row['accuracy']:.3f}（与原记录相同）；吞吐 {row['throughput']:.0f} token/s，"
              f"原记录 {row['reference_throughput']:.0f}（{row['throughput_change']:+.1%}）")
```

```text
退出码: 0
准确率 0.445（与原记录相同）；吞吐 38162 token/s，原记录 38243（-0.2%）
准确率 0.300（与原记录相同）；吞吐 6072 token/s，原记录 6032（+0.7%）
准确率 0.465（与原记录相同）；吞吐 34767 token/s，原记录 34992（-0.6%）
准确率 0.465（与原记录相同）；吞吐 35369 token/s，原记录 35545（-0.5%）
准确率 0.465（与原记录相同）；吞吐 35618 token/s，原记录 35448（+0.5%）
准确率 0.465（与原记录相同）；吞吐 35666 token/s，原记录 35409（+0.7%）
```

这一步我们看重放的结果，并总结整个实验。

重放的退出码为 0：六条记录在 200 道题上的答案全部逐题复现（准确率分别为 0.445、0.300、0.465、0.465、0.465、0.465，与原记录相同）；六条记录重新测得的吞吐分别为 38162、6072、34767、35369、35618、35666 token/s，和原记录（38243、6032、34992、35545、35448、35409 token/s）相差在 `-0.6%` 到 `+0.7%` 之间。

小结这一节的内容：
- **循环能跑通**：agent 在无 GPU 沙箱里提议、pLLM 做 `explain` 与 `check` 检查、GPU 进程测准确率和吞吐、结果写进 `ledger.jsonl`、最后用 `--replay` 重放。
- **门槛起作用**：DoLa（吞吐比 0.152）和未合并调用的 early exit 起点（吞吐比 0.894）都因为没达到 90% 吞吐门槛而被自动淘汰。
- **分数可复现，agent 的提议不可复现**：每次让 agent 跑搜索，它写出的代码可能不同；但一旦写进 `ledger.jsonl`，用 `--replay` 就能在新进程里复现相同的逐题答案和一致的门槛判断（前提是同样的硬件和 ledger 里记下的固定版本）。
- **这次的准确率差别是噪声**：0.465 对 0.445（净 +4 题，配对符号检验双侧 p 值 0.50）来自切换执行路径的数值差异：从不跳层的 early exit 起点也是 0.465。
要让搜索真正有意义，需要更大的题集或多次重复、能被训练或调参真正改变行为的候选，这些不在这个演示里。

如果你想让自己的编码 agent 编写 pLLM 负载，可以把 [agent skill](/pllm/docs/agent/skill/) 和 [agent 参考](/pllm/docs/agent/reference/) 交给它；仓库里的 `tools/ai_eval` 也用同样的沙箱评测“只看文档的 agent 能不能写对负载”。

回到 [文档首页](/pllm/docs/)。

</div>

<div class="zine-doc-footer">
  <a class="zine-pill-link" href="/pllm/#docs">← Back to pLLM Documentation Hub</a>
  <a class="zine-pill-link" href="https://github.com/LinesHogan/pLLM/blob/main/doc/examples/13_algorithm_search.ipynb" target="_blank" rel="noopener">Edit / View on GitHub ↗</a>
</div>
