---
layout: default
permalink: /pllm/
title: pLLM Project
description: "pLLM (Programmable LLM): Declarative programmability layer for stateful and adaptive inference-time algorithms in vLLM — overview, documentation, and engineering blog series."
nav: true
nav_order: 3
---

{% assign sync = site.data.pllm_sync %}
{% assign pllm_posts = site.posts | where: "pllm_synced", true | sort: "pllm_order" %}

<div class="post">

<header class="zine-masthead">
  <div class="zine-kicker" data-i18n-zh="开源系统项目 · Programmable LLM · vLLM 0.27–0.34" data-i18n-en="Open-Source Systems Project &middot; Programmable LLM &middot; vLLM 0.27–0.34">Open-Source Systems Project &middot; Programmable LLM &middot; vLLM 0.27–0.34</div>
  <h1 class="post-title" data-i18n-zh="pLLM：可编程大语言模型推理层" data-i18n-en="pLLM: Programmable LLM">pLLM: Programmable LLM</h1>
  <p class="zine-deck" data-i18n-zh="面向未修改 vLLM 的类 eBPF 声明式可编程层：在连续批处理与静态 CUDA Graph 下运行有状态、自适应的推理期算法、每请求快权重与在线自蒸馏训练。" data-i18n-en="An eBPF-like declarative programmability layer for unmodified vLLM—running stateful, adaptive inference-time algorithms, per-request fast weights, and online decode-time training inside continuous batching and CUDA graphs.">An eBPF-like declarative programmability layer for unmodified vLLM—running stateful, adaptive inference-time algorithms, per-request fast weights, and online decode-time training inside continuous batching and CUDA graphs.</p>
</header>

<div class="zine-dispatch" data-title-zh="pLLM：可编程大语言模型推理层" data-title-en="pLLM: Programmable LLM">
  <div class="zine-dispatch-links">
    <a class="zine-pill-link primary" href="https://github.com/LinesHogan/pLLM" target="_blank" rel="noopener">GitHub &middot; LinesHogan/pLLM ↗</a>
    <a class="zine-pill-link accent" data-i18n-zh="文档中心 ↓" data-i18n-en="Documentation ↓" href="#docs">Documentation ↓</a>
    <a class="zine-pill-link" data-i18n-zh="技术专栏 ({{ pllm_posts.size }}) ↓" data-i18n-en="Blog Series ({{ pllm_posts.size }}) ↓" href="#blog">Blog Series ({{ pllm_posts.size }}) ↓</a>
    <a class="zine-pill-link" data-i18n-zh="ICML 2026 论文 (ESamp) ↗" data-i18n-en="ICML 2026 Paper (ESamp) ↗" href="https://proceedings.mlr.press/v306/zeng26m.html" target="_blank" rel="noopener">ICML 2026 Paper (ESamp) ↗</a>
    {% if sync.commit_sha %}
    <a class="zine-pill-link" href="{{ sync.commit_url }}" target="_blank" rel="noopener" title="Auto-synced from LinesHogan/pLLM">Sync @ {{ sync.commit_sha }} ↗</a>
    {% endif %}
  </div>
  <div class="zine-lang-switch" role="group" aria-label="Language">
    <button type="button" class="zine-lang-btn is-active" data-lang-target="en" aria-pressed="true">EN</button>
    <button type="button" class="zine-lang-btn" data-lang-target="zh" lang="zh-CN" aria-pressed="false">中文</button>
  </div>
</div>

<div class="zine-stat-grid">
  <div class="zine-stat-card">
    <div class="zine-stat-value">96.3%</div>
    <div class="zine-stat-label" data-i18n-zh="原生 vLLM 吞吐占比（7B ESamp 128 并发边训边引）" data-i18n-en="Native vLLM throughput retained (7B ESamp online train + guide @ 128 seqs)">Native vLLM throughput retained (7B ESamp online train + guide @ 128 seqs)</div>
  </div>
  <div class="zine-stat-card">
    <div class="zine-stat-value">111.4 tok/s</div>
    <div class="zine-stat-label" data-i18n-zh="4B In-Place TTT 四并发吞吐（单请求超越官方 HF 循环）" data-i18n-en="4-way concurrent 4B In-Place TTT throughput (beats 1-stream HF at bs=1)">4-way concurrent 4B In-Place TTT throughput (beats 1-stream HF at bs=1)</div>
  </div>
  <div class="zine-stat-card">
    <div class="zine-stat-value">15 Families</div>
    <div class="zine-stat-label" data-i18n-zh="支持 Qwen2/3/3.5、Llama、DeepSeek V2/V3 MLA、Mixtral 等及量化/TP" data-i18n-en="Qwen2/3/3.5, Llama, DeepSeek V2/V3 MLA, Mixtral, Gemma 2/3, GPTQ/AWQ/FP8 &amp; TP">Qwen2/3/3.5, Llama, DeepSeek V2/V3 MLA, Mixtral, Gemma 2/3, GPTQ/AWQ/FP8 &amp; TP</div>
  </div>
  <div class="zine-stat-card">
    <div class="zine-stat-value">0 Forks</div>
    <div class="zine-stat-label" data-i18n-zh="兼容原版 vLLM 0.27–0.34，启动前 CPU 静态核验多租户隔离契约" data-i18n-en="Runs atop unmodified vLLM 0.27–0.34 with CPU pre-flight isolation checks">Runs atop unmodified vLLM 0.27–0.34 with CPU pre-flight isolation checks</div>
  </div>
</div>

<div id="markdown-content">

<div class="zine-lang-panel" data-lang-panel="en" lang="en" markdown="1">

## Why pLLM?

**pLLM** (**Programmable LLM**) decouples inference-time algorithm design from serving-engine internals. Prototyping activation steering, linear probes, hypernetwork LoRA, or decode-time test-time training takes a few dozen lines of PyTorch at `batch_size = 1`, yet running that same logic inside production vLLM runs into packed continuous batching, frozen CUDA graphs (where Python hooks never fire on replay), paged per-request state management, KV preemption rollback, and multi-stream hazards.

Instead of maintaining a private fork of vLLM's scheduler or `GPUModelRunner`, you subclass `pllm.Workload` to declare **which internal representations to read, what per-request or engine-wide state to own, and where outputs or parameter updates take effect**, writing standard PyTorch tensor math. pLLM statically verifies row-permutation invariance and multi-tenant isolation on CPU before touching a GPU, then compiles your math, autograd backward passes, and lock-free MVCC state transitions into vLLM's continuous batching and CUDA graphs.

| Inference-Time Capability                                                                                         | Why Raw PyTorch Hooks or vLLM Forks Break                                            | Declarative Primitive in pLLM                                                                              |
| ----------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------- |
| **Inspect hidden states & per-layer logits** (probes, SAEs, Logit Lens, DoLa)                                     | Requests interleave dynamically; Python hooks never fire inside replayed CUDA graphs | `pllm.hidden(layer)`, `pllm.residual(layer)`, `pllm.layer_logits(...)`, `@pllm.export`, `@pllm.add_logits` |
| **Apply per-request or session overlays** (CAA, ReFT, hypernetwork LoRA)                                          | Manual slot eviction and quantized projection overlays leak VRAM or corrupt state    | `pllm.RequestState` + `pllm.model.ActivationPatch` / `pllm.model.Adapter`                                  |
| **Train fast weights online during decode** (ESamp, In-Place TTT, TLM, TTRL)                                      | Synchronous autograd stalls decoding; concurrent streams tear in-place parameters    | `@pllm.train` / `@pllm.train_in_forward` (CUDA-graph captured & side-stream overlapped)                    |
| **Customize forward schedules, attention & KV layouts** (early exit, looped depth, sink/window KV, lookup tables) | Ad-hoc layer loops and KV reads violate append-only block-table invariants           | `@pllm.custom_forward`, `@pllm.custom_attention`, `@pllm.custom_residual`, `pllm.input_tokens`             |

### Quickstart: Per-Request Decode-Time Distillation (ESamp)

Each concurrent request owns an independent rank-4 MLP trained online via `AdamW` at every decode step to predict deep hidden states from shallow layer-0 states, while sampling suppresses shallowly predictable tokens:

```python
from dataclasses import dataclass
import torch
from torch import nn
import pllm
from pllm import testing


@dataclass(frozen=True)
class ESampLite(pllm.Workload):
    beta: float = 0.5
    distiller = pllm.RequestState(
        init=pllm.FromModule(
            lambda m: nn.Sequential(nn.Linear(m.hidden_size, 4), nn.Linear(4, m.hidden_size))
        ),
        optimizer=pllm.AdamW(1e-3),
    )

    @pllm.train(state="distiller")
    def loss(
        self,
        shallow: torch.Tensor = pllm.hidden(0),
        deep: torch.Tensor = pllm.hidden(-1),
        f: nn.Module = pllm.module("distiller"),
    ) -> torch.Tensor:
        return (f(shallow) - deep).square().mean(-1)

    @pllm.guide_sampling(beta=pllm.cfg("beta"))
    def predict(
        self,
        shallow: torch.Tensor = pllm.hidden(0),
        f: nn.Module = pllm.module("distiller"),
    ) -> torch.Tensor:
        return f(shallow)


cfg = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=128)
print(pllm.explain(ESampLite(), cfg))        # Inspect execution plan & VRAM budgets on CPU
print(testing.check(ESampLite(), cfg))       # Verify multi-tenant contracts on CPU in < 2s
```

### Installation, Serving & Coding Agent Skill

```bash
# Install into an existing vLLM 0.27–0.34 environment (or pip install -e ".[vllm032,test]")
python -m pip install -e . --no-deps

# Launch an OpenAI-compatible SSE server with per-session state routing
pllm serve --model Qwen/Qwen2.5-0.5B-Instruct --workload pllm.workloads:DoLaWorkload --port 8000

# Install packaged authoring & migration skills for Claude Code, Codex, or Cursor
pllm skill --install claude   # or: codex | cursor
```

</div>

<div class="zine-lang-panel is-hidden" data-lang-panel="zh" lang="zh-CN" markdown="1">

## 为什么做 pLLM？

**pLLM**（**Programmable LLM**，可编程大语言模型推理层）将推理期算法表达与推理引擎底层解耦。在单序列脚本里写激活引导、探针监控、超网络动态 LoRA 或解码期在线训练只需几十行 PyTorch；但放进生产级 vLLM 时，连续批处理会把多租户词元拍扁重排，静态 CUDA Graph 重放会跳过 Python hook，每请求分页状态回收、KV 抢占回滚与双流并发同步都必须从头处理。

无需 fork 或修改上游 vLLM 源码，只需继承 `pllm.Workload` 声明**读模型的哪个内部表示、持有什么请求级或实例级状态、结果与梯度更新作用于何处**，并编写普通 PyTorch 张量数学。pLLM 在启动 GPU 前先于 CPU 上静态核验行重排不变性与多租户隔离契约，再把张量程序、反向传播与无锁多版本状态转移编译进连续批处理与 CUDA Graph。

| 推理期能力                                                                        | 直接写 PyTorch Hook 或维护 vLLM 分支的阻碍              | 在 pLLM 中的声明式原语                                                                                     |
| --------------------------------------------------------------------------------- | ------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| **读取隐藏状态或逐层 logits**（探针、SAE、Logit Lens、DoLa）                      | 多请求动态交错排布，CUDA Graph 重放时不执行 Python hook | `pllm.hidden(layer)`、`pllm.residual(layer)`、`pllm.layer_logits(...)`、`@pllm.export`、`@pllm.add_logits` |
| **按请求或会话叠加激活 / 权重**（CAA、ReFT、超网络动态 LoRA）                     | 手工管理槽位分配、回收与量化投影叠加极易泄漏显存或串味  | `pllm.RequestState` + `pllm.model.ActivationPatch` / `pllm.model.Adapter`                                  |
| **解码期在线训练快权重**（ESamp、In-Place TTT、测试时学习、TTRL）                 | 同步反向传播阻塞主解码流，双流并发又会导致原地读写撕裂  | `@pllm.train` / `@pllm.train_in_forward`（捕获进图，在旁路 stream 与前向重叠）                             |
| **改写 forward、注意力读法或 KV 布局**（早退跳层、循环深度、滑窗/Sink、查表记忆） | 手改层循环与块表极易破坏只追加分页 KV 的底层内存不变量  | `@pllm.custom_forward`、`@pllm.custom_attention`、`@pllm.custom_residual`、`pllm.input_tokens`             |

### 快速示例：解码期每请求在线蒸馏（ESamp）

每个并发请求独立持有一份秩为 4 的双层小网络与 `AdamW` 动量，在每步解码时在线拟合浅层与深层隐藏状态，并在采样尾声压低浅层即可猜中的平庸候选词：

```python
from dataclasses import dataclass
import torch
from torch import nn
import pllm
from pllm import testing


@dataclass(frozen=True)
class ESampLite(pllm.Workload):
    beta: float = 0.5
    distiller = pllm.RequestState(
        init=pllm.FromModule(
            lambda m: nn.Sequential(nn.Linear(m.hidden_size, 4), nn.Linear(4, m.hidden_size))
        ),
        optimizer=pllm.AdamW(1e-3),
    )

    @pllm.train(state="distiller")
    def loss(
        self,
        shallow: torch.Tensor = pllm.hidden(0),
        deep: torch.Tensor = pllm.hidden(-1),
        f: nn.Module = pllm.module("distiller"),
    ) -> torch.Tensor:
        return (f(shallow) - deep).square().mean(-1)

    @pllm.guide_sampling(beta=pllm.cfg("beta"))
    def predict(
        self,
        shallow: torch.Tensor = pllm.hidden(0),
        f: nn.Module = pllm.module("distiller"),
    ) -> torch.Tensor:
        return f(shallow)


cfg = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=128)
print(pllm.explain(ESampLite(), cfg))        # 在 CPU 上查看执行计划与显存预算
print(testing.check(ESampLite(), cfg))       # 2 秒内在 CPU 上核验多租户隔离契约
```

### 安装、在线服务与 Coding Agent 集成

```bash
# 装入已有的 vLLM 0.27–0.34 环境（或从零安装 python -m pip install -e ".[vllm032,test]"）
python -m pip install -e . --no-deps

# 启动兼容 OpenAI 协议的流式 SSE 服务（支持 vllm_xargs 会话级状态隔离）
pllm serve --model Qwen/Qwen2.5-0.5B-Instruct --workload pllm.workloads:DoLaWorkload --port 8000

# 为 Claude Code、Codex 或 Cursor 一键安装内置编写规范与旧项目迁移清单
pllm skill --install claude   # 亦可选 codex | cursor
```

</div>

<section id="docs" class="zine-hub-section">
  <div class="zine-section-head">
    <div>
      <h2 data-i18n-zh="pLLM 文档中心（自动同步）" data-i18n-en="pLLM Documentation Hub (Auto-Synced)">pLLM Documentation Hub (Auto-Synced)</h2>
      <p class="zine-section-sub" data-i18n-zh="以下全部文档、API 参考手册与可运行教程直接从 LinesHogan/pLLM 仓库自动同步构建。" data-i18n-en="All guides, agent specifications, API references, and tutorials below are automatically synced and rendered from LinesHogan/pLLM.">All guides, agent specifications, API references, and tutorials below are automatically synced and rendered from LinesHogan/pLLM.</p>
    </div>
    <a class="zine-pill-link primary" data-i18n-zh="打开完整文档总览页 →" data-i18n-en="Open Full Docs Index →" href="{{ '/pllm/docs/' | relative_url }}">Open Full Docs Index →</a>
  </div>

{% for cat in sync.doc_categories %}

  <h3 class="zine-cat-heading" data-i18n-zh="{{ cat.title_zh | escape }}" data-i18n-en="{{ cat.title_en | escape }}">{{ cat.title_en }}</h3>
  <div class="zine-doc-grid">
    {% for item in cat.items %}
    <a class="zine-doc-card" href="{{ item.url | relative_url }}">
      <div class="zine-doc-card-top">
        <span class="zine-lang-badge">{{ item.lang }}{% if item.kind == "notebook" %} &middot; ipynb{% endif %}</span>
        <span class="zine-doc-path">{{ item.rel_path | remove_first: "doc/" }}</span>
      </div>
      <div class="zine-doc-title">{{ item.title }}</div>
      {% if item.summary != "" %}
      <p class="zine-doc-summary">{{ item.summary }}</p>
      {% endif %}
    </a>
    {% endfor %}
  </div>
  {% endfor %}
</section>

<section id="blog" class="zine-hub-section">
  <div class="zine-section-head">
    <div>
      <h2 data-i18n-zh="pLLM 技术随笔与机制拆解（中英双语）" data-i18n-en="pLLM Engineering Essays &amp; Deep Dives (Bilingual)">pLLM Engineering Essays &amp; Deep Dives (Bilingual)</h2>
      <p class="zine-section-sub" data-i18n-zh="十篇面向推理系统社区的平权独立文章，无预设仓库门槛，随 pLLM 开源仓库更新自动同步至本站博客。" data-i18n-en="Ten standalone inference-systems articles assuming zero prior familiarity with pLLM—automatically synced from the pLLM repository to this blog.">Ten standalone inference-systems articles assuming zero prior familiarity with pLLM—automatically synced from the pLLM repository to this blog.</p>
    </div>
    <a class="zine-pill-link" data-i18n-zh="查看全部个人博客 →" data-i18n-en="All Blog Posts →" href="{{ '/blog/' | relative_url }}">All Blog Posts →</a>
  </div>

{% if sync.anatomy_diagram and sync.anatomy_diagram != "" %}

  <pre class="zine-anatomy-box"><code>{{ sync.anatomy_diagram | escape }}</code></pre>

{% endif %}

  <ul class="post-list">
    {% for post in pllm_posts %}
    {% assign title_zh = post.title_zh | default: post.title %}
    {% assign title_en = post.title_en | default: post.title %}
    {% assign desc_zh = post.description_zh | default: post.description %}
    {% assign desc_en = post.description_en | default: post.description %}
    {% assign badge_zh = post.pllm_badge_zh | default: "技术拆解" %}
    {% assign badge_en = post.pllm_badge_en | default: "Deep Dive" %}
    {% assign read_time = post.content | number_of_words | divided_by: 360 | plus: 1 %}
    <li>
      <div class="zine-series-meta">
        <span class="zine-series-number">#{{ post.pllm_order | prepend: "0" | slice: -2, 2 }}</span>
        <span class="zine-lang-badge" data-i18n-zh="{{ badge_zh | escape }}" data-i18n-en="{{ badge_en | escape }}">{{ badge_en }}</span>
        <span class="post-meta">{{ read_time }} min read &middot; EN / 中文</span>
      </div>
      <h3>
        <a class="post-title" data-i18n-zh="{{ title_zh | escape }}" data-i18n-en="{{ title_en | escape }}" href="{{ post.url | relative_url }}">{{ title_en }}</a>
      </h3>
      <p data-i18n-zh="{{ desc_zh | escape }}" data-i18n-en="{{ desc_en | escape }}">{{ desc_en }}</p>
    </li>
    {% endfor %}
  </ul>
</section>

<div id="citation" class="zine-hub-section" markdown="1">

<h2 data-i18n-zh="引用（Citation）" data-i18n-en="Citation">Citation</h2>

<p data-i18n-zh="如果 pLLM 对你的研究或推理系统工作有帮助，欢迎引用我们的 ICML 2026 论文（ESamp）：" data-i18n-en="If you find pLLM useful in your research or inference systems work, please cite our ICML 2026 paper on Exploratory Sampling (ESamp):">If you find pLLM useful in your research or inference systems work, please cite our ICML 2026 paper on Exploratory Sampling (ESamp):</p>

```bibtex
@InProceedings{pmlr-v306-zeng26m,
  title = 	 {Large Language Models Explore by Latent Distilling},
  author =       {Zeng, Yuanhao and Lu, Ao and Li, Lufei and Zhang, Zheng and Li, Yexin and Ren, Kan},
  booktitle = 	 {Proceedings of the 43rd International Conference on Machine Learning},
  pages = 	 {153483--153507},
  year = 	 {2026},
  editor = 	 {Zhang, Tong and Dudik, Miroslav and Jaggi, Martin and Agarwal, Alekh and Li, Sharon and Schuurmans, Dale and Zhu, Jerry and Berkenkamp, Felix and Dong, Hanze and Bietti, Alberto},
  volume = 	 {306},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {06--11 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://raw.githubusercontent.com/mlresearch/v306/main/assets/zeng26m/zeng26m.pdf},
  url = 	 {https://proceedings.mlr.press/v306/zeng26m.html}
}
```

</div>

</div>

</div>

<script defer src="{{ '/assets/js/bilingual-post.js' | relative_url }}"></script>
