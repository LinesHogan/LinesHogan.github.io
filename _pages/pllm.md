---
layout: default
permalink: /pllm/
title: pLLM Project
description: "pLLM (Programmable LLM): Declarative programmability layer for stateful and adaptive inference-time algorithms in vLLM — visual architecture, bilingual documentation, and engineering blog series."
nav: true
nav_order: 3
---

{% assign sync = site.data.pllm_sync %}
{% assign pllm_posts = site.posts | where: "pllm_synced", true | sort: "pllm_order" %}

<link rel="stylesheet" href="{{ '/assets/css/pllm-hub.css' | relative_url | bust_file_cache }}">

<div class="post">

<header class="zine-masthead">
  <div class="zine-kicker" data-i18n-zh="开源推理系统 · Programmable LLM · Release v0.5.0 · vLLM 0.27–0.34" data-i18n-en="Open-Source Inference System &middot; Programmable LLM &middot; Release v0.5.0 &middot; vLLM 0.27–0.34">Open-Source Inference System &middot; Programmable LLM &middot; Release v0.5.0 &middot; vLLM 0.27–0.34</div>
  <h1 class="post-title" data-i18n-zh="pLLM：可编程大语言模型推理层" data-i18n-en="pLLM: Programmable LLM">pLLM: Programmable LLM</h1>
  <p class="zine-deck" data-i18n-zh="面向未修改 vLLM 的类 eBPF 声明式可编程层——无需拉私有分支，即可将有状态推理期算法、每请求快权重与解码期在线训练直接编译进连续批处理与静态 CUDA Graph。" data-i18n-en="An eBPF-like declarative programmability layer for unmodified vLLM—compiling stateful inference-time algorithms, per-request fast weights, and online decode-time training into continuous batching and CUDA graphs with zero engine forks.">An eBPF-like declarative programmability layer for unmodified vLLM—compiling stateful inference-time algorithms, per-request fast weights, and online decode-time training into continuous batching and CUDA graphs with zero engine forks.</p>
</header>

<div class="zine-dispatch" data-title-zh="pLLM：可编程大语言模型推理层" data-title-en="pLLM: Programmable LLM">
  <div class="zine-dispatch-links">
    <a class="zine-pill-link primary" href="https://github.com/LinesHogan/pLLM" target="_blank" rel="noopener">GitHub &middot; LinesHogan/pLLM ↗</a>
    <a class="zine-pill-link" href="https://github.com/LinesHogan/pLLM/releases/tag/v0.5.0" target="_blank" rel="noopener">Release v0.5.0 ↗</a>
    <a class="zine-pill-link accent" data-i18n-zh="双语文档中心 ↓" data-i18n-en="Documentation ↓" href="#docs">Documentation ↓</a>
    <a class="zine-pill-link" data-i18n-zh="技术专栏 ({{ pllm_posts.size }}) ↓" data-i18n-en="Blog Series ({{ pllm_posts.size }}) ↓" href="#blog">Blog Series ({{ pllm_posts.size }}) ↓</a>
    <a class="zine-pill-link" data-i18n-zh="ICML 2026 论文 (ESamp) ↗" data-i18n-en="ICML 2026 Paper (ESamp) ↗" href="https://proceedings.mlr.press/v306/zeng26m.html" target="_blank" rel="noopener">ICML 2026 Paper (ESamp) ↗</a>
    {% if sync.commit_sha %}
    <a class="zine-pill-link" href="{{ sync.commit_url }}" target="_blank" rel="noopener" title="Auto-synced from LinesHogan/pLLM">Sync @ {{ sync.commit_sha }} ↗</a>
    {% endif %}
  </div>
</div>

<div class="zine-lang-panel" data-lang-panel="en" lang="en">
  <figure class="pllm-hero-figure">
    <img src="{{ '/assets/img/pllm/doc/assets/pllm-architecture-en.svg' | relative_url }}" alt="pLLM Architecture: Declarative Workload, CPU Pre-Flight Verifier, and Unmodified vLLM Dual-Stream Execution">
    <figcaption class="pllm-fig-caption">Architecture Overview &middot; Declare what internal activations to read, what state to own, and where effects apply; pLLM verifies multi-tenant contracts on CPU and compiles into vLLM's main &amp; side CUDA Graph streams.</figcaption>
  </figure>
</div>

<div class="zine-lang-panel is-hidden" data-lang-panel="zh" lang="zh-CN">
  <figure class="pllm-hero-figure">
    <img src="{{ '/assets/img/pllm/doc/assets/pllm-architecture-zh.svg' | relative_url }}" alt="pLLM 架构总览：声明式负载、CPU 静态契约核验与未修改 vLLM 双流执行">
    <figcaption class="pllm-fig-caption">架构总览 · 声明读取位置、状态归属与作用站点；pLLM 在 CPU 上完成多租户隔离契约核验，并编译进原版 vLLM 的主旁双流静态 CUDA Graph。</figcaption>
  </figure>
</div>

<div class="pllm-bento-grid">
  <div class="pllm-bento-card">
    <div class="pllm-card-kicker" data-i18n-zh="实测吞吐对比 · RTX 4090 · CUDA Graphs 开启" data-i18n-en="Measured Throughput &middot; RTX 4090 &middot; CUDA Graphs ON">Measured Throughput &middot; RTX 4090 &middot; CUDA Graphs ON</div>
    <h2 class="pllm-card-heading" data-i18n-zh="边解码边在线训练，保持 96%+ 原生吞吐" data-i18n-en="96%+ Native Throughput with Online Training">96%+ Native Throughput with Online Training</h2>

    <div class="pllm-bench-group">
      <div class="pllm-bench-subtitle" data-i18n-zh="Qwen2.5-7B-Instruct · 128 并发序列 (8×16) · 每请求独立 AdamW 小网络" data-i18n-en="Qwen2.5-7B-Instruct &middot; 128 Concurrent Seqs (8&times;16) &middot; Per-Request AdamW MLP">Qwen2.5-7B-Instruct &middot; 128 Concurrent Seqs (8&times;16) &middot; Per-Request AdamW MLP</div>
      <div class="pllm-bar-row">
        <div class="pllm-bar-meta">
          <span class="pllm-bar-name" data-i18n-zh="原生 vLLM 基线" data-i18n-en="Native vLLM Baseline">Native vLLM Baseline</span>
          <span class="pllm-bar-val">5,894 tok/s (100%)</span>
        </div>
        <div class="pllm-bar-track"><div class="pllm-bar-fill" style="width: 100%"></div></div>
      </div>
      <div class="pllm-bar-row">
        <div class="pllm-bar-meta">
          <span class="pllm-bar-name" data-i18n-zh="pLLM 在线训练 (@pllm.train)" data-i18n-en="pLLM Online Training (@pllm.train)">pLLM Online Training (@pllm.train)</span>
          <span class="pllm-bar-val">5,738 tok/s (97.4%)</span>
        </div>
        <div class="pllm-bar-track"><div class="pllm-bar-fill emerald" style="width: 97.4%"></div></div>
      </div>
      <div class="pllm-bar-row">
        <div class="pllm-bar-meta">
          <span class="pllm-bar-name" data-i18n-zh="pLLM 在线训练 + 采样引导 (ESamp)" data-i18n-en="pLLM Train + Survivor Guidance (ESamp)">pLLM Train + Survivor Guidance (ESamp)</span>
          <span class="pllm-bar-val">5,678 tok/s (96.3%)</span>
        </div>
        <div class="pllm-bar-track"><div class="pllm-bar-fill red" style="width: 96.3%"></div></div>
      </div>
    </div>

    <div class="pllm-bench-group">
      <div class="pllm-bench-subtitle" data-i18n-zh="4B In-Place TTT 社区检查点 · 4096 Prompt · 快权重在线更新" data-i18n-en="4B In-Place TTT Checkpoint &middot; 4096 Prompt &middot; Chunked Fast Weights">4B In-Place TTT Checkpoint &middot; 4096 Prompt &middot; Chunked Fast Weights</div>
      <div class="pllm-bar-row">
        <div class="pllm-bar-meta">
          <span class="pllm-bar-name" data-i18n-zh="官方 Hugging Face 实现 (仅支持单请求)" data-i18n-en="Official Hugging Face Loop (bs=1 only)">Official Hugging Face Loop (bs=1 only)</span>
          <span class="pllm-bar-val">54.8 tok/s</span>
        </div>
        <div class="pllm-bar-track"><div class="pllm-bar-fill" style="width: 49.2%"></div></div>
      </div>
      <div class="pllm-bar-row">
        <div class="pllm-bar-meta">
          <span class="pllm-bar-name" data-i18n-zh="pLLM 单请求 (bs=1)" data-i18n-en="pLLM Single Request (bs=1)">pLLM Single Request (bs=1)</span>
          <span class="pllm-bar-val">59.6 tok/s (1.09&times;)</span>
        </div>
        <div class="pllm-bar-track"><div class="pllm-bar-fill blue" style="width: 53.5%"></div></div>
      </div>
      <div class="pllm-bar-row">
        <div class="pllm-bar-meta">
          <span class="pllm-bar-name" data-i18n-zh="pLLM 4 并发请求 (bs=4)" data-i18n-en="pLLM 4 Concurrent Requests (bs=4)">pLLM 4 Concurrent Requests (bs=4)</span>
          <span class="pllm-bar-val">111.4 tok/s (2.03&times;)</span>
        </div>
        <div class="pllm-bar-track"><div class="pllm-bar-fill mustard" style="width: 100%"></div></div>
      </div>
    </div>

  </div>

  <div class="pllm-bento-card blue">
    <div class="pllm-card-kicker" data-i18n-zh="系统边界与模型生态" data-i18n-en="Runtime Scope &amp; Model Matrix">Runtime Scope &amp; Model Matrix</div>
    <h2 class="pllm-card-heading" data-i18n-zh="零侵入兼容 15 类主流架构" data-i18n-en="Zero-Fork Support Across 15 Families">Zero-Fork Support Across 15 Families</h2>

    <div class="pllm-kpi-grid">
      <div class="pllm-kpi-tile">
        <div class="pllm-kpi-num">0 Forks</div>
        <div class="pllm-kpi-caption" data-i18n-zh="不改上游 vLLM 源码，不注册全局插件" data-i18n-en="Unmodified upstream vLLM, zero global plugins">Unmodified upstream vLLM, zero global plugins</div>
      </div>
      <div class="pllm-kpi-tile">
        <div class="pllm-kpi-num">0.27–0.34</div>
        <div class="pllm-kpi-caption" data-i18n-zh="跨 8 个大版本兼容 vLLM 执行引擎" data-i18n-en="Verified across vLLM 0.27 through 0.34">Verified across vLLM 0.27 through 0.34</div>
      </div>
      <div class="pllm-kpi-tile">
        <div class="pllm-kpi-num">&lt; 2s CPU</div>
        <div class="pllm-kpi-caption" data-i18n-zh="无 GPU 静态核验多租户串味与抢占回滚" data-i18n-en="Pre-flight isolation &amp; preemption check on CPU">Pre-flight isolation &amp; preemption check on CPU</div>
      </div>
      <div class="pllm-kpi-tile">
        <div class="pllm-kpi-num">3 Modes</div>
        <div class="pllm-kpi-caption" data-i18n-zh="批处理 make_llm · 异步 · OpenAI SSE 服务" data-i18n-en="make_llm &middot; make_async_llm &middot; pllm serve">make_llm &middot; make_async_llm &middot; pllm serve</div>
      </div>
    </div>

    <div class="pllm-chip-cloud">
      <span class="pllm-chip hi">DeepSeek V2 / V3 (Latent MLA)</span>
      <span class="pllm-chip hi">Qwen3 &amp; Qwen3.5 (MoE)</span>
      <span class="pllm-chip">Qwen2 / 2.5 / MoE</span>
      <span class="pllm-chip">Llama 3</span>
      <span class="pllm-chip">Mistral &amp; Mixtral</span>
      <span class="pllm-chip">Gemma 2 / 3</span>
      <span class="pllm-chip">Phi-3</span>
      <span class="pllm-chip">GLM-4</span>
      <span class="pllm-chip">gpt-oss</span>
      <span class="pllm-chip hi">GPTQ / AWQ / FP8</span>
      <span class="pllm-chip hi">Tensor Parallelism</span>
    </div>

  </div>
</div>

<div id="markdown-content">

<div class="pllm-showcase">
  <div class="pllm-tab-bar" role="tablist" aria-label="pLLM Capability Examples">
    <button type="button" class="pllm-tab-btn is-active" data-pllm-tab="esamp" role="tab" aria-selected="true" data-i18n-zh="1 · 解码期在线训练 (ESamp)" data-i18n-en="1 &middot; Decode-Time Training (ESamp)">1 &middot; Decode-Time Training (ESamp)</button>
    <button type="button" class="pllm-tab-btn" data-pllm-tab="dola" role="tab" aria-selected="false" data-i18n-zh="2 · 观测内部与对比解码 (DoLa)" data-i18n-en="2 &middot; Layer Logit Contrast (DoLa)">2 &middot; Layer Logit Contrast (DoLa)</button>
    <button type="button" class="pllm-tab-btn" data-pllm-tab="adapter" role="tab" aria-selected="false" data-i18n-zh="3 · 每请求动态 LoRA / 引导" data-i18n-en="3 &middot; Per-Request LoRA &amp; Steering">3 &middot; Per-Request LoRA &amp; Steering</button>
    <button type="button" class="pllm-tab-btn" data-pllm-tab="serve" role="tab" aria-selected="false" data-i18n-zh="4 · 在线服务与 Agent Skill" data-i18n-en="4 &middot; OpenAI SSE Server &amp; Skill">4 &middot; OpenAI SSE Server &amp; Skill</button>
  </div>

  <div class="pllm-tab-panel" data-pllm-panel="esamp" markdown="1">

<div class="pllm-step-pills">
  <span class="pllm-step-pill" data-i18n-zh="读：pllm.hidden(0), pllm.hidden(-1)" data-i18n-en="Read: pllm.hidden(0), pllm.hidden(-1)">Read: pllm.hidden(0), pllm.hidden(-1)</span>
  <span class="pllm-step-arrow">→</span>
  <span class="pllm-step-pill" data-i18n-zh="存：每请求独立 RequestState + AdamW" data-i18n-en="State: Per-Request MLP + AdamW">State: Per-Request MLP + AdamW</span>
  <span class="pllm-step-arrow">→</span>
  <span class="pllm-step-pill" data-i18n-zh="效：旁路流 @train + 采样尾声 @guide_sampling" data-i18n-en="Effect: Side-Stream @train + @guide_sampling">Effect: Side-Stream @train + @guide_sampling</span>
</div>

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
        init=pllm.FromModule(lambda m: nn.Sequential(nn.Linear(m.hidden_size, 4), nn.Linear(4, m.hidden_size))),
        optimizer=pllm.AdamW(1e-3),
    )

    @pllm.train(state="distiller")
    def loss(self, shallow: torch.Tensor = pllm.hidden(0), deep: torch.Tensor = pllm.hidden(-1),
             f: nn.Module = pllm.module("distiller")) -> torch.Tensor:
        return (f(shallow) - deep).square().mean(-1)

    @pllm.guide_sampling(beta=pllm.cfg("beta"))
    def predict(self, shallow: torch.Tensor = pllm.hidden(0), f: nn.Module = pllm.module("distiller")) -> torch.Tensor:
        return f(shallow)


cfg = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=128)
print(pllm.explain(ESampLite(), cfg))
print(testing.check(ESampLite(), cfg))
```

  </div>

  <div class="pllm-tab-panel is-hidden" data-pllm-panel="dola" markdown="1">

<div class="pllm-step-pills">
  <span class="pllm-step-pill" data-i18n-zh="读：pllm.layer_logits((12, -1))" data-i18n-en="Read: pllm.layer_logits((12, -1))">Read: pllm.layer_logits((12, -1))</span>
  <span class="pllm-step-arrow">→</span>
  <span class="pllm-step-pill" data-i18n-zh="存：无状态（全程驻留静态 CUDA Graph）" data-i18n-en="State: Stateless (stays inside CUDA Graph)">State: Stateless (stays inside CUDA Graph)</span>
  <span class="pllm-step-arrow">→</span>
  <span class="pllm-step-pill" data-i18n-zh="效：@pllm.add_logits 逐行修正" data-i18n-en="Effect: @pllm.add_logits contrast">Effect: @pllm.add_logits contrast</span>
</div>

```python
import torch
import pllm
from pllm import testing


class LayerContrast(pllm.Workload):
    @pllm.add_logits
    def contrast(self, logits: torch.Tensor = pllm.layer_logits((12, -1))) -> torch.Tensor:
        return 0.5 * (logits[:, 1] - logits[:, 0])


config = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=16)
print(testing.check(LayerContrast(), config))
```

  </div>

  <div class="pllm-tab-panel is-hidden" data-pllm-panel="adapter" markdown="1">

<div class="pllm-step-pills">
  <span class="pllm-step-pill" data-i18n-zh="读：Prompt 平均表示生成每请求低秩因子" data-i18n-en="Read: Prompt hidden -> Hypernetwork factors">Read: Prompt hidden -> Hypernetwork factors</span>
  <span class="pllm-step-arrow">→</span>
  <span class="pllm-step-pill" data-i18n-zh="存：分页 RequestState 低秩权重 a, b" data-i18n-en="State: Paged RequestState factors (a, b)">State: Paged RequestState factors (a, b)</span>
  <span class="pllm-step-arrow">→</span>
  <span class="pllm-step-pill" data-i18n-zh="效：pllm.model.Adapter 叠加至量化/稠密投影" data-i18n-en="Effect: pllm.model.Adapter overlay in-graph">Effect: pllm.model.Adapter overlay in-graph</span>
</div>

```python
import torch
import pllm


class PromptLoRA(pllm.Workload):
    lora = pllm.RequestState(
        fields={
            "a": pllm.Field(shape=lambda m: (4, m.hidden_size)),
            "b": pllm.Field(shape=lambda m: (m.hidden_size, 4)),
        }
    )
    overlay = pllm.model.Adapter(
        state="lora",
        targets=(pllm.model.Projection(layers=(8, 12, 16), name="down_proj"),),
        operator="low_rank",
    )

    @pllm.write(state="lora", tokens="prefill")
    def generate_adapter(self, h: torch.Tensor = pllm.hidden(4)) -> dict[str, torch.Tensor]:
        seed = h[:, :4]
        return {"a": seed.unsqueeze(-1).expand(-1, 4, h.shape[-1]) * 0.01, "b": torch.zeros(h.shape[0], h.shape[-1], 4, device=h.device, dtype=h.dtype)}
```

  </div>

  <div class="pllm-tab-panel is-hidden" data-pllm-panel="serve" markdown="1">

<div class="pllm-step-pills">
  <span class="pllm-step-pill" data-i18n-zh="安装：uv 虚拟环境 + vLLM 0.27–0.34" data-i18n-en="Install: uv venv + vLLM 0.27–0.34">Install: uv venv + vLLM 0.27–0.34</span>
  <span class="pllm-step-arrow">→</span>
  <span class="pllm-step-pill" data-i18n-zh="服务：OpenAI 流式 SSE + 会话级状态路由" data-i18n-en="Serve: OpenAI SSE + Session State Routing">Serve: OpenAI SSE + Session State Routing</span>
  <span class="pllm-step-arrow">→</span>
  <span class="pllm-step-pill" data-i18n-zh="协作：一键导出 Claude / Codex / Cursor Skill" data-i18n-en="Agent: Built-in Claude / Codex / Cursor Skill">Agent: Built-in Claude / Codex / Cursor Skill</span>
</div>

```bash
# 1. Recommended: create a virtual environment with uv (or install into an existing vLLM 0.27–0.34 env via `uv pip install -e . --no-deps`)
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e ".[vllm032,test]"

# 2. Start OpenAI-compatible HTTP server (callers pass extra_body={"vllm_xargs": {"pllm_route": "session-42"}})
pllm serve --model Qwen/Qwen2.5-0.5B-Instruct --workload pllm.workloads:DoLaWorkload --port 8000

# 3. Install packaged authoring & HF/PEFT migration skills for your coding agent
pllm skill --install claude   # or: codex | cursor
```

  </div>
</div>

<section id="docs" class="zine-hub-section">
  <div class="zine-section-head">
    <div>
      <h2 data-i18n-zh="pLLM 双语文档中心（面向开发者与研究者）" data-i18n-en="pLLM Bilingual Documentation Hub">pLLM Bilingual Documentation Hub</h2>
      <p class="zine-section-sub" data-i18n-zh="收录全部面向人类开发者的教程、指南、架构解析、API 手册与算法示例（均提供完整中英双语，随 pLLM 仓库自动同步更新）。" data-i18n-en="All human-facing tutorials, user guides, serving how-tos, design explanations, references, and algorithm examples—fully bilingual (EN / 中文) and auto-synced from pLLM.">All human-facing tutorials, user guides, serving how-tos, design explanations, references, and algorithm examples—fully bilingual (EN / 中文) and auto-synced from pLLM.</p>
    </div>
    <a class="zine-pill-link primary" data-i18n-zh="文档导航地图 →" data-i18n-en="Docs Roadmap →" href="{{ '/pllm/docs/' | relative_url }}">Docs Roadmap →</a>
  </div>

  <div class="pllm-filter-bar" role="group" aria-label="Filter documentation categories">
    <button type="button" class="pllm-filter-btn is-active" data-doc-filter="all" aria-pressed="true" data-i18n-zh="全部文档" data-i18n-en="All Docs">All Docs</button>
    {% for cat in sync.doc_categories %}
    <button type="button" class="pllm-filter-btn" data-doc-filter="{{ cat.id }}" aria-pressed="false" data-i18n-zh="{{ cat.short_zh | escape }} ({{ cat.items.size }})" data-i18n-en="{{ cat.short_en | escape }} ({{ cat.items.size }})">{{ cat.short_en }} ({{ cat.items.size }})</button>
    {% endfor %}
  </div>

{% for cat in sync.doc_categories %}

  <div class="pllm-doc-cat-block" data-doc-cat-block="{{ cat.id }}">
    <h3 class="zine-cat-heading" data-i18n-zh="{{ cat.title_zh | escape }}" data-i18n-en="{{ cat.title_en | escape }}">{{ cat.title_en }}</h3>
    <div class="zine-doc-grid">
      {% for item in cat.items %}
      <a class="zine-doc-card" data-cat="{{ cat.id }}" href="{{ item.url | relative_url }}">
        <div class="zine-doc-card-top">
          <span class="zine-lang-badge">EN / 中文{% if item.kind == "notebook" %} &middot; ipynb{% endif %}</span>
          <span class="zine-doc-path">{{ item.rel_path | remove_first: "doc/" }}</span>
        </div>
        <div class="zine-doc-title" data-i18n-zh="{{ item.title_zh | escape }}" data-i18n-en="{{ item.title_en | escape }}">{{ item.title_en }}</div>
        {% if item.summary_en != "" or item.summary_zh != "" %}
        <p class="zine-doc-summary" data-i18n-zh="{{ item.summary_zh | escape }}" data-i18n-en="{{ item.summary_en | escape }}">{{ item.summary_en }}</p>
        {% endif %}
      </a>
      {% endfor %}
    </div>
  </div>
  {% endfor %}
</section>

<section id="blog" class="zine-hub-section">
  <div class="zine-section-head">
    <div>
      <h2 data-i18n-zh="pLLM 技术随笔与机制拆解（中英双语）" data-i18n-en="pLLM Engineering Essays &amp; Deep Dives (Bilingual)">pLLM Engineering Essays &amp; Deep Dives (Bilingual)</h2>
      <p class="zine-section-sub" data-i18n-zh="十篇面向推理系统社区的平权独立文章，配有完整架构与内存解剖插图，随 pLLM 仓库更新自动同步至个人博客。" data-i18n-en="Ten standalone inference-systems deep dives with visual architecture &amp; memory diagrams—automatically synced from the pLLM repository to this blog.">Ten standalone inference-systems deep dives with visual architecture &amp; memory diagrams—automatically synced from the pLLM repository to this blog.</p>
    </div>
    <a class="zine-pill-link" data-i18n-zh="查看全部个人博客 →" data-i18n-en="All Blog Posts →" href="{{ '/blog/' | relative_url }}">All Blog Posts →</a>
  </div>

  <div class="zine-lang-panel" data-lang-panel="en" lang="en">
    <figure class="pllm-hero-figure">
      <img src="{{ '/assets/img/pllm/blog/assets/system-anatomy-en.svg' | relative_url }}" alt="Inference Engine Anatomy Map for the 10-Part Deep-Dive Series">
      <figcaption class="pllm-fig-caption">System Anatomy Map &middot; Each standalone article dissects one concrete subsystem of a modern continuous-batching inference engine.</figcaption>
    </figure>
  </div>

  <div class="zine-lang-panel is-hidden" data-lang-panel="zh" lang="zh-CN">
    <figure class="pllm-hero-figure">
      <img src="{{ '/assets/img/pllm/blog/assets/system-anatomy-zh.svg' | relative_url }}" alt="现代连续批处理推理引擎解剖路标（十篇系列文章对应位置）">
      <figcaption class="pllm-fig-caption">推理引擎解剖路标 · 每篇独立文章对应现代连续批处理推理引擎上的一个具体硬件或调度部位。</figcaption>
    </figure>
  </div>

  <div class="pllm-blog-grid">
    {% for post in pllm_posts %}
    {% assign title_zh = post.title_zh | default: post.title %}
    {% assign title_en = post.title_en | default: post.title %}
    {% assign desc_zh = post.description_zh | default: post.description %}
    {% assign desc_en = post.description_en | default: post.description %}
    {% assign badge_zh = post.pllm_badge_zh | default: "技术拆解" %}
    {% assign badge_en = post.pllm_badge_en | default: "Deep Dive" %}
    {% assign read_time = post.content | number_of_words | divided_by: 360 | plus: 1 %}
    <a class="pllm-blog-card" href="{{ post.url | relative_url }}">
      <div class="zine-series-meta">
        <span class="zine-series-number">#{{ post.pllm_order | prepend: "0" | slice: -2, 2 }}</span>
        <span class="zine-lang-badge" data-i18n-zh="{{ badge_zh | escape }}" data-i18n-en="{{ badge_en | escape }}">{{ badge_en }}</span>
        <span class="post-meta">{{ read_time }} min</span>
      </div>
      <div class="pllm-blog-card-title" data-i18n-zh="{{ title_zh | escape }}" data-i18n-en="{{ title_en | escape }}">{{ title_en }}</div>
      <p class="pllm-blog-card-desc" data-i18n-zh="{{ desc_zh | escape }}" data-i18n-en="{{ desc_en | escape }}">{{ desc_en }}</p>
    </a>
    {% endfor %}
  </div>
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
<script defer src="{{ '/assets/js/pllm-hub.js' | relative_url | bust_file_cache }}"></script>
