---
layout: default
permalink: /pllm/
title: pLLM Project
description: "pLLM (Programmable LLM): Stop forking vLLM to test a new idea. Read hidden states, attach per-request fast weights, or run online decode-time training—compiled directly into unmodified vLLM’s continuous batching and CUDA graphs."
nav: true
nav_order: 3
---

{% assign sync = site.data.pllm_sync %}
{% assign pllm_posts = site.posts | where: "pllm_synced", true | sort: "pllm_order" %}
{% assign arch_posts = pllm_posts | where: "pllm_track", "systems-architecture" %}
{% assign intern_posts = pllm_posts | where: "pllm_track", "engine-internals" %}
{% assign debug_posts = pllm_posts | where: "pllm_track", "debugging-notes" %}

<link rel="stylesheet" href="{{ '/assets/css/pllm-hub.css' | relative_url | bust_file_cache }}">

<div class="post">

<header class="zine-masthead">
  <div class="zine-kicker" data-i18n-zh="pLLM (Programmable LLM) · 开源推理编程层 v0.5.1 · 兼容 vLLM 0.27–0.34" data-i18n-en="pLLM (Programmable LLM) &middot; Open-Source Release v0.5.1 &middot; vLLM 0.27–0.34">pLLM (Programmable LLM) &middot; Open-Source Release v0.5.1 &middot; vLLM 0.27–0.34</div>
  <h1 class="post-title" data-i18n-zh="别再为了试个新算法，去改 vLLM 源码了。" data-i18n-en="Stop forking vLLM to test a new idea.">Stop forking vLLM to test a new idea.</h1>
  <p class="zine-deck" data-i18n-zh="读取内部隐藏状态、按请求挂载私有快权重，或在解码时运行在线训练——直接编译进未修改原版 vLLM 的连续批处理与静态 CUDA Graph。" data-i18n-en="Read hidden states, attach per-request fast weights, or run online decode-time training—compiled directly into unmodified vLLM’s continuous batching and CUDA graphs.">Read hidden states, attach per-request fast weights, or run online decode-time training—compiled directly into unmodified vLLM’s continuous batching and CUDA graphs.</p>
</header>

<div class="zine-dispatch" data-title-zh="pLLM：别再为了试新算法去改 vLLM 源码了" data-title-en="pLLM: Stop forking vLLM to test a new idea.">
  <div class="zine-dispatch-links">
    <a class="zine-pill-link primary" href="https://github.com/LinesHogan/pLLM" target="_blank" rel="noopener">GitHub &middot; LinesHogan/pLLM ↗</a>
    <a class="zine-pill-link" href="https://github.com/LinesHogan/pLLM/releases/tag/v0.5.1" target="_blank" rel="noopener">Release v0.5.1 ↗</a>
    <a class="zine-pill-link accent" data-i18n-zh="上手教程与文档 ↓" data-i18n-en="Tutorials &amp; Docs ↓" href="#docs">Tutorials &amp; Docs ↓</a>
    <a class="zine-pill-link" data-i18n-zh="背后的故事与拆解 ({{ pllm_posts.size }}) ↓" data-i18n-en="Essays &amp; Stories ({{ pllm_posts.size }}) ↓" href="#blog">Essays &amp; Stories ({{ pllm_posts.size }}) ↓</a>
    <a class="zine-pill-link" data-i18n-zh="ICML 2026 论文 (ESamp) ↗" data-i18n-en="ICML 2026 Paper (ESamp) ↗" href="https://proceedings.mlr.press/v306/zeng26m.html" target="_blank" rel="noopener">ICML 2026 Paper (ESamp) ↗</a>
    {% if sync.commit_sha %}
    <a class="zine-pill-link" href="{{ sync.commit_url }}" target="_blank" rel="noopener" title="Auto-synced from LinesHogan/pLLM">Sync @ {{ sync.commit_sha }} ↗</a>
    {% endif %}
  </div>
</div>

<div class="pllm-story-grid">
  <div class="pllm-story-card pain">
    <div class="pllm-story-step" data-i18n-zh="01 · 令人头疼的断层" data-i18n-en="01 &middot; The Painful Gap">01 &middot; The Painful Gap</div>
    <h2 class="pllm-story-title" data-i18n-zh="HF 写半天慢得跑不动，改 vLLM 又得死磕几周" data-i18n-en="Half a day in HF—or weeks fighting a vLLM fork">Half a day in HF—or weeks fighting a vLLM fork</h2>
    <p class="pllm-story-body" data-i18n-zh="做跨层对比解码、动态快权重或边解码边训练（如 TTT、ESamp）时：在 Hugging Face 里写 bs=1 循环只要半天，但慢得没法做大规模评测；想上 vLLM 跑满吞吐，就得 fork 源码跟分页 KV、抢占调度和静态图死磕数周，vLLM 一升级代码就全废了。" data-i18n-en="Writing a bs=1 Hugging Face loop for contrastive decoding, fast weights, or online training (TTT, ESamp) takes half a day—yet runs too slowly for real benchmarks. Porting it to vLLM means wrestling for weeks with paged KV, preemption, and CUDA graphs, until the next vLLM upgrade breaks it all.">Writing a bs=1 Hugging Face loop for contrastive decoding, fast weights, or online training (TTT, ESamp) takes half a day—yet runs too slowly for real benchmarks. Porting it to vLLM means wrestling for weeks with paged KV, preemption, and CUDA graphs, until the next vLLM upgrade breaks it all.</p>
  </div>

  <div class="pllm-story-card declare">
    <div class="pllm-story-step" data-i18n-zh="02 · 你只需要写什么" data-i18n-en="02 &middot; What You Write">02 &middot; What You Write</div>
    <h2 class="pllm-story-title" data-i18n-zh="几十行纯 Python，只声明算法关心的三件事" data-i18n-en="A few dozen lines of Python. Declare three things.">A few dozen lines of Python. Declare three things.</h2>
    <p class="pllm-story-body" data-i18n-zh="完全不改 vLLM 源码。在 pLLM 里只写标准 PyTorch 张量数学，声明三件事：① 读哪一层的内部状态；② 每个请求维护什么私有状态或优化器；③ 在哪里干预前向计算或词表采样。" data-i18n-en="Leave upstream vLLM untouched. Using standard PyTorch tensor math, you simply declare: (1) which layer's internal state to read, (2) what private state or optimizer each request carries, and (3) where to intervene in the forward pass or sampler.">Leave upstream vLLM untouched. Using standard PyTorch tensor math, you simply declare: (1) which layer's internal state to read, (2) what private state or optimizer each request carries, and (3) where to intervene in the forward pass or sampler.</p>
  </div>

  <div class="pllm-story-card compile">
    <div class="pllm-story-step" data-i18n-zh="03 · 底层如何跑满速度" data-i18n-en="03 &middot; How It Runs Fast">03 &middot; How It Runs Fast</div>
    <h2 class="pllm-story-title" data-i18n-zh="CPU 两秒验完隔离安全，自动编译进原版静态图" data-i18n-en="Checked on CPU in 2s. Compiled into stock vLLM.">Checked on CPU in 2s. Compiled into stock vLLM.</h2>
    <p class="pllm-story-body" data-i18n-zh="底层就像 vLLM 上的 eBPF：运行前先在 CPU 上花不到 2 秒做静态契约校验，确保多租户隔离与抢占安全；随后自动把逻辑内联编译进原版 vLLM 的 CUDA Graph，独立的在线训练任务还会自动放到旁路 Stream 并行重叠。" data-i18n-en="Under the hood, it works like eBPF for vLLM: checking multi-tenant isolation and preemption safety on CPU in under 2 seconds before touching a GPU, then compiling your logic inline into unmodified vLLM's CUDA graphs—overlapping independent training work on a side stream automatically.">Under the hood, it works like eBPF for vLLM: checking multi-tenant isolation and preemption safety on CPU in under 2 seconds before touching a GPU, then compiling your logic inline into unmodified vLLM's CUDA graphs—overlapping independent training work on a side stream automatically.</p>
  </div>
</div>

<div class="zine-lang-panel" data-lang-panel="en" lang="en">
  <figure class="pllm-hero-figure">
    <img src="{{ '/assets/img/pllm/doc/assets/pllm-architecture-en.svg' | relative_url | bust_file_cache }}" alt="pLLM Overview: From Plain Python Workload to Production vLLM Execution">
    <figcaption class="pllm-fig-caption">From Idea to Serving &middot; Write a plain-Python Workload (left); pLLM checks multi-tenant safety on CPU in &lt; 2s (center) and compiles it directly into stock vLLM (right).</figcaption>
  </figure>
</div>

<div class="zine-lang-panel is-hidden" data-lang-panel="zh" lang="zh-CN">
  <figure class="pllm-hero-figure">
    <img src="{{ '/assets/img/pllm/doc/assets/pllm-architecture-zh.svg' | relative_url | bust_file_cache }}" alt="pLLM 工作流程总览：从几十行纯 Python 到未修改 vLLM 满血并发">
    <figcaption class="pllm-fig-caption">从研究代码到满血并发 · 左侧用纯 Python 声明你的算法逻辑；中间在 CPU 上不到 2 秒完成多租户安全校验；右侧自动编译进未修改的原版 vLLM。</figcaption>
  </figure>
</div>

<div class="pllm-bento-grid">
  <div class="pllm-bento-card">
    <div class="pllm-card-kicker" data-i18n-zh="真实评测与线上吞吐 · 单张 RTX 4090 · 原版 vLLM 开启静态图" data-i18n-en="Real-World Speed &middot; Single RTX 4090 &middot; Stock vLLM + CUDA Graphs">Real-World Speed &middot; Single RTX 4090 &middot; Stock vLLM + CUDA Graphs</div>
    <h2 class="pllm-card-heading" data-i18n-zh="写研究代码的轻松，跑生产级引擎的速度" data-i18n-en="Prototype in Python. Keep Native vLLM Speed.">Prototype in Python. Keep Native vLLM Speed.</h2>

    <div class="pllm-bench-group">
      <div class="pllm-bench-subtitle" data-i18n-zh="1 · 激活引导（Steering）与实时特征导出 · Qwen2.5-7B · 几乎零延迟损耗" data-i18n-en="1 &middot; Activation Steering &amp; Live Hidden Readout &middot; Qwen2.5-7B &middot; Near-Zero Tax">1 &middot; Activation Steering &amp; Live Hidden Readout &middot; Qwen2.5-7B &middot; Near-Zero Tax</div>
      <div class="pllm-tpot-strip">
        <div class="pllm-tpot-pill">
          <div class="pllm-tpot-val">16.52 ms <span class="delta">(+0.02 ms)</span></div>
          <div class="pllm-tpot-lbl" data-i18n-zh="单请求流式引导每词延迟 (99.7% 吞吐)" data-i18n-en="bs=1 steering TPOT (99.7% speed)">bs=1 steering TPOT (99.7% speed)</div>
        </div>
        <div class="pllm-tpot-pill">
          <div class="pllm-tpot-val">22.02 ms <span class="delta">(+0.05 ms)</span></div>
          <div class="pllm-tpot-lbl" data-i18n-zh="128 并发激活引导延迟 (98.3% 吞吐)" data-i18n-en="128-seq steering TPOT (98.3% speed)">128-seq steering TPOT (98.3% speed)</div>
        </div>
        <div class="pllm-tpot-pill">
          <div class="pllm-tpot-val">22.17 ms <span class="delta">(+0.20 ms)</span></div>
          <div class="pllm-tpot-lbl" data-i18n-zh="引导 + 全维特征实时回传 CPU (98.0%)" data-i18n-en="Steer + full 3584-d CPU export (98%)">Steer + full 3584-d CPU export (98%)</div>
        </div>
      </div>
      <div class="pllm-bar-row">
        <div class="pllm-bar-meta">
          <span class="pllm-bar-name" data-i18n-zh="原版 vLLM 基线 (128 并发)" data-i18n-en="Stock vLLM Baseline (128 concurrent)">Stock vLLM Baseline (128 concurrent)</span>
          <span class="pllm-bar-val">5,719 tok/s &middot; 21.97 ms (100%)</span>
        </div>
        <div class="pllm-bar-track"><div class="pllm-bar-fill" style="width: 100%"></div></div>
      </div>
      <div class="pllm-bar-row">
        <div class="pllm-bar-meta">
          <span class="pllm-bar-name" data-i18n-zh="开启第 14 层激活引导 (Steering)" data-i18n-en="With Layer-14 Activation Steering">With Layer-14 Activation Steering</span>
          <span class="pllm-bar-val">5,623 tok/s &middot; 22.02 ms (98.3%)</span>
        </div>
        <div class="pllm-bar-track"><div class="pllm-bar-fill emerald" style="width: 98.3%"></div></div>
      </div>
      <div class="pllm-bar-row">
        <div class="pllm-bar-meta">
          <span class="pllm-bar-name" data-i18n-zh="激活引导 + 整层特征逐词回传 CPU (零丢行)" data-i18n-en="Steering + Full Residual Streamed to CPU">Steering + Full Residual Streamed to CPU</span>
          <span class="pllm-bar-val">5,604 tok/s &middot; 22.17 ms (98.0%)</span>
        </div>
        <div class="pllm-bar-track"><div class="pllm-bar-fill blue" style="width: 98.0%"></div></div>
      </div>
    </div>

    <div class="pllm-bench-group">
      <div class="pllm-bench-subtitle" data-i18n-zh="2 · 128 并发边解码边跑独立在线训练 (ESamp · 每个请求挂独立 MLP + AdamW)" data-i18n-en="2 &middot; 128-Request Online Training During Decode (ESamp &middot; Private MLP + AdamW Each)">2 &middot; 128-Request Online Training During Decode (ESamp &middot; Private MLP + AdamW Each)</div>
      <div class="pllm-bar-row">
        <div class="pllm-bar-meta">
          <span class="pllm-bar-name" data-i18n-zh="逐词跑 AdamW 在线训练 (原版基线 5,894 tok/s)" data-i18n-en="Per-token AdamW training (vs. 5,894 native)">Per-token AdamW training (vs. 5,894 native)</span>
          <span class="pllm-bar-val">5,738 tok/s (97.4%)</span>
        </div>
        <div class="pllm-bar-track"><div class="pllm-bar-fill emerald" style="width: 97.4%"></div></div>
      </div>
      <div class="pllm-bar-row">
        <div class="pllm-bar-meta">
          <span class="pllm-bar-name" data-i18n-zh="在线训练 + 采样分布实时引导同时开启 (ESamp)" data-i18n-en="Online training + live guided sampling (ESamp)">Online training + live guided sampling (ESamp)</span>
          <span class="pllm-bar-val">5,678 tok/s (96.3%)</span>
        </div>
        <div class="pllm-bar-track"><div class="pllm-bar-fill red" style="width: 96.3%"></div></div>
      </div>
    </div>

    <div class="pllm-bench-group">
      <div class="pllm-bench-subtitle" data-i18n-zh="3 · 为 4B Test-Time Training 直接解锁并发批处理 (对比官方仅 bs=1 实现)" data-i18n-en="3 &middot; Unlocking Batch Concurrency for 4B Test-Time Training (vs. Official bs=1)">3 &middot; Unlocking Batch Concurrency for 4B Test-Time Training (vs. Official bs=1)</div>
      <div class="pllm-bar-row">
        <div class="pllm-bar-meta">
          <span class="pllm-bar-name" data-i18n-zh="官方 Hugging Face 实现 (仅支持单请求 bs=1)" data-i18n-en="Official Hugging Face Code (bs=1 only)">Official Hugging Face Code (bs=1 only)</span>
          <span class="pllm-bar-val">54.8 tok/s (1.00&times;)</span>
        </div>
        <div class="pllm-bar-track"><div class="pllm-bar-fill" style="width: 31.9%"></div></div>
      </div>
      <div class="pllm-bar-row">
        <div class="pllm-bar-meta">
          <span class="pllm-bar-name" data-i18n-zh="pLLM 单请求 (~25 行 Python 声明)" data-i18n-en="pLLM Single Request (~25 lines of Python)">pLLM Single Request (~25 lines of Python)</span>
          <span class="pllm-bar-val">73.0 tok/s (1.33&times;)</span>
        </div>
        <div class="pllm-bar-track"><div class="pllm-bar-fill blue" style="width: 42.6%"></div></div>
      </div>
      <div class="pllm-bar-row">
        <div class="pllm-bar-meta">
          <span class="pllm-bar-name" data-i18n-zh="pLLM 4 请求并发批处理 (直接解锁 3 倍+ 吞吐)" data-i18n-en="pLLM 4 Concurrent Requests (Batching Unlocked)">pLLM 4 Concurrent Requests (Batching Unlocked)</span>
          <span class="pllm-bar-val">171.6 tok/s (3.13&times;)</span>
        </div>
        <div class="pllm-bar-track"><div class="pllm-bar-fill mustard" style="width: 100%"></div></div>
      </div>
    </div>

  </div>

  <div class="pllm-bento-card blue">
    <div class="pllm-card-kicker" data-i18n-zh="告别分支维护税 · 从实验一路通往线上服务" data-i18n-en="Zero Maintenance Tax &middot; From Idea to Production">Zero Maintenance Tax &middot; From Idea to Production</div>
    <h2 class="pllm-card-heading" data-i18n-zh="vLLM 升级代码不废，上 GPU 前先抓出并发隐患" data-i18n-en="Upgrade vLLM Anytime. Catch Bugs Before Renting a GPU.">Upgrade vLLM Anytime. Catch Bugs Before Renting a GPU.</h2>

    <div class="pllm-kpi-grid">
      <div class="pllm-kpi-tile">
        <div class="pllm-kpi-num" data-i18n-zh="0 行改引擎" data-i18n-en="0 Engine Edits">0 Engine Edits</div>
        <div class="pllm-kpi-caption" data-i18n-zh="直接装在原版 vLLM 上，彻底告别维护私有分支" data-i18n-en="Runs on pip-installed vLLM; never rebase a fork again">Runs on pip-installed vLLM; never rebase a fork again</div>
      </div>
      <div class="pllm-kpi-tile">
        <div class="pllm-kpi-num">15 Families</div>
        <div class="pllm-kpi-caption" data-i18n-zh="兼容 vLLM 0.27–0.34、主流架构、量化与多卡并行" data-i18n-en="Works across vLLM 0.27–0.34, quantization &amp; multi-GPU TP">Works across vLLM 0.27–0.34, quantization &amp; multi-GPU TP</div>
      </div>
      <div class="pllm-kpi-tile">
        <div class="pllm-kpi-num" data-i18n-zh="CPU 2 秒验错" data-i18n-en="&lt; 2s CPU Check">&lt; 2s CPU Check</div>
        <div class="pllm-kpi-caption" data-i18n-zh="不用租 GPU，在笔记本上提前查出多请求串味与抢占漏洞" data-i18n-en="Catches cross-request leaks on your laptop without a GPU">Catches cross-request leaks on your laptop without a GPU</div>
      </div>
      <div class="pllm-kpi-tile">
        <div class="pllm-kpi-num" data-i18n-zh="在线秒级切层" data-i18n-en="Live Layer Switch">Live Layer Switch</div>
        <div class="pllm-kpi-caption" data-i18n-zh="服务运行中随时切换观测层或引导层，无需停机重录图" data-i18n-en="Switch probe, steering, or LoRA layers with zero downtime">Switch probe, steering, or LoRA layers with zero downtime</div>
      </div>
      <div class="pllm-kpi-tile">
        <div class="pllm-kpi-num">+0.05 ms</div>
        <div class="pllm-kpi-caption" data-i18n-zh="128 并发开启激活引导增加的每词延迟（单请求 +0.02 ms）" data-i18n-en="Per-token latency added by steering at 128 concurrency">Per-token latency added by steering at 128 concurrency</div>
      </div>
      <div class="pllm-kpi-tile">
        <div class="pllm-kpi-num" data-i18n-zh="一套代码上线" data-i18n-en="Eval to Serving">Eval to Serving</div>
        <div class="pllm-kpi-caption" data-i18n-zh="同一个 Python 类无缝跑批量评测、异步调用与 OpenAI 接口" data-i18n-en="Same Python class runs batch evals, async &amp; OpenAI SSE">Same Python class runs batch evals, async &amp; OpenAI SSE</div>
      </div>
    </div>

    <div class="pllm-chip-cloud">
      <span class="pllm-chip hi">Qwen2 / Qwen3 / Qwen3.5</span>
      <span class="pllm-chip hi">DeepSeek V2 / V3 (MLA)</span>
      <span class="pllm-chip hi">Llama 3</span>
      <span class="pllm-chip">Mistral &amp; Mixtral</span>
      <span class="pllm-chip">Gemma 2 / 3</span>
      <span class="pllm-chip">Phi-3</span>
      <span class="pllm-chip">GLM-4</span>
      <span class="pllm-chip">gpt-oss</span>
      <span class="pllm-chip hi">GPTQ / AWQ / FP8</span>
      <span class="pllm-chip hi">Multi-GPU TP</span>
      <span class="pllm-chip hi">Claude / Codex / Cursor Skill</span>
    </div>

  </div>
</div>

<div id="markdown-content">

<section class="zine-hub-section" style="margin-bottom: 2.2rem;">
  <div class="zine-section-head" style="margin-bottom: 0.9rem;">
    <div>
      <h2 data-i18n-zh="代码长什么样？二十行标准 PyTorch 搞定" data-i18n-en="See It in Code: ~20 Lines of Plain PyTorch">See It in Code: ~20 Lines of Plain PyTorch</h2>
      <p class="zine-section-sub" data-i18n-zh="不用碰调度器、不用手写分页块表、不用管 CUDA Graph 怎么录制——只写你的算法逻辑本身。" data-i18n-en="No scheduler surgery, no manual block tables, no CUDA graph boilerplate—just write the algorithm itself.">No scheduler surgery, no manual block tables, no CUDA graph boilerplate—just write the algorithm itself.</p>
    </div>
  </div>

<div class="pllm-showcase">
  <div class="pllm-tab-bar" role="tablist" aria-label="pLLM Capability Examples">
    <button type="button" class="pllm-tab-btn is-active" data-pllm-tab="esamp" role="tab" aria-selected="true" data-i18n-zh="1 · 边解码边在线训练 (ESamp)" data-i18n-en="1 &middot; Online Training During Decode">1 &middot; Online Training During Decode</button>
    <button type="button" class="pllm-tab-btn" data-pllm-tab="route" role="tab" aria-selected="false" data-i18n-zh="2 · 在线秒级切层与激活引导" data-i18n-en="2 &middot; Switch Steering Layers Live">2 &middot; Switch Steering Layers Live</button>
    <button type="button" class="pllm-tab-btn" data-pllm-tab="dola" role="tab" aria-selected="false" data-i18n-zh="3 · 跨层对比解码 (DoLa)" data-i18n-en="3 &middot; Cross-Layer Contrast (DoLa)">3 &middot; Cross-Layer Contrast (DoLa)</button>
    <button type="button" class="pllm-tab-btn" data-pllm-tab="adapter" role="tab" aria-selected="false" data-i18n-zh="4 · 按请求实时生成独立 LoRA" data-i18n-en="4 &middot; Per-Request Dynamic LoRA">4 &middot; Per-Request Dynamic LoRA</button>
    <button type="button" class="pllm-tab-btn" data-pllm-tab="serve" role="tab" aria-selected="false" data-i18n-zh="5 · 一行命令启动 OpenAI 服务" data-i18n-en="5 &middot; Serve as OpenAI Endpoint">5 &middot; Serve as OpenAI Endpoint</button>
  </div>

  <div class="pllm-tab-panel" data-pllm-panel="esamp" markdown="1">

<div class="pllm-step-pills">
  <span class="pllm-step-pill" data-i18n-zh="读哪层：第 0 层与末层隐藏状态" data-i18n-en="Read: Layer 0 &amp; final hidden states">Read: Layer 0 &amp; final hidden states</span>
  <span class="pllm-step-arrow">→</span>
  <span class="pllm-step-pill" data-i18n-zh="存什么：每个请求独立小网络 + AdamW" data-i18n-en="State: Private MLP + AdamW per request">State: Private MLP + AdamW per request</span>
  <span class="pllm-step-arrow">→</span>
  <span class="pllm-step-pill" data-i18n-zh="怎么干预：边训边引导词表采样" data-i18n-en="Intervene: Train online &amp; guide sampling">Intervene: Train online &amp; guide sampling</span>
</div>

```python
from dataclasses import dataclass
from torch import nn
import torch
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

  <div class="pllm-tab-panel is-hidden" data-pllm-panel="route" markdown="1">

<div class="pllm-step-pills">
  <span class="pllm-step-pill" data-i18n-zh="存什么：当前目标层号 + 引导向量" data-i18n-en="State: Target layer ID + steering vector">State: Target layer ID + steering vector</span>
  <span class="pllm-step-arrow">→</span>
  <span class="pllm-step-pill" data-i18n-zh="随时切换：第 4 / 12 / 20 层在线秒切（免重录图）" data-i18n-en="Switch live: Jump across layers 4 / 12 / 20">Switch live: Jump across layers 4 / 12 / 20</span>
  <span class="pllm-step-arrow">→</span>
  <span class="pllm-step-pill" data-i18n-zh="怎么干预：叠加引导向量并实时回传特征" data-i18n-en="Intervene: Steer activations &amp; stream probes">Intervene: Steer activations &amp; stream probes</span>
</div>

```python
import torch
import pllm
from pllm import testing


class DynamicSteering(pllm.Workload):
    ctrl = pllm.InstanceState(
        {"layer": pllm.Field((), torch.int64), "vector": pllm.Field(lambda m: (m.hidden_size,))},
        init=pllm.Imported(),
    )
    route = pllm.model.LayerRoute("ctrl", over=(4, 12, 20))
    steer = route.patch(operator="add", decode=True)

    @pllm.export(name="probe", max_rows=8, max_cols=1)
    def capture(self, r: torch.Tensor = route.residual()) -> torch.Tensor:
        return r.float().norm(dim=-1, keepdim=True)


cfg = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=32)
print(testing.check(DynamicSteering(), cfg))
```

  </div>

  <div class="pllm-tab-panel is-hidden" data-pllm-panel="dola" markdown="1">

<div class="pllm-step-pills">
  <span class="pllm-step-pill" data-i18n-zh="读哪层：同时取出第 12 层与末层词表分布" data-i18n-en="Read: Layer 12 &amp; final-layer logits">Read: Layer 12 &amp; final-layer logits</span>
  <span class="pllm-step-arrow">→</span>
  <span class="pllm-step-pill" data-i18n-zh="存什么：完全无状态（零额外显存开销）" data-i18n-en="State: Stateless (zero extra memory)">State: Stateless (zero extra memory)</span>
  <span class="pllm-step-arrow">→</span>
  <span class="pllm-step-pill" data-i18n-zh="怎么干预：跨层对比修正输出分布" data-i18n-en="Intervene: Contrastive logit adjustment">Intervene: Contrastive logit adjustment</span>
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
  <span class="pllm-step-pill" data-i18n-zh="读哪层：从输入提示词提取特征" data-i18n-en="Read: Extract features from prompt">Read: Extract features from prompt</span>
  <span class="pllm-step-arrow">→</span>
  <span class="pllm-step-pill" data-i18n-zh="存什么：每个请求一份专属低秩权重" data-i18n-en="State: Dedicated LoRA weights per request">State: Dedicated LoRA weights per request</span>
  <span class="pllm-step-arrow">→</span>
  <span class="pllm-step-pill" data-i18n-zh="怎么干预：自动挂载至模型投影层" data-i18n-en="Intervene: Attach to model projections">Intervene: Attach to model projections</span>
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
  <span class="pllm-step-pill" data-i18n-zh="第一步：用 uv 装入原版 vLLM 环境" data-i18n-en="1. Install with uv alongside stock vLLM">1. Install with uv alongside stock vLLM</span>
  <span class="pllm-step-arrow">→</span>
  <span class="pllm-step-pill" data-i18n-zh="第二步：一行命令暴露标准 OpenAI 接口" data-i18n-en="2. Launch standard OpenAI HTTP server">2. Launch standard OpenAI HTTP server</span>
  <span class="pllm-step-arrow">→</span>
  <span class="pllm-step-pill" data-i18n-zh="第三步：让 Claude / Cursor 自动帮你迁移旧代码" data-i18n-en="3. Let Claude / Cursor port your HF code">3. Let Claude / Cursor port your HF code</span>
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
</section>

<section id="docs" class="zine-hub-section">
  <div class="zine-section-head">
    <div>
      <h2 data-i18n-zh="上手教程与双语文档" data-i18n-en="Runnable Tutorials &amp; Bilingual Docs">Runnable Tutorials &amp; Bilingual Docs</h2>
      <p class="zine-section-sub" data-i18n-zh="从 5 分钟快速入门、可交互运行的算法 Notebook，到上线部署与完整 API 查询（全部支持中英切换，随仓库自动同步）。" data-i18n-en="From a 5-minute quickstart and runnable algorithm notebooks to serving guides and API references—all in English &amp; Chinese, auto-synced from GitHub.">From a 5-minute quickstart and runnable algorithm notebooks to serving guides and API references—all in English &amp; Chinese, auto-synced from GitHub.</p>
    </div>
    <a class="zine-pill-link primary" data-i18n-zh="打开文档总导航 →" data-i18n-en="Docs Roadmap →" href="{{ '/pllm/docs/' | relative_url }}">Docs Roadmap →</a>
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
      <h2 data-i18n-zh="为什么要做 pLLM，以及底层是怎么跑通的" data-i18n-en="Why &amp; How We Built It: Essays and Engineering Stories">Why &amp; How We Built It: Essays and Engineering Stories</h2>
      <p class="zine-section-sub" data-i18n-zh="共 {{ pllm_posts.size }} 篇双语文章：既有“为什么推理创新不该靠拉分支”的行业与架构思考，也有把连续批处理掰开揉碎讲清楚的机制拆解与真实踩坑复盘。" data-i18n-en="{{ pllm_posts.size }} standalone bilingual reads—covering why inference research hits an engine wall, how continuous batching works inside, and real compiler/GPU debugging stories.">{{ pllm_posts.size }} standalone bilingual reads—covering why inference research hits an engine wall, how continuous batching works inside, and real compiler/GPU debugging stories.</p>
    </div>
    <a class="zine-pill-link" data-i18n-zh="全站博客分类目录 →" data-i18n-en="All Blog Categories →" href="{{ '/blog/' | relative_url }}">All Blog Categories →</a>
  </div>

  <div class="pllm-filter-bar" role="group" aria-label="Filter blog tracks">
    <button type="button" class="pllm-filter-btn is-active" data-pllm-blog-filter="all" aria-pressed="true" data-i18n-zh="全部文章 ({{ pllm_posts.size }})" data-i18n-en="All Articles ({{ pllm_posts.size }})">All Articles ({{ pllm_posts.size }})</button>
    <button type="button" class="pllm-filter-btn" data-pllm-blog-filter="systems-architecture" aria-pressed="false" data-i18n-zh="行业思考与架构设计 ({{ arch_posts.size }})" data-i18n-en="Why &amp; Architecture ({{ arch_posts.size }})">Why &amp; Architecture ({{ arch_posts.size }})</button>
    <button type="button" class="pllm-filter-btn" data-pllm-blog-filter="engine-internals" aria-pressed="false" data-i18n-zh="推理引擎底层拆解 ({{ intern_posts.size }})" data-i18n-en="How Engines Work Inside ({{ intern_posts.size }})">How Engines Work Inside ({{ intern_posts.size }})</button>
    <button type="button" class="pllm-filter-btn" data-pllm-blog-filter="debugging-notes" aria-pressed="false" data-i18n-zh="真实编译与 GPU 踩坑记 ({{ debug_posts.size }})" data-i18n-en="Real Debugging Stories ({{ debug_posts.size }})">Real Debugging Stories ({{ debug_posts.size }})</button>
  </div>

  <div class="pllm-blog-cat-block" data-pllm-blog-block="systems-architecture">
    <h3 class="zine-cat-heading" data-i18n-zh="一 · 为什么要做这件事：行业趋势、状态层设计与弯路复盘" data-i18n-en="Part 1 &middot; Why We Built It: Inference Economics, State Design &amp; Lessons Learned">Part 1 &middot; Why We Built It: Inference Economics, State Design &amp; Lessons Learned</h3>
    <div class="zine-lang-panel" data-lang-panel="en" lang="en">
      <figure class="pllm-hero-figure">
        <img src="{{ '/assets/img/pllm/blog/assets/system-anatomy-en.svg' | relative_url | bust_file_cache }}" alt="Visual Guide to the 11-Part Inference Engine Series">
        <figcaption class="pllm-fig-caption">Visual Reading Map &middot; Where each standalone article sits inside a modern LLM serving engine.</figcaption>
      </figure>
    </div>
    <div class="zine-lang-panel is-hidden" data-lang-panel="zh" lang="zh-CN">
      <figure class="pllm-hero-figure">
        <img src="{{ '/assets/img/pllm/blog/assets/system-anatomy-zh.svg' | relative_url | bust_file_cache }}" alt="十一篇推理系统文章在引擎中的对应位置">
        <figcaption class="pllm-fig-caption">阅读路标 · 每篇独立文章对应现代推理引擎里的一个真实痛点与核心模块。</figcaption>
      </figure>
    </div>
    <div class="pllm-blog-grid">
      {% for post in arch_posts %}
      {% assign title_zh = post.title_zh | default: post.title %}
      {% assign title_en = post.title_en | default: post.title %}
      {% assign desc_zh = post.description_zh | default: post.description %}
      {% assign desc_en = post.description_en | default: post.description %}
      {% assign badge_zh = post.pllm_badge_zh | default: "架构与复盘" %}
      {% assign badge_en = post.pllm_badge_en | default: "Architecture" %}
      {% assign code_lbl = post.pllm_code | default: "#01" %}
      {% assign read_time = post.content | number_of_words | divided_by: 360 | plus: 1 %}
      <a class="pllm-blog-card" href="{{ post.url | relative_url }}">
        <div class="zine-series-meta">
          <span class="zine-series-number">{{ code_lbl }}</span>
          <span class="zine-lang-badge" data-i18n-zh="{{ badge_zh | escape }}" data-i18n-en="{{ badge_en | escape }}">{{ badge_en }}</span>
          <span class="post-meta">{{ read_time }} min</span>
        </div>
        <div class="pllm-blog-card-title" data-i18n-zh="{{ title_zh | escape }}" data-i18n-en="{{ title_en | escape }}">{{ title_en }}</div>
        <p class="pllm-blog-card-desc" data-i18n-zh="{{ desc_zh | escape }}" data-i18n-en="{{ desc_en | escape }}">{{ desc_en }}</p>
      </a>
      {% endfor %}
    </div>
  </div>

  <div class="pllm-blog-cat-block" data-pllm-blog-block="engine-internals">
    <h3 class="zine-cat-heading" data-i18n-zh="二 · 把现代推理引擎拆开看：带宽账、静态图与并发隔离" data-i18n-en="Part 2 &middot; How Modern Serving Engines Actually Work Under the Hood">Part 2 &middot; How Modern Serving Engines Actually Work Under the Hood</h3>
    <div class="pllm-blog-grid">
      {% for post in intern_posts %}
      {% assign title_zh = post.title_zh | default: post.title %}
      {% assign title_en = post.title_en | default: post.title %}
      {% assign desc_zh = post.description_zh | default: post.description %}
      {% assign desc_en = post.description_en | default: post.description %}
      {% assign badge_zh = post.pllm_badge_zh | default: "机制拆解" %}
      {% assign badge_en = post.pllm_badge_en | default: "Deep Dive" %}
      {% assign code_lbl = post.pllm_code | default: "#03" %}
      {% assign read_time = post.content | number_of_words | divided_by: 360 | plus: 1 %}
      <a class="pllm-blog-card" href="{{ post.url | relative_url }}">
        <div class="zine-series-meta">
          <span class="zine-series-number">{{ code_lbl }}</span>
          <span class="zine-lang-badge" data-i18n-zh="{{ badge_zh | escape }}" data-i18n-en="{{ badge_en | escape }}">{{ badge_en }}</span>
          <span class="post-meta">{{ read_time }} min</span>
        </div>
        <div class="pllm-blog-card-title" data-i18n-zh="{{ title_zh | escape }}" data-i18n-en="{{ title_en | escape }}">{{ title_en }}</div>
        <p class="pllm-blog-card-desc" data-i18n-zh="{{ desc_zh | escape }}" data-i18n-en="{{ desc_en | escape }}">{{ desc_en }}</p>
      </a>
      {% endfor %}
    </div>
  </div>

  <div class="pllm-blog-cat-block" data-pllm-blog-block="debugging-notes">
    <h3 class="zine-cat-heading" data-i18n-zh="三 · 真实踩坑与排障实录：当编译器、半精度浮点遇上并发状态" data-i18n-en="Part 3 &middot; Real Debugging Stories: Compilers, Floating-Point Traps &amp; Live State">Part 3 &middot; Real Debugging Stories: Compilers, Floating-Point Traps &amp; Live State</h3>
    <div class="zine-lang-panel" data-lang-panel="en" lang="en">
      <figure class="pllm-hero-figure">
        <img src="{{ '/assets/img/pllm/blog/assets/personal-roadmap-en.svg' | relative_url | bust_file_cache }}" alt="Debugging Stories Overview Map">
        <figcaption class="pllm-fig-caption">Four Real Bugs, Four Layers of the Stack &middot; From scheduler timing and CUDA Graph memory to FP16 rounding and torch.compile.</figcaption>
      </figure>
    </div>
    <div class="zine-lang-panel is-hidden" data-lang-panel="zh" lang="zh-CN">
      <figure class="pllm-hero-figure">
        <img src="{{ '/assets/img/pllm/blog/assets/personal-roadmap-zh.svg' | relative_url | bust_file_cache }}" alt="四篇真实排障手记在技术栈中的对应位置">
        <figcaption class="pllm-fig-caption">四个真实诡异 Bug · 分别发生在多步请求调度、CUDA Graph 显存、半精度浮点舍入与 torch.compile 编译阶段。</figcaption>
      </figure>
    </div>
    <div class="pllm-blog-grid">
      {% for post in debug_posts %}
      {% assign title_zh = post.title_zh | default: post.title %}
      {% assign title_en = post.title_en | default: post.title %}
      {% assign desc_zh = post.description_zh | default: post.description %}
      {% assign desc_en = post.description_en | default: post.description %}
      {% assign badge_zh = post.pllm_badge_zh | default: "排障实录" %}
      {% assign badge_en = post.pllm_badge_en | default: "Debugging" %}
      {% assign code_lbl = post.pllm_code | default: "N01" %}
      {% assign read_time = post.content | number_of_words | divided_by: 360 | plus: 1 %}
      <a class="pllm-blog-card" href="{{ post.url | relative_url }}">
        <div class="zine-series-meta">
          <span class="zine-series-number">{{ code_lbl }}</span>
          <span class="zine-lang-badge" data-i18n-zh="{{ badge_zh | escape }}" data-i18n-en="{{ badge_en | escape }}">{{ badge_en }}</span>
          <span class="post-meta">{{ read_time }} min</span>
        </div>
        <div class="pllm-blog-card-title" data-i18n-zh="{{ title_zh | escape }}" data-i18n-en="{{ title_en | escape }}">{{ title_en }}</div>
        <p class="pllm-blog-card-desc" data-i18n-zh="{{ desc_zh | escape }}" data-i18n-en="{{ desc_en | escape }}">{{ desc_en }}</p>
      </a>
      {% endfor %}
    </div>
  </div>
</section>

<div id="citation" class="zine-hub-section" markdown="1">

<h2 data-i18n-zh="论文引用（Citation）" data-i18n-en="Citation">Citation</h2>

<p data-i18n-zh="如果 pLLM 帮你在研究中省去了改引擎的苦力活，欢迎引用我们的 ICML 2026 论文（ESamp）：" data-i18n-en="If pLLM saves you from maintaining a vLLM fork in your research or systems work, please cite our ICML 2026 paper (ESamp):">If pLLM saves you from maintaining a vLLM fork in your research or systems work, please cite our ICML 2026 paper (ESamp):</p>

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

<script defer src="{{ '/assets/js/bilingual-post.js' | relative_url | bust_file_cache }}"></script>
<script defer src="{{ '/assets/js/pllm-hub.js' | relative_url | bust_file_cache }}"></script>
