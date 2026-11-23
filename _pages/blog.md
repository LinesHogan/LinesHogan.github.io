---
layout: default
permalink: /blog/
title: blog
nav: true
nav_order: 1
pagination:
  enabled: false
---

{% assign all_posts = site.posts %}
{% assign essay_posts = site.posts | where: "blog_track", "essays" %}
{% assign agent_posts = site.posts | where: "blog_track", "agent-research" %}
{% assign arch_posts = site.posts | where: "blog_track", "systems-architecture" | sort: "pllm_order" %}
{% assign intern_posts = site.posts | where: "blog_track", "engine-internals" | sort: "pllm_order" %}
{% assign debug_posts = site.posts | where: "blog_track", "debugging-notes" | sort: "pllm_order" %}

<link rel="stylesheet" href="{{ '/assets/css/pllm-hub.css' | relative_url | bust_file_cache }}">

<div class="post">

<header class="zine-masthead">
  <div class="zine-kicker" data-i18n-zh="研究专题、推理系统与随笔 · 分类目录" data-i18n-en="Research, Inference Systems &amp; Essays &middot; Curated Index">Research, Inference Systems &amp; Essays &middot; Curated Index</div>
  <h1 class="post-title" data-i18n-zh="博客与技术专栏" data-i18n-en="Blog &amp; Technical Anthology">Blog &amp; Technical Anthology</h1>
  <p class="zine-deck" data-i18n-zh="收录 {{ all_posts.size }} 篇双语文章，按“观点随笔”、“Agent 与评估研究”、“推理架构与状态层”、“引擎底层机制”及“编译与排障手记”五大专题归档索引。" data-i18n-en="{{ all_posts.size }} bilingual essays, research dispatches, systems deep dives, and compiler/GPU debugging postmortems—organized by topic for fast navigation.">{{ all_posts.size }} bilingual essays, research dispatches, systems deep dives, and compiler/GPU debugging postmortems—organized by topic for fast navigation.</p>
</header>

<div class="zine-blog-toolbar">
  <div class="pllm-filter-bar" style="margin: 0;" role="group" aria-label="Filter blog categories">
    <button type="button" class="pllm-filter-btn is-active" data-blog-cat-btn="all" aria-pressed="true" data-i18n-zh="全部文章 ({{ all_posts.size }})" data-i18n-en="All Posts ({{ all_posts.size }})">All Posts ({{ all_posts.size }})</button>
    <button type="button" class="pllm-filter-btn" data-blog-cat-btn="essays" aria-pressed="false" data-i18n-zh="观点与随笔 ({{ essay_posts.size }})" data-i18n-en="Essays ({{ essay_posts.size }})">Essays ({{ essay_posts.size }})</button>
    <button type="button" class="pllm-filter-btn" data-blog-cat-btn="agent-research" aria-pressed="false" data-i18n-zh="Agent 与评估研究 ({{ agent_posts.size }})" data-i18n-en="Agent Research ({{ agent_posts.size }})">Agent Research ({{ agent_posts.size }})</button>
    <button type="button" class="pllm-filter-btn" data-blog-cat-btn="systems-architecture" aria-pressed="false" data-i18n-zh="推理架构与思辨 ({{ arch_posts.size }})" data-i18n-en="Systems Architecture ({{ arch_posts.size }})">Systems Architecture ({{ arch_posts.size }})</button>
    <button type="button" class="pllm-filter-btn" data-blog-cat-btn="engine-internals" aria-pressed="false" data-i18n-zh="引擎底层机制 ({{ intern_posts.size }})" data-i18n-en="Engine Internals ({{ intern_posts.size }})">Engine Internals ({{ intern_posts.size }})</button>
    <button type="button" class="pllm-filter-btn" data-blog-cat-btn="debugging-notes" aria-pressed="false" data-i18n-zh="编译与排障手记 ({{ debug_posts.size }})" data-i18n-en="Debugging Notes ({{ debug_posts.size }})">Debugging Notes ({{ debug_posts.size }})</button>
  </div>

  <div class="zine-blog-controls-row">
    <div class="zine-blog-search-wrap">
      <input
        type="search"
        class="zine-blog-search"
        data-blog-search
        aria-label="Search blog posts"
        placeholder="Search titles, topics, or keywords..."
        data-placeholder-zh="搜索文章标题、技术关键词或标签…"
        data-placeholder-en="Search titles, topics, or keywords..."
      >
    </div>
    <div class="zine-blog-view-group" role="group" aria-label="View mode">
      <button type="button" class="pllm-tab-btn is-active" data-blog-view-btn="grouped" aria-pressed="true" data-i18n-zh="按专题分组" data-i18n-en="Grouped by Topic">Grouped by Topic</button>
      <button type="button" class="pllm-tab-btn" data-blog-view-btn="chrono" aria-pressed="false" data-i18n-zh="按时间排序" data-i18n-en="Chronological">Chronological</button>
    </div>
  </div>
</div>

<div id="blog-empty-notice" class="zine-blog-empty is-hidden" data-i18n-zh="未找到匹配当前筛选或关键词的文章。" data-i18n-en="No articles match the current category or search query.">No articles match the current category or search query.</div>

<div id="blog-view-grouped">

  <section class="zine-blog-section" data-blog-sec="essays">
    <div class="zine-blog-sec-header">
      <h2 class="zine-blog-sec-title" data-i18n-zh="观点与随笔" data-i18n-en="Essays &amp; Perspectives">Essays &amp; Perspectives</h2>
      <p class="zine-blog-sec-desc" data-i18n-zh="关于 AI 浪潮、研究品味与实事求是的思考" data-i18n-en="Reflections on the AI wave, research taste, and intellectual honesty">Reflections on the AI wave, research taste, and intellectual honesty</p>
    </div>
    <ul class="post-list">
      {% for post in essay_posts %}
      {% assign title_zh = post.title_zh | default: post.title %}
      {% assign title_en = post.title_en | default: post.title %}
      {% assign desc_zh = post.description_zh | default: post.description %}
      {% assign desc_en = post.description_en | default: post.description %}
      {% assign b_zh = post.blog_badge_zh | default: "观点随笔" %}
      {% assign b_en = post.blog_badge_en | default: "Essay" %}
      {% assign read_time = post.content | number_of_words | divided_by: 280 | plus: 1 %}
      <li class="zine-blog-item" data-post-cat="essays" data-post-search="{{ title_zh | escape }} {{ title_en | escape }} {{ desc_zh | escape }} {{ desc_en | escape }} {{ post.tags | join: ' ' }}">
        <div class="zine-blog-meta-top">
          <span class="zine-cat-pill" data-cat="essays" data-i18n-zh="观点与随笔" data-i18n-en="Essays">Essays</span>
          <span class="zine-lang-badge" data-i18n-zh="{{ b_zh | escape }}" data-i18n-en="{{ b_en | escape }}">{{ b_en }}</span>
          <span class="post-meta">{{ read_time }} min &middot; {{ post.date | date: "%Y-%m-%d" }}</span>
        </div>
        <h3><a class="post-title" data-i18n-zh="{{ title_zh | escape }}" data-i18n-en="{{ title_en | escape }}" href="{{ post.url | relative_url }}">{{ post.title }}</a></h3>
        <p data-i18n-zh="{{ desc_zh | escape }}" data-i18n-en="{{ desc_en | escape }}">{{ post.description }}</p>
      </li>
      {% endfor %}
    </ul>
  </section>

  <section class="zine-blog-section" data-blog-sec="agent-research">
    <div class="zine-blog-sec-header">
      <h2 class="zine-blog-sec-title" data-i18n-zh="Agent 自进化与评估研究（分辨率假说专题）" data-i18n-en="Agent Evolution &amp; Evaluation Research (The Resolution Hypothesis)">Agent Evolution &amp; Evaluation Research (The Resolution Hypothesis)</h2>
      <p class="zine-blog-sec-desc" data-i18n-zh="超千万次调用实证拆解：为什么冻结权重 Agent 进化卡在评估分辨率" data-i18n-en="Auditing >10M LLM calls: why frozen-weight agent evolution stalls on evaluation resolution">Auditing &gt;10M LLM calls: why frozen-weight agent evolution stalls on evaluation resolution</p>
    </div>
    <ul class="post-list">
      {% for post in agent_posts %}
      {% assign title_zh = post.title_zh | default: post.title %}
      {% assign title_en = post.title_en | default: post.title %}
      {% assign desc_zh = post.description_zh | default: post.description %}
      {% assign desc_en = post.description_en | default: post.description %}
      {% assign b_zh = post.blog_badge_zh | default: "研究专题" %}
      {% assign b_en = post.blog_badge_en | default: "Research" %}
      {% assign read_time = post.content | number_of_words | divided_by: 280 | plus: 1 %}
      {% if post.redirect == blank %}{% assign post_href = post.url | relative_url %}{% else %}{% assign post_href = post.redirect | relative_url %}{% endif %}
      <li class="zine-blog-item" data-post-cat="agent-research" data-post-search="{{ title_zh | escape }} {{ title_en | escape }} {{ desc_zh | escape }} {{ desc_en | escape }} {{ post.tags | join: ' ' }}">
        {% if post.thumbnail %}
        <div class="row">
          <div class="col-sm-9">
        {% endif %}
            <div class="zine-blog-meta-top">
              <span class="zine-cat-pill" data-cat="agent-research" data-i18n-zh="Agent 与评估研究" data-i18n-en="Agent Research">Agent Research</span>
              <span class="zine-lang-badge" data-i18n-zh="{{ b_zh | escape }}" data-i18n-en="{{ b_en | escape }}">{{ b_en }}</span>
              <span class="post-meta">{{ read_time }} min &middot; {{ post.date | date: "%Y-%m-%d" }}</span>
            </div>
            <h3><a class="post-title" data-i18n-zh="{{ title_zh | escape }}" data-i18n-en="{{ title_en | escape }}" href="{{ post_href }}">{{ post.title }}</a></h3>
            <p data-i18n-zh="{{ desc_zh | escape }}" data-i18n-en="{{ desc_en | escape }}">{{ post.description }}</p>
        {% if post.thumbnail %}
          </div>
          <div class="col-sm-3">
            <img class="card-img" src="{{ post.thumbnail | relative_url }}" style="object-fit: cover; height: 90%" alt="thumbnail">
          </div>
        </div>
        {% endif %}
      </li>
      {% endfor %}
    </ul>
  </section>

  <section class="zine-blog-section" data-blog-sec="systems-architecture">
    <div class="zine-blog-sec-header">
      <h2 class="zine-blog-sec-title" data-i18n-zh="推理系统架构、趋势思辨与复盘（pLLM 专栏）" data-i18n-en="Inference Economics, PTTS Architecture &amp; Postmortems (pLLM)">Inference Economics, PTTS Architecture &amp; Postmortems (pLLM)</h2>
      <p class="zine-blog-sec-desc" data-i18n-zh="推理经济账、统一分页张量状态层（PTTS）与编译器架构取舍" data-i18n-en="Post-Bitter-Lesson inference economics, unified tensor state layer (PTTS), and scheduler postmortems">Post-Bitter-Lesson inference economics, unified tensor state layer (PTTS), and scheduler postmortems</p>
    </div>
    <ul class="post-list">
      {% for post in arch_posts %}
      {% assign title_zh = post.title_zh | default: post.title %}
      {% assign title_en = post.title_en | default: post.title %}
      {% assign desc_zh = post.description_zh | default: post.description %}
      {% assign desc_en = post.description_en | default: post.description %}
      {% assign b_zh = post.pllm_badge_zh | default: "架构与复盘" %}
      {% assign b_en = post.pllm_badge_en | default: "Architecture" %}
      {% assign read_time = post.content | number_of_words | divided_by: 360 | plus: 1 %}
      <li class="zine-blog-item" data-post-cat="systems-architecture" data-post-search="{{ title_zh | escape }} {{ title_en | escape }} {{ desc_zh | escape }} {{ desc_en | escape }} {{ post.tags | join: ' ' }} {{ post.pllm_code }}">
        <div class="zine-blog-meta-top">
          <span class="zine-cat-pill" data-cat="systems-architecture" data-i18n-zh="推理架构与思辨" data-i18n-en="Systems Architecture">Systems Architecture</span>
          <span class="zine-series-number">{{ post.pllm_code }}</span>
          <span class="zine-lang-badge" data-i18n-zh="{{ b_zh | escape }}" data-i18n-en="{{ b_en | escape }}">{{ b_en }}</span>
          <span class="post-meta">{{ read_time }} min &middot; {{ post.date | date: "%Y-%m-%d" }}</span>
        </div>
        <h3><a class="post-title" data-i18n-zh="{{ title_zh | escape }}" data-i18n-en="{{ title_en | escape }}" href="{{ post.url | relative_url }}">{{ post.title }}</a></h3>
        <p data-i18n-zh="{{ desc_zh | escape }}" data-i18n-en="{{ desc_en | escape }}">{{ post.description }}</p>
      </li>
      {% endfor %}
    </ul>
  </section>

  <section class="zine-blog-section" data-blog-sec="engine-internals">
    <div class="zine-blog-sec-header">
      <h2 class="zine-blog-sec-title" data-i18n-zh="连续批处理推理引擎底层机制拆解（pLLM 专栏）" data-i18n-en="Continuous-Batching Engine Internals Deep Dives (pLLM)">Continuous-Batching Engine Internals Deep Dives (pLLM)</h2>
      <p class="zine-blog-sec-desc" data-i18n-zh="带宽配额、IEEE 754 位变换、编译态快权重、静态图指针寻址、GPU MVCC、GQA/MLA 与多租户核验" data-i18n-en="Bandwidth quotas, IEEE 754 bit tricks, compiled fast weights, frozen CUDA Graph pointers, GPU MVCC, GQA/MLA, and verification">Bandwidth quotas, IEEE 754 bit tricks, compiled fast weights, frozen CUDA Graph pointers, GPU MVCC, GQA/MLA, and verification</p>
    </div>
    <ul class="post-list">
      {% for post in intern_posts %}
      {% assign title_zh = post.title_zh | default: post.title %}
      {% assign title_en = post.title_en | default: post.title %}
      {% assign desc_zh = post.description_zh | default: post.description %}
      {% assign desc_en = post.description_en | default: post.description %}
      {% assign b_zh = post.pllm_badge_zh | default: "机制拆解" %}
      {% assign b_en = post.pllm_badge_en | default: "Deep Dive" %}
      {% assign read_time = post.content | number_of_words | divided_by: 360 | plus: 1 %}
      <li class="zine-blog-item" data-post-cat="engine-internals" data-post-search="{{ title_zh | escape }} {{ title_en | escape }} {{ desc_zh | escape }} {{ desc_en | escape }} {{ post.tags | join: ' ' }} {{ post.pllm_code }}">
        <div class="zine-blog-meta-top">
          <span class="zine-cat-pill" data-cat="engine-internals" data-i18n-zh="引擎底层机制" data-i18n-en="Engine Internals">Engine Internals</span>
          <span class="zine-series-number">{{ post.pllm_code }}</span>
          <span class="zine-lang-badge" data-i18n-zh="{{ b_zh | escape }}" data-i18n-en="{{ b_en | escape }}">{{ b_en }}</span>
          <span class="post-meta">{{ read_time }} min &middot; {{ post.date | date: "%Y-%m-%d" }}</span>
        </div>
        <h3><a class="post-title" data-i18n-zh="{{ title_zh | escape }}" data-i18n-en="{{ title_en | escape }}" href="{{ post.url | relative_url }}">{{ post.title }}</a></h3>
        <p data-i18n-zh="{{ desc_zh | escape }}" data-i18n-en="{{ desc_en | escape }}">{{ post.description }}</p>
      </li>
      {% endfor %}
    </ul>
  </section>

  <section class="zine-blog-section" data-blog-sec="debugging-notes">
    <div class="zine-blog-sec-header">
      <h2 class="zine-blog-sec-title" data-i18n-zh="编译器、融合算子与多版本状态排障手记（LayerRoute）" data-i18n-en="Compiler, Triton Kernel &amp; MVCC Debugging Postmortems (LayerRoute)">Compiler, Triton Kernel &amp; MVCC Debugging Postmortems (LayerRoute)</h2>
      <p class="zine-blog-sec-desc" data-i18n-zh="记录 torch.compile 函数化回写覆盖、L2 早退内核、IEEE 754 半精度吸收与连续批处理生命周期排障" data-i18n-en="Real-world postmortems on PyTorch AOT functionalization, L2 early-exit Triton, FP16/BF16 rounding traps, and PTTS state pins">Real-world postmortems on PyTorch AOT functionalization, L2 early-exit Triton, FP16/BF16 rounding traps, and PTTS state pins</p>
    </div>
    <ul class="post-list">
      {% for post in debug_posts %}
      {% assign title_zh = post.title_zh | default: post.title %}
      {% assign title_en = post.title_en | default: post.title %}
      {% assign desc_zh = post.description_zh | default: post.description %}
      {% assign desc_en = post.description_en | default: post.description %}
      {% assign b_zh = post.pllm_badge_zh | default: "排障手记" %}
      {% assign b_en = post.pllm_badge_en | default: "Postmortem" %}
      {% assign read_time = post.content | number_of_words | divided_by: 360 | plus: 1 %}
      <li class="zine-blog-item" data-post-cat="debugging-notes" data-post-search="{{ title_zh | escape }} {{ title_en | escape }} {{ desc_zh | escape }} {{ desc_en | escape }} {{ post.tags | join: ' ' }} {{ post.pllm_code }}">
        <div class="zine-blog-meta-top">
          <span class="zine-cat-pill" data-cat="debugging-notes" data-i18n-zh="编译与排障手记" data-i18n-en="Debugging Notes">Debugging Notes</span>
          <span class="zine-series-number">{{ post.pllm_code }}</span>
          <span class="zine-lang-badge" data-i18n-zh="{{ b_zh | escape }}" data-i18n-en="{{ b_en | escape }}">{{ b_en }}</span>
          <span class="post-meta">{{ read_time }} min &middot; {{ post.date | date: "%Y-%m-%d" }}</span>
        </div>
        <h3><a class="post-title" data-i18n-zh="{{ title_zh | escape }}" data-i18n-en="{{ title_en | escape }}" href="{{ post.url | relative_url }}">{{ post.title }}</a></h3>
        <p data-i18n-zh="{{ desc_zh | escape }}" data-i18n-en="{{ desc_en | escape }}">{{ post.description }}</p>
      </li>
      {% endfor %}
    </ul>
  </section>

</div>

<div id="blog-view-chrono" class="is-hidden">
  <ul class="post-list">
    {% for post in all_posts %}
    {% assign title_zh = post.title_zh | default: post.title %}
    {% assign title_en = post.title_en | default: post.title %}
    {% assign desc_zh = post.description_zh | default: post.description %}
    {% assign desc_en = post.description_en | default: post.description %}
    {% assign cat_id = post.blog_track | default: "essays" %}
    {% assign b_zh = post.pllm_badge_zh | default: post.blog_badge_zh | default: "文章" %}
    {% assign b_en = post.pllm_badge_en | default: post.blog_badge_en | default: "Post" %}
    {% assign read_time = post.content | number_of_words | divided_by: 320 | plus: 1 %}
    {% if post.redirect == blank %}{% assign post_href = post.url | relative_url %}{% else %}{% assign post_href = post.redirect | relative_url %}{% endif %}
    {% case cat_id %}
      {% when "essays" %}{% assign c_zh = "观点与随笔" %}{% assign c_en = "Essays" %}
      {% when "agent-research" %}{% assign c_zh = "Agent 与评估研究" %}{% assign c_en = "Agent Research" %}
      {% when "systems-architecture" %}{% assign c_zh = "推理架构与思辨" %}{% assign c_en = "Systems Architecture" %}
      {% when "engine-internals" %}{% assign c_zh = "引擎底层机制" %}{% assign c_en = "Engine Internals" %}
      {% when "debugging-notes" %}{% assign c_zh = "编译与排障手记" %}{% assign c_en = "Debugging Notes" %}
      {% else %}{% assign c_zh = "博客" %}{% assign c_en = "Blog" %}
    {% endcase %}
    <li class="zine-blog-item" data-post-cat="{{ cat_id }}" data-post-search="{{ title_zh | escape }} {{ title_en | escape }} {{ desc_zh | escape }} {{ desc_en | escape }} {{ post.tags | join: ' ' }} {{ post.pllm_code }}">
      <div class="zine-blog-meta-top">
        <span class="zine-cat-pill" data-cat="{{ cat_id }}" data-i18n-zh="{{ c_zh }}" data-i18n-en="{{ c_en }}">{{ c_en }}</span>
        {% if post.pllm_code %}<span class="zine-series-number">{{ post.pllm_code }}</span>{% endif %}
        <span class="zine-lang-badge" data-i18n-zh="{{ b_zh | escape }}" data-i18n-en="{{ b_en | escape }}">{{ b_en }}</span>
        <span class="post-meta">{{ read_time }} min &middot; {{ post.date | date: "%Y-%m-%d" }}</span>
      </div>
      <h3><a class="post-title" data-i18n-zh="{{ title_zh | escape }}" data-i18n-en="{{ title_en | escape }}" href="{{ post_href }}">{{ post.title }}</a></h3>
      <p data-i18n-zh="{{ desc_zh | escape }}" data-i18n-en="{{ desc_en | escape }}">{{ post.description }}</p>
    </li>
    {% endfor %}
  </ul>
</div>

</div>

<script defer src="{{ '/assets/js/bilingual-post.js' | relative_url | bust_file_cache }}"></script>
<script defer src="{{ '/assets/js/pllm-hub.js' | relative_url | bust_file_cache }}"></script>
