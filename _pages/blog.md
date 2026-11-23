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
{% assign research_posts = site.posts | where: "blog_track", "research" %}
{% assign systems_posts = site.posts | where: "blog_track", "systems" | sort: "pllm_order" %}
{% assign eng_posts = site.posts | where: "blog_track", "engineering-notes" | sort: "pllm_order" %}

<link rel="stylesheet" href="{{ '/assets/css/pllm-hub.css' | relative_url | bust_file_cache }}">

<div class="post">

<header class="zine-masthead">
  <div class="zine-kicker" data-i18n-zh="随笔、算法研究、系统与工程笔记" data-i18n-en="Essays, Research, Systems &amp; Engineering Notes">Essays, Research, Systems &amp; Engineering Notes</div>
  <h1 class="post-title" data-i18n-zh="博客" data-i18n-en="Blog">Blog</h1>
  <p class="zine-deck" data-i18n-zh="关于算法、机器学习系统与 AI 行业的思考随笔、研究笔记与一线工程记录（共 {{ all_posts.size }} 篇，支持中英切换与分类检索）。" data-i18n-en="Essays, research notes, systems deep dives, and engineering dispatches across algorithms, ML systems, and AI ({{ all_posts.size }} bilingual posts).">Essays, research notes, systems deep dives, and engineering dispatches across algorithms, ML systems, and AI ({{ all_posts.size }} bilingual posts).</p>
</header>

<div class="zine-blog-toolbar">
  <div class="pllm-filter-bar" style="margin: 0;" role="group" aria-label="Filter blog categories">
    <button type="button" class="pllm-filter-btn is-active" data-blog-cat-btn="all" aria-pressed="true" data-i18n-zh="全部 ({{ all_posts.size }})" data-i18n-en="All ({{ all_posts.size }})">All ({{ all_posts.size }})</button>
    <button type="button" class="pllm-filter-btn" data-blog-cat-btn="essays" aria-pressed="false" data-i18n-zh="随笔与思考 ({{ essay_posts.size }})" data-i18n-en="Essays ({{ essay_posts.size }})">Essays ({{ essay_posts.size }})</button>
    <button type="button" class="pllm-filter-btn" data-blog-cat-btn="research" aria-pressed="false" data-i18n-zh="算法与研究 ({{ research_posts.size }})" data-i18n-en="Algorithms &amp; Research ({{ research_posts.size }})">Algorithms &amp; Research ({{ research_posts.size }})</button>
    <button type="button" class="pllm-filter-btn" data-blog-cat-btn="systems" aria-pressed="false" data-i18n-zh="系统与架构 ({{ systems_posts.size }})" data-i18n-en="Systems &amp; Architecture ({{ systems_posts.size }})">Systems &amp; Architecture ({{ systems_posts.size }})</button>
    <button type="button" class="pllm-filter-btn" data-blog-cat-btn="engineering-notes" aria-pressed="false" data-i18n-zh="工程与排障笔记 ({{ eng_posts.size }})" data-i18n-en="Engineering &amp; Debugging ({{ eng_posts.size }})">Engineering &amp; Debugging ({{ eng_posts.size }})</button>
  </div>

  <div class="zine-blog-controls-row">
    <div class="zine-blog-search-wrap">
      <input
        type="search"
        class="zine-blog-search"
        data-blog-search
        aria-label="Search blog posts"
        placeholder="Search titles, topics, or keywords..."
        data-placeholder-zh="搜索标题、主题或关键词…"
        data-placeholder-en="Search titles, topics, or keywords..."
      >
    </div>
    <div class="zine-blog-view-group" role="group" aria-label="View mode">
      <button type="button" class="pllm-tab-btn is-active" data-blog-view-btn="grouped" aria-pressed="true" data-i18n-zh="按分类浏览" data-i18n-en="By Category">By Category</button>
      <button type="button" class="pllm-tab-btn" data-blog-view-btn="chrono" aria-pressed="false" data-i18n-zh="按时间排序" data-i18n-en="Chronological">Chronological</button>
    </div>
  </div>
</div>

<div id="blog-empty-notice" class="zine-blog-empty is-hidden" data-i18n-zh="未找到匹配当前筛选或关键词的文章。" data-i18n-en="No articles match the current category or search query.">No articles match the current category or search query.</div>

<div id="blog-view-grouped">

  <section class="zine-blog-section" data-blog-sec="essays">
    <div class="zine-blog-sec-header">
      <h2 class="zine-blog-sec-title" data-i18n-zh="随笔与思考" data-i18n-en="Essays &amp; Perspectives">Essays &amp; Perspectives</h2>
      <p class="zine-blog-sec-desc" data-i18n-zh="关于技术浪潮、研究品味与行业观察的独立随笔" data-i18n-en="Personal essays on technology shifts, research taste, and industry observations">Personal essays on technology shifts, research taste, and industry observations</p>
    </div>
    <ul class="post-list">
      {% for post in essay_posts %}
      {% assign title_zh = post.title_zh | default: post.title %}
      {% assign title_en = post.title_en | default: post.title %}
      {% assign desc_zh = post.description_zh | default: post.description %}
      {% assign desc_en = post.description_en | default: post.description %}
      {% assign b_zh = post.blog_badge_zh | default: "随笔" %}
      {% assign b_en = post.blog_badge_en | default: "Essay" %}
      {% assign read_time = post.content | number_of_words | divided_by: 280 | plus: 1 %}
      <li class="zine-blog-item" data-post-cat="essays" data-post-search="{{ title_zh | escape }} {{ title_en | escape }} {{ desc_zh | escape }} {{ desc_en | escape }} {{ post.tags | join: ' ' }}">
        <div class="zine-blog-meta-top">
          <span class="zine-cat-pill" data-cat="essays" data-i18n-zh="随笔与思考" data-i18n-en="Essays">Essays</span>
          <span class="zine-lang-badge" data-i18n-zh="{{ b_zh | escape }}" data-i18n-en="{{ b_en | escape }}">{{ b_en }}</span>
          <span class="post-meta">{{ read_time }} min &middot; {{ post.date | date: "%Y-%m-%d" }}</span>
        </div>
        <h3><a class="post-title" data-i18n-zh="{{ title_zh | escape }}" data-i18n-en="{{ title_en | escape }}" href="{{ post.url | relative_url }}">{{ post.title }}</a></h3>
        <p data-i18n-zh="{{ desc_zh | escape }}" data-i18n-en="{{ desc_en | escape }}">{{ post.description }}</p>
      </li>
      {% endfor %}
    </ul>
  </section>

  <section class="zine-blog-section" data-blog-sec="research">
    <div class="zine-blog-sec-header">
      <h2 class="zine-blog-sec-title" data-i18n-zh="算法与研究" data-i18n-en="Algorithms &amp; Research">Algorithms &amp; Research</h2>
      <p class="zine-blog-sec-desc" data-i18n-zh="围绕大模型、智能体、强化学习与评估方法的论文解读与研究笔记" data-i18n-en="Paper dispatches and research notes on LLMs, agents, learning, and evaluation">Paper dispatches and research notes on LLMs, agents, learning, and evaluation</p>
    </div>
    <ul class="post-list">
      {% for post in research_posts %}
      {% assign title_zh = post.title_zh | default: post.title %}
      {% assign title_en = post.title_en | default: post.title %}
      {% assign desc_zh = post.description_zh | default: post.description %}
      {% assign desc_en = post.description_en | default: post.description %}
      {% assign b_zh = post.blog_badge_zh | default: "研究笔记" %}
      {% assign b_en = post.blog_badge_en | default: "Research" %}
      {% assign read_time = post.content | number_of_words | divided_by: 280 | plus: 1 %}
      {% if post.redirect == blank %}{% assign post_href = post.url | relative_url %}{% else %}{% assign post_href = post.redirect | relative_url %}{% endif %}
      <li class="zine-blog-item" data-post-cat="research" data-post-search="{{ title_zh | escape }} {{ title_en | escape }} {{ desc_zh | escape }} {{ desc_en | escape }} {{ post.tags | join: ' ' }}">
        {% if post.thumbnail %}
        <div class="row">
          <div class="col-sm-9">
        {% endif %}
            <div class="zine-blog-meta-top">
              <span class="zine-cat-pill" data-cat="research" data-i18n-zh="算法与研究" data-i18n-en="Algorithms &amp; Research">Algorithms &amp; Research</span>
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

  <section class="zine-blog-section" data-blog-sec="systems">
    <div class="zine-blog-sec-header">
      <h2 class="zine-blog-sec-title" data-i18n-zh="系统与架构" data-i18n-en="Systems &amp; Architecture">Systems &amp; Architecture</h2>
      <p class="zine-blog-sec-desc" data-i18n-zh="机器学习系统设计、软硬件协同、底层机制拆解与架构复盘" data-i18n-en="ML systems design, hardware-software codesign, internals deep dives, and architecture postmortems">ML systems design, hardware-software codesign, internals deep dives, and architecture postmortems</p>
    </div>
    <ul class="post-list">
      {% for post in systems_posts %}
      {% assign title_zh = post.title_zh | default: post.title %}
      {% assign title_en = post.title_en | default: post.title %}
      {% assign desc_zh = post.description_zh | default: post.description %}
      {% assign desc_en = post.description_en | default: post.description %}
      {% assign b_zh = post.pllm_badge_zh | default: "系统与架构" %}
      {% assign b_en = post.pllm_badge_en | default: "Systems" %}
      {% assign read_time = post.content | number_of_words | divided_by: 360 | plus: 1 %}
      <li class="zine-blog-item" data-post-cat="systems" data-post-search="{{ title_zh | escape }} {{ title_en | escape }} {{ desc_zh | escape }} {{ desc_en | escape }} {{ post.tags | join: ' ' }} {{ post.pllm_code }} pLLM">
        <div class="zine-blog-meta-top">
          <span class="zine-cat-pill" data-cat="systems" data-i18n-zh="系统与架构" data-i18n-en="Systems &amp; Architecture">Systems &amp; Architecture</span>
          {% if post.blog_series_en %}<span class="zine-series-number">{{ post.blog_series_en }} {{ post.pllm_code }}</span>{% endif %}
          <span class="zine-lang-badge" data-i18n-zh="{{ b_zh | escape }}" data-i18n-en="{{ b_en | escape }}">{{ b_en }}</span>
          <span class="post-meta">{{ read_time }} min &middot; {{ post.date | date: "%Y-%m-%d" }}</span>
        </div>
        <h3><a class="post-title" data-i18n-zh="{{ title_zh | escape }}" data-i18n-en="{{ title_en | escape }}" href="{{ post.url | relative_url }}">{{ post.title }}</a></h3>
        <p data-i18n-zh="{{ desc_zh | escape }}" data-i18n-en="{{ desc_en | escape }}">{{ post.description }}</p>
      </li>
      {% endfor %}
    </ul>
  </section>

  <section class="zine-blog-section" data-blog-sec="engineering-notes">
    <div class="zine-blog-sec-header">
      <h2 class="zine-blog-sec-title" data-i18n-zh="工程与排障笔记" data-i18n-en="Engineering &amp; Debugging">Engineering &amp; Debugging</h2>
      <p class="zine-blog-sec-desc" data-i18n-zh="一线开发中的疑难 Bug 排查、性能调优与底层工具链踩坑记录" data-i18n-en="Field notes from tracking down subtle bugs, profiling bottlenecks, and wrestling with toolchains">Field notes from tracking down subtle bugs, profiling bottlenecks, and wrestling with toolchains</p>
    </div>
    <ul class="post-list">
      {% for post in eng_posts %}
      {% assign title_zh = post.title_zh | default: post.title %}
      {% assign title_en = post.title_en | default: post.title %}
      {% assign desc_zh = post.description_zh | default: post.description %}
      {% assign desc_en = post.description_en | default: post.description %}
      {% assign b_zh = post.pllm_badge_zh | default: "排障笔记" %}
      {% assign b_en = post.pllm_badge_en | default: "Debugging" %}
      {% assign read_time = post.content | number_of_words | divided_by: 360 | plus: 1 %}
      <li class="zine-blog-item" data-post-cat="engineering-notes" data-post-search="{{ title_zh | escape }} {{ title_en | escape }} {{ desc_zh | escape }} {{ desc_en | escape }} {{ post.tags | join: ' ' }} {{ post.pllm_code }} pLLM">
        <div class="zine-blog-meta-top">
          <span class="zine-cat-pill" data-cat="engineering-notes" data-i18n-zh="工程与排障笔记" data-i18n-en="Engineering &amp; Debugging">Engineering &amp; Debugging</span>
          {% if post.blog_series_en %}<span class="zine-series-number">{{ post.blog_series_en }} {{ post.pllm_code }}</span>{% endif %}
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
      {% when "essays" %}{% assign c_zh = "随笔与思考" %}{% assign c_en = "Essays" %}
      {% when "research" %}{% assign c_zh = "算法与研究" %}{% assign c_en = "Algorithms & Research" %}
      {% when "systems" %}{% assign c_zh = "系统与架构" %}{% assign c_en = "Systems & Architecture" %}
      {% when "engineering-notes" %}{% assign c_zh = "工程与排障笔记" %}{% assign c_en = "Engineering & Debugging" %}
      {% else %}{% assign c_zh = "博客" %}{% assign c_en = "Blog" %}
    {% endcase %}
    <li class="zine-blog-item" data-post-cat="{{ cat_id }}" data-post-search="{{ title_zh | escape }} {{ title_en | escape }} {{ desc_zh | escape }} {{ desc_en | escape }} {{ post.tags | join: ' ' }} {{ post.pllm_code }}">
      <div class="zine-blog-meta-top">
        <span class="zine-cat-pill" data-cat="{{ cat_id }}" data-i18n-zh="{{ c_zh }}" data-i18n-en="{{ c_en }}">{{ c_en }}</span>
        {% if post.blog_series_en %}<span class="zine-series-number">{{ post.blog_series_en }} {{ post.pllm_code }}</span>{% endif %}
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
