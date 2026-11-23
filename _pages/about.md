---
layout: about
title: about
permalink: /
subtitle:

profile:
  align: right
  image: pic_profile.jpg
  image_circular: false # crops the image to make it circular
  more_info:

selected_papers: true # includes a list of papers marked as "selected={true}"
social: true # includes social icons at the bottom of the page

announcements:
  enabled: true # includes a list of news items
  scrollable: true # adds a vertical scroll bar if there are more than 3 news items
  limit: 5 # leave blank to include all the news in the `_news` folder

latest_posts:
  enabled: false
  scrollable: true # adds a vertical scroll bar if there are more than 3 new posts items
  limit: 3 # leave blank to include all the blog posts
---

<div class="zine-kicker">PhD Student &middot; ShanghaiTech &times; BIGAI &middot; ByteDance Intern</div>

I am Zeng Yuanhao (曾缘浩), a 1st-year PhD student at <a href="https://www.shanghaitech.edu.cn/en/">ShanghaiTech University</a> and <a href="https://eng.bigai.ai/">BIGAI</a>. I received my bachelor's degree from the CS Experimental Class at BUPT. I am currently interning at ByteDance.

My research focuses on algorithm-system codesign for large language model agents — I design efficient algorithms that solve real-world challenges and build systems that make them practical. I enjoy (very) smart ideas in theory, algorithms, and systems, and will share some cool insights in my blogs.

I develop <a href="{{ '/pllm/' | relative_url }}">pLLM</a> (<a href="https://github.com/LinesHogan/pLLM" target="_blank" rel="noopener">Programmable LLM</a>)—built so researchers can stop forking vLLM just to test a new inference-time idea. With a few dozen lines of Python, pLLM compiles hidden-state interventions, per-request fast weights, and online decode-time training directly into unmodified vLLM's continuous batching and CUDA graphs.

Our latest study introduces **[The Resolution Hypothesis](https://arxiv.org/abs/2611.23068)** ([project page](/resolution-hypothesis/) · [blog](/blog/2026/resolution-hypothesis-behind-the-scenes/)), a large-scale audit (>10M LLM calls, 400K CPU hours) showing that frozen-weight agent evolution stalls not from failing to propose better skills, but because low-resolution evaluations cannot tell real improvements apart from noise.

My earlier work introduces **Exploratory Sampling (ESamp)**, a decoding-time method that encourages LLMs to explore semantically diverse reasoning paths via online latent distillation (ICML 2026).

<div class="zine-bio-actions">
  <a class="zine-pill-link primary" href="{{ '/blog/' | relative_url }}">Read Blog</a>
  <a class="zine-pill-link accent" href="{{ '/pllm/' | relative_url }}">pLLM Project</a>
  <a class="zine-pill-link" href="{{ '/resolution-hypothesis/' | relative_url }}">Resolution Hypothesis</a>
  <a class="zine-pill-link" href="https://github.com/LinesHogan/pLLM" target="_blank" rel="noopener">pLLM on GitHub ↗</a>
</div>
