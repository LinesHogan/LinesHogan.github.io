---
layout: page
title: "Support Matrix"
description: "pLLM Documentation · API, Config & Support Reference"
permalink: /pllm/docs/reference/support-matrix/
nav: false
---

<link rel="stylesheet" href="{{ '/assets/css/pllm-hub.css' | relative_url | bust_file_cache }}">

<div class="zine-dispatch" data-title-zh="支持矩阵" data-title-en="Support Matrix">
  <div class="zine-dispatch-links">
    <a class="zine-pill-link primary" data-i18n-zh="← 返回 pLLM 文档中心" data-i18n-en="← pLLM Docs Hub" href="/pllm/#docs">← pLLM Docs Hub</a>
    <a class="zine-pill-link" data-i18n-zh="文档导航地图" data-i18n-en="Docs Roadmap" href="/pllm/docs/">Docs Roadmap</a>
    <span class="zine-lang-badge" data-i18n-zh="API 与配置参考手册" data-i18n-en="API, Config &amp; Support Reference">API, Config &amp; Support Reference</span>
  </div>
  <a class="zine-pill-link" data-i18n-zh="GitHub 源码 ↗" data-i18n-en="View on GitHub ↗" href="https://github.com/LinesHogan/pLLM/blob/main/doc/reference/support-matrix.md" target="_blank" rel="noopener">View on GitHub ↗</a>
</div>

<div id="markdown-content">

<div class="zine-lang-panel" data-lang-panel="en" lang="en" markdown="1">

<div id="en-support-matrix"></div>

This page lists what pLLM supports by model family, feature, vLLM version, and parallelism mode. Each restriction includes its reason; for unsupported algorithm families and why, see the [Capability Map](/pllm/docs/guides/capability-map/) and [Design Tradeoffs](/pllm/docs/explain/design-tradeoffs/), and for steps to onboard another model family, see [Using New Model Families](/pllm/docs/howto/new-model-families/).

## vLLM Versions {#en-vllm-versions}

| Item | Support | Notes |
| --- | --- | --- |
| vLLM versions | 0.27 – 0.34 | Covers `vllm>=0.27.0,<0.35`. `0.27.1` and `0.32.0` run full GPU regression suites in this repository; `0.33`–`0.34` adapt to upstream interface changes (requiring PyTorch 2.14 / CUDA 13 driver) |
| Model runner | vLLM MRv2 | pLLM selects MRv2 automatically (setting `VLLM_USE_V2_MODEL_RUNNER=1` on versions such as 0.27); MRv1 is not supported |
| Tensor parallelism | vLLM 0.32.0 and later | See [Tensor Parallelism](#en-tensor-parallelism) below |
| Pipeline & data parallelism | Not supported | Must be `1` |
| Online serving | `pllm.make_async_llm` and `pllm serve` | `ModelConfig(multiprocessing="auto")` automatically isolates `EngineCore` in a subprocess for async handles and multi-GPU runs; supports continuous admission, SSE streaming, disconnect cancellation, `pllm_route` session routing, in-flight `import_state()` / `train_batch()` / `feed()`, and no-workload operation. See [Online Serving Guide](/pllm/docs/howto/online-serving/) |

## Models and Features {#en-models-and-features}

pLLM admits models by vLLM's **concrete model class**, covering 15 decoder architecture families (16 native model classes; see [Using New Model Families](/pllm/docs/howto/new-model-families/)). Unverified subclasses are rejected at admission.

| Feature | Qwen2 / 2.5, Llama, Qwen3, Mistral, Phi-3, GLM-4, Gemma 2 / 3 | Mixtral, Qwen2-MoE, Qwen3-MoE | Qwen3.5 (Gated DeltaNet hybrid, incl. MoE) | DeepSeek V2 / V3 | gpt-oss |
| --- | --- | --- | --- | --- | --- |
| Reads & exports (`pllm.hidden`, `residual`, `final_hidden`, `layer_logits`), probes | ✓ | ✓ | ✓ | ✓ | ✓ |
| `@pllm.add_logits`, `@pllm.guide_sampling`, request-state training (e.g., ESamp) | ✓ (Gemma 2 requires dense guidance ¹) | ✓ | ✓ | ✓ | ✓ |
| `pllm.model.ActivationPatch` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `pllm.model.Adapter`, `@pllm.train_in_forward` | ✓ All projections | q/k/v ² | Full-attention v, dense MLP ³ | — ⁵ | — ⁵ |
| `@pllm.custom_residual` | ✓ | ✓ (except `site="value"`) | ✓ (`site="value"` full-attention layers only) | ✓ (except `site="value"`) | ✓ (except `site="value"`) |
| `@pllm.custom_attention`, static KV layouts | ✓ | ✓ | — ⁴ | ✓ (MLA compressed latents ⁶) | — ⁵ |
| `@pllm.custom_forward`, `pllm.PositionState` | ✓ | ✓ | — ⁴ | — ⁵ | — ⁵ |
| Full-model causal training (`pllm.causal_model`: TLM, TTRL) | ✓ (except Gemma 2 ⁷) | — ⁷ | — ⁷ | — ⁷ | — ⁷ |
| Host-only training (`pllm.host.make_host`) | Model-agnostic | Model-agnostic | Model-agnostic | Model-agnostic | Model-agnostic |

1. Gemma 2's `final_logit_softcapping` requires exact full-vocabulary guidance rather than bounded top-K guidance; GLM-4 and Gemma 2 / 3 four-norm structures are aligned with native branch ordering.
2. Sparse MoE MLP layers route tokens to experts without a single unified dense projection, so routed layers in `Mixtral`, `Qwen2-MoE`, and `Qwen3-MoE` expose attention `q/k/v` (unrouted dense layers in `Qwen2-MoE` and `Qwen3-MoE` also expose MLP deltas); `site="value"` relies on a dense `v_proj` slice.
3. Gated DeltaNet layers have no q/k/v projections; Qwen3.5's full-attention layers interleave per-head queries with output gates, so modifying query alone would alter the gate—hence only full-attention `v` projections and dense MLPs are exposed.
4. Gated DeltaNet recurrent state does not reside in the paged KV cache, so pLLM cannot replay or re-read those layers from KV pages alone.
5. DeepSeek V2/V3 MLA has no standalone V projection and splits MoE by expert; gpt-oss uses sink-enabled sliding-window attention and MXFP4 expert projections. Both expose read-only ops, `ActivationPatch`, and non-`value` `@pllm.custom_residual` rewrites, but not `Adapter` or full-model training.
6. `@pllm.custom_attention` on DeepSeek V2/V3 computes attention directly over compressed latent pages (`kv_lora_rank` + `qk_rope_head_dim`) via matrix absorption (`W_UK_T` / `W_UV`), supporting `keep="all"` only (no static `window` buffer).
7. Full-model training backward passes are implemented for these 7 dense architectures and require a BF16 base model with unquantized embeddings and LM head; Gemma 2 ties shared embeddings without a separate `lm_head`.

Kimi K2.x INT4 text towers can be sliced into a 6-layer DeepSeek V3 test slice for structural testing with the same feature boundaries as DeepSeek V3; see [Using New Model Families](/pllm/docs/howto/new-model-families/).

## CUDA Graphs and Eager {#en-cuda-graphs-and-eager}

By default, `ModelConfig(cuda_graphs="auto")` uses CUDA graphs whenever possible and falls back to eager execution otherwise, recording the reason in `pllm.explain`'s `forward` line and `llm.diagnostics().eager_reason`; `cuda_graphs=True` raises an error in those cases.

| Pattern | Model Forward |
| --- | --- |
| Reads & exports, sampling effects, request-state training (backward captured alongside AdamW), `@pllm.write` | vLLM CUDA graph |
| `ActivationPatch`, `Adapter` backed by `RequestState` or `InstanceState`, `@pllm.custom_residual`, `@pllm.custom_attention`, token-history lookups | vLLM CUDA graph (vLLM captures the base model ops and overlay reads without Inductor fusion) |
| `@pllm.custom_forward` | Does not use vLLM's full-model graph; pLLM schedules layer segments and captures its own sliced graphs |
| `cuda_graphs=False` | Eager |

## Tensor Parallelism {#en-tensor-parallelism}

Requires vLLM 0.32.0 or later with `ModelConfig(tensor_parallel_size=N)` (`multiprocessing="auto"` enables subprocess isolation automatically).

| Supported Multi-GPU | Requires `tensor_parallel_size=1` |
| --- | --- |
| Hidden reads & exports (delivered from rank 0 only), sampling guidance, request-state training (ESamp, output-layer LoRA, etc.), `@pllm.custom_forward`, `ActivationPatch` | `per_head` adapter deltas, `Adapter` with optimizer / side-stream reads-writes or initialized via `FromModule` / `ModelCheckpoint` / `Fed` |
| Dense/low-rank deltas on `q/k/v`, `gate/up`, `down` backed by `Zeros`/`Computed`/`Pretrained`/`Imported` (including sharded `@pllm.train_in_forward` updates) | Full-model causal training (`pllm.causal_model`), state feeding (`pllm.Fed`) |
| `@pllm.custom_residual` without `pllm.value(...)` / `pllm.input_tokens` dependencies; `@pllm.custom_attention` directly reading local-rank head KV | `@pllm.custom_residual` fed `pllm.value(...)` / `pllm.input_tokens`, or `@pllm.custom_attention` using `pllm.attention(...)` fact tables / `window` buffers |

Memory reservations reported by `pllm.explain` are per GPU.

## Quantization and Attention Backends {#en-quantization-and-attention-backends}

| Item | Support | Notes |
| --- | --- | --- |
| Quantized base models | GPTQ, AWQ, and FP8 checkpoints, admitted under the same family rules | `Adapter`-style edits add a high-precision correction to the quantized op's output, matching vLLM's LoRA behavior on quantized models; full-model training dequantizes transiently during backward (QLoRA-style); methods reading `pllm.operand("weight")` keep an additional dequantized weight copy resident |
| Quantized KV (`kv_cache_dtype`) | Incompatible with `@pllm.custom_attention` | Custom reads directly read KV pages (including MLA latent pages) and require unquantized K/V |
| Attention backend | `@pllm.custom_attention` requires FlashAttention 2 (or the MLA backend for DeepSeek V2/V3) | Other features are backend-agnostic |
| Prefix caching | Supported for read-only workloads, stateless `@pllm.custom_residual`, and `Adapter` / `last_tokens=0` `ActivationPatch` reading only immutable `InstanceState` (non-`Imported`/`Fed`, no writer methods) | Must be disabled when reading `RequestState`, mutable or externally updated `InstanceState`, `last_tokens > 0`, `@pllm.train_in_forward`, or `@pllm.custom_attention`: cached KV depends on request-private or dynamically changing state |

</div>

<div class="zine-lang-panel is-hidden" data-lang-panel="zh" lang="zh-CN" markdown="1">

<div id="zh-sec-1"></div>

本页按模型、功能、vLLM 版本和并行方式列出 pLLM 支持什么。每条限制后面写了原因；不支持的算法族及其原因见
[能力地图](/pllm/docs/guides/capability-map/) 和 [设计取舍](/pllm/docs/explain/design-tradeoffs/)，换模型的具体步骤见
[在新模型家族上使用](/pllm/docs/howto/new-model-families/)。

## vLLM 版本 {#zh-sec-2}

| 项目 | 支持 | 说明 |
| --- | --- | --- |
| vLLM 版本 | 0.27 – 0.34 | 覆盖 `vllm>=0.27.0,<0.35`。其中 `0.27.1` 与 `0.32.0` 在本仓库跑过完整 GPU 回归；`0.33`–`0.34` 适配了上游接口变更（依赖 PyTorch 2.14 / CUDA 13 驱动） |
| model runner | vLLM 的 MRv2 | pLLM 自己选定 MRv2（在 0.27 等版本自动设置 `VLLM_USE_V2_MODEL_RUNNER=1`），不支持 MRv1 |
| 张量并行 | vLLM 0.32.0 及以上 | 见下文[张量并行](#zh-sec-5) |
| 流水线并行、数据并行 | 不支持 | 必须为 1 |
| 在线服务 | `pllm.make_async_llm` 与 `pllm serve` | `ModelConfig(multiprocessing="auto")` 默认自动为异步句柄与多卡隔离 EngineCore 子进程；支持连续准入、流式 SSE、断连取消、`pllm_route` 会话路由，以及在途 `import_state()` / `train_batch()` / `feed()` 与无负载运行，见 [在线服务指南](/pllm/docs/howto/online-serving/) |

## 模型与功能 {#zh-sec-3}

pLLM 按 vLLM 的**具体模型类**准入，共覆盖 15 类解码器架构（16 个原生模型类，见 [在新模型家族上使用](/pllm/docs/howto/new-model-families/)），任意未核对子类不予准入。

| 功能 | Qwen2 / 2.5、Llama、Qwen3、Mistral、Phi-3、GLM-4、Gemma 2 / 3 | Mixtral、Qwen2-MoE、Qwen3-MoE | Qwen3.5（Gated DeltaNet 混合，含 MoE 版） | DeepSeek V2 / V3 | gpt-oss |
| --- | --- | --- | --- | --- | --- |
| 读取与导出（`pllm.hidden`、`residual`、`final_hidden`、`layer_logits`）、探针 | ✓ | ✓ | ✓ | ✓ | ✓ |
| `@pllm.add_logits`、`@pllm.guide_sampling`、请求状态训练（ESamp 一类） | ✓（Gemma 2 需稠密引导 ¹） | ✓ | ✓ | ✓ | ✓ |
| `pllm.model.ActivationPatch` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `pllm.model.Adapter`、`@pllm.train_in_forward` | ✓ 全部投影 | q/k/v ² | 全注意力层的 v、稠密 MLP ³ | — ⁵ | — ⁵ |
| `@pllm.custom_residual` | ✓ | ✓（`site="value"` 除外） | ✓（`site="value"` 仅全注意力层） | ✓（`site="value"` 除外） | ✓（`site="value"` 除外） |
| `@pllm.custom_attention`、静态 KV 布局 | ✓ | ✓ | — ⁴ | ✓（MLA 压缩潜变量 ⁶） | — ⁵ |
| `@pllm.custom_forward`、`pllm.PositionState` | ✓ | ✓ | — ⁴ | — ⁵ | — ⁵ |
| 整模型因果训练（`pllm.causal_model`：TLM、TTRL） | ✓（Gemma 2 除外 ⁷） | — ⁷ | — ⁷ | — ⁷ | — ⁷ |
| 只在宿主上训练（`pllm.host.make_host`） | 与模型无关 | 与模型无关 | 与模型无关 | 与模型无关 | 与模型无关 |

1. Gemma 2 的 `final_logit_softcapping` 需要全词表精确引导，不适用有限 top-K 引导；GLM-4 与 Gemma 2 / 3 的四层归一化结构已精确对齐原生分支顺序。
2. 稀疏 MoE 层的 MLP 把 token 路由给专家，无统一稠密投影，因此 `Mixtral`、`Qwen2-MoE` 与 `Qwen3-MoE` 路由层开放 attention `q/k/v`（`Qwen2-MoE` 与 `Qwen3-MoE` 的非路由稠密层同样开放 MLP 增量）；`site="value"` 依赖稠密 `v_proj` 切片。
3. Gated DeltaNet 层没有 q/k/v 投影；Qwen3.5 的全注意力层把每个头的 query 与输出门交错存放，单独改 query 会改到门，所以只开放全注意力层 `v` 投影与稠密 MLP。
4. Gated DeltaNet 层的递归状态不在分页 KV cache 里，pLLM 无法只靠 KV 页重放或改读这些层。
5. DeepSeek V2/V3 的 MLA 无独立 V 投影且 MoE 按专家拆分；gpt-oss 使用带 sink 的滑动窗口 attention 与 MXFP4 专家投影，二者均开放只读、`ActivationPatch` 与非 `value` 的 `@pllm.custom_residual` 改写，不开放 `Adapter` 与整模型训练。
6. DeepSeek V2/V3 上的 `@pllm.custom_attention` 直接对压缩潜变量页（`kv_lora_rank` + `qk_rope_head_dim`）通过矩阵吸收（`W_UK_T` / `W_UV`）计算注意力，仅支持 `keep="all"`，不支持静态 `window` 缓冲。
7. 整模型训练的反向按这 7 类稠密架构实现，并要求 BF16 底座、embedding 与 LM head 不量化；Gemma 2 绑定共享 embedding，没有独立 `lm_head`。

Kimi K2.x 的 INT4 文本塔可以切成六层的 DeepSeek V3 切片做结构测试，功能边界同 DeepSeek V3，见
[在新模型家族上使用](/pllm/docs/howto/new-model-families/)。

## CUDA graph 与 eager {#zh-sec-4}

默认 `ModelConfig(cuda_graphs="auto")`：能进 CUDA graph 就进，不能进时模型以 eager 运行，原因写在 `pllm.explain` 的
`forward` 行和 `llm.diagnostics().eager_reason` 里；`cuda_graphs=True` 时这些情况直接报错。

| 写法 | 模型 forward |
| --- | --- |
| 读取与导出、采样效果、请求状态训练（反向与 AdamW 一起捕获）、`@pllm.write` | vLLM 的 CUDA graph |
| `ActivationPatch`、基于 `RequestState` 或 `InstanceState` 的 `Adapter`、`@pllm.custom_residual`、`@pllm.custom_attention`、按 token 历史查表 | vLLM 的 CUDA graph（vLLM 捕获模型原本的算子与覆盖读表，不做 inductor 融合） |
| `@pllm.custom_forward` | 不用 vLLM 的整模型 graph；pLLM 按层段调度并捕获自己的切片图 |
| `cuda_graphs=False` | eager |

## 张量并行 {#zh-sec-5}

需要 vLLM 0.32.0 及以上，设 `ModelConfig(tensor_parallel_size=N)`（`multiprocessing="auto"` 会自动启用子进程隔离）。

| 可以多卡 | 需要 `tensor_parallel_size=1` |
| --- | --- |
| hidden 读取与导出（只从 rank 0 交付）、采样引导、请求状态训练（ESamp、输出层 LoRA 一类）、`@pllm.custom_forward`、`ActivationPatch` | per-head 适配器增量、带优化器 / 侧链读写或使用 `FromModule` / `ModelCheckpoint` / `Fed` 初始化的 `Adapter` |
| `q/k/v`、`gate/up`、`down` 上基于 `Zeros`/`Computed`/`Pretrained`/`Imported` 的 dense/low-rank 增量（含分片 `@pllm.train_in_forward` 更新） | 整模型因果训练（`pllm.causal_model`）、状态 feed（`pllm.Fed`） |
| 无 `pllm.value(...)` / `pllm.input_tokens` 依赖的 `@pllm.custom_residual`；直接改读本卡本地头 KV 的 `@pllm.custom_attention` | 给 `@pllm.custom_residual` 喂 `pllm.value(...)` / `pllm.input_tokens`，或使用 `pllm.attention(...)` 事实表 / `window` 缓冲的 `@pllm.custom_attention` |

`pllm.explain` 报告的显存预留是每张卡的数。

## 量化与 attention 后端 {#zh-sec-6}

| 项目 | 支持 | 说明 |
| --- | --- | --- |
| 量化底座 | GPTQ、AWQ、FP8 检查点，按同一家族的规则准入 | `Adapter` 一类改写像 vLLM 对量化模型做 LoRA 一样，在量化算子的输出上加一份高精度修正；整模型训练在反向时临时反量化（QLoRA 的做法）；读 `pllm.operand("weight")` 的方法会让该投影多常驻一份反量化后的权重 |
| 量化 KV（`kv_cache_dtype`） | 与 `@pllm.custom_attention` 不兼容 | 定制读法直接读 KV 页（含 MLA 潜变量页），需要未量化的 K/V |
| attention 后端 | `@pllm.custom_attention` 需要 FlashAttention 2（或 DeepSeek V2/V3 的 MLA 后端） | 其他功能不限后端 |
| 前缀缓存 | 只读、无状态 `@pllm.custom_residual`，以及仅读取不可变 `InstanceState`（非 `Imported`/`Fed`、无写入方法）的 `Adapter` 与 `last_tokens=0` `ActivationPatch` 可开启 | 读取 `RequestState`、可变或外部更新 `InstanceState`、`last_tokens > 0`、`@pllm.train_in_forward` 或 `@pllm.custom_attention` 时必须关闭：缓存里的 KV 依赖请求私有或动态变化的状态 |

</div>

</div>

<div class="zine-doc-footer">
  <a class="zine-pill-link" data-i18n-zh="← 返回 pLLM 项目与文档中心" data-i18n-en="← Back to pLLM Project &amp; Docs" href="/pllm/#docs">← Back to pLLM Project &amp; Docs</a>
  <a class="zine-pill-link" href="https://github.com/LinesHogan/pLLM/blob/main/doc/reference/support-matrix.md" target="_blank" rel="noopener">GitHub &middot; doc/reference/support-matrix.md ↗</a>
</div>

<script defer src="/assets/js/bilingual-post.js"></script>
