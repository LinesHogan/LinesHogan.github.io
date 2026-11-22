---
layout: page
title: "11 Lookup Memory: N-Gram Priors and Engram-Style Residuals"
description: "pLLM Documentation · Algorithm Family Examples (06–13)"
permalink: /pllm/docs/examples/11_lookup_memory/
nav: false
---

<link rel="stylesheet" href="{{ '/assets/css/pllm-hub.css' | relative_url | bust_file_cache }}">

<div class="zine-dispatch" data-title-zh="11 查表记忆：n-gram 先验与 Engram 式残差" data-title-en="11 Lookup Memory: N-Gram Priors and Engram-Style Residuals" data-desc-zh="pLLM 文档 · 算法族示例与实测 (06–13)" data-desc-en="pLLM Documentation · Algorithm Family Examples (06–13)">
  <div class="zine-dispatch-links">
    <a class="zine-pill-link primary" data-i18n-zh="← 返回 pLLM 文档中心" data-i18n-en="← pLLM Docs Hub" href="/pllm/#docs">← pLLM Docs Hub</a>
    <a class="zine-pill-link" data-i18n-zh="文档导航地图" data-i18n-en="Docs Roadmap" href="/pllm/docs/">Docs Roadmap</a>
    <span class="zine-lang-badge" data-i18n-zh="可执行教程 (.ipynb)" data-i18n-en="Notebook (.ipynb)">Notebook (.ipynb)</span>
  </div>
  <a class="zine-pill-link" data-i18n-zh="在 GitHub 打开 .ipynb ↗" data-i18n-en="Open .ipynb on GitHub ↗" href="https://github.com/LinesHogan/pLLM/blob/main/doc/examples/11_lookup_memory.ipynb" target="_blank" rel="noopener">Open .ipynb on GitHub ↗</a>
</div>

<div id="markdown-content">

<div class="zine-lang-panel" data-lang-panel="en" lang="en" markdown="1">

<div id="en-sec-1"></div>

After reading Tutorials 00, 07, and 09, let's examine inference-time algorithms that look up tables keyed by context tokens. Two representative examples are modifying logits directly with an n-gram language model prior (or kNN-LM-style adjustments), and DeepSeek's Engram—hashing n-grams into a large embedding table, applying a gate, and adding the result into the residual stream.

Bringing such algorithms into an inference engine raises three practical questions:
1. **How to obtain context tokens**: lookup keys depend solely on token IDs, not hidden states. Inside an inference engine, you need each row's current token ID and its preceding tokens—and they must remain accurate under continuous batching, chunked prefill, and preemption recomputation.
2. **Where to place large tables**: these tables are often huge (an Engram table can exceed the size of the model itself), so you must decide whether they live in GPU VRAM or host memory.
3. **How expensive PCIe transfers are**: when a large table sits in host memory, every step pulls rows over PCIe. You want to see how many bytes move per step and how much throughput drops before running anything.

In pLLM, these three pieces are expressed as follows:
- `pllm.input_tokens(history=n)` gives each row `[current token, previous, ..., n back]` (`int64`, padded with `-1` before the start of a request). Data comes directly from vLLM's GPU-resident token history with zero GPU-to-CPU synchronization.
- Tables are loaded from safetensors via `InstanceState(init=pllm.Pretrained(file))`. Placement is explicit: `pllm.Pretrained(..., placement="device")` (default; stays in VRAM and never silently spills to host) or `placement="host"` (pinned host memory read directly per row by the GPU). pLLM never silently moves a table—that is an expensive choice made explicitly by the author, with costs itemized by `pllm.explain`.
- The same lookup result can feed either `@pllm.add_logits` (n-gram prior) or `@pllm.custom_residual` (Engram-style residual).

Before starting, note these limitations and costs in this example:
- The table here is counted from a tiny synthetic corpus rather than trained; `scale=10` causes the model to recite the corpus, which only shows that table lookup is wired up, not a quality improvement.
- In the Engram-style residual step, the table stores next-token word embeddings from the corpus; adding them into a model never trained for that architecture has only a weak effect. Real Engram trains the table and gate jointly with the model.
- Throughput has a cost: the logits prior adds a vocab-wide vector per row; on 7B in this run, `device` placement reaches `98%` of vanilla vLLM and `host` placement reaches `96%`, a modest additional cost (measured in Step 5).
- Outputs and numbers below come from RTX 4090, vLLM 0.32.0; Steps 1–4 and 6–7 use `Qwen/Qwen2.5-0.5B-Instruct`, while Step 5 throughput uses `Qwen/Qwen2.5-7B-Instruct` (bf16).

We proceed in seven steps:
1. Inspect what `pllm.input_tokens` returns;
2. Count an n-gram table from a short corpus;
3. Write the n-gram prior workload and compare device vs. host placement with `explain`;
4. Run contract checks and text generation;
5. Benchmark throughput;
6. Feed the same lookup into the residual stream (Engram style);
7. Run GPU contract checks.

First, import dependencies and prepare the model configuration.

```python
import os

os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"  # Show only vLLM errors to suppress startup logs

import tempfile
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer
from vllm import SamplingParams

import pllm
from pllm import testing
from pllm.benchmark import throughput

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
CONFIG = pllm.ModelConfig(MODEL, max_model_len=512, max_num_seqs=8, max_num_batched_tokens=512,
                          gpu_memory_utilization=0.4, seed=0)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
```

## Step 1: What `pllm.input_tokens` Returns {#en-step-1-what-pllminput_tokens-returns}

Before writing the lookup logic, let's verify what tokens each row actually receives via `pllm.input_tokens` during prefill and decode.

Below we write a small export-only workload `ShowTokens`. It reads `[current token, previous, two back]` per row via `pllm.input_tokens(history=2)` and casts it to `.float()` for host export.

Keep this rule in mind for what each row represents:
- If a computation feeds into `custom_residual`, it runs for **every scheduled token** (including every prompt token).
- Other methods (such as `@pllm.export` here and `@pllm.add_logits` later) run for **sampled rows**, i.e., the last token of the prompt and each decode token generated thereafter.

Let's run it to inspect the token IDs.

```python
seen: list[list[int]] = []


@dataclass(frozen=True)
class ShowTokens(pllm.Workload):
    @pllm.export(name="ids", max_rows=4, max_cols=3)
    def ids(self, ids: torch.Tensor = pllm.input_tokens(history=2)) -> torch.Tensor:
        return ids.float()  # [rows, 3]: current token, previous, two tokens back

    def on_export(self, event: pllm.io.TraceEvent) -> None:
        seen.extend(event.tensor().long().tolist())


with pllm.make_llm(CONFIG, workload=ShowTokens()) as llm:
    output = llm.generate(["The capital of France is"], SamplingParams(max_tokens=4, temperature=0.0),
                          use_tqdm=False)[0]
print("prompt 的 token:", [tokenizer.decode([t]) for t in output.prompt_token_ids])
print("生成的 token:  ", [tokenizer.decode([t]) for t in output.outputs[0].token_ids])
for row in seen:
    print(row, "->", [tokenizer.decode([t]) for t in row])
```

```text
prompt 的 token: ['The', ' capital', ' of', ' France', ' is']
生成的 token:   [' Paris', '.', ' It', ' is']
[374, 9625, 315] -> [' is', ' France', ' of']
[12095, 374, 9625] -> [' Paris', ' is', ' France']
[13, 12095, 374] -> ['.', ' Paris', ' is']
[1084, 13, 12095] -> [' It', '.', ' Paris']
```

Four rows were exported, corresponding to four sampling steps:
- Row 1 `[374, 9625, 315]` -> `[' is', ' France', ' of']`: the current token is the prompt's final token `' is'` (whose output samples the first generated token `' Paris'`), preceded by `' France'` and `' of'`.
- Row 2 `[12095, 374, 9625]` -> `[' Paris', ' is', ' France']`: the current token is now `' Paris'` generated in the previous step, preceded by `' is'` and `' France'`.
- On each subsequent step, the current token is the token generated in the previous step. Four tokens were generated in total; the 4th token `' is'` was sampled by row 4, after which the request finished without being fed back as an input, so it does not appear in the exported table.

## Step 2: Counting an N-Gram Table from a Corpus {#en-step-2-counting-an-n-gram-table-from-a-corpus}

Having verified the ordering of `pllm.input_tokens`, let's build a table to query. We want to see clearly whether the model retrieves next-token bonuses when encountering familiar bigram contexts, while also making the table large enough to show the difference between VRAM and host-memory placement.

We prepare a 4-sentence synthetic corpus describing a fictional festival, the "Lantern Fair of Quillmere." At every position in the corpus, we use `(current token, previous token)` as the context and count which tokens follow and how often.

Table layout:
- `2048` buckets total, with contexts mapped to bucket indices by a simple integer hash `ngram_keys`.
- Each bucket stores two fields:
  1. `context`: the 2 `int64` token IDs corresponding to that bucket, checked on lookup to prevent false hits from hash collisions.
  2. `bonus`: a `bfloat16` logits bonus vector of width `151936` (Qwen2.5's LM head width, including padding). Next tokens seen in the corpus receive `1 + log(count)`, and all other entries are `0`.

We save both tensors into a single safetensors file.

```python
corpus = (
    "The Lantern Fair of Quillmere is held every year on the first full moon of autumn. "
    "Families carry paper lanterns shaped like herons down to the Silverbrook river, and the oldest baker in town "
    "hands out honey cakes stamped with a crescent moon. The fair ends at midnight, when the lanterns are set afloat "
    "and the bell of the old granary rings thirteen times."
)
BUCKETS = 2048
VOCAB = 151936  # Qwen2.5 LM head width (including padding)


def ngram_keys(current: torch.Tensor, previous: torch.Tensor, buckets: int) -> torch.Tensor:
    return (current * 1000003 + previous + 1).remainder(buckets)


def fit_table(text: str, buckets: int, path: str, vocab: int = VOCAB) -> int:
    """Count what follows each bigram in the corpus and write a hash table."""
    ids = tokenizer(text).input_ids
    follows: dict[tuple[int, int], dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for t in range(1, len(ids) - 1):
        follows[(ids[t], ids[t - 1])][ids[t + 1]] += 1
    context = torch.full((buckets, 2), -2, dtype=torch.int64)  # Context stored per bucket for collision checks
    bonus = torch.zeros(buckets, vocab, dtype=torch.bfloat16)  # One logits bonus row per bucket
    for (current, previous), nexts in follows.items():
        key = int(ngram_keys(torch.tensor(current), torch.tensor(previous), buckets))
        context[key] = torch.tensor([current, previous])
        bonus[key] = 0
        for token, count in nexts.items():
            bonus[key, token] = 1.0 + torch.log(torch.tensor(float(count)))
    save_file({"context": context, "bonus": bonus}, path)
    return len(follows)


TABLE = os.path.join(tempfile.gettempdir(), "lantern_ngram.safetensors")
print("语料", len(tokenizer(corpus).input_ids), "个 token，", fit_table(corpus, BUCKETS, TABLE), "个不同的上下文；",
      f"表文件 {os.path.getsize(TABLE) / 2**20:.0f} MiB")
```

```text
语料 78 个 token， 74 个不同的上下文； 表文件 594 MiB
```

The corpus has `78` tokens and `74` distinct contexts, producing a `594 MiB` table file. Since only `74` buckets are populated, most rows in this table are zeros—we intentionally sized it to `2048 × 151936` so the next step clearly contrasts device vs. host placement.

## Step 3: Authoring the N-Gram Prior and Comparing Placement {#en-step-3-authoring-the-n-gram-prior-and-comparing-placement}

Now we write the `NGramPrior` workload: at each step, look up the table using the current and previous token; if the bucket's stored context matches the row, extract that `151936`-wide bonus row, multiply by `scale`, and add it to the logits.

Two patterns in the code are worth noting:
1. **One `InstanceState` holds the whole file**: `table` declares two fields loaded by name from the safetensors file via `init=pllm.Pretrained(self.path)`. `context` needs `int64`, so it uses `pllm.Field((self.buckets, 2), torch.int64)`; `bonus` matches the model's vocabulary width, written as a shape function `lambda m: (self.buckets, m.vocab_size)` where `m` holds model dimensions. `placement` applies to the entire state: with `"host"`, both `context` and `bonus` live in host memory.
2. **Intermediate result `lookup`**: `lookup` itself has no decorator. It reads `pllm.input_tokens(history=1)`, hashes to bucket keys, checks `context == ids`, and returns `bonus[keys].float() * self.scale` on hits. `prior` (`@pllm.add_logits()`) then consumes this intermediate result via `pllm.result("lookup")` and adds it to logits.

Before launching on GPU, let's compare the resource accounting of `placement="device"` and `placement="host"` on CPU using `pllm.explain`.

```python
@dataclass(frozen=True)
class NGramPrior(pllm.Workload):
    path: str = TABLE
    scale: float = 0.0
    buckets: int = BUCKETS
    placement: Literal["device", "host"] = "device"

    @property
    def table(self) -> pllm.InstanceState:
        return pllm.InstanceState({"context": pllm.Field((self.buckets, 2), torch.int64),
                                   "bonus": lambda m: (self.buckets, m.vocab_size)},
                                  init=pllm.Pretrained(self.path, placement=self.placement))

    def lookup(self, ids: torch.Tensor = pllm.input_tokens(history=1),
               context: torch.Tensor = pllm.parameter("table", "context"),
               bonus: torch.Tensor = pllm.parameter("table", "bonus")) -> torch.Tensor:
        keys = ngram_keys(ids[:, 0], ids[:, 1], self.buckets)
        hit = (context[keys] == ids).all(-1, keepdim=True)  # Hit only if the stored bucket context matches this row
        return torch.where(hit, bonus[keys].float(), 0.0) * self.scale

    @pllm.add_logits()
    def prior(self, extra: torch.Tensor = pllm.result("lookup")) -> torch.Tensor:
        return extra
```

```python
for placement in ("device", "host"):
    report = pllm.explain(NGramPrior(scale=10.0, placement=placement), CONFIG)
    for line in str(report).splitlines():
        if line.startswith(("table ", "memory ")):
            print(line.replace(f"pretrained {TABLE}", "pretrained <TABLE>"))
    print()
```

```text
table  instance  622362624  622362638  field 0: int64 x 1 pages, field 1: bfloat16 x 1 pages; source: pretrained <TABLE> sha256 45f6aaccc61895503b2f82d8ac0b3be7e29afeacec37e3387b0c8bb0d4a8c492 (622362784 bytes); placement: device; host bytes: 0; logical transfer bytes/row: 0; decode bytes/step: 0
memory   peak 736516646 bytes funded by pLLM [per-rank tensor reservations; excludes vLLM model-graph/activation pools and GPU-dependent BLAS workspace; execution_peak_budget_bytes]

table  instance  622362624  0  field 0: int64 x 1 pages, field 1: bfloat16 x 1 pages; source: pretrained <TABLE> sha256 45f6aaccc61895503b2f82d8ac0b3be7e29afeacec37e3387b0c8bb0d4a8c492 (622362784 bytes); placement: host; host bytes: 622362638; logical transfer bytes/row: 303888; decode bytes/step: 2431104
memory   peak 114154008 bytes funded by pLLM [per-rank tensor reservations; excludes vLLM model-graph/activation pools and GPU-dependent BLAS workspace; execution_peak_budget_bytes]
```

Let's check both `explain` reports line by line:
- **With `placement="device"` (table in VRAM)**: `table` reserves `622362638` bytes in VRAM, with `host bytes: 0`, `logical transfer bytes/row: 0`, and `decode bytes/step: 0`. Total workload peak VRAM reservation (`memory peak`) is `736516646` bytes.
- **With `placement="host"` (table in host memory)**: `table` VRAM reservation drops to `0`, while host memory usage is `622362638` bytes. `explain` also calculates PCIe transfer volume: `logical transfer bytes/row: 303888` (`303888 = 151936 × 2 + 2 × 8` bytes, matching one `151936`-wide BF16 bonus row plus two `int64` context IDs), giving `decode bytes/step: 2431104` across 8 decode rows at `max_num_seqs=8`. Correspondingly, pLLM's peak VRAM reservation drops to `114154008` bytes.
- **File identity tracking**: `source` records the file path and its `sha256` (`45f6aaccc61895503b2f82d8ac0b3be7e29afeacec37e3387b0c8bb0d4a8c492`). If the underlying table file changes, the workload's identity changes with it.

Note that `explain` reports byte counts and does not convert them to latency; Step 5 measures how much those transfers slow down actual generation.

## Step 4: Contract Checks and Text Generation {#en-step-4-contract-checks-and-text-generation}

Before running in vLLM, we verify on CPU with `testing.check` that the workload satisfies all batching contracts (row permutation invariance, request isolation, mixed prefill/decode, chunked prefill, preemption recomputation, and equivalence to the uninstrumented model when `scale=0`).

Below we run `testing.check(NGramPrior(scale=10.0), CONFIG, neutral=NGramPrior(scale=0.0))` directly against the `594 MiB` (`2048`-bucket) table with `scale=0.0` as `neutral`; on CPU this takes ~`28` seconds with ~`6.6 GiB` peak process memory.

```python
print(testing.check(NGramPrior(scale=10.0), CONFIG, neutral=NGramPrior(scale=0.0)))
```

```text
status  contract / reason
PASS    tensor_parallel_size=1: admission
PASS    compile
PASS    row permutation
PASS    request isolation
PASS    mixed prefill/decode
PASS    chunking / tokens=
PASS    publish_lag
PASS    preempted
PASS    neutral
```

All 9 contracts `PASS`. In particular, `mixed prefill/decode`, `chunking / tokens=`, and `preempted` verify that when prompts are chunked, mixed with decode rows in a batch, or preempted and recomputed, every row's token history from `pllm.input_tokens(history=1)` stays identical.

With contracts passing, we generate text on GPU using the `2048`-bucket (`594 MiB`) table. We ask two questions about the Lantern Fair using greedy decoding (`temperature=0.0`) for `40` tokens across four configurations:
1. Vanilla vLLM (no workload);
2. `scale=0.0` (workload attached, bonus scaled by 0);
3. `scale=10.0` with table in VRAM (`placement="device"`);
4. `scale=10.0` with table in host memory (`placement="host"`).

```python
questions = ["Tell me about the Lantern Fair of Quillmere.", "What happens at the end of the Lantern Fair?"]
prompts = [tokenizer.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
           for q in questions]
greedy = SamplingParams(max_tokens=40, temperature=0.0)


def answers(llm) -> list[str]:
    return [output.outputs[0].text for output in llm.generate(prompts, greedy, use_tqdm=False)]


with pllm.make_llm(CONFIG) as plain:
    baseline = answers(plain)
for text in baseline:
    print("原生 vLLM:", repr(text))
results = {}
for workload in (NGramPrior(scale=0.0), NGramPrior(scale=10.0), NGramPrior(scale=10.0, placement="host")):
    with pllm.make_llm(CONFIG, workload=workload) as llm:
        results[(workload.scale, workload.placement)] = answers(llm)
print("scale=0 与原生 vLLM 逐字相同:", results[(0.0, "device")] == baseline)
for text in results[(10.0, "device")]:
    print("scale=10:", repr(text))
print("表放在主机内存与放在显存，输出逐字相同:", results[(10.0, "host")] == results[(10.0, "device")])
```

```text
原生 vLLM: 'The Lantern Fair of Quillmere is a traditional festival celebrated in the small town of Quillmere, located in the United States. This event is a significant cultural and economic event that has been held'
原生 vLLM: 'The Lantern Fair is a traditional Chinese festival that celebrates the end of the Lantern Festival. The end of the Lantern Fair typically marks the end of the festival and the beginning of the New Year. Here are'
scale=0 与原生 vLLM 逐字相同: True
scale=10: 'The Lantern Fair of Quillmere is held every year on the first full moon of autumn. Families carry paper lanterns shaped like herons down to the riverbank to light them. The fair ends'
scale=10: 'The Lantern Fair of Quillmere is held every year on the first full moon of autumn. Families carry paper lanterns shaped like herons down to the river to light them. The fair ends at'
表放在主机内存与放在显存，输出逐字相同: True
```

Inspect the generated responses:
- **Vanilla vLLM**: because the model has never seen our fictional town of Quillmere, it invents `"a small town, located in the United States"` for the first question and answers `"traditional Chinese festival"` for the second.
- **`scale=0`**: output matches vanilla vLLM verbatim (`True`).
- **`scale=10`**: both answers open by reciting our synthetic corpus: `"The Lantern Fair of Quillmere is held every year on the first full moon of autumn. Families carry paper lanterns shaped like herons down to the"`. They then diverge into `"riverbank to light them."` and `"river to light them."` (where the corpus had `"Silverbrook river"`), before returning to `"The fair ends"`.
- **Host vs. device placement**: both produce verbatim identical outputs (`True`): placement affects only speed and VRAM, never numerical results.

To reiterate: this table was counted from a tiny synthetic corpus rather than trained; `scale=10` reciting the corpus shows that token-history lookup and logit modification work end-to-end, not an improvement in generation quality.

## Step 5: Measuring Throughput Cost {#en-step-5-measuring-throughput-cost}

Earlier, `explain` showed that device placement avoids PCIe transfers while still processing a `151936`-wide vector per row, whereas host placement also transfers `303888` bytes per row over PCIe. Let's measure how much each slows down execution and whether PCIe transfers can be hidden.

0.5B sufficed for earlier steps; for throughput we switch to `Qwen/Qwen2.5-7B-Instruct` because a 0.5B decode step takes only a few milliseconds (amplifying lookup/transfer overhead), whereas 7B reflects typical deployment scale (bf16 weights ~15 GiB, requiring a 24 GiB GPU). 7B's LM head has width `152064`, so we build a `2048 × 152064` table (also ~594 MiB) from the same corpus, moving `152064 × 2 + 2 × 8 = 304144` bytes per row over PCIe.

Benchmark setup: `128` concurrent requests (`8` prompts × `16` samples × `256` tokens, `ignore_eos=True`), measured via `pllm.benchmark.throughput`: 1 warmup round followed by the median of 3 timed rounds, timing the entire `generate` call (including prefill).

```python
many = [f"Explain the mathematical idea numbered {i}:" for i in range(8)]
sampling = SamplingParams(n=16, max_tokens=256, temperature=0.8, top_p=0.95, ignore_eos=True, seed=7)
BIG = pllm.ModelConfig("Qwen/Qwen2.5-7B-Instruct", max_model_len=512, max_num_seqs=128, max_num_batched_tokens=512,
                       gpu_memory_utilization=0.8, seed=0)
BIG_TABLE = os.path.join(tempfile.gettempdir(), "lantern_ngram_7b.safetensors")
fit_table(corpus, BUCKETS, BIG_TABLE, vocab=152064)  # Qwen2.5-7B LM head width

rates = {}
with pllm.make_llm(BIG) as plain:
    rates["原生 vLLM"] = throughput(plain, many, sampling)
for placement in ("device", "host"):
    with pllm.make_llm(BIG, workload=NGramPrior(BIG_TABLE, scale=10.0, placement=placement)) as llm:
        rates[f"表在 {placement}"] = throughput(llm, many, sampling)
for name, rate in rates.items():
    print(f"{name:10s}: {rate:6.0f} token/s，为原生 vLLM 的 {rate / rates['原生 vLLM']:.0%}")
```

```text
原生 vLLM   :   5877 token/s，为原生 vLLM 的 100%
表在 device :   5742 token/s，为原生 vLLM 的 98%
表在 host   :   5615 token/s，为原生 vLLM 的 96%
```

Measured results (7B, 128 concurrent requests; ~128 rows per decode step):
- **Vanilla vLLM**: `5877 token/s` (`100%`), ~`21.8 ms` per decode step.
- **Table on `device` (VRAM)**: `5742 token/s`, `98%` of vanilla vLLM (~`22.3 ms`/step). This cost comes from fetching a `152064`-wide BF16 bonus row per sampled row, scaling it, and adding it to logits.
- **Table on `host` (host memory)**: `5615 token/s`, `96%` of vanilla vLLM (~`22.8 ms`/step). On top of `device`, each row reads `304144` bytes over PCIe; `host` adds only ~`0.5 ms` per step over `device`, even though moving `128 × 304144 ≈ 38.9 MB` at the measured ~`23.7 GB/s` takes ~`1.6 ms`—meaning most of the transfer overlaps with the model's forward pass.

Compared to 0.5B, a 7B decode step is much longer (streaming 15 GB of weights), so the same 38.9 MB lookup and transfer is a smaller fraction of step time, shrinking the relative overhead of both placements.

Why does overlap work? When a lookup method depends only on `pllm.input_tokens` and read-only tables, pLLM schedules it at the start of the step (`request.prepare`) on a `side` CUDA stream concurrently with the model forward. When reading host tables, pLLM uses only a small fraction of SMs (~1/4), because host-memory reads spend most of their time waiting on PCIe and a few thread blocks saturate PCIe bandwidth, leaving the remaining SMs for the forward pass. In `pllm.explain`, such a method shows up as `lookup  compute  request.prepare / side / captured` (as seen on `NGramPrior` here), allowing authors to verify side-stream overlap directly. Row-wise reads on host tables—`table[idx]`, `torch.index_select`, `torch.nn.functional.embedding`, row-wise indexing inside `torch.vmap`, and slicing a view (such as `table[:, ::2]`) before row indexing—all take this path; other operations on host tables are rejected during planning.

This yields a simple rule of thumb: PCIe bytes per step = rows × `logical transfer bytes/row` from `explain`, divided by PCIe bandwidth. When overlap applies, added wall time is much smaller than raw transfer time, though transfers longer than a forward pass cannot be fully hidden; furthermore, if lookup depends on intermediate hidden states rather than just tokens and static tables, it cannot start early at `request.prepare`. Narrow Engram-style tables (like Step 6, width `896`) move two orders of magnitude fewer bytes per row, making host placement proportionally cheaper. Ultimately, table placement trades VRAM for throughput, and `explain` lets you price that tradeoff upfront.

## Step 6: Feeding Lookup Results into the Residual Stream (Engram Style) {#en-step-6-feeding-lookup-results-into-the-residual-stream-engram-style}

So far we added lookup outputs to final-layer logits. What if, like DeepSeek's Engram, you want to inject retrieved vectors into an intermediate layer's residual stream?

In plain terms, Engram hashes context n-grams to look up an embedding table, gates the retrieved vector, and adds it into the residual stream; when the table is huge, it can reside in host memory and be read row-by-row.

Below we build a minimal `EngramLite` using the same corpus and hash function:
- **Table construction (`fit_memory`)**: still `2048` buckets, but instead of `151936`-wide logits bonuses, each bucket stores the word embedding of the next token in the corpus (width equal to the model's hidden size `896`, averaged if a context precedes multiple tokens).
- **State declaration (`memory`)**: both fields use `pllm.Field` (`context` is `(self.buckets, 2)` `int64`, and `memory` uses `lambda width, _: (self.buckets, width)` to read the model's hidden width), so they live in one `InstanceState`.
- **Lookup and injection**: `lookup` reads `pllm.input_tokens(history=1)`, hashes keys, and checks context; `add` (`@pllm.custom_residual(site="attention", layers=(20,))`) adds `self.gate * e` to the residual stream after attention at layer 20.

Two things differ from `NGramPrior`:
1. Because `lookup`'s output feeds `custom_residual`, it runs for **every scheduled token** (including every prompt token), not just sampled rows.
2. `lookup` depends only on input token IDs and the static table (not earlier hidden states). pLLM detects this and executes `lookup` ahead of layer 0 on the `side` CUDA stream.

Let's define `EngramLite` and inspect `lookup`'s scheduling line in `pllm.explain`.

```python
# Read the embedding matrix directly from the model's safetensors on CPU without loading the full model.
EMBEDDING = "model.embed_tokens.weight"
embedding = pllm.io.checkpoint_tensors(pllm.io.checkpoint_directory(MODEL), (EMBEDDING,))[EMBEDDING]


def fit_memory(text: str, buckets: int, path: str) -> None:
    """Engram-style narrow table: each bucket stores the embedding of the next token in the corpus (averaged if multiple)."""
    ids = tokenizer(text).input_ids
    context = torch.full((buckets, 2), -2, dtype=torch.int64)
    rows, counts = torch.zeros(buckets, embedding.shape[1]), torch.zeros(buckets, 1)
    for t in range(1, len(ids) - 1):
        key = int(ngram_keys(torch.tensor(ids[t]), torch.tensor(ids[t - 1]), buckets))
        context[key] = torch.tensor([ids[t], ids[t - 1]])
        rows[key] += embedding[ids[t + 1]].float()
        counts[key] += 1
    save_file({"context": context, "memory": (rows / counts.clamp_min(1)).bfloat16()}, path)


MEMORY = os.path.join(tempfile.gettempdir(), "lantern_memory.safetensors")
fit_memory(corpus, BUCKETS, MEMORY)


@dataclass(frozen=True)
class EngramLite(pllm.Workload):
    path: str = MEMORY
    gate: float = 0.0
    buckets: int = BUCKETS

    @property
    def memory(self) -> pllm.InstanceState:
        return pllm.InstanceState({"context": pllm.Field((self.buckets, 2), torch.int64),
                                   "memory": pllm.Field(lambda width, _: (self.buckets, width))},
                                  init=pllm.Pretrained(self.path))

    def lookup(self, ids: torch.Tensor = pllm.input_tokens(history=1),
               context: torch.Tensor = pllm.parameter("memory", "context"),
               memory: torch.Tensor = pllm.parameter("memory", "memory")) -> torch.Tensor:
        keys = ngram_keys(ids[:, 0], ids[:, 1], self.buckets)
        hit = (context[keys] == ids).all(-1, keepdim=True)
        return torch.where(hit, memory[keys].float(), 0.0)  # [every scheduled token, hidden]

    @pllm.custom_residual(site="attention", layers=(20,))
    def add(self, x: torch.Tensor, y: torch.Tensor, e: torch.Tensor = pllm.result("lookup")) -> torch.Tensor:
        return (x.float() + y.float() + self.gate * e).to(x.dtype)
```

```python
for line in str(pllm.explain(EngramLite(gate=8.0), CONFIG)).splitlines():
    if line.startswith(("forward", "lookup", "add ")):
        print(line)
```

```text
forward  CUDA graphs: no effect needs eager
add  custom_residual  model site or inactive method
lookup  compute  request.prepare / side / captured  memory (entry, 0)
```

```python
for gate in (0.0, 8.0):
    with pllm.make_llm(CONFIG, workload=EngramLite(gate=gate)) as llm:
        for text in answers(llm):
            print(f"gate={gate}:", repr(text))
```

```text
gate=0.0: 'The Lantern Fair of Quillmere is a traditional festival celebrated in the small town of Quillmere, located in the United States. This event is a significant cultural and economic event that has been held'
gate=0.0: 'The Lantern Fair is a traditional Chinese festival that takes place in the month of Lantern Festival, which is the 15th day of the first lunar month. The Lantern Fair is a time for people'
gate=8.0: 'The Lantern Fair of Quillmere is a traditional festival celebrated in the small town of Quillmere, located in the United States. This event is a significant cultural and economic event that has been held'
gate=8.0: 'The Lantern Fair is a traditional Chinese festival that is celebrated in many parts of China, including the Pearl River Delta region. At the end of the Lantern Fair, there are several significant events that take place'
```

Look at the three lines from `explain`:
- `forward  CUDA graphs: no effect needs eager`: the model still runs inside CUDA graphs without falling back to eager mode.
- `lookup  compute  request.prepare / side / captured  memory (entry, 0)`: `lookup`'s execution anchor is `request.prepare`, running on the `side` stream and captured in CUDA graphs (`captured`). Before the main stream starts layer 0, the side stream is already performing the table lookup concurrently with the model forward.

Now look at the generated text:
- Raising `gate` from `0.0` to `8.0` leaves the first answer unchanged and alters only the second half of the second answer without reciting the corpus. This matches our upfront caveat: the table only holds raw next-token word embeddings from the corpus, so injecting them into layer 20 of an untrained model has a weak effect; in real Engram, the table and gate are trained jointly with the model.
- You may notice that `gate=0.0` here differs slightly in the second half of the second answer from Step 4's "vanilla vLLM" output. Attaching `custom_residual` captures the model's original operators inside CUDA graphs without Inductor fusion, producing a slightly different floating-point rounding order than default vanilla vLLM (as discussed in Example 09). To verify strict equivalence at `gate=0.0`, compare bitwise against a baseline using the same graph settings—which is exactly what `testing.check_on_gpu` does next.

## Step 7: GPU Contract Checks {#en-step-7-gpu-contract-checks}

Finally, we run `testing.check_on_gpu` on GPU for both `NGramPrior` (`scale=0.0` as `neutral`) and `EngramLite` (`gate=0.0` as `neutral`), verifying four runtime contracts in the real vLLM engine:
1. `explain/probe`: the `explain` plan computed solely from `config.json` matches recomputation after loading the real model;
2. `host syncs`: zero GPU-to-CPU synchronizations over 8 steady decode steps;
3. `preempted`: after request preemption and recomputation, per-token inputs, generated token IDs, and terminal state match identically;
4. `neutral`: at `scale=0.0` or `gate=0.0`, greedy decoding and fixed-seed sampling are bitwise identical to the vanilla model under the same graph settings.

```python
print(testing.check_on_gpu(NGramPrior(scale=10.0), CONFIG, neutral=NGramPrior(scale=0.0)))
print(testing.check_on_gpu(EngramLite(gate=8.0), CONFIG, neutral=EngramLite(gate=0.0)))
```

```text
status  contract / reason
PASS    explain/probe
PASS    host syncs: eight steady decode steps
PASS    preempted: fixed per-token inputs, token IDs and terminal request state
PASS    neutral: greedy and seeded sampling bitwise
status  contract / reason
PASS    explain/probe
PASS    host syncs: eight steady decode steps
PASS    preempted: fixed per-token inputs, token IDs and terminal request state
PASS    neutral: greedy and seeded sampling bitwise
```

All four GPU checks `PASS` for both workloads: zero GPU-to-CPU syncs over 8 decode steps (reading token history introduces no sync either), exact reproduction after preemption recompute, and bitwise parity with vanilla vLLM under matching settings when `scale` or `gate` is `0`.

### Summary {#en-summary}

Recapping what this guide covered:
- `pllm.input_tokens(history=n)` provides `[current token, previous, ..., n back]` per row directly on GPU without GPU-to-CPU synchronization.
- `InstanceState(..., init=pllm.Pretrained(file_path))` loads tables from safetensors and records the file's `sha256` hash in the workload identity.
- Placement is specified explicitly via `placement="device"` or `placement="host"`; pLLM never silently relocates tables, and `pllm.explain` reports VRAM reservation, host memory footprint, and PCIe bytes/row so you can price `host` placement upfront.
- The same lookup method can feed either `@pllm.add_logits` (per sampled row) or `@pllm.custom_residual` (per scheduled token, overlapping early on the side stream when depending only on token IDs and static tables).
- Whether lookup memory improves quality depends on how the table and model were trained; pLLM handles bringing lookup and injection cleanly into inference.

Importable versions of both workloads live in `examples/engram_lite.py` and `examples/ngram_prior.py` in the repository.

### Next {#en-next}

In the next tutorial, [12 Adaptive Depth](/pllm/docs/examples/12_adaptive_depth/), we look at how to let each token decide for itself which model layers to skip.

</div>

<div class="zine-lang-panel is-hidden" data-lang-panel="zh" lang="zh-CN" markdown="1">

<div id="zh-sec-1"></div>

读过教程 00、07 和 09 之后，我们来看一类“按上下文 token 查表”的推理算法。两个典型例子是：用 n-gram 语言模型先验（或 kNN-LM 式的修正）直接改动 logits，以及 DeepSeek 的 Engram——按 n-gram 哈希去查一张很大的嵌入表，经过门控后加进模型的残差流。

要把这类算法放进推理引擎，会遇到三个实际问题：
1. **怎么拿到上下文 token**：这类算法的查表键只由 token id 决定，不看 hidden state。在推理引擎里，你先得拿到“每一行对应的当前 token id 和它前面几个 token”，而且在连续批处理、分块 prefill、抢占重算下都要对。
2. **大表放哪里**：这类表往往很大（比如 Engram 的表可以比模型本身还大），你需要决定把它放在显存还是主机内存。
3. **跨 PCIe 搬数据的代价有多大**：大表放进主机内存时，每一步都要经 PCIe 搬数据。每步搬多少字节、吞吐慢多少，需要在写代码时就看得到。

在 pLLM 里，这三件事的写法是：
- `pllm.input_tokens(history=n)` 给每一行返回 `[当前 token, 前一个, ..., 前 n 个]`（`int64`，请求开始之前的位置填 `-1`）。数据来自 vLLM 自己在 GPU 上维护的 token 历史，不需要 GPU 到 CPU 的同步。
- 表用 `InstanceState(init=pllm.Pretrained(文件))` 从 safetensors 文件读入。表放哪里由你写明：`pllm.Pretrained(..., placement="device")`（默认，放显存，放不下也不会悄悄挪到主机）或 `placement="host"`（放在锁页主机内存里，GPU 直接按行读）。pLLM 不会替你自动搬表——这是一个昂贵的决定，由作者显式做出；`pllm.explain` 会把代价列出来。
- 同一个查表结果既可以送进 `@pllm.add_logits`（做 n-gram 先验），也可以送进 `@pllm.custom_residual`（做 Engram 式残差）。

开始动手前，先把这个示例里的限制和代价说清楚：
- 这里的表是从一小段虚构语料“数”出来的，不是训练出来的；`scale=10` 的 n-gram 先验会让模型复述语料，这只说明查表生效，不是质量提升。
- Engram 式残差那一步，表里存的是“语料里下一个 token 的词嵌入”，直接加进一个没按这种结构训练过的模型，效果很弱；真正的 Engram 里，表和门控是和模型一起训练的。
- 吞吐有代价：logits 先验每行要加一个词表宽的向量；在 7B 上测，这次运行里表放显存时吞吐为原生 vLLM 的 `98%`；表放主机内存时还要经 PCIe 读表，吞吐为原生 vLLM 的 `96%`，多出的代价不大（实测见第五步）。
- 本文的输出和数字来自以下环境：RTX 4090、vLLM 0.32.0；前四步和第六、七步用 `Qwen/Qwen2.5-0.5B-Instruct`，第五步的吞吐用 `Qwen/Qwen2.5-7B-Instruct`（bf16）。

下面我们按七步来做实验：
1. 看看 `pllm.input_tokens` 给出什么；
2. 从一段语料数出一张 n-gram 表；
3. 写 n-gram 先验负载，用 `explain` 对比表放显存和放主机内存；
4. 跑契约检查和文本生成；
5. 实测吞吐；
6. 把同样的查表送进残差流（Engram 式）；
7. 在 GPU 上跑契约检查。

先导入依赖并准备好模型配置。

```python
import os

os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"  # 只显示 vLLM 的报错，省掉启动日志

import tempfile
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer
from vllm import SamplingParams

import pllm
from pllm import testing
from pllm.benchmark import throughput

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
CONFIG = pllm.ModelConfig(MODEL, max_model_len=512, max_num_seqs=8, max_num_batched_tokens=512,
                          gpu_memory_utilization=0.4, seed=0)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
```

## 第一步：`pllm.input_tokens` 给出什么 {#zh-sec-2}

在写查表逻辑之前，我们先验证一件事：模型在跑 prefill 和 decode 时，每一行通过 `pllm.input_tokens` 拿到的究竟是哪些 token？

下面写一个只做导出的小负载 `ShowTokens`。它用 `pllm.input_tokens(history=2)` 读取每一行的 `[当前 token, 前一个, 前两个]`，因为导出值是浮点数，我们把它转成 `.float()` 导出到主机。

这里有一条关于“每一行代表什么”的规则：
- 如果一个计算结果被送进 `custom_residual`，它是按**每个被调度的 token** 算的（包括 prompt 里的每一个 token）。
- 其他方法（比如这里的 `@pllm.export` 导出，以及后面的 `@pllm.add_logits`）则是按**被采样的行**算的，也就是 prompt 的最后一个 token，以及之后每一步生成的 decode token。

我们来跑一下看看拿到的 token id。

```python
seen: list[list[int]] = []


@dataclass(frozen=True)
class ShowTokens(pllm.Workload):
    @pllm.export(name="ids", max_rows=4, max_cols=3)
    def ids(self, ids: torch.Tensor = pllm.input_tokens(history=2)) -> torch.Tensor:
        return ids.float()  # [行, 3]：当前 token、前一个、前两个

    def on_export(self, event: pllm.io.TraceEvent) -> None:
        seen.extend(event.tensor().long().tolist())


with pllm.make_llm(CONFIG, workload=ShowTokens()) as llm:
    output = llm.generate(["The capital of France is"], SamplingParams(max_tokens=4, temperature=0.0),
                          use_tqdm=False)[0]
print("prompt 的 token:", [tokenizer.decode([t]) for t in output.prompt_token_ids])
print("生成的 token:  ", [tokenizer.decode([t]) for t in output.outputs[0].token_ids])
for row in seen:
    print(row, "->", [tokenizer.decode([t]) for t in row])
```

```text
prompt 的 token: ['The', ' capital', ' of', ' France', ' is']
生成的 token:   [' Paris', '.', ' It', ' is']
[374, 9625, 315] -> [' is', ' France', ' of']
[12095, 374, 9625] -> [' Paris', ' is', ' France']
[13, 12095, 374] -> ['.', ' Paris', ' is']
[1084, 13, 12095] -> [' It', '.', ' Paris']
```

上面导出了 4 行，对应 4 次采样：
- 第一行 `[374, 9625, 315]` 对应 `[' is', ' France', ' of']`：当前 token 是 prompt 的最后一个 token `' is'`（这一行的输出采样出了第一个生成 token `' Paris'`），它前面两个 token 是 `' France'` 和 `' of'`。
- 第二行 `[12095, 374, 9625]` 对应 `[' Paris', ' is', ' France']`：当前 token 变成了上一步刚刚生成的 `' Paris'`，前面两个变成了 `' is'` 和 `' France'`。
- 之后每一行的“当前 token”都是上一步生成的 token。我们一共生成了 4 个 token，最后一个 `' is'` 是第 4 行采样出来的，它生成之后请求就结束了，没有再作为输入被读入，所以不在导出的表里。

## 第二步：从一段语料数出一张 n-gram 表 {#zh-sec-3}

确认了 `pllm.input_tokens` 的顺序后，我们来准备一张要查的表。我们想要看清当模型遇到熟悉的二元上下文时，能不能从表里查出下一个 token 的加分；同时我们也想构造一张体积够大的表，用来观察放显存和放主机内存的差别。

我们准备一段 4 句话的虚构语料，介绍一个虚构的节日“Lantern Fair of Quillmere”。对语料里的每个位置，我们用“当前 token + 前一个 token”作为上下文，数一数它后面跟着哪个 token、出现了几次。

表的结构设计如下：
- 一共 `2048` 个桶，上下文通过一个简单的整数哈希函数 `ngram_keys` 映射到桶号。
- 每个桶存两样东西：
  1. `context`：这个桶对应的上下文（2 个 `int64` 的 token id），用来在查表时核对，避免哈希冲突时误命中。
  2. `bonus`：一行长 `151936`（Qwen2.5 的 LM head 宽度，含填充）的 `bfloat16` logits 加分向量。在语料里出现过的下一个 token 得分为 `1 + log(次数)`，其余位置为 `0`。

我们把这两个张量写成同一个 safetensors 文件。

```python
corpus = (
    "The Lantern Fair of Quillmere is held every year on the first full moon of autumn. "
    "Families carry paper lanterns shaped like herons down to the Silverbrook river, and the oldest baker in town "
    "hands out honey cakes stamped with a crescent moon. The fair ends at midnight, when the lanterns are set afloat "
    "and the bell of the old granary rings thirteen times."
)
BUCKETS = 2048
VOCAB = 151936  # Qwen2.5 的 LM head 宽度（含填充）


def ngram_keys(current: torch.Tensor, previous: torch.Tensor, buckets: int) -> torch.Tensor:
    return (current * 1000003 + previous + 1).remainder(buckets)


def fit_table(text: str, buckets: int, path: str, vocab: int = VOCAB) -> int:
    """数一数语料里每个“前两个 token”后面跟着什么，写成一张哈希表。"""
    ids = tokenizer(text).input_ids
    follows: dict[tuple[int, int], dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for t in range(1, len(ids) - 1):
        follows[(ids[t], ids[t - 1])][ids[t + 1]] += 1
    context = torch.full((buckets, 2), -2, dtype=torch.int64)  # 每个桶存它对应的上下文，用来核对
    bonus = torch.zeros(buckets, vocab, dtype=torch.bfloat16)  # 每个桶一行 logits 加分
    for (current, previous), nexts in follows.items():
        key = int(ngram_keys(torch.tensor(current), torch.tensor(previous), buckets))
        context[key] = torch.tensor([current, previous])
        bonus[key] = 0
        for token, count in nexts.items():
            bonus[key, token] = 1.0 + torch.log(torch.tensor(float(count)))
    save_file({"context": context, "bonus": bonus}, path)
    return len(follows)


TABLE = os.path.join(tempfile.gettempdir(), "lantern_ngram.safetensors")
print("语料", len(tokenizer(corpus).input_ids), "个 token，", fit_table(corpus, BUCKETS, TABLE), "个不同的上下文；",
      f"表文件 {os.path.getsize(TABLE) / 2**20:.0f} MiB")
```

```text
语料 78 个 token， 74 个不同的上下文； 表文件 594 MiB
```

语料一共 `78` 个 token，有 `74` 个不同的上下文，写出来的表文件大小是 `594 MiB`。因为只有 `74` 个桶有数据，这张表里几乎全是 `0` 的加分行——我们故意让它有 `2048 × 151936` 这么大，好在下一步清楚地看出表放在显存和放在主机内存的差别。

## 第三步：写 n-gram 先验负载，对比表的位置 {#zh-sec-4}

现在我们来写 `NGramPrior` 负载：每一步根据当前行和前一个 token 查表，如果桶里存的上下文和当前行一致，就把那一行 `151936` 宽的加分取出来乘上 `scale`，加到 logits 上。

代码里有两处写法值得留意：
1. **一份 `InstanceState` 装下整个文件**：`table` 有两个字段，`init=pllm.Pretrained(self.path)` 按字段名从刚才保存的 safetensors 文件里读入。`context` 要指定 `int64`，所以写成 `pllm.Field((self.buckets, 2), torch.int64)`；`bonus` 的宽度是模型的词表宽度，写成形状函数 `lambda m: (self.buckets, m.vocab_size)`，`m` 是模型的尺寸。`placement` 作用于整份状态：放主机内存时，`context` 和 `bonus` 一起放过去。
2. **中间结果 `lookup`**：`lookup` 方法本身没有加装饰器。它读取 `pllm.input_tokens(history=1)`，算桶号、核对 `context == ids`，命中时返回 `bonus[keys].float() * self.scale`。接着，`prior` 方法用 `@pllm.add_logits()` 装饰，通过 `pllm.result("lookup")` 拿到这个中间结果并加到 logits 上。

写好负载之后，我们先不在 GPU 上启动模型，直接在 CPU 上用 `pllm.explain` 对比 `placement="device"` 和 `placement="host"` 的资源账单。

```python
@dataclass(frozen=True)
class NGramPrior(pllm.Workload):
    path: str = TABLE
    scale: float = 0.0
    buckets: int = BUCKETS
    placement: Literal["device", "host"] = "device"

    @property
    def table(self) -> pllm.InstanceState:
        return pllm.InstanceState({"context": pllm.Field((self.buckets, 2), torch.int64),
                                   "bonus": lambda m: (self.buckets, m.vocab_size)},
                                  init=pllm.Pretrained(self.path, placement=self.placement))

    def lookup(self, ids: torch.Tensor = pllm.input_tokens(history=1),
               context: torch.Tensor = pllm.parameter("table", "context"),
               bonus: torch.Tensor = pllm.parameter("table", "bonus")) -> torch.Tensor:
        keys = ngram_keys(ids[:, 0], ids[:, 1], self.buckets)
        hit = (context[keys] == ids).all(-1, keepdim=True)  # 桶里存的上下文和这一行的一致才算命中
        return torch.where(hit, bonus[keys].float(), 0.0) * self.scale

    @pllm.add_logits()
    def prior(self, extra: torch.Tensor = pllm.result("lookup")) -> torch.Tensor:
        return extra
```

```python
for placement in ("device", "host"):
    report = pllm.explain(NGramPrior(scale=10.0, placement=placement), CONFIG)
    for line in str(report).splitlines():
        if line.startswith(("table ", "memory ")):
            print(line.replace(f"pretrained {TABLE}", "pretrained <TABLE>"))
    print()
```

```text
table  instance  622362624  622362638  field 0: int64 x 1 pages, field 1: bfloat16 x 1 pages; source: pretrained <TABLE> sha256 45f6aaccc61895503b2f82d8ac0b3be7e29afeacec37e3387b0c8bb0d4a8c492 (622362784 bytes); placement: device; host bytes: 0; logical transfer bytes/row: 0; decode bytes/step: 0
memory   peak 736516646 bytes funded by pLLM [per-rank tensor reservations; excludes vLLM model-graph/activation pools and GPU-dependent BLAS workspace; execution_peak_budget_bytes]

table  instance  622362624  0  field 0: int64 x 1 pages, field 1: bfloat16 x 1 pages; source: pretrained <TABLE> sha256 45f6aaccc61895503b2f82d8ac0b3be7e29afeacec37e3387b0c8bb0d4a8c492 (622362784 bytes); placement: host; host bytes: 622362638; logical transfer bytes/row: 303888; decode bytes/step: 2431104
memory   peak 114154008 bytes funded by pLLM [per-rank tensor reservations; excludes vLLM model-graph/activation pools and GPU-dependent BLAS workspace; execution_peak_budget_bytes]
```

我们来逐项核对 `explain` 给出的两份账单：
- **当 `placement="device"`（表放显存）时**：`table` 在显存里预留了 `622362638` 字节，主机内存（`host bytes`）为 `0`，跨总线传输（`logical transfer bytes/row` 和 `decode bytes/step`）都是 `0`。pLLM 为整个负载预留的显存峰值（`memory peak`）是 `736516646` 字节。
- **当 `placement="host"`（表放主机内存）时**：`table` 的显存预留降为 `0`，主机内存占用 `622362638` 字节。此时 `explain` 算出了跨 PCIe 读表的传输量：每行逻辑传输 `303888` 字节（`logical transfer bytes/row: 303888`，`303888 = 151936 × 2 + 2 × 8` 字节，正好是一行 `151936` 宽的 BF16 加分加上一行两个 `int64` 的上下文），在 `max_num_seqs=8` 时，8 个 decode 行每步要搬 `2431104` 字节（`decode bytes/step: 2431104`）。相应地，pLLM 预留的显存峰值降到了 `114154008` 字节。
- **文件身份记录**：`source` 一栏记下了文件路径和它的 `sha256`（`45f6aaccc61895503b2f82d8ac0b3be7e29afeacec37e3387b0c8bb0d4a8c492`）。这意味着如果底层的表文件变了，负载的身份也就跟着变了。

注意 `explain` 只报字节数，不会把字节数换算成耗时；搬这些字节到底慢多少，需要到第五步去实测。

## 第四步：契约检查与文本生成 {#zh-sec-5}

在放进 vLLM 生成之前，我们先用 `testing.check` 在 CPU 上验证负载是否满足批处理契约（行置换不变、请求隔离、混合 prefill/decode、分块 prefill、抢占重算，以及 `scale=0` 时与不挂负载一致）。

下面我们直接用同一张 `594 MiB`（`2048` 个桶）的大表跑 `testing.check(NGramPrior(scale=10.0), CONFIG, neutral=NGramPrior(scale=0.0))`，把 `scale=0.0` 作为 `neutral` 对照；在 CPU 上约 `28` 秒跑完，进程内存峰值约 `6.6 GiB`。

```python
print(testing.check(NGramPrior(scale=10.0), CONFIG, neutral=NGramPrior(scale=0.0)))
```

```text
status  contract / reason
PASS    tensor_parallel_size=1: admission
PASS    compile
PASS    row permutation
PASS    request isolation
PASS    mixed prefill/decode
PASS    chunking / tokens=
PASS    publish_lag
PASS    preempted
PASS    neutral
```

9 项契约全部 `PASS`。其中 `mixed prefill/decode`、`chunking / tokens=` 和 `preempted` 检查的正是：当 prompt 被切块、和 decode 行混在一个 batch 里执行，或者请求被抢占后重新计算时，每一行通过 `pllm.input_tokens(history=1)` 读到的 token 历史是否保持不变。

契约检查通过后，我们换回那张 `2048` 个桶（`594 MiB`）的表，在 GPU 上实际生成文本。我们准备两个关于 Lantern Fair 的问题，用贪心解码（`temperature=0.0`）各生成 `40` 个 token，依次对比四种设置：
1. 原生 vLLM（不挂负载）；
2. `scale=0.0`（挂上负载但加分乘 0）；
3. `scale=10.0`，表放在显存（`placement="device"`）；
4. `scale=10.0`，表放在主机内存（`placement="host"`）。

```python
questions = ["Tell me about the Lantern Fair of Quillmere.", "What happens at the end of the Lantern Fair?"]
prompts = [tokenizer.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
           for q in questions]
greedy = SamplingParams(max_tokens=40, temperature=0.0)


def answers(llm) -> list[str]:
    return [output.outputs[0].text for output in llm.generate(prompts, greedy, use_tqdm=False)]


with pllm.make_llm(CONFIG) as plain:
    baseline = answers(plain)
for text in baseline:
    print("原生 vLLM:", repr(text))
results = {}
for workload in (NGramPrior(scale=0.0), NGramPrior(scale=10.0), NGramPrior(scale=10.0, placement="host")):
    with pllm.make_llm(CONFIG, workload=workload) as llm:
        results[(workload.scale, workload.placement)] = answers(llm)
print("scale=0 与原生 vLLM 逐字相同:", results[(0.0, "device")] == baseline)
for text in results[(10.0, "device")]:
    print("scale=10:", repr(text))
print("表放在主机内存与放在显存，输出逐字相同:", results[(10.0, "host")] == results[(10.0, "device")])
```

```text
原生 vLLM: 'The Lantern Fair of Quillmere is a traditional festival celebrated in the small town of Quillmere, located in the United States. This event is a significant cultural and economic event that has been held'
原生 vLLM: 'The Lantern Fair is a traditional Chinese festival that celebrates the end of the Lantern Festival. The end of the Lantern Fair typically marks the end of the festival and the beginning of the New Year. Here are'
scale=0 与原生 vLLM 逐字相同: True
scale=10: 'The Lantern Fair of Quillmere is held every year on the first full moon of autumn. Families carry paper lanterns shaped like herons down to the riverbank to light them. The fair ends'
scale=10: 'The Lantern Fair of Quillmere is held every year on the first full moon of autumn. Families carry paper lanterns shaped like herons down to the river to light them. The fair ends at'
表放在主机内存与放在显存，输出逐字相同: True
```

看看生成的回答：
- **原生 vLLM**：因为模型没见过我们虚构的 Quillmere 小镇，面对第一个问题它编造了一个“位于美国的小镇”；面对第二个问题则回答成了“中国的传统节日”。
- **`scale=0`**：输出与原生 vLLM 逐字相同（`True`）。
- **`scale=10`**：两个问题的回答在开头都复述了我们虚构语料里的句子：`"The Lantern Fair of Quillmere is held every year on the first full moon of autumn. Families carry paper lanterns shaped like herons down to the"`。接着它们分别写出了 `"riverbank to light them."` 和 `"river to light them."`——在这里离开了语料（语料原文接的是 `"Silverbrook river"`），随后又回到了语料里的 `"The fair ends"`。
- **表放主机内存 vs 放显存**：两者的生成结果逐字相同（`True`）：表放在哪里只影响速度和显存，不影响结果。

这里要再次说明：这张表是从一小段虚构语料直接“数”出来的，不是训练出来的；`scale=10` 让模型复述语料，只说明按 token 历史查表并修改 logits 的链路生效了，不是生成质量的提升。

## 第五步：实测吞吐代价 {#zh-sec-6}

前面 `explain` 告诉我们，表放显存时每行不用跨 PCIe 搬数据，但每行都要处理一个 `151936` 宽的向量；表放主机内存时，每行还要经 PCIe 搬 `303888` 字节。现在我们来实测这两种放法各慢多少，以及跨 PCIe 的搬运能不能被藏住。

前面几步用 0.5B 就够了；吞吐换到 `Qwen/Qwen2.5-7B-Instruct` 上测：0.5B 一步 decode 只有几毫秒，查表和搬运的占比被放大，7B 才是这类算法常见的部署规模（bf16 权重约 15 GiB，需要一张 24 GiB 的卡）。7B 的 LM head 宽 `152064`，所以用同一段语料再建一张 `2048 × 152064` 的表（同样约 594 MiB），每行跨 PCIe 搬 `152064 × 2 + 2 × 8 = 304144` 字节。

测试方法：同时跑 `128` 个并发请求（`8` 个 prompt × 每个采样 `16` 条 × 每条生成 `256` 个 token，`ignore_eos=True`）。用 `pllm.benchmark.throughput` 测 token/s：每个配置先预热一轮，再计时三轮取中位数，每轮计整次 `generate` 的时间（含 prefill）。

```python
many = [f"Explain the mathematical idea numbered {i}:" for i in range(8)]
sampling = SamplingParams(n=16, max_tokens=256, temperature=0.8, top_p=0.95, ignore_eos=True, seed=7)
BIG = pllm.ModelConfig("Qwen/Qwen2.5-7B-Instruct", max_model_len=512, max_num_seqs=128, max_num_batched_tokens=512,
                       gpu_memory_utilization=0.8, seed=0)
BIG_TABLE = os.path.join(tempfile.gettempdir(), "lantern_ngram_7b.safetensors")
fit_table(corpus, BUCKETS, BIG_TABLE, vocab=152064)  # Qwen2.5-7B 的 LM head 宽度

rates = {}
with pllm.make_llm(BIG) as plain:
    rates["原生 vLLM"] = throughput(plain, many, sampling)
for placement in ("device", "host"):
    with pllm.make_llm(BIG, workload=NGramPrior(BIG_TABLE, scale=10.0, placement=placement)) as llm:
        rates[f"表在 {placement}"] = throughput(llm, many, sampling)
for name, rate in rates.items():
    print(f"{name:10s}: {rate:6.0f} token/s，为原生 vLLM 的 {rate / rates['原生 vLLM']:.0%}")
```

```text
原生 vLLM   :   5877 token/s，为原生 vLLM 的 100%
表在 device :   5742 token/s，为原生 vLLM 的 98%
表在 host   :   5615 token/s，为原生 vLLM 的 96%
```

实测结果如下（7B，128 个并发请求；每个 decode 步按 128 行换算）：
- **原生 vLLM**：`5877 token/s`（`100%`），每个 decode 步约 `21.8 ms`。
- **表在 `device`（显存）**：`5742 token/s`，为原生 vLLM 的 `98%`，每个 decode 步约 `22.3 ms`。这一档的代价来自每个被采样的行都要从表里取出一行 `152064` 宽的 BF16 加分向量，乘上 `scale` 后加到 logits 上。
- **表在 `host`（主机内存）**：`5615 token/s`，为原生 vLLM 的 `96%`，每个 decode 步约 `22.8 ms`。在 `device` 的基础上，每行还要经 PCIe 读 `304144` 字节；`host` 比 `device` 每步只多约 `0.5 ms`，而这一步经 PCIe 搬 `128 × 304144` ≈ `38.9 MB`，按实测的约 `23.7 GB/s` 算需要约 `1.6 ms`——也就是说大部分搬运和模型 forward 重叠执行了。

和 0.5B 比，7B 一步 decode 长得多（读 15 GB 权重），同样 38.9 MB 的查表和搬运在整步里占得少，所以两种放法的相对代价都比小模型上小。

为什么能重叠？当查表方法只依赖 `pllm.input_tokens` 和只读的表时，pLLM 在每步开始（`request.prepare`）就把它放到旁路 CUDA stream（`side`）上执行，和模型 forward 同时进行；读主机表时，pLLM 只用一小部分 SM（约四分之一）去读，读主机内存的大部分时间在等 PCIe，少量线程块就能把 PCIe 带宽用满，其余 SM 留给 forward。在 `pllm.explain` 里，这样的方法显示为 `lookup  compute  request.prepare / side / captured`（本示例的 `NGramPrior` 也是这样），作者可以用这一行自查查表是否走上了旁路重叠。对主机表的按行读取——`table[idx]`、`torch.index_select`、`torch.nn.functional.embedding`、`torch.vmap` 里的按行读，以及先取视图（如 `table[:, ::2]`）再按行读——都走这条路径；对主机表做其他运算会在规划阶段被拒绝。

由此可以得到一个估算方法：每步跨 PCIe 字节数 = 行数 × `explain` 报的 `logical transfer bytes/row`，除以 PCIe 带宽得到搬运时间。能重叠时真正多出来的时间远小于它，但搬运时间超过一次 forward 时就藏不住了；另外，如果查表依赖模型中间层的 hidden（而不只是 token 和静态表），它就不能提前到 `request.prepare` 执行。像第六步那种 Engram 式的窄表，每行只有 hidden size `896` 宽，每行搬的字节少两个数量级，放主机内存的代价相应更小。总之，表放哪里是用显存换吞吐的取舍，代价可以用 `explain` 事先估出来。

## 第六步：把查表结果送进残差流（Engram 式） {#zh-sec-7}

前面我们把查表结果加到了最后一层的 logits 上。那如果像 DeepSeek 的 Engram 那样，想把查出来的向量加进模型中间层的残差流里呢？

用白话讲，Engram 的思路是：用上下文的 n-gram 哈希去查一张嵌入表，把查到的向量经过门控后加进残差流；当表很大时，可以把表放在主机内存里按行读取。

下面我们用同样的语料和同样的哈希函数，写一个缩小版的 `EngramLite`：
- **建表（`fit_memory`）**：依然是 `2048` 个桶，但这次每个桶不再存 `151936` 宽的 logits 加分，而是存“语料里接下来那个 token 的词嵌入”（宽度为模型的 hidden size `896`，如果同一个上下文后面接过多个 token 就取平均）。
- **状态声明（`memory`）**：这次两个字段都用了 `pllm.Field`（`context` 是 `(self.buckets, 2)` 的 `int64`，`memory` 用 `lambda width, _: (self.buckets, width)` 获取模型的 hidden 宽度），所以可以放在同一个 `InstanceState` 里。
- **查表与注入**：`lookup` 方法和前面一样，读 `pllm.input_tokens(history=1)`、算哈希、核对上下文；`add` 方法用 `@pllm.custom_residual(site="attention", layers=(20,))` 装饰，把 `self.gate * e` 加到第 20 层 attention 之后的残差流里。

这里有两个和 `NGramPrior` 不同的地方：
1. 因为 `lookup` 的结果送进了 `custom_residual`，所以它是对**每个被调度的 token** 都计算的（包括 prompt 里的每一个 token），而不再只是被采样的行。
2. `lookup` 只依赖输入的 token id 和那张不变的表（不依赖前面层的 hidden state）。pLLM 识别出这一点后，会把 `lookup` 放在模型 forward 开始之前、在旁路 CUDA stream（`side`）上提前执行。

我们先定义 `EngramLite`，然后看一眼 `pllm.explain` 里对 `lookup` 的调度说明。

```python
# 词嵌入矩阵直接从模型的 safetensors 文件里读（在 CPU 上），不用加载整个模型。
EMBEDDING = "model.embed_tokens.weight"
embedding = pllm.io.checkpoint_tensors(pllm.io.checkpoint_directory(MODEL), (EMBEDDING,))[EMBEDDING]


def fit_memory(text: str, buckets: int, path: str) -> None:
    """Engram 式的窄表：每个桶存“语料里接下来那个 token”的词嵌入（多个就取平均）。"""
    ids = tokenizer(text).input_ids
    context = torch.full((buckets, 2), -2, dtype=torch.int64)
    rows, counts = torch.zeros(buckets, embedding.shape[1]), torch.zeros(buckets, 1)
    for t in range(1, len(ids) - 1):
        key = int(ngram_keys(torch.tensor(ids[t]), torch.tensor(ids[t - 1]), buckets))
        context[key] = torch.tensor([ids[t], ids[t - 1]])
        rows[key] += embedding[ids[t + 1]].float()
        counts[key] += 1
    save_file({"context": context, "memory": (rows / counts.clamp_min(1)).bfloat16()}, path)


MEMORY = os.path.join(tempfile.gettempdir(), "lantern_memory.safetensors")
fit_memory(corpus, BUCKETS, MEMORY)


@dataclass(frozen=True)
class EngramLite(pllm.Workload):
    path: str = MEMORY
    gate: float = 0.0
    buckets: int = BUCKETS

    @property
    def memory(self) -> pllm.InstanceState:
        return pllm.InstanceState({"context": pllm.Field((self.buckets, 2), torch.int64),
                                   "memory": pllm.Field(lambda width, _: (self.buckets, width))},
                                  init=pllm.Pretrained(self.path))

    def lookup(self, ids: torch.Tensor = pllm.input_tokens(history=1),
               context: torch.Tensor = pllm.parameter("memory", "context"),
               memory: torch.Tensor = pllm.parameter("memory", "memory")) -> torch.Tensor:
        keys = ngram_keys(ids[:, 0], ids[:, 1], self.buckets)
        hit = (context[keys] == ids).all(-1, keepdim=True)
        return torch.where(hit, memory[keys].float(), 0.0)  # [每个被调度的 token, hidden]

    @pllm.custom_residual(site="attention", layers=(20,))
    def add(self, x: torch.Tensor, y: torch.Tensor, e: torch.Tensor = pllm.result("lookup")) -> torch.Tensor:
        return (x.float() + y.float() + self.gate * e).to(x.dtype)
```

```python
for line in str(pllm.explain(EngramLite(gate=8.0), CONFIG)).splitlines():
    if line.startswith(("forward", "lookup", "add ")):
        print(line)
```

```text
forward  CUDA graphs: no effect needs eager
add  custom_residual  model site or inactive method
lookup  compute  request.prepare / side / captured  memory (entry, 0)
```

```python
for gate in (0.0, 8.0):
    with pllm.make_llm(CONFIG, workload=EngramLite(gate=gate)) as llm:
        for text in answers(llm):
            print(f"gate={gate}:", repr(text))
```

```text
gate=0.0: 'The Lantern Fair of Quillmere is a traditional festival celebrated in the small town of Quillmere, located in the United States. This event is a significant cultural and economic event that has been held'
gate=0.0: 'The Lantern Fair is a traditional Chinese festival that takes place in the month of Lantern Festival, which is the 15th day of the first lunar month. The Lantern Fair is a time for people'
gate=8.0: 'The Lantern Fair of Quillmere is a traditional festival celebrated in the small town of Quillmere, located in the United States. This event is a significant cultural and economic event that has been held'
gate=8.0: 'The Lantern Fair is a traditional Chinese festival that is celebrated in many parts of China, including the Pearl River Delta region. At the end of the Lantern Fair, there are several significant events that take place'
```

看 `explain` 输出的三行信息：
- `forward  CUDA graphs: no effect needs eager`：模型依然运行在 CUDA graph 里，不需要退回 eager 模式。
- `lookup  compute  request.prepare / side / captured  memory (entry, 0)`：`lookup` 的执行锚点是 `request.prepare`，运行在 `side`（旁路 stream）上，并且被捕获进 CUDA graph（`captured`）。也就是说，在主 stream 开始跑第 0 层之前，旁路 stream 就已经在查表了，查表与主 stream 的模型前向重叠执行。

再看生成的文本：
- 把 `gate` 从 `0.0` 调到 `8.0`，第一个问题的回答没有变化，第二个问题的回答只改变了后半段，模型并没有复述语料。这正是我们在开头说明的：表里存的只是“语料里下一个 token 的词嵌入”，直接加进一个没按这种结构训练过的模型第 20 层里，效果很弱；在真正的 Engram 里，表和门控是和模型一起训练出来的。
- 你可能会注意到，这里 `gate=0.0` 时第二个问题的回答，和前面第四步里“原生 vLLM”的回答在后半段不完全一样。这是因为挂上 `custom_residual` 后，vLLM 在 CUDA graph 里捕获的是模型原本的算子（不做 inductor 融合），浮点数舍入顺序与默认的原生 vLLM 略有不同（这一点在示例 09 里讲过）。要验证 `gate=0.0` 是否严格等价，应该和同样 graph 设置下的基准做逐位比对——这就是下一步 `testing.check_on_gpu` 要做的事。

## 第七步：GPU 契约检查 {#zh-sec-8}

最后，我们在 GPU 上对 `NGramPrior`（以 `scale=0.0` 为 `neutral`）和 `EngramLite`（以 `gate=0.0` 为 `neutral`）分别运行 `testing.check_on_gpu`，检查它们在真实 vLLM 引擎里的四项契约：
1. `explain/probe`：只看 `config.json` 算出的 `explain` 与加载真实模型后重算的结果一致；
2. `host syncs`：8 个稳定的 decode 步内没有发生 GPU 到 CPU 的同步；
3. `preempted`：请求被抢占重算后，每个 token 的输入、生成的 token ID 和最终状态完全一致；
4. `neutral`：当 `scale=0.0` 或 `gate=0.0` 时，贪心解码和固定种子的随机采样与同样 graph 设置下的原生模型逐位相同（bitwise）。

```python
print(testing.check_on_gpu(NGramPrior(scale=10.0), CONFIG, neutral=NGramPrior(scale=0.0)))
print(testing.check_on_gpu(EngramLite(gate=8.0), CONFIG, neutral=EngramLite(gate=0.0)))
```

```text
status  contract / reason
PASS    explain/probe
PASS    host syncs: eight steady decode steps
PASS    preempted: fixed per-token inputs, token IDs and terminal request state
PASS    neutral: greedy and seeded sampling bitwise
status  contract / reason
PASS    explain/probe
PASS    host syncs: eight steady decode steps
PASS    preempted: fixed per-token inputs, token IDs and terminal request state
PASS    neutral: greedy and seeded sampling bitwise
```

两个负载在 GPU 上的四项检查全部 `PASS`：8 个 decode 步内没有 GPU 到 CPU 的同步（读 token 历史也没有引入同步），抢占重算能够严格复现，把 `scale` 或 `gate` 设为 `0` 时与同样设置的原生 vLLM 逐位相同。

### 小结 {#zh-sec-9}

回顾这一节的内容：
- `pllm.input_tokens(history=n)` 为每一行提供 `[当前 token, 前一个, ..., 前 n 个]`，数据直接从 GPU 上取，不需要 GPU 到 CPU 的同步。
- `InstanceState(..., init=pllm.Pretrained(文件路径))` 从 safetensors 读表，文件的 `sha256` 哈希会记入负载身份。
- 表放哪里由 `placement="device"` 或 `placement="host"` 显式指定，pLLM 不会替你悄悄搬表；`pllm.explain` 会报出显存预留、主机内存占用以及每行跨 PCIe 搬运的字节数，用它可以事先估算 `host` 放法的吞吐代价。
- 同一个查表方法既可以接 `@pllm.add_logits`（按采样行算），也可以接 `@pllm.custom_residual`（按每个被调度的 token 算，且只依赖 token id 和静态表时会在旁路 stream 提前执行）。
- 查表记忆能不能带来质量提升，取决于表和模型是怎么训练的；pLLM 负责的是把查表和注入放进推理过程。


仓库里的 `examples/engram_lite.py` 和 `examples/ngram_prior.py` 是这两个负载可直接 import 的版本。

### 下一步 {#zh-sec-10}

下一篇教程 [12 自适应深度](/pllm/docs/examples/12_adaptive_depth/) 中，我们来看怎么让每个 token 自己决定跳过模型里的哪些层。

</div>

</div>

<div class="zine-doc-footer">
  <a class="zine-pill-link" data-i18n-zh="← 返回 pLLM 项目与文档中心" data-i18n-en="← Back to pLLM Project &amp; Docs" href="/pllm/#docs">← Back to pLLM Project &amp; Docs</a>
  <a class="zine-pill-link" href="https://github.com/LinesHogan/pLLM/blob/main/doc/examples/11_lookup_memory.ipynb" target="_blank" rel="noopener">GitHub &middot; doc/examples/11_lookup_memory.ipynb ↗</a>
</div>

<script defer src="/assets/js/bilingual-post.js"></script>
