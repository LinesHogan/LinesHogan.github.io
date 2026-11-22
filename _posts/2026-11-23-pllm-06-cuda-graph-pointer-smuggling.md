---
layout: post
title: "图是死的，表是活的：怎么在冻结的 CUDA Graph 里偷渡动态指针？"
title_en: "Dead Graphs, Live Tables: Smuggling Dynamic Pointers and Autograd Inside Frozen CUDA Graphs"
date: 2026-11-23 03:05:00 +0800
description: "借用无绑定描述符表（Bindless Descriptor Table）与空位吸收槽建模静态图动态寻址，展示如何把外部搬家张量的裸指针当成 int64 填表并在 Triton 内还原，以及从“流事件记录器”视角预判预热污染、跨流 GC 与纯函数编译器克隆冲突（pLLM 技术专栏 · 支持 EN / 中文 一键切换）"
description_en: "Production serving engines capture decode steps into static CUDA Graphs to eliminate Python/driver launch overhead, freezing every kernel's tensor shapes and raw device addresses at warmup. (pLLM Engineering Series · Bilingual EN / 中文)"
tags: pllm vllm inference-systems
categories: systems
pllm_synced: true
pllm_order: 6
pllm_badge_zh: "极客拆解 · 静态图寻址"
pllm_badge_en: "Deep Dive · Frozen Graphs"
related_posts: true
---

<div class="zine-dispatch" data-title-zh="图是死的，表是活的：怎么在冻结的 CUDA Graph 里偷渡动态指针？" data-title-en="Dead Graphs, Live Tables: Smuggling Dynamic Pointers and Autograd Inside Frozen CUDA Graphs">
  <div class="zine-dispatch-links">
    <a class="zine-pill-link" data-i18n-zh="pLLM 项目主页与文档" data-i18n-en="pLLM Project &amp; Docs" href="/pllm/">pLLM 项目主页与文档</a>
    <a class="zine-pill-link" data-i18n-zh="GitHub · pLLM ↗" data-i18n-en="GitHub &middot; pLLM ↗" href="https://github.com/LinesHogan/pLLM" target="_blank" rel="noopener">GitHub · pLLM ↗</a>
  </div>
  <div class="zine-lang-switch" role="group" aria-label="Language">
    <button type="button" class="zine-lang-btn" data-lang-target="en" aria-pressed="false">EN</button>
    <button type="button" class="zine-lang-btn is-active" data-lang-target="zh" lang="zh-CN" aria-pressed="true">中文</button>
  </div>
</div>

<div class="zine-lang-panel" data-lang-panel="zh" lang="zh-CN" markdown="1">

<div class="zine-tldr" markdown="1">

**TL;DR：** 为了省掉逐个发射小算子的 CPU 开销，生产推理引擎会把解码步录成静态 CUDA Graph，代价是图内算子的张量形状与裸显存地址在预热那一刻就被锁死。想在不重录图、零主机同步的前提下跑通每步人数在变、座位在变、甚至外部输入缓冲区都在搬家的在线反向传播与 `AdamW`，核心建模手段是借用图形学与系统里的**无绑定描述符表（Bindless Descriptor Table）**：把一切运行期变化（包括外部张量的 64 位裸指针）降维成固定地址 `int64` 表里的整数。文末从“录制器真正记录的是什么”这一视角，统一解释预热防污、跨流 GC 与纯函数编译器冲突三类常见故障的排查逻辑。

</div>

> **pLLM 技术随笔与机制拆解（6/10 · 极客拆解 · 静态图寻址）** · [pLLM 项目介绍与文档中心](/pllm/) · [专栏目录与系统解剖图](/pllm/#blog) · [仓库源码原文 ↗](https://github.com/LinesHogan/pLLM/blob/main/blog/zh/06-cuda-graph-pointer-smuggling.md)

在单卡脚本里调通一段在线学习算法（比如每生成一个词，就拿首尾层隐藏状态算一次损失、反向求导并用 `AdamW` 更新当前请求专属的轻量权重）之后，搬进并发推理引擎遇到的第一个性能悬崖，通常不在 GPU 上，而在 CPU 上。

逐字解码时，每个请求每步只处理 1 个词。GPU 执行单个小算子只需几微秒，而 CPU 每发射一个算子都要穿过 Python 解释器、PyTorch 分发器和 CUDA 驱动。在线训练加上反向求导和动量更新，一步里零零碎碎有五六十个小算子：

```text
即时执行 (Eager —— CPU 逐个发令，发射间隙把 GPU 切碎):
  CPU: [发 k1][发 k2][发 k3][发 k4]...
                 \       \       \
  GPU:           [k1]空转 [k2]空转 [k3]...   (0.5B 模型关掉静态图，吞吐直接掉一半)

静态图回放 (CUDA Graph —— 预热录成驱动指令序列，运行期单次提交):
  预热期 (1次):  走一遍算子流 ──► 驱动锁死网格尺寸与所有张量的裸显存地址 (data_ptr)
  运行期 (每步): CPU: [ cudaGraphLaunch ]
                GPU: [k1][k2][k3][k4][k5]... (硬件队列连续执行，Python 不再介入)
```

为了消掉这笔发射开销，在线训练步也必须像主干模型那样录进静态 CUDA Graph。

但只要把两边的约束摆在一起，矛盾就出来了：CUDA Graph 要求算子看到的**张量形状恒定、物理基地址恒定、且全程不能有 GPU 到 CPU 的同步等待**；而外面的连续批处理引擎恰恰相反——每一步进场的活跃请求数在变，哪个请求排在批次的第几行在变，甚至上游组件（比如采样器）递过来的输入张量隔几步就会释放旧块、换一个新显存地址。

---

怎么对这个矛盾建模？

遇到“指令流必须静态、操作对象却在动态变化”的约束，不妨想想操作系统虚拟内存页表，或者现代图形 API（Vulkan / DirectX 12）里的**无绑定描述符表（Bindless Descriptor Table）**是怎么干的：着色器代码编译死之后不能改，但它可以绑定一张物理基地址永远不动的整数描述符表；每帧开画前，CPU 把本帧真正要读的纹理槽位和内存地址写进表里，GPU 顺着表做一次间接寻址。

把这个模式原样搬进 PyTorch 和 Triton，整套动态训练图的数据结构就收束成了一句话：**图是死的，表是活的。**

我们在启动时按最大并发上限，一次性分配两样基地址永不改变的显存块：一块是存放各参数字段的物理槽池 `pool[capacity, *field_shape]`，另一块是一张只有二十多 KB 的固定 `int64` 路由表 `TableLayout`（每步由 CPU 在锁页内存里填好数字推上显卡，异步拷贝只需约 6 微秒）。

顺着这张表，三层由浅入深的动态性可以逐一消解。

**第一层：座位每步洗牌，用槽号间接寻址解决。**  
这一步第 0 行是请求 A，下一步变成了请求 C。图里的内核不绑定任何具体请求的张量，只绑定槽池基地址和路由表地址：先去表的 `rows` 段读出当前行对应的源槽号 `src` 与目标写槽号 `dst`，再去 `pool[src]` 读参数、算完写回 `pool[dst]`。外头怎么换座位，变的只是表里的几个整数。

**第二层：在场人数不满上限，用“空位吸收槽（Padding Sink）”替代多档位录图。**  
上限设为 $$R = 128$$ 行、$$S = 128$$ 个请求状态，可线上低谷期某一步只来了 30 个请求，剩下 98 行空位怎么处理？如果模仿无状态前向那样按 `(行数, 状态数)` 组合分档录十几张图，每张图私有的反向激活显存池会把宝贵的 KV 缓存空间吃光。

想只录**唯一一张满容量图**，只需在设计路由表的读写列数时做一个不对称切分——源槽表 `sources` 开 `S + 1` 列（`0..S`），而目标写槽表 `destinations` 只开 `S` 列（`0..S-1`），把多出来的第 `S` 列专门留作吸收无效行的哨兵槽：

```text
固定形状路由表 (满编 R = 128)              固定基地址状态池
┌────────────────────────────┐
│ 行 0..29  (active=1)       │──────────► 状态 0..29 : 正常读 sources[s]，写回 destinations[s]
├────────────────────────────┤
│ 行 30..127 (active=0)      │──────────► 状态 S (空位吸收槽):
│ 状态号统一填 S              │            • 读取段 sources[F, S+1]  : 槽号填 -1，掩码读出全 0.0 (防 NaN)
└────────────────────────────┘            • 求导段 loss * active     : 权重为 0.0，对真实状态零梯度贡献
                                          • 写入段 destinations[F, S]: 无第 S 列，写内核直接跳过落盘
```

这样一来，无论本步来 1 个请求还是 128 个请求，静态图执行的永远是同一套规整张量运算，空行既不会算出 `NaN` 污染梯度，也不会触发任何多余的显存写入。（同理，哪怕算法要求按运行时的动态数值条件只更新部分请求，CPU 也不用调 `.tolist()` 同步等结果——直接给在场请求都分配好新槽位，图内内核发现某请求未命中更新条件时，把旧槽位原样拷进新槽位即可。）

---

**第三层：上游每步重新分配地址的外部张量，怎么在死图里读取？**

自己分配的状态池可以保证基地址不变，但宿主引擎传过来的外部张量（比如采样器刚产出的反馈张量）不归我们管——上游隔几步调整批次大小时就会释放旧张量、新分配一块显存。如果把这个外部张量直接作为算子参数录进 CUDA Graph，下一步回放时算子就会冲向已经释放的悬空旧地址。

初看这像个死结：不传张量怎么读数据？传了张量地址又会被锁死。

破局点在于跳出 PyTorch 的 `Tensor` 包装看硬件本质：**在 64 位统一虚拟寻址的 GPU 上，设备指针（Device Pointer）在物理表示上就是一个普通的 64 位整数。**

CUDA Graph 只禁止“作为内核入口参数的张量换地址”，从不禁止“固定地址缓冲区里存的整数换数值”。那我们干脆不把外部张量传进图里——每步回放前，让 CPU 调一次 `external_tensor.data_ptr()` 取出它的 64 位裸地址，连同两个维度的步长（`stride0`, `stride1`），当成三个普通 `int64` 填进固定路由表的 `descriptor` 段：

```text
固定地址路由表 (0xAAAA0000 —— 录制时绑定，永不改变)
┌────────────────────────────────────────────────────────┐
│ descriptor[0]: 0x7f8812340000  (本步外部张量的 data_ptr) │
│ descriptor[1]: 3584            (stride 0)              │
│ descriptor[2]: 1               (stride 1)              │
└───────────────────────┬────────────────────────────────┘
                        │ 1. tl.load(descriptor) 读出 64 位整数
                        │ 2. .to(tl.pointer_type(...)) 在寄存器内转回设备指针 T*
                        ▼
每步动态搬家的上游外部张量 (本步刚分配在 0x7f8812340000)
┌────────────────────────────────────────────────────────┐
│ [ row 0 ][ row 1 ][ row 2 ] ...                        │
└────────────────────────────────────────────────────────┘
```

在 Triton 内核里，只需十行代码就能完成这次“整数还原指针”的间接读取：

```python
@triton.jit
def _gather_rows(descriptor, rows, output, width: tl.constexpr):
    row = tl.program_id(0) // tl.cdiv(width, 256)
    col = (tl.program_id(0) % tl.cdiv(width, 256)) * 256 + tl.arange(0, 256)
    # 录制器只看到恒定的 descriptor 地址；运行期读出的 int64 直接转回类型指针解引用
    address = tl.load(descriptor).to(tl.pointer_type(output.dtype.element_ty))
    stride0 = tl.load(descriptor + 1)
    stride1 = tl.load(descriptor + 2)
    index = tl.load(rows + row)
    value = tl.load(address + index * stride0 + col * stride1, col < width, 0)
    tl.store(output + row * width + col, value, col < width)
```

一行 `.to(tl.pointer_type(...))`，就把静态计算图的生命周期与外部缓冲区的分配生命周期干净拆开了。

---

掌握了上面这套间接寻址模型之后，自己在项目里动手录复杂状态图时，怎样才能不踩进莫名其妙的崩溃和性能倒退里？

不少文章会把录图时的故障列成一串零散的注意事项，其实只要建立一个核心认知，所有问题都能自己提前推出来：**CUDA Graph 录制器看不懂你的 Python 数学意图，它只是忠实地把“录制窗口期间当前 CUDA Stream 上发生的每一次内存写操作和每一次分配器事件”原样录下来；而且它跟追求“纯函数无副作用”的图编译器天生路数不同。**

顺着这个视角往外推，三个最容易栽跟头的现场就全串起来了：

1. **既然预热和录制跑的是真指令，原地写算子就会污染线上状态。**  
   正式调用 `torch.cuda.graph` 之前，惯例要先在私有流上跑 3 遍预热（warmup），让 cuBLAS 和 Triton 完成内部工作区分配。但如果你的图末尾挂着原地写池子的 `AdamW`，直接跑 3 遍预热就会拿未初始化的缓冲把线上真参数连覆盖 3 次。顺着模型想，解法自然就有了：进入预热和录制前先调一句 `destinations.fill_(-1)` 把所有目标写槽屏蔽掉，录完图再填回真实槽号回放，4 遍捕获过程对状态池就成了严格只读操作。
2. **既然分配器事件也会被录进去，任何跨流内存释放都会炸毁录制窗口。**  
   线上偶发“捕获失败（operation not permitted when stream is capturing）”时，新手常去查是不是自己写了 `.item()`，却忽略了后台的 Python 循环垃圾回收器（GC）：录制中途分配包装对象时，GC 可能突然按计数触发，顺手析构掉一个几步前在**另一条 CUDA Stream** 上用过的死张量；C++ 析构函数向缓存分配器发出的跨流释放通知一插进当前捕获流，录图就会立刻作废。因此录制窗口外围必须用 `gc.disable()` 护住。（同理，若在图内使用设备端条件 `IF` 分支节点，PyTorch C++ 底层在条件节点开闭时会把同一个图显存池的引用计数多加两次，而 `CUDAGraph.reset()` 只减一次；排查显存泄漏时盯着图池引用计数看，就能定位到必须手动调一次底层池作用域释放来配平。）
3. **当原地状态池遇上 `torch.compile`，记得用剖析器核对总耗时而非只数内核个数。**  
   把训练步录进静态图后，常有人想到再套一层 `@torch.compile` 把五六十个碎算子融合成大内核。我们实测套上之后，旁路流每步发射的内核数确实从 59 个降到了 25 个，但每步 GPU 总耗时反而从约 **180 微秒恶化到了近 260 微秒**。打开剖析时间线一看，编译器在 `AdamW` 前面凭空塞进了一个耗时近 40 微秒的 `clone_paged_adamw_slice` 内核——因为 AOTAutograd 的函数式化（Functionalization）要求图内算子全是无副作用的纯函数，看到你往固定大池子的切片里原地写数据，它为了自保先把池子切片克隆了一份副本、算完再拷回去。在静态图已经消除了 CPU 发射税的前提下，盲目叠加纯函数编译反而会赔上显存带宽。

---

这套基于单张固定 `int64` 描述符表的捕获机制，目前承担着开源扩展层 **[pLLM](https://github.com/LinesHogan/pLLM)**（`pllm`）里所有逐请求在线训练与状态写入任务的图执行；在 7B 模型、128 并发下，回放包含前向、反向求导与分页 `AdamW` 的整步训练图只需约 600 微秒。

工程上没有白吃的午餐，这套建模换取“单图零重录”的代价有两条：一是低并发时段，空位吸收槽所在的行依然会陪跑满编的矩阵乘；二是由于有效行与空位行同批执行，用户编写的损失函数在行维度（`dim=0`）上必须严格独立——一旦混入跨行求均值等归约操作，空位行的零值就会直接稀释正常请求的梯度。

</div>

<div class="zine-lang-panel is-hidden" data-lang-panel="en" lang="en" markdown="1">

<div class="zine-tldr" markdown="1">

**TL;DR:** Production serving engines capture decode steps into static CUDA Graphs to eliminate Python/driver launch overhead, freezing every kernel's tensor shapes and raw device addresses at warmup. Running online backward passes and `AdamW` inside a single never-recaptured graph—while active request counts, batch row seats, and even upstream input buffer addresses change every step—comes down to borrowing **Bindless Descriptor Tables** from modern graphics APIs: reduce all runtime dynamism (including 64-bit raw pointers to relocated external tensors) to integers inside a fixed-address `int64` routing table. We close by viewing CUDA Graph capture as a literal stream-and-allocator event recorder, turning warmup corruption, cross-stream GC crashes, and `@torch.compile` clone regressions into predictable consequences of one mental model.

</div>

> **pLLM Engineering Essays & Deep Dives (6/10 · Deep Dive · Frozen Graphs)** · [pLLM Project & Docs Hub](/pllm/) · [Series Index & Anatomy Map](/pllm/#blog) · [Markdown Source on GitHub ↗](https://github.com/LinesHogan/pLLM/blob/main/blog/en/06-cuda-graph-pointer-smuggling.md)

After getting an online learning loop working in a single-request script—say, computing a token-level loss from hidden states, running `autograd` backward, and stepping per-request `AdamW` weights on every decode token—moving it into a concurrent serving engine hits an immediate wall. That first wall isn't on the GPU; it is on the CPU.

During autoregressive decode, each request processes 1 token per step. A tiny GPU kernel finishes in a few microseconds, whereas launching it from Python walks through the interpreter, the PyTorch dispatcher, and the CUDA driver. Add backward gradients and optimizer updates, and a single step fires fifty to sixty small kernels:

```text
Eager execution (CPU dispatches one kernel at a time; launch gaps starve the GPU):
  CPU: [launch k1][launch k2][launch k3][launch k4]...
                    \          \          \
  GPU:             [k1] idle  [k2] idle  [k3]...   (Disabling CUDA Graphs on 0.5B cuts throughput in half)

Static replay (CUDA Graph — recorded once at warmup, replayed in a single driver call):
  Warmup (once):   Run stream ──► driver freezes grid dims & every tensor's raw address (data_ptr)
  Runtime (step):  CPU: [ cudaGraphLaunch ]
                   GPU: [k1][k2][k3][k4][k5]...    (Hardware queue runs back-to-back, zero Python)
```

To erase that launch tax, online training has to live inside a static CUDA Graph just like the trunk model.

Put both sets of constraints side by side, however, and they collide head-on: CUDA Graphs require **constant tensor shapes, constant base device pointers, and zero GPU-to-CPU synchronization**; continuous batching changes active request counts every step, shuffles which request sits in which batch row, and periodically frees and reallocates upstream input tensors (like sampler outputs) at brand-new HBM addresses.

---

How do you model that contradiction?

Whenever an instruction stream must stay static while the objects it touches move around, look at OS virtual memory page tables or **Bindless Descriptor Tables** in Vulkan and DirectX 12: compiled shader code never changes, because it binds a single fixed-address integer table; before each frame, the CPU writes the active resource slots and buffer addresses into that table, and the GPU dereferences them indirectly.

Carry that exact pattern into PyTorch and Triton, and the entire architecture reduces to six words: **dead graphs, live tables.**

At engine startup, we allocate two buffers whose base addresses never move for the lifetime of the server: a physical state pool `pool[capacity, *field_shape]` holding parameter and optimizer pages, and a ~20 KB fixed-address `int64` routing table `TableLayout` (populated by the CPU in pinned memory each step and copied asynchronously in ~6 microseconds).

Three layers of runtime dynamism dissolve into that table one by one.

**Layer 1: Row seats shuffle every step—solve with slot indirection.**  
Request A sits at batch row 0 on step $$t$$, while Request C sits at row 0 on step $$t+1$$. Graph kernels never bind per-request tensors; they bind only the pool base pointer and the routing table, reading `src` and `dst` slot indices from the table's `rows` segment before indexing `pool[src]` and `pool[dst]`. Reseating requests just changes integers in the table.

**Layer 2: Active requests fall below capacity—use an asymmetric Padding Sink instead of multi-bucket graphs.**  
Suppose capacity is $$R = 128$$ rows and $$S = 128$$ states, yet an off-peak step arrives with only 30 live requests. Recording dozens of `(rows, states)` bucket graphs—as stateless forward passes do—would multiply private backward-activation memory pools across buckets and starve the KV cache.

Capturing **one single full-capacity graph** takes an asymmetric column split in the routing table: size the read table `sources` to `S + 1` columns (`0..S`), size the write table `destinations` to only `S` columns (`0..S-1`), and dedicate column `S` as a sentinel **Padding Sink**:

```text
Fixed-shape routing table (R = 128)        Fixed-base state pool
┌────────────────────────────┐
│ Rows 0..29  (active=1)     │──────────► States 0..29 : read sources[s], write destinations[s]
├────────────────────────────┤
│ Rows 30..127 (active=0)    │──────────► State S (Padding Sink):
│ All mapped to state S      │            • Read sources[F, S+1]  : slot -1, masked load returns 0.0 (no NaN)
└────────────────────────────┘            • Loss * active         : weight 0.0, zero gradient into live states
                                          • Write destinations[F,S]: no column S, write kernel skips store
```

Whether 1 request or 128 requests show up, the frozen graph runs the exact same dense tensor shapes without generating `NaN`s or stray HBM stores. (By the same token, when a workload updates only requests satisfying a dynamic data-dependent condition, the CPU never calls `.tolist()` to check the mask—it pre-allocates fresh destination slots for every live request, and an in-graph kernel copies old slots forward whenever a row's runtime predicate evaluates false.)

---

**Layer 3: How does a frozen graph read an upstream tensor whose HBM address changes every few steps?**

We own the state pool, so its base pointer stays put. We do _not_ own external tensors handed over by the host engine (such as feedback tensors freshly allocated by the sampler)—whenever upstream resizes a batch, the caching allocator frees the old block and hands back a new address. Binding that external `Tensor` directly as a kernel argument bakes a dangling pointer into the CUDA Graph.

At first glance that looks impossible: how do you read a tensor without passing it into the graph?

Step beneath PyTorch's `Tensor` wrapper to the hardware truth: **under 64-bit unified virtual addressing on modern GPUs, a device pointer is physically just an ordinary 64-bit integer.**

CUDA Graphs forbid changing a kernel's entry argument pointers; they never forbid changing `int64` values stored inside a fixed-address buffer. So we never pass the external tensor into the graph at all—before replay, the CPU calls `external_tensor.data_ptr()` and writes its raw 64-bit address plus `(stride0, stride1)` as three plain `int64` integers into the `descriptor` segment of the fixed routing table:

```text
Fixed-address routing table (0xAAAA0000 — bound once at capture, never moves)
┌─────────────────────────────────────────────────────────────┐
│ descriptor[0]: 0x7f8812340000  (external_tensor.data_ptr()) │
│ descriptor[1]: 3584            (stride 0)                   │
│ descriptor[2]: 1               (stride 1)                   │
└───────────────────────┬─────────────────────────────────────┘
                        │ 1. tl.load(descriptor) loads 64-bit integer
                        │ 2. .to(tl.pointer_type(...)) casts back to T* in registers
                        ▼
Relocated upstream tensor (allocated at 0x7f8812340000 on this step)
┌─────────────────────────────────────────────────────────────┐
│ [ row 0 ][ row 1 ][ row 2 ] ...                             │
└─────────────────────────────────────────────────────────────┘
```

Inside Triton, reconstructing the typed device pointer and gathering rows takes ten lines:

```python
@triton.jit
def _gather_rows(descriptor, rows, output, width: tl.constexpr):
    row = tl.program_id(0) // tl.cdiv(width, 256)
    col = (tl.program_id(0) % tl.cdiv(width, 256)) * 256 + tl.arange(0, 256)
    # Capture binds only the constant `descriptor` address; runtime casts the loaded int64 back to T*
    address = tl.load(descriptor).to(tl.pointer_type(output.dtype.element_ty))
    stride0 = tl.load(descriptor + 1)
    stride1 = tl.load(descriptor + 2)
    index = tl.load(rows + row)
    value = tl.load(address + index * stride0 + col * stride1, col < width, 0)
    tl.store(output + row * width + col, value, col < width)
```

One `.to(tl.pointer_type(...))` cast cleanly decouples the lifetime of the frozen CUDA Graph from the allocation lifetime of upstream tensors.

---

Once you build stateful CUDA Graphs yourself, how do you anticipate subtle crashes and slowdowns instead of playing whack-a-mole?

Most postmortems list CUDA Graph pitfalls as disconnected trivia. In practice, nearly all of them follow directly from one mental model: **a CUDA Graph recorder does not understand your Python math—it literally records every memory store and every allocator event that touches the capturing CUDA Stream during the capture window, and its in-place world clashes directly with pure-functional compilers.**

Walk forward from that single principle, and three classic failure modes become obvious ahead of time:

1. **Because warmup and capture execute real instructions, in-place writers will corrupt live state unless masked.**  
   Before `torch.cuda.graph` capture, standard practice runs 3 warmup iterations on a private stream so cuBLAS and Triton finish lazy workspace allocations. If your graph ends with an in-place `AdamW` kernel over the state pool, 3 warmup runs will overwrite live parameters three times with uninitialized scratch gradients! Following the model makes the safeguard immediate: call `destinations.fill_(-1)` before warmup + capture so all four passes become strictly read-only against the pool, then populate real destination slots on replay.
2. **Because allocator events are recorded too, any cross-stream free aborts capture.**  
   When `operation not permitted when stream is capturing` fires intermittently in production, engineers usually hunt for stray `.item()` calls while overlooking Python's cyclic garbage collector (`gc`): allocating wrapper objects mid-capture can trip a GC threshold, finalizing a dead tensor last used steps ago on _another_ CUDA Stream. Its C++ destructor notifies the caching allocator across streams and kills capture on the spot—so always bracket capture windows with `gc.disable()`. (Similarly, if you use device-side conditional `IF` nodes inside a graph, PyTorch's C++ internals increment the graph memory pool refcount twice across conditional scope entry/exit while `CUDAGraph.reset()` decrements it once; watching pool refcounts exposes the exact scope release needed to prevent pool leaks.)
3. **When an in-place state pool meets `@torch.compile`, verify total wall time in a profiler—never just kernel counts.**  
   Wrapping a captured training step in `@torch.compile` looks like an easy win to fuse fifty small ops. When we measured it, side-stream kernel launches dropped from 59 to 25—yet step duration _regressed_ from **~180 µs to nearly 260 µs**! Opening the trace showed why: AOTAutograd's Functionalization pass requires pure side-effect-free ops, so seeing an in-place slice mutation on the persistent pool caused it to inject a ~40 µs `clone_paged_adamw_slice` kernel to copy the pool slice out and back. Once CUDA Graphs have already removed CPU launch overhead, layering pure-functional compilation on top of in-place state pools trades away precious HBM bandwidth.

---

This single-descriptor-table capture architecture drives every per-request online training and state-mutation graph in **[pLLM](https://github.com/LinesHogan/pLLM)** (`pllm`); at 128-way concurrency on a 7B model, replaying a full forward, backward, and paged `AdamW` step takes ~600 microseconds.

Nothing in systems engineering is free, and buying "one graph, zero recaptures" carries two explicit costs: at low concurrency, Padding Sink rows still ride along through full-capacity GEMMs; and because live rows and sink rows share the batch tensor, user loss functions must stay strictly independent along `dim=0`—any cross-row mean or batch norm mixes sink zeros into live request gradients.

</div>

<script defer src="/assets/js/bilingual-post.js"></script>
