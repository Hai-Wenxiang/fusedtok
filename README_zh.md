# fusedtok

[![CI](https://github.com/Hai-Wenxiang/fusedtok/actions/workflows/ci.yml/badge.svg)](https://github.com/Hai-Wenxiang/fusedtok/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/fusedtok.svg)](https://pypi.org/project/fusedtok/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/Hai-Wenxiang/fusedtok/blob/main/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://github.com/Hai-Wenxiang/fusedtok/blob/main/pyproject.toml)

**面向 LLM 推理的融合 CUDA 算子库** —— RMSNorm / RoPE / SwiGLU / 解码注意力
等，支持**torch 张量零拷贝**：对比 PyTorch SDPA 最高 **8.7 倍加速**
（解码注意力，RTX 3060，见[性能基准](#性能基准)）。

**English version: [README.md](https://github.com/Hai-Wenxiang/fusedtok/blob/main/README.md)**

## 为什么选 fusedtok

LLM 推理框架每生成一个 token，都要跑一大堆小而吃带宽的算子；
eager 模式下每个中间结果都要在显存里来回读写。`fusedtok` 把它们
融合成单个 kernel，省掉中间张量的搬运和启动开销。

## 算子总览

共 38 个算子与辅助函数（1.x 系列冻结的 `fusedtok.__all__`，见下方"API
稳定性"）。`axpy` 是 v0.x 骨架保留至今的入门演示算子——可用，但不是
性能特性。

| 状态 | 算子 | 说明 |
| :---: | --- | --- |
| ✅ | RMSNorm（含残差） | LLaMA/Qwen 风格，融合残差加法 |
| ✅ | LayerNorm | 含仿射变换 |
| ✅ | RoPE | 交错与 NeoX 两种布局，支持 kv-cache `pos_offset` |
| ✅ | SwiGLU | 融合 MLP 激活 |
| ✅ | Softmax（按行） | 数值稳定版 |
| ✅ | SiLU / GeLU / GeLU-tanh / ReLU / Tanh / Sigmoid | 逐元素 |
| ✅ | add / mul | 逐元素二元（融合加残差模式） |
| ✅ | top-k / top-p（核采样） | 到达票据 radix + 早退压缩，缓存 CUDA 图整管线回放；并列取最靠前下标（131k k=50 上两张卡从持平到领先，全 k 范围如此） |
| ✅ | argmax / temperature | 贪心解码辅助 |
| ✅ | sample_topp | 融合 top-p（nucleus）采样：softmax -> 截取 top-p 集合 -> 按种子抽签，用全局质量做阈值 |
| ✅ | sample_topk | 融合 top-k 采样：softmax -> 保留 k 个 -> 在幸存者内重新归一化 -> 按种子抽签（131k 上 1.9-2.2x vs topk+multinomial 组合式） |
| ✅ | sample_minp | 融合 min-p 采样（v1.3）：保留所有 p >= min_p × p_max 的 token -> 重新归一化 -> 按种子抽签——值阈值截断，无需全局质量归约，核宽度天然自适应 |
| ✅ | sample_topp/minp/topk_batched | 批量采样（v1.4）：一次调用处理整个 `[行数, 词表]` 的 logits，每行按各自种子各出一个 token——每行原封不动地复用单行管线（逐行结果一致）；相比逐行循环，收益纯粹来自省掉逐行的提交开销：受提交延迟限制的主机（如 Windows/WDDM）上墙上时钟时间快 4-6 倍，尖峰解码分布下与 torch 原生批量 multinomial 同档 |
| ✅ | repetition penalty | CTRL 风格，作用于已生成的 token |
| ✅ | decode_step | 整个解码步一次融合调用：重复惩罚 -> 温度 -> top-p 采样，一次调用一次回读 |
| ✅ | decode_step_batched | 整个批的融合解码步（v1.5）：逐行 ragged 历史经逐行惩罚位图进入同一融合管线，每行按种子各出一个 token——每行跑单行 `decode_step` 管线，一致到文档化的 ulp 边界为止 |
| ✅ | quantize_int8 / dequantize_int8 / qadd_int8 | 对称逐张量 INT8，融合的反量化-相加-再量化 |
| ✅ | qgemm | INT8 矩阵乘，int32 精确：cp.async 双缓冲流水线 IMMA GEMM + 运行时 tile 调优（64x64 / 128x128）+ 每 warp 一行的 GEMV（M=1 解码；速度约为 fp16 投影的 2 倍） |
| ✅ | qgemm_perchannel | 真实 INT8 推理所用的 W8A8 布局：逐输出通道权重 scale 融合进同一 kernel 的 epilogue，零开销 |
| ✅ | attention_decode | 解码步因果注意力：GQA + 连续 kv-cache，在线 softmax、长 cache 自动 flash-decoding 切分、支持逐序列有效长度（`lens`）；**float32 / bfloat16 / float16 存储**（半精度 cache = 解码字节减半，softmax 仍 float32） |
| ✅ | kv_append | 连续解码循环的 cache 写入侧（v1.3）：每序列一个新 token 的 k/v 行原地 scatter 到 cache 第 `lens[b]` 行（一个微型 kernel，f32/bf16/fp16） |
| ✅ | attention_decode_paged | 1.2 主打特性：同样的解码注意力跑在 **vLLM 式块池 kv-cache** `[Nb, Hkv, P, D]` 上，经每序列块表间接寻址——cache 内存零碎片；任意合法表均可，f32/bf16/fp16 存储，约为连续版 1.09-1.11x 开销（对比预展开头参考 7.8x / 4.3x vs SDPA） |
| ✅ | kv_append_paged | 分页循环的 cache 写入侧：每序列一个新 token 的 k/v 行原地 scatter 到池中 `lens[b]` 位置（一个微型 kernel，f32/bf16/fp16） |
| ✅ | attention_prefill | 新序列 S 行注意力（因果 / 双向）；便捷路径——重度 prefill 仍建议交给 SDPA/flash（诚实约 0.45x） |
| ✅ | axpy | `a*x + b` —— v0.x 的入门演示算子，为 API 兼容保留 |

## 安装

```bash
pip install fusedtok
```

PyPI 提供预编译 wheel（CUDA 12.4 构建）：**Linux x86_64**（manylinux，
cp310-cp313）与 **Windows x86_64**（cp311-cp313）。其他平台或 Python 版本
pip 会自动从源码构建：

```bash
git clone https://github.com/Hai-Wenxiang/fusedtok.git
cd fusedtok
pip install .
```

**环境要求**：

- **RTX 30 系（Ampere）或更新**的 NVIDIA 显卡 —— 如 RTX 3060/3090、RTX 4080、RTX 5090、A100、H100
- CUDA Toolkit ≥ 12.0
- C++17 编译器（Windows 用 MSVC，Linux 用 GCC/Clang）；Python 3.10+

<details>
<summary>什么是 "compute capability"（计算能力）？点击展开</summary>

计算能力是 NVIDIA 给每代 GPU 架构的**版本编号**，不是性能分数。CUDA 代码必须针对特定架构编译才能运行。
构建产物包含计算能力 8.0（A100）与 8.6（RTX 30）的原生 cubin，以及 compute_86 PTX 回退：
Ampere 原生运行，更新架构（RTX 40/50 等）由驱动即时编译 PTX。

| 计算能力 | 架构 | 代表显卡 |
|---|---|---|
| 7.5 | Turing | GTX 16xx、RTX 20xx（不支持） |
| 8.0 / 8.6 | Ampere | A100、RTX 30xx |
| 8.9 | Ada | RTX 40xx（走 PTX） |
| 9.0 | Hopper | H100（走 PTX） |
| 12.0 | Blackwell | RTX 50xx（走 PTX） |

查看自己的显卡：运行 `nvidia-smi` 看到型号后，在 https://developer.nvidia.com/cuda-gpus 查询对应计算能力。

</details>

## 用法

numpy 进 / numpy 出，或 torch 进 / torch 出 —— 并支持 **CUDA 零拷贝**：
kernel 通过 `data_ptr()` 直接读写 torch 显存缓冲区，无暂存拷贝、无主机端同步。

```python
import numpy as np
import torch
import fusedtok

x = np.random.randn(4, 1024).astype(np.float32)
w = np.random.rand(1024).astype(np.float32)

# CPU 参考实现（正确性基准，任何机器可跑）
y = fusedtok.rmsnorm(x, w, eps=1e-6)

# 暂存式 CUDA：拷入 GPU、跑 kernel、拷回
y = fusedtok.rmsnorm(x, w, cuda=True)

# torch 张量零拷贝：kernel 直接在 torch 自己的缓冲区上运行，
# 与其他 torch 操作保持流式顺序
xt, wt = torch.from_numpy(x).cuda(), torch.from_numpy(w).cuda()
yt = fusedtok.rmsnorm(xt, wt)          # -> CUDA torch 张量

# 带 kv-cache 位置偏移的 RoPE，NeoX（LLaMA-HF）布局
q = torch.randn(1, 4096, device="cuda")          # 只传入新 token
q_rot, k_rot = fusedtok.rope(q, k=None, pos_offset=1023, neox=True)

# GQA kv-cache 上的注意力：解码步一次调用，注意力分数全程不物化
# （不会物化成显存里的中间张量），变长 batch 共享同一份 cache 张量
out = fusedtok.attention_decode(
    q_heads,                                    # [B, Hq, D] 新 token
    k_cache, v_cache,                           # [B, Hkv, T, D]
    lens=torch.tensor([1023, 512], dtype=torch.int32, device="cuda"))
# 连续 cache 逐 token 增长：先 append 新行，再解码
fusedtok.kv_append(k_cache, v_cache, k_new, v_new, lens)
out = fusedtok.attention_decode(q_heads, k_cache, v_cache, lens + 1)
# ……或跑在分页（vLLM 式块池）cache 上：池 [Nb, Hkv, P, D]
# + 每序列块表；每步先 append 新 token 再解码
fusedtok.kv_append_paged(k_pool, v_pool, block_table, k_new, v_new, lens)
out = fusedtok.attention_decode_paged(q_heads, k_pool, v_pool,
                                      block_table, lens + 1)
# 新序列 prefill（默认因果；便捷路径）
ctx = fusedtok.attention_prefill(q_all, k_all, v_all, causal=True)

# 采样这一侧：整个解码步一次融合调用
token = fusedtok.decode_step(logits, sampled_ids, penalty=1.1,
                             p=0.9, temperature=0.8, seed=step)
# 也可以分步执行：
logits = fusedtok.repetition_penalty(logits, sampled_ids, penalty=1.1)
token = fusedtok.sample_topp(logits, p=0.9, temperature=0.8, seed=step)
# top-k 采样变体（在 k 个幸存者内重新归一化）
token = fusedtok.sample_topk(logits, k=50, temperature=0.8, seed=step)
# 并发解码整批出 token：[行数, 词表] 进、每行一个（种子默认 0..行数-1）
tokens = fusedtok.sample_topp_batched(batch_logits, p=0.9,
                                      temperature=0.8)
# ……含逐行重复惩罚：ragged 历史，"扁平数组 + 偏移"是服务端快路径
tokens = fusedtok.decode_step_batched(batch_logits, flat_ids,
                                      penalty=1.1, p=0.9,
                                      ids_offsets=offsets)
```

一个最小的逐 token 采样循环：

```python
import torch, fusedtok as ft

h = torch.zeros(1, 4096, device="cuda")            # 解码状态
w = torch.load("rms_weight.pt").cuda()             # float32 权重
generated = []
for step in range(256):
    h = ft.rmsnorm(h, w, residual=h)               # 残差相加与归一化融合在同一个 kernel
    q = ft.rope(q, k=None, pos_offset=step, neox=True)
    logits = model_output(h)                       # 你的模型
    tok = ft.decode_step(logits, generated, penalty=1.1,
                         p=0.9, temperature=0.8, seed=step)
    generated.append(int(tok))
```

所有函数接受 float32 的 numpy 数组或 torch 张量（其他 dtype 会被拷贝转换），
返回同族的 float32 输出。CUDA torch 张量会自动选择零拷贝路径；所有搬运张量
数据的算子（elementwise / 归一化 / RoPE / attention）的 CUDA 张量支持
**bfloat16**，attention 类算子额外支持 **float16** —— kernel 内部以 float32
计算、在读写边界转换（norm 权重自动升精度到 float32；采样/选择类算子保持
float32）。

完整可运行的算子巡览见 `examples/demo.py`；分主题的使用手册见
[中文使用指南](https://github.com/Hai-Wenxiang/fusedtok/blob/main/docs/zh/usage.md)——
快速上手、执行模型（三条路径 / dtype / CUDA graph）、注意力、
采样契约、INT8 工作流、基准测试读法、常见问题与词汇表，每主题
一个页面；英文版在[这里](https://github.com/Hai-Wenxiang/fusedtok/blob/main/docs/en/usage.md)。

## 正确性

- 每个算子均附带 **CPU 参考实现** 与逐元素对拍测试（pytest）
- 无 GPU 的机器也能运行测试（CUDA 用例自动跳过）

## API 稳定性

1.0 冻结了当时的 34 个公开算子与辅助函数；此后新增算子均走小版本
发布，目前 `fusedtok.__all__` 共 38 个，1.x 系列内签名保持不变。
包内附带类型存根（`__init__.pyi`，PEP 561
`py.typed`）。破坏性变更需要新的大版本并保留一个
弃用窗口。确定性承诺：选择类并列取最靠前下标；采样按种子可复现
（批量版按逐行种子）。

## 性能基准

RTX 3060（sm_86）、float32、torch 零拷贝张量、CUDA event 计时（**独立 3 轮
取平均**，逐轮数值在 JSON 中），对比等价的 PyTorch 参考实现（组合 eager
表达式；attention 参考使用**预展开**头 —— `repeat_interleave` 在计时区
之外）。每算子取最大形状；完整数据：`docs/benchmarks/benchmark_rtx3060.json`，可用
`python benchmarks/bench.py` 复现：

| 算子 | 形状 | fusedtok | PyTorch 参考 | 加速比 |
|---|---|---:|---:|---:|
| attention_decode（GQA） | T=16384, D=128 | 874 µs | 7698 µs（SDPA） | **8.81x** |
| attention_decode_paged（GQA） | T=16384, D=128, P=16 | 972 µs | 7667 µs（SDPA） | **7.89x** |
| RoPE NeoX (q+k) | [8192×4096] | 1636 µs | 10013 µs | **6.12x** |
| kv_append（连续 cache 写入） | B=8, T=4096 | 14 µs | 46 µs（高级索引） | **3.31x** |
| RMSNorm（含残差） | [4096×4096] | 610 µs | 2052 µs | **3.36x** |
| sample_topp p=0.9（峰值分布） | [131072] | 148 µs | 382 µs（排序+掩码+multinomial） | **2.59x** |
| attention_decode bf16 | T=16384, D=128 | 856 µs | 1813 µs（SDPA bf16） | **2.12x** |
| sample_minp p=0.05（峰值分布） | [131072] | 144 µs | 209 µs（掩码+multinomial） | **1.45x** |
| SwiGLU | [4096×4096] | 609 µs | 1026 µs | **1.68x** |
| sample_topk k=50 | [131072] | 160 µs | 294 µs（topk+multinomial） | **1.84x** |
| top-k (k=50) | [131072] | 83 µs | 127 µs（CUB） | **1.54x** |
| LayerNorm | [4096×4096] | 449 µs | 612 µs | **1.36x** |
| top-k（k=4096，中段 k） | [131072] | 109 µs | 129 µs | **1.18x** |
| Softmax | [4096×4096] | 410 µs | 431 µs | **1.05x** |
| SiLU / GeLU / add | [4096×4096] | ~412-614 µs | ~408-607 µs | ~1.0x |
| argmax | [131072] | 56 µs | 46 µs | 0.81x（事件计时在 WDDM 上抖动大；墙上时钟探针 1.12x，见下） |
| int8 qgemm pc（W8A8） | [4096×4096×4096] | 3573 µs（38.5 TOPS） | 2061 µs（cuBLASLt + 广播） | 0.58x（如实） |
| int8 qgemm（IMMA） | [4096×11008×4096] | 9582 µs（38.5 TOPS） | 4510 µs（cuBLASLt） | 0.47x（如实） |
| attention_prefill（因果） | S=1024, D=128 | 5802 µs | 2612 µs（SDPA flash） | 0.45x（如实） |
| sample_minp p=0.05（宽核） | [131072] | 484 µs | 216 µs | 0.45x（如实：一次加宽重试加一次 32-64k 排序；torch 的布尔掩码组合式不用排序——min-p 的胜出场景见上方峰值行） |
| sample_topp p=0.9（平坦最坏） | [131072] | 1357 µs | 496 µs | 0.37x（如实，见下） |

批量采样（v1.4）与批量解码步（v1.5）—— 一次调用处理整个
`[8, 131072]` 批，参考为 torch 原生批量抽签（softmax + 2-D
multinomial；decode 行另加 gather 惩罚；逐轮数值在 JSON）：

| 算子 | 形状 | fusedtok | PyTorch 参考 | 加速比 |
|---|---|---:|---:|---:|
| sample_topk_batched k=50 | [8×131072] | 199 µs | 307 µs（topk+multinomial） | **1.54x** |
| sample_minp_batched p=0.05 | [8×131072] | 224 µs | 300 µs（掩码+multinomial） | **1.34x** |
| sample_topp_batched p=0.9 | [8×131072] | 268 µs | 219 µs（multinomial） | 0.82x（参考侧 WDDM 波动，见下） |
| decode_step_batched（惩罚 1.3，~64 token 历史） | [8×131072] | 315 µs | 269 µs（惩罚+softmax+multinomial） | 0.86x（对比逐行循环 **5.2x**，见下方说明） |
| sample_topp_batched（平坦最坏） | [8×131072] | 3608 µs | 215 µs | 0.06x（如实，同单行说明） |

按行 kernel（归一化、softmax）自 v0.4.1 起按形状在首次调用时自动调优
线程块大小；上表为调优后的数字。

![fusedtok 对比 PyTorch 参考](https://raw.githubusercontent.com/Hai-Wenxiang/fusedtok/main/docs/benchmarks/benchmark_rtx3060.png)

**RTX 5060 Ti（Blackwell，sm_120）** —— 同套测试，每算子最大形状（完整
数据：`docs/benchmarks/benchmark_rtx5060ti.json`）：

| 算子 | 形状 | fusedtok | PyTorch 参考 | 加速比 |
|---|---|---:|---:|---:|
| RoPE NeoX (q+k) | [8192×4096] | 1385 µs | 8371 µs | **6.05x** |
| attention_decode（GQA） | T=16384, D=128 | 574 µs | 2682 µs（SDPA） | **4.67x** |
| attention_decode_paged（GQA） | T=16384, D=128, P=16 | 627 µs | 2681 µs（SDPA） | **4.28x** |
| RMSNorm（含残差） | [4096×4096] | 505 µs | 1657 µs | **3.28x** |
| sample_topp p=0.9（峰值分布） | [131072] | 62 µs | 155 µs（排序+掩码+multinomial） | **2.48x** |
| kv_append（连续 cache 写入） | B=8, T=4096 | 9 µs | 21 µs（高级索引） | **2.20x** |
| sample_topk k=50 | [131072] | 45 µs | 93 µs（topk+multinomial） | **2.06x** |
| SwiGLU | [4096×4096] | 504 µs | 858 µs | **1.70x** |
| top-k (k=50) | [131072] | 27 µs | 41 µs（CUB） | **1.51x** |
| attention_decode bf16 | T=16384, D=128 | 546 µs | 641 µs（SDPA bf16） | **1.17x** |
| sample_minp p=0.05（峰值分布） | [131072] | 62 µs | 73 µs（掩码+multinomial） | **1.19x** |
| top-k（k=4096，中段 k） | [131072] | 50 µs | 54 µs（CUB） | 1.09x |
| LayerNorm / Softmax | [4096×4096] | ~346 µs | ~344-350 µs | ~1.0x |
| argmax | [131072] | 17 µs | 14 µs | 0.81x（事件计时有噪声；墙上时钟探针 0.96x） |
| int8 qgemm pc（W8A8） | [4096×4096×4096] | 2071 µs（66.3 TOPS） | 1139 µs（cuBLASLt + 广播） | 0.55x（如实） |
| attention_prefill（因果） | S=1024, D=128 | 3301 µs | 1421 µs（SDPA flash） | 0.43x（如实） |
| int8 qgemm（IMMA） | [4096×11008×4096] | 5465 µs（67.4 TOPS） | 2178 µs（cuBLASLt） | 0.40x（如实） |
| sample_minp p=0.05（宽核） | [131072] | 262 µs | 74 µs | 0.28x（如实：同 3060 行的宽核说明） |
| sample_topp p=0.9（平坦最坏） | [131072] | 1053 µs | 165 µs | 0.16x（如实，见下） |

批量采样（v1.4）与批量解码步（v1.5）同 `[8, 131072]` 形状：

| 算子 | 形状 | fusedtok | PyTorch 参考 | 加速比 |
|---|---|---:|---:|---:|
| sample_topk_batched k=50 | [8×131072] | 95 µs | 115 µs（topk+multinomial） | **1.21x** |
| sample_minp_batched p=0.05 | [8×131072] | 118 µs | 112 µs（掩码+multinomial） | 0.95x（持平） |
| sample_topp_batched p=0.9 | [8×131072] | 131 µs | 83 µs（multinomial） | 0.63x |
| decode_step_batched（惩罚 1.3，~64 token 历史） | [8×131072] | 142 µs | 99 µs（惩罚+softmax+multinomial） | 0.70x（对比逐行循环墙钟约 **4x**） |
| sample_topp_batched（平坦最坏） | [8×131072] | 1755 µs | 83 µs | 0.05x（如实） |

小形状下 Blackwell 的优势更大（softmax 1.7x、RMSNorm 3.1x @256 行、
attention decode 3.78x @T=4096 跑出约 187 GB/s）——形状越大启动开销占比
越低；完整扫描见 JSON。

![fusedtok 对比 PyTorch 参考（RTX 5060 Ti）](https://raw.githubusercontent.com/Hai-Wenxiang/fusedtok/main/docs/benchmarks/benchmark_rtx5060ti.png)

PyPI wheel 附带 sm_80/sm_86 原生 cubin 与 compute_86 PTX 回退 —— 已在
Blackwell（sm_120）驱动上验证 JIT 运行正确。

融合算子（RoPE / RMSNorm / SwiGLU）优势明显：eager 模式的中间张量要在显存间
来回搬运。v0.4 选择管线（到达票据 radix 轮 + 早退压缩，缓存 CUDA 图整管线
回放）在两张卡上小 k 场景均超过 torch 的 CUB radix select；v1.0 重调
（块内排序阈值与排序 chunk 双双从 2048 降到 1024 —— 单个 block 双调排序
（bitonic sort）2048 个 key 正是中段 k 退步的全部来源）让中段 k 窗口
也持平到领先。attention_decode 自 v1.1 起接受 bfloat16 / float16 cache：
kv-cache 字节减半，同 dtype 对比仍保持领先——batch=1 时 kernel 受延迟
限制，相对自身 f32 路径的绝对提升有限，batch 越大收益越大。
融合采样器在真实解码形态的 logits 上
胜过 eager 组合式（峰值行的组合参考本身在 WDDM 上逐轮波动约 15%，
逐轮数值在 JSON 里）；平坦分布下 sample_topp 如实落后 —— 此时核覆盖
约九成词表，管线实际上要给全词表排序。v1.2 用三个 token 逐位不变的
改动把该最坏情况耗时压到约 1/8.5（3060 上 n=131072 实测 18.2ms ->
2.2ms）：按 p×总质量 下界自适应跳窗；全词表快路径（窗口等于词表时
跳过选择阶段）；串行走查改为批量载入——承担契约的严格顺序浮点加法
一步不动（它正是与 CPU 对拍一致的确定性契约本身），只有数据载入被
流水化。v1.3 又用检查点二分再快约 1.6 倍（3060 上 2.2ms -> 1.4ms、
5060 Ti 上 1.6ms -> 1.0ms——第一趟走查记录前缀和、第二趟用二分定位
后续走，token 仍逐位不变）——但该场景仍是 torch 全并行排序占优。
sample_minp（v1.3）在尖峰 logits 上胜出，宽核行如实落后（一次加宽
重试加一次 32-64k 排序；torch 的布尔掩码组合式从不排序）——v1.4 给
min-p 补上了 top-p 自 v1.2 就有的自适应跳窗（用一个只算一次的全局
总量推出充分下界：宽核行直接跳过 x8 阶梯的中间档位，该行快 28-30%，
token 逐位不变）。
v1.4 的批量采样器把整个 `[行数, 词表]` 批一次调用送完：每行原封不动
地复用单行管线（逐行结果一致，扩窗循环按行跟踪完成状态），对比逐行
循环的收益纯粹来自批处理——B=8 在受提交开销限制的主机上墙上时钟
时间快 4-6 倍（3060：topp 1340 -> 274 µs、minp 1399 -> 237 µs；
5060 Ti：3.9x / 4.3x），尖峰 logits 下与 torch 原生批量 multinomial
处于同一档位（sample_topk_batched 明确胜出），平坦最坏则比单行版
再低一档、劣势同样如实标注。v1.5 把批处理扩展到整个解码步：
`decode_step_batched` 用逐行词表位图把逐行重复惩罚也装进同一融合
管线——B=8 尖峰行对比逐行循环 `decode_step` 快 5.2 倍（3060 上
1676 -> 321 µs，torch 原生"惩罚+softmax+multinomial"组合为 266 µs），
中尾行 3.1 倍。
attention_decode 在解码场景优势大（单次启动把 GQA cache 一遍流完，
而 SDPA 要额外做头展开且在小查询下效率偏低）；
attention_decode_paged（v1.2）为免碎片的 vLLM 式块池布局只付约
1.09-1.11x 的块表间接开销，切片调度一致时输出与连续版逐位相同；
attention_prefill 是如实的便捷路径，约为 SDPA flash 后端的 0.45x ——
设计上不用 tensor core，重度 prefill 请继续用 SDPA/FlashAttention。
kv_append（v1.3）单次启动把一个新 token 的 k/v 行写入连续 cache
（比手写高级索引快数倍，具体倍数随参考侧高级索引的耗时波动；算子
本身很小、受启动开销限制——量的是每步解码的固定成本）。
INT8 解码 GEMV 只搬运 fp16 投影一半的字节并跑满内存带宽（2 倍）；
流水线化 IMMA GEMM（v1.0 重写：cp.async 双缓冲 slab、运行时 tile 调优
64x64 / 128x128）在 3060 上约 38 TOPS、5060 Ti 上约 67 TOPS —— 是
v0.4 kernel 的 2-4 倍 —— 但 cuBLASLt（torch._int_mm）在逐张量行上仍保持
约 2.1-2.5 倍领先（W8A8 行的差距只有约 1.7-1.8 倍）：它的 tile 流水线更深、
epilogue 按架构精调。目前 qgemm 的定位是
精确 / 可图捕获 / 零拷贝的 INT8 路径，而非最快的路径 —— 数字诚实，
CUTLASS 级调度留作后续工作。逐通道变体（qgemm_perchannel，真实 INT8
推理所用的 W8A8 布局）把每输出通道的 scale 乘法融合进同一 epilogue，
kernel 侧零开销 —— 组合式 torch 参考要为广播乘法单独多跑一趟 kernel，
这正是它 0.55-0.58x 的来源。

自 1.1.1 起上表采样行测量**固定**的 logits（bench 播种 torch RNG）；v1.2 起
峰值行尖峰改为 +20、平坦行改用接近均匀的 logits —— 两种分布形态都
确定地名副其实（v1.1 的峰值行在 n=131072 恰好坐在覆盖边界上，逐种子
在两种形态间翻转）。argmax 行是对含主机同步调用的事件计时，在 WDDM
上摆动大（跨轮 0.73-1.24x）；把同步排除在计时环外的墙上时钟探针测得
1.12x（3060）/ 0.96x（5060 Ti）—— v1.2 为每次调用省掉一次 CUDA 提交
与一次分配。

## 开发

完整指南（测试规则、错误契约、确定性约定）见 [CONTRIBUTING.md](https://github.com/Hai-Wenxiang/fusedtok/blob/main/CONTRIBUTING.md)。快速上手：

```bash
# Windows：需在 VS 开发者命令行（vcvars64）中执行
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
# 在仓库根目录：PYTHONPATH 指向构建产物，conftest.py 会自动加 python/ 目录
$env:PYTHONPATH = "$PWD/build"        # Windows
PYTHONPATH=$PWD/build                 # Linux
python -m pytest tests -q
python benchmarks/bench.py            # GPU 基准测试 + 出图
```

- 支持 Windows / Linux
- Windows 下由 MSVC 配合 nvcc 编译
- CI 在每次推送时构建并运行 CPU 测试套件

## 路线图

- v0.2（已完成）：bf16 零拷贝、radix-select top-k/top-p、融合 top-p（nucleus）采样、
  单读 softmax、CUDA graph 验证
- v0.3（已完成）：chunk-merge 选择排序 + 并行 nucleus 计数、bf16x4/x8 向量化、
  INT8 量化/反量化工具
- v0.4.1（已完成）：按行 kernel（归一化/softmax）运行时线程块自动调优
- v0.4（已完成）：到达票据选择管线（无 cooperative launch、早退压缩、缓存 CUDA 图）、全库 stream 化（CUDA graph 真捕获）、INT8 计算路径（IMMA qgemm + 解码 GEMV）、融合 decode_step 采样
- v0.5（已完成）：attention —— GQA 解码注意力（连续 kv-cache、长 cache 自动 flash-decoding 切分、每序列长度）+ 分块 prefill 路径（诚实约 0.45x vs SDPA flash，定位便捷路径）；每 GPU 单图 benchmark；Windows wheel 进入 PyPI 发布管线
- 1.0（已发布）：流水线化 tensor-core INT8 GEMM（cp.async 双缓冲、运行时 tile 调优；3060 上 17 -> 39 TOPS）与逐通道权重 scale（W8A8）、融合 top-k 采样（vs topk+multinomial 组合式 2.1x）、top-k 中段 k 补平、文本卫生门禁、wheel 矩阵扩容（Linux cp310-313 / Windows cp311-313）、API 冻结
- 1.1（已发布）：半精度 attention —— `attention_decode` / `attention_prefill` 接受 bfloat16 与 float16 cache（float32 计算，解码路径字节减半）；并行 exp 预计算使平坦分布采样最坏情况耗时减半且 token 逐位不变
- 1.2（已发布）：分页 kv-cache attention —— `attention_decode_paged` 跑在 vLLM 式块池 `[Nb, Hkv, P, D]` + 每序列块表上（连续版 1.09-1.13x 开销，任意合法表均可）与 `kv_append_paged`（原地 cache 写入侧）；平坦分布采样最坏情况压到约 1/8.5（自适应跳窗 + 全词表快路径 + 批量载入串行走查，token 逐位不变）；argmax 减负（每次调用少一次提交少一次分配）
- 1.2.1（已发布）：审计驱动的加固 —— 修复选择工作区在超过 131072 词表（Qwen 级）时的越界；lens/块表/token id 改为主机侧校验 + 设备张量信任边界（带 `lens` 的 CUDA graph 捕获从此可用）；零拷贝路径补齐空输入与 dtype/连续性防护；基准带宽数字如实化（四行此前虚高 1.5 倍）；编译零警告（MSVC /W3 + GCC -Wall -Wextra）；文档重组为主题页并全面重写中文表述
- 1.3（已发布）：`sample_minp`（min-p 采样——相对 p_max 的值阈值核，无需全局质量归约，核宽度天然自适应）与 `kv_append`（连续 cache 的写入侧）；采样串行走查获得检查点二分（walk 1 记录前缀和、walk 2 二分续走——平坦最坏再压到约 1/1.6，token 逐位不变）；零拷贝助手拒绝 CPU 操作数（宿主指针进 kernel 会毒化 CUDA 上下文）
- 1.3.1（已发布）：审计驱动加固 —— 补齐 staged 路径的 lens/块表值校验（此前坏值会变成静默 GPU 越界写）与整数输入防护；修复两个潜伏采样走查 bug（stride≥2 检查点续走重复计数、自适应扩窗质量读错 workspace 字——中尾分布提速约 28% 且 token 逐位不变）；softmax 调优器封顶修复 sanitizer 门禁；kernel/启动代码清理合一；基准表全量重生成（新增 minp 峰值与 kv_append 行）与文档大修（陈旧数字、中文呆板残留、词汇表补条）
- 1.4（已发布）：批量采样 —— `sample_topp/minp/topk_batched` 一次调用采样整个 `[行数, 词表]` 批（每行与单行 API 一致、每行独立种子、各行按自己的核宽度分批完成）；min-p 获得自适应扩窗跳变（用一个只算一次的全局总量推出充分下界——宽核行直接跳过阶梯中间档位，token 逐位不变）
- 1.4.1（已发布）：审计驱动加固 —— 补齐 staged 路径批量采样的形状校验（此前 rows/n 直接信任、未对照缓冲区）；扩窗下界公式合一与批量时序器清理、两种扩窗模式统一惰性 totals 缓存、同步收窄到调用方流；基准表以发布版本戳重生成；文档大修（陈旧数字、呆板措辞、词汇表与 FAQ 补条）
- 1.5（已发布）：批量版 `decode_step` —— 逐行 ragged 历史经逐行惩罚位图，一次调用跑完"惩罚 -> 温度 -> 采样"整链。原 1.5 另一候选（批量尝试内的逐行独立窗口，让一个宽核行不再抬高整批统一窗口）做了三种实现、在 B=8/B=32 实测净损失或持平（串行逆 CDF 走查主导每轮开销、归并梯子每级有发射底价、按窗口分桶只会翻倍底价）后撤销，数字与结论记录在 CHANGELOG——若走查并行化可重开
- 后续候选（未排期）：bf16/fp16 tensor-core prefill（重写级）；CUTLASS 级 INT8 GEMM 调度（当前 qgemm 定位是精确/可图捕获/零拷贝路径，而非最快路径）；16-bit radix key（动确定性契约）

## 社区

- [贡献指南](https://github.com/Hai-Wenxiang/fusedtok/blob/main/CONTRIBUTING.md) —— 环境搭建、规则与 PR 流程
- [行为准则](https://github.com/Hai-Wenxiang/fusedtok/blob/main/CODE_OF_CONDUCT.md)
- [安全策略](https://github.com/Hai-Wenxiang/fusedtok/blob/main/SECURITY.md)
- [更新日志](https://github.com/Hai-Wenxiang/fusedtok/blob/main/CHANGELOG.md)

## 许可证

MIT —— 见 [LICENSE](https://github.com/Hai-Wenxiang/fusedtok/blob/main/LICENSE)；第三方声明见 [NOTICES.md](https://github.com/Hai-Wenxiang/fusedtok/blob/main/NOTICES.md)。
