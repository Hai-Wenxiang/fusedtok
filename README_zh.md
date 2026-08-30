# fusedtok

[![CI](https://github.com/Hai-Wenxiang/fusedtok/actions/workflows/ci.yml/badge.svg)](https://github.com/Hai-Wenxiang/fusedtok/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/fusedtok.svg)](https://pypi.org/project/fusedtok/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/Hai-Wenxiang/fusedtok/blob/main/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://github.com/Hai-Wenxiang/fusedtok/blob/main/pyproject.toml)

**面向 LLM 推理的融合 CUDA 算子库** —— RMSNorm / RoPE / SwiGLU / 解码注意力
等，支持**torch 张量零拷贝**：对比 PyTorch SDPA 最高 **9.3 倍加速**
（attention decode，RTX 3060，见[性能基准](#性能基准)）。

**English version: [README.md](https://github.com/Hai-Wenxiang/fusedtok/blob/main/README.md)**

## 为什么做

LLM 推理框架中，每个 token 都要触发大量小而受内存带宽限制的算子，每次启动都要在显存间来回搬运数据。

`fusedtok` 将它们融合为单一 kernel，削减显存流量与启动开销。

## 算子路线

| 状态 | 算子 | 说明 |
| :---: | --- | --- |
| ✅ | RMSNorm（含残差） | LLaMA/Qwen 风格，融合残差加法 |
| ✅ | LayerNorm | 含仿射变换 |
| ✅ | RoPE | 交错与 NeoX 两种布局，支持 kv-cache `pos_offset` |
| ✅ | SwiGLU | 融合 MLP 激活 |
| ✅ | Softmax（按行） | 数值稳定版 |
| ✅ | SiLU / GeLU / GeLU-tanh / ReLU / Tanh / Sigmoid | 逐元素 |
| ✅ | add / mul | 逐元素二元（融合加残差模式） |
| ✅ | top-k / top-p（核采样） | 到达票据 radix + 早退压缩，缓存 CUDA 图整管线回放；平局取先下标（131k k=50 上 1.5x vs torch/CUB，双卡全 k 范围持平到领先） |
| ✅ | argmax / temperature | 贪心解码辅助 |
| ✅ | sample_topp | 融合核采样：softmax -> top-p -> 种子抽取，全局质量阈值 |
| ✅ | sample_topk | 融合 top-k 采样：softmax -> top-k -> 窗口内重归一 -> 种子抽取（131k 上 2.1x / 1.9x vs topk+multinomial 组合式） |
| ✅ | repetition penalty | CTRL 风格，作用于已采样 token |
| ✅ | decode_step | 整个解码步融合：惩罚 -> 温度 -> 核采样，一次调用一次回读 |
| ✅ | quantize_int8 / dequantize_int8 / qadd_int8 | 对称 per-tensor INT8，融合反量化-加-重量化 |
| ✅ | qgemm | INT8 矩阵乘，int32 精确：cp.async 双缓冲流水线 IMMA GEMM + 运行时 tile 调优（64x64 / 128x128）+ 每 warp 一行的 GEMV（M=1 解码；比 fp16 投影快 2 倍） |
| ✅ | qgemm_perchannel | 真实 INT8 推理所用的 W8A8 布局：逐输出通道权重 scale 融合进同一 kernel 的 epilogue，零开销 |
| ✅ | attention_decode | 解码步因果注意力：GQA + 连续 kv-cache，在线 softmax、长 cache 自动 flash-decoding 切分、支持每序列长度 |
| ✅ | attention_prefill | 新序列 S 行注意力（因果 / 双向）；便捷路径——重度 prefill 仍属 SDPA/flash 领地（诚实约 0.45x） |

## 安装

```bash
pip install fusedtok
```

PyPI 提供预编译 wheel（CUDA 12.4 构建）：**Linux x86_64**（manylinux，
cp310）与 **Windows x86_64**（cp312）。其他平台或 Python 版本 pip 会自动
从源码构建：

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

# GQA kv-cache 上的注意力：解码步一次调用，分数不落盘，
# 变长 batch 共享同一份 cache 张量
out = fusedtok.attention_decode(
    q_heads,                                    # [B, Hq, D] 新 token
    k_cache, v_cache,                           # [B, Hkv, T, D]
    lens=torch.tensor([1023, 512], dtype=torch.int32, device="cuda"))
# 新序列 prefill（默认因果；便捷路径）
ctx = fusedtok.attention_prefill(q_all, k_all, v_all, causal=True)

# 采样侧：整个解码步一次融合调用
token = fusedtok.decode_step(logits, sampled_ids, penalty=1.1,
                             p=0.9, temperature=0.8, seed=step)
# 或分步执行：
logits = fusedtok.repetition_penalty(logits, sampled_ids, penalty=1.1)
token = fusedtok.sample_topp(logits, p=0.9, temperature=0.8, seed=step)
# top-k 采样变体（在 k 个幸存者内重归一）
token = fusedtok.sample_topk(logits, k=50, temperature=0.8, seed=step)
```

一个最小的逐 token 采样循环：

```python
import torch, fusedtok as ft

h = torch.zeros(1, 4096, device="cuda")            # 解码状态
w = torch.load("rms_weight.pt").cuda()             # float32 权重
generated = []
for step in range(256):
    h = ft.rmsnorm(h, w, residual=h)               # 融合加 + 归一
    q = ft.rope(q, k=None, pos_offset=step, neox=True)
    logits = model_output(h)                       # 你的模型
    tok = ft.decode_step(logits, generated, penalty=1.1,
                         p=0.9, temperature=0.8, seed=step)
    generated.append(int(tok))
```

所有函数接受 float32 的 numpy 数组或 torch 张量（其他 dtype 会被拷贝转换），
返回同族的 float32 输出。CUDA torch 张量会自动选择零拷贝路径；CUDA 张量
也支持 **bfloat16** —— kernel 内部以 float32 计算、在读写边界转换
（norm 权重自动升精度到 float32；采样/选择类算子保持 float32）。

完整可运行的算子巡览见 `examples/demo.py`。

## 正确性

- 每个算子均附带 **CPU 参考实现** 与逐元素对拍测试（pytest）
- 无 GPU 的机器也能运行测试（CUDA 用例自动跳过）

## API 稳定性

1.0 冻结公开接口：`fusedtok.__all__` 中的 30 个算子与辅助函数在 1.x
系列内保持签名不变。包内附带类型存根（`__init__.pyi`，PEP 561
`py.typed`）。新算子走小版本发布；破坏性变更需要新的大版本并保留一个
弃用窗口。确定性承诺：选择类平局取最早下标；采样按种子可复现。

## 性能基准

RTX 3060（sm_86）、float32、torch 零拷贝张量、CUDA event 计时（**独立 3 轮
取平均**，逐轮数值在 JSON 中），对比等价的 PyTorch 参考实现（组合 eager
表达式；attention 参考使用**预展开**头 —— `repeat_interleave` 在计时区
之外）。每算子取最大形状；完整数据：`docs/benchmark_rtx3060.json`，可用
`python benchmarks/bench.py` 复现：

| 算子 | 形状 | fusedtok | PyTorch 参考 | 加速比 |
|---|---|---:|---:|---:|
| attention_decode（GQA） | T=16384, D=128 | 853 µs | 7614 µs（SDPA） | **8.92x** |
| RoPE NeoX (q+k) | [8192×4096] | 1641 µs | 10061 µs | **6.13x** |
| RMSNorm（含残差） | [4096×4096] | 614 µs | 2061 µs | **3.36x** |
| SwiGLU | [4096×4096] | 614 µs | 1025 µs | **1.67x** |
| top-k (k=50) | [131072] | 79 µs | 137 µs | **1.75x** |
| top-k（k=4096，中段 k） | [131072] | 113 µs | 127 µs | 1.12x |
| LayerNorm | [4096×4096] | 446 µs | 616 µs | **1.38x** |
| Softmax | [4096×4096] | 414 µs | 432 µs | 1.04x |
| SiLU / GeLU / add | [4096×4096] | ~412 µs | ~411 µs | ~1.0x |
| sample_topk k=50 | [131072] | 133 µs | 282 µs（topk+multinomial） | **2.13x** |
| argmax | [131072] | 65 µs | 45 µs | 0.69x（含主机回读） |
| int8 qgemm（IMMA） | [4096×4096×4096] | 3554 µs（38.7 TOPS） | 1634 µs（cuBLASLt） | 0.46x（诚实） |
| int8 qgemm pc（W8A8） | [4096×4096×4096] | 3553 µs（38.7 TOPS） | 2046 µs（cuBLASLt + 广播） | 0.58x（诚实） |
| attention_prefill（因果） | S=1024, D=128 | 5732 µs | 2560 µs（SDPA flash） | 0.45x（诚实） |

按行 kernel（归一化、softmax）自 v0.4.1 起按形状在首次调用时自动调优
线程块大小；上表为调优后的数字。

![fusedtok 对比 PyTorch 参考](https://raw.githubusercontent.com/Hai-Wenxiang/fusedtok/main/docs/benchmark_rtx3060.png)

**RTX 5060 Ti（Blackwell，sm_120）** —— 同套测试，每算子最大形状（完整
数据：`docs/benchmark_rtx5060ti.json`）：

| 算子 | 形状 | fusedtok | PyTorch 参考 | 加速比 |
|---|---|---:|---:|---:|
| RoPE NeoX (q+k) | [8192×4096] | 1384 µs | 8368 µs | **6.04x** |
| attention_decode（GQA） | T=16384, D=128 | 575 µs | 2682 µs（SDPA） | **4.67x** |
| RMSNorm（含残差） | [4096×4096] | 504 µs | 1657 µs | **3.29x** |
| SwiGLU | [4096×4096] | 504 µs | 858 µs | **1.70x** |
| top-k (k=50) | [131072] | 27 µs | 41 µs（CUB） | **1.50x** |
| top-k（k=4096，中段 k） | [131072] | 50 µs | 54 µs（CUB） | 1.09x |
| LayerNorm / Softmax | [4096×4096] | ~345 µs | ~348 µs | 1.0x |
| sample_topk k=50 | [131072] | 49 µs | 93 µs（topk+multinomial） | **1.91x** |
| argmax | [131072] | 17 µs | 14 µs | 0.83x（含主机回读） |
| int8 qgemm（IMMA） | [4096×4096×4096] | 2063 µs（66.6 TOPS） | 800 µs（cuBLASLt） | 0.39x（诚实） |
| int8 qgemm pc（W8A8） | [4096×4096×4096] | 2079 µs（66.1 TOPS） | 1142 µs（cuBLASLt + 广播） | 0.55x（诚实） |
| attention_prefill（因果） | S=1024, D=128 | 3291 µs | 1421 µs（SDPA flash） | 0.43x（诚实） |

小形状下 Blackwell 的优势更大（softmax 2.5x、RMSNorm 3.2x @256 行、
attention decode 3.8x @T=4096 跑出 235 GB/s）——形状越大启动开销占比
越低；完整扫描见 JSON。

![fusedtok 对比 PyTorch 参考（RTX 5060 Ti）](https://raw.githubusercontent.com/Hai-Wenxiang/fusedtok/main/docs/benchmark_rtx5060ti.png)

PyPI wheel 附带 sm_80/sm_86 原生 cubin 与 compute_86 PTX 回退 —— 已在
Blackwell（sm_120）驱动上验证 JIT 运行正确。

融合算子（RoPE / RMSNorm / SwiGLU）优势明显：eager 模式的中间张量要在显存间
来回搬运。v0.4 选择管线（到达票据 radix 轮 + 早退压缩，缓存 CUDA 图整管线
回放）在两张卡上小 k 场景均超过 torch 的 CUB radix select；v1.0 重调
（块内排序阈值与排序 chunk 双双从 2048 降到 1024 —— 单 block 位排序
2048 个 key 正是中段 k 退步的全部来源）让中段 k 窗口也持平到领先
（k=4096 @131k：1.12x / 1.09x）。attention_decode 在解码场景优势大（单次启动把 GQA cache 一遍流完，
有效带宽最高约 157 GB/s，而 SDPA 要付头展开或小查询低效的代价）；
attention_prefill 是诚实的便捷路径，约为 SDPA flash 后端的 0.45x ——
设计上不用 tensor core，重度 prefill 请继续用 SDPA/FlashAttention。
INT8 解码 GEMV 只搬运 fp16 投影一半的字节并跑满内存带宽（2 倍）；
流水线化 IMMA GEMM（v1.0 重写：cp.async 双缓冲 slab、运行时 tile 调优
64x64 / 128x128）在 3060 上约 39 TOPS、5060 Ti 上约 67 TOPS —— 是
v0.4 kernel 的 2-4 倍 —— 但 cuBLASLt（torch._int_mm）仍保持约 2.2-2.6 倍
领先：它的 tile 流水线更深、epilogue 按架构精调。目前 qgemm 的定位是
精确 / 可图捕获 / 零拷贝的 INT8 路径，而非最快的路径 —— 数字诚实，
CUTLASS 级调度留作后续工作。逐通道变体（qgemm_perchannel，真实 INT8
推理所用的 W8A8 布局）把每输出通道的 scale 乘法融合进同一 epilogue，
kernel 侧零开销 —— 组合式 torch 参考要单独付广播乘法的钱，这正是它
0.55-0.58x 的来源。

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

- v0.2（已完成）：bf16 零拷贝、radix-select top-k/top-p、融合核采样、
  单读 softmax、CUDA graph 验证
- v0.3（已完成）：chunk-merge 选择排序 + 并行核计数、bf16x4/x8 向量化、
  INT8 量化/反量化工具
- v0.4.1（已完成）：按行 kernel（归一化/softmax）运行时线程块自动调优
- v0.4（已完成）：到达票据选择管线（无 cooperative launch、早退压缩、缓存 CUDA 图）、全库 stream 化（CUDA graph 真捕获）、INT8 计算路径（IMMA qgemm + 解码 GEMV）、融合 decode_step 采样
- v0.5（已完成）：attention —— GQA 解码注意力（连续 kv-cache、长 cache 自动 flash-decoding 切分、每序列长度）+ 分块 prefill 路径（诚实约 0.45x vs SDPA flash，定位便捷路径）；每 GPU 单图 benchmark；Windows wheel 进入 PyPI 发布管线
- 1.0（开发中）：流水线化 tensor-core INT8 GEMM（cp.async 双缓冲、运行时 tile 调优；3060 上 17 -> 39 TOPS）与逐通道权重 scale（W8A8）、融合 top-k 采样（vs topk+multinomial 组合式 2.1x）、top-k 中段 k 补平、文本卫生门禁、wheel 矩阵扩容、API 冻结

## 社区

- [贡献指南](https://github.com/Hai-Wenxiang/fusedtok/blob/main/CONTRIBUTING.md) —— 环境搭建、规则与 PR 流程
- [行为准则](https://github.com/Hai-Wenxiang/fusedtok/blob/main/CODE_OF_CONDUCT.md)
- [安全策略](https://github.com/Hai-Wenxiang/fusedtok/blob/main/SECURITY.md)
- [更新日志](https://github.com/Hai-Wenxiang/fusedtok/blob/main/CHANGELOG.md)

## 许可证

MIT —— 见 [LICENSE](https://github.com/Hai-Wenxiang/fusedtok/blob/main/LICENSE)；第三方声明见 [NOTICES.md](https://github.com/Hai-Wenxiang/fusedtok/blob/main/NOTICES.md)。
