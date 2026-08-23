# fusedtok

[![CI](https://github.com/Hai-Wenxiang/fusedtok/actions/workflows/ci.yml/badge.svg)](https://github.com/Hai-Wenxiang/fusedtok/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/fusedtok.svg)](https://pypi.org/project/fusedtok/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/Hai-Wenxiang/fusedtok/blob/main/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://github.com/Hai-Wenxiang/fusedtok/blob/main/pyproject.toml)

**面向 LLM 推理的融合 CUDA 算子库** —— RMSNorm / RoPE / SwiGLU 等，支持
**torch 张量零拷贝**：对比 PyTorch eager 最高 **6.2 倍加速**（RoPE，
RTX 3060，见[性能基准](#性能基准)）。

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
| ✅ | top-k / top-p（核采样） | radix-select，平局取先下标（131k 词表 1.4x） |
| ✅ | argmax / temperature | 贪心解码辅助 |
| ✅ | sample_topp | 融合核采样：softmax -> top-p -> 种子抽取，单 kernel |
| ✅ | repetition penalty | CTRL 风格，作用于已采样 token |
| ⏳ | INT8/FP8 量化路线 | v0.3 计划 |

## 安装

```bash
pip install fusedtok
```

PyPI 提供 Linux x86_64 预编译 wheel（manylinux，CUDA 12.4 构建）。
Windows（或无匹配 wheel 的平台）下 pip 会自动从源码构建：

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

# 采样侧：惩罚 + 融合核采样（单 kernel，可复现种子）
logits = fusedtok.repetition_penalty(logits, sampled_ids, penalty=1.1)
token = fusedtok.sample_topp(logits, p=0.9, temperature=0.8, seed=step)
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
    logits = ft.repetition_penalty(logits, generated, 1.1)
    tok = ft.sample_topp(logits, p=0.9, temperature=0.8, seed=step)
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

## 性能基准

RTX 3060（sm_86）、float32、torch 零拷贝张量、CUDA event 计时，对比等价的
PyTorch eager 表达式（完整数据：`docs/benchmark_rt3060.json`，可用
`python benchmarks/bench.py` 复现）：

| 算子 | 形状 | fusedtok | PyTorch eager | 加速比 |
|---|---|---:|---:|---:|
| RoPE NeoX (q+k) | [2048×4096] | 416 µs | 2570 µs | **6.2x** |
| RMSNorm（含残差） | [1024×4096] | 260 µs | 538 µs | **2.1x** |
| SwiGLU | [1024×4096] | 153 µs | 257 µs | **1.7x** |
| LayerNorm | [1024×4096] | 168 µs | 162 µs | ~1.0x |
| SiLU | [1024×4096] | 105 µs | 112 µs | ~1.0x |
| Softmax | [1024×4096] | 159 µs | 115 µs | 0.7x |
| argmax | [131072] | 36 µs | 46 µs | **1.3x** |
| top-k (k=50) | [131072] | 168 µs | 129 µs | 0.8x |

![fusedtok 对比 PyTorch eager](https://raw.githubusercontent.com/Hai-Wenxiang/fusedtok/main/docs/benchmark_rt3060.png)

**RTX 5060 Ti（Blackwell，sm_120）** —— 同套测试，torch 2.11/cu128，亮点：

| 算子 | 形状 | fusedtok | PyTorch eager | 加速比 |
|---|---|---:|---:|---:|
| RoPE NeoX (q+k) | [512×4096] | 29 µs | 240 µs | **8.3x** |
| RMSNorm（含残差） | [4096×4096] | 512 µs | 1662 µs | **3.3x** |
| Softmax | [1024×4096] | 20 µs | 51 µs | **2.6x** |
| SwiGLU | [4096×4096] | 504 µs | 858 µs | **1.7x** |
| argmax | [32000] | 11 µs | 22 µs | **1.9x** |
| LayerNorm | [1024×4096] | 27 µs | 28 µs | ~1.0x |

![fusedtok 对比 PyTorch eager（RTX 5060 Ti）](https://raw.githubusercontent.com/Hai-Wenxiang/fusedtok/main/docs/benchmark_rt5060ti.png)

PyPI wheel 附带 sm_80/sm_86 原生 cubin 与 compute_86 PTX 回退 —— 已在
Blackwell（sm_120）驱动上验证 JIT 运行正确。

融合算子（RoPE / RMSNorm / SwiGLU）优势明显：eager 模式的中间张量要在显存间
来回搬运。纯带宽受限的逐元素算子与 PyTorch 调优 kernel 跑出相同的
~330-500 GB/s（silu、gelu、add ≈ 持平）。Softmax 与 top-k 仍落后于 PyTorch
的 CUB 内核 —— 数字诚实，改进列入 v0.2 规划。

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
- v0.4：decoupled-lookback 选择（多 SM 卡达 CUB 级）、INT8 GEMM、
  block size 自动调优
- v0.4+：轻量融合 attention；PyPI 预编译 wheel

## 社区

- [贡献指南](https://github.com/Hai-Wenxiang/fusedtok/blob/main/CONTRIBUTING.md) —— 环境搭建、规则与 PR 流程
- [行为准则](https://github.com/Hai-Wenxiang/fusedtok/blob/main/CODE_OF_CONDUCT.md)
- [安全策略](https://github.com/Hai-Wenxiang/fusedtok/blob/main/SECURITY.md)
- [更新日志](https://github.com/Hai-Wenxiang/fusedtok/blob/main/CHANGELOG.md)

## 许可证

MIT —— 见 [LICENSE](https://github.com/Hai-Wenxiang/fusedtok/blob/main/LICENSE)；第三方声明见 [NOTICES.md](https://github.com/Hai-Wenxiang/fusedtok/blob/main/NOTICES.md)。
