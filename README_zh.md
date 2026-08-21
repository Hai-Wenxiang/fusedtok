# fusedtok

**面向 LLM 推理的融合 CUDA 算子库** —— RMSNorm / RoPE / SwiGLU，一行 `pip install` 即可用。

**English version: [README.md](README.md)**

## 为什么做

LLM 推理框架中，每个 token 都要触发大量小而受内存带宽限制的算子，每次启动都要在显存间来回搬运数据。

`fusedtok` 将它们融合为单一 kernel，削减显存流量与启动开销。

## 算子路线

| 状态 | 算子 | 说明 |
| :---: | --- | --- |
| ✅ | RMSNorm（含残差） | 暴力版，v0.1 |
| 🚧 | RoPE | v0.1 计划 |
| 🚧 | SwiGLU | v0.1 计划 |
| ⏳ | top-p / top-k 采样 | v0.2 计划 |
| ⏳ | INT8/FP8 量化路线 | v0.3 计划 |

## 安装

```bash
pip install fusedtok   # 尚未发布 —— v0.1 前请从源码构建
```

源码构建：

```bash
git clone https://github.com/Hai-Wenxiang/fusedtok.git
cd fusedtok
pip install .
```

**环境要求**：

- **RTX 30 系（Ampere）或更新**的 NVIDIA 显卡 —— 如 RTX 3060/3090、RTX 4080、RTX 5090、A100、H100
- CUDA Toolkit ≥ 12.0

<details>
<summary>什么是 "compute capability"（计算能力）？点击展开</summary>

计算能力是 NVIDIA 给每代 GPU 架构的**版本编号**，不是性能分数。CUDA 代码必须针对特定架构编译才能运行。
本库的 kernel 面向计算能力 8.0 及以上，因为所依赖的特性（如 `__nv_bfloat16`）从这一代才开始提供。

| 计算能力 | 架构 | 代表显卡 |
|---|---|---|
| 7.5 | Turing | GTX 16xx、RTX 20xx |
| 8.0 / 8.6 | Ampere | A100、RTX 30xx |
| 8.9 | Ada | RTX 40xx |
| 9.0 | Hopper | H100 |
| 12.0 | Blackwell | RTX 50xx、B200 |

查看自己的显卡：运行 `nvidia-smi` 看到型号后，在 https://developer.nvidia.com/cuda-gpus 查询对应计算能力。

</details>

## 用法（预览）

```python
import fusedtok

x = [1.0, 2.0, 3.0]
y = fusedtok.axpy(x, 2.0, 1.0, cuda=True)   # 当前骨架算子

# 对展平的 [rows, cols] 张量做 RMSNorm,可选融合残差
h = fusedtok.rmsnorm(x, w=[1.0, 1.0, 1.0], rows=1, cols=3, eps=1e-6,
                     residual=skip, cuda=True)
```

> **说明**：最终 API 目标是 torch 张量零拷贝；当前骨架使用纯列表以保持零依赖，便于框架搭建期学习。

## 正确性

- 每个算子均附带 **CPU 参考实现** 与逐元素对拍测试（pytest）
- 无 GPU 的机器也能运行测试（CUDA 用例自动跳过）

## 性能基准

随 v0.1 发布 —— 在 RTX 3060（sm_86）上对比 PyTorch eager。

## 开发

```bash
# Windows：需在 VS 开发者命令行（vcvars64）中执行
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
py -3.12 -m pytest tests -q      # 需将 build 目录加入 PYTHONPATH
```

- 支持 Windows / Linux
- Windows 下由 MSVC 配合 nvcc 编译
- CI 双平台构建

## 许可证

[MIT](LICENSE)
