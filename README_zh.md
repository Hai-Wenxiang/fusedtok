# fusedtok

**面向 LLM 推理的融合 CUDA 算子库** — RMSNorm / RoPE / SwiGLU,一行 `pip install` 即可用。

**English version: [README.md](README.md)**

## 为什么做

LLM 推理框架中每个 token 要触发大量小而受内存带宽限制的算子,每次启动都要在显存间来回搬运。
`fusedtok` 将它们融合为单一 kernel,削减显存流量与启动开销。

## 算子路线

| 状态 | 算子 | 说明 |
|---|---|---|
| 🚧 | RMSNorm(含残差) | v0.1 计划 |
| 🚧 | RoPE | v0.1 计划 |
| 🚧 | SwiGLU | v0.1 计划 |
| ⏳ | top-p / top-k 采样 | v0.2 计划 |
| ⏳ | INT8/FP8 量化路线 | v0.3 计划 |

## 安装

```bash
pip install fusedtok   # 尚未发布 — v0.1 前请从源码构建
```

源码构建:

```bash
git clone https://github.com/Hai-Wenxiang/fusedtok.git
cd fusedtok
pip install .
```

环境要求:CUDA Toolkit >= 12.0,计算能力 >= 8.0(Ampere 及以上)。

## 用法(预览)

```python
import fusedtok

x = [1.0, 2.0, 3.0]
y = fusedtok.axpy(x, 2.0, 1.0, cuda=True)   # 当前骨架算子
```

> 最终 API 目标是 torch 张量零拷贝;当前骨架用纯列表保持零依赖,便于框架搭建期学习。

## 正确性

每个算子都附带 CPU 参考实现和逐元素对拍测试(pytest)。
无 GPU 机器也能跑测试(CUDA 用例自动跳过)。

## 性能基准

随 v0.1 发布 — 在 RTX 3060 (sm_86) 上对比 PyTorch eager。

## 开发

```bash
# Windows:需在 VS 开发者命令行(vcvars64)中执行
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
py -3.12 -m pytest tests -q      # 需将 build 目录加入 PYTHONPATH
```

支持 Windows / Linux。Windows 下 MSVC 配合 nvcc;CI 双平台构建。

## 许可证

MIT
