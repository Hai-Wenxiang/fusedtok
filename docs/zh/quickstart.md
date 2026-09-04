# 快速上手

这一页带你从 `pip install` 到把 kernel 跑上 GPU，几分钟即可完成。
如果你在评估这个库，从这里开始；想了解背后的执行机制，看
[执行模型](execution.md)；各个算子家族的深入内容见
[注意力](attention.md)、[采样](sampling.md)、[INT8](int8.md)。

**其他语言：** [English quickstart](../en/quickstart.md)

## 安装

```bash
pip install fusedtok
```

PyPI 上的预编译 wheel（CUDA 12.4 构建）覆盖 **Linux x86_64**
（manylinux，CPython 3.10–3.13）和 **Windows x86_64**（3.11–3.13）。
其他平台或 Python 版本，pip 会自动从源码构建：

```bash
git clone https://github.com/Hai-Wenxiang/fusedtok.git
cd fusedtok
pip install .
```

源码构建需要：RTX 30 系（Ampere）或更新的 NVIDIA 显卡、CUDA
Toolkit 12.0 以上、C++17 编译器。预编译 wheel 只需要配套的驱动。
完整的架构支持表和 RTX 40/50 的 JIT 兼容说明见
[常见问题](faq.md#支持哪些显卡)。

## 三十秒入门

```python
import numpy as np
import fusedtok

x = np.random.randn(4, 1024).astype(np.float32)
w = np.random.rand(1024).astype(np.float32) + 0.5

y = fusedtok.rmsnorm(x, w)              # CPU 参考实现，什么机器都能跑
y = fusedtok.rmsnorm(x, w, cuda=True)   # 暂存式：拷上 GPU 算完再拷回来
```

每个算子都同时接受 numpy 数组和 torch 张量（装了 torch 的话）。
传入 CUDA 上的 torch 张量时自动走**零拷贝路径**：kernel 直接读写
torch 自己的显存缓冲区，在 torch 当前流上启动，全程不经过主机中转。

```python
import torch

xt = torch.from_numpy(x).cuda()
wt = torch.from_numpy(w).cuda()
yt = fusedtok.rmsnorm(xt, wt)           # 零拷贝 CUDA，输出也在 GPU 上
```

## 体验一下快在哪

两个代表性算子，零拷贝路径上各只需一次调用：

```python
# GQA kv-cache 上的解码注意力：一次启动把整个 cache 顺序读一遍
q = torch.randn(1, 32, 128, device="cuda")              # [B, Hq, D]
k_cache = torch.randn(1, 8, 16384, 128, device="cuda")  # [B, Hkv, T, D]
v_cache = torch.randn(1, 8, 16384, 128, device="cuda")
lens = torch.tensor([16384], dtype=torch.int32, device="cuda")
out = fusedtok.attention_decode(q, k_cache, v_cache, lens)

# 整个解码步的采样链路合并成一次调用、一次回读
logits = torch.randn(131072, device="cuda")
token = fusedtok.decode_step(logits, [], penalty=1.1,
                             p=0.9, temperature=0.8, seed=0)

# 并发服务一整批：一次调用，每行按各自种子各出一个 token
# （把 logits 做出尖峰才像真实解码输出——平坦随机 logits 下批量
# 优势会缩小，见基准测试页）
batch_logits = torch.randn(8, 131072, device="cuda")
batch_logits[torch.arange(8, device="cuda"),
             batch_logits.argmax(dim=1)] += 20.0
tokens = fusedtok.sample_topp_batched(batch_logits, p=0.9)
```

仓库里的 `examples/demo.py` 会把每个算子都逐个演示一遍并与解析
参考对拍，本身就是一份可执行的文档。

## 接下来读什么

- [执行模型](execution.md) —— 三条执行路径、dtype 规则、流与
  CUDA graph、错误契约
- [注意力算子](attention.md) —— 解码注意力、连续与分页 append 写侧、prefill
- [采样与选择](sampling.md) —— top-k / top-p / min-p、融合采样器、确定性契约
- [INT8 路径](int8.md) —— 量化工具与整数精确矩阵乘
- [基准测试](benchmarks.md) —— 数字怎么测的、表格怎么读
- [常见问题](faq.md) —— 排错与词汇表
