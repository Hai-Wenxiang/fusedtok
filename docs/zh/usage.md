# fusedtok 中文使用指南

这里是 fusedtok 文档的目录页：每个链接打开一个聚焦的主题页。
算子总表与性能数据见 [README](../../README_zh.md)；每个算子的
可运行逐算子示例见 [`examples/demo.py`](../../examples/demo.py)。

**其他语言：** [English usage guide](../en/usage.md)

## 目录

| 页面 | 内容 |
|---|---|
| [快速上手](quickstart.md) | 安装、第一批 kernel、快路径初体验 |
| [执行模型](execution.md) | 三条执行路径、dtype 规则、流与 CUDA graph、错误契约 |
| [注意力算子](attention.md) | 连续与分页 kv-cache 上的解码注意力、append 写侧、prefill |
| [采样与选择](sampling.md) | top-k / top-p / min-p / argmax、融合采样器、同 token 确定性契约 |
| [INT8 路径](int8.md) | 量化工具、整数精确 qgemm、W8A8 逐通道 scale |
| [基准测试](benchmarks.md) | 测量协议、怎么读表（包括如实标注的落败行）、如何复现 |
| [常见问题](faq.md) | 显卡支持表、常见拒绝原因、计时注意事项、词汇表 |

## 一分钟版本

```python
import numpy as np
import torch
import fusedtok

x = np.random.randn(4, 1024).astype(np.float32)
w = np.random.rand(1024).astype(np.float32) + 0.5

y = fusedtok.rmsnorm(x, w)                    # CPU 参考实现
y = fusedtok.rmsnorm(x, w, cuda=True)         # 暂存式 CUDA
yt = fusedtok.rmsnorm(torch.from_numpy(x).cuda(),
                      torch.from_numpy(w).cuda())   # 零拷贝 CUDA
```

numpy 进 numpy 出，torch 进 torch 出。CUDA torch 张量自动选择
零拷贝路径：kernel 直接在 torch 自己的设备缓冲上运行、与其他
torch 操作保持流序、没有中转拷贝也没有主机同步。完整入门见
[快速上手](quickstart.md)。
