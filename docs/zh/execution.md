# 执行模型

fusedtok 的所有算子共用一套执行模型。这一页把它一次讲清楚，
各主题页就能专注于算子本身：三条分发路径、dtype 规则、流与
CUDA graph 的行为、以及错误契约。

**其他语言：** [English: the execution model](../en/execution.md)

## 三条执行路径

每个算子接受同样语义的三种输入形式，输入形式自动决定执行路径：

| 你传入 | 路径 | 实际发生什么 |
|---|---|---|
| numpy 数组（默认） | **CPU 参考实现** | C++ float32 参考实现，完全不碰 GPU |
| numpy 数组 + `cuda=True` | **暂存式 CUDA** | 输入拷上 GPU、跑 kernel、结果拷回 |
| CUDA torch 张量 | **零拷贝 CUDA** | kernel 经 `data_ptr()` 直接读写 torch 的显存缓冲——没有中转拷贝，没有主机同步 |

```python
import numpy as np, torch, fusedtok

x = np.random.randn(4, 1024).astype(np.float32)
w = np.random.rand(1024).astype(np.float32) + 0.5

y1 = fusedtok.rmsnorm(x, w)                # CPU 参考实现
y2 = fusedtok.rmsnorm(x, w, cuda=True)     # 暂存式 CUDA（numpy 进出）

xt, wt = torch.from_numpy(x).cuda(), torch.from_numpy(w).cuda()
yt = fusedtok.rmsnorm(xt, wt)              # 零拷贝：CUDA torch 进出
```

输出跟随输入的"家族"：numpy 进 numpy 出，CUDA torch 进 CUDA torch
出（CPU torch 进则经参考路径出 CPU torch）。

推理循环要的就是零拷贝路径：kernel 挂在 torch 的**当前流**上
（`torch.cuda.current_stream()`），与其他 GPU 操作按普通的流顺序
协作，没有任何隐藏传输。

CPU 参考实现是正确性的基准：它实现的是同一套算法（在需要的地方
连累加顺序都一致——见[采样契约](sampling.md#同-token-保证)），
无 GPU 的机器也能跑，测试对拍全靠它。

## dtype 规则

| 算子家族 | numpy | CUDA torch |
|---|---|---|
| 逐元素、激活、归一化、RoPE | float32 | float32、bfloat16 |
| `attention_decode`、`attention_decode_paged`、`attention_prefill`、`kv_append`、`kv_append_paged` | float32 | float32、bfloat16、float16 |
| 选择与采样（`topk`、`sample_*` 等） | float32 | float32 |
| INT8 算子（`qgemm` 等） | int8 操作数、float32 scale/输出 | 同左 |

几条值得记住的规则：

- **半精度输入、float32 计算。** 读入时在内存边界升到 float32，
  写出时按「最近偶数舍入」（round-to-nearest-even）规则舍回半精度。
  attention 的 softmax 和所有累加器在任何 dtype 下都是 float32，
  数值差异只来自输入本身的舍入。
- **输出 dtype 跟随输入**：bf16 进 bf16 出；attention 额外支持
  fp16 的往返。
- 归一化的权重（`weight`、`bias`、`residual`）在激活为半精度时
  自动升到 float32——模型权重（checkpoint）通常本来就是按 fp32
  存这些参数的。
- CPU / 暂存路径恒为 float32（numpy 没有 bf16/fp16）。
- 其他 dtype（float64 等）在 CPU / 暂存路径会被拷贝转换成
  float32；零拷贝路径直接报 `TypeError` 拒绝。

## 流与 CUDA graph

零拷贝启动挂在调用方的当前流上，torch 的流语义全部适用。整个
库都支持 CUDA graph 捕获，照常使用 `torch.cuda.graph` 即可。两条
实操要点：

1. **捕获前先热身。** 首次调用可能会为该形状分配 workspace
   （attention 切分路径、选择管线）或微基准测试启动配置（行
   kernel 的线程块大小、qgemm tile）。这些动作设计上只发生在
   捕获之外，热身一次就把它们解决掉。
2. **进图的 kernel 从设备内存读取每次调用的参数。** 两次 replay
   之间写进张量的新内容，下一次 replay 能看到；原地改写 +
   replay 会重新计算（测试钉住了这一点）。而以 kernel 参数形式
   传递的每次调用**数值**（比如采样的 `seed`）和任何 kernel
   参数一样，在捕获时就固定了。

```python
g = torch.cuda.CUDAGraph()
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        out = fusedtok.rmsnorm(xt, wt)     # 热身（调优/workspace）
torch.cuda.current_stream().wait_stream(s)
with torch.cuda.graph(g):
    out = fusedtok.rmsnorm(xt, wt)
g.replay()                                  # 整批一次 replay
```

设计上明确的例外（各算子文档均有注明）：

- 融合采样器返回主机端的 `int`，每次调用以一次很小的
  设备到主机回读收尾——它们本来就不打算被捕获。
- 批量采样器（`sample_*_batched`）返回主机侧 int64 张量/数组，
  而且扩窗循环要根据回读结果重新发射 kernel——同样不可捕获
  （契约一致）。
- `quantize_int8` / `qadd_int8` 必须把归约出的 scale 读回主机
  才能组织第二遍 kernel，所以调用中途会同步一次调用方的流。
- 零拷贝路径上，整数输入（attention 的 `lens`、分页的
  `block_table`、重复惩罚的 token id）如果本身就是 CUDA 张量，
  **直接信任**——校验它们的值需要同步流，而同步会破坏图捕获。
  主机侧来源的值（list、numpy、CPU 张量）则在上传前校验。

## 错误契约

自 1.0 起稳定，`tests/test_api.py` 钉死：

| 异常 | 触发场景 | 例子 |
|---|---|---|
| `ValueError` | 形状与取值范围 | `k` 超出 `[0, n]`、`p` 超出 `(0, 1]`、负的 `pos_offset`、`lens` 超过 cache 长度、形状不匹配、主机侧来源的 `lens`/`block_table` 取值越界、非连续张量（见下） |
| `TypeError` | dtype / 设备族问题 | 零拷贝路径收到 float64、主输入在 CUDA 而某个操作数在 CPU、q/k/v dtype 不一致 |
| `RuntimeError` | CUDA 执行失败 | kernel 启动错误、驱动错误、拷贝失败 |

注意连续性规则：零拷贝 kernel 直接按地址访问显存，非连续张量
会被拒绝（`ValueError`）而不是被读错——需要时请显式传
`.contiguous()` 的视图。

## 接下来

- [注意力算子](attention.md)
- [采样与选择](sampling.md)
- [INT8 路径](int8.md)
