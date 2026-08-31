# fusedtok 中文使用指南

按主题组织的 fusedtok 使用文档。算子总表与性能数据见
[README](../../README_zh.md)；每个算子的可运行巡览见
[`examples/demo.py`](../../examples/demo.py)。本指南先讲清执行模型（所有算子
共用），再逐族讲解算子。

**其他语言：** [English usage guide](../en/usage.md)

- [安装与第一步](#安装与第一步)
- [三条执行路径](#三条执行路径)
- [dtype 支持](#dtype-支持)
- [流与 CUDA graph](#流与-cuda-graph)
- [注意力](#注意力)
- [归一化与 RoPE](#归一化与-rope)
- [elementwise 与激活](#elementwise-与激活)
- [采样与选择](#采样与选择)
- [INT8 路径](#int8-路径)
- [错误契约](#错误契约)
- [运行基准测试](#运行基准测试)

## 安装与第一步

```bash
pip install fusedtok
```

预编译 wheel 覆盖 Linux x86_64（cp310-cp313）与 Windows x86_64
（cp311-cp313），以 CUDA 12.4 构建。Ampere（RTX 30，sm_80/86）原生运行随包
cubin；更新的 GPU（RTX 40/50、H100 等）由驱动 JIT PTX fallback。源码构建需要
CUDA Toolkit >= 12.0 与 C++17 编译器。

确认扩展能看到可用 GPU：

```python
import fusedtok
print(fusedtok.cuda_available())   # 有可用 CUDA 设备时为 True
```

所有算子都能跑 CPU（C++ 编写的 float32 参考实现），同一份代码在无 GPU 的
机器上也能工作——写测试与调试时很方便。

## 三条执行路径

每个算子接受同样语义的三种输入形式，形式自动决定执行路径：

| 你传入 | 路径 | 发生什么 |
|---|---|---|
| numpy 数组（默认） | **CPU 参考** | C++ float32 参考实现，不碰 GPU |
| numpy 数组 + `cuda=True` | **暂存 CUDA** | 输入拷入 GPU、跑 kernel、拷回 |
| CUDA torch 张量 | **零拷贝 CUDA** | kernel 经 `data_ptr()` 直接读写 torch 缓冲，无暂存拷贝、无主机同步 |

```python
import numpy as np
import torch
import fusedtok

x = np.random.randn(4, 1024).astype(np.float32)
w = np.random.rand(1024).astype(np.float32) + 0.5

y1 = fusedtok.rmsnorm(x, w)                    # CPU 参考实现
y2 = fusedtok.rmsnorm(x, w, cuda=True)         # 暂存 CUDA（numpy 进出）

xt, wt = torch.from_numpy(x).cuda(), torch.from_numpy(w).cuda()
yt = fusedtok.rmsnorm(xt, wt)                  # 零拷贝：CUDA torch 进出
```

输出跟随输入族：numpy 进 numpy 出，CUDA torch 进 CUDA torch 出。推理循环要
的正是零拷贝路径——kernel 在 torch 当前流上启动，与其他 GPU 操作正确交错，
没有隐藏传输。

## dtype 支持

| 算子族 | numpy | CUDA torch |
|---|---|---|
| elementwise、激活、归一化、RoPE | float32 | float32、bfloat16 |
| `attention_decode`、`attention_prefill` | float32 | float32、bfloat16、float16 |
| 选择与采样（`topk`、`sample_*` 等） | float32 | float32 |
| INT8 算子（`qgemm` 等） | int8 操作数、float32 scale/输出 | 同左 |

值得记住的规则：

- 半精度输入保持 float32 **计算**：读入时加宽、写出时按最近邻舍入收窄。
  attention 的 softmax 与累加器在所有 dtype 下都是 float32，数值差异仅来自
  输入舍入。
- attention 的输出**匹配输入 dtype**；其他族同样如此（bf16 进 bf16 出）。
- 归一化的权重（`weight`、`bias`、`residual`）在激活为半精度时自动升精度到
  float32——检查点通常以 fp32 存这些参数。
- CPU/暂存路径恒为 float32（numpy 没有 bf16/fp16）。

## 流与 CUDA graph

零拷贝启动挂在 torch 的**当前流**上（`torch.cuda.current_stream()`），普通的
流顺序规则全部适用。整个库可被 CUDA graph 捕获：照常用 `torch.cuda.graph`
即可。

两个实操要点：

- **捕获前先做一次预热调用**。首次调用可能分配按形状缓存的 workspace
  （attention 切分路径、选择管线）或做配置调优（行 kernel 线程块、qgemm
  tile）——设计上它们都发生在捕获之外，预热把这两件事提前解决。
- 缓存进 graph 的 kernel 从设备内存读取每次调用的参数，因此 replay 能观察
  到两次 replay 之间写入张量的新内容。原地改写 + replay 会重新计算（测试
  有断言钉住这一点）。

```python
g = torch.cuda.CUDAGraph()
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        out = fusedtok.rmsnorm(xt, wt)         # 预热（调优/workspace）
torch.cuda.current_stream().wait_stream(s)
with torch.cuda.graph(g):
    out = fusedtok.rmsnorm(xt, wt)
g.replay()                                      # 整批一次 replay
```

## 注意力

### attention_decode——解码步

```python
out = fusedtok.attention_decode(q, k_cache, v_cache, lens)
```

- `q`：`[B, Hq, D]`——新 token 的各 query 头。
- `k_cache`、`v_cache`：`[B, Hkv, T, D]`——**连续** kv-cache（`T` 是分配
  行数，不一定是已用行数）。
- `lens`：可选 `[B]` int32（torch 张量或 numpy）——每个序列的有效 cache 长
  度。`None` 表示 `T` 行全部有效。长度为 0 的序列输出零行。

GQA 映射是**连续分组**：q 头 `h` 使用 kv 头 `h // (Hq // Hkv)`。
`Hq == Hkv` 退化为普通 MHA。这与 LLaMA 式检查点的布局一致（同组的 q 头相
邻存放）。

约束：`Hq % Hkv == 0`，`D` 为 4 的倍数且不超过 512。长 cache 自动切分
（flash-decoding 风格）：序列切成若干片、并行算 partial、再归并——无论 `T`
多大都只是一次调用。

性能定位：解码注意力受带宽约束（每个 token 把整个 kv-cache 流式读一遍）。
f32 大致跑到有效带宽对比 SDPA；bf16/fp16 cache 把字节减半。batch 为 1 时
kernel 受延迟约束，半精度省的绝对时间有限——字节节省随 batch 放大。

### attention_prefill——新序列

```python
ctx = fusedtok.attention_prefill(q_all, k_all, v_all, causal=True)
```

- `q`：`[B, Hq, S, D]`，`k`/`v`：`[B, Hkv, S, D]`；`causal=True` 时第 `i`
  行 query 只看 key 行 `[0, i]`，`causal=False` 时看全部。
- GQA/dtype/dim 规则与 decode 相同。

诚实的定位：这是**便捷路径**——单个分块 kernel、不用 tensor core。它让小
prefill 与混合负载留在 fusedtok 里；重型 prefill 属于 SDPA /
FlashAttention 的地盘（诚实比率见基准表）。

## 归一化与 RoPE

```python
h = fusedtok.rmsnorm(x, w, residual=r, eps=1e-6)
# y = (x + r) * rsqrt(mean((x + r)^2) + eps) * w
# residual=None 则不加；x 可以是 [rows, cols] 或 [cols]

y = fusedtok.layernorm(x, w, b, eps=1e-6)
# y = (x - mean) / sqrt(var + eps) * w + b

q_rot, k_rot = fusedtok.rope(q, k, theta=10000.0, pos_offset=0, neox=True)
```

`rmsnorm` 的融合 residual 是解码循环的主力：一个 kernel 读 `x` 与 `r`、
写出归一化后的和——没有中间张量。

RoPE 作用于 `[seq, dim]`、`dim` 为偶数：

- `neox=False`：交错配对 `(2j, 2j+1)`（RoFormer 原始形式）
- `neox=True`：行两半 rotate-half（GPT-NeoX / LLaMA-HuggingFace 检查点）
- `pos_offset` 是第 0 行的绝对位置——向已有序列解码时传 cache 长度
  （只处理 query 时 `k=None`）。
- `k` 可选（仅 q 调用返回 `(q_rot, None)`）。

## elementwise 与激活

`silu`、`gelu`（erf 形式）、`gelu_tanh`、`relu`、`tanh`、`sigmoid`、
`softmax`（按行、数值稳定）、`add`、`mul`，以及融合 MLP 门 `swiglu(gate,
up)`（= `silu(gate) * up`）。全部遵循上面的三条执行路径与 dtype 规则。
`temperature(x, t)` 缩放 logits；`axpy(x, a, b)` 一趟算出 `a * x + b`。

## 采样与选择

所有选择类算子平局取**最早下标**；采样**按种子确定**（splitmix 风格 hash
RNG——可复现，非密码学安全）。

```python
i = fusedtok.argmax(logits)                      # 平局取最早下标
vals, idxs = fusedtok.topk(logits, 50)           # 降序，(值, 下标)
vals, idxs = fusedtok.topp(probs, 0.9)           # 输入 = 概率
```

- `topk` 接受原始分数，`k` 取值 `[0, n]`。
- `topp` 接受**概率**（已 softmax），`p` 取值 `(0, 1]`；返回最小的 top-p
  集合（含跨越元素）。

融合采样器从 logits 到 token 一次 GPU 往返：

```python
tok = fusedtok.sample_topp(logits, p=0.9, temperature=0.8, seed=step)
tok = fusedtok.sample_topk(logits, k=50, temperature=0.8, seed=step)
tok = fusedtok.decode_step(logits, history, penalty=1.1,
                           p=0.9, temperature=0.8, seed=step)
```

- `sample_topp`：softmax(logits / T) -> 用**全局质量**阈值切核 -> 反 CDF
  抽取。
- `sample_topk`：softmax(logits / T) -> 保留 k 个 -> 在幸存者内重归一 ->
  抽取。`k=1` 恰为贪心；`k >= vocab` 采样整个分布。
- `decode_step`：对 `history` 施加 CTRL 式 `repetition_penalty`、再温度、
  再核采样——一次调用一次回读。与三个算子按同顺序组合的结果一致（同种
  子）。
- `repetition_penalty(logits, token_ids, penalty)` 也单独暴露：正 logit 除以
  `penalty`、负 logit 乘以 `penalty`（`penalty=1.0` 关闭）。

同 token 保证：固定种子下 CPU / 暂存 / 零拷贝三条路径抽出同一个 token
（已文档化的舍入边界：CPU 用精确 `exp`、GPU 用 `__expf`；parity 测试钉住
了可能出差异的位置——CDF 边界上的抽取可能差一个元素）。

平坦分布注意：当核覆盖几乎整个词表（接均匀的 logits）时，`sample_topp`
会反复扩窗，比 torch 的全并行排序慢——基准里如实记录。真实解码 logits
是尖峰状的；平坦情形是最坏情况，不是常态。

## INT8 路径

```python
q, scale = fusedtok.quantize_int8(x)        # 对称逐张量：
                                            # scale = max|x|/127
x_back = fusedtok.dequantize_int8(q, scale)
qy, s_out = fusedtok.qadd_int8(qa, sa, qb, sb)   # 融合 反量化-加-再量化
```

矩阵乘——LLM 友好布局（两个操作数都沿 K 行主序，`activations @
linear_weight.T` 无需转置）：

```python
y = fusedtok.qgemm(a_q, a_scale, b_q, b_scale)
# y[M, N] = (A_q[M, K] @ B_q[N, K]^T) * (a_scale * b_scale)

y = fusedtok.qgemm_perchannel(a_q, a_scale, b_q, b_scales)
# y[M, N] = (A_q @ B_q^T) * (a_scale * b_scales[j])   # W8A8
```

- `M == 1` 分发到带宽型 GEMV kernel（解码步）；更大 `M` 走带运行时 tile
  调优的 tensor-core IMMA 流水线。
- **精确性契约**：整数累加是精确 int32，合并 scale 在写出时乘一次——
  CPU、暂存、零拷贝三条路径的结果**逐位一致**。`qgemm_perchannel` 在
  `b_scales` 为常向量时与逐张量 `qgemm` 逐位相等。
- `qgemm_perchannel` 是真实 INT8 推理使用的布局（SmoothQuant /
  TensorRT-LLM 风格 W8A8）：每个输出通道一个 scale，吸收单一逐张量 scale
  吃不掉的权重离群值。
- 诚实的性能：大 GEMM 上 cuBLASLt（`torch._int_mm`）仍快约 2.2-2.6x；
  fusedtok 的 INT8 路径卖点是精确 / 可 graph 捕获 / 零拷贝。每 token 常见
  的解码 GEMV 以满带宽搬 fp16 投影一半的字节。

## 错误契约

自 1.0 起稳定（测试钉住）：

- `ValueError`——形状不匹配与取值越界（`k` 超出 `[0, n]`、`p` 超出
  `(0, 1]`、负的 `pos_offset`、`lens` 超过 `T` 等）
- `TypeError`——dtype 族或设备族错误（float64 输入、需要 CUDA 张量时给了
  CPU 张量、q/k/v 混用 dtype 等）
- `RuntimeError`——CUDA 执行失败（kernel 启动或驱动错误）

## 运行基准测试

```bash
python benchmarks/bench.py            # 全套，几分钟
```

协议：CUDA event 计时，每个配置 3 轮独立计时（各带预热），报告均值、逐轮
值保存在 JSON 里。产物落在 `docs/`：每 GPU 一份 JSON + 一张单面板加速比图
（文件名含设备名）。当前 RTX 3060 与 RTX 5060 Ti 的数字见 README 基准节。
