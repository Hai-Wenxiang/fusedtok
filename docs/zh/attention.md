# 注意力算子

fusedtok 提供五个注意力相关入口：解码步主力 `attention_decode`、
分页 cache 变体 `attention_decode_paged`（v1.2）、配套的两条写侧
`kv_append_paged`（v1.2）与 `kv_append`（v1.3，连续 cache），以及
prefill 便捷路径 `attention_prefill`。
这一页讲清各种布局、GQA 映射、变长 batch，以及性能上的诚实定位。

**其他语言：** [English: attention operators](../en/attention.md)

- [attention_decode——解码步](#attention_decode解码步)
- [attention_decode_paged——vLLM 式块池 cache](#attention_decode_pagedvllm-式块池-cache)
- [kv_append_paged——往池里写入新 token](#kv_append_paged往池里写入新-token)
- [kv_append——往连续 cache 写入新 token（v1.3）](#kv_append往连续-cache-写入新-tokenv13)
- [attention_prefill——新序列](#attention_prefill新序列)
- [性能定位](#性能定位)

## attention_decode——解码步

```python
out = fusedtok.attention_decode(q, k_cache, v_cache, lens)
```

- `q`：`[B, Hq, D]`——新 token 的各个 query 头。
- `k_cache` / `v_cache`：`[B, Hkv, T, D]`——**连续**排布的
  kv-cache（`T` 是分配的行数，不一定是已用行数）。
- `lens`：可选 `[B]` int32——每个序列的有效 cache 长度。
  `None` 表示 `T` 行全部有效。长度为 0 的序列输出零行。

GQA 映射是**连续分组**：q 头 `h` 对应 kv 头 `h // (Hq // Hkv)`；
`Hq == Hkv` 时就是普通 MHA。LLaMA 系检查点就是这个布局（同组的
q 头相邻存放）。

约束：`Hq` 必须是 `Hkv` 的整数倍；`D` 是 4 的倍数且不超过 512。
cache 很长时自动走 flash-decoding 式切分：序列切成若干片、各片
并行算出部分结果、再归并——无论 `T` 多大都只是一次调用。注意力
分数全程不物化（不会写到显存里落成中间张量）；q/K/V 各恰好读一遍。

`lens` 的取值校验只针对主机侧来源的输入（list、numpy、CPU 张量），
在上传前完成；CUDA 上的 `lens` 张量直接信任——读回主机需要同步
流、会破坏 CUDA graph 捕获（与裸设备指针同一信任边界）。图捕获前
在捕获外先热身一次该形状（切分 workspace 必须先存在）。

零拷贝路径的存储 dtype：float32、bfloat16、float16。半精度 cache
把解码步的搬运字节减半（这正是解码步的瓶颈），softmax 仍是
float32；输出 dtype 与输入一致。

## attention_decode_paged——vLLM 式块池 cache

```python
out = fusedtok.attention_decode_paged(q, k_pool, v_pool, block_table, lens)
```

同样的数学，跑在**分页**排布的 cache 上——也就是真实推理服务
在用的内存形态：

- `k_pool` / `v_pool`：`[Nb, Hkv, P, D]`——由定长 token 块组成
  的池（`P` = 每块 token 数，从池的形状读出）。不再给每个序列
  预留一段连续空间，token 分散在池中各处：序列增长、收缩、驱逐
  都不会让 cache 产生碎片。
- `block_table`：`[B, S]` int32——序列 `b` 的第 `t` 个 token 存
  在池块 `block_table[b, t // P]` 的偏移 `t % P` 处。**任意合法
  的表都可以**——块号乱序、块共享、有空洞都行，kernel 走间接
  寻址。
- `lens`：可选 `[B]`——每个序列的有效长度（`None` = 每个序列
  用满表宽 `S * P`）。

GQA 映射、零长度序列输出零行的约定、dtype 矩阵与连续版完全
一致；切分管线和 workspace 也是共用的。分页版特有的规则：

- GQA 组大小（`Hq // Hkv`）限 1/2/4/8/16（其他倍数请用连续版）。
- 主机侧来源的 `block_table` / `lens` 取值在上传前校验
  （`ValueError`）；设备上的张量直接信任（不同步流）。
- CUDA graph 捕获前同样需要捕获外热身一次。

间接寻址的实测开销（3060，b=1，GQA 32/8，D=128，T=16384，
P=16）：连续版的 **1.13 倍**（5060 Ti 上 1.09 倍；基准表同口径），
且切片调度一致时输出逐位相同。

## kv_append_paged——往池里写入新 token

```python
fusedtok.kv_append_paged(k_pool, v_pool, block_table, k_new, v_new, lens)
```

分页解码循环的 cache 写入侧：序列 `b` 的新行 `k_new[b]` /
`v_new[b]`（各 `[Hkv, D]`）写到池块 `block_table[b, lens[b] // P]`
的偏移 `lens[b] % P` 处。

- **原地写**（返回 `None`）。块表归调度器管：本算子只往已映射
  的块里写数据，绝不动表项。
- 主机路径要求 float32 且 C 连续的池——若传入其他 dtype 或非
  连续布局，转换会产生副本、写入被静默丢弃，所以直接以
  `TypeError` 拒绝而不是默默出错。torch 路径支持 f32/bf16/fp16
  全部存储组合。
- 一个微型 kernel、流序、可图捕获（照常先热身）。

典型循环：在 `lens[b]` 处 append，再以 `lens + 1` 解码：

```python
for step in range(n_steps):
    fusedtok.kv_append_paged(k_pool, v_pool, block_table,
                             k_new, v_new, lens)          # 写在 lens 处
    out = fusedtok.attention_decode_paged(q, k_pool, v_pool,
                                          block_table, lens + 1)
    lens += 1
    # ……下一个 token 的 k_new/v_new 由模型给出
```

## kv_append——往连续 cache 写入新 token（v1.3）

```python
fusedtok.kv_append(k_cache, v_cache, k_new, v_new, lens)
```

连续解码循环的 cache 写入侧（`kv_append_paged` 的孪生算子）：序列
`b` 的新行 `k_new[b]` / `v_new[b]`（各 `[Hkv, D]`）写到
`k_cache[b]` / `v_cache[b]` 的第 `lens[b]` 行。

- **原地写**（返回 `None`）；`lens` 必填（写入位置就是各序列的当前
  长度）。
- 主机路径要求 float32 且 C 连续的 cache——若传入其他 dtype 或
  非连续布局，转换会产生副本、写入被静默丢弃，所以直接以
  `TypeError` 拒绝。torch 路径支持 f32/bf16/fp16。
- 一个微型 kernel、流序、可 CUDA graph 捕获。
- 主机侧来源的 `lens` 取值在 `[0, T)` 内校验；设备上的张量直接信任
  （标准零拷贝信任边界）。
- 性能（基准表）：3060 上 16.7µs vs torch 高级索引 88.6µs
  （5.31x）、5060 Ti 上 9.4µs vs 20.8µs（2.21x）——算子本身很小、
  受启动开销限制，量的是每步解码的固定成本。

典型循环与分页版一致：在 `lens[b]` 处 append，再以 `lens + 1` 解码：

```python
for step in range(n_steps):
    fusedtok.kv_append(k_cache, v_cache, k_new, v_new, lens)
    out = fusedtok.attention_decode(q, k_cache, v_cache, lens + 1)
    lens += 1
    # ……下一个 token 的 k_new/v_new 由模型给出
```

## attention_prefill——新序列

```python
ctx = fusedtok.attention_prefill(q_all, k_all, v_all, causal=True)
```

- `q`：`[B, Hq, S, D]`，`k` / `v`：`[B, Hkv, S, D]`；`causal=True`
  时第 `i` 行 query 只看 key 的 `[0, i]` 行（prefill 对角线），
  `causal=False` 时看全部。
- GQA / dtype / 维度规则与 decode 相同。

诚实定位：这是**便捷路径**——单个分块 kernel，不用 tensor core。
它存在的意义是让小 prefill 和混合负载留在 fusedtok 里；重度
prefill 请交给 SDPA / FlashAttention（基准表里如实标着约 0.45x
的比值）。

## 性能定位

解码注意力受**显存带宽**约束：每个 token 都要把整个 kv-cache
顺序读一遍。可以期待的是：

- f32 decode 在长 cache 上达到或超过 SDPA 的有效带宽
  （README 表里 3060 @T=16384 最高 8.91x——参考实现需要额外做头
  展开，且在小 query 下效率偏低）。
- bf16/fp16 cache 把字节减半。batch 为 1 时 kernel 受延迟限制，
  绝对收益有限，batch 越大收益越大。
- 分页间接寻址比连续版多付 1.09–1.13 倍。
- prefill 刻意不与 flash 后端竞争。

测试协议与复现方法见[基准测试](benchmarks.md)。
