# INT8 路径

fusedtok 的 INT8 路径覆盖量化存储与量化计算：对称逐张量量化
工具、整数精确的 `qgemm` 矩阵乘，以及真实 INT8 推理部署所用的
逐通道变体 `qgemm_perchannel`（W8A8）。

**其他语言：** [English: the INT8 path](../en/int8.md)

- [量化工具](#量化工具)
- [矩阵乘](#矩阵乘)
- [精确性契约](#精确性契约)
- [性能，如实说](#性能如实说)

## 量化工具

```python
q, scale = fusedtok.quantize_int8(x)     # scale = max|x|/127，
                                         # 每条路径都返回 Python float
x_back = fusedtok.dequantize_int8(q, scale)
qy, s_out = fusedtok.qadd_int8(qa, sa, qb, sb)   # 融合 反量化-加-再量化
```

- `quantize_int8`：对称逐张量——`scale = max|x| / 127`，
  `q = clamp(round(x / scale), -127, 127)`。scale 在源头就读回
  主机一次（所有消费方要的都是主机 float），因此每条路径的
  返回类型完全一致。
- `dequantize_int8(q, scale)`：`x ~= q * scale`。接受拆开的一对：
  `dequantize_int8(*quantize_int8(x))`。零拷贝路径要求 int8 且
  C 连续（dtype 不对会被当原始字节读）。
- `qadd_int8(qa, sa, qb, sb)`：按 float32 计算 `qa*sa + qb*sb`，
  再用输出自己的逐张量 scale 重新量化——一遍 kernel 走完，不用
  反量化 -> 相加 -> 量化的三趟往返。

`quantize_int8` / `qadd_int8` 是异步启动器契约里唯一注明过的
例外：第二遍 kernel 要用主机上的 absmax 来组织，所以调用中途
会同步一次调用方的流（拷贝走调用方的流、且带错误检查）。

## 矩阵乘

两个操作数都沿 K 行主序——对 LLM 最友好的布局，
`activations @ linear_weight.T` 不需要转置：

```python
y = fusedtok.qgemm(a_q, a_scale, b_q, b_scale)
# y[M, N] = (A_q[M, K] int8 @ B_q[N, K] int8 ^T) * (a_scale * b_scale)

y = fusedtok.qgemm_perchannel(a_q, a_scale, b_q, b_scales)
# y[M, N] = (A_q @ B_q^T) * (a_scale * b_scales[j])    # W8A8
```

- `M == 1` 时走带宽型的每 warp 一行 GEMV kernel（解码步的投影）；
  更大的 `M` 走带运行时 tile 调优的 tensor-core IMMA 流水线
  （64x64 或 128x128 tile，cp.async 双缓冲）。
- `qgemm_perchannel` 是真实 INT8 推理用的布局（SmoothQuant /
  TensorRT-LLM 风格的 W8A8）：激活带一个逐张量 scale，权重带
  **每个输出通道一个** scale（`b_scales[j]`，长度 N 的 float32
  向量）。逐通道 scale 能吃掉单一逐张量 scale 消化不了的权重
  离群值——端到端测试量化了它在尖刺权重上 5 倍以上的误差改善。

零拷贝路径要求 int8 且 C 连续的操作数（scale 向量为 float32）；
非连续张量会被拒绝，而不是被读错。

## 精确性契约

整数累加是精确的 int32，合并 scale 只在写出时乘一次：CPU、
暂存、零拷贝三条路径的结果**逐位一致**。`qgemm_perchannel`
的逐元素 scale 按 `float32(a_scale * b_scales[j])` 单次舍入合成，
与 CPU 参考的顺序一致；`b_scales` 为常向量时，输出与逐张量
`qgemm` 逐位相等。这两条都有测试钉住。

不玩容差游戏：INT8 结果跨路径不一致就是 bug。

## 性能，如实说

- **解码 GEMV**（`M == 1`）只搬运 fp16 投影一半的字节并跑满
  内存带宽——约为 fp16 投影的 2 倍。这是每个 token 的热路径，
  也是 INT8 权重的意义所在。
- **流水线化 IMMA GEMM** 在 3060 上约 39 TOPS、5060 Ti 上约
  67 TOPS（是 v0.4 kernel 的 2–4 倍），但 cuBLASLt
  （`torch._int_mm`）凭借更深的 tile 流水线和按架构精调的
  epilogue 仍领先约 2.1–2.6 倍。fusedtok 的 INT8 路径定位是
  **精确 / 可图捕获 / 零拷贝**，不是最快；CUTLASS 级调度在
  路线图上。
- `qgemm_perchannel` 的 scale 乘法融合进同一个 epilogue，kernel
  侧零开销——反而是组合式 torch 参考要为 scale 广播单独付钱，
  它的比值就是这么来的。
