# 常见问题

**其他语言：** [English FAQ](../en/faq.md)

## 支持哪些显卡？

Ampere（RTX 30 系，计算能力 8.0/8.6）及更新架构。wheel 附带
sm_80/sm_86 原生 cubin 外加 compute_86 PTX 回退，更新的架构
（RTX 40/50、H100）由驱动 JIT 运行——已在 Blackwell（sm_120）
上验证。Turing（GTX 16xx / RTX 20xx，cc 7.5）及更早架构不支持。

| 计算能力 | 架构 | 代表显卡 | fusedtok 的运行方式 |
|---|---|---|---|
| 7.5 | Turing | GTX 16xx、RTX 20xx | 不支持 |
| 8.0 / 8.6 | Ampere | A100、RTX 30xx | 原生 cubin |
| 8.9 | Ada | RTX 40xx | PTX JIT |
| 9.0 | Hopper | H100 | PTX JIT |
| 12.0 | Blackwell | RTX 50xx | PTX JIT |

先跑 `nvidia-smi` 看型号，再到
https://developer.nvidia.com/cuda-gpus 查计算能力。

## 明明有 GPU 却提示 "no CUDA device"？

- `fusedtok.cuda_available()` 返回 False 说明 CUDA 上下文建不起
  来：确认 `nvidia-smi` 能跑、驱动版本配得上你的 CUDA 运行时
  （12.0 级以上）、机器上确实看得到设备。
- 没有 GPU 时 CPU 参考功能全部可用——只有 CUDA 路径需要卡。

## 为什么第一次调用偏慢？

每个形状的首次调用可能做一次性工作：分配 workspace
（attention 切分路径、选择管线）或微基准测试启动配置（行
kernel 线程块、qgemm tile）。选择结果会按进程缓存。捕获
CUDA graph 或计时之前先热身一次（见
[执行模型](execution.md#流与-cuda-graph)）。

## 张量为什么被拒绝了？

| 报错 | 常见原因 | 解决 |
|---|---|---|
| `TypeError: ... must be float32, bfloat16 or float16` | 零拷贝路径不支持的 dtype | 用 `.to(torch.float32)` 转换（或支持的 bf16） |
| `ValueError: ... must be contiguous` | 零拷贝路径收到跨步视图（kernel 按裸指针访问） | 传 `.contiguous()` |
| `TypeError: ... must be a CUDA tensor when the primary input is on CUDA` | 设备混用 | 把所有操作数搬到同一设备 |
| `ValueError: lens entries must be in [0, T]` | 主机侧 `lens` 越界 | 改正数值（CUDA 上的 `lens` 张量直接信任——请自行校验） |
| `RuntimeError: ... launch ...` | CUDA 侧失败 | 用 `nvidia-smi` 查上下文/显存；带复现脚本提 issue |

完整契约见[执行模型——错误契约](execution.md#错误契约)。

## 为什么 CPU 和 GPU 偶尔抽的 token 不一样？

采样按种子确定，CPU / 暂存 / 零拷贝三条路径结果一致——唯一的
例外：抽签恰好落在 CDF 某个 exp 舍入边界上时（CPU 用精确 `exp`，
GPU 用约 2 ulp 的 `__expf`），两边各抽到相邻的一个元素，两个都是
该分布的合法样本。超大词表 + 接近均匀的 logits 时这个效应会成
规模出现（很小的排名窗口，n=152064 实测约 14 个排名）。另有一种
更少见的 GPU 侧边界：全局 softmax 总量靠逐 block 浮点原子加累加，
调度顺序在进程间可能不同，压在边界上的抽签在进程重启后可能落到
相邻 token。细节见
[采样——同 token 保证](sampling.md#同-token-保证)。

## 为什么批量采样在平坦分布上比 torch 慢？

`torch.multinomial` 从不排序——它对整个分布做一次布尔掩码就直接
抽签。fusedtok 的采样器必须给核**排序**（选择管线的职责），而接近
均匀的 logits 下核约占九成词表，所以这个场景下如实落后（批量
0.05-0.06x、单行 0.17-0.25x）。真实解码的 logits 是尖峰状的，那
才是采样器与原生 multinomial 同档甚至更快的场景。细节见
[采样——平坦分布](sampling.md#平坦分布如实的最坏情况)。

## 采样器是密码学安全的吗？

不是。随机源是对种子做 splitmix 风格哈希——可复现、分布均匀，
但不是 CSPRNG。应用需要密码学随机的话请自行引入。

## 能捕获 CUDA graph 吗？

能，全库都行，先热身。例外：融合采样器返回主机 int（每次调用
以一次回读收尾）；批量采样器（`_batched` 系列）的扩窗循环还要
根据回读结果重新发射 kernel，返回 int64 张量也一样进不了图；
`quantize_int8` / `qadd_int8` 中途同步一次以合成 scale。选择管线
自己维护内部 CUDA 图——不需要你管。

## Windows 计时注意事项

Windows 的 GeForce 驱动跑在 WDDM 模式：kernel 提交要花几十微秒，
主机端计时噪声大（Linux 下的驱动模型没有这层提交开销，数字通常
更稳）。基准测试用 CUDA event（见
[基准测试](benchmarks.md#测试协议)）；自己微基准时请优先用
event 而不是墙上时钟，并预期 argmax 这类微小算子的数字会摆。

## 词汇表

- **GQA**（分组查询注意力）——`Hq` 个 query 头共享 `Hkv < Hq`
  个 key/value 头，按连续分组对应（`q 头 h -> kv 头 h // (Hq/Hkv)`）。
- **kv-cache**——已生成 token 的 key/value 缓存；解码步每步把
  整个 cache 读一遍。
- **分页 cache / 块表（paged cache / block table）**——token
  分散住在定长块组成的池里、按序列用块表索引的 cache 排布
  （vLLM 的设计）。序列增长、收缩、驱逐都不会产生碎片。
- **flash-decoding**——长序列解码策略：把 cache 切成片、并行算
  出各片的部分 softmax、再归并。
- **nucleus（核）**——采样的截断集合：top-p 按"累计概率质量达到
  p"取前缀，min-p 按"概率不低于 min_p × 最大概率"取前缀。
- **批量采样（batched sampling）**——一次调用采样整个
  `[行数, 词表]` 批（`_batched` 系列采样器）：每行用自己的种子
  跑单行管线原样；对比逐行循环的提速来自省掉每行的启动/提交
  开销，数学本身不变。
- **multinomial**——torch 的按概率加权抽样算子
  （`torch.multinomial`），本库采样器的基准参考实现（按行标注
  与 softmax、top-k 或布尔掩码组合使用）。
- **radix（基数）**——选择管线按 key 的高位字节逐轮分桶的排序
  手法（radix sort 的思想）。
- **SDPA**——PyTorch 的 `scaled_dot_product_attention`，attention
  算子的官方参考实现（本库基准表的对比对象）。
- **cooperative launch（协作组全网格启动）**——CUDA 的一次性
  启动全部 block 并提供全网格同步的原语；本库的选择管线用普通
  kernel 启动 + 到达票据替代了它。
- **到达票据（arrival ticket）**——选择管线的跨 block 定序技巧：
  最后到达计数器的 block 拍板本轮结果，用普通 kernel 启动替代
  全网格栅栏。
- **物化（materialize）**——把中间张量写进显存。fusedtok 的卖点
  之一就是分数、中间结果不物化。
- **W8A8**——INT8 推理布局：权重 8 位带逐输出通道 scale，激活
  8 位带逐张量 scale。
- **WDDM**——Windows 显示驱动模型；相比 Linux 下的驱动模式多了
  kernel 提交延迟。
- **零拷贝（zero-copy）**——kernel 经 `data_ptr()` 直接操作
  torch 的设备缓冲；无中转拷贝、无主机同步。
