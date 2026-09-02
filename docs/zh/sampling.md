# 采样与选择

选择类算子（top-k、top-p、argmax）和融合采样器（`sample_topp`、
`sample_topk`、`decode_step`）共用一条管线和一份确定性契约。
这一页把两者讲清，包括 CPU 与 GPU 抽签可能不一致的精确边界。

**其他语言：** [English: sampling and selection](../en/sampling.md)

- [选择算子](#选择算子)
- [融合采样器](#融合采样器)
- [sample_minp——按最大值阈值截断的采样（v1.3）](#sample_minp按最大值阈值截断的采样v13)
- [同 token 保证](#同-token-保证)
- [平坦分布——诚实的最坏情况](#平坦分布诚实的最坏情况)
- [管线是怎么工作的](#管线是怎么工作的)

## 选择算子

所有选择类算子值相同时都取**最靠前的下标**（tie 取最早）。

```python
i = fusedtok.argmax(logits)                # 并列时取最靠前的下标
vals, idxs = fusedtok.topk(logits, 50)     # 降序，返回 (值, 下标)
vals, idxs = fusedtok.topp(probs, 0.9)     # 输入是概率
```

- `topk` 接原始分数，`k` 取值 `[0, n]`。
- `topp` 接**概率**（已经 softmax 过），`p` 取值 `(0, 1]`；返回
  最小的 top-p 集合，跨越边界的那一个元素也包含在内。

`argmax` 返回一个 `int`——里面包含返回主机值所必需的一次
设备到主机回读。零拷贝路径上 argmax 用两个自我复位的 workspace
槽位，让热路径只剩一次 kernel 启动、零额外分配（v1.2 优化）。

## 融合采样器

采样器从 logits 到 token 一次 GPU 往返：

```python
tok = fusedtok.sample_topp(logits, p=0.9, temperature=0.8, seed=step)
tok = fusedtok.sample_topk(logits, k=50, temperature=0.8, seed=step)
tok = fusedtok.decode_step(logits, history, penalty=1.1,
                           p=0.9, temperature=0.8, seed=step)
```

- `sample_topp`：softmax(logits / T) -> 用**全局质量**阈值切核
  （nucleus）-> 反 CDF 抽签。如果第一个候选窗口没盖住整个核，
  窗口自动加宽——v1.2 起是自适应跳宽（见下文）。
- `sample_topk`：softmax(logits / T) -> 保留 k 个 -> **在 k 个
  幸存者之内**重新归一 -> 抽签。窗口构造性地盖住分布，所以没有
  质量阈值、没有加宽循环。`k = 1` 恰好等于贪心；`k >= 词表大小`
  就是对全分布采样。
- `decode_step`：对 `history` 施加 CTRL 式重复惩罚、再温度、再
  核采样——一次调用一次回读，结果与按同样顺序组合三个算子
  （同种子）完全一致。
- `repetition_penalty(logits, token_ids, penalty)` 也单独暴露：
  正 logit 除以 `penalty`、负 logit 乘以 `penalty`
  （`penalty=1.0` 即关闭）。

采样**按种子确定**：抽签用的均匀数来自 splitmix 风格的哈希
（可复现，但不是密码学安全的随机源）。主机侧来源的 token id
在上传前按词表校验；CUDA 上的 id 张量直接信任（不同步流）。

## sample_minp——按最大值阈值截断的采样（v1.3）

```python
tok = fusedtok.sample_minp(logits, min_p=0.1, temperature=0.8, seed=step)
```

min-p（出自 2024 年的 Min-P Sampling 论文，llama.cpp、vLLM 等推理栈
均已支持）用**相对峰值的值阈值**代替累计质量来截断：保留所有概率
不低于 ``min_p × 最大概率`` 的 token，在核内重新归一化，再用同一个
种子哈希抽签。

- 天然自适应：尖峰的解码 logits 得到很小的核，接均匀的 logits 得到
  很宽的核——调用方不需要猜窗口。
- ``min_p = 1.0`` 只保留恰好处于最大值的 token（唯一最大值时恰好是
  贪心；并列最大值时并列参与抽签）。
- 按种子确定，与其余采样器共用 RNG 与同 token 保证（含 CPU 精确
  ``exp`` vs GPU ``__expf`` 的边界 caveat）。
- 实现说明：exp 列本就按行最大归一（``exps[0] == 1.0``），核就是第一
  个低于 ``min_p`` 的元素处截断的前缀——不需要全局质量归约，串行走查
  直接继承 v1.3 的检查点二分。

## 同 token 保证

固定种子下，CPU / 暂存 / 零拷贝三条路径抽出**同一个 token**——
这是被测试钉住的承诺，不是愿景。唯一有文档记录的边界：CPU
参考实现用精确 `exp`，GPU kernel 用 `__expf`（约 2 ulp 误差）。
当一次抽签恰好落在 CDF 的某个 exp 舍入边界上时，CPU 与 GPU 可能
各抽到相邻的一个元素——两个都是该分布的合法样本。

在超大词表 + 接近均匀的 logits 下，这个边界效应会成规模地出现：
微小的逐元素舍入差沿着严格顺序的 CDF 走查累加，CPU 与 GPU 抽出
的 token 会隔一个很小的**排名窗口**（n=152064 实测约差 14 个
排名；32k 词表约差 1 个）。GPU 自身对同一种子恒定位一致，扩窗
调度也从不改变抽出的 token。

为什么不修？严格顺序的浮点加法**就是**确定性契约本身——把求和
并行化会改变历史上每个种子抽出的每一个 token。v1.2 的优化批次
原封不动保留了加法顺序，只把载入做了流水化（平坦最坏情况提速
8.5 倍，token 逐位不变）。

## 平坦分布——诚实的最坏情况

当核（nucleus）盖住几乎整个词表（接均匀的 logits）时，
`sample_topp` 实际上要给全词表排序，torch 的全并行排序仍然更快
——基准表里如实标着 0.10–0.17x。v1.2 用三个不破坏契约的改动把
该最坏情况的耗时压到原来的约 1/8.5（3060 上 n=131072 实测
18.2ms -> 2.2ms）：

1. **自适应跳窗**——失败的窗口尝试会把它累计的质量留在
   workspace 槽里；结合全局 softmax 总量可以推出必要下界
   `w >= W * p * T / C`（排名第 W 之后的元素都不超过 C/W），
   平坦分布由此一步跳到（几乎）全词表，不必沿阶梯逐级加宽。
2. **全词表快路径**——窗口等于词表时，radix 选择是纯浪费（所有
   key 都存活），换成一次朴素的并行打包。
3. **串行走查批量载入**——承担契约的顺序加法仍然顺序执行，但
   载入按无分支批次流水化（朴素的一读一加走查纯吃 L2 延迟——
   占平坦情况耗时的 97%）。

真实解码的 logits 是尖峰状的；平坦是最坏情况，不是常态。v1.3 又加了一刀
同样保持 token 逐位不变的优化：walk 1 按批边界把前缀和记进共享内存，
walk 2 用二分查找定位目标所在的批、从那里续走（续走的前缀与从头走
逐位相同——同样的加数按同样的顺序），平坦最坏情况再降约 1.6 倍。

## 管线是怎么工作的

给好奇者的背景（用这些算子不需要以下知识）：

- **到达票据（arrival ticket）radix 轮**：候选 key 按基数分轮做
  直方图细化；每轮最后一个到达的 block 拍板边界——全是普通
  kernel 启动，没有全网格栅栏，不需要 cooperative launch。
- **早退压缩**：当某个基数边界桶里的候选数不超过 1024 时，单个
  block 在共享内存里把幸存者排序，不再继续细化。
- **合并梯 sorting**：k 更大时，各 block 先分块排序，再按层级
  逐层合并，每层一次启动。
- **缓存 CUDA 图**：整条序列按 (n, k, mode) 捕获一次，之后每次
  调用就是一次图启动；每次调用的指针经由设备侧参数块传递，
  replay 能看到新张量。
