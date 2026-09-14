# 采样与选择

选择类算子（top-k、top-p、argmax）和融合采样器（`sample_topp`、
`sample_topk`、`sample_minp`、`sample_topa`、`sample_nsigma`、
`sample_eta`、`sample_typical`、
`decode_step`，以及各自的 `_batched` 批量变体）共用一条管线和一份确定性契约，`logit_penalties` 则把 HF
的三件惩罚合并成一次调用。这一页把两者讲清，包括
CPU 与 GPU 抽签可能不一致的精确边界。

**其他语言：** [English: sampling and selection](../en/sampling.md)

- [选择算子](#选择算子)
- [融合采样器](#融合采样器)
- [sample_minp——按最大值阈值截断的采样（v1.3）](#sample_minp按最大值阈值截断的采样v13)
- [sample_topa——平方峰值截断采样（v1.8）](#sample_topa平方峰值截断采样v18)
- [sample_nsigma——按离散度截断的采样（v1.8）](#sample_nsigma按离散度截断的采样v18)
- [sample_tfs——尾部自由采样（v2.1）](#sample_tfs尾部自由采样v21)
- [sample_xtc——排除顶部选择采样（v2.2）](#sample_xtc排除顶部选择采样v22)
- [sample_dry——序列级重复惩罚采样（v2.3）](#sample_dry序列级重复惩罚采样v23)
- [sample_eta——熵自适应截断采样（v1.6）](#sample_eta熵自适应截断采样v16)
- [sample_typical——局部典型采样（v1.6）](#sample_typical局部典型采样v16)
- [logit_penalties——一次调用套齐 HF 三件惩罚（v1.6.1）](#logit_penalties一次调用完成-hf-三种惩罚v161)
- [logit_penalties_batched——整批一行搞定（v1.7）](#logit_penalties_batched整批一行搞定v17)
- [批量采样——每个解码步一次调用（v1.4）](#批量采样每个解码步一次调用v14)
- [批量解码步——含重复惩罚（v1.5）](#批量解码步含重复惩罚v15)
- [同 token 保证](#同-token-保证)
- [平坦分布——不回避的最坏情况](#平坦分布不回避的最坏情况)
- [管线是怎么工作的](#管线是怎么工作的)

## 选择算子

所有选择类算子在值相同时都取**最靠前的下标**（并列时取下标更小者）。

```python
i = fusedtok.argmax(logits)                # 并列时取最靠前的下标
vals, idxs = fusedtok.topk(logits, 50)     # 降序，返回 (值, 下标)
vals, idxs = fusedtok.topp(probs, 0.9)     # 输入是概率
idxs = fusedtok.argmax_batched(batch_logits)   # 每行一个下标（v1.8）
```

- `topk` 接原始分数，`k` 取值 `[0, n]`。
- `topp` 接**概率**（已经 softmax 过），`p` 取值 `(0, 1]`；返回恰好
  凑满 p 的最小 top-p 集合（跨过 p 阈值的那一个元素也算在内）。

`argmax` 返回主机侧的 `int`，因此需要一次设备到主机的回读。
零拷贝路径上 argmax 用两个会自动清零复用的 workspace 槽位，让热路径
只剩一次 kernel 启动、零额外分配（v1.2 优化）。

`argmax_batched`（v1.8）把同样的规则按行跑在二维 `[行数, 词表]`
的整批上——一次 launch 完成整批贪心解码，不需要种子也不需要
温度。与单行版不同，它返回 int64 下标且零拷贝路径**不做主机
回读**：CUDA 输入返回 CUDA 张量（与调用方的其他操作按流排序），
CPU 输入返回 CPU 张量或 numpy 数组；workspace 经一次调用预留后，
launcher 即可做 CUDA graph 捕获——首次调用可能触发分配，无法在
外层捕获内执行，请先预热。每行的到达槽位每次调用清一次（整批一次小
memset，不是每行一次），因为这片槽位与选择类算子的暂存区共用
区域；每行的最后到达块发布该行的下标——就是单行的到达票据
模式按行拆开。

## 融合采样器

采样器从 logits 到 token 一次 GPU 往返：

```python
tok = fusedtok.sample_topp(logits, p=0.9, temperature=0.8, seed=step)
tok = fusedtok.sample_topk(logits, k=50, temperature=0.8, seed=step)
tok = fusedtok.decode_step(logits, history, penalty=1.1,
                           p=0.9, temperature=0.8, seed=step)
```

- `sample_topp`：softmax(logits / T) -> 用**全局质量**阈值切出核
  （nucleus）-> 逆 CDF（累计分布函数的反函数）抽签。如果第一个
  候选窗口没盖住整个核，窗口自动扩窗——v1.2 起是自适应扩窗跳变
  （见下文）。
- `sample_topk`：softmax(logits / T) -> 保留 k 个 -> **在 k 个
  幸存者之内**重新归一化 -> 抽签。这个窗口天生就盖得住整个分布
  （不需要猜），所以没有质量阈值、也没有扩窗循环。`k = 1` 恰好
  等于贪心；`k >= 词表大小` 就是对全分布采样。
- `decode_step`：对 `history` 施加 CTRL 式重复惩罚，再施加温度、
  再做核采样——一次调用一次回读，结果与按同样顺序组合三个算子
  （同种子）完全一致。
- `repetition_penalty(logits, token_ids, penalty)` 也单独暴露：
  正 logit 除以 `penalty`、负 logit 乘以 `penalty`
  （`penalty=1.0` 即关闭）。
- `logit_penalties(logits, token_ids, *, repetition, presence,
  frequency)` 把上面的 CTRL 规则与出现惩罚、频率惩罚合并成一次
  调用——详见[下面的小节](#logit_penalties一次调用完成-hf-三种惩罚v161)。

采样**按种子确定**：抽签用的均匀数来自 splitmix 风格的哈希
（可复现，但不是密码学安全的随机源）。主机侧来源的 token id
在上传前按词表校验；CUDA 上的 id 张量直接信任（不对流做同步）。

## sample_minp——按最大值阈值截断的采样（v1.3）

```python
tok = fusedtok.sample_minp(logits, min_p=0.1, temperature=0.8, seed=step)
```

min-p（出自 2024 年的 Min-P Sampling 论文，llama.cpp、vLLM 等推理栈
均已支持）用**相对峰值的值阈值**代替累计质量来截断：保留所有概率
不低于 `min_p × 最大概率` 的 token，在核内重新归一化，再用同一个
种子哈希抽签。

- 取值范围：`min_p` 必须在 `(0, 1]` 内（越界抛 `ValueError`），
  `temperature` 必须大于 0。
- 天然自适应：尖峰的解码 logits 得到很小的核，接近均匀的 logits
  得到很宽的核——调用方不需要猜窗口。
- `min_p = 1.0` 只保留恰好处于最大值的 token（唯一最大值时恰好是
  贪心；并列最大值时并列参与抽签）。
- 按种子确定，与其余采样器共用 RNG 与同 token 保证（含 CPU 精确
  `exp` vs GPU `__expf` 的边界注意事项）。
- 实现说明：exp 列本就按行最大归一（`exps[0] == 1.0`），核就是
  截断到首个低于 `min_p` 的元素处的前缀——不需要全局质量归约，
  串行遍历（serial walk）直接继承 v1.3 的检查点二分。v1.4 起扩窗循环也像 top-p
  一样自适应跳变：失败的窗口留下累计质量，配上一个只算一次的
  全局总量即可推出必定盖住核的下界（`w >= W + (T - C) / min_p`，
  其中 W 是失败窗口的宽度、C 是它的累计质量、T 是惰性计算的全局
  总量），宽核从此跳过 x8 阶梯的中间档位（宽核行约快 30%，token
  逐位不变）。

## sample_topa——平方峰值截断采样（v1.8）

```python
tok = fusedtok.sample_topa(logits, top_a=0.2, temperature=0.8, seed=step)
```

top-a 采样（开源采样栈的 top-a 规则，HF transformers 以
`TopALogitsWarper`、vLLM 以 `top_a` 之名支持）用**峰值平方**驱动的
值阈值截断：保留所有概率不低于 `top_a × 最大概率²` 的 token，在核内
重新归一化，再用同一个种子哈希抽签。平方是关键：阈值的下降比峰值
本身更快——分布越尖，保留下来的头部越紧凑；分布接近平坦，则几乎
整个词表都保留——同等取值下，其窗口自适应幅度比 `min_p` 更大。

- 取值范围：`top_a` 必须在 `(0, 1]` 内（越界抛 `ValueError`），
  `temperature` 必须大于 0。
- `top_a = 1.0` 的阈值是 `p_max²`：只有一个明显占优的 token 时
  退化为贪心；但只要第二名的概率仍够得到平方峰值，两个最大值
  就都留在抽签池里（`min_p = 1.0` 在同样的分布下只保留最大概率
  token）。
- 核永远至少保留一个 token：只要 `top_a <= 1`，阈值就不会超过最大
  概率（因为 `p_max <= 1`），实现里另加 min-token 守卫，即使有浮点
  误差这条下限也不会破。
- 按种子确定，与其余采样器共用 RNG 与同 token 保证。截断阈值来自
  全局 softmax 总量（一次浮点原子归约），因此与 top-p、eta 属于同一类
  文档已注明的舍入边界行为：CPU 与 GPU 之间、以及跨进程运行时，落在同一边界上的抽签可能相差
  相邻排名；同进程、同一份输入缓冲内的结果逐位稳定。
- 实现说明：在按行最大归一的 exp 列（`exps[0] == 1.0`）里，
  最大概率是 `p_max = 1 / total`，截断值换算成 exp 阈值就是
  `e_i >= top_a / total`——用 exptotal pass 已产出的总量做一次除法
  即可，不需要额外归约。核和 min-p 一样是值阈值前缀，扩窗下界复用
  min-p 的充分质量公式（除数换成推导出的截断值）；总量每次尝试都
  重算（每次尝试的开头 memset 会清掉它的槽位），首次扩窗后由主机
  缓存——与 eta 的做法完全一致。

## sample_nsigma——按离散度截断的采样（v1.8）

```python
tok = fusedtok.sample_nsigma(logits, nsigma=1.5, temperature=0.8,
                             seed=step)
```

top-nσ 采样（Shi 等 2024 年论文 *Top-nσ: Not All Logits Are You
Need*，Qwen 解码栈背后的极值过滤器）按分布**自身离散度**截断：先把
logits 除以温度，保留所有缩放后 logit 不低于 `均值 − nsigma × 标准差`
的 token（均值与标准差都对整行取），在核内重新归一化，再用同一个
种子哈希抽签。截断和 min-p 一样是值阈值，只是写在 logit 空间里，
连 softmax 归约都不需要——只要整行的前两阶矩。

- 取值范围：`nsigma` 必须大于 0（越界抛 `ValueError`），
  `temperature` 必须大于 0。论文的常用区间是 `1.0`–`3.0`：
  取 `1.0` 时，普通解码行大约只留下最靠近均值的前三分之二；
  取 `3.0` 几乎全保留；再大的 `nsigma` 就逼近普通采样。
- 接近平坦的行全保留：`sigma -> 0` 时，阈值降到不高于任何一个 logit（全相等的行
  正是这种情形的极限）。
- 核永远至少保留一个 token：阈值不会超过行最大值（均值在最大值
  下方，`nsigma × sigma >= 0`），实现里另加 min-token 守卫，即使有
  浮点误差这条下限也不会破。
- 按种子确定，与其余采样器共用 RNG 与同 token 保证。截断阈值来自
  整行的矩累加器（逐线程舍入模式固定，块结果升为 double 后原子
  加——方差的相减运算若用 float 会放大到达序漂移），因此属于
  文档记录的边界情形：CPU 与 GPU 之间，落在同一边界上的抽签可能相差相邻
  排名；同进程、同一份输入缓冲内的结果逐位稳定。
- 实现说明：记 `d = l - max`，门槛换算成 exp 阈值是
  `exp(mean(d) - nsigma × sigma_d)`——与最大值无关且永不超过 1，
  所以核是值阈值前缀，扩窗下界复用 min-p 的充分质量公式（除数换成
  这个阈值）。每次尝试跑一遍 expmax 加一遍新的矩 pass（外加扩窗界
  要用的全局总量），结构与 eta 一致。

## sample_tfs——尾部自由采样（v2.1）

```python
tok = fusedtok.sample_tfs(logits, z=0.95, temperature=0.8, seed=step)
```

尾部自由采样（Filazzola & Trottet 2023，llama.cpp/KoboldAI 等广泛支持）
按排序概率曲线 CDF 的**二阶导数**截断：保留排序后概率曲线仍在"弯曲"
（归一化的二阶导数绝对值 ≥ `1-z`）的前缀。与 min-p 或 top-a 使用固定值
阈值不同，TFS 的截断点是数据依赖的——它标记了排序概率分布从弯曲变为
平坦的位置。

- 取值范围：`z` 必须在 `(0, 1]` 内（越界抛 `ValueError`），
  `temperature` 必须大于 0。`z = 1.0` 保留全部（阈值 = 0，全通过）；
  越低剪得越激进。论文推荐值 `0.95`。
- 核永远至少保留一个 token（排序最高的 token 总被保留），实现里另加
  min-token 守卫。
- 按种子确定，与其余采样器共用 RNG。截断索引是排序概率的纯函数，
  属于文档记录的边界情形。
- 实现说明：d2 的计算天然是串行的（每个元素依赖三个连续概率），
  需要两遍（第一遍找最大值，第二遍找截断），但排序窗口很小。扩窗
  策略与 sample_typical 相同——诚实的 x8 阶梯。批量版与其他批量采样器共用同一条分块管线（sequencer 模式 6）：每轮尝试一次流同步、一次整体回读。

## sample_xtc——排除顶部选择采样（v2.2）

```python
tok = fusedtok.sample_xtc(logits, top_n=3, probability=0.8, seed=step)
tok = fusedtok.sample_xtc_batched(batch_logits, 3, 0.8, seeds=seeds)
```

XTC（Exclude Top Choices，排除顶部选择）专治大模型千篇一律的"模板"输出：以
`probability` 的概率，在抽签之前把概率最高的前 `top_n` 个 token 整体移出采样池，
把模型从最熟练的续写上推开。

- 是否触发压制由每个种子确定（一个加盐 splitmix 哈希），同一个
  `(seed, 输入)` 组合永远做出同一个决定。
- `top_n` >= 0、`probability` 属于 [0, 1]（越界抛 `ValueError`）；
  `top_n = 0` 或 `probability = 0` 退化为普通 softmax 抽签。
- 至少保留一个 token：当 `top_n` >= 窗口宽度时，钳制会恰好留下最小
  候选。
- 抽签分布横跨整个词表——压制只移除一个前缀，所以管线直接以全词表
  窗口运行（2.2.1 修复；此前词表 > 1024 时 1024 名之后的尾部会被静默
  截掉）。全词表排序是这一语义的诚实代价；批量版通过共享分块管线摊薄
  它。

## sample_dry——序列级重复惩罚采样（v2.3）

```python
tok = fusedtok.sample_dry(logits, history, allowed_length=2,
                          multiplier=1.75, temperature=0.8, seed=step)
tok = fusedtok.sample_dry_batched(batch_logits, histories,
                                  allowed_length=2, multiplier=1.75,
                                  seeds=seeds)
```

DRY（Don't Repeat Yourself）是 token 级 `repetition_penalty` 的序列级
版本：它不再惩罚"出现过的 token"，而是惩罚"会延续一段重复序列的
token"。扫描窗口 = 最近 64 个历史 token。对每个后缀长度
`L ∈ [allowed_length, 窗口)`，当前 L 后缀在窗口内每一次更早的出现都会
贡献它的后继 token；候选 token 保留其中的最大指数
（`L - allowed_length + 1`）。

- 应用约定与 `repetition_penalty` 一致：正 logit 除以
  `multiplier ** 指数`，负 logit 相乘——幅度只会缩小。
- `multiplier` >= 1.0（1.0 等于关闭惩罚）；`allowed_length` 属于
  [1, 64]（越界抛 `ValueError`）；`temperature` > 0。默认值
  (2, 1.75) 与 DRY 社区惯用量级一致。
- 抽签是惩罚行上的普通全词表 softmax 逆变换（与
  `sample_xtc(top_n=0)` 同一条路径），逐种子确定，与其他采样器共用
  同一个 splitmix 契约。
- `token_ids` 携带历史：单行传一维整数数组；批量传逐行序列 /
  二维数组 / 扁平数组加 offsets（即 `decode_step_batched` 的布局）。
  主机侧取值会做范围校验；设备端 torch 张量按文档记载的信任边界
  处理。
- 实现注：GPU 路径先用一遍扫描生成每行至多 64 项的触发表，再用一遍
  grid-stride 把惩罚写进行数据的 scratch 副本；批量版对不等长历史
  每行一个扫描 block、对所有行做一次融合 apply，然后走普通全词表
  批量抽签。每行与单行算子在该行上按文档记载的 ulp 边界一致。

## sample_eta——熵自适应截断采样（v1.6）

```python
tok = fusedtok.sample_eta(logits, eta=0.3, temperature=0.8, seed=step)
```

（示例取 `0.3` 是为了演示 API；实际服务端常用 `1e-3` 量级。）

eta 截断（Hewitt 等 2022 年的论文 *Truncation Sampling as Language
Model Desmoothing*，llama.cpp、vLLM 以 `eta_cutoff` 之名支持）的截断
阈值由分布**自身形状**决定：先算分布的熵 H（单位 nat），保留所有
概率不低于 `eta × min(1, exp(-H))` 的 token，在核内重新归一化，再用
同一个种子哈希抽签。门槛与熵反向移动：分布越尖，熵越接近零，门槛被抬到 `eta`
附近，低概率尾部被剪掉；分布越平，`exp(-H)` 越接近零，门槛跟着
下沉（几乎全保留）。对比一下：`min_p` 的门槛相对峰值，`top_p`
的门槛相对累计质量，eta 的门槛则跟着熵走。（本库实现的是 llama.cpp 与 HF 常用的
简化阈值 `eta × min(1, exp(-H))`；论文原式为
`min(ε, √ε·exp(-H))`，方向一致、数值不同。）

- 取值范围：`eta` 必须在 `(0, 1]` 内（越界抛 `ValueError`），
  `temperature` 必须大于 0。常用 `eta` 值非常小（`1e-3` 量级）——
  真正的塑形工作由熵因子完成。
- 核永远至少保留一个 token：截断阈值以最大概率为上界（概率的
  加权几何平均不会超过最大值），实现里另加 min-token 守卫，即使有
  浮点误差这条下限也不会破。
- 按种子确定，与其余采样器共用 RNG 与同 token 保证。截断阈值
  来自熵累加器（浮点原子加）与全局总量，因此属于文档记录的边界
  情形：CPU 与 GPU 之间、以及跨进程运行时，落在同一边界上的抽签可能相差一个相邻排名；
  同进程、同一份输入缓冲内的结果逐位稳定。
- 实现说明：截断和 min-p 一样是值阈值前缀，但阈值需要熵 H——
  每次尝试多跑一遍全词表 pass（熵累加器 `s = Σ e_i × (l_i - max)`，
  `H = log(total) - s / total`），扩窗下界复用 min-p 的充分质量
  公式（把 `min_p` 换成推导出的截断值）。

## sample_typical——局部典型采样（v1.6）

```python
tok = fusedtok.sample_typical(logits, typical=0.9, temperature=0.8,
                              seed=step)
```

局部典型采样（Meister 等 2022）按"每个 token 的意外度
（`-log p_i`）与分布熵 `H` 有多接近"升序，保留质量达到 `typical`
的最小集合，在集合内重新归一化，再用同一个种子哈希抽签。与
top-p/min-p 不同，保留集**不是**值排序的前缀：过自信与过意外的
token 被对称地剪掉——这正是该设计的本意。

- 取值范围：`typical` 必须在 `(0, 1]` 内（越界抛 `ValueError`），
  `temperature` 必须大于 0。取值接近 1 时趋近于对全词表的普通
  采样。
- 保留带至少包含一个 token——以意外度最接近熵的 token 为起点，
  实现里另加 min-token 守卫，浮点误差也动不了这条下限。
- 按种子确定，与其余采样器共用 RNG 与同 token 保证。带成员关系
  由熵累加器推导，因此同样属于文档记录的边界情形（CPU 与 GPU 在
  舍入边界上取相邻排名；同进程、同一份输入缓冲内逐位稳定）。
- 实现说明：在值排序的窗口上，保留集是一个**连续带**（意外度
  `|log p_i + H|` 沿值序呈 U 形——先降到谷底 `log p_i = -H` 再
  回升）。串行遍历器从谷底出发按意外度升序扩带（合并两条单调
  臂），只要窗口还没到全词表，带一旦触及窗口尾部就拒绝并扩窗
  （质量达标与否都一样）——典型集的成员可能落在窗口之外，宁可
  扩窗也不能截错；到了全词表窗口，遍历得到的带质量就是整行总量
  （求和顺序差在 ulp 以内），照常抽签，和其他采样器的全窗兜底
  一致。没有解析的扩窗下界可用，只能老老实实按 x8 阶梯逐级加窗（全词表窗口必然覆盖，此时带质量即
  全量）。

## logit_penalties——一次调用完成 HF 三种惩罚（v1.6.1）

```python
penalized = fusedtok.logit_penalties(logits, history,
                                     repetition=1.2, presence=0.1,
                                     frequency=0.05)
```

`logit_penalties` 把 HF 风格的三种采样惩罚一次施加到一行 logits
上。对 `token_ids` 里每个**去重后**的 id（设它在 ids 里出现了
`c` 次）：

```
v = logit[id]
v = v / repetition if v > 0 else v * repetition  # 正值除、负值乘
if presence    != 0.0: v -= presence
if frequency   != 0.0: v -= c * frequency
penalized[id] = v
```

- 组合顺序与 HF 处理器一致：先按 CTRL 规则缩放，再做出现惩罚的减法偏移，再按
  计数加权做频率偏移。`repetition` 必须大于 0（越界抛
  `ValueError`）；`presence` 和 `frequency` 是直接从 logit 里减去的偏移量，不限范围。
- 重复的 id 不会叠加惩罚：同一个 id 写三遍也只罚一次，`c = 3`
  只进频率项。这与 GPU 直方图（每个 id 计一个数）和"每个去重 id
  恰好罚一次"的 CPU 参考一致——也就是 v1.5.2 落地的那个修复。
- 没被点名的 logit 原样通过；空的 `token_ids` 是精确的空操作。
  其余参数取默认值时，`logit_penalties(..., repetition=1.2)` 与单独
  的 `repetition_penalty` 逐位一致。
- **精确契约，不属于采样器那类舍入边界行为**：计数是整数，输出没有任何
  一个值被多个线程碰过——没有原子到达序的舍入、没有熵推导的
  阈值，所以 CPU 参考与所有 GPU 路径（staged、零拷贝、原地）逐位
  一致，跨进程也一样。
- id 直方图走按词表大小缓存的 workspace，分配发生在流捕获之外
  （attention workspace 的老套路），所以本算子可以进 CUDA graph；
  唯一的例外是首次调用就撞上捕获的情形——这时直方图会借用输出
  缓冲，也是唯一 `out` 不能与 `logits` 同址的场景（那种罕见路径上
  launcher 会抛出明确的错误）。捕获前先热身一次即可继续用原地调用。
- 批量版见[下一节](#logit_penalties_batched整批一行搞定v17)。

## logit_penalties_batched——整批一行搞定（v1.7）

```python
penalized = fusedtok.logit_penalties_batched(batch_logits, histories,
                                             repetition=1.2, presence=0.1,
                                             frequency=0.05)
```

`logit_penalties_batched` 把三件惩罚一次施加到整个 `[行数, 词表]`
批上。每行的语义与单行 `logit_penalties` **完全相同**——组合顺序、
"每个去重 id 只罚一次、按行内计数 c 进频率项"的规则、取值守卫——
所以每行的输出与单行算子跑该行**逐位一致**，所有路径、跨进程也是。
与批量采样器不同，这里没有种子（本算子不是随机过程），零拷贝路径
的输出也留在设备上。

- `logits` 为 2-D 连续 float32；返回值与输入同一类型族（CUDA 张量进
  →CUDA 张量出，否则是 float32 的 CPU 数组/张量）。
- `token_ids` 携带逐行不等长历史，形式与 `decode_step_batched`
  完全一致：每行一个 id 列表的列表、一个 2-D 整数数组（每行的所有
  列都算数——补位请用合法 id，并记得补位 id 自己也计一次数），或
  一维扁平 id 数组加 `ids_offsets`（行数 + 1 个单调不减、从 0 开始
  的偏移）。取值必须在 `[0, 词表)` 内；主机侧数值上传前校验，设备
  侧 id 张量直接信任。
- 逐行直方图走按（行数, 词表）缓存的 workspace，分配发生在流捕获
  之外，所以本算子热身后可以进 CUDA graph；冷捕获会退到一条逐行
  路径（借输出行当直方图 scratch，更慢但结果相同——Python 包装层
  总是返回新张量，包装层用户不受影响）。裸 launcher 的原地
  `out == logits` 在热路径上可用。
- 空行是精确的空操作；空批返回 `[0, 词表]` 的空结果。

## 批量采样——每个解码步一次调用（v1.4）

```python
tokens = fusedtok.sample_topp_batched(batch_logits, p=0.9, seeds=seeds)
```

`sample_topp_batched` / `sample_minp_batched` / `sample_topa_batched` /
`sample_nsigma_batched` / `sample_tfs_batched` /
`sample_topk_batched` /
`sample_eta_batched` / `sample_typical_batched`
一次调用采样整个 `[行数, 词表]` 批，每行返回一个 token。返回值是
**主机侧**的 int64：torch 输入回 CPU torch 张量，numpy 输入回 numpy
数组——扩窗循环不可避免地要回读主机，因此与单行版一样不可做
CUDA graph 捕获。

- `logits` 为二维、连续、float32。
- `seeds` 每行一个整数，接受列表、numpy 数组或 torch 张量（CUDA
  张量会先搬回主机）；取值校验为非负且小于 2^63。默认（`None`）
  取 `0..行数-1`，相同内容的行也能各自独立抽签。服务循环里记得
  每步换种子——惯用写法是 `step * rows + arange(rows)`——否则每一步
  都在复用同一套逐行随机流。
- 每行原封不动地跑单行管线——同样的 kernel、同样的累加顺序、
  逐行一致（含扩窗循环：各行按自己的核宽度完成，先完成的行被
  跳过，宽核的行继续扩窗重试）。1.6 的两个批量成员自 1.6.1 起
  也保持这一性质：批量 typical 与单行 kernel 一样，带触及窗口
  尾部就回去扩窗（1.6.1 之前会直接从截断带抽签——本版已修复）。
- 行按固定 32 行一组分块处理，超大批次也只是分块流过一个容量有上限的 workspace。
- 批处理换来的收益：把逐行的 Python/启动开销合并成一次。在 B=8、
  约 64 token 历史的墙上时钟探针下，受提交开销限制的主机（如
  Windows/WDDM）看到的时间只有
  逐行循环的 1/4 到 1/6（[8, 131072] 在 3060 上：topp 1340 ->
  274 µs、minp 1399 -> 237 µs；README 中的事件计时基准表量的是
  GPU 时间，协议不同）。尖峰 logits 下与 torch 原生批量
  multinomial 处于同一档位，`sample_topk_batched` 明确胜出
  （1.29x / 1.17x）；平坦最坏情况则比单行版再低一档
  （0.05-0.06x），差距同样如实给出。
- `decode_step` 的批量版见下一节（v1.5）。

## 批量解码步——含重复惩罚（v1.5）

```python
tokens = fusedtok.decode_step_batched(
    batch_logits, sampled_ids, penalty=1.3, seeds=seeds)
```

`decode_step_batched` 为整个 `[行数, 词表]` 批一次跑完融合解码
链——对每行自己的历史做重复惩罚、温度缩放、核采样——每行返回
一个 token（主机侧 int64，契约与批量采样器相同）。

- `sampled_ids` 承载逐行历史：逐行序列的嵌套列表（天然不等长）、
  二维整数数组（每行取其**全部**列——需要补齐的场合请自行填充（padding）
  或改用不等长形式），或扁平一维整数数组加 `ids_offsets`
  （`行数 + 1` 个单调不减的条目、从 0 到扁平长度；跳过逐行 Python 的
  服务端快路径）。取值必须在 `[0, 词表)` 内。
- 每行把自己的历史记录进一个词表长度的位图，每次读 logit 都先对**原始
  值**施加惩罚再做温度缩放——与 `decode_step` 的组合顺序一致，
  因此除文档记录的 ulp 边界外逐行一致。
- `penalty=1.0` 或历史全空时不会产生任何位图读写（此时调用与
  `sample_topp_batched` 完全等价）。
- 逐（行，种子）确定；不可 CUDA graph 捕获；行按 32 行一组分块。
- 批处理收益：B=8 墙上时钟探针、约 64 token 历史下，尖峰
  logits 比逐行循环 `decode_step` 快 5.2 倍（3060：1676 ->
  321 µs；5060 Ti：646 -> 145 µs），与 torch 原生"惩罚 +
  softmax + 批量 multinomial"组合慢约两成（3060 墙钟探针；5060 Ti
  表中为 147 vs 100 µs）；中尾 logits 快
  3.1 倍（3060：17.3 -> 5.5 ms）。

## 同 token 保证

固定种子下，CPU / 暂存 / 零拷贝三条路径抽出**同一个 token**——
这不是口号，而是测试一条条钉死的行为。唯一有文档记录的边界：CPU
参考实现用精确 `exp`，GPU kernel 用 `__expf`（约 2 ulp 误差）。
当一次抽签恰好落在 CDF 的某个 exp 舍入边界上时，CPU 与 GPU 可能
各抽到相邻的一个元素——两个都是该分布的合法样本。

在超大词表 + 接近均匀的 logits 下，这个边界效应会成规模地出现：
微小的逐元素舍入差沿着严格顺序的 CDF 遍历累加，CPU 与 GPU 抽出
的 token 会隔一个很小的**排名窗口**（n=152064 实测约差 14 个
排名；32k 词表约差 1 个）。GPU 自身对同一种子恒定位一致，扩窗
调度也从不改变抽出的 token。另有一个更细的同类边界（v1.4 批量
工作时暴露；这一性质其实自 1.2 的单行 API 起就存在）：全局
softmax 总量靠逐 block 的浮点原子加累加，而 GPU 调度这些 block
的到达顺序在进程之间可能不同——恰好落在 CDF 边界上的抽签因此
可能在进程重启后抽到相邻 token。实测在 131k 词表下抽 8 行，其中
1 行恰好压在边界上、多次运行间会翻转——对真实负载来说，撞上
这种边界的概率可以忽略不计。批量采样器每行用的是同一套
累加模式，因此某行与它的单独调用也可能差这一个边界 token；
对照测试（parity）按"精确命中、或最多只差一个相邻排名"的标准加以验证。

为什么不修？严格顺序的浮点加法**就是**确定性契约本身——把求和
并行化会改变历史上每个种子抽出的每一个 token。v1.2 的优化批次
原封不动保留了加法顺序，只把载入做了流水化（平坦最坏情况提速
8.5 倍，token 逐位不变）。

## 平坦分布——不回避的最坏情况

当核（nucleus）盖住几乎整个词表（接近均匀的 logits）时，
`sample_topp` 实际上要给全词表排序，torch 的全并行排序仍然更快
——基准表里明确标着 0.16-0.26x 的差距。v1.2 用三个不破坏契约的改动把
该最坏情况的耗时压到约 1/8.5（快约 8.5 倍；3060 上 n=131072
实测 18.2ms -> 2.2ms）：

1. **自适应扩窗跳变**——失败的窗口尝试会把它累计的质量留在
   workspace 槽里；结合全局 softmax 总量可以推出必要下界
   `w >= W * p * T / C`（排名第 W 之后的元素都不超过 C/W），
   平坦分布由此一步跳到（几乎）全词表，不必沿阶梯逐级扩窗。
2. **全词表快路径**——窗口等于词表时，radix 选择是纯浪费（所有
   key 都存活），换成一次朴素的并行打包。
3. **串行遍历批量载入**——承担契约的顺序加法仍然顺序执行，但
   载入按无分支批次流水化（朴素的一读一加遍历纯吃 L2 延迟——
   占平坦情况耗时的 97%）。

真实解码的 logits 是尖峰状的；平坦是最坏情况，不是常态。v1.3 又加
了一项同样保持 token 逐位不变的优化：第一趟遍历（walk 1）按批边界
把前缀和记进共享内存，第二趟（walk 2）用二分查找定位目标所在的
批、从那里续走（续走的前缀与从头走逐位相同——同样的加数按同样
的顺序），平坦最坏情况再快约 1.6 倍。

## 管线是怎么工作的

背景知识（选读；使用这些算子不需要以下内容）：

- **到达票据（arrival ticket）radix 轮**：候选 key 按基数（radix）
  分轮做直方图细化；每轮最后一个到达的 block 拍板边界——全是普通
  kernel 启动，没有全网格栅栏，也不需要 cooperative launch
  （协作组全网格启动）。
- **早退压缩**：当某个基数边界桶里的候选数不超过 1024 时，单个
  block 在共享内存里把幸存者排序，不再继续细化。
- **合并阶梯排序（merge-ladder sort）**：k 更大时，各 block 先分块
  排序，再按层级逐层合并，每层一次启动。
- **缓存 CUDA graph**：整条序列按 (n, k, mode) 捕获一次，之后每次
  调用就是一次图启动；每次调用的指针经由设备侧参数块传递，
  replay 能看到新张量。
