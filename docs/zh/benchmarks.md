# 基准测试：协议与读表指南

fusedtok 发布的数字出自同一个脚本、同一套协议和一条诚实原则。
这一页把三者都写清楚，你可以复现表里的每一行，也能看懂为什么
有些行是输的。

**其他语言：** [English: benchmarks](../en/benchmarks.md)

## 测试协议

```bash
python benchmarks/bench.py            # 全套，几分钟
```

- **CUDA event 计时**，走零拷贝 torch 路径，绝不用墙上时钟
  （Windows 的 WDDM 模式下主机端给 GPU 计时毫无意义）。
- 每个配置跑 **3 轮独立计时**，各带自己的预热；表格报告**均值**，
  JSON 里保留每一轮的值，波动可审计。
- torch 参考实现是推理循环真正会手写的那种组合表达式（eager
  组合式）；attention 的参考使用**预先展开**的头——
  `repeat_interleave` 放在计时区之外——因为这才是与 fusedtok
  所算内容的公平对比。
- 采样行测量**固定的 logits**：bench 给 torch 的 RNG 播种，每次
  运行、每台机器抽到的分布完全一致。自 1.2 起，峰值行用 +20 的
  尖峰、平坦行用接均匀的 logits，两个 regime 都确定性符合各自
  标签。
- `--iters N` 调整每轮迭代次数（发布数据用 60，默认 100 便于
  快速检查）。

产物落在 `docs/benchmarks/`：每 GPU 一份 JSON + 一张单面板加速比
图（文件名含设备名）。README 的基准表就从这些 JSON 生成——同源
同舍入。

## 表格怎么读

- 头条表格展示**每算子最大形状**；更小的形状在 JSON 里
  （Blackwell 上小形状的优势反而更大——形状越大，启动开销占比
  越低）。
- **（诚实）** 标注的是 fusedtok 输掉的行：attention_prefill 对
  SDPA 的 flash 后端（约 0.45x）、INT8 GEMM 对 cuBLASLt
  （0.40-0.58x）、平坦分布的 sample_topp 对 torch 全并行排序
  （0.17-0.25x，单行与批量同理）与宽核的 sample_minp 行
  （0.28-0.39x——一次加宽重试加一次 32-64k 排序；torch 的布尔掩码
  组合式从不排序）。这些是设计范围的声明，不是测量噪声——各
  [主题页](usage.md)有解释。
- **带宽列**（GB/s）只统计该算子必须搬运的字节（如 softmax 2
  张量、rmsnorm+res 3 张量）；INT8 行的 **TOPS** 按每秒稠密 MAC
  数 ×2 计算。
- **argmax** 行是对含主机同步调用的事件计时，在 WDDM 上摆动大
  （跨轮 0.73–1.24x）；把同步排除在计时环外的墙上时钟探针测得
  1.12x（3060）/ 0.96x（5060 Ti）。两种数字都在行内注明。
- 采样行的**组合参考**（排序+掩码+multinomial 等）自身在 WDDM
  上逐轮波动约 15%；JSON 里的逐轮值能看出是哪一侧在动。

## 当前发布数字

见 [README](../../README_zh.md#性能基准) 中 RTX 3060 与
RTX 5060 Ti 的表格。复现方法：

```bash
git clone https://github.com/Hai-Wenxiang/fusedtok.git
cd fusedtok
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
PYTHONPATH=$PWD/build python benchmarks/bench.py --iters 60
```

（Windows 请在 VS 开发者命令行里执行 cmake 配置；见
[CONTRIBUTING](../../CONTRIBUTING.md)。）

## 诚实原则

1. README 里的每个数字都能用同目录树里的 JSON 按上述协议再生成。
2. 输的行与赢的行并列发布，输的原因（设计范围 vs 实测差距）在
   正文写明。
3. 协议或输入分布一旦变更，都会记入 CHANGELOG（如 1.1.1 的播种
   修正、1.2 的峰值/平坦 regime 修正——两者都纠正过本仓库先前
   发布过的数字）。
