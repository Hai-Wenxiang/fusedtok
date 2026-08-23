# fusedtok

[![CI](https://github.com/Hai-Wenxiang/fusedtok/actions/workflows/ci.yml/badge.svg)](https://github.com/Hai-Wenxiang/fusedtok/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/fusedtok.svg)](https://pypi.org/project/fusedtok/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/Hai-Wenxiang/fusedtok/blob/main/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://github.com/Hai-Wenxiang/fusedtok/blob/main/pyproject.toml)

**闈㈠悜 LLM 鎺ㄧ悊鐨勮瀺鍚?CUDA 绠楀瓙搴?* 鈥斺€?RMSNorm / RoPE / SwiGLU 绛夛紝鏀寔
**torch 寮犻噺闆舵嫹璐?*锛氬姣?PyTorch eager 鏈€楂?**6.2 鍊嶅姞閫?*锛圧oPE锛?
RTX 3060锛岃[鎬ц兘鍩哄噯](#鎬ц兘鍩哄噯)锛夈€?

**English version: [README.md](https://github.com/Hai-Wenxiang/fusedtok/blob/main/README.md)**

## 涓轰粈涔堝仛

LLM 鎺ㄧ悊妗嗘灦涓紝姣忎釜 token 閮借瑙﹀彂澶ч噺灏忚€屽彈鍐呭瓨甯﹀闄愬埗鐨勭畻瀛愶紝姣忔鍚姩閮借鍦ㄦ樉瀛橀棿鏉ュ洖鎼繍鏁版嵁銆?

`fusedtok` 灏嗗畠浠瀺鍚堜负鍗曚竴 kernel锛屽墛鍑忔樉瀛樻祦閲忎笌鍚姩寮€閿€銆?

## 绠楀瓙璺嚎

| 鐘舵€?| 绠楀瓙 | 璇存槑 |
| :---: | --- | --- |
| 鉁?| RMSNorm锛堝惈娈嬪樊锛?| LLaMA/Qwen 椋庢牸锛岃瀺鍚堟畫宸姞娉?|
| 鉁?| LayerNorm | 鍚豢灏勫彉鎹?|
| 鉁?| RoPE | 浜ら敊涓?NeoX 涓ょ甯冨眬锛屾敮鎸?kv-cache `pos_offset` |
| 鉁?| SwiGLU | 铻嶅悎 MLP 婵€娲?|
| 鉁?| Softmax锛堟寜琛岋級 | 鏁板€肩ǔ瀹氱増 |
| 鉁?| SiLU / GeLU / GeLU-tanh / ReLU / Tanh / Sigmoid | 閫愬厓绱?|
| 鉁?| add / mul | 閫愬厓绱犱簩鍏冿紙铻嶅悎鍔犳畫宸ā寮忥級 |
| 鉁?| top-k / top-p锛堟牳閲囨牱锛?| 骞冲眬鍙栧厛涓嬫爣 |
| 鉁?| argmax / temperature | 璐績瑙ｇ爜杈呭姪 |
| 鉁?| repetition penalty | CTRL 椋庢牸锛屼綔鐢ㄤ簬宸查噰鏍?token |
| 鈴?| INT8/FP8 閲忓寲璺嚎 | v0.3 璁″垝 |

## 瀹夎

```bash
pip install fusedtok
```

PyPI 鎻愪緵 Linux x86_64 棰勭紪璇?wheel锛坢anylinux锛孋UDA 12.4 鏋勫缓锛夈€?
Windows锛堟垨鏃犲尮閰?wheel 鐨勫钩鍙帮級涓?pip 浼氳嚜鍔ㄤ粠婧愮爜鏋勫缓锛?

```bash
git clone https://github.com/Hai-Wenxiang/fusedtok.git
cd fusedtok
pip install .
```

**鐜瑕佹眰**锛?

- **RTX 30 绯伙紙Ampere锛夋垨鏇存柊**鐨?NVIDIA 鏄惧崱 鈥斺€?濡?RTX 3060/3090銆丷TX 4080銆丷TX 5090銆丄100銆丠100
- CUDA Toolkit 鈮?12.0
- C++17 缂栬瘧鍣紙Windows 鐢?MSVC锛孡inux 鐢?GCC/Clang锛夛紱Python 3.10+

<details>
<summary>浠€涔堟槸 "compute capability"锛堣绠楄兘鍔涳級锛熺偣鍑诲睍寮€</summary>

璁＄畻鑳藉姏鏄?NVIDIA 缁欐瘡浠?GPU 鏋舵瀯鐨?*鐗堟湰缂栧彿**锛屼笉鏄€ц兘鍒嗘暟銆侰UDA 浠ｇ爜蹇呴』閽堝鐗瑰畾鏋舵瀯缂栬瘧鎵嶈兘杩愯銆?
鏋勫缓浜х墿鍖呭惈璁＄畻鑳藉姏 8.0锛圓100锛変笌 8.6锛圧TX 30锛夌殑鍘熺敓 cubin锛屼互鍙?compute_86 PTX 鍥為€€锛?
Ampere 鍘熺敓杩愯锛屾洿鏂版灦鏋勶紙RTX 40/50 绛夛級鐢遍┍鍔ㄥ嵆鏃剁紪璇?PTX銆?

| 璁＄畻鑳藉姏 | 鏋舵瀯 | 浠ｈ〃鏄惧崱 |
|---|---|---|
| 7.5 | Turing | GTX 16xx銆丷TX 20xx锛堜笉鏀寔锛?|
| 8.0 / 8.6 | Ampere | A100銆丷TX 30xx |
| 8.9 | Ada | RTX 40xx锛堣蛋 PTX锛?|
| 9.0 | Hopper | H100锛堣蛋 PTX锛?|
| 12.0 | Blackwell | RTX 50xx锛堣蛋 PTX锛?|

鏌ョ湅鑷繁鐨勬樉鍗★細杩愯 `nvidia-smi` 鐪嬪埌鍨嬪彿鍚庯紝鍦?https://developer.nvidia.com/cuda-gpus 鏌ヨ瀵瑰簲璁＄畻鑳藉姏銆?

</details>

## 鐢ㄦ硶

numpy 杩?/ numpy 鍑猴紝鎴?torch 杩?/ torch 鍑?鈥斺€?骞舵敮鎸?**CUDA 闆舵嫹璐?*锛?
kernel 閫氳繃 `data_ptr()` 鐩存帴璇诲啓 torch 鏄惧瓨缂撳啿鍖猴紝鏃犳殏瀛樻嫹璐濄€佹棤涓绘満绔悓姝ャ€?

```python
import numpy as np
import torch
import fusedtok

x = np.random.randn(4, 1024).astype(np.float32)
w = np.random.rand(1024).astype(np.float32)

# CPU 鍙傝€冨疄鐜帮紙姝ｇ‘鎬у熀鍑嗭紝浠讳綍鏈哄櫒鍙窇锛?
y = fusedtok.rmsnorm(x, w, eps=1e-6)

# 鏆傚瓨寮?CUDA锛氭嫹鍏?GPU銆佽窇 kernel銆佹嫹鍥?
y = fusedtok.rmsnorm(x, w, cuda=True)

# torch 寮犻噺闆舵嫹璐濓細kernel 鐩存帴鍦?torch 鑷繁鐨勭紦鍐插尯涓婅繍琛岋紝
# 涓庡叾浠?torch 鎿嶄綔淇濇寔娴佸紡椤哄簭
xt, wt = torch.from_numpy(x).cuda(), torch.from_numpy(w).cuda()
yt = fusedtok.rmsnorm(xt, wt)          # -> CUDA torch 寮犻噺

# 甯?kv-cache 浣嶇疆鍋忕Щ鐨?RoPE锛孨eoX锛圠LaMA-HF锛夊竷灞€
q = torch.randn(1, 4096, device="cuda")          # 鍙紶鍏ユ柊 token
q_rot, k_rot = fusedtok.rope(q, k=None, pos_offset=1023, neox=True)

# 閲囨牱渚?
logits = fusedtok.repetition_penalty(logits, sampled_ids, penalty=1.1)
values, indices = fusedtok.topk(logits, k=50)
```

鎵€鏈夊嚱鏁版帴鍙?float32 鐨?numpy 鏁扮粍鎴?torch 寮犻噺锛堝叾浠?dtype 浼氳鎷疯礉杞崲锛夛紝
杩斿洖鍚屾棌鐨?float32 杈撳嚭銆侰UDA torch 寮犻噺浼氳嚜鍔ㄩ€夋嫨闆舵嫹璐濊矾寰勩€?

瀹屾暣鍙繍琛岀殑绠楀瓙宸¤瑙?`examples/demo.py`銆?

## 姝ｇ‘鎬?

- 姣忎釜绠楀瓙鍧囬檮甯?**CPU 鍙傝€冨疄鐜?* 涓庨€愬厓绱犲鎷嶆祴璇曪紙pytest锛?
- 鏃?GPU 鐨勬満鍣ㄤ篃鑳借繍琛屾祴璇曪紙CUDA 鐢ㄤ緥鑷姩璺宠繃锛?

## 鎬ц兘鍩哄噯

RTX 3060锛坰m_86锛夈€乫loat32銆乼orch 闆舵嫹璐濆紶閲忋€丆UDA event 璁℃椂锛屽姣旂瓑浠风殑
PyTorch eager 琛ㄨ揪寮忥紙瀹屾暣鏁版嵁锛歚docs/benchmark_results.json`锛屽彲鐢?
`python benchmarks/bench.py` 澶嶇幇锛夛細

| 绠楀瓙 | 褰㈢姸 | fusedtok | PyTorch eager | 鍔犻€熸瘮 |
|---|---|---:|---:|---:|
| RoPE NeoX (q+k) | [2048脳4096] | 416 碌s | 2570 碌s | **6.2x** |
| RMSNorm锛堝惈娈嬪樊锛?| [1024脳4096] | 260 碌s | 538 碌s | **2.1x** |
| SwiGLU | [1024脳4096] | 153 碌s | 257 碌s | **1.7x** |
| LayerNorm | [1024脳4096] | 168 碌s | 162 碌s | ~1.0x |
| SiLU | [1024脳4096] | 105 碌s | 112 碌s | ~1.0x |
| Softmax | [1024脳4096] | 159 碌s | 115 碌s | 0.7x |
| argmax | [131072] | 36 碌s | 46 碌s | **1.3x** |
| top-k (k=50) | [131072] | 168 碌s | 129 碌s | 0.8x |

![fusedtok 瀵规瘮 PyTorch eager](https://raw.githubusercontent.com/Hai-Wenxiang/fusedtok/main/docs/benchmark_rt3060.png)

铻嶅悎绠楀瓙锛圧oPE / RMSNorm / SwiGLU锛変紭鍔挎槑鏄撅細eager 妯″紡鐨勪腑闂村紶閲忚鍦ㄦ樉瀛橀棿
鏉ュ洖鎼繍銆傜函甯﹀鍙楅檺鐨勯€愬厓绱犵畻瀛愪笌 PyTorch 璋冧紭 kernel 璺戝嚭鐩稿悓鐨?
~330-500 GB/s锛坰ilu銆乬elu銆乤dd 鈮?鎸佸钩锛夈€係oftmax 涓?top-k 浠嶈惤鍚庝簬 PyTorch
鐨?CUB 鍐呮牳 鈥斺€?鏁板瓧璇氬疄锛屾敼杩涘垪鍏?v0.2 瑙勫垝銆?

## 寮€鍙?

瀹屾暣鎸囧崡锛堟祴璇曡鍒欍€侀敊璇绾︺€佺‘瀹氭€х害瀹氾級瑙?[CONTRIBUTING.md](https://github.com/Hai-Wenxiang/fusedtok/blob/main/CONTRIBUTING.md)銆傚揩閫熶笂鎵嬶細

```bash
# Windows锛氶渶鍦?VS 寮€鍙戣€呭懡浠よ锛坴cvars64锛変腑鎵ц
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
# 鍦ㄤ粨搴撴牴鐩綍锛歅YTHONPATH 鎸囧悜鏋勫缓浜х墿锛宑onftest.py 浼氳嚜鍔ㄥ姞 python/ 鐩綍
$env:PYTHONPATH = "$PWD/build"        # Windows
PYTHONPATH=$PWD/build                 # Linux
python -m pytest tests -q
python benchmarks/bench.py            # GPU 鍩哄噯娴嬭瘯 + 鍑哄浘
```

- 鏀寔 Windows / Linux
- Windows 涓嬬敱 MSVC 閰嶅悎 nvcc 缂栬瘧
- CI 鍦ㄦ瘡娆℃帹閫佹椂鏋勫缓骞惰繍琛?CPU 娴嬭瘯濂椾欢

## 璺嚎鍥?

- v0.2锛歜f16 鏀寔銆乺adix-select top-k/top-p锛圕UB 绾ч€熷害锛夈€佽瀺鍚堥噰鏍?
  锛坰oftmax+top-p+鎶芥牱鍗曡稛瀹屾垚锛夈€丆UDA graph 鍙嬪ソ鎵瑰鐞?
- v0.3锛欼NT8/FP8 閲忓寲璺嚎銆乥lock size 鑷姩璋冧紭
- v0.4+锛氳交閲忚瀺鍚?attention锛汸yPI 棰勭紪璇?wheel

## 绀惧尯

- [璐＄尞鎸囧崡](https://github.com/Hai-Wenxiang/fusedtok/blob/main/CONTRIBUTING.md) 鈥斺€?鐜鎼缓銆佽鍒欎笌 PR 娴佺▼
- [琛屼负鍑嗗垯](https://github.com/Hai-Wenxiang/fusedtok/blob/main/CODE_OF_CONDUCT.md)
- [瀹夊叏绛栫暐](https://github.com/Hai-Wenxiang/fusedtok/blob/main/SECURITY.md)
- [鏇存柊鏃ュ織](https://github.com/Hai-Wenxiang/fusedtok/blob/main/CHANGELOG.md)

## 璁稿彲璇?

MIT 鈥斺€?瑙?[LICENSE](https://github.com/Hai-Wenxiang/fusedtok/blob/main/LICENSE)锛涚涓夋柟澹版槑瑙?[NOTICES.md](https://github.com/Hai-Wenxiang/fusedtok/blob/main/NOTICES.md)銆?
