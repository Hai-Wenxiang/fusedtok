"""Type stubs for fusedtok.

Every operator accepts either a numpy array or a torch tensor and
returns the same family: float32 numpy in -> float32 numpy out; a CUDA
torch tensor in -> CUDA torch tensor out (the zero-copy path). Torch is
an OPTIONAL dependency and stubs must not import it, so the array types
below are expressed as ``Array = ndarray | Any``: numpy is the concrete
half, and a checker resolves passing a torch tensor through the ``Any``
half. IDEs still get the full signature surface, which is the point.

Error contract (stable since 1.0):
- ``ValueError``  - shapes, out-of-range values (k, p, temperature, ...),
- ``TypeError``   - wrong dtype, wrong device family, mixed inputs,
- ``RuntimeError`` - CUDA execution failures (launch faults, allocs).

Selection determinism (stable since 1.0): top-k / top-p / argmax resolve
ties toward the EARLIEST index. Sampling (sample_topp / sample_topk /
decode_step) is deterministic per seed; the RNG is a splitmix-style hash
- reproducible, NOT cryptographically secure.
"""

from typing import Any, Optional, Union

from numpy import ndarray

Array = Union[ndarray, Any]

__version__: str

__all__ = [
    "cuda_available",
    "axpy",
    "rmsnorm",
    "layernorm",
    "softmax",
    "rope",
    "swiglu",
    "silu",
    "gelu",
    "gelu_tanh",
    "relu",
    "tanh",
    "sigmoid",
    "add",
    "mul",
    "temperature",
    "repetition_penalty",
    "argmax",
    "topk",
    "topp",
    "sample_topp",
    "sample_topk",
    "quantize_int8",
    "dequantize_int8",
    "qadd_int8",
    "qgemm",
    "qgemm_perchannel",
    "decode_step",
    "attention_decode",
    "attention_prefill",
]

def cuda_available() -> bool: ...
def axpy(x: Array, a: float = 1.0, b: float = 0.0, *, cuda: bool = False) -> Array: ...
def rmsnorm(x: Array, weight: Array, *, residual: Optional[Array] = None,
            eps: float = 1e-6, cuda: bool = False) -> Array: ...
def layernorm(x: Array, weight: Array, bias: Array, *, eps: float = 1e-6,
              cuda: bool = False) -> Array: ...
def softmax(x: Array, *, cuda: bool = False) -> Array: ...
def rope(q: Array, k: Optional[Array] = None, *, theta: float = 10000.0,
         pos_offset: int = 0, neox: bool = False,
         cuda: bool = False) -> "tuple[Array, Optional[Array]]": ...
def swiglu(gate: Array, up: Array, *, cuda: bool = False) -> Array: ...
def silu(x: Array, *, cuda: bool = False) -> Array: ...
def gelu(x: Array, *, cuda: bool = False) -> Array: ...
def gelu_tanh(x: Array, *, cuda: bool = False) -> Array: ...
def relu(x: Array, *, cuda: bool = False) -> Array: ...
def tanh(x: Array, *, cuda: bool = False) -> Array: ...
def sigmoid(x: Array, *, cuda: bool = False) -> Array: ...
def add(a: Array, b: Array, *, cuda: bool = False) -> Array: ...
def mul(a: Array, b: Array, *, cuda: bool = False) -> Array: ...
def temperature(x: Array, t: float, *, cuda: bool = False) -> Array: ...
def repetition_penalty(logits: Array, token_ids: Array, penalty: float,
                       *, cuda: bool = False) -> Array: ...
def argmax(x: Array, *, cuda: bool = False) -> int: ...
def topk(x: Array, k: int, *,
         cuda: bool = False) -> "tuple[Array, Array]": ...
def topp(probs: Array, p: float, *,
         cuda: bool = False) -> "tuple[Array, Array]": ...
def sample_topp(logits: Array, p: float, *, temperature: float = 1.0,
                seed: int = 0, cuda: bool = False) -> int: ...
def sample_topk(logits: Array, k: int, *, temperature: float = 1.0,
                seed: int = 0, cuda: bool = False) -> int: ...
def quantize_int8(x: Array) -> "tuple[Array, Any]": ...
def dequantize_int8(q: Array, scale: Any) -> Array: ...
def qadd_int8(qa: Array, sa: float, qb: Array,
              sb: float) -> "tuple[Array, Any]": ...
def qgemm(a_q: Array, a_scale: float, b_q: Array, b_scale: float, *,
          cuda: bool = False) -> Array: ...
def qgemm_perchannel(a_q: Array, a_scale: float, b_q: Array, b_scales: Array,
                     *, cuda: bool = False) -> Array: ...
def decode_step(logits: Array, sampled_ids: Array, penalty: float = 1.0, *,
                p: float = 0.9, temperature: float = 1.0, seed: int = 0,
                cuda: bool = False) -> int: ...
def attention_decode(q: Array, k_cache: Array, v_cache: Array,
                     lens: Optional[Array] = None,
                     *, cuda: bool = False) -> Array: ...
def attention_prefill(q: Array, k: Array, v: Array, causal: bool = True,
                      *, cuda: bool = False) -> Array: ...
