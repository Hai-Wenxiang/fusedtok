"""Type stubs for fusedtok.

Every operator accepts either a numpy array or a torch tensor and
returns the same family: float32 numpy in -> float32 numpy out; a CUDA
torch tensor in -> CUDA torch tensor out (the zero-copy path). The
samplers are the documented exception: they return tokens (int /
int64 array), and the batched variants always return int64 on the
HOST (a CPU torch tensor or numpy array) - the widening loop's host
readback is inherent to returning tokens at all. Torch is
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
sample_minp / the sample_*_batched variants / decode_step /
decode_step_batched) is deterministic per seed (per row-seed for the
batched variants); the RNG is a splitmix-style hash - reproducible,
NOT cryptographically secure.
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
    "sample_minp",
    "sample_topp_batched",
    "sample_topk_batched",
    "sample_minp_batched",
    "quantize_int8",
    "dequantize_int8",
    "qadd_int8",
    "qgemm",
    "qgemm_perchannel",
    "decode_step",
    "decode_step_batched",
    "attention_decode",
    "kv_append",
    "attention_decode_paged",
    "attention_prefill",
    "kv_append_paged",
]

def cuda_available() -> bool: ...
def axpy(x: Array, a: float = 1.0, b: float = 0.0,
         *, cuda: bool = False) -> Array: ...
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
def sample_minp(logits: Array, min_p: float, *, temperature: float = 1.0,
                seed: int = 0, cuda: bool = False) -> int: ...
def sample_topp_batched(logits: Array, p: float, *,
                        temperature: float = 1.0,
                        seeds: Optional[Array] = None,
                        cuda: bool = False) -> Array: ...
def sample_topk_batched(logits: Array, k: int, *, temperature: float = 1.0,
                        seeds: Optional[Array] = None,
                        cuda: bool = False) -> Array: ...
def sample_minp_batched(logits: Array, min_p: float, *,
                        temperature: float = 1.0,
                        seeds: Optional[Array] = None,
                        cuda: bool = False) -> Array: ...
def quantize_int8(x: Array) -> "tuple[Array, float]": ...
def dequantize_int8(q: Array, scale: float) -> Array: ...
def qadd_int8(qa: Array, sa: float, qb: Array,
              sb: float) -> "tuple[Array, float]": ...
def qgemm(a_q: Array, a_scale: float, b_q: Array, b_scale: float, *,
          cuda: bool = False) -> Array: ...
def qgemm_perchannel(a_q: Array, a_scale: float, b_q: Array, b_scales: Array,
                     *, cuda: bool = False) -> Array: ...
def decode_step(logits: Array, sampled_ids: Array, penalty: float = 1.0, *,
                p: float = 0.9, temperature: float = 1.0, seed: int = 0,
                cuda: bool = False) -> int: ...
def decode_step_batched(logits: Array, sampled_ids: Array,
                        penalty: float = 1.0, *, p: float = 0.9,
                        temperature: float = 1.0,
                        seeds: Optional[Array] = None,
                        ids_offsets: Optional[Array] = None,
                        cuda: bool = False) -> Array: ...
def attention_decode(q: Array, k_cache: Array, v_cache: Array,
                     lens: Optional[Array] = None,
                     *, cuda: bool = False) -> Array: ...
def kv_append(k_cache: Array, v_cache: Array, k_new: Array, v_new: Array,
              lens: Array, *, cuda: bool = False) -> None: ...
def attention_decode_paged(q: Array, k_pool: Array, v_pool: Array,
                           block_table: Array,
                           lens: Optional[Array] = None,
                           *, cuda: bool = False) -> Array: ...
def kv_append_paged(k_pool: Array, v_pool: Array, block_table: Array,
                    k_new: Array, v_new: Array, lens: Array,
                    *, cuda: bool = False) -> None: ...
def attention_prefill(q: Array, k: Array, v: Array, causal: bool = True,
                      *, cuda: bool = False) -> Array: ...
