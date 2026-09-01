"""The 1.0 public API surface: frozen names, complete docs, stubs shipped.

1.0 freezes the public names in ``fusedtok.__all__``. These tests make
an accidental addition or removal a test failure, keep every public
callable documented (a public op without a docstring is an API bug),
and pin the packaging promises of PEP 561 (py.typed + stubs travel with
the wheel).
"""

import pathlib
from importlib import import_module

import pytest

fusedtok = import_module("fusedtok")

# The frozen surface. Additions require a feature deprecation note and a
# minor version bump AFTER 1.0; removals/re-names require a major bump.
FROZEN_API = frozenset({
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
    "attention_decode_paged",
    "attention_prefill",
    "kv_append_paged",
})


def test_all_is_exactly_the_frozen_surface():
    assert set(fusedtok.__all__) == FROZEN_API


def test_every_public_name_is_importable_and_documented():
    for name in FROZEN_API:
        obj = getattr(fusedtok, name, None)
        assert obj is not None, f"__all__ name missing: {name}"
        assert callable(obj), f"{name} is not callable"
        doc = getattr(obj, "__doc__", None)
        assert doc and doc.strip(), f"{name} has no docstring"


def test_version_present_and_semver_like():
    v = fusedtok.__version__
    parts = v.split(".")
    assert len(parts) == 3 and all(p.isdigit() for p in parts), (
        f"__version__ {v!r} is not MAJOR.MINOR.PATCH")


def test_no_public_leaks_beyond_all():
    # callables that sneak into the module namespace (imports, helpers)
    # must not look public: everything public either lives in __all__ or
    # starts with an underscore
    public = {n for n in vars(fusedtok)
              if not n.startswith("_") and callable(getattr(fusedtok, n))}
    leaked = public - FROZEN_API - {"_fusedtok"}
    # _fusedtok is the compiled extension module object; it is not a
    # callable, the filter above keeps it only defensively
    assert not leaked, f"undocumented public callables: {sorted(leaked)}"


def test_stub_file_covers_the_frozen_surface():
    # the stub must ship next to the package source and name every
    # frozen symbol (text-level check: the stub is a static artifact,
    # importing it is not a thing)
    pkg_dir = pathlib.Path(fusedtok.__file__).parent
    stub = pkg_dir / "__init__.pyi"
    assert stub.is_file(), "__init__.pyi missing from the package"
    assert (pkg_dir / "py.typed").is_file(), \
        "py.typed marker missing (PEP 561)"
    text = stub.read_text(encoding="utf-8")
    for name in FROZEN_API:
        assert f"def {name}(" in text, f"{name} not stubbed"


@pytest.mark.parametrize("op", ["sample_topp", "sample_topk",
                                "decode_step", "temperature"])
def test_error_contract_value_errors(op):
    # the documented contract: value problems raise ValueError
    logits = __import__("numpy").zeros(16, dtype="float32")
    fn = getattr(fusedtok, op)
    with pytest.raises(ValueError):
        if op == "sample_topp":
            fn(logits, 0.0)                    # p must be in (0, 1]
        elif op == "sample_topk":
            fn(logits, 0)                      # k must be >= 1
        elif op == "decode_step":
            fn(logits, [0], 0.0)               # penalty must be > 0
        else:
            fn(logits, 0.0)                    # temperature must be > 0
