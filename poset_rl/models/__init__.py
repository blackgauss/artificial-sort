"""Model registry — add new architectures with @register('name').

Usage
-----
@register("mymodel")
class MyModel(nn.Module):
    ...

# anywhere else
cls = get_model("mymodel")
model = cls(hidden=64, ...)
"""
from __future__ import annotations
from typing import Callable, Dict

_REGISTRY: Dict[str, type] = {}


def register(name: str):
    """Class decorator: @register('mymodel')"""
    def _inner(cls):
        _REGISTRY[name] = cls
        return cls
    return _inner


def get_model(name: str) -> type:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[name]


def list_models():
    return list(_REGISTRY.keys())


# ── trigger registration decorators ───────────────────────────────────────
from poset_rl.models import mlp        # noqa: E402 F401
from poset_rl.models import attention  # noqa: E402 F401
try:
    from poset_rl.models import jax_mlp        # noqa: E402 F401
    from poset_rl.models import jax_attention  # noqa: E402 F401
except ImportError:
    pass  # JAX not installed — torch-only mode
