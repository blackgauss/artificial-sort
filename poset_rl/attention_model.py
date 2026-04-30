"""Backward-compatibility shim — import from poset_rl.models.attention instead."""
from poset_rl.models.attention import AttentionActorCritic, _pair_features, PAIR_FEAT_DIM  # noqa: F401

TORCH_AVAILABLE = True

__all__ = ["AttentionActorCritic", "_pair_features", "PAIR_FEAT_DIM", "TORCH_AVAILABLE"]
