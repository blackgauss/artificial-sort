"""poset_rl — RL environment for learning efficient sorting strategies.

Quick start
-----------
>>> from poset_rl import PosetEnv, get_model, ExperimentConfig
>>> from poset_rl.train import train_from_config
"""

from poset_rl.env import PosetEnv
from poset_rl.datasets import make_sampler
from poset_rl.config import ExperimentConfig, ModelConfig, TrainConfig, EvalConfig
from poset_rl.models import get_model, list_models, register

# convenience re-exports for common models
from poset_rl.models.mlp import ActorCritic
from poset_rl.models.attention import AttentionActorCritic

TORCH_AVAILABLE = True
try:
    import jax  # noqa: F401
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

__all__ = [
    "PosetEnv",
    "make_sampler",
    "ExperimentConfig", "ModelConfig", "TrainConfig", "EvalConfig",
    "get_model", "list_models", "register",
    "ActorCritic", "AttentionActorCritic",
    "TORCH_AVAILABLE", "JAX_AVAILABLE",
]
