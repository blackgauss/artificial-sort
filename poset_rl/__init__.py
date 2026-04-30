"""Poset RL — learn efficient sorting strategies on partial orders.

Quick start
-----------
>>> from poset_rl import PosetEnv, ActorCritic, AttentionActorCritic
>>> from poset_rl.train import train
>>> from poset_rl.bench import benchmark, print_table
"""

from .env import PosetEnv
from .model import PolicyNet, ActorCritic, TORCH_AVAILABLE
from .attention_model import AttentionActorCritic
from .datasets import make_sampler

__all__ = [
    "PosetEnv",
    "PolicyNet",
    "ActorCritic",
    "AttentionActorCritic",
    "TORCH_AVAILABLE",
    "make_sampler",
]
