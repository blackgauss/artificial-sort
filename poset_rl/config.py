"""Experiment configuration dataclasses — serialisable to/from YAML.

Usage
-----
cfg = ExperimentConfig(
    name  = "my_run",
    model = ModelConfig(name="jax_attention", hidden=64, nhead=2, nlayers=1),
    train = TrainConfig(n_or_range=[3,4,5,6], episodes=5000),
    eval  = EvalConfig(ns=[3,4,5,6], episodes=300),
)
cfg.to_yaml("configs/my_run.yaml")

cfg2 = ExperimentConfig.from_yaml("configs/my_run.yaml")
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Union

try:
    import yaml
    _YAML_OK = True
except ImportError:
    _YAML_OK = False


@dataclass
class ModelConfig:
    name:    str                      # registry key: "mlp", "attention", "jax_mlp", ...
    hidden:  int  = 64
    nhead:   int  = 2
    nlayers: int  = 1
    kwargs:  dict = field(default_factory=dict)

    @property
    def framework(self) -> str:
        return "jax" if self.name.startswith("jax_") else "torch"


@dataclass
class TrainConfig:
    n_or_range:     Union[int, List[int]] = 5
    episodes:       int                   = 3000
    lr:             float                 = 3e-3
    gamma:          float                 = 0.99
    grad_clip:      float                 = 1.0
    value_coef:     float                 = 0.5
    log_interval:   int                   = 500
    seed:           int                   = 42
    out_csv:        str                   = "training.csv"
    batch_episodes: int                   = 8


@dataclass
class EvalConfig:
    ns:       List[int] = field(default_factory=lambda: [3, 4, 5, 6])
    episodes: int       = 300


@dataclass
class ExperimentConfig:
    name:  str
    model: ModelConfig
    train: TrainConfig       = field(default_factory=TrainConfig)
    eval:  EvalConfig        = field(default_factory=EvalConfig)

    # ── YAML serialisation ────────────────────────────────────────────────

    def to_yaml(self, path: str):
        if not _YAML_OK:
            raise ImportError("pip install pyyaml")
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        if not _YAML_OK:
            raise ImportError("pip install pyyaml")
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(
            name  = d["name"],
            model = ModelConfig(**d["model"]),
            train = TrainConfig(**d["train"]),
            eval  = EvalConfig(**d.get("eval", {})),
        )
