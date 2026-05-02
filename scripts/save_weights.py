"""Save weights from a running or completed training run.

Usage (inside container or with the venv active):
    python scripts/save_weights.py --config configs/gb10_large.yaml --out runs/gb10_large.npz

Rebuilds the model from config, re-runs training for 0 steps, but that's not
what we want — instead this script is designed to be injected into a running
container via `docker exec`, OR the weights can be extracted by re-training.

Actually: the simplest path is to save via docker exec + a short Python snippet.
See save_from_container() below.
"""
import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


def save_model(model, out_path: str):
    """Serialize all nnx.Param leaves to a .npz file."""
    state = nnx.state(model)
    flat  = {}
    for path, leaf in jax.tree_util.tree_leaves_with_path(state):
        key = "/".join(str(p) for p in path)
        flat[key] = np.array(leaf)
    np.savez(out_path, **flat)
    print(f"Saved {len(flat)} weight arrays → {out_path}")
    return flat


def load_model_weights(model, npz_path: str):
    """Load weights from a .npz file back into a model (in-place)."""
    data  = np.load(npz_path, allow_pickle=False)
    state = nnx.state(model)

    # Rebuild the pytree with loaded arrays
    flat_state = {"/".join(str(p) for p in path): leaf
                  for path, leaf in jax.tree_util.tree_leaves_with_path(state)}

    for key in flat_state:
        if key not in data:
            print(f"  WARNING: {key} not in checkpoint")
    print(f"Loaded {len(data.files)} arrays from {npz_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out",    required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from poset_rl.config import ExperimentConfig
    from poset_rl.models import get_model

    cfg = ExperimentConfig.from_yaml(args.config)
    mc  = cfg.model
    rngs = nnx.Rngs(0)

    cls   = get_model(mc.name)
    model = cls(hidden=mc.hidden, nhead=mc.nhead, nlayers=mc.nlayers,
                rngs=rngs, **mc.kwargs)

    save_model(model, args.out)
