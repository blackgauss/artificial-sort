"""Poset environment for RL.

State:  n×n matrix of known relations, flattened to a float32 vector.
        known[i,j] =  1  →  i > j confirmed
                     -1  →  j > i confirmed
                      0  →  unknown
Action: pick an unordered pair (i,j) with i<j — n*(n-1)/2 discrete actions.
Reward: -1 per explicit comparison query (deterministic proxy for compute cost).
        Transitive inferences after each query are free (zero cost).
"""
import numpy as np


class PosetEnv:
    def __init__(self, true_rel: np.ndarray):
        """true_rel: full adjacency matrix of the ground-truth poset relation:
        true_rel[i,j] = 1 if i>j, -1 if j>i, 0 if incomparable (but we assume no ties). For total orders use 1/-1.
        """
        self.true_rel = true_rel.copy()
        self.n = self.true_rel.shape[0]
        self.known = np.zeros((self.n, self.n), dtype=np.int8)
        self.steps = 0

    @staticmethod
    def total_order_from_perm(perm: np.ndarray) -> np.ndarray:
        n = len(perm)
        rel = np.zeros((n, n), dtype=np.int8)
        # perm is array of elements in descending order: perm[0] > perm[1] > ...
        pos = np.empty(n, dtype=int)
        pos[perm] = np.arange(n)
        for i in range(n):
            for j in range(n):
                if pos[i] < pos[j]:
                    rel[i, j] = 1
                elif pos[i] > pos[j]:
                    rel[i, j] = -1
        return rel

    def reset(self):
        self.known.fill(0)
        self.steps = 0
        return self._obs()

    def _obs(self) -> np.ndarray:
        """Return the full known matrix as a flat float32 vector (length n²)."""
        return self.known.astype(np.float32).ravel()

    def legal_actions_mask(self) -> np.ndarray:
        """Float32 mask of length n*(n-1)/2; 1.0 = legal, 0.0 = already known."""
        m = self.n * (self.n - 1) // 2
        mask = np.zeros(m, dtype=np.float32)
        idx = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.known[i, j] == 0 and self.known[j, i] == 0:
                    mask[idx] = 1.0
                idx += 1
        return mask

    def _pair_from_action(self, action: int) -> tuple[int, int]:
        idx = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if idx == action:
                    return i, j
                idx += 1
        raise IndexError("action out of range")

    def step(self, action: int):
        i, j = self._pair_from_action(action)
        # reveal comparison according to true_rel
        val = self.true_rel[i, j]
        if val == 1:
            self.known[i, j] = 1
            self.known[j, i] = -1
        elif val == -1:
            self.known[i, j] = -1
            self.known[j, i] = 1
        else:
            # incomparable; mark both directions as special code 2/-2
            self.known[i, j] = 2
            self.known[j, i] = -2
        self.steps += 1
        # propagate transitive closure: any new inference is free
        self._close_transitivity()
        reward = -1.0
        done = (self.legal_actions_mask().sum() == 0)
        return self._obs(), reward, done, {}

    def _close_transitivity(self):
        """Floyd-Warshall-style single pass to add transitive inferences."""
        n = self.n
        changed = True
        while changed:
            changed = False
            for a in range(n):
                for b in range(n):
                    if self.known[a, b] != 1:
                        continue  # a > b known
                    for c in range(n):
                        if self.known[b, c] != 1:
                            continue  # b > c known
                        # infer a > c
                        if self.known[a, c] == 0:
                            self.known[a, c] = 1
                            self.known[c, a] = -1
                            changed = True

    def render(self):
        print("Known relations (rows > cols):")
        print(self.known)
