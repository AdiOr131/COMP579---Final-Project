import abc
import random
import pickle
from typing import Any, Optional, List

import numpy as np
from board import board
from features import feature, info, error


__all__ = [
    "Learner",
    "FeatureTD0Learner",
]

# ───────────────────────────── base interface ──────────────────────────────

class Learner(abc.ABC):
    """Minimal RL‑learner interface so RLAgent can treat any algorithm the same."""

    @abc.abstractmethod
    def select_action(self, s: Any, eps: float) -> int: ...

    @abc.abstractmethod
    def update(
        self,
        s: Any,
        a: Optional[int],
        r: float,
        s_next: Any,
        a_next: Optional[int],
        done: bool,
    ) -> None: ...

    # DQN‑style optional hooks (ignored by TD(0))
    def store_transition(self, *_, **__):
        pass

    def learn(self):
        pass

    # checkpoint helpers -----------------------------------------------------
    @abc.abstractmethod
    def save(self, path: str): ...

    @abc.abstractmethod
    def load(self, path: str): ...


# ─────────────────────── TD(0) after‑state learner ─────────────────────────

class FeatureTD0Learner(Learner):
    """After‑state TD(0) learner using an *additive* set of n‑tuple features.

    For each legal move we evaluate Q(s,a) = r(a) + γ·V(afterstate).
    Learning updates the *afterstate* value function V approximated by the
    sum of all registered feature tables.
    """

    def __init__(self, alpha: float = 0.1, gamma: float = 0.99):
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.features: List[feature] = []
        # --- statistics (optional) ---
        self._scores: List[float] = []
        self._maxtile: List[int] = []

    # ──────────────────────── feature management ──────────────────────────

    def add_feature(self, feat: feature) -> None:
        """Register a pre‑constructed feature table.

        *No* sparse override here — the Pattern class indexes `weight` as a list;
        replacing it with a dict would break indexing semantics.
        """
        self.features.append(feat)
        usage_bytes = feat.size() * 4  # float32
        if   usage_bytes >= 1 << 30:
            usage = f"{usage_bytes >> 30} GB"
        elif usage_bytes >= 1 << 20:
            usage = f"{usage_bytes >> 20} MB"
        elif usage_bytes >= 1 << 10:
            usage = f"{usage_bytes >> 10} KB"
        else:
            usage = f"{usage_bytes} B"
        info(f"Registered feature: {feat.name()} (size = {feat.size()}, {usage})")

    # ─────────────────────── policy (ϵ‑greedy) ────────────────────────────

    def select_action(self, s: int, eps: float) -> int:
        """Return an action ∈ {0,1,2,3} using ϵ‑greedy after‑state values."""
        if random.random() < eps:
            return random.randrange(4)

        best_a, best_q = 0, -float("inf")
        for a in range(4):
            after = board(s)
            r = after.move(a)
            if r == -1:
                continue  # illegal
            v = sum(f.estimate(after) for f in self.features)
            q = r + self.gamma * v
            if q > best_q:
                best_a, best_q = a, q
        return best_a

    # ───────────────────────── TD(0) update ───────────────────────────────

    def update(
        self,
        s: int,
        a: Optional[int],
        r: float,
        s_next: int,
        a_next: Optional[int],
        done: bool,
    ) -> None:
        if a is None or a < 0 or a > 3:
            return  # ignore invalid calls
        if not self.features:
            return  # nothing to train yet

        # --- current afterstate value --------------------------------------
        after0 = board(s)
        r0 = after0.move(a)
        if r0 == -1:
            return  # illegal move slipped through
        v0 = sum(f.estimate(after0) for f in self.features)

        # --- bootstrap target ---------------------------------------------
        if done:
            target = r0  # no future value
        else:
            best_q = -float("inf")
            for a2 in range(4):
                after1 = board(s_next)
                r1 = after1.move(a2)
                if r1 == -1:
                    continue
                v1 = sum(f.estimate(after1) for f in self.features)
                best_q = max(best_q, r1 + self.gamma * v1)
            target = r0 + self.gamma * (0.0 if best_q == -float("inf") else best_q)

        # --- weight update --------------------------------------------------
        delta = target - v0
        step = self.alpha * delta / len(self.features)
        for f in self.features:
            f.update(after0, step)

    # ────────────────────────── utils / I/O ───────────────────────────────

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.features, f)
        info(f"[FeatureTD0] saved feature list → {path}")

    def load(self, path: str):
        try:
            with open(path, "rb") as f:
                self.features = pickle.load(f)
            info(f"[FeatureTD0] loaded feature list ← {path}")
        except FileNotFoundError:
            error(f"Cannot load learner weights: {path} (file not found)")

    # ────────────────────────── simple stats (optional) ───────────────────

    def record_episode(self, b: board, score: float):
        self._scores.append(score)
        self._maxtile.append(max(b.at(i) for i in range(16)))

    def flush_stats(self, n: int, unit: int = 1000):
        if n % unit != 0 or not self._scores:
            return
        avg_score = sum(self._scores) / len(self._scores)
        max_score = max(self._scores)
        info(f"{n}\tavg = {avg_score:.1f}\tmax = {max_score}")
        # tile distribution
        counts = [self._maxtile.count(t) for t in range(16)]
        coef = 100 / len(self._scores)
        for t in range(1, 16):
            if counts[t]:
                win = sum(counts[t:]) * coef
                share = counts[t] * coef
                info(f"\t{1<<t}\t{win:.1f}%\t({share:.1f}%)")
        self._scores.clear(); self._maxtile.clear()
