import abc
import random
import collections
import pickle
from typing import Any, Callable, Deque, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from board import board
from features import feature, info, error

try:
    board.lookup.init()
except Exception:
    pass

class Learner(abc.ABC):
    """
    Abstract interface for RL learners.
    Subclasses should implement:
      - select_action(state, eps)
      - update(s, a, r, s_next, a_next, done)
      - optional store_transition/learn for experience-based
      - save(path), load(path)
    """
    @abc.abstractmethod
    def select_action(self, s: Any, eps: float) -> int:
        pass

    @abc.abstractmethod
    def update(
        self,
        s: Any,
        a: Optional[int],
        r: float,
        s_next: Any,
        a_next: Optional[int],
        done: bool
    ) -> None:
        pass

    def store_transition(
        self, s: Any, a: int, r: float, s_next: Any, done: bool
    ) -> None:
        # default: no-op for tabular or on-policy learners
        pass

    def learn(self) -> None:
        # default: no-op
        pass

    @abc.abstractmethod
    def save(self, path: str) -> None:
        pass

    @abc.abstractmethod
    def load(self, path: str) -> None:
        pass


class FeatureTD0Learner(Learner):
    """
    N-tuple afterstate Q‑learning: off-policy TD(0) on afterstates (no popup) with detailed stats.
    """
    def __init__(self, alpha: float = 0.1, gamma: float = 0.99, sparse: bool = True):
        self.alpha = alpha
        self.gamma = gamma
        self.sparse = sparse
        self.features: list[feature] = []
        self.scores: list[float] = []
        self.maxtile: list[int] = []

    def add_feature(self, feat: feature) -> None:
        orig_size = feat.size()
        if self.sparse:
            feat.weight = collections.defaultdict(float)
        self.features.append(feat)
        sign = f"{feat.name()}, size = {orig_size}"
        usage = orig_size * 4
        if   usage >= (1 << 30): size = f"{usage >> 30}GB"
        elif usage >= (1 << 20): size = f"{usage >> 20}MB"
        elif usage >= (1 << 10): size = f"{usage >> 10}KB"
        else:                    size = f"{usage}B"
        info(f"{sign} ({size})")

    def select_action(self, s: int, eps: float) -> int:
        """ε-greedy over afterstate Q-values (no popup simulation)."""
        if random.random() < eps:
            return random.randrange(4)
        best_a, best_v = 0, -float('inf')
        for a in range(4):
            b = board(s).clone()
            reward = b.move(a)
            if reward == -1:
                continue
            # value = estimated V(afterstate)
            value = sum(f.estimate(b) for f in self.features)
            q = reward + self.gamma * value
            if q > best_v:
                best_v, best_a = q, a
        return best_a

    def update(self, s: int, a: int, r: float, s_next: int, a_next: Any, done: bool) -> None:
        """Q-learning update on afterstates."""
        # compute afterstate for (s,a)
        after0 = board(s).clone()
        reward0 = after0.move(a)
        if reward0 == -1:
            return
        # current estimate
        v0 = sum(f.estimate(after0) for f in self.features)
        # compute max next afterstate value from same s (off-policy):
        if done:
            v_next_max = 0.0
        else:
            values = []
            for a2 in range(4):
                after1 = board(s).clone()
                r1 = after1.move(a2)
                if r1 == -1:
                    continue
                values.append(sum(f.estimate(after1) for f in self.features))
            v_next_max = max(values) if values else 0.0
        # Q-learning target: r + gamma * v_next_max
        delta = r + self.gamma * v_next_max - v0
        adjust = self.alpha * delta / len(self.features)
        # update weights on after0
        for f in self.features:
            f.update(after0, adjust)

    def make_statistic(self, n: int, b: board, score: int, unit: int = 1000) -> None:
        # (unchanged)
        self.scores.append(score)
        self.maxtile.append(max(b.at(i) for i in range(16)))
        if n % unit == 0:
            if len(self.scores) != unit or len(self.maxtile) != unit:
                error("wrong statistic size for show statistics")
                exit(2)
            avg_score = sum(self.scores) / unit
            max_score = max(self.scores)
            info(f"{n}	avg = {avg_score:.1f}	max = {max_score}")
            stat = [self.maxtile.count(t) for t in range(16)]
            coef = 100.0 / unit
            c = 0
            t = 1
            while c < unit and t < 16:
                cnt = stat[t]
                if cnt:
                    accu = sum(stat[t:])
                    tile = 1 << t
                    winrate = accu * coef
                    share = cnt * coef
                    info(f"	{tile}	{winrate:.1f}%	({share:.1f}%)")
                c += cnt; t += 1
            self.scores.clear(); self.maxtile.clear()
            
    def save(self, path: str) -> None:
        """Pickle the feature list."""
        with open(path, 'wb') as f:
            pickle.dump(self.features, f)
        info(f"Saved features to {path}")

    def load(self, path: str) -> None:
        """Load feature list from pickle."""
        try:
            with open(path, 'rb') as f:
                self.features = pickle.load(f)
            info(f"Loaded features from {path}")
        except FileNotFoundError:
            pass

class FeatureTDLambdaLearner(FeatureTD0Learner):
    """
    N-tuple afterstate TD(λ) learner with eligibility traces.
    """
    def __init__(self, alpha: float = 0.1, gamma: float = 0.99, lam: float = 0.8, sparse: bool = True):
        super().__init__(alpha, gamma, sparse)
        self.lam = lam
        self.traces: list[float] = []

    def add_feature(self, feat: feature) -> None:
        """Add a feature and initialize its eligibility trace."""
        # capture original table size before sparse override
        orig_size = feat.size()
        super().add_feature(feat)
        self.traces.append(0.0)

    def reset_traces(self) -> None:
        """Reset all eligibility traces (call at episode start)."""
        self.traces = [0.0] * len(self.features)

    def select_action(self, s: int, eps: float) -> int:
        # same as TD0
        return super().select_action(s, eps)

    def update(self, s: int, a: Any, r: float, s_next: int, a_next: Any, done: bool) -> None:
        """TD(λ) update with accumulating eligibility traces."""
        b0 = board(s)
        v0 = sum(f.estimate(b0) for f in self.features)
        b1 = board(s_next)
        v1 = 0.0 if done else sum(f.estimate(b1) for f in self.features)
        delta = r + self.gamma * v1 - v0
        # accumulate traces and update weights
        for i, f in enumerate(self.features):
            self.traces[i] = self.gamma * self.lam * self.traces[i] + 1.0
            f.update(b1, self.alpha * delta * self.traces[i])

    def make_statistic(self, n: int, b: board, score: int, unit: int = 1000) -> None:
        super().make_statistic(n, b, score, unit)

    def save(self, path: str) -> None:
        super().save(path)

    def load(self, path: str) -> None:
        super().load(path)
