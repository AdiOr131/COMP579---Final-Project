import numpy as np

def epsilon_greedy(Q: dict, s: int, n_actions: int, eps: float) -> int:
    if np.random.rand() < eps or s not in Q:
        return np.random.randint(n_actions)
    return int(np.argmax(Q[s]))