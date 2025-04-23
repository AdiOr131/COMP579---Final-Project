from typing import Any

# The learner only needs to expose select_action() and update();
# our FeatureTD0Learner already satisfies that contract.

class RLAgent:
    """Simple ε‑greedy episode runner compatible with FeatureTD0Learner.

    Parameters
    ----------
    env : Any
        An environment that provides `reset()` → state and
        `step(action)` → (next_state, reward, done, info).
    learner : Any
        Must implement `select_action(state, eps)` and
        `update(state, action, reward, next_state, a_next, done)`.
    epsilon : float, optional
        Initial exploration rate.
    decay : float, optional
        Multiplicative ε decay applied after every episode.
    eps_min : float, optional
        Minimum exploration rate.
    """

    def __init__(
        self,
        env: Any,
        learner: Any,
        epsilon: float = 0.1,
        decay: float = 0.9995,
        eps_min: float = 0.01,
    ) -> None:
        self.env = env
        self.ln = learner
        self.eps = epsilon
        self.decay = decay
        self.eps_min = eps_min

    # ───────────────────────────── public API ──────────────────────────────

    def run_episode(self) -> float:
        """Play one full game and train online.

        Returns
        -------
        float
            Sum of rewards obtained during the episode (i.e. game score).
        """
        state = self.env.reset()
        done = False
        total = 0.0

        while not done:
            # ε‑greedy action selection via learner
            action = self.ln.select_action(state, self.eps)

            # environment transition
            nxt_state, reward, done, _ = self.env.step(action)
            total += reward

            # one‑step TD update (afterstate learner ignores `a_next`)
            self.ln.update(state, action, reward, nxt_state, None, done)

            state = nxt_state
        # decay ε after the episode ends
        self.eps = max(self.eps_min, self.eps * self.decay)
        return total

    # convenience: play many episodes with optional callback ----------------

    def train(
        self,
        episodes: int,
        callback: Any | None = None,
    ) -> None:
        """Run multiple episodes and call *callback* after each one.

        The callback receives `(episode_idx, score, env)` so you can log
        statistics or render periodically.
        """
        for ep in range(1, episodes + 1):
            score = self.run_episode()
            if callback is not None:
                callback(ep, score, self.env)