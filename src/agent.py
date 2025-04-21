from typing import Any

class RLAgent:
    """
    Single‐episode runner that works with both ApproxQLearning and DQNLearner.
    """

    def __init__(self, env: Any, learner: Any, epsilon=0.1, decay=0.9995, eps_min=0.01):
        self.env     = env
        self.ln      = learner
        self.eps     = epsilon
        self.decay   = decay
        self.eps_min = eps_min

    def run_episode(self) -> float:
        s, total, done = self.env.reset(), 0.0, False

        while not done:
            # ─── action selection ─────────────────────────
            if hasattr(self.ln, "select_action"):
                # both ApproxQLearning and DQNLearner implement this
                a = self.ln.select_action(s, self.eps)
            else:
                # pure random fallback
                a = self.env.sample_move()

            # ─── environment step ─────────────────────────
            s2, r, done, _ = self.env.step(a)
            total += r

            # ─── learning step ───────────────────────────
            if hasattr(self.ln, "store_transition"):
                # DQN path
                self.ln.store_transition(s, a, r, s2, done)
                self.ln.learn()
            else:
                # ApproxQLearning path
                self.ln.update(s, a, r, s2, None, done)

            s = s2

        # decay ε
        self.eps = max(self.eps_min, self.eps * self.decay)
        return total