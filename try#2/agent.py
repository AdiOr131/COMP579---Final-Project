class Agent:
    def __init__(self, env, learner, eps=1e-2, decay=0.999, eps_min=1e-3):
        self.env, self.ln = env, learner
        self.eps, self.decay, self.eps_min = eps, decay, eps_min

    def run_episode(self):
        s = self.env.reset()
        done, total = False, 0
        while not done:
            a = self.ln.select_action(s, self.eps)
            s2, r, done, _ = self.env.step(a)
            self.ln.update(s, a, r, s2, done)
            s = s2; total += r
        self.eps = max(self.eps_min, self.eps * self.decay)
        return total