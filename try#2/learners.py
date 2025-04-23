import random
from board import Board
from features import Pattern

class TD0Learner:
    """After-state TD(0) with n-tuple features."""
    def __init__(self, tuples, alpha=0.01, gamma=0.99):
        self.alpha, self.gamma = alpha, gamma
        self.feats = [Pattern(t) for t in tuples]

    # ------------- helper ---------------------------------------------
    def _V(self, board):                   # value of an after-state
        return sum(f.value(board) for f in self.feats)

    # ------------- ε-greedy policy ------------------------------------
    def select_action(self, s, eps):
        if random.random() < eps: return random.randrange(4)
        best_a, best_q = 0, -1e9
        for a in range(4):
            b = Board(s).clone()
            r = b.move(a)
            if r == -1: continue
            q = r + self.gamma * self._V(b)
            if q > best_q: best_q, best_a = q, a
        return best_a

    # ------------- TD(0) update ---------------------------------------
    def update(self, s, a, r, s_next, done):
        # current after-state
        after  = Board(s).clone()      # copy the original board
        if after.move(a) == -1:        # illegal move? abort update
            return
        v0 = self._V(after)            # current value estimate V(s,a)

        # 2) Compute max_a′ Q(s′,a′) for the *next* true state s_next
        if done:
            v_next = 0.0               # terminal
        else:
            q_vals = []
            for a2 in range(4):
                b1 = Board(s_next).clone()  # start from the post-popup board
                r1 = b1.move(a2)            # slide direction a2
                if r1 == -1:                # illegal ⇒ skip
                    continue
                q_vals.append(r1 + self.gamma * self._V(b1))
            v_next = max(q_vals) if q_vals else 0.0

        # 3) TD(0) target and error
        delta   = r + self.gamma * v_next - v0

        # 4) Update every active LUT (equal share)
        adj = self.alpha * delta / len(self.feats)
        for f in self.feats:
            f.update(after, adj)

# ------------ convenience factory ------------------------------------
DEFAULT_TUPLES = [
    [0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15],   # rows
    [0,4,8,12], [1,5,9,13], [2,6,10,14], [3,7,11,15]    # columns
]
def make_learner(alpha=0.01, gamma=0.99):
    return TD0Learner(DEFAULT_TUPLES, alpha, gamma)