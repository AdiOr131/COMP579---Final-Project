import random
from board import Board

class Game2048Env:
    ACTIONS = (0,1,2,3)              # up, right, down, left

    def __init__(self, seed=None):
        if seed is not None: random.seed(seed)
        self.b = Board()

    # gym-like api ------------------------------------------------------
    def reset(self):
        self.b.reset(); return self.b.raw          # state = 64-bit int

    def step(self, a):
        r = self.b.move(a)
        illegal = (r == -1)
        if illegal:
            r = 0                      # punish only by lost turn
        else:
            self.b.popup()
        done = not self.can_move()
        return self.b.raw, r, done, {"illegal":illegal}

    def sample_move(self): return random.choice(self.ACTIONS)
    def can_move(self):
        return any(self.b.clone().move(a)!=-1 for a in self.ACTIONS)