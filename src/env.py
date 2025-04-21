import random
from board import board
from gui_render import BoardView

def _ascii_render(raw_value: int) -> None: #console debugging
    tiles = [(raw_value >> (4 * i)) & 0xF for i in range(16)]
    for row in range(4):
        print(" ".join(f"{(1 << t) if t else 0:>6}" for t in tiles[4*row:4*(row+1)]))
    print("-" * 30)

class Game2048Env:
    ACTIONS = (0, 1, 2, 3)               # 0=UP 1=RIGHT 2=DOWN 3=LEFT

    def __init__(
        self,
        seed: int | None = None,
        ascii_render: bool = False,
        gui: bool = False
    ):
        if seed is not None:
            random.seed(seed)

        self.b = board()                      # your 2048 bitboard
        self.num_moves = len(self.ACTIONS)

        # Rendering switches
        self._ascii = ascii_render
        self._gui_view = BoardView() if gui else None

    def _maybe_render(self):
        if self._ascii:
            _ascii_render(self.b.raw)
        if self._gui_view:
            self._gui_view.draw(self.b.raw)

    def reset(self):
        self.b.reset()
        self._maybe_render()
        return self.b.raw

    def step(self, action: int) -> tuple[int, int, bool, dict]:
        reward = self.b.move(action)
        illegal = (reward == -1)
        if illegal:
            reward = 0
        else:
            self.b.popup()     # only add a tile on valid moves
        done = not self.b.can_move()
        self._maybe_render()
        return self.b.raw, reward, done, {"illegal": illegal}

    def sample_move(self) -> int:
        return random.choice(self.ACTIONS)

    def render(self):
        self._maybe_render()