import random
from typing import List, Tuple

__all__ = ["board"]

class board:
    """64‑bit bit‑board implementation for the 2048 game.

    Layout (index order):
        0  1  2  3
        4  5  6  7
        8  9 10 11
       12 13 14 15

    Internally the 16 four‑bit tiles are packed into a 64‑bit integer (`raw`) in
    little‑endian order.  The value stored in each nibble is *log₂(tileValue)*.
    A zero nibble therefore represents an empty cell.
    """

    # ───────────────────────── initializer ────────────────────────────

    def __init__(self, raw: int = 0):
        self.raw: int = int(raw)

    # ────────────────────────── basic utils ───────────────────────────

    def __int__(self) -> int:
        return self.raw

    # ----- row & tile helpers -------------------------------------------------

    def fetch(self, i: int) -> int:
        """Return *row i* as a 16‑bit integer (little‑endian)."""
        return (self.raw >> (i << 4)) & 0xFFFF

    def place(self, i: int, r: int) -> None:
        """Overwrite *row i* with the 16‑bit value *r*."""
        self.raw = (self.raw & ~(0xFFFF << (i << 4))) | ((r & 0xFFFF) << (i << 4))

    def at(self, i: int) -> int:
        """Return log₂ of the tile at board index *i* (0‑15)."""
        return (self.raw >> (i << 2)) & 0x0F

    def set(self, i: int, t: int) -> None:
        """Set board index *i* to log₂ value *t*.  (0 clears the tile)."""
        self.raw = (self.raw & ~(0x0F << (i << 2))) | ((t & 0x0F) << (i << 2))

    # ----- container dunder ---------------------------------------------------

    def __getitem__(self, i: int) -> int:
        return self.at(i)

    def __setitem__(self, i: int, t: int) -> None:
        self.set(i, t)

    # ----- comparisons --------------------------------------------------------

    def __eq__(self, other) -> bool:
        return isinstance(other, board) and self.raw == other.raw

    def __lt__(self, other) -> bool:
        return isinstance(other, board) and self.raw < other.raw

    def __hash__(self) -> int:
        return hash(self.raw)

    # ──────────────────────── lookup structure ─────────────────────────

    class lookup:
        """Static slide/merge lookup for each possible 16‑bit row."""

        find: List["board.lookup.entry"] = [None] * 65536  # type: ignore

        class entry:
            def __init__(self, row: int):
                V = [
                    (row >> 0) & 0x0F,
                    (row >> 4) & 0x0F,
                    (row >> 8) & 0x0F,
                    (row >> 12) & 0x0F,
                ]
                L, sc_l = board.lookup.entry._slide_left(V)
                V.reverse()            # mirror for right move
                R, sc_r = board.lookup.entry._slide_left(V)
                R.reverse()
                self.raw: int = row
                self.left: int = (L[0] << 0) | (L[1] << 4) | (L[2] << 8) | (L[3] << 12)
                self.right: int = (R[0] << 0) | (R[1] << 4) | (R[2] << 8) | (R[3] << 12)
                self.score: int = sc_l  # sc_l == sc_r

            # fast apply helpers ------------------------------------------------
            def move_left(self, raw: int, sc: int, i: int) -> Tuple[int, int]:
                return raw | (self.left << (i << 4)), sc + self.score

            def move_right(self, raw: int, sc: int, i: int) -> Tuple[int, int]:
                return raw | (self.right << (i << 4)), sc + self.score

            # internal slide algorithm -----------------------------------------
            @staticmethod
            def _slide_left(row: List[int]) -> Tuple[List[int], int]:
                buf = [t for t in row if t]
                res: List[int] = []
                score = 0
                while buf:
                    if len(buf) >= 2 and buf[0] == buf[1]:
                        buf[1] += 1  # merge
                        score += 1 << buf[1]
                        buf = buf[1:]
                    res.append(buf[0])
                    buf = buf[1:]
                return res + [0] * (4 - len(res)), score

        # build LUT -------------------------------------------------------------
        @classmethod
        def init(cls) -> None:
            if cls.find[0] is not None:  # already built
                return
            cls.find = [cls.entry(row) for row in range(65536)]

    # build lookup at import‑time for plug‑and‑play -----------------------------

    # ───────────────────── high‑level board ops ────────────────────────
    # -- random helpers -------------------------------------------------

    def init(self) -> None:
        """Reset to the game’s initial state (two random tiles)."""
        self.raw = 0
        self.popup()
        self.popup()

    def reset(self):  # env convenience alias
        self.init()

    def popup(self) -> None:
        """Spawn a 2‑tile (90 %) or 4‑tile (10 %) at a random empty cell."""
        empty = [i for i in range(16) if self.at(i) == 0]
        if empty:
            self.set(random.choice(empty), 1 if random.random() < 0.9 else 2)

    # -- move dispatcher ------------------------------------------------

    def move(self, op: int) -> int:
        return (
            self.move_up()    if op == 0 else
            self.move_right() if op == 1 else
            self.move_down()  if op == 2 else
            self.move_left()  if op == 3 else
            -1
        )

    # -- primitive moves ------------------------------------------------

    def move_left(self) -> int:
        move = 0
        prev = self.raw
        score = 0
        for i in range(4):
            move, score = self.lookup.find[self.fetch(i)].move_left(move, score, i)  # type: ignore[index]
        self.raw = move
        return score if move != prev else -1

    def move_right(self) -> int:
        move = 0
        prev = self.raw
        score = 0
        for i in range(4):
            move, score = self.lookup.find[self.fetch(i)].move_right(move, score, i)  # type: ignore[index]
        self.raw = move
        return score if move != prev else -1

    def move_up(self) -> int:
        self.rotate_clockwise()
        score = self.move_right()
        self.rotate_counterclockwise()
        return score

    def move_down(self) -> int:
        self.rotate_clockwise()
        score = self.move_left()
        self.rotate_counterclockwise()
        return score

    # ─────────────────── board transforms (bit‑twiddling) ──────────────

    def transpose(self) -> None:
        self.raw = (
            (self.raw & 0xF0F00F0FF0F00F0F)
            | ((self.raw & 0x0000F0F00000F0F0) << 12)
            | ((self.raw & 0x0F0F00000F0F0000) >> 12)
        )
        self.raw = (
            (self.raw & 0xFF00FF0000FF00FF)
            | ((self.raw & 0x00000000FF00FF00) << 24)
            | ((self.raw & 0x00FF00FF00000000) >> 24)
        )

    def mirror(self) -> None:
        self.raw = (
            ((self.raw & 0x000F000F000F000F) << 12)
            | ((self.raw & 0x00F000F000F000F0) << 4)
            | ((self.raw & 0x0F000F000F000F00) >> 4)
            | ((self.raw & 0xF000F000F000F000) >> 12)
        )

    def flip(self) -> None:
        self.raw = (
            ((self.raw & 0x000000000000FFFF) << 48)
            | ((self.raw & 0x00000000FFFF0000) << 16)
            | ((self.raw & 0x0000FFFF00000000) >> 16)
            | ((self.raw & 0xFFFF000000000000) >> 48)
        )

    # composite rotations ------------------------------------------------

    def rotate(self, r: int = 1) -> None:
        r = ((r % 4) + 4) % 4
        if r == 1:
            self.rotate_clockwise()
        elif r == 2:
            self.reverse()
        elif r == 3:
            self.rotate_counterclockwise()

    def rotate_clockwise(self) -> None:
        self.transpose(); self.mirror()

    def rotate_counterclockwise(self) -> None:
        self.transpose(); self.flip()

    def reverse(self) -> None:
        self.mirror(); self.flip()

    # ─────────────────────── convenience helpers ──────────────────────

    def clone(self) -> "board":
        return board(self.raw)

    def can_move(self) -> bool:
        return any(self.clone().move(op) != -1 for op in range(4))

    # ──────────────────────── pretty‑print UI ─────────────────────────

    def __str__(self) -> str:
        out = ["+" + "-" * 24 + "+"]
        for r in range(4):
            row = "|" + "".join(f"{(1 << self.at(r * 4 + c)) & -2:6d}" for c in range(4)) + "|"
            out.append(row)
        out.append("+" + "-" * 24 + "+")
        return "\n".join(out)
try:
    board.lookup.init()  # type: ignore
except Exception:
    pass