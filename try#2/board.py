import random

class Board:
    """64-bit little-endian bit-board for 2048."""

    def __init__(self, raw: int = 0):
        self.raw = raw

    # ─── 4-bit tile helpers ────────────────────────────────────────────
    def at(self, idx: int) -> int:
        return (self.raw >> (idx << 2)) & 0xF          # log2 value (0 = empty)

    def set(self, idx: int, val: int) -> None:
        self.raw = (self.raw & ~(0xF << (idx << 2))) | (val << (idx << 2))

    # ─── look-up table for row slides (left / right) ───────────────────
    class LUT:
        LEFT  = [0]*65536
        RIGHT = [0]*65536
        SCORE = [0]*65536   # reward for merges

    @staticmethod
    def _init_LUT():
        def slide(row):                       # row = list of four 4-bit tiles
            out, score = [], 0
            buf = [t for t in row if t]
            while buf:
                t = buf.pop(0)
                if buf and t == buf[0]:
                    t += 1; buf.pop(0)
                    score += 1 << t
                out.append(t)
            return out + [0]*(4-len(out)), score

        for r in range(65536):
            row = [(r >> (i*4)) & 0xF for i in range(4)]
            left , sc = slide(row)
            right, _  = slide(row[::-1]); right = right[::-1]
            encode = lambda t: sum(t[i] << (i*4) for i in range(4))
            Board.LUT.LEFT [r] = encode(left)
            Board.LUT.RIGHT[r] = encode(right)
            Board.LUT.SCORE[r] = sc
    _init_LUT = staticmethod(_init_LUT)

    # ─── rotate / mirror helpers (needed for up/down) ──────────────────
    def transpose(self):             # bit-magic swap rows <-> columns
        self.raw = (self.raw & 0xF0F00F0FF0F00F0F)     \
                  |((self.raw & 0x0000F0F00000F0F0)<<12)\
                  |((self.raw & 0x0F0F00000F0F0000)>>12)
        self.raw = (self.raw & 0xFF00FF0000FF00FF)     \
                  |((self.raw & 0x00FF00FF00000000)>>24)\
                  |((self.raw & 0x00000000FF00FF00)<<24)

    def mirror(self):                 # horizontal reflection
        self.raw = ((self.raw & 0x000F000F000F000F)<<12)\
                  |((self.raw & 0x00F000F000F000F0)<<4) \
                  |((self.raw & 0x0F000F000F000F00)>>4) \
                  |((self.raw & 0xF000F000F000F000)>>12)

    # ─── single-row slide utilities ───────────────────────────────────
    def _row(self, i): return (self.raw >> (i*16)) & 0xFFFF
    def _set_row(self, i, r): self.raw = (self.raw & ~(0xFFFF << (i*16))) | (r << (i*16))

    # ─── moves: 0=UP 1=RIGHT 2=DOWN 3=LEFT ────────────────────────────
    def move(self, op: int) -> int:
        if op == 1: return self._h_slide(Board.LUT.RIGHT)
        if op == 3: return self._h_slide(Board.LUT.LEFT )
        self.transpose()
        sc = self._h_slide(Board.LUT.RIGHT if op == 0 else Board.LUT.LEFT)
        self.transpose()
        return sc

    def _h_slide(self, lut):
        prev, score = self.raw, 0
        for i in range(4):
            r = self._row(i)
            score += Board.LUT.SCORE[r]
            self._set_row(i, lut[r])
        return score if self.raw != prev else -1

    # ─── misc helpers ─────────────────────────────────────────────────
    def popup(self):
        empty = [i for i in range(16) if self.at(i) == 0]
        if empty:
            self.set(random.choice(empty), 1 if random.random()<0.9 else 2)

    def reset(self):
        self.raw = 0; self.popup(); self.popup()

    def clone(self): return Board(self.raw)
Board._init_LUT()   # build lookup table once