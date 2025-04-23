import abc
import struct
import typing
from sys import stderr
from board import board

# ──────────────────────────── helpers ────────────────────────────────

def info(msg: str):
    print(msg, file=stderr)

def error(msg: str):
    print(msg, file=stderr)


# ──────────────────────────── base table ─────────────────────────────

class feature(abc.ABC):
    """Base class for n‑tuple feature tables (weight look‑ups)."""

    def __init__(self, length: int):
        self.weight = feature.alloc(length)

    # --- list‑like helpers -------------------------------------------------
    def __getitem__(self, i: int) -> float:
        return self.weight[i]

    def __setitem__(self, i: int, v: float) -> None:
        self.weight[i] = v

    def __len__(self) -> int:
        return len(self.weight)

    def size(self) -> int:
        return len(self.weight)

    # --- abstract interface ----------------------------------------------
    @abc.abstractmethod
    def estimate(self, b: board) -> float:  # value‑function estimate
        ...

    @abc.abstractmethod
    def update(self, b: board, u: float) -> float:  # gradient / TD‑step
        ...

    @abc.abstractmethod
    def name(self) -> str:  # printable id
        ...

    # --- (de)serialisation -------------------------------------------------
    def dump(self, b: board, out: typing.Callable = info) -> None:
        out(f"{b}\nestimate = {self.estimate(b)}")

    def write(self, output: typing.BinaryIO) -> None:
        name = self.name().encode("utf-8")
        output.write(struct.pack("I", len(name)))
        output.write(name)
        size = len(self.weight)
        output.write(struct.pack("Q", size))
        output.write(struct.pack(f"{size}f", *self.weight))

    def read(self, input: typing.BinaryIO) -> None:
        size = struct.unpack("I", input.read(4))[0]
        name = input.read(size).decode("utf-8")
        if name != self.name():
            error(f"unexpected feature: {name} (expected {self.name()})")
            exit(1)
        size = struct.unpack("Q", input.read(8))[0]
        if size != len(self.weight):
            error(f"unexpected feature size {size} for {self.name()} ({self.size()} expected)")
            exit(1)
        self.weight = list(struct.unpack(f"{size}f", input.read(size * 4)))
        if len(self.weight) != size:
            error("unexpected end of binary")
            exit(1)

    # --- memory guard ------------------------------------------------------
    @staticmethod
    def alloc(num: int) -> list[float]:
        """Allocate a weight table while enforcing a 1‑GiB global cap."""
        if not hasattr(feature.alloc, "total"):
            feature.alloc.total = 0
            feature.alloc.limit = (1 << 30) // 4  # 1 GiB worth of floats
        feature.alloc.total += num
        if feature.alloc.total > feature.alloc.limit:
            error("memory limit exceeded" + str(feature.alloc.total))
            exit(1)
        return [0.0] * num


# ──────────────────────────── n‑tuple feature ─────────────────────────

class pattern(feature):
    """N‑tuple pattern feature with optional rotational / reflection symmetry.

    Args:
        patt: list of tile indices (0–15) defining the tuple order.
        iso:  number of isomorphic transformations:
              * 1 = none
              * 4 = rotations
              * 8 = rotations + mirror (default)
    """

    def __init__(self, patt: list[int], iso: int = 8):
        if not patt:
            error("pattern cannot be empty")
            exit(1)
        if iso not in (1, 4, 8):
            error("iso must be 1, 4, or 8")
            exit(1)

        super().__init__(1 << (len(patt) * 4))  # dense table size: 16^|patt|

        # Build all unique isomorphic variants of the index pattern.
        self.isom: list[list[int]] = []
        canonical = board(0xFEDCBA9876543210)  # index → position mapping board

        # rotations ---------------------------------------------------------
        for r in range(4):
            b = canonical.clone()
            b.rotate(r)
            self.isom.append([b.at(t) for t in patt])
            if iso == 1:
                break  # no more transforms requested
        # mirror ------------------------------------------------------------
        if iso == 8:
            mir = canonical.clone(); mir.mirror()
            for r in range(4):
                b = mir.clone(); b.rotate(r)
                self.isom.append([b.at(t) for t in patt])

    # ----------------------------------------------------------------------
    #  value lookup / TD updates
    # ----------------------------------------------------------------------
    def estimate(self, b: board) -> float:
        return sum(self.weight[self._index_of(iso, b)] for iso in self.isom)

    def update(self, b: board, u: float) -> float:
        adjust = u / len(self.isom)
        val = 0.0
        for iso in self.isom:
            idx = self._index_of(iso, b)
            self.weight[idx] += adjust
            val += self.weight[idx]
        return val

    # ----------------------------------------------------------------------
    #  misc helpers
    # ----------------------------------------------------------------------
    def name(self) -> str:
        return f"{len(self.isom[0])}-tuple pattern {self._name_of(self.isom[0])}"

    # low‑level index helpers ----------------------------------------------
    @staticmethod
    def _index_of(patt: list[int], b: board) -> int:
        idx = 0
        for i, pos in enumerate(patt):
            idx |= b.at(pos) << (4 * i)
        return idx

    @staticmethod
    def _name_of(patt: list[int]) -> str:
        return "".join(f"{p:x}" for p in patt)

    # pretty printer -------------------------------------------------------
    def dump(self, b: board, out: typing.Callable = info) -> None:
        for iso in self.isom:
            idx = self._index_of(iso, b)
            tiles = [(idx >> (4 * i)) & 0x0F for i in range(len(iso))]
            out(f"#{self._name_of(iso)}[{self._name_of(tiles)}] = {self.weight[idx]}")