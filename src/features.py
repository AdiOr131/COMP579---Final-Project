import abc, struct, typing
from board import board
from sys import stderr

def info(msg: str):
    print(msg, file=stderr)
def error(msg: str):
    print(msg, file=stderr)

class feature(abc.ABC):
    """feature and weight table for n-tuple networks"""

    def __init__(self, length: int):
        self.weight = feature.alloc(length)

    def __getitem__(self, i: int) -> float:
        return self.weight[i]
    def __setitem__(self, i: int, v: float) -> None:
        self.weight[i] = v
    def __len__(self) -> int:
        return len(self.weight)
    def size(self) -> int:
        return len(self.weight)

    @abc.abstractmethod
    def estimate(self, b: board) -> float:
        pass

    @abc.abstractmethod
    def update(self, b: board, u: float) -> float:
        pass

    @abc.abstractmethod
    def name(self) -> str:
        pass

    def dump(self, b: board, out: typing.Callable = info) -> None:
        out(f"{b}\nestimate = {self.estimate(b)}")

    def write(self, output: typing.BinaryIO) -> None:
        name = self.name().encode('utf-8')
        output.write(struct.pack('I', len(name))); output.write(name)
        size = len(self.weight)
        output.write(struct.pack('Q', size))
        output.write(struct.pack(f'{size}f', *self.weight))

    def read(self, input: typing.BinaryIO) -> None:
        size = struct.unpack('I', input.read(4))[0]
        name = input.read(size).decode('utf-8')
        if name != self.name(): error(f'unexpected feature: {name}'); exit(1)
        size2= struct.unpack('Q', input.read(8))[0]
        if size2!=len(self.weight): error(f'unexpected size {size2}'); exit(1)
        self.weight = list(struct.unpack(f'{size2}f', input.read(size2*4)))

    @staticmethod
    def alloc(num: int) -> list[float]:
        if not hasattr(feature.alloc, "total"):
            feature.alloc.total = 0
            feature.alloc.limit = (1<<30)//4
        feature.alloc.total += num
        if feature.alloc.total > feature.alloc.limit:
            error("memory limit exceeded"); exit(1)
        return [0.0]*num


class pattern(feature):
    """
    n-tuple pattern feature, with isomorphic rotations/reflections
    """
    def __init__(self, patt: list[int], iso: int = 8):
        super().__init__(1 << (len(patt) * 4))
        if not patt:
            error("no pattern defined"); exit(1)

        self.isom = [None]*iso
        for i in range(iso):
            idx = board(0xfedcba9876543210)
            if i>=4: idx.mirror()
            idx.rotate(i)
            self.isom[i] = [idx.at(t) for t in patt]

    def estimate(self, b: board) -> float:
        v = 0.0
        for iso in self.isom:
            idx = self.indexof(iso, b)
            v += self.weight[idx]
        return v

    def update(self, b: board, u: float) -> float:
        adj = u/len(self.isom)
        v = 0.0
        for iso in self.isom:
            idx = self.indexof(iso, b)
            self.weight[idx] += adj
            v += self.weight[idx]
        return v

    def name(self) -> str:
        return f"{len(self.isom[0])}-tuple"

    def indexof(self, patt: list[int], b: board) -> int:
        idx = 0
        for i,pos in enumerate(patt):
            idx |= b.at(pos) << (4*i)
        return idx

    def nameof(self, patt: list[int]) -> str:
        return "".join(f"{p:x}" for p in patt)

    def dump(self, b: board, out: typing.Callable = info) -> None:
        for iso in self.isom:
            idx = self.indexof(iso, b)
            tiles = [(idx >> (4*i)) & 0x0f for i in range(len(iso))]
            out(f"#{self.nameof(iso)}[{self.nameof(tiles)}]={self.weight[idx]}")