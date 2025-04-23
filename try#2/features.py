import numpy as np
from board import Board

class Pattern:
    """One n-tuple feature with all 8 isomorphic rotations/reflections."""
    def __init__(self, coords, iso=8):
        self.len = len(coords)
        self.weight = np.zeros( 1 << (4*self.len), dtype=np.float32 )
        # build 8 equivalence-classes of coordinates
        self.isom = []
        for k in range(iso):
            idx = Board(0xFEDCBA9876543210)        # coordinate board
            if k >= 4: idx.mirror()
            idx.transpose(); idx.rotate = lambda *_:None   # cheap hack
            for _ in range(k%4): idx.transpose(); idx.mirror()
            self.isom.append([ idx.at(c) for c in coords ])

    def _index(self, patt, board):
        idx = 0
        for i,p in enumerate(patt):
            idx |= board.at(p) << (4*i)
        return idx

    def value(self, board):
        return sum(self.weight[self._index(p,board)] for p in self.isom)

    def update(self, board, delta):
        adj = delta / len(self.isom)
        for p in self.isom:
            i = self._index(p,board)
            self.weight[i] += adj