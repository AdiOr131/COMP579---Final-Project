import tkinter as tk

# tile colours (log2 value → bg, fg)
COLORS = {
    0:  ("#cdc1b4", "#776e65"),
    1:  ("#eee4da", "#776e65"),      # 2
    2:  ("#ede0c8", "#776e65"),      # 4
    3:  ("#f2b179", "#f9f6f2"),      # 8
    4:  ("#f59563", "#f9f6f2"),      # 16
    5:  ("#f67c5f", "#f9f6f2"),      # 32
    6:  ("#f65e3b", "#f9f6f2"),      # 64
    7:  ("#edcf72", "#f9f6f2"),      # 128
    8:  ("#edcc61", "#f9f6f2"),      # 256
    9:  ("#edc850", "#f9f6f2"),      # 512
    10: ("#edc53f", "#f9f6f2"),      # 1024
    11: ("#edc22e", "#f9f6f2"),      # 2048
}

class BoardView:
    def __init__(self, size=4, tile_px=100):
        self.size, self.tile_px = size, tile_px
        self.window = tk.Tk()
        self.window.title("2048 RL")
        self.canvas = tk.Canvas(
            self.window,
            width=size * tile_px,
            height=size * tile_px,
            bg="#bbada0",
            highlightthickness=0,
        )
        self.canvas.pack()
        self.tiles = [
            self.canvas.create_rectangle(
                c*tile_px, r*tile_px, (c+1)*tile_px, (r+1)*tile_px,
                outline="#bbada0", width=5
            )
            for r in range(size) for c in range(size)
        ]
        self.labels = [
            self.canvas.create_text(
                c*tile_px + tile_px/2,
                r*tile_px + tile_px/2,
                font=("Clear Sans", 32, "bold")
            )
            for r in range(size) for c in range(size)
        ]

    def draw(self, raw_value: int):
        for idx in range(16):
            v = (raw_value >> (4*idx)) & 0xF
            bg, fg = COLORS.get(v, COLORS[11])
            self.canvas.itemconfig(self.tiles[idx], fill=bg)
            self.canvas.itemconfig(self.labels[idx], text="" if v == 0 else str(1 << v), fill=fg)
        self.window.update_idletasks()
        self.window.update()