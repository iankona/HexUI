


class Object:pass
class anarchyfile:
    def __init__(self, bp, anarchyclasses):
        self.filepath = ""
        self.head = Object()
        self.blocks = {}
        self.read_blocks(bp)

        self.read_head(bp)
        self.read_data(anarchyclasses)


    def read_blocks(self, bp):
        self.filepath = bp.rdpath

        self.blocks["b_head"] = [0, "head", 8, bp.readslice(8), 0, "head"]
        while True:
            if bp.remainsize() < 12: break
            [
            leftflag, leftchar, size, 块, rightflag, rightchar
            ] = bp.readuint32(), bp.readchar(4), bp.readuint32seek0(), bp.readslice(bp.readuint32()), bp.readuint32(), bp.readchar(4)

            self.blocks[f"b_{leftchar}"] = [leftflag, leftchar, size, 块, rightflag, rightchar]

        self.blocks["b_endof"] = [0, "endof", bp.remainsize(), bp.readremainslice(), 0, "endof"]


    def read_head(self, bp):
        a, b, c, bp, e, f = self.blocks["b_head"]
        self.head.magic = bp.readhex(4)


    def read_data(self, anarchyclasses):
        for leftflag, leftchar, size, 块, rightflag, rightchar in self.blocks.values():
            if leftchar in anarchyclasses: setattr(self, leftchar, anarchyclasses[leftchar](块))





