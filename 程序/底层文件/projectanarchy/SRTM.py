from .LRTM import LRTM

class SRTM:
    def __init__(self, bp):
        self.材质列表 = []
        self.read_block_data(bp)


    def read_block_data(self, bp):
        for i in range(bp.readuint32()):
            leftflag, leftchar, bn, rightflag, rightchar = bp.readuint32(), bp.readchar(4), bp.readslice(bp.readuint32()), bp.readuint32(), bp.readchar(4)
            self.材质列表.append( LRTM(bn) )

