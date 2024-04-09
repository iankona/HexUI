
from .havokbinary import readhavokint

def hkxSkinBinding(bp, flag, hkafile):
    if flag == 15: return hkxSkinBinding15(bp, hkafile)


class hkxSkinBinding15:
    def __init__(self, bp, hkafile):
        self.mesh = readhavokint(bp) -1
        self.nodeNames = [hkafile.readname(bp) for i in range(readhavokint(bp))]
        self.bindPose = [[bp.readfloat32(4),  bp.readfloat32(4),  bp.readfloat32(4), bp.readfloat32(4)] for i in range(readhavokint(bp))]
        self.initSkinTransform = [bp.readfloat32(4),  bp.readfloat32(4),  bp.readfloat32(4), bp.readfloat32(4)]



