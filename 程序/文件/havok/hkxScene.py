from .havokBinary import readhavokint

def hkxScene(bp, flag, hkafile):
    if flag == 32798: return hkxScene32798(bp, hkafile)


class hkxScene32798:
    def __init__(self, bp, hkafile):
        self.modeller = hkafile.readname(bp)
        self.asset = hkafile.readname(bp)
        self.sceneLength = bp.readfloat32()
        self.numFrames = readhavokint(bp)
        self.appliedTransform = [bp.readfloat32(4), bp.readfloat32(4), bp.readfloat32(4)]