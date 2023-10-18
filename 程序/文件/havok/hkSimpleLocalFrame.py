from .havokBinary import readhavokint


def hkSimpleLocalFrame(bp, flag, hkafile):
    if flag == 3: return hkSimpleLocalFrame3(bp, hkafile)
    if flag == 17: return hkSimpleLocalFrame17(bp, hkafile)
    if flag == 21: return hkSimpleLocalFrame21(bp, hkafile)


class hkSimpleLocalFrame17:
    def __init__(self, bp, hkafile):
        self.transform = [bp.readfloat32(4), bp.readfloat32(4), bp.readfloat32(4), bp.readfloat32(4)]
        self.name = hkafile.readname(bp)


class hkSimpleLocalFrame3:
    def __init__(self, bp, hkafile):
        self.transform = [bp.readfloat32(4), bp.readfloat32(4), bp.readfloat32(4), bp.readfloat32(4)]
        self.children = [ readhavokint(bp)-1 for i in range(readhavokint(bp))]


class hkSimpleLocalFrame21:
    def __init__(self, bp, hkafile):
        self.transform = [bp.readfloat32(4), bp.readfloat32(4), bp.readfloat32(4), bp.readfloat32(4)]
        self.parentFrame = readhavokint(bp)-1
        self.name = hkafile.readname(bp)