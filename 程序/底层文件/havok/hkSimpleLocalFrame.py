from .havokbinary import readhavokint


def hkSimpleLocalFrame(bp, flag, havokfile):
    if flag == 3: return hkSimpleLocalFrame3(bp, havokfile)
    if flag == 17: return hkSimpleLocalFrame17(bp, havokfile)
    if flag == 21: return hkSimpleLocalFrame21(bp, havokfile)


class hkSimpleLocalFrame17:
    def __init__(self, bp, havokfile):
        self.transform = [bp.readfloat32(4), bp.readfloat32(4), bp.readfloat32(4), bp.readfloat32(4)]
        self.name = havokfile.readname(bp)


class hkSimpleLocalFrame3:
    def __init__(self, bp, havokfile):
        self.transform = [bp.readfloat32(4), bp.readfloat32(4), bp.readfloat32(4), bp.readfloat32(4)]
        self.children = [ readhavokint(bp)-1 for i in range(readhavokint(bp))]


class hkSimpleLocalFrame21:
    def __init__(self, bp, havokfile):
        self.transform = [bp.readfloat32(4), bp.readfloat32(4), bp.readfloat32(4), bp.readfloat32(4)]
        self.parentFrame = readhavokint(bp)-1
        self.name = havokfile.readname(bp)