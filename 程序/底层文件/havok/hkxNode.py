from .havokbinary import readhavokint


def hkxNode(bp, flag, havokfile):
    if flag == 52: return hkxNode52(bp, havokfile) # [52,0]
    if flag == 276: return hkxNode276(bp, havokfile) # [20,1]
    if flag == 284: return hkxNode284(bp, havokfile) # [28,1]
    if flag == 308: return hkxNode308(bp, havokfile) # [52,1]
    if flag == 1300: return hkxNode1300(bp, havokfile) # [20,5]
    if flag == 1332: return hkxNode1332(bp, havokfile) # [52,5]

class hkxNode52:
    def __init__(self, bp, havokfile):
        self.name = havokfile.readname(bp)
        num_keyFrame = readhavokint(bp)
        self.keyFrames = [[bp.readfloat32(4), bp.readfloat32(4), bp.readfloat32(4), bp.readfloat32(4)] for i in range(num_keyFrame)]
        num_child = readhavokint(bp)
        self.children = [ readhavokint(bp)-1 for i in range(num_child)]


class hkxNode276:
    def __init__(self, bp, havokfile):
        self.name = havokfile.readname(bp)
        num_keyFrame = readhavokint(bp)
        self.keyFrames = [[bp.readfloat32(4), bp.readfloat32(4), bp.readfloat32(4), bp.readfloat32(4)] for i in range(num_keyFrame)]
        self.bone = bp.readuint8()


class hkxNode284:
    def __init__(self, bp, havokfile):
        self.name = havokfile.readname(bp)
        self.object = readhavokint(bp) - 1
        num_keyFrame = readhavokint(bp)
        self.keyFrames = [[bp.readfloat32(4), bp.readfloat32(4), bp.readfloat32(4), bp.readfloat32(4)] for i in range(num_keyFrame)]
        self.bone = bp.readuint8()


class hkxNode308:
    def __init__(self, bp, havokfile):
        self.name = havokfile.readname(bp)
        num_keyFrame = readhavokint(bp)
        self.keyFrames = [[bp.readfloat32(4), bp.readfloat32(4), bp.readfloat32(4), bp.readfloat32(4)] for i in range(num_keyFrame)]
        num_child = readhavokint(bp)
        self.children = [ readhavokint(bp)-1 for i in range(num_child)]
        self.bone = bp.readuint8()


class hkxNode1300:
    def __init__(self, bp, havokfile):
        self.name = havokfile.readname(bp)
        num_keyFrame = readhavokint(bp)
        self.keyFrames = [[bp.readfloat32(4), bp.readfloat32(4), bp.readfloat32(4), bp.readfloat32(4)] for i in range(num_keyFrame)]
        self.selected = bp.readuint8()
        self.bone = bp.readuint8()


class hkxNode1332:
    def __init__(self, bp, havokfile):
        self.name = havokfile.readname(bp)
        num_keyFrame = readhavokint(bp)
        self.keyFrames = [[bp.readfloat32(4), bp.readfloat32(4), bp.readfloat32(4), bp.readfloat32(4)] for i in range(num_keyFrame)]
        num_child = readhavokint(bp)
        self.children = [ readhavokint(bp)-1 for i in range(num_child)]
        self.selected = bp.readuint8()
        self.bone = bp.readuint8()
