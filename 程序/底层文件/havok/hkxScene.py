from .havokbinary import readhavokint

def hkxScene(bp, flag, havokfile):
    if flag == 32798: return hkxScene32798(bp, havokfile) # 
    if flag == 46646: return hkxScene46646(bp, havokfile) # [54,182]


class hkxScene32798:
    def __init__(self, bp, havokfile):
        self.modeller = havokfile.readname(bp)
        self.asset = havokfile.readname(bp)
        self.sceneLength = bp.readfloat32()
        self.numFrames = readhavokint(bp)
        self.appliedTransform = [bp.readfloat32(4), bp.readfloat32(4), bp.readfloat32(4)]


class hkxScene46646:
    def __init__(self, bp, havokfile):
        self.modeller = havokfile.readname(bp)
        self.sceneLength = bp.readuint8()
        self.numFrames = readhavokint(bp)
        self.rootNode = readhavokint(bp) - 1
        num_mesh = readhavokint(bp)
        self.meshs = [readhavokint(bp)-1 for i in range(num_mesh)]
        num_material = readhavokint(bp)
        self.materials = [readhavokint(bp)-1 for i in range(num_material)]
        num_externalTexture = readhavokint(bp)
        self.externalTextures = [readhavokint(bp)-1 for i in range(num_externalTexture)]
        num_skinBinding = readhavokint(bp)
        self.skinBindings = [readhavokint(bp)-1 for i in range(num_skinBinding)]
        self.appliedTransform = [bp.readfloat32(4), bp.readfloat32(4), bp.readfloat32(4)]