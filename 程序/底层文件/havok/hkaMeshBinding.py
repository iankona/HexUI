


from .havokbinary import readhavokint

def hkaMeshBinding(bp, flag, havokfile):
    if flag == 46: return hkaMeshBinding46(bp, havokfile) # [92]
    if flag == 47: return hkaMeshBinding47(bp, havokfile) # [94]

class hkaMeshBinding46:
    def __init__(self, bp, havokfile):
        self.originalSkeletonName = havokfile.readname(bp)
        self.name = havokfile.readname(bp)
        self.skeleton = readhavokint(bp) - 1
        numboneFromSkinMeshTransform = readhavokint(bp)
        self.boneFromSkinMeshTransforms = [ [bp.readfloat32(4), bp.readfloat32(4), bp.readfloat32(4), bp.readfloat32(4)] for i in range(numboneFromSkinMeshTransform) ]

class hkaMeshBinding47:
    def __init__(self, bp, havokfile):
        self.mesh = readhavokint(bp) - 1
        self.originalSkeletonName = havokfile.readname(bp)
        self.name = havokfile.readname(bp)
        self.skeleton = readhavokint(bp) - 1
        numboneFromSkinMeshTransform = readhavokint(bp)
        self.boneFromSkinMeshTransforms = [ [bp.readfloat32(4), bp.readfloat32(4), bp.readfloat32(4), bp.readfloat32(4)] for i in range(numboneFromSkinMeshTransform) ]
