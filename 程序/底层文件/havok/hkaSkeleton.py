from .havokbinary import readhavokint


def hkaSkeleton(bp, flag, havokfile):

    if flag == 158: return hkaSkeleton158(bp, havokfile)
    if flag == 30: return hkaSkeleton30(bp, havokfile)


class hkaSkeleton158:
    def __init__(self, bp, havokfile):
        self.name = havokfile.readname(bp)

        parentIndiceNumber = readhavokint(bp)
        parentIndiceType, self.parentIndices = readhavokint(bp), [ readhavokint(bp) for i in range(parentIndiceNumber) ]

        boneNumber = readhavokint(bp)
        boneType   = readhavokint(bp) # hkaBone
        self.bonenames           = [ havokfile.readname(bp) for i in range(boneNumber) ]
        if bp.readuint8seek0() == 0 or bp.readuint8seek0() == 1:
            self.bonelockTranslation = bp.readuint8(boneNumber)
        else:
            self.bonelockTranslation = [ 0 for i in range(boneNumber) ]

        referencePoseNumber  = readhavokint(bp)
        self.referencePose   =  [ [bp.readfloat32(4), bp.readfloat32(4), bp.readfloat32(4)] for i in range(referencePoseNumber) ]


        localFrameNumber = readhavokint(bp)
        localFrameType, self.localFrames = readhavokint(bp), [ readhavokint(bp)-1 for i in range(localFrameNumber) ]
        partitionType,  self.partitions  = readhavokint(bp), [ readhavokint(bp)   for i in range(localFrameNumber) ]


class hkaSkeleton30:
    def __init__(self, bp, havokfile):
        self.name = havokfile.readname(bp)

        parentIndiceNumber = readhavokint(bp)
        parentIndiceType, self.parentIndices = readhavokint(bp), [ readhavokint(bp) for i in range(parentIndiceNumber) ]

        boneNumber = readhavokint(bp)
        boneType   = readhavokint(bp) # hkaBone
        self.bonenames = [ havokfile.readname(bp) for i in range(boneNumber) ]
        if bp.readuint8seek0() == 0 or bp.readuint8seek0() == 1:
            self.bonelockTranslation = bp.readuint8(boneNumber)
        else:
            self.bonelockTranslation = [ 0 for i in range(boneNumber) ]

        referencePoseNumber  = readhavokint(bp)
        self.referencePose   =  [ [bp.readfloat32(4), bp.readfloat32(4), bp.readfloat32(4)] for i in range(referencePoseNumber) ]


