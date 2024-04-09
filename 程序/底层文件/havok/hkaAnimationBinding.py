from .havokbinary import readhavokint

def hkaAnimationBinding(bp, flag, hkafile):
    if flag == 39: return hkaAnimationBinding39(bp, hkafile)
    if flag == 7: return hkaAnimationBinding7(bp, hkafile)


class hkaAnimationBinding39:
    def __init__(self, bp, hkafile):
        self.originalSkeletonName = hkafile.readname(bp)
        self.animation = readhavokint(bp) - 1 # 数据块索引

        transformTrackToBoneIndiceNumber, transformTrackToBoneIndiceType = readhavokint(bp), readhavokint(bp)
        self.transformTrackToBoneIndices = [ readhavokint(bp) for i in range(transformTrackToBoneIndiceNumber) ]

        self.blendHint = readhavokint(bp)


class hkaAnimationBinding7:
    def __init__(self, bp, hkafile):
        self.originalSkeletonName = hkafile.readname(bp)
        self.animation = readhavokint(bp) - 1 # 数据块索引

        transformTrackToBoneIndiceNumber, transformTrackToBoneIndiceType = readhavokint(bp), readhavokint(bp)
        self.transformTrackToBoneIndices = [ readhavokint(bp) for i in range(transformTrackToBoneIndiceNumber) ]

