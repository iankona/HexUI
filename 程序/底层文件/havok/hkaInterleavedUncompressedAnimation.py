from .havokbinary import readhavokint



def hkaInterleavedUncompressedAnimation(bp, flag, hkafile):
    if flag == 39: return hkaInterleavedUncompressedAnimation39(bp, hkafile)

class hkaInterleavedUncompressedAnimation39:
    def __init__(self, bp, hkafile):
        # hkaAnimation
        self.type = readhavokint(bp)
        self.duration = bp.readfloat32()
        self.numberOfTransformTracks = readhavokint(bp)

        annotationTrackNumber = readhavokint(bp) # 骨骼名称列表
        annotationTrackType, self.annotationTracks = readhavokint(bp), [hkafile.readname(bp) for i in range(annotationTrackNumber) ]
        # hkaInterleavedUncompressedAnimation
        transformNumber = readhavokint(bp)
        self.transforms = [ [bp.readfloat32(4), bp.readfloat32(4), bp.readfloat32(4)] for i in range(transformNumber) ]


