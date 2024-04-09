from .havokbinary import readhavokint


def hkaDefaultAnimatedReferenceFrame(bp, flag, hkafile):
    if flag == 30: return hkaDefaultAnimatedReferenceFrame30(bp, hkafile)


class hkaDefaultAnimatedReferenceFrame30:
    def __init__(self, bp, hkafile):
        self.up = bp.readfloat32(4)
        self.forward = bp.readfloat32(4)
        self.duration = bp.readfloat32()

        referenceFrameSampleNumber, enumNumber = readhavokint(bp), readhavokint(bp)
        self.referenceFrameSamples = [bp.readfloat32(enumNumber) for i in range(referenceFrameSampleNumber) ]

