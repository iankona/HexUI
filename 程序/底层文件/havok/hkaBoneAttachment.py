from .havokbinary import readhavokint

def hkaBoneAttachment(bp, flag, hkafile):
    if flag == 31: return hkaBoneAttachment31(bp, hkafile)



class hkaBoneAttachment31:
    def __init__(self, bp, hkafile):
        self.originalSkeletonName = hkafile.readname(bp)
        self.boneFromAttachment = [bp.readfloat32(4),  bp.readfloat32(4),  bp.readfloat32(4), bp.readfloat32(4)]
        self.attachment = readhavokint(bp)-1
        self.name = hkafile.readname(bp)
        self.boneIndex = readhavokint(bp)