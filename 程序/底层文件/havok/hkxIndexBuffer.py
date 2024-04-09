
from .havokbinary import readhavokint


def hkxIndexBuffer(bp, flag, hkafile):
    if flag == 21: return hkxIndexBuffer21(bp, hkafile)




class hkxIndexBuffer21:
    def __init__(self, bp, hkafile):
        self.indexType = readhavokint(bp)
        numindices32, typeindices32 = readhavokint(bp), readhavokint(bp)
        self.indices32 = [readhavokint(bp) for i in range(numindices32)]
        self.length = readhavokint(bp)