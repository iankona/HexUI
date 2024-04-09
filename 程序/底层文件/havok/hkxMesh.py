
from .havokbinary import readhavokint

def hkxMesh(bp, flag, hkafile):
    if flag == 1: return hkxMesh1(bp, hkafile)


class hkxMesh1:
    def __init__(self, bp, hkafile):
        self.sections = [readhavokint(bp)-1 for i in range(readhavokint(bp))]
