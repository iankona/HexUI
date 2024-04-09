
from .havokbinary import readhavokint



def hkxMeshSection(bp, flag, hkafile):
    if flag == 7: return hkxMeshSection7(bp, hkafile)


class hkxMeshSection7:
    def __init__(self, bp, hkafile):
        self.vertexBuffer = readhavokint(bp)-1
        self.indexBuffers = [readhavokint(bp)-1 for i in range(readhavokint(bp))]
        self.material = readhavokint(bp)-1
