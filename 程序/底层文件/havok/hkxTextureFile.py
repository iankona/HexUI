
from .havokbinary import readhavokint

def hkxTextureFile(bp, flag, hkafile):
    if flag == 7: return hkxTextureFile7(bp, hkafile)


class hkxTextureFile7:
    def __init__(self, bp, hkafile):
        self.filename = hkafile.readname(bp)
        self.name = hkafile.readname(bp)
        self.originalFilename = hkafile.readname(bp)