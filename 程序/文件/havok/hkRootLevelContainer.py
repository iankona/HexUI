
from .havokBinary import readhavokint

def hkRootLevelContainer(bp, flag, hkafile):
    if flag == 0: return hkRootLevelContainer0(bp, hkafile)


class hkRootLevelContainer0:
    def __init__(self, bp, hkafile):
        self.numnamedVariant = readhavokint(bp)
        self.parentClassname = hkafile.readname(bp)
        self.names      = [ hkafile.readname(bp) for i in range(self.numnamedVariant) ]
        self.classNames = [ hkafile.readname(bp) for i in range(self.numnamedVariant) ]
        self.variants   = [ readhavokint(bp)-1   for i in range(self.numnamedVariant) ]
