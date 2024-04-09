
from .havokbinary import readhavokint

def hkRootLevelContainer(bp, flag, havokfile):
    if flag == 0: return hkRootLevelContainer0(bp, havokfile)


class hkRootLevelContainer0:
    def __init__(self, bp, havokfile):
        self.numnamedVariant = readhavokint(bp)
        self.parentClassname = havokfile.readname(bp)
        self.names      = [ havokfile.readname(bp) for i in range(self.numnamedVariant) ]
        self.classNames = [ havokfile.readname(bp) for i in range(self.numnamedVariant) ]
        self.variants   = [ readhavokint(bp)-1   for i in range(self.numnamedVariant) ]
