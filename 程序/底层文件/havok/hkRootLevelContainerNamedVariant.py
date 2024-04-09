from .havokBinary import readhavokint

def hkRootLevelContainerNamedVariant(bp, flag, hkafile):
    if flag == 0: return hkRootLevelContainerNamedVariant0(bp, hkafile)


class hkRootLevelContainerNamedVariant0:
    def __init__(self, bp, hkafile):
        self.numnamedVariant = readhavokint(bp)
        self.parentClassname = hkafile.readname(bp)
        self.names      = [ hkafile.readname(bp) for i in range(self.numnamedVariant) ]
        self.classNames = [ hkafile.readname(bp) for i in range(self.numnamedVariant) ]
        self.variants   = [ readhavokint(bp)-1   for i in range(self.numnamedVariant) ]