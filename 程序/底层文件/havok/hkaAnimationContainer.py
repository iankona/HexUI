from .havokbinary import readhavokint

def hkaAnimationContainer(bp, flag, havokfile):
    if flag == 0: return hkaAnimationContainer0(bp, havokfile)
    if flag == 6: return hkaAnimationContainer6(bp, havokfile)
    if flag == 7: return hkaAnimationContainer7(bp, havokfile)
    if flag == 15: return hkaAnimationContainer15(bp, havokfile)
    if flag == 17: return hkaAnimationContainer17(bp, havokfile)
    if flag == 25: return hkaAnimationContainer25(bp, havokfile) # [50]



class hkaAnimationContainer0:
    def __init__(self, bp, havokfile):
        pass # None


class hkaAnimationContainer6:
    def __init__(self, bp, havokfile):
        self.animations = [ readhavokint(bp) - 1 for i in range(readhavokint(bp)) ]
        self.bindings   = [ readhavokint(bp) - 1 for i in range(readhavokint(bp)) ]


class hkaAnimationContainer7:
    def __init__(self, bp, havokfile):
        self.skeletons  = [ readhavokint(bp) - 1 for i in range(readhavokint(bp)) ]
        self.animations = [ readhavokint(bp) - 1 for i in range(readhavokint(bp)) ]
        self.bindings   = [ readhavokint(bp) - 1 for i in range(readhavokint(bp)) ]


class hkaAnimationContainer15:
    def __init__(self, bp, havokfile):
        self.skeletons   = [ readhavokint(bp) - 1 for i in range(readhavokint(bp)) ]
        self.animations  = [ readhavokint(bp) - 1 for i in range(readhavokint(bp)) ]
        self.bindings    = [ readhavokint(bp) - 1 for i in range(readhavokint(bp)) ]
        self.attachments = [ readhavokint(bp) - 1 for i in range(readhavokint(bp)) ]


class hkaAnimationContainer17:
    def __init__(self, bp, havokfile):
        self.skeletons = [ readhavokint(bp) - 1 for i in range(readhavokint(bp)) ]
        self.skins     = [ readhavokint(bp) - 1 for i in range(readhavokint(bp)) ]


class hkaAnimationContainer25:
    def __init__(self, bp, havokfile):
        self.skeletons   = [ readhavokint(bp) - 1 for i in range(readhavokint(bp)) ]
        self.attachments = [ readhavokint(bp) - 1 for i in range(readhavokint(bp)) ]
        self.skins       = [ readhavokint(bp) - 1 for i in range(readhavokint(bp)) ]