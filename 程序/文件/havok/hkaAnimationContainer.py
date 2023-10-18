from .havokBinary import readhavokint

def hkaAnimationContainer(bp, flag, hkafile):
    if flag == 6: return hkaAnimationContainer6(bp, hkafile)
    if flag == 7: return hkaAnimationContainer7(bp, hkafile)
    if flag == 0: return hkaAnimationContainer0(bp, hkafile)
    if flag == 15: return hkaAnimationContainer15(bp, hkafile)


class hkaAnimationContainer6:
    def __init__(self, bp, hkafile):
        self.animations = [ readhavokint(bp) - 1 for i in range(readhavokint(bp)) ]
        self.bindings   = [ readhavokint(bp) - 1 for i in range(readhavokint(bp)) ]


class hkaAnimationContainer7:
    def __init__(self, bp, hkafile):
        self.skeletons  = [ readhavokint(bp) - 1 for i in range(readhavokint(bp)) ]
        self.animations = [ readhavokint(bp) - 1 for i in range(readhavokint(bp)) ]
        self.bindings   = [ readhavokint(bp) - 1 for i in range(readhavokint(bp)) ]

class hkaAnimationContainer0:
    def __init__(self, bp, hkafile):
        pass # None


class hkaAnimationContainer15:
    def __init__(self, bp, hkafile):
        self.skeletons   = [ readhavokint(bp) - 1 for i in range(readhavokint(bp)) ]
        self.animations  = [ readhavokint(bp) - 1 for i in range(readhavokint(bp)) ]
        self.bindings    = [ readhavokint(bp) - 1 for i in range(readhavokint(bp)) ]
        self.attachments = [ readhavokint(bp) - 1 for i in range(readhavokint(bp)) ]

