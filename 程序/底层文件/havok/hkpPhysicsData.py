from .havokbinary import readhavokint


def hkpPhysicsData(bp, flag, havokfile):
    if flag == 2: return hkpPhysicsData2(bp, havokfile)


class hkpPhysicsData2:
    def __init__(self, bp, havokfile):
        numsystem = readhavokint(bp)
        self.systems = [readhavokint(bp) for i in range(numsystem)] 