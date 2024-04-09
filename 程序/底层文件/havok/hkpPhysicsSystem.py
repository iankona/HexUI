from .havokbinary import readhavokint


def hkpPhysicsSystem(bp, flag, havokfile):
    if flag == 162: return hkpPhysicsSystem162(bp, havokfile)


class hkpPhysicsSystem162:
    def __init__(self, bp, havokfile):
        numrigidBodie = readhavokint(bp)
        self.rigidBodies = [readhavokint(bp) for i in range(numrigidBodie)]
        self.name = havokfile.readname(bp)
        self.active = bp.readuint8()
