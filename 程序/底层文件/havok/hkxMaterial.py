

from .havokbinary import readhavokint

def hkxMaterial(bp, flag, hkafile):
    if flag == 15166: return hkxMaterial15166(bp, hkafile)

class hkxMaterial15166:
    def __init__(self, bp, hkafile):
        self.name = hkafile.readname(bp)
        self.stages = [bp.readuint8(4) for i in range(readhavokint(bp))]
        self.diffuseColor = bp.readfloat32(4)
        self.ambientColor = bp.readfloat32(4)
        self.specularColor = bp.readfloat32(4)
        self.emissiveColor = bp.readfloat32(4)
        self.uvMapScale = bp.readfloat32(2)
        self.uvMapOffset = bp.readfloat32(2)
        self.uvMapAlgorithm = readhavokint(bp)
        self.specularMultiplier = bp.readfloat32()
        self.specularExponent = bp.readfloat32()
        self.transparency = readhavokint(bp)

