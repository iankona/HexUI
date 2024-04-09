
class NiSourceTexture:
    def __init__(self, bp, niffile):
        self.贴图名称 = None
        self.读取成员(bp, niffile)


    def 读取成员(self, bp, niffile):
        unknowsed = bp.read(12)      #  FF, FF, FF, FF, 0, 0, 0, 0, FF, FF, FF, FF
        # NiSourceTexture
        是外部贴图 = bp.readuint8()
        名称索引 = bp.readuint32()
        贴图名称 = self.贴图名称 = niffile.names[名称索引]




