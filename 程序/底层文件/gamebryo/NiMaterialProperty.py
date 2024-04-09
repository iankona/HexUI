class NiMaterialProperty:
    def __init__(self, bp, niffile):
        self.名称 = None
        self.读取成员(bp, niffile)


    def 读取成员(self, bp, niffile):
        名称索引 = bp.readint32()   # NiMaterialProperty列表中，名称索引有大量的 FF, FF, FF, FF，有非正常材质名
        if 名称索引 == -1:
            self.名称 = ""
        else:
            self.名称 = niffile.names[名称索引]
        unknowse = bp.read(8)     # 0, 0, 0, 0, FF, FF, FF, FF
        unfloats = bp.readfloat32(14)
