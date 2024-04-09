class NiTexturingProperty:
    def __init__(self, bp, niffile):
        self.NiSourceTextureID列表 = []
        self.读取成员(bp, niffile)


    def 读取成员(self, bp, niffile):
        unknowse = bp.read(12)       #  FF, FF, FF, FF, 0, 0, 0, 0, FF, FF, FF, FF
        未知标记, 贴图数量 = bp.readuint16(), bp.readuint32()
        # NiTexturingProperty
        for i in range(贴图数量):
            贴图存在 = bp.readuint8()
            if 贴图存在 == 1:
                NiSourceTextureID, 是引用贴图, 材质名称索引 = bp.readuint32(), bp.readuint8(), bp.readuint32()
                self.NiSourceTextureID列表.append(NiSourceTextureID)
                # print(niffile.名称列表[材质名称索引])
        接下来贴图数 = bp.readuint32()
        for j in range(接下来贴图数):
            贴图存在, NiSourceTextureID, 是引用贴图, 材质名称索引, 贴图编号 = bp.readuint8(), bp.readuint32(), bp.readuint8(), bp.readuint32(), bp.readuint32()
            self.NiSourceTextureID列表.append(NiSourceTextureID)
            # print(niffile.名称列表[材质名称索引])


