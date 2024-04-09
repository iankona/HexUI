class NiSkinningMeshModifier:
    def __init__(self, bp, niffile):
        self.骨骼ID列表 = []
        self.读取成员(bp, niffile)

    def 读取成员(self, bp, niffile):
        unknowse = bp.read(18)
        position, matrix, scale = bp.readfloat32(3), [bp.readfloat32(3), bp.readfloat32(3), bp.readfloat32(3)], bp.readfloat32()
        骨骼总数 = bp.readuint32()
        骨骼ID列表 = [bp.readuint32() for x in range(骨骼总数)]
        # 接下来不在读取

        self.骨骼ID列表 = 骨骼ID列表