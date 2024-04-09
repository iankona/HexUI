
class NiMesh:
    def __init__(self, bp, niffile):
        self.名称 = None
        self.position = None
        self.matrix = None
        self.scale = None
        self.属性ID列表 = None
        self.NiDataStreamID列表 = []
        self.修改器ID列表 = None
        self.读取成员(bp, niffile)


    def 读取成员(self, bp, niffile):
        # NiObjectNET
        名称索引 = bp.readuint32()
        if 名称索引 == 0xFFFFFFFF:
            self.名称 = ""
        else:
            self.名称 = niffile.names[名称索引]
        ID数量 =  bp.readuint32()
        ID列表 = [bp.readuint32() for i in range(ID数量)]
        FF结尾 =  bp.readuint32()        # 大部分都为，0x  FF, FF, FF, FF 或者 27
        # NiAVObject
        未知标记 = bp.readuint16()      # 16，22，23
        self.position = bp.readfloat32(3)
        self.matrix = [bp.readfloat32(3), bp.readfloat32(3), bp.readfloat32(3)]
        self.scale = bp.readfloat32()

        属性ID数量 =  bp.readuint32()
        self.属性ID列表 = [bp.readuint32() for i in range(属性ID数量)]  # NiMaterialProperty, NiTexturingProperty, NiAlphaProperty, NiSpecularProperty, NiVertexColorProperty
        FF结尾     =  bp.readuint32()        # 大部分都为，0x  FF, FF, FF, FF

        # NiMaterial
        # 0, 0, 0, 0,   FF，FF, FF, FF,   00
        材质总数 =  bp.readuint32()
        材质列表 = [bp.readuint32() for i in range(材质总数)]
        FF中间   =  bp.readuint32()
        材质索引 = [bp.readuint32() for i in range(材质总数)]
        材质是否需要更新 = bp.readuint8()

        # NiMesh
        # 0, 0, 0, 0,    1, 0,    0,
        网格类型, NiDataStream区域总数, 网格是实例 = bp.readuint32(), bp.readuint16(), bp.readuint8() # meshprimtype
        包围盒中心, 包围盒弧度 = bp.readfloat32(3), bp.readfloat32()
        self.NiDataStream区域总数 =  NiDataStream区域总数

        # NiDataStream
        NiDataStreamID数量 = bp.readuint32()
        for i in range(NiDataStreamID数量):
            NiDataStreamID, 是标准网格, 映射总数 = bp.readuint32(), bp.readuint8(), bp.readuint16()  # 映射总数固定为1
            NiDataStream区域编号列表 = [bp.readuint16() for i in range(NiDataStream区域总数)]
            标识总数 = bp.readuint32()
            标识列表 = [ [niffile.names[bp.readuint32()], bp.readuint32()] for i in range(标识总数)] # 标识名称, 标识编号
            self.NiDataStreamID列表.append( [NiDataStreamID, 标识列表] )

        # NiSkinningMeshModifier
        修改器ID总数 =  bp.readuint32()
        self.修改器ID列表 = [bp.readuint32() for x in range(修改器ID总数)]




