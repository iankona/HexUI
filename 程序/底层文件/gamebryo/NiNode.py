

class NiNode:
    def __init__(self, bp, niffile):
        self.名称 = None
        self.position = None
        self.matrix = None
        self.scale = None
        self.children = []
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
        FF结尾 =  bp.readuint32()        # 大部分都为，0x  FF, FF, FF, FF
        # NiAVObject
        未知标记 = bp.readuint16()      # 16，22，23，u16(2,1),
        self.position =  bp.readfloat32(3)
        self.matrix   = [bp.readfloat32(3), bp.readfloat32(3), bp.readfloat32(3)]
        self.scale    =  bp.readfloat32()

        属性ID数量 =  bp.readuint32()
        属性ID列表 = [bp.readuint32() for i in range(属性ID数量)]
        FF结尾     =  bp.readuint32()        # 大部分都为，0x  FF, FF, FF, FF

        # NiNode
        子ID数量 = bp.readuint32()
        for x in range(子ID数量):
            索引 = bp.readuint32()
            if 索引 == 0xFFFFFFFF: continue
            self.children.append(索引)

        灯光ID数量 =  bp.readuint32()
        灯光ID列表 = [bp.readuint32() for x in range(灯光ID数量)]





