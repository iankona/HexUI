

class MBUS:
    def __init__(self, bp):
        self.顶点划分信息列表 = []
        self.read_block_data(bp)


    def read_block_data(self, bp):
        固定字节 = bp.readuint8(8)
        for i in range(bp.readuint32()):
            固定字节 = bp.readuint8(16) # [0, 0, 0, 0, 255, 255, 0, 0, 255, 255, 255, 255, 3, 0, 0, 0]
            变化浮点 = bp.readfloat32(5)
            材质名称 = bp.readchar(bp.readuint32())
            末尾FFFF = bp.readuint8(2)

        for j in range(bp.readuint32()): # 64字节/块
            [loop左值, loop个数], [固定00, 固定00] = bp.readuint32(2), bp.readuint32(2)
            [vert左值, vert个数], [固定00, 固定00] = bp.readuint32(2), bp.readuint32(2)
            包围方盒, 材质索引, 字节补齐 = bp.readfloat32(6), bp.readuint32(), bp.readuint8(4)  # FFFFFFFF
            self.顶点划分信息列表.append( [loop左值, loop个数, vert左值, vert个数, 材质索引] )

