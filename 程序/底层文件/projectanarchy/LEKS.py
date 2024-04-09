

class LEKS:
    def __init__(self, bp):
        self.骨骼节点列表 = []
        self.read_block_data(bp)


    def read_block_data(self, bp):
        固定00, 骨骼总数 = bp.readuint16(2)
        for i in range(骨骼总数):
            骨骼名称, 父骨骼ID = bp.readchar(bp.readuint32()),  bp.readint16()
            head, rotation_quaternion, directe, unkown_rotation_quaternion = bp.readfloat32(3), bp.readfloat32(4), bp.readfloat32(3), bp.readfloat32(4)
            self.骨骼节点列表.append( [骨骼名称, 父骨骼ID, head, rotation_quaternion, directe, unkown_rotation_quaternion] )
