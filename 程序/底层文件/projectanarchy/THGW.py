

class THGW:
    def __init__(self, bp):
        self.顶点骨骼组列表 = []
        self.read_block_data(bp)


    def read_block_data(self, bp):
        固定字节 = bp.readuint8(4) # [1, 0, 0, 0]
        while True: # 顶点总数 # 顶点index
            if bp.remainsize() < 2: break
            权重组 = [ [bp.readuint16(), bp.readu16float32()] for x in range(bp.readuint16()) ] # [骨骼ID, 权重值]
            self.顶点骨骼组列表.append( 权重组 )

