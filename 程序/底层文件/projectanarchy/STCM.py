class STCM:
    def __init__(self, bp):
        self.形态键列表 = []
        self.read_block_data(bp)


    def read_block_data(self, bp):
        while True:
            if bp.remainsize() <= 8: break
            形态键名称 = bp.readchar(bp.readuint32())
            相对顶点列表 = [ [bp.readuint32(), bp.readfloat32(4)[0:3], bp.readfloat32(4)[0:3]]  for i in range(bp.readuint32())]  # vertindex, leftposition, rightposition
            self.形态键列表.append([形态键名称, 相对顶点列表])

