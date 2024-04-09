class EHCM:
    def __init__(self, bp):
        self.形态键数量 = 0
        self.read_block_data(bp)


    def read_block_data(self, bp):
        self.形态键数量 = bp.readuint32()

