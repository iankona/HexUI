from .anarchymeshfile import anarchymeshfile


class Object: pass
class HSMV:
    def __init__(self, bp):
        self.head = Object()
        self.mesh = Object()
        self.read_block_data(bp)


    def read_block_data(self, bp):
        self.head.magic = bp.readuint8(4)
        self.head.version = bp.readuint8(4)
        self.mesh = anarchymeshfile(bp)
        self.tail = bp.readslice(28) # 末尾28字节，bound?
