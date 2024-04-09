
class Object: pass
class gamebryofile:
    def __init__(self, bp, gamebryoclasses):
        self.header = Object()
        self.names = []
        self.classes = []
        self.blockindexs = []
        self.block_sizes = []
        self.datas = []

        self.readheader(bp)
        self.readblocks(bp, gamebryoclasses)


    def readheader(self, bp):
        self.filepath = bp.rdpath
        h = self.header
        h.magic1 = bp.readchar(38)
        h.magic2 = bp.readuint8()
        h.fileversion = bp.readuint8(4)[::-1]
        is_little = bp.readuint8()
        if is_little: bp.endian = "<"      # 左端存小数，是为小端。
        h.unknowuint32 = bp.readuint32()  # userversion or platform


    def readblocks(self, bp, gamebryoclasses):
        numblock = bp.readuint32()
        self.classes = [bp.readchar(bp.readuint32()) for x in range(bp.readuint16())]
        self.blockindexs = [bp.readuint16() for x in range(numblock)]
        self.block_sizes = [bp.readuint32() for x in range(numblock)]

        numname, namemaxlength = bp.readuint32(2)
        self.names = [bp.readchar(bp.readuint32()) for x in range(numname)]
        numgroup = bp.readuint32()

        for size, index in zip(self.block_sizes, self.blockindexs):
            bn = bp.readslice(size)
            classname = self.classes[index]
            if "NiDataStream" in classname: classname = "NiDataStream"
            o = gamebryoclasses[classname](bn, self)
            o.type = classname
            self.datas.append(o)
