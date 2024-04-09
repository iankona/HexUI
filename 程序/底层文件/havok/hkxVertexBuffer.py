
from .havokbinary import readhavokint, readhavokuint

def hkxVertexBuffer(bp, flag, hkafile):
    if flag == 3: return hkxVertexBuffer3(bp, hkafile)

class hkxVertexBuffer3:
    def __init__(self, bp, hkafile):
        self.data_name_index = readhavokint(bp)
        self.data = hkxVertexBufferVertexData(bp, hkafile)
        datatype1, numdesc, unknowindex = readhavokint(bp), readhavokint(bp), readhavokint(bp) # ?  0 6 animations
        datatype8, self.byteOffsets = bp.readuint8(), [readhavokint(bp) for i in range(numdesc)]
        datatype8, self.types       = bp.readuint8(), [readhavokint(bp) for i in range(numdesc)]
        datatype8, self.usages      = bp.readuint8(), [readhavokint(bp) for i in range(numdesc)]
        datatype8, self.byteStrides = bp.readuint8(), [readhavokint(bp) for i in range(numdesc)]
        self.numElementss = bp.readuint8(numdesc)

class hkxVertexBufferVertexData:
    def __init__(self, bp, hkafile):
        numdata, typedata = readhavokint(bp), bp.readuint8()
        self.vectorData = [readhavokuint(bp) for i in range(numdata)] # readhavokuint(bp, typedata), 8
        numdata, typedata = readhavokint(bp), bp.readuint8()
        self.floatData = [readhavokuint(bp) for i in range(numdata)]  # readhavokuint(bp, typedata), 8
        numdata, typedata = readhavokint(bp), bp.readuint8()
        self.uint32Data = [readhavokuint(bp) for i in range(numdata)] # readhavokuint(bp, typedata), 8
        self.uint8Data = bp.readuint8(readhavokint(bp))
        self.numVerts = readhavokint(bp)
        self.vectorStride = readhavokint(bp)
        self.floatStride = readhavokint(bp)
        self.uint32Stride = readhavokint(bp)
        self.uint8Stride = readhavokint(bp)


class DataType:
    HKX_DT_NONE = 0
    HKX_DT_UINT8 = 1
    HKX_DT_INT16 = 2
    HKX_DT_UINT32 = 3
    HKX_DT_FLOAT = 4

class DataUsage:
    HKX_DU_NONE = 0
    HKX_DU_POSITION = 1
    HKX_DU_COLOR = 2
    HKX_DU_NORMAL = 4
    HKX_DU_TANGENT = 8
    HKX_DU_BINORMAL = 16
    HKX_DU_TEXCOORD = 32
    HKX_DU_BLENDWEIGHTS = 64
    HKX_DU_BLENDINDICES = 128
    HKX_DU_USERDATA = 256