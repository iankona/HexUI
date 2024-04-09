# 代码参考了blender3.6.4内置的fbx导入插件
import zlib      


class fbxnode:
    def __init__(self, bp): 
        self.blocks = {}
        self.block_section1(bp)
        self.block_section2(bp)
        self.block_section3(bp)
        self.block_section4(bp)
        self.iteration_blocks_to_values()
        self.iteration_blocks_to_attributes()


    def block_section1(self, bp):
        bx = bp.copy()
        sliceright, numdata, sizedata, numchar = bx.readuint32(), bx.readuint32(), bx.readuint32(), bx.readuint8()                  
        self.blockname = bx.readchar(numchar) # maybe type ?
        self.__numdata__, self.__sizedata__ = numdata, sizedata
        self.blocks["b_info"] = bp.readslice(13 + numchar)


    def block_section2(self, bp):
        self.blocks["b_data"] = bp.readslice(self.__sizedata__)


    def block_section3(self, bp):
        count = -1
        while True:
            count += 1
            if bp.remainsize() < 13: break
            if bp.readuint32seek0() == 0: break
            sliceleft, sliceright = bp.tell(), bp.readuint32seek0()
            self.blocks[f"b_prop{count}"] = bp.readslice(sliceright - sliceleft)


    def block_section4(self, bp):
        if bp.size() <= 0: return None
        self.blocks["b_beof"] = bp.readremainslice()



    def iteration_blocks_to_values(self):
        bp = self.blocks["b_data"]
        self.__values__ = []
        for i in range(self.__numdata__):
            datatype = bp.readchar()
            match datatype:
                case "S" : self.__values__.append(bp.readchar(bp.readuint32()))
                case "R" : self.__values__.append(bp.read()) # 
                case "B" : self.__values__.append(bp.read()) # 字节
                case "C" : self.__values__.append(bp.readuint8()) # bool
                case "Y" : self.__values__.append(bp.readint16()) # int16
                case "I" : self.__values__.append(bp.readint32()) # int32
                case "L" : self.__values__.append(bp.readint64()) # int64
                case "F" : self.__values__.append(bp.readfloat32()) # float32
                case "D" : self.__values__.append(bp.readfloat64()) # float64
                case "i" : self.__values__.append(self.__data__array__(bp, "i"))
                case "l" : self.__values__.append(self.__data__array__(bp, "l"))
                case "f" : self.__values__.append(self.__data__array__(bp, "f"))
                case "d" : self.__values__.append(self.__data__array__(bp, "d"))
                case  _  : break

    def __data__array__(self, bp, datatype):
        numelem, compress, datasize = bp.readuint32(3)

        if compress == 1: 
            by = bp.tobpbytes(zlib.decompress(bp.read(datasize)))
        else:
            by = bp.tobpbytes(bp.read(datasize))

        result = []
        match datatype:
            case "i" : result = by.readint32(numelem)
            case "l" : result = by.readint64(numelem)
            case "f" : result = by.readfloat32(numelem)
            case "d" : result = by.readfloat64(numelem)
        return result
    

    def iteration_blocks_to_attributes(self):
        self.__attributes__ = []
        for keychar, bp in self.blocks.items(): 
            if "b_prop" in keychar: self.__attributes__.append(fbxnode(bp))




    # def block_read(self, bp):
    #     by = bp.copy()
    #     sliceleft = by.tell()
    #     sliceright, numvalue, datasize, numchar = by.readuint32(), by.readuint32(), by.readuint32(), by.readuint8()

    #     asize = 13 + numchar
    #     bsize = datasize
    #     csize = sliceright - sliceleft - datasize - 13 - numchar

    #     self.blocks["b_section1"] = bp.readslice(asize)
    #     self.blocks["b_section2"] = bp.readslice(bsize)
    #     self.blocks["b_section3"] = bp.readslice(csize)