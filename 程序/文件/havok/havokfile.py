from .havokBinary import readhavokint, readhavokintseek0

from . import hkaSplineCompressedAnimationData
from . import hkaInterleavedUncompressedAnimationData


class havokfile:
    def __init__(self, bp, havokclasses):
        # magic, [30, 13, 176, 202, 206, 250, 17, 208] # hka # [1E, 0D, B0, CA, CE, FA, 11, D0] # 1E0DB0CA, CEFA11D0 # CAB00D1E, D011FACE
        self.names = ["", ""]
        self.types = []
        self.datas = []

        self.blocks = {}
        self.block_section(bp, havokclasses)

        self.readhkaSplineCompressedAnimationData()
        self.readhkaInterleavedUncompressedAnimationData()


    def block_section(self, bp, havokclasses):
        bp.readslice(31) # head
        sizes = []
        while True:
            if bp.remainsize() < 16: break
            if bp.readuint8seek0() == 4: 
                mpleft, classname, mpright = bp.tell(), self.__block__type__(bp), bp.tell()
                sizes.append([f"b_type_{classname}", mpright - mpleft])
                continue
            if bp.readuint8seek0() == 8: 
                mpleft, classname, mpright = bp.tell(), self.__block__data__(bp, havokclasses), bp.tell()
                sizes.append([f"b_data_{classname}", mpright - mpleft])
                continue
            break

        
        bx = bp.copy()
        self.filepath = bx.rdpath
        self.blocks["b_head"] = bx.readslice(31)
        for name, size in sizes: 
            self.blocks[name] = bx.readslice(size)
        self.blocks["b_beof"] = bx.readremainslice()


    def __block__type__(self, bp):
        typeflag, classname, layout, parantindex, numelement = bp.readuint8(), self.readname(bp), bp.readuint8(), bp.readuint8(), readhavokint(bp)
        elements = []
        for i in range(numelement):
            varname, vartype, havoktype = self.readname(bp), bp.readuint8(), ""
            if vartype in [16, 18, 48, 50]: havoktype = self.readname(bp)
            if vartype in [66, 70]: havoktype = str(bp.readuint8())
            elements.append([varname, vartype, havoktype])
        self.types.append([typeflag, classname, layout, parantindex, numelement, elements]) 
        return classname


    def readname(self, bp):
        havokint = readhavokint(bp)
        if havokint < 0: 
            return self.names[havokint]
        else:
            chars = bp.readchar(havokint)
            self.names.insert(0, chars)
            return chars


    def __block__data__(self, bp, havokclasses):
        typeflag, classindex = bp.readuint8(), readhavokint(bp)-1
        classname = self.types[classindex][1]
        if classname in ["hkaSkeleton", "hkxScene"]: 
            classflag  = bp.readuint16()
        else: 
            classflag = readhavokint(bp)
        self.datas.append([typeflag, classname, classflag, havokclasses[classname](bp, classflag, self)])
        return classname



    def readhkaSplineCompressedAnimationData(self):
        for [typeflag, classname, classflag, datainstance] in self.datas:
            if classname != "hkaSplineCompressedAnimation": continue
            hkaSplineCompressedAnimationData.函数(self, datainstance)
            # filedata.action.tracknodes.items()

    def readhkaInterleavedUncompressedAnimationData(self):
        for [typeflag, classname, classflag, datainstance] in self.datas:
            if classname != "hkaInterleavedUncompressedAnimation": continue
            hkaInterleavedUncompressedAnimationData.函数(self, datainstance)
            # filedata.action.tracknodes.items()




