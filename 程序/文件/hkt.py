




class havokfiletype2:
    def __init__(self, bp, havokclasses):
        self.head = Object()
        self.names = ["", ""]
        self.types = {}
        self.datas = []


        self.read_head(bp)
        self.read_types(bp)
        self.read_datas(bp, havokclasses)
        # self.readhkaSplineCompressedAnimationData()
        # self.readhkaInterleavedUncompressedAnimationData()


    def read_blocks(self, bp):
        magichex = bp.readhexseek0(8)
        magicuint8 = bp.readuint8seek0(8)
        print(magichex, magicuint8)
        self.blocks["b_magic"] = bp.readslice(8)
        match magicuint8:
            case [30, 13, 176, 202, 206, 250, 17, 208]: # hka # [1E, 0D, B0, CA, CE, FA, 11, D0] # 1E0DB0CA, CEFA11D0 # CAB00D1E, D011FACE
                self.blocks["b_sdk"] = bp.readslice(21)
            case [87, 224, 224, 87, 16, 192, 192, 16]:  # hkt # [57, E0, E0, 57, 10, CO, CO, 10] # 57E0E057, 10COCO10
                pass













    def readheader(self, bp):
        self.filepath = bp.rdpath
        h = self.header
        h.magic1 = bp.readhex(4)
        h.magic2 = bp.readhex(4)
        h.userversion = bp.readuint8()
        h.classversion = bp.readuint8()
        h.havokversion = bp.readchar(bp.readuint8()//2)
        h.unknowsed = bp.readuint8(6)
        self.tells.append(bp.tell())





    def readtypes(self, bp):
        while True:
            test = bp.readuint8seek0()
            if test != 4: break
            typeflag, classname, layout, parantindex, numelement = bp.readuint8(), self.readname(bp), bp.readuint8(), bp.readuint8(), bp.readuint8()//2
            elements = []
            for i in range(numelement):
                varname, vartype, havoktype = self.readname(bp), bp.readuint8(), ""
                if vartype in [16, 18, 48, 50]: havoktype = self.readname(bp)
                if vartype in [66, 70]: havoktype = str(bp.readuint8())
                elements.append([varname, vartype, havoktype])
            self.types[classname] = [typeflag, classname, layout, parantindex, numelement, elements]
        self.tells.append(bp.tell())


    def readname(self, bp):
        havokint = readhavokint(bp)
        if havokint < 0: return self.names[havokint]
        else:
            chars = bp.readchar(havokint)
            self.names.insert(0, chars)
            return chars



    def readblocks(self, bp, havokclasses):
        classnames = list(self.types.keys())
        while True:
            test = bp.readuint8seek0()
            if test != 8: break
            try:
                typeflag, classnameindex = bp.readuint8(), readhavokint(bp)-1
                classname = classnames[classnameindex]
                if classname in ["hkaSkeleton", "hkxScene"]: classflag  = bp.readuint16()
                else: classflag = readhavokint(bp)
                self.datas.append([classname, typeflag, classnameindex, classflag, havokclasses[classname](bp, classflag, self)])
                self.tells.append(bp.tell())
            except:
                self.tells.append(bp.tell())


    def readhkaSplineCompressedAnimationData(self):
        self.spacename = "BONESPACE"
        for [classname, typeflag, classnameindex, classflag, datainstance] in self.datas:
            if classname != "hkaSplineCompressedAnimation": continue
            hkaSplineCompressedAnimationData.函数(self, datainstance)


    def readhkaInterleavedUncompressedAnimationData(self):
        self.spacename = "BASISSPACE"
        for [classname, typeflag, classnameindex, classflag, datainstance] in self.datas:
            if classname != "hkaInterleavedUncompressedAnimation": continue
            hkaInterleavedUncompressedAnimationData.函数(self, datainstance)

