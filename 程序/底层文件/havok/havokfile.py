from .havokbinary import readhavokint, readhavokintseek0



class havokfile:
    def __init__(self, bp, havokclasses):
        # magic, [30, 13, 176, 202, 206, 250, 17, 208] # hka # [1E, 0D, B0, CA, CE, FA, 11, D0] # 1E0DB0CA, CEFA11D0 # CAB00D1E, D011FACE
        self.names = ["", ""]
        self.types = []
        self.datas = []

        self.blocks = {}
        self.block_section(bp, havokclasses)




    def block_section(self, bp, havokclasses):
        bp.readslice(31) # head
        sizes = []
        while True:
            if bp.remainsize() < 16: break
            if bp.readuint8seek0() == 4: 
                mpleft, classname, mpright = bp.tell(), self.__block__type__(bp), bp.tell()
                sizes.append([f"b_type_{classname}", mpright - mpleft])
                continue
            try:
                if bp.readuint8seek0() == 8: 
                    mpleft, classname, mpright = bp.tell(), self.__block__data__(bp, havokclasses), bp.tell()
                    sizes.append([f"b_data_{classname}", mpright - mpleft])
                    continue
            except Exception as e:
                print(e)
                break
            break

        
        bx = bp.copy()
        self.filepath = bx.rdpath
        self.blocks["b_head"] = bx.readslice(31)
        for name, size in sizes: 
            name = self.__block__name__(name)
            self.blocks[name] = bx.readslice(size)
        self.blocks["b_beof"] = bx.readremainslice()



    def __block__name__(self, name):
        if name not in self.blocks: return name
        for i in range(1000):
            iname = f"{name}_{i+1}"
            if iname not in self.blocks: return iname
        raise ValueError("ERROR::havokfile同类型对象超过1000个...")



    def __block__type__(self, bp):
        typeflag, classname, layout, parantindex, numelement = bp.readuint8(), self.readname(bp), bp.readuint8(), readhavokint(bp), readhavokint(bp)
        elements = []
        for i in range(numelement):
            varname, vartype, datatype, havoktype = self.readname(bp), bp.readuint8(), None, ""
            if vartype in [16, 18, 48, 50]: 
                havoktype = self.readname(bp)
            if vartype in [66, 68, 70, 72]: 
                datatype = str(bp.readuint8())
            if vartype in [80]: 
                datatype = str(bp.readuint8())
                havoktype = self.readname(bp)
            elements.append([varname, vartype, datatype, havoktype])
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
        if classname in ["hkaSkeleton", "hkxScene", "hkpBvCompressedMeshShape", "hkxNode"]: 
            classflag  = bp.readuint16()
        elif classname in ["hkaSkeletonMapper", "hkpPhysicsSystem"]:
            classflag  = bp.readuint8()
        else: 
            classflag = readhavokint(bp)
        self.datas.append([typeflag, classname, classflag, havokclasses[classname](bp, classflag, self)])
        return classname








