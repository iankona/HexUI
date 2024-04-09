
class Object: pass
class anarchymeshfile:
    def __init__(self, bp):
        self.head = Object()
        self.blocks = {}
        self.顶点大小 = 0
        self.顶点描述列表 = []
        self.顶点总数 = 0
        self.Loop总数 = 0
        self.顶点列表 = []
        self.顶点UV列表 = []
        self.顶点Loop列表 = []

        self.read_blocks(bp)
        self.read_head()
        self.read_vert()
        self.read_loop()


    def read_blocks(self, bp):
        self.filepath = bp.rdpath

        magic, version = bp.readuint32seek0(2)
        self.blocks["b_head"] = bp.readslice(8)
        self.blocks["b_datatype"] = bp.readslice(12 + bp.readuint32seek0(2)[1])
        match version:
            case 1: self.blocks["b_mesh_info"] = bp.readslice(32)
            case 5: self.blocks["b_mesh_info"] = bp.readslice(38)
            case _: print(f"Anarchymeshfile::未知mesh_info_type_{version}")

        self.顶点大小, self.顶点描述列表 = self.read_vert_info(self.blocks["b_datatype"])
        match version:
            case 1: self.顶点总数, self.Loop总数, Face总数, 附加标识, 附加末尾 = self.read_mesh_info_version1(self.blocks["b_mesh_info"])
            case 5: self.顶点总数, self.Loop总数, Face总数, 附加标识, 附加末尾 = self.read_mesh_info_version5(self.blocks["b_mesh_info"])
        if 附加末尾 == [0xFF, 0xFF]:
            self.blocks["b_mesh_info_endof"] = bp.readslice(64)


        顶点块大小 = self.顶点总数*self.顶点大小
        self.blocks["b_vert"] = bp.readslice(顶点块大小)

        if self.顶点总数 > 0xFFFF:
            self.blocks["b_loop"] = bp.readslice(self.Loop总数*4)
        else:
            self.blocks["b_loop"] = bp.readslice(self.Loop总数*2)


    def read_vert_info(self, bx):
        leftflag, bn, rightflag = bx.readuint8(4), bx.readslice(bx.readuint32()), bx.readuint8(4)  # [11, 10, 2, 1]
        顶点大小 = bn.readuint16()
        描述字典 = {}
        for i in range(20):
            index, datatype = bn.readuint8(2)
            if index == 0xFF: continue
            描述字典[index] = datatype

        typeindc列表= sorted(描述字典.keys())  # sort()是列表的一个排序方法，直接修改原列表，没有返回值。# sorted()使用范围更广，不局限于列表，能接受所有迭代器，返回排好序的新列表。
        顶点描述列表 = [[index, 描述字典[index]] for index in typeindc列表]
        return 顶点大小, 顶点描述列表


    def read_mesh_info_version1(self, bx): # 32
        顶点总数, 未知05uint8 = bx.readuint32(), bx.readuint8(5)
        Loop总数, 未知04uint8 = bx.readuint32(), bx.readuint8(4)
        Face总数, 未知05uint8 = bx.readuint32(), bx.readuint8(5)
        附加标识, 未知02uint8 = bx.readuint32(), bx.readuint8(2)
        return 顶点总数, Loop总数, Face总数, 附加标识, 未知02uint8


    def read_mesh_info_version5(self, bx): # 38
        顶点总数, 未知09uint8 = bx.readuint32(), bx.readuint8(9)
        Loop总数, 未知04uint8 = bx.readuint32(), bx.readuint8(4)
        Face总数, 未知07uint8 = bx.readuint32(), bx.readuint8(7)
        附加标识, 未知02uint8 = bx.readuint32(), bx.readuint8(2)
        return 顶点总数, Loop总数, Face总数, 附加标识, 未知02uint8


    def read_head(self):
        bx = self.blocks["b_head"]
        self.head.magic = bx.readuint8(4)
        self.head.version = bx.readuint32()


    def read_vert(self):
        bx = self.blocks["b_vert"]
        for i in range(self.顶点总数):
            bn = bx.readslice(self.顶点大小)
            data48列表, data80列表, data00列表, data32列表 = [], [], [], []
            for index, datatype in self.顶点描述列表:
                match datatype:
                    case 48: data48列表.append(bn.readfloat32(3))
                    case 80: data80列表.append(bn.readuint8(4))
                    case  0: data00列表.append(bn.readuint8(4))
                    case 32: data32列表.append(bn.readfloat32(2))
                    # case  _: print(f"Anarchymeshfile::未知vert_data_type_{datatype}")
            self.顶点列表.append(data48列表[0])
            self.顶点UV列表.append(data32列表[0])


    def read_loop(self):
        by = self.blocks["b_loop"]
        if self.顶点总数 > 0xFFFF:
            self.顶点Loop列表 = [by.readuint32() for i in range(self.Loop总数)]
        else:
            self.顶点Loop列表 = [by.readuint16() for i in range(self.Loop总数)]





