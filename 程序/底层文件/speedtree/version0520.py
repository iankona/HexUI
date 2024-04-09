
class Object: pass
class 类:
    def __init__(self, bp):
        self.filepath = ""
        self.head = Object()
        self.blocks = {}
        self.材质列表 = []

        self.树干网格列表 = []
        self.树枝网格列表 = []
        self.树叶网格列表 = []
        # 中途需要计算各自块大小，所有分开，不合并
        self.树干网格列表1 = []
        self.树枝网格列表1 = []
        self.树叶网格列表1 = []

        self.read_blocks(bp)
        self.read_head()
        self.read_material()
        self.read_mesh_data()

        self.细枝网格列表 = []
        self.细枝网格列表1 = []
        # self.read_mesh_data_细枝()


    def read_blocks(self, bp):
        self.filepath = bp.rdpath
        self.blocks_section0(bp)
        self.blocks_section1_wind(bp)
        self.blocks_section2(bp)

        树干numvertlist = self.read_mesh_info_type1(self.blocks["b_mesh_info_树干"])
        树枝numvertlist = self.read_mesh_info_type1(self.blocks["b_mesh_info_树枝"])
        树叶numvertlist = self.read_mesh_info_type1(self.blocks["b_mesh_info_树叶"])

        树干mbuslist    = self.read_mesh_mbus_type1(self.blocks["b_mbus"], 树干numvertlist)
        树枝mbuslist    = self.read_mesh_mbus_type1(self.blocks["b_mbus"], 树枝numvertlist)
        树叶mbuslist    = self.read_mesh_mbus_type1(self.blocks["b_mbus"], 树叶numvertlist)

        细枝nummeshlist = self.read_mesh_info_type2(self.blocks["b_mesh_info_细枝"])
        细枝mbuslist    = self.read_mesh_mbus_type2(self.blocks["b_mbus"], 细枝nummeshlist)
        size1, size2, size3 = self.计算_data_划分大小(细枝mbuslist)
        data = self.blocks["b_data"]
        self.blocks["b_vert"] = data.readslice(size1)
        self.blocks["b_xizi"] = data.readslice(size2)
        self.blocks["b_mowi"] = data.readslice(size3)

        vert = self.blocks["b_vert"]
        树干element, 树枝element, 树叶element, 树干size, 树枝size, 树叶size = self.计算_树干树枝树叶vert大小(vert.size(), 树干numvertlist, 树枝numvertlist, 树叶numvertlist)
        self.blocks["b_vert_树干"] = vert.readslice(树干size)
        self.blocks["b_vert_树枝"] = vert.readslice(树枝size)
        self.blocks["b_vert_树叶"] = vert.readslice(树叶size)

        loop = self.blocks["b_loop"]
        树干size, 树枝size, 树叶size, 树干numlooplist, 树枝numlooplist, 树叶numlooplist = self.计算_树干树枝树叶loop大小(树干mbuslist, 树枝mbuslist, 树叶mbuslist)
        self.blocks["b_loop_树干"] = loop.readslice(树干size)
        self.blocks["b_loop_树枝"] = loop.readslice(树枝size)
        self.blocks["b_loop_树叶"] = loop.readslice(树叶size)

        self.树干element = 树干element
        self.树枝element = 树枝element
        self.树叶element = 树叶element

        self.树干mbuslist = 树干mbuslist
        self.树枝mbuslist = 树枝mbuslist
        self.树叶mbuslist = 树叶mbuslist

        self.树干numvertlist = 树干numvertlist
        self.树枝numvertlist = 树枝numvertlist
        self.树叶numvertlist = 树叶numvertlist

        self.树干numlooplist = 树干numlooplist
        self.树枝numlooplist = 树枝numlooplist
        self.树叶numlooplist = 树叶numlooplist


    def blocks_section0(self, bp):
        self.blocks["b_head"] = bp.readslice(36)
        self.blocks["b_bound"] = bp.readslice(24)
        self.blocks["b_lod"] = bp.readslice(20)
        self.blocks["b_collision_object"] = bp.readslice(4 + bp.readuint32seek0()*284)

    def blocks_section1_wind(self, bp):
        self.blocks["b_wind"] = bp.readslice(216)

    def blocks_section2(self, bp):
        self.blocks["b_material"] = bp.readslice(4 + bp.readuint32seek0()*1648)
        self.blocks["b_mesh_info_树干"] = self.block_mesh_info_type1(bp)
        self.blocks["b_mesh_info_树枝"] = self.block_mesh_info_type1(bp)
        self.blocks["b_mesh_info_树叶"] = self.block_mesh_info_type1(bp)
        self.blocks["b_mesh_info_细枝"] = self.block_mesh_info_type2(bp)
        self.blocks["b_custom_data"] = bp.readslice(16)
        self.blocks["b_block_info"] = bp.readslice(112)
        size1, size2, size3, size4 = self.read_block_info()
        self.blocks["b_data"] = bp.readslice(size1)
        self.blocks["b_mbus"] = bp.readslice(size2)
        self.blocks["b_unkn"] = bp.readslice(size3)
        self.blocks["b_loop"] = bp.readslice(size4)
        self.blocks["b_endof"] = bp.readremainslice()


    def block_mesh_info_type1(self, bp):
        bx = bp.readsliceseek0(4+3*16)
        if bx.readuint32seek0() == 0:
            size0 = 4
        else:
            size0 = 4
            for i in range(bx.readuint32()):
                if bx.readuint32seek0() == 0:
                    size0 += 4
                    bx.readslice(4)
                else:
                    size0 += 16
                    bx.readslice(16)

        return bp.readslice(size0)

    def block_mesh_info_type2(self, bp):
        size0 = 4 + bp.readuint32seek0()*8
        return bp.readslice(size0)


    def read_block_info(self):
        bp = self.blocks["b_block_info"]
        bp.seek(92)
        values = bp.readuint32seek0(5)
        size1 = values[0] * 4
        size2 = values[1] * 4
        size3 = values[2] * 1
        size4 = values[3] * 2
        return size1, size2, size3, size4


    def read_mesh_info_type1(self, bp):
        类型numvertlist = []
        for i in range(bp.readuint32()):
            顶点个数 = bp.readuint32()
            if 顶点个数 == 0:
                continue
            else:
                valuezero, valuefloat, 网格个数 = bp.readuint32(3)
                类型numvertlist.append([顶点个数, valuezero, valuefloat, 网格个数])
        return 类型numvertlist


    def read_mesh_mbus_type1(self, bp, 类型numvertlist):
        类型mbuslist = []
        for 顶点个数, valuezero, valuefloat, 网格个数 in 类型numvertlist:
            mbuslist = [bp.readuint32(5) for i in range(网格个数)]  # 材质索引, loop左值, loop个数, vert左值, vert个数
            类型mbuslist.append(mbuslist)
        return 类型mbuslist


    def read_mesh_info_type2(self, bp):
        类型nummeshlist = []
        for j in range(bp.readuint32()): 类型nummeshlist.append([bp.readuint32(), bp.readfloat32()]) # valuemesh, valuefloat
        return 类型nummeshlist


    def read_mesh_mbus_type2(self, bp, 类型nummeshlist):
        类型mbuslist = []
        for 网格个数, valuefloat in 类型nummeshlist:
            mbuslist = [bp.readuint32(5) for i in range(网格个数)]
            类型mbuslist.append(mbuslist)
        return 类型mbuslist


    def 计算_data_划分大小(self, 细枝mbuslist):
        data = self.blocks["b_data"]
        size2 = 0
        for mbuslist in 细枝mbuslist:
            for 材质索引, vert左值, vert个数, loop左值, loop个数 in mbuslist: size2 += vert个数*84 #

        size1 = data.size() - size2 - 96
        size3 = 96
        return size1, size2, size3


    def 计算_树干树枝树叶vert大小(self, blocksize, 树干numvertlist, 树枝numvertlist, 树叶numvertlist):
        树干numvert = 0
        for numvert, index, valuefloat, valuemesh in 树干numvertlist: 树干numvert += numvert
        树枝numvert = 0
        for numvert, index, valuefloat, valuemesh in 树枝numvertlist: 树枝numvert += numvert
        树叶numvert = 0
        for numvert, index, valuefloat, valuemesh in 树叶numvertlist: 树叶numvert += numvert

        for 树干element in [32, 40, 44]:
            for 树枝element in [32, 40, 44]:
                for 树叶element in [32, 40, 44]:
                    树干size = 树干numvert * 树干element
                    树枝size = 树枝numvert * 树枝element
                    树叶size = 树叶numvert * 树叶element
                    if 树干size + 树枝size + 树叶size == blocksize: return 树干element, 树枝element, 树叶element, 树干size, 树枝size, 树叶size
        return [0, 0, 0, 0, 0, 0]


    def 计算_树干树枝树叶loop大小(self, 树干mbuslist, 树枝mbuslist, 树叶mbuslist):
        def 函数(类型mbuslist):
            类型numloop = 0
            类型numlooplist = []
            for mbuslist in 类型mbuslist:
                numloop = 0
                for 材质索引, loop左值, loop个数, vert左值, vert个数 in mbuslist: numloop += loop个数
                类型numloop += numloop
                类型numlooplist.append(numloop)
            return 类型numloop, 类型numlooplist

        树干numloop, 树干numlooplist = 函数(树干mbuslist)
        树枝numloop, 树枝numlooplist = 函数(树枝mbuslist)
        树叶numloop, 树叶numlooplist = 函数(树叶mbuslist)
        return 树干numloop*2, 树枝numloop*2, 树叶numloop*2, 树干numlooplist, 树枝numlooplist, 树叶numlooplist


    def read_head(self):
        bp = self.blocks["b_head"]
        self.head.magic = bp.readchar(10) #  "SRT 05.2.0"


    def read_material(self):
        bx = self.blocks["b_material"]
        材质数量 = bx.readuint32()
        for i in range(材质数量):
            材质块 = bx.readslice(1648)
            贴图名称列表 = []
            for j in range(5):
                贴图名称 = 材质块.readchar(256)
                if 贴图名称 == "": continue
                贴图名称列表.append(贴图名称)
            self.材质列表.append(["", 贴图名称列表])


    def read_mesh_data(self):
        [
        self.树干网格列表,
        self.树干网格列表1,
        ] = self.read_data_type1(self.树干element, self.树干numvertlist, self.blocks["b_vert_树干"], self.树干numlooplist, self.blocks["b_loop_树干"], self.树干mbuslist)

        [
        self.树枝网格列表,
        self.树枝网格列表1,
        ] = self.read_data_type1(self.树枝element, self.树枝numvertlist, self.blocks["b_vert_树枝"], self.树枝numlooplist, self.blocks["b_loop_树枝"], self.树枝mbuslist)

        [
        self.树叶网格列表,
        self.树叶网格列表1,
        ] = self.read_data_type1(self.树叶element, self.树叶numvertlist, self.blocks["b_vert_树叶"], self.树叶numlooplist, self.blocks["b_loop_树叶"], self.树叶mbuslist)


    def read_data_type1(self, 类型element, 类型numvertlist, vert, 类型numlooplist, loop, 类型mbuslist):
        类型网格列表 = []
        类型网格列表1 = []
        for [顶点个数, valuezero, valuefloat, 网格个数], Loop个数, mbuslist in zip(类型numvertlist, 类型numlooplist, 类型mbuslist):
            with vert.readslice(顶点个数*类型element) as bp:
                顶点列表 = [bp.readfloat32(3) for i in range(顶点个数)]
                顶点列表1 = [bp.readfloat32(3) for i in range(顶点个数)]
                顶点UV列表 = [bp.readfloat32(2) for i in range(顶点个数)]
            顶点Loop列表 = loop.readuint16(Loop个数)
            类型网格列表.append([顶点列表, 顶点UV列表, 顶点Loop列表, mbuslist])
            类型网格列表1.append([顶点列表1, 顶点UV列表, 顶点Loop列表, mbuslist])
        return 类型网格列表, 类型网格列表1

























