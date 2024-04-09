# https://github.com/WolvenKit/WolvenKit/blob/main/WolvenKit.Common/RED3/SRT/Srtfile.cs




class Object: pass
class 类:
    def __init__(self, bp):
        self.filepath = ""
        self.head = Object()
        self.blocks = {}
        self.blocks_mesh = []
        self.材质列表 = []
        self.材质列表1 = []
        self.网格列表 = []
        self.网格列表1 = []
        self.read_blocks(bp)
        self.read_head()
        self.read_mesh_data()


    def read_blocks(self, bp):
        self.filepath = bp.rdpath

        self.blocks["b_head"] = bp.readslice(20)
        self.blocks["b_bound"] = bp.readslice(24)
        self.blocks["b_lod"] = bp.readslice(20)
        self.blocks["b_wind"] = bp.readslice(1376)
        self.blocks["b_chartable"] = self.block_chartable(bp)
        self.blocks["b_collision_object"] = bp.readslice(4 + bp.readuint32seek0()*36)
        self.blocks["b_numbillboard"] = self.block_numbillboard(bp)
        self.blocks["b_vertical_billboard"] = self.block_vertical_billboard(bp)
        self.blocks["b_horizontal_billboard"] = bp.readslice(84)
        self.blocks["b_custom_data"] = bp.readslice(20)
        self.blocks["b_render_data"] = self.block_render_data(bp)

        self.blocks["b_mbus"] = self.block_mbus(bp)

        字符名称列表 = self.read_chartable()
        区块信息列表 = self.read_render_data(字符名称列表)
        网格信息列表 = self.read_mbus()
        for i, [渲染块索引, vert个数, loop个数] in enumerate(网格信息列表):
            贴图名称列表, vert大小 = 区块信息列表[渲染块索引]
            sizevert = vert个数 * vert大小
            sizeloop = loop个数 * 2
            sizezero = sizeloop % 4
            vert = bp.readslice(sizevert)
            loop = bp.readslice(sizeloop)
            zero = bp.readslice(sizezero)
            self.blocks[f"b_vert_{i}"] = vert
            self.blocks[f"b_loop_{i}"] = loop
            self.blocks[f"b_zero_{i}"] = zero
            self.blocks_mesh.append([vert, loop, zero, [vert个数, vert大小, loop个数, 贴图名称列表]])
        self.blocks["b_endof"] = bp.readremainslice()


    def block_chartable(self, bp):
        size1 = 4 + bp.readuint32seek0()*8
        bx = bp.readsliceseek0(size1)
        size2 = 0
        for i in range(bx.readuint32()): size2 += bx.readuint32(2)[1] # valuezero, valuechar
        return bp.readslice(size1+size2)

    def read_chartable(self):
        bp = self.blocks["b_chartable"]

        numchars = []
        for i in range(bp.readuint32()): numchars.append(bp.readuint32(2)[1])

        字符名称列表 = []
        for numchar in numchars: 字符名称列表.append(bp.readchar(numchar))
        return 字符名称列表

    def block_numbillboard(self, bp):
        width, top_position, bottom_position, numbillboard = bp.readuint32seek0(4)
        if numbillboard == 0:
            return bp.readslice(16)
        else:
            size1 = 16 + numbillboard*16 + numbillboard
            if size1 % 4 == 0:
                pad0 = 0
            else:
                pad0 = 4 - size1 % 4
            size1 += pad0
            return bp.readslice(size1)

    def block_vertical_billboard(self, bp):
        numvert, numloop = bp.readuint32seek0(2)
        size = 8 + numvert*8 + numloop*2
        return bp.readslice(size)

    def block_render_data(self, bp):
        渲染块个数, valuezero, 存在渲染块镜像, 存在单独块镜像 = bp.readuint32seek0(4)
        size = 16 + 渲染块个数*720 + 720
        if 存在渲染块镜像: size += 渲染块个数*720
        if 存在单独块镜像: size += 720
        return bp.readslice(size)

    def read_render_data(self, 字符名称列表):
        bp = self.blocks["b_render_data"]
        渲染块个数, valuezero, 存在渲染块镜像, 存在单独块镜像 = bp.readuint32(4)
        渲染块列表 = []
        for i in range(渲染块个数): 渲染块列表.append(bp.readslice(720))

        区块信息列表 = []
        for bx in 渲染块列表:
            贴图名称列表 = []
            for i in range(8):
                index = bx.readuint32(2)[0]
                贴图名称列表.append(字符名称列表[index])

            bx.readslice(636)
            区块信息列表.append([贴图名称列表, bx.readuint32()]) # vert大小 # seek 700

        return 区块信息列表

    def block_mbus(self, bp):
        bx = bp.readsliceseek0(1024)
        分类总数 = bx.readuint32()
        size1 = 4 + 分类总数*24

        网格总数 = 0
        for i in range(分类总数): 网格总数 += bx.readuint32(6)[0]
        size2 = 网格总数*40
        return bp.readslice(size1+size2)

    def read_mbus(self):
        bp = self.blocks["b_mbus"]
        网格总数 = 0
        for i in range(bp.readuint32()):
            网格总数 += bp.readuint32(6)[0]

        网格信息列表 = []
        for j in range(网格总数):
            values = bp.readuint32(10)
            渲染块索引, vert个数, loop个数 = values[2], values[3], values[6]
            网格信息列表.append([渲染块索引, vert个数, loop个数])  # [渲染块索引, vert左值, vert个数, loop左值, loop个数, 材质索引]

        return 网格信息列表


    def read_head(self):
        bx = self.blocks["b_head"]
        self.head.magic = bx.readchar(10) #  "SRT 07.0.0"




    def read_mesh_data(self):
        for i, [vert, loop, zero, [vert个数, vert大小, loop个数, 贴图名称列表]] in enumerate(self.blocks_mesh):
            贴图列表 = 贴图名称列表[0:2] + 贴图名称列表[4:5]
            贴图列表1 = 贴图名称列表[2:4]
            if 贴图列表1 == []: 贴图列表1 = 贴图列表
            self.材质列表.append(["", 贴图列表]) # 材质名称, 贴图名称列表
            self.材质列表1.append(["", 贴图列表1]) # 材质名称, 贴图名称列表
            [
            顶点列表,
            顶点列表1,
            顶点UV列表]  = self.read_mesh_vert(vert, vert个数, vert大小)
            顶点Loop列表 = self.read_mesh_loop(loop, loop个数)
            self.网格列表.append([顶点列表, 顶点UV列表, 顶点Loop列表, i]) # 材质索引
            self.网格列表1.append([顶点列表1, 顶点UV列表, 顶点Loop列表, i]) # 材质索引


    def read_mesh_vert(self, vert, vert个数, vert大小):
        顶点列表 = []
        顶点列表1 = []
        顶点UV列表 = []
        for i in range(vert个数):
            with vert.readslice(vert大小) as bp:
                values = bp.readfloat16(8)
            positon = values[0:3]
            uv      = values[4:6]
            positon1 = values[3:4] + values[6:8]
            顶点列表.append(positon)
            顶点列表1.append(positon1)
            顶点UV列表.append(uv)

        return 顶点列表, 顶点列表1, 顶点UV列表


    def read_mesh_loop(self, loop, loop个数):
        return [loop.readuint16() for i in range(loop个数)] # 顶点Loop列表




