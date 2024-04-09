import 界面文件.bcontext.bsfunction as bs
import 底层文件.bpformat.byfunction as by
import 底层文件

from . import tkinter_file


class 类:
    def __init__(self, frametree, 菜单):
        self.frametree = frametree
        子菜单 = tk.Menu(菜单, tearoff=False)
        菜单.add_cascade(label = "SpeedTree", menu=子菜单)
        子菜单.add_command(label="文件解析", command=self.文件解析)
        子菜单.add_command(label="文件分块", command=self.文件分块)

        # 子菜单.add_command(label="dataposition", command=self.dataposition)
        # 子菜单.add_separator() # 添加分割线


    def context(self, item=""):
        if item == "": item = self.frametree.tree.selection()[0]
        bs.bscontext.frametree = self.frametree
        bs.bscontext.item = item
        global by
        by = func.context(self.frametree.bpdict[item].copy()) # bp = self.frametree.bpdict[item]
        return item


    def 文件解析(self):
        for item in self.frametree.tree.selection():
            self.context(item)
            bp = by.filepath(by.rdpath()).bp
            srtfile = 文件.srt.类(bp)
            for char, value in srtfile.blocks.items():
                bs.insertblock(text=char, bp=value.copy())
     


    def 文件分块(self):
        for item in self.frametree.tree.selection():
            self.context(item)
            if by.readcharseek0(10) == "SRT 07.0.0": self.__文件分块0700__()


    def __文件分块0700__(self):
        bs.insertblock(text="b_head", bp=by.readslice(20).bp)
        bs.insertblock(text="b_bound", bp=by.readslice(24).bp)
        bs.insertblock(text="b_lod", bp=by.readslice(20).bp)
        bs.insertblock(text="b_wind", bp=by.readslice(1376).bp)

        with by.readsliceseek0(1024):
            numenum = by.readuint32()
            numchar_list = []
            for i in range(numenum): numchar_list.append(by.readuint32(2)[1])
            size = 4 + 8*numenum
            for numchar in numchar_list: size += numchar
            chartable = []
            for numchar in numchar_list: chartable.append(by.readchar(numchar))
            # for char in chartable: print(char)

        with bs.insertblock(text="b_chartable", bp=by.readsliceseek0(size).bp), by.readslice(size):
            bs.insertblock(text="", bp=by.readslice(4).bp)
            bs.insertblock(text="", bp=by.readslice(8*numenum).bp)
            for i, numchar in enumerate(numchar_list): 
                bs.insertblock(text=f"{i}, {by.readcharseek0(numchar)}", bp=by.readslice(numchar).bp)

        with by.readsliceseek0(32):
            # num_collision_object = by.readuint32seek0()
            # size = 4 + num_collision_object*36
            size = 4 + by.readuint32seek0()*36
        bs.insertblock(text="b_collision_object", bp=by.readslice(size).bp)

        with by.readsliceseek0(32):
            width, top_position, bottom_position, numbillboard = by.readuint32seek0(4)
            if numbillboard == 0:
                size = 16
            else:
                size = 16 + numbillboard*16 + numbillboard
                if size % 4 == 0:
                    pad0 = 0
                else:
                    pad0 = 4 - size % 4
                size += pad0
        bs.insertblock(text="b_numbillboard", bp=by.readslice(size).bp)

        with by.readsliceseek0(32):
            numvert, numloop = by.readuint32seek0(2)
            size = 8 + numvert*8 + numloop*2
        bs.insertblock(text="b_vertical_billboard", bp=by.readslice(size).bp)

        bs.insertblock(text="b_horizontal_billboard", bp=by.readslice(84).bp)

        bs.insertblock(text="b_custom_data", bp=by.readslice(20).bp)

        with by.readsliceseek0(32):
            渲染块个数, valuezero, 存在渲染块镜像, 存在单独块镜像 = by.readuint32seek0(4)
            size = 16 + 渲染块个数*720 + 720
            if 存在渲染块镜像: size += 渲染块个数*720
            if 存在单独块镜像: size += 720        
        with bs.insertblock(text="b_render_data", bp=by.readsliceseek0(size).bp), by.readslice(size):
            bs.insertblock(text=f"", bp=by.readslice(16).bp)
            for i in range(渲染块个数):
                with bs.insertblock(text=f"{i}_渲染块", bp=by.readsliceseek0(720).bp), by.readslice(720):
                    self.__渲染块划分__(chartable)
            for i in range(渲染块个数):
                bs.insertblock(text=f"{i}_渲染块镜像", bp=by.readslice(720).bp)
            
            bs.insertblock(text=f"单独块", bp=by.readslice(720).bp)
            if 存在单独块镜像: bs.insertblock(text=f"单独块镜像", bp=by.readslice(720).bp)

        with by.readsliceseek0(1024):
            分类总数 = by.readuint32()
            网格总数 = 0
            for i in range(分类总数): 网格总数 += by.readuint32(6)[0]
            网格信息列表 = [] 
            for j in range(网格总数): 网格信息列表.append(by.readuint32(10)) 
            size1 = 4 
            size2 = 分类总数*24
            size3 = 网格总数*40
            size = size1 + size2 + size3
        with bs.insertblock(text="b_mbus", bp=by.readsliceseek0(size).bp), by.readslice(size):
            bs.insertblock(text=f"", bp=by.readslice(4).bp)
            bs.insertblock(text=f"", bp=by.readslice(分类总数*24).bp)
            for i in range(网格总数): # 渲染块索引, vert个数, loop个数 = values[2], values[3], values[6]
                bs.insertblock(text=f"{i}, {by.readuint32seek0(10)}", bp=by.readslice(40).bp)

        bs.insertblock(text="余下", bp=by.readremainslice().bp) 

    def __渲染块划分__(self, chartable):
        for i in range(8):
            index1, index2 = by.readuint32seek0(2)
            bs.insertblock(text=f"{chartable[index1]}, {chartable[index2]}", bp=by.readslice(8).bp)
        bs.insertblock(text="余下", bp=by.readremainslice().bp) 


    def __文件分块0700__(self):
        bs.insertblock(text="b_head", bp=by.readslice(36).bp)
        bs.insertblock(text="b_bound", bp=by.readslice(24).bp)
        bs.insertblock(text="b_lod", bp=by.readslice(20).bp)
        with by.readsliceseek0(4):
            numblock = by.readuint32()
            size = 4 + numblock*284
        with bs.insertblock(text="b_collision_object", bp=by.readsliceseek0(size).bp), by.readslice(size):
            bs.insertblock(text="", bp=by.readslice(size).bp)
            for i in range(numblock): bs.insertblock(text="块_{i}_284", bp=by.readslice(size).bp)
        
        bs.insertblock(text="b_wind", bp=by.readslice(284).bp)

        with by.readsliceseek0(4):
            numblock = by.readuint32()
            size = 4 + numblock*1648
        with bs.insertblock(text="b_material", bp=by.readsliceseek0(size).bp), by.readslice(size):
            bs.insertblock(text="", bp=by.readslice(size).bp)
            for i in range(numblock): bs.insertblock(text="块_{i}_1648", bp=by.readslice(size).bp)