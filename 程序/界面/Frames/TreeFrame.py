import tkinter as tk
import tkinter.ttk as ttk
import tkinter.filedialog
import os


import 文件

from .Tree右键Memu import TreeRightMemu
from .Tree选项Frame import TreeItemFrame

from .. import 缓存
import 程序.文件.bpformat.function_bpnumpy as by

class TreeFrame(ttk.Widget):
    def __init__(self, master=None, **kwargs):
        ttk.Widget.__init__(self, master, "ttk::frame", kwargs)
        self.bpdict = {}
        self.建立列()
        self.建立表()
        self.建立菜单()

        self.添加右键菜单()
        self.设置列()


    def 建立列(self, numcolumn=32):
        charlist = [f"#{i+1}" for i in range(numcolumn)]
        self.columns = charlist[:]
        self.columnstitle = charlist[:]


    def 建立表(self):
        xbar = ttk.Scrollbar(self, orient='horizontal')   # 水平
        xbar.pack(side="bottom", fill="x")                # must be top, bottom, left, or right

        ybar = ttk.Scrollbar(self, orient='vertical')     # 垂直
        ybar.pack(side="right", fill="y")

        self.tree = ttk.Treeview(self, columns=self.columns)
        self.itemframe = TreeItemFrame(self)
        self.itemframe.pack(side='top', fill='x')
        self.tree.pack(side='left', fill='both', expand=1)

        xbar.config(command=self.tree.xview)
        ybar.config(command=self.tree.yview)
        self.tree.config(xscrollcommand=xbar.set, yscrollcommand=ybar.set)


    def 建立菜单(self):
        self.右键菜单 = tk.Menu(self.tree, tearoff=False)
        self.tree.bind("<ButtonRelease-1>", self.显示状态) # 左键单击离开
        self.tree.bind("<ButtonRelease-3>", self.显示菜单) # 右键单击离开
        


    def 显示菜单(self, event):
        self.event = event  # <ButtonRelease event state=Mod1|Button3 num=3 x=221 y=36>  <class 'tkinter.Event'>
        self.右键菜单.post(event.x_root, event.y_root)


    def 显示状态(self, event):
        # self.右键菜单.post(event.x_root, event.y_root)
        item = self.tree.selection()[-1]
        self.itemframe.状态显示.delete(0, "end")
        if self.bpdict[item] is not None: 
            bp = self.bpdict[item].copy()
            self.itemframe.状态显示.insert(0, f"size: {bp.size()} , uint16: {bp.readuint16seek0()} , uint32: {bp.readuint32seek0()} ,")



    def 添加右键菜单(self):
        self.右键菜单.add_command(label="打开文件", command=self.打开浏览文件多选)

        self.右键菜单.add_separator() # 添加分割线
        self.右键菜单.add_command(label="删除行", command=self.删除行)
        self.右键菜单.add_command(label="复制行标题", command=self.复制行标题)
        self.右键菜单.add_separator() # 添加分割线
        self.解析菜单 = TreeRightMemu(self, self.右键菜单)
        self.右键菜单.add_separator() # 添加分割线
        # self.添加右键菜单2()
        # self.右键菜单.add_separator() # 添加分割线
        self.右键菜单.add_command(label="数据集to右边", command=self.数据集to右边)



    def 设置列(self):
        self.tree.column("#0", width=200, stretch=0)
        # for column, columnstitle in zip(self.columns[0:self.numhide], self.columnstitle[0:self.numhide]):
        #     self.tree.column(column, width=80, stretch=False, anchor='center')
        #     self.tree.heading(column, text=columnstitle)
        for column, columnstitle in zip(self.columns, self.columnstitle):
            self.tree.column(column, width=40, stretch=False, anchor='center')
            self.tree.heading(column, text=columnstitle)



    def 打开浏览文件单选(self):
        filepath = tkinter.filedialog.askopenfilename()
        if filepath == "": return ""
        self.打开文件(filepath)


    def 打开浏览文件多选(self):
        # filepaths = tkinter.filedialog.askopenfilenames(filetypes=[("text file", "*.txt"), ("all", "*.*")], )
        filepaths = tkinter.filedialog.askopenfilenames()
        for filepath in filepaths:
            if filepath == "": continue
            self.打开文件(filepath)



    def 打开文件(self, filepath):
        filename = os.path.basename(filepath)           # 带扩展名
        rootitem = self.insertblock("", text=filename, bp=文件.bpformat.bpnumpy.类().filepath(filepath))
        return rootitem


    def 关闭文件(self, rootitem):
        self.bpdict[rootitem].close()


    def insertblock(self, fitem="", index="end", text="", bp=None):
        item = self.tree.insert(fitem, index=index, text=text, values=[uint8 for uint8 in bp.readseek0(32)])
        self.bpdict[item] = bp
        return item


    def insertbytes(self, fitem="", index="end", text="", bp=None):
        item = self.tree.insert(fitem, index=index, text=text, values=[uint8 for uint8 in bp.readseek0(32)])
        self.bpdict[item] = bp
        return item


    def insertvalue(self, fitem="", index="end", text="", values=[]):
        item = self.tree.insert(fitem, index=index, text=text, values=values[0:32])
        self.bpdict[item] = None
        return item
    

    def itemblock(self, item="", text="", bp=None, values=[]):
        self.tree.item(item, text=text, values=[uint8 for uint8 in bp.readseek0(32)])
        self.bpdict[item] = bp
        return item


    def itemvalue(self, item="", text="", bp=None, values=[]):
        self.tree.item(item, text=text, values=values)
        self.bpdict[item] = None
        return item


    def 删除行(self):
        selection = self.tree.selection()
        rootitems = self.tree.get_children()
        for item in selection:
            if item in rootitems: self.关闭文件(item)
            try:
                self.tree.delete(item)
                self.bpdict.pop(item)
            except:
                pass


    def 复制行标题(self):
        item = self.tree.identify_row(self.event.y)
        chars = self.tree.item(item)["text"]
        缓存.根界面.clipboard_clear()
        缓存.根界面.clipboard_append(chars)



    def 数据集to右边(self):
        缓存.选择集列表 = []
        selectitems = self.tree.selection()
        for item in selectitems:
            if self.bpdict[item] is None: continue
            缓存.选择集列表.append(self.bpdict[item].copy())


    # def 添加右键菜单2(self):
    #     列样式菜单 = tk.Menu(self.右键菜单, tearoff=False)
    #     self.右键菜单.add_cascade(label = "列样式风格", menu=列样式菜单)
    #     列样式菜单.add_command(label="显示所有列", command=self.显示所有列)
    #     列样式菜单.add_command(label="隐藏参数列", command=self.隐藏参数列)
    #     列样式菜单.add_separator() # 添加分割线
    #     列样式菜单.add_command(label="所有列左对齐", command=self.所有列左对齐)
    #     列样式菜单.add_command(label="所有列居中", command=self.所有列居中)
    #     列样式菜单.add_command(label="所有列右对齐", command=self.所有列右对齐)
    #     列样式菜单.add_separator() # 添加分割线
    #     列样式菜单.add_command(label="列左对齐", command=self.列左对齐)
    #     列样式菜单.add_command(label="列居中", command=self.列居中)
    #     列样式菜单.add_command(label="列右对齐", command=self.列右对齐)



    # def 双击修改单元格(self):
    #     self.tree.bind('<Double-1>', self.双击修改单元格)




    # def 隐藏参数列(self):
    #     self.tree["displaycolumns"] = self.tree["columns"][self.numhide:]


    # def 显示所有列(self):
    #     self.tree["displaycolumns"] = self.tree["columns"]


    # def 隐藏左侧栏(self):
    #     self.tree["show"] = "headings"


    # def 显示左侧栏(self):
    #     self.tree["show"] = "tree headings"





    # def 插入子项(self):
    #     selectitems = self.tree.selection()
    #     for item in selectitems:
    #         bp, name, format = self.bpdict[item]
    #         for i in range(5): self.insertblock(item, bp=bp)
    #         self.tree.item(item, open=True)


    # def 更新行(self):
    #     selectitems = self.tree.selection()
    #     for item in selectitems:
    #         fitem = self.tree.parent(item)
    #         index = self.tree.index(item)
    #         self.resettreedata(fitem, index+1)


    # def 更新子行(self):
    #     selectitems = self.tree.selection()
    #     for item in selectitems:
    #         children = self.tree.get_children(item)
    #         self.resettreedata(item, len(children))


    # def resettreedata(self, fitem, length):
    #     bp, name, format = self.bpdict[fitem]
    #     children = self.tree.get_children(item)[0: length]
    #     for item in children:
    #         name, format = self.tree.item(item)["values"][0: 2]


    # def 双击修改单元格(self, event):
    #     # self.tree.identify_region(event.x, event.y) # cell, tree, heading
    #     item = self.tree.identify_row(event.y)
    #     column = self.tree.identify_column(event.x)
    #     if column not in ["#1", "#2"]: return ""
    #     x, y, width, height = self.tree.bbox(item=item, column=column) # #0 代表行heading所在的列
    #     默认值 = self.tree.item(item)["values"][int(column[1:])-1]
    #     输入框 = ttk.Entry(self.tree, width=width//7)
    #     输入框.insert(index=0, string=默认值)
    #     输入框.place(x=x, y=y)
    #     def 输入框内容保存到单元格(event):
    #         if item not in self.tree.get_children():
    #             self.tree.item(item, text=输入框.get())
    #             self.tree.set(item=item, column=column, value=输入框.get())
    #         输入框.destroy()
    #     输入框.bind("<Return>", 输入框内容保存到单元格)
    #     输入框.focus()









    # def 所有列左对齐(self):
    #     columns, displaycolumns = self.tree["columns"], self.tree["displaycolumns"]
    #     self.tree["displaycolumns"] = columns
    #     for column in columns:
    #         self.tree.column(column, anchor='w') # must be n, ne, e, se, s, sw, w, nw, or center
    #     self.tree["displaycolumns"] = displaycolumns

    # def 所有列居中(self):
    #     columns, displaycolumns = self.tree["columns"], self.tree["displaycolumns"]
    #     self.tree["displaycolumns"] = columns
    #     for column in columns:
    #         self.tree.column(column, anchor='center')
    #     self.tree["displaycolumns"] = displaycolumns


    # def 所有列右对齐(self):
    #     columns, displaycolumns = self.tree["columns"], self.tree["displaycolumns"]
    #     self.tree["displaycolumns"] = columns
    #     for column in columns:
    #         self.tree.column(column, anchor='e')
    #     self.tree["displaycolumns"] = displaycolumns


    # def 列左对齐(self):
    #     column = self.tree.identify_column(self.event.x)
    #     self.tree.column(column, anchor='w')

    # def 列居中(self):
    #     column = self.tree.identify_column(self.event.x)
    #     self.tree.column(column, anchor='center')

    # def 列右对齐(self):
    #     column = self.tree.identify_column(self.event.x)
    #     self.tree.column(column, anchor='e')





    def 打开默认文件(self, filepath=""):
        缓存.选择集列表 = []
        filepaths = [
            # r"E:\Program_StructFiles\GuJianQT3\asset\characters\actress1\models\actress1_default_bodya.model",
            # # r"E:\Program_StructFiles\GuJianQT3\asset\models\common\building\common_building_bridge_01.model",
            # # r"E:\Program_StructFiles\GuJianQT3\asset\maps\m01\elems.xxx",
            # r"E:\Program_StructFiles\GuJianQT3\asset\maps\m01\TerrainData\Config.vtc",
            # r"E:\Program_StructFiles\GuJianQT3\asset\maps\m01\TerrainData\ocm\Sector_00_00.ocm",
            # r"E:\Program_StructFiles\GuJianQT3\asset\maps\m01\TerrainData\ocm\Sector_00_01.ocm",
            # r"E:\Program_StructFiles\GuJianQT3\asset\maps\m01\TerrainData\Physics\Sector_00_00.hkt",
            # r"E:\Program_StructFiles\GuJianQT3\asset\maps\m01\TerrainData\Physics\Sector_00_01.hkt",
            # r"E:\Program_StructFiles\GuJianQT3\asset\maps\m01\TerrainData\Sectors\Sector_00_00.hmap",
            # r"E:\Program_StructFiles\GuJianQT3\asset\maps\m01\TerrainData\Sectors\Sector_00_00.mesh",
            # r"E:\Program_StructFiles\GuJianQT3\asset\maps\m01\TerrainData\Sectors\Sector_00_01.hmap",
            # r"E:\Program_StructFiles\GuJianQT3\asset\maps\m01\TerrainData\Sectors\Sector_00_01.mesh",
            # # r"E:\Program_StructFiles\GuJianQT3\asset\maps\m01\TerrainData\AuxiliaryTextures\Normalmap_00_00.dds",
            # # r"E:\Program_StructFiles\GuJianQT3\asset\maps\m01\TerrainData\AuxiliaryTextures\Normalmap_00_01.dds",
            # # r"E:\Program_StructFiles\GuJianQT3\asset\maps\m01\TerrainData\AuxiliaryTextures\Weightmap0_00_00.dds",
            # # r"E:\Program_StructFiles\GuJianQT3\asset\maps\m01\TerrainData\AuxiliaryTextures\Weightmap0_00_01.dds",
            # r"E:\Program_StructFiles\GuJianQT2\Models\Axis.MODEL",
            # r"E:\Program_StructFiles\GuJianQT2\Models\Bone.MODEL",
            # r"E:\Program_StructFiles\GuJianQT2\Models\MagicBall.model",
            # r"E:\Program_StructFiles\GuJianQT2\Models\MissingModel.model",
            # r"E:\Program_StructFiles\GuJianQT2\Models\Sphere.model",
            # r"E:\Program_StructFiles\GuJianQT2\Models\plant\plant_002.vmesh",
            # r"E:\Program_StructFiles\GuJianQT2\Maps\n01p1\n01p1Navwall.vmesh",
            # r"E:\Program_StructFiles\GuJianQT2\Maps\n01p1\TerrainData\Config.vtc", # mesh version 3
            # r"E:\Program_StructFiles\GuJianQT2\Maps\n01p1\TerrainData\Sectors\Sector_00_00.hkp",
            # r"E:\Program_StructFiles\GuJianQT2\Maps\n01p1\TerrainData\Sectors\Sector_00_00.hmap",
            # r"E:\Program_StructFiles\GuJianQT2\Maps\n01p1\TerrainData\Sectors\Sector_00_00.mesh",
            # r"E:\Program_StructFiles\GuJianQT2\Models\cave\cave_519.vmesh",
            # r"E:\Program_StructFiles\GuJianQT3\asset\navmesh\m01\1010.navmesh",
            r"D:\Program_checkpoints\Gf_style2.ckpt",

        ]

        for filepath in filepaths:
            rootitem = self.打开文件(filepath)
            # 缓存.选择集列表.append(self.bpdict[rootitem].copy())






        # 缓存.选择集列表 = []
        # # filepath = r"E:\struct_files\asset\characters\actress1\anims\32I_c.hka"
        # # filepath = r"E:\Program_struct_files\Gujian3\asset\characters\actress1\anims\32I_c.hka"
        # # filepath = r"E:\struct_files\Avatar_Kiana_C4_YN\Avatar_Kiana_C4_YN__out_anim_Avatar_Kiana_C4_Ani_Attack_QTE_AS.hkx"
        # filepath = r"F:\Avatar_Kiana_C4_YN\Avatar_Kiana_C4_YN.fbx"
        # filepath = r"E:\Program_StructFiles\GuJianQT3\asset\characters\actress1\models\actress1_default_bodya.model"
        # rootitem = self.打开文件(filepath)
        # 缓存.选择集列表.append(self.bpdict[rootitem].copy())
        # self.tree.selection_set(rootitem)
        # self.解析菜单.menuhka.文件解析()
        # self.解析菜单.menuhka.dataposition()
        # self.解析菜单.menuhkx.文件解析()
        # self.解析菜单.menuhkx.dataposition()
        # self.解析菜单.menufbx.文件解析()