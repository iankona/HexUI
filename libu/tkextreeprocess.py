import tkinter as tk
import tkinter.ttk as ttk
import tkinter.filedialog


try:
    from . import tkfunction as UI
except:
    import tkfunction as UI


class TreeProcess():
    def __init__(self):
        # 主界面
        with UI.PanedWindowHPack(), UI.Title("文件解析"):
            with UI.FramePack(width=460, pack_propagate=0, side="top"): 
                with UI.FramePack(side="top", fill="x"):
                    UI.LabelGrid(text="索引列表：", row=0, column=0), UI.EntryGrid(width=42, row=0, column=1), UI.ButtonGrid(text="索引划分", width=10, row=0, column=2)
                    UI.LabelGrid(text="大小列表：", row=1, column=0), UI.EntryGrid(width=42, row=1, column=1), UI.ButtonGrid(text="大小划分", width=10, row=1, column=2)
                    UI.LabelGrid(text="状态显示：", row=2, column=0), UI.EntryGrid(width=42, row=2, column=1), UI.ButtonGrid(text="层级划分", width=10, row=2, column=2)
                UI.ScrollBarVPack(side="right", fill="y")
                UI.ScrollBarHPack(side="bottom", fill="x")
                UI.TreeViewPack(side="top", fill="both", expand=1)
            with UI.FramePack(side="top", fill="both"):
                with UI.FramePack( side="top", fill="x"):
                    UI.LabelGrid(text="偏移大小：", row=0, column=0), UI.EntryGrid(width=15, row=0, column=1), UI.LabelGrid(text="表格行数：", row=0, column=2), UI.EntryGrid(width=15, row=0, column=3), UI.ButtonGrid(text="开启弹窗", width=10, row=0, column=4)
                    UI.LabelGrid(text="单区大小：", row=1, column=0), UI.EntryGrid(width=15, row=1, column=1), UI.LabelGrid(text="每行列数：", row=1, column=2), UI.EntryGrid(width=15, row=1, column=3), UI.ButtonGrid(text="切换端序", width=10, row=1, column=4)
                    UI.LabelGrid(text="解析格式：", row=2, column=0), UI.EntryGrid(width=15, row=2, column=1), UI.LabelGrid(text="显示留行：", row=2, column=2), UI.EntryGrid(width=15, row=2, column=3), UI.ButtonGrid(text="当前小端", width=10, row=2, column=4)
                UI.ScrollBarVPack(side="right", fill="y")
                UI.ScrollBarHPack(side="bottom", fill="x")
                UI.TreeViewPack(side="top", fill="both", expand=1)

        # 弹窗
        with UI.Toplevel(geometry="500x205+100+50", overrideredirect=True): # width, height = 500, 205 # +23
            UI.LabelGrid(text="  uint8:", row=0, column=0), UI.EntryGrid(width=15, row=0, column=1), UI.LabelGrid(text="   int8:", row=0, column=2), UI.EntryGrid(width=15, row=0, column=3), UI.LabelGrid(text="   char:", row=0, column=0), UI.EntryGrid(width=15, row=0, column=1), UI.LabelGrid(text="  uint8:", row=0, column=0), UI.EntryGrid(width=15, row=0, column=1) 
            UI.LabelGrid(text=" uint16:", row=1, column=0), UI.EntryGrid(width=15, row=1, column=1), UI.LabelGrid(text="  int16:", row=1, column=2), UI.EntryGrid(width=15, row=1, column=3), UI.LabelGrid(text="   utf8:", row=0, column=0), UI.EntryGrid(width=15, row=0, column=1), UI.LabelGrid(text="  uint8:", row=0, column=0), UI.EntryGrid(width=15, row=0, column=1)
            UI.LabelGrid(text=" uint32:", row=2, column=0), UI.EntryGrid(width=15, row=2, column=1), UI.LabelGrid(text="  int32:", row=2, column=2), UI.EntryGrid(width=15, row=2, column=3), UI.LabelGrid(text="    dgk:", row=0, column=0), UI.EntryGrid(width=15, row=0, column=1), UI.LabelGrid(text="  uint8:", row=0, column=0), UI.EntryGrid(width=15, row=0, column=1) 
            UI.LabelGrid(text=" uint64:", row=2, column=0), UI.EntryGrid(width=15, row=2, column=1), UI.LabelGrid(text="  int64:", row=2, column=2), UI.EntryGrid(width=15, row=2, column=3), UI.LabelGrid(text="  ascii:", row=0, column=0), UI.EntryGrid(width=15, row=0, column=1), UI.LabelGrid(text="  uint8:", row=0, column=0), UI.EntryGrid(width=15, row=0, column=1)

        # frame = ttk.Frame(self)
        # frame.pack(side="left") # 
        # width = 25
        # row, column = 0, 0
        # self.__entry__uint8__,   row = Tree基础显示.__标签输入部件__(frame, text="  uint8：", row=row, column=column, width=width), row + 1
        # self.__entry__uint16__,  row = Tree基础显示.__标签输入部件__(frame, text=" uint16：", row=row, column=column, width=width), row + 1
        # self.__entry__uint32__,  row = Tree基础显示.__标签输入部件__(frame, text=" uint32：", row=row, column=column, width=width), row + 1
        # self.__entry__uint64__,  row = Tree基础显示.__标签输入部件__(frame, text=" uint64：", row=row, column=column, width=width), row + 1
        # frame.grid_rowconfigure((4,), minsize=6, weight=0)
        # row = 5
        # self.__entry__int8__,    row = Tree基础显示.__标签输入部件__(frame, text="   int8：", row=row, column=column, width=width), row + 1
        # self.__entry__int16__,   row = Tree基础显示.__标签输入部件__(frame, text="  int16：", row=row, column=column, width=width), row + 1
        # self.__entry__int32__,   row = Tree基础显示.__标签输入部件__(frame, text="  int32：", row=row, column=column, width=width), row + 1
        # self.__entry__int64__,   row = Tree基础显示.__标签输入部件__(frame, text="  int64：", row=row, column=column, width=width), row + 1

        # row, column = 0, column+2
        # self.__entry__char__,    row = Tree基础显示.__标签输入部件__(frame, text="   char：", row=row, column=column, width=width), row + 1
        # self.__entry__utf8__,    row = Tree基础显示.__标签输入部件__(frame, text="   utf8：", row=row, column=column, width=width), row + 1
        # row = 6
        # self.__entry__float16__, row = Tree基础显示.__标签输入部件__(frame, text="float16：", row=row, column=column, width=width), row + 1
        # self.__entry__float32__, row = Tree基础显示.__标签输入部件__(frame, text="float32：", row=row, column=column, width=width), row + 1
        # self.__entry__float64__, row = Tree基础显示.__标签输入部件__(frame, text="float64：", row=row, column=column, width=width), row + 1


        # 选择集列表 = []
        # items = self.frametree.treeview.selection()
        # for item in items:
        #     if self.frametree.bpdict[item] is None: continue
        #     bp = self.frametree.bpdict[item].copy()
        #     选择集列表.append(bp)
        # self.frameview.frameoption.选择集列表 = 选择集列表



# import tkinter as tk
# import tkinter.ttk as ttk
# import tkinter.filedialog

# import os
# import 底层文件

# from . import Tree右键菜单


# class __Tree右键菜单__:
#     def __init__(self, frametreeview, 菜单):
#         self.menu = Tree右键菜单.stream2tkinter.类(frametreeview, 菜单)


# class TreeFrame(ttk.Widget):
#     def __init__(self, master=None, **kwargs):
#         ttk.Widget.__init__(self, master, "ttk::frame", kwargs)
#         self.bpdict = {}
#         self.__添加__部件__()

#     def insertblock(self, fitem="", text="", bp=None):
#         item = self.treeview.insert(fitem, index="end", text=text, values=[uint8 for uint8 in bp.readseek0(32)])
#         self.bpdict[item] = bp
#         return item

#     def insertbytes(self, fitem="", text="", bp=None):
#         item = self.treeview.insert(fitem, index="end", text=text, values=[uint8 for uint8 in bp.readseek0(32)])
#         self.bpdict[item] = bp
#         return item

#     def insertvalue(self, fitem="", text="", values=[]):
#         item = self.treeview.insert(fitem, index="end", text=text, values=values[0:32])
#         self.bpdict[item] = None
#         return item
    
#     def itemblock(self, item="", text="", bp=None):
#         if text == "": 
#             self.treeview.item(item, values=[uint8 for uint8 in bp.readseek0(32)])
#         else:
#             self.treeview.item(item, text=text, values=[uint8 for uint8 in bp.readseek0(32)])
#         self.bpdict[item] = bp
#         return item

#     def itembytes(self, item="", text="", bp=None):
#         if text == "": 
#             self.treeview.item(item, values=[uint8 for uint8 in bp.readseek0(32)])
#         else:
#             self.treeview.item(item, text=text, values=[uint8 for uint8 in bp.readseek0(32)])
#         self.bpdict[item] = bp
#         return item
    
#     def itemvalue(self, item="", text="", values=[]):
#         if text == "": 
#             self.treeview.item(item, values=values)
#         else:
#             self.treeview.item(item, text=text, values=values)
#         self.bpdict[item] = None
#         return item



#     def __添加__部件__(self):
#         frameb = self.__添加__选项__区域__()
#         framec = self.__添加__表格__区域__()
#         rmenua = self.__添加__右键__菜单__()
#         self.treeview = framec.treeview
#         framec.rightmenu = rmenua
#         self.frameoption = frameb
#         self.__frametreeview__ = framec
#         self.rightmenu = rmenua

#     def __添加__选项__区域__(self):
#         frame = __ItemFrame__(self)
#         frame.pack(side='top', fill='x')
#         return frame

#     def __添加__表格__区域__(self):
#         frame = __TreeFrame__(self, height=500)
#         frame.pack_propagate(0) # 同时pack_propagate(0)，才能调整面板高度
#         frame.pack(side='left', fill='both', expand=1)
#         return frame

#     def __添加__右键__菜单__(self):
#         menu = tk.Menu(None, tearoff=False)
#         menu.add_command(label="打开文件", command=lambda:批量打开文件(self))
#         menu.add_command(label="打开文件夹", command=lambda:打开文件夹(self))
#         menu.add_separator() # 添加分割线
#         menu.add_command(label="删除行", command=lambda:删除节点(self))
#         menu.add_command(label="复制行标题", command=self.__复制行标题__)
#         menu.add_separator() # 添加分割线
#         self.__format__menu__ = __Tree右键菜单__(self, menu)
#         menu.add_separator() # 添加分割线
#         menu.add_command(label="数据集to右边", command=None)
#         return menu

#     def __复制行标题__(self):
#         item = self.treeview.identify_row(self.event.y)
#         chars = self.treeview.item(item)["text"]
#         self.clipboard_clear()
#         self.clipboard_append(chars)



# def 批量打开文件(frametreeview):
#     filepaths = tkinter.filedialog.askopenfilenames()
#     filepaths = [filepath for filepath in filepaths if filepath != "" and os.path.isfile(filepath) == True]
#     for filepath in filepaths:
#         basename = os.path.basename(filepath)           # 带扩展名
#         frametreeview.insertblock("", text=basename, bp=底层文件.bpformat.bpnumpy.类().filepath(filepath))


# def 打开文件夹(frametreeview):
#     filedir = tkinter.filedialog.askdirectory()
#     if filedir == "": return None # FileNotFoundError: [WinError 3] 系统找不到指定的路径。: ''
    
#     filepaths = []
#     for filename in os.listdir(filedir):
#         filepath = os.path.join(filedir, filename)
#         if os.path.isfile(filepath): filepaths.append(filepath)
        
#     for filepath in filepaths:
#         basename = os.path.basename(filepath)           # 带扩展名
#         frametreeview.insertblock("", text=basename, bp=底层文件.bpformat.bpnumpy.类().filepath(filepath))





# def 删除节点(frametreeview):
#     selection = frametreeview.treeview.selection()
#     rootitems = frametreeview.treeview.get_children()
#     for item in selection:
#         if item in rootitems: frametreeview.bpdict[item].close()
#         try:
#             frametreeview.treeview.delete(item)
#             frametreeview.bpdict.pop(item)
#         except Exception as e:
#             pass # print(e)




# def __标签输入部件__(self, text="", row=0, column=0, width=15, value=""):
#     标签 = ttk.Label(self, text=text)
#     标签.grid(row=row, column=column, sticky="e")
#     输入 = ttk.Entry(self, width=width)
#     输入.insert(index=0, string=value)
#     输入.grid(row=row, column=column+1, sticky="we")
#     return 输入


# def __标签标签部件__(self, text="", row=0, column=0, width=15):
#     标签 = ttk.Label(self, text=text)
#     标签.grid(row=row, column=column, sticky="e")
#     输出 = ttk.Label(self, width=width)
#     输出.grid(row=row, column=column+1, sticky="we")
#     return 输出

# def __标签选取部件__(self, text="", row=0, column=0, width=15, values=[0, 1, 2, 3, 4, 5, 6, 7], index=0):
#     标签 = ttk.Label(self, text=text)
#     标签.grid(row=row, column=column, sticky="e")
#     选取 = ttk.Combobox(self, width=width, values=values)
#     选取.current(index) # 选择第1个
#     选取.grid(row=row, column=column+1, sticky="we")
#     return 选取

# def __标签按钮部件__(self, text="", row=0, column=0, width=15, command=None):
#     按钮 = ttk.Button(self, text=text, width=width, command=command)
#     按钮.grid(row=row, column=column, sticky="we")
#     return 按钮


# class __ItemFrame__(ttk.Widget):
#     def __init__(self, master=None, **kwargs):
#         ttk.Widget.__init__(self, master, "ttk::frame", kwargs)
#         self.frametreeview = master
#         self.__添加__参数__部件__()

#     def __添加__参数__部件__(self):
#         width = 42
#         row, column = 0, 0
#         self.索引列表, row = __标签输入部件__(self, text="索引列表：", row=row, column=column, width=width, value="48/56/128"), row + 1
#         self.大小列表, row = __标签输入部件__(self, text="大小列表：", row=row, column=column, width=width, value="24/48/96"), row + 1
#         self.状态显示, row = __标签输入部件__(self, text="状态显示：", row=row, column=column, width=width, value=""), row + 1

#         width = 10
#         row, column = 0, column + 2
#         self.索引分割, row = __标签按钮部件__(self, text="索引分割", row=row, column=column, width=width, command=self.__按索引新建子节点__), row + 1
#         self.大小分割, row = __标签按钮部件__(self, text="大小分割", row=row, column=column, width=width, command=self.__按大小新建子节点__), row + 1
#         self.层级分割, row = __标签按钮部件__(self, text="层级分割", row=row, column=column, width=width, command=self.__按索引新建层节点__), row + 1


#     def __按索引新建子节点__(self):
#         numchars = self.索引列表.get().split("/")
#         numbers = [0] + [int(numchar) for numchar in numchars if numchar != ""]
#         sizes = [ numbers[i+1]-numbers[i] for i in range(len(numbers)-1)]
#         sizes = [ size for size in sizes if size > 0 ]
#         for item in self.frametreeview.treeview.selection(): self.__新建子节点__(item, sizes)


#     def __按大小新建子节点__(self):
#         numchars = self.大小列表.get().split("/")
#         sizes = [int(numchar) for numchar in numchars if numchar != ""]
#         sizes = [ size for size in sizes if size > 0 ]
#         for item in self.frametreeview.treeview.selection(): self.__新建子节点__(item, sizes)


#     def __新建子节点__(self, item, sizes):
#         bp = self.frametreeview.bpdict[item].copy()
#         bps = [bp.readslice(size) for size in sizes]
#         sizes.append(bp.remainsize())
#         bps.append(bp.readremainslice())
#         for size, bp in zip(sizes, bps): citem = self.frametreeview.insertblock(item, text=f"块_{size}", bp=bp)  
#         char = self.frametreeview.treeview.item(citem)["text"] 
#         self.frametreeview.treeview.item(citem, text="余"+char)


#     def __新建层节点__(self, item, sizes):
#         bp = self.frametreeview.bpdict[item].copy()
#         bps = [bp.readslice(size) for size in sizes]
#         sizes.append(bp.remainsize())
#         bps.append(bp.readremainslice())

#         fitem = self.frametreeview.treeview.parent(item)
#         if fitem == "":
#             for size, bp in zip(sizes, bps): citem = self.frametreeview.insertblock(item, text=f"块_{size}", bp=bp)
#         else:
#             citem = self.frametreeview.itemblock(item, text=f"块_{sizes[0]}", bp=bps[0])
#             for size, bp in zip(sizes[1:], bps[1:]): citem = self.frametreeview.insertblock(fitem, text=f"块_{size}", bp=bp) 

#         char = self.frametreeview.treeview.item(citem)["text"] 
#         self.frametreeview.treeview.item(citem, text="余"+char)
        
#     def __按索引新建层节点__(self):
#         numchars = self.索引列表.get().split("/")
#         numbers = [0] + [int(numchar) for numchar in numchars if numchar != ""]
#         sizes = [ numbers[i+1]-numbers[i] for i in range(len(numbers)-1)]
#         sizes = [ size for size in sizes if size > 0 ]
#         for item in self.frametreeview.treeview.selection(): self.__新建层节点__(item, sizes)


#     def __按大小新建层节点__(self):
#         numchars = self.大小列表.get().split("/")
#         sizes = [int(numchar) for numchar in numchars if numchar != ""]
#         sizes = [ size for size in sizes if size > 0 ]
#         for item in self.frametreeview.treeview.selection(): self.__新建层节点__(item, sizes)
  


# class __TreeFrame__(ttk.Widget):
#     def __init__(self, master=None, **kwargs):
#         ttk.Widget.__init__(self, master, "ttk::frame", kwargs)
#         self.frametreeview = master
#         self.__bars__ = self.__添加__滚动__区域__()
#         self.treeview = self.__添加__表格__区域__()
#         self.__绑定函数__()
#         self.leftpopup = None
#         self.rightmenu = None

#     def __添加__滚动__区域__(self):
#         xbar = ttk.Scrollbar(self, orient='horizontal')   # 水平
#         xbar.pack(side="bottom", fill="x")                # must be top, bottom, left, or right
#         ybar = ttk.Scrollbar(self, orient='vertical')     # 垂直
#         ybar.pack(side="right", fill="y")
#         return xbar, ybar
    
#     def __添加__表格__区域__(self):
#         columns = [f"#{i+1}" for i in range(32)]
#         treeview = ttk.Treeview(self, columns=columns)
#         treeview.pack(side='left', fill='both')

#         return treeview


#     def __绑定函数__(self):
#         xbar, ybar = self.__bars__
#         xbar.config(command=self.treeview.xview)
#         ybar.config(command=self.treeview.yview)
#         self.treeview.config(xscrollcommand=xbar.set, yscrollcommand=ybar.set)
#         self.treeview.bind("<ButtonRelease-3>", self.__显示右键菜单__) # 右键单击离开


#     def __显示右键菜单__(self, event):
#         self.rightmenu.post(event.x_root, event.y_root)

# import tkinter as tk
# import tkinter.ttk as ttk

# from . import Tree基础显示


# __格式__步长__字典__ = {
#     "bin":1, 
#     "bin8":1,        
#     "hex":1,
#     "gbk":1,    
#     "char":1,
#     "utf8":1, # 可变字节，步长只能为1，为其他步长，会造成少读字符
#     "int8": 1,
#     "int16": 2,
#     "int32": 4,
#     "int64": 8,
#     "uint8": 1,
#     "uint16": 2,
#     "uint32": 4,
#     "uint64": 8,
#     "5u8uint64": 5, 
#     "float16": 2,
#     "float32": 4,
#     "float64": 8, 
#     "i8float32": 4,
#     "u8float32": 4,
#     "i16float32": 4,
#     "u16float32": 4,          
#     }


# __格式__宽度__字典__ = {
#     "bin":80, 
#     "bin8":80,        
#     "hex":40,
#     "gbk":40,    
#     "char":40,
#     "utf8":40,
#     "int8": 40,
#     "uint8": 40,

#     "int16": 60,
#     "uint16": 60,

#     "int32": 80,
#     "uint32": 80,

#     "int64": 120,
#     "uint64": 120,
#     "5u8uint64": 120,

#     "float16": 100,
#     "float32": 100,
#     "float64": 100, 
#     "i8float32": 100,
#     "u8float32": 100,
#     "i16float32": 100,
#     "u16float32": 100,                
#     }


# __解析__格式__列表__ = [
#     "bin",  
#     "bin8",        
#     "hex",
#     "gbk",
#     "char",
#     "utf8",
#     "hex+char",
#     "hex+uint8",
#     "uint8+char",
#     "bin8+gbk",    
#     "bin8+utf8",   
#     "uint8+gbk",    
#     "uint8+utf8",  
#     "uint8+bin8+gbk",    
#     "uint8+bin8+utf8",
#     "uint8",
#     "uint16",
#     "uint32",
#     "uint64",
#     "int8",
#     "int16",
#     "int32",
#     "int64",
#     "float16",
#     "float32",
#     "float64",
#     "uint8+uint16",
#     "uint8+uint32",    
#     "i8float32",
#     "u8float32",
#     "i16float32",
#     "u16float32",
#     "int8+i8float32",
#     "uint8+u8float32",
#     "int16+i16float32",
#     "uint16+u16float32",
#     "5u8uint64",
#     "uint8+5u8uint64",    
#     ]


# class ViewFrame(ttk.Widget):
#     def __init__(self, master=None, **kwargs):
#         ttk.Widget.__init__(self, master, "ttk::frame", kwargs)
#         self.item = None
#         self.column = None
#         self.__添加__部件__()


#     def __添加__部件__(self):
#         frameb = self.__添加__选项__区域__()
#         framec = self.__添加__表格__区域__()
#         popupa = self.__添加__左键__弹窗__(frameb)
#         rmenua = self.__添加__右键__菜单__(frameb)
#         self.treeview = framec.treeview
#         framec.leftpopup = popupa
#         framec.rightmenu = rmenua
#         self.frameoption = frameb
#         self.__frametreeview__ = framec
#         self.rightmenu = rmenua

#     def __添加__选项__区域__(self):
#         frame = __ItemFrame__(self)
#         frame.pack(side='top', fill='x')
#         return frame

#     def __添加__表格__区域__(self):
#         frame = __TreeFrame__(self, height=500)
#         frame.pack_propagate(0) # 同时pack_propagate(0)，才能调整面板高度
#         frame.pack(side='top', fill='both', expand=1)
#         return frame

#     def __添加__左键__弹窗__(self, frameoption):
#         popup = __EditPopup__(self, frameoption)
#         return popup

#     def __添加__右键__菜单__(self, frameoption):
#         menu = tk.Menu(None, tearoff=False)
#         menu.add_command(label="右键分行", command=frameoption.__函数__右键分行__)
#         menu.add_command(label="右键后行", command=frameoption.__函数__右键后行__)
#         menu.add_command(label="右键末行", command=frameoption.__函数__右键末行__)
#         menu.add_command(label="右键并行", command=frameoption.__函数__右键并行__)
#         menu.add_separator() # 添加分割线
#         menu.add_command(label="跳转首行", command=self.__跳转首行__)
#         menu.add_command(label="跳转末行", command=self.__跳转末行__)        
#         menu.add_separator() # 添加分割线
#         menu.add_command(label="显示左侧栏", command=self.__显示左侧栏__)
#         menu.add_command(label="隐藏左侧栏", command=self.__隐藏左侧栏__)
#         menu.add_separator() # 添加分割线
#         menu.add_command(label="复制行内容", command=self.__复制行内容__)
#         menu.add_command(label="复制行文本", command=self.__复制行文本__)
#         return menu


#     def __跳转首行__(self):
#         item = self.treeview.get_children("")[0]
#         self.treeview.see(item)

#     def __跳转末行__(self):
#         item = self.treeview.get_children("")[-1]
#         self.treeview.see(item)

#     def __隐藏左侧栏__(self):
#         self.treeview["show"] = "headings"

#     def __显示左侧栏__(self):
#         self.treeview["show"] = "tree headings"

#     def __复制行内容__(self):
#         values = self.treeview.item(self.item)["values"]
#         self.clipboard_clear()
#         self.clipboard_append(str(values))

#     def __复制行文本__(self):
#         values = self.treeview.item(self.item)["values"]
#         chars = ""
#         for char in values:
#             if isinstance(char, int): char = str(char)
#             chars += char
#         self.clipboard_clear()
#         self.clipboard_append(chars)


# def __刷新__表格__行域__(self, numrow):
#     items = self.treeview.get_children("")
#     numitem = len(items)
#     if numitem < numrow:
#         for i in range(numitem, numrow) : self.treeview.insert("", index="end", text="", values=[]) 
#     if numitem == numrow:
#         pass
#     if numitem > numrow:
#         for item in items[numrow:][::-1]: self.treeview.delete(item)
#     return self.treeview.get_children("")


# class __ItemFrame__(ttk.Widget):
#     def __init__(self, master=None, **kw):
#         ttk.Widget.__init__(self, master, "ttk::frame", kw)
#         self.frametreeview = master
#         self.__添加__参数__部件__()
#         self.选择集列表 = []


#     def __添加__参数__部件__(self):
#         width = 15
#         row, column = 0, 0
#         self.__设置__偏移大小__, row = Tree基础显示.__标签选取部件__(self, text="偏移大小：", row=row, column=column, width=width), row + 1
#         self.__设置__单区大小__, row = Tree基础显示.__标签输入部件__(self, text="单区大小：", row=row, column=column, width=width, value=10240), row + 1
#         self.__设置__解析格式__, row = Tree基础显示.__标签选取部件__(self, text="解析格式：", row=row, column=column, width=width, values=__解析__格式__列表__, index=8), row + 1

#         row, column = 0, column + 2
#         self.__设置__表格行数__, row = Tree基础显示.__标签输入部件__(self, text="表格行数：", row=row, column=column, width=width, value=256), row + 1
#         self.__设置__每行列数__, row = Tree基础显示.__标签输入部件__(self, text="每行列数：", row=row, column=column, width=width, value=16), row + 1
#         self.__设置__显示留行__, row = Tree基础显示.__标签输入部件__(self, text="显示留行：", row=row, column=column, width=width, value=2), row + 1

#         row, column = 0, column + 2
#         self.__按钮__弹窗开闭__, row = Tree基础显示.__标签按钮部件__(self, text="", row=row, column=column, width=width, command=None), row + 1
#         self.__按钮__端序切换__, row = Tree基础显示.__标签按钮部件__(self, text="", row=row, column=column, width=width, command=None), row + 1
#         self.__按钮__刷新显示__, row = Tree基础显示.__标签按钮部件__(self, text="刷新显示", row=row, column=column, width=width, command=self.__函数__刷新显示__), row + 1


#     def 读取刷新设置(self):
#         self.格式列表 = self.__设置__解析格式__.get().split("+")
#         self.偏移大小 = int(self.__设置__偏移大小__.get())
#         self.单区大小 = int(self.__设置__单区大小__.get())
#         self.表格行数 = int(self.__设置__表格行数__.get())
#         self.每行列数 = int(self.__设置__每行列数__.get())
#         self.显示留行 = int(self.__设置__显示留行__.get())
#         __刷新__表格__行域__(self.frametreeview, self.表格行数)
#         return self.偏移大小, self.单区大小, self.格式列表, self.表格行数, self.每行列数, self.显示留行
    
#     def __组内行数__(self):
#         return len(self.格式列表) * len(self.选择集列表)

#     def __有效行数__(self):
#         return self.表格行数 // self.组内行数

#     def __单格大小__(self):
#         单元格字节数最大值 = 1
#         for 格式 in self.格式列表:
#             步长 = __格式__步长__字典__[格式]
#             if 步长 > 单元格字节数最大值: 单元格字节数最大值 = 步长
#         return 单元格字节数最大值

#     def __倍数列表__(self):
#         列表 = []
#         for 格式 in self.格式列表:
#             步长 = __格式__步长__字典__[格式]
#             倍数 = self.单格大小 // 步长
#             列表.append(倍数)
#         return 列表

#     def __数据大小__(self):
#         数据块大小最大值 = 0
#         for bp in self.选择集列表:
#             数据块大小 = bp.size()
#             if 数据块大小 > 数据块大小最大值: 数据块大小最大值 = 数据块大小
#         return 数据块大小最大值


#     def __分段数量__(self):
#         段数 = self.数据大小  // self.单区大小
#         余数 = self.数据大小  %  self.单区大小
#         if 余数 > 0: 段数 += 1
#         return 段数

#     def 读取刷新参数(self):
#         self.单格大小 = self.__单格大小__()
#         self.倍数列表 = self.__倍数列表__()
#         self.数据大小 = self.__数据大小__()
#         self.分段数量 = self.__分段数量__()
#         self.组内行数 = self.__组内行数__()
#         self.有效行数 = self.__有效行数__()
#         self.分段长度 = 1.0 / self.分段数量
#         return self.单格大小, self.倍数列表, self.分段数量, self.组内行数, self.有效行数, self.分段长度

#     def 读取右键参数(self):
#         self.表索引 = self.rowindex // self.组内行数
#         self.列索引 = self.columnindex
#         return self.表索引, self.列索引

#     def __函数__刷新显示__(self): __刷新显示__(self)

#     def __函数__拖动显示__(self): __拖动显示__(self)

#     def __函数__换页显示__(self): __换页显示__(self)  

#     def __函数__右键分行__(self): __右键分行__(self) 

#     def __函数__右键后行__(self): __右键后行__(self) 

#     def __函数__右键末行__(self): __右键末行__(self) 

#     def __函数__右键并行__(self): __右键并行__(self) 
     

# def __刷新显示__(self):
#     self.读取刷新设置()
#     self.读取刷新参数()
#     self.列数列表列表 = [ [] for i in range(self.分段数量) ]
#     self.段索引 = 0
#     self.行索引 = 0
#     self.前置大小 = self.偏移大小
#     列数列表 = __读取__当前列数列表__(self.列数列表列表,  0, self.每行列数, self.单区大小, self.单格大小)
#     末尾列表 = __初始__末尾列数列表__(self.列数列表列表, -1, self.每行列数, self.数据大小, self.单区大小, self.单格大小)
#     范围列数列表 = 列数列表[self.行索引: self.行索引+self.有效行数]
#     显示数据列表 = __生成__数据显示列表__(self.选择集列表, self.前置大小, 范围列数列表, self.格式列表, self.倍数列表, self.单格大小)
#     __刷新__表格显示处理__(self.frametreeview, 显示数据列表)
#     __刷新__表格列数处理__(self.frametreeview, 范围列数列表)
#     __刷新__表格列宽处理__(self.frametreeview, self.格式列表)
#     显示索引列表 = __生成__数据索引列表__(self.选择集列表, self.前置大小, 范围列数列表, self.格式列表, self.倍数列表, self.单格大小)
#     __刷新__表格左边侧栏__(self.frametreeview, self.选择集列表, 显示数据列表, 显示索引列表, self.格式列表)


# def __拖动显示__(self):
#     self.段索引 = __拖动显示__计算分段索引__(self.ybarfloat, self.分段长度, self.分段数量)
#     列数列表 = __读取__当前列数列表__(self.列数列表列表, self.段索引, self.每行列数, self.单区大小, self.单格大小)
#     self.分行长度 = self.分段长度 / len(列数列表)
#     self.行索引 = __拖动显示__计算分行索引__(self.ybarfloat, self.分段长度, self.段索引, self.分行长度, len(列数列表), self.显示留行)
#     self.前置大小 = __滚条显示__简化前置偏移__(列数列表, self.段索引, self.行索引, self.偏移大小, self.单区大小, self.单格大小)
#     范围列数列表 = 列数列表[self.行索引: self.行索引+self.有效行数]
#     显示数据列表 = __生成__数据显示列表__(self.选择集列表, self.前置大小, 范围列数列表, self.格式列表, self.倍数列表, self.单格大小)
#     __刷新__表格显示处理__(self.frametreeview, 显示数据列表)
#     __刷新__表格列数处理__(self.frametreeview, 范围列数列表)
#     __刷新__表格列宽处理__(self.frametreeview, self.格式列表)
#     显示索引列表 = __生成__数据索引列表__(self.选择集列表, self.前置大小, 范围列数列表, self.格式列表, self.倍数列表, self.单格大小)
#     __刷新__表格左边侧栏__(self.frametreeview, self.选择集列表, 显示数据列表, 显示索引列表, self.格式列表)


# def __换页显示__(self):
#     if 1 < self.段索引 < self.分段数量 - 1:
#         上表列表 = __读取__当前列数列表__(self.列数列表列表, self.段索引-1, self.每行列数, self.单区大小, self.单格大小)
#         下表列表 = __读取__当前列数列表__(self.列数列表列表, self.段索引+1, self.每行列数, self.单区大小, self.单格大小)
#     self.段索引, self.行索引 = __换页显示__计算段行索引__(self.ybarrange, self.列数列表列表, self.段索引, self.行索引, self.有效行数) 
#     列数列表 = __读取__当前列数列表__(self.列数列表列表, self.段索引, self.每行列数, self.单区大小, self.单格大小) # 防止下方 len(列数列表) == 0
#     self.分行长度 = self.分段长度 / len(列数列表)
#     self.前置大小 = __滚条显示__简化前置偏移__(列数列表, self.段索引, self.行索引, self.偏移大小, self.单区大小, self.单格大小)
#     范围列数列表 = self.列数列表列表[self.段索引][self.行索引: self.行索引+self.有效行数]
#     显示数据列表 = __生成__数据显示列表__(self.选择集列表, self.前置大小, 范围列数列表, self.格式列表, self.倍数列表, self.单格大小)
#     __刷新__表格显示处理__(self.frametreeview, 显示数据列表)
#     __刷新__表格列数处理__(self.frametreeview, 范围列数列表)
#     __刷新__表格列宽处理__(self.frametreeview, self.格式列表)
#     显示索引列表 = __生成__数据索引列表__(self.选择集列表, self.前置大小, 范围列数列表, self.格式列表, self.倍数列表, self.单格大小)
#     __刷新__表格左边侧栏__(self.frametreeview, self.选择集列表, 显示数据列表, 显示索引列表, self.格式列表)


# def __右键分行__(self):
#     self.读取右键参数()
#     __右键分行__列数列表操作__(self.列数列表列表[self.段索引], self.行索引+self.表索引, self.列索引)
#     范围列数列表 = self.列数列表列表[self.段索引][self.行索引: self.行索引+self.有效行数]
#     显示数据列表 = __生成__数据显示列表__(self.选择集列表, self.前置大小, 范围列数列表, self.格式列表, self.倍数列表, self.单格大小)
#     __刷新__表格显示处理__(self.frametreeview, 显示数据列表)
#     __刷新__表格列数处理__(self.frametreeview, 范围列数列表)
#     __刷新__表格列宽处理__(self.frametreeview, self.格式列表)
#     显示索引列表 = __生成__数据索引列表__(self.选择集列表, self.前置大小, 范围列数列表, self.格式列表, self.倍数列表, self.单格大小)
#     __刷新__表格左边侧栏__(self.frametreeview, self.选择集列表, 显示数据列表, 显示索引列表, self.格式列表)


# def __右键后行__(self):
#     self.读取右键参数()
#     __右键后行__列数列表操作__(self.列数列表列表[self.段索引], self.行索引+self.表索引, self.列索引)
#     范围列数列表 = self.列数列表列表[self.段索引][self.行索引: self.行索引+self.有效行数]
#     显示数据列表 = __生成__数据显示列表__(self.选择集列表, self.前置大小, 范围列数列表, self.格式列表, self.倍数列表, self.单格大小)
#     __刷新__表格显示处理__(self.frametreeview, 显示数据列表)
#     __刷新__表格列数处理__(self.frametreeview, 范围列数列表)
#     __刷新__表格列宽处理__(self.frametreeview, self.格式列表)
#     显示索引列表 = __生成__数据索引列表__(self.选择集列表, self.前置大小, 范围列数列表, self.格式列表, self.倍数列表, self.单格大小)
#     __刷新__表格左边侧栏__(self.frametreeview, self.选择集列表, 显示数据列表, 显示索引列表, self.格式列表)


# def __右键末行__(self):
#     self.读取右键参数()
#     __右键末行__列数列表操作__(self.列数列表列表[self.段索引], self.行索引+self.表索引, self.列索引)
#     范围列数列表 = self.列数列表列表[self.段索引][self.行索引: self.行索引+self.有效行数]
#     显示数据列表 = __生成__数据显示列表__(self.选择集列表, self.前置大小, 范围列数列表, self.格式列表, self.倍数列表, self.单格大小)
#     __刷新__表格显示处理__(self.frametreeview, 显示数据列表)
#     __刷新__表格列数处理__(self.frametreeview, 范围列数列表)
#     __刷新__表格列宽处理__(self.frametreeview, self.格式列表)
#     显示索引列表 = __生成__数据索引列表__(self.选择集列表, self.前置大小, 范围列数列表, self.格式列表, self.倍数列表, self.单格大小)
#     __刷新__表格左边侧栏__(self.frametreeview, self.选择集列表, 显示数据列表, 显示索引列表, self.格式列表)


# def __右键并行__(self):
#     self.读取右键参数()
#     __右键并行__列数列表操作__(self.列数列表列表[self.段索引], self.行索引+self.表索引, self.列索引)
#     范围列数列表 = self.列数列表列表[self.段索引][self.行索引: self.行索引+self.有效行数]
#     显示数据列表 = __生成__数据显示列表__(self.选择集列表, self.前置大小, 范围列数列表, self.格式列表, self.倍数列表, self.单格大小)
#     __刷新__表格显示处理__(self.frametreeview, 显示数据列表)
#     __刷新__表格列数处理__(self.frametreeview, 范围列数列表)
#     __刷新__表格列宽处理__(self.frametreeview, self.格式列表)
#     显示索引列表 = __生成__数据索引列表__(self.选择集列表, self.前置大小, 范围列数列表, self.格式列表, self.倍数列表, self.单格大小)
#     __刷新__表格左边侧栏__(self.frametreeview, self.选择集列表, 显示数据列表, 显示索引列表, self.格式列表)

            
# def __读取__当前列数列表__(列数列表列表, 段索引, 每行列数, 单区大小, 单格大小):
#     列数列表 = 列数列表列表[段索引]
#     if 列数列表 != []: return 列数列表
#     单行大小 = 每行列数 * 单格大小
#     行数 = 单区大小  // 单行大小
#     余数 = 单区大小  %  单行大小
#     for i in range(行数): 列数列表.append(每行列数)
#     if 余数 > 0: 列数列表.append(每行列数)
#     return 列数列表


# def __初始__末尾列数列表__(列数列表列表, 段索引, 每行列数, 数据大小, 单区大小, 单格大小):
#     列数列表 = 列数列表列表[段索引]
#     末表大小 = 数据大小 % 单区大小
#     单行大小 = 每行列数 * 单格大小
#     行数 = 末表大小  // 单行大小
#     余数 = 末表大小  %  单行大小
#     for i in range(行数): 列数列表.append(每行列数)
#     if 余数 > 0: 列数列表.append(每行列数)
#     return 列数列表


# def __右键分行__列数列表操作__(列数列表, 行索引, 列索引): 
#     当前列数 = 列数列表[行索引]
#     本行列数 = 列索引 - 1
#     if 本行列数 < 0: 本行列数 = 0
#     下行列数 = 当前列数 - 本行列数
#     列数列表[行索引] = 本行列数
#     列数列表.insert(行索引+1, 下行列数) # index > len，会在列表末尾加1个元素


# def __右键后行__列数列表操作__(列数列表, 行索引, 列索引):
#     当前列数 = 列数列表[行索引]
#     本行列数 = 列索引 - 1
#     if 本行列数 < 0: 本行列数 = 0
#     下行列数 = 当前列数 - 本行列数
#     列数列表[行索引] = 本行列数
#     try:
#         列数列表[行索引+1] = 列数列表[行索引+1] + 本行列数
#     except:
#         列数列表.insert(行索引+1, 下行列数)

# def __右键末行__列数列表操作__(列数列表, 行索引, 列索引):
#     当前列数 = 列数列表[行索引]
#     本行列数 = 列索引 - 1
#     if 本行列数 < 0: 本行列数 = 0
#     下行列数 = 当前列数 - 本行列数
#     列数列表[行索引] = 本行列数

#     缓存列数 = 列数列表[-1]
#     末行列数 = 缓存列数 + 下行列数
#     整除数量 = 末行列数  // 缓存列数
#     整除余数 = 末行列数  %  缓存列数
#     for i in range(1, 整除数量): 列数列表.append(缓存列数)
#     if 整除余数 > 0: 列数列表.append(整除余数)
    

# def __右键并行__列数列表操作__(列数列表, 行索引, 列索引):
#     if 行索引 >= len(列数列表)-1: return None
#     当前列数 = 列数列表[行索引]
#     下行列数 = 列数列表[行索引+1]
#     列数列表[行索引] = 当前列数 + 下行列数
#     列数列表.pop(行索引+1)


# # 显示算法
# def __拖动显示__计算分段索引__(ybarfloat, 分段长度, 分段数量):
#     段索引 = round(ybarfloat // 分段长度)
#     if 段索引 < 0 : 段索引 = 0
#     if 段索引 >= 分段数量: 段索引 = 分段数量 - 1
#     return 段索引

# def __拖动显示__计算分行索引__(ybarfloat, 分段长度, 段索引, 分行长度, 行总数, 显示留行):
#     行索引 = round((ybarfloat - 分段长度*段索引) // 分行长度)
#     if 行索引 < 0: 行索引 = 0
#     if ybarfloat < 0: 行索引 = 0
#     标位索引 = 行总数 - 显示留行 # 行索引 使用方式 切片 列数列表[行总数: ], 不用减1，不是 使用方式 访问元素 列数列表[行总数] 
#     if 行索引 > 标位索引: 行索引 = 标位索引
#     return 行索引


# def __滚条显示__简化前置偏移__(列数列表, 段索引, 行索引, 偏移大小, 单区大小, 单格大小):
#     分段偏移 = 段索引 * 单区大小
#     列段偏移 = 0
#     for 本行列数 in 列数列表[0: 行索引]:
#         列段偏移 += 本行列数 * 单格大小
#     return 偏移大小 + 分段偏移 + 列段偏移


# def __换页显示__计算段行索引__(ybarrange, 列数列表列表, 段索引, 行索引, 有效行数):
#     分段数量 = len(列数列表列表)
#     本表行数 = len(列数列表列表[段索引])
#     标索引 = 行索引 + ybarrange*有效行数
#     if 标索引 < 0: 
#         if 段索引-1 < 0: return 0, 0
#         上表行数 = len(列数列表列表[段索引-1])
#         行索引 = 上表行数 - 有效行数
#         if 行索引 < 0: 行索引 = 0
#         return 段索引-1, 行索引

#     if 标索引 >= 本表行数:
#         if 段索引+1 >= 分段数量: 
#             行索引 = 本表行数 - 有效行数
#             if 行索引 < 0: 行索引 = 0
#             return 分段数量-1, 行索引
#         # 下表行数 = len(列数列表列表[段索引+1])
#         return 段索引+1, 0

#     return 段索引, 标索引

    
# # 显示算法
# def __生成__数据显示列表__(数据列表, 前置偏移, 范围列数列表, 格式列表, 倍数列表, 单格大小):
#     for bp in 数据列表: 
#         bp.index = bp.mpleft
#         bp.seekclosecheck(前置偏移)

#     显示列表 = []
#     for 列数 in 范围列数列表:
#         for 格式, 倍数 in zip(格式列表, 倍数列表):
#             for bp in 数据列表: 
#                 结果列表 = 读取格式seek0(bp, 格式, 倍数, 列数)
#                 显示列表.append(结果列表)
#         for bp in 数据列表: bp.seekclosecheck(单格大小*列数)

#     return 显示列表

# def __生成__数据索引列表__(数据列表, 前置偏移, 范围列数列表, 格式列表, 倍数列表, 单格大小):
#     for bp in 数据列表: 
#         bp.index = bp.mpleft
#         bp.seekclosecheck(前置偏移)

#     索引列表 = []
#     for 列数 in 范围列数列表:
#         for 格式, 倍数 in zip(格式列表, 倍数列表):
#             for bp in 数据列表: 索引列表.append([bp.tell(), bp.slicetell()])
#         for bp in 数据列表: bp.seekclosecheck(单格大小*列数)
#     return 索引列表



# def 读取格式seek0(bp, 格式, 倍数, 列数):
#     # tkinter 会自动处理 [[], [], [], [], [], , ]
#     列表 = []
#     match 格式:
#         case "bin": 列表 = bp.readbinqueueseek0(倍数*列数)            
#         case "hex": 列表 = bp.readhexqueueseek0(倍数*列数)      
#         case "char": 列表 = bp.readcharqueueseek0(倍数*列数)
#         case "int8": 列表 = bp.readint8seek0(倍数*列数)
#         case "int16": 列表 = bp.readint16seek0(倍数*列数)   
#         case "int32": 列表 = bp.readint32seek0(倍数*列数) 
#         case "int64": 列表 = bp.readint64seek0(倍数*列数) 
#         case "uint8": 列表 = bp.readuint8seek0(倍数*列数)
#         case "uint16": 列表 = bp.readuint16seek0(倍数*列数)
#         case "uint32": 列表 = bp.readuint32seek0(倍数*列数)
#         case "uint64": 列表 = bp.readuint64seek0(倍数*列数)
#         case "float16": 列表 = bp.readfloat16seek0(倍数*列数)
#         case "float32": 列表 = bp.readfloat32seek0(倍数*列数)
#         case "float64": 列表 = bp.readfloat64seek0(倍数*列数)
#         case "i8float32": 列表 = bp.readi8float32seek0(倍数*列数)
#         case "u8float32": 列表 = bp.readu8float32seek0(倍数*列数)
#         case "i16float32": 列表 = bp.readi16float32seek0(倍数*列数)
#         case "u16float32": 列表 = bp.readu16float32seek0(倍数*列数)
        
#     if not isinstance(列表, list): 列表 = [列表] # 处理num==1情况
#     if 格式 == "bin8": 列表 = bpformat_readbin8seek0(bp, 倍数, 列数)
    
#     结果列表 = []
#     for i in range(0, len(列表), 倍数): 结果列表.append(列表[i:i+倍数])
#     if 格式 == "gbk": 结果列表 = bpformat_readgbkseek0(bp, 倍数, 列数)    
#     if 格式 == "utf8": 结果列表 = bpformat_readutf8seek0(bp, 倍数, 列数)
#     return 结果列表

# def bpformat_readbin8seek0(bp, 倍数, 列数):
#     列表 = []
#     列表 = bp.readbinseek0(倍数*列数)
#     if 倍数*列数 == 1: 列表 = [列表]  
#     binchar8列表 = []
#     for binchar in 列表:
#         前缀, 后列表 = binchar[0:2], binchar[2:]
#         前零 = ""
#         for i in range(8-len(后列表)): 前零 += "0"
#         binchar8列表.append(前缀+前零+后列表)
#     return binchar8列表

# def bpformat_readgbkseek0(bp, 倍数, 列数): #
#     # bin(bytebin) # TypeError: 'bytes' object cannot be interpreted as an integer
#     binchar8列表 = bpformat_readbin8seek0(bp, 倍数, 列数)
    
#     描述列表 = []
#     跳位 = 0
#     for i in range(len(binchar8列表)):
#         if i < 跳位: continue
#         length = len(binchar8列表[i:])
#         if length >= 2:
#             if binchar8列表[i].startswith("0b1") and binchar8列表[i+1].startswith("0b1"):
#                 跳位 += 2
#                 描述列表.append([i, 2])
#                 continue
#         跳位 += 1
#         描述列表.append([i, 1])

#     结果列表 = []
#     bx = bp.readsliceseek0(倍数*列数)
#     for index, size in 描述列表:
#         if size == 1: 
#             char = chr(bx.readuint8())
#             结果列表.append(char)
#         else:
#             结果列表.append(bx.readgbk(size))
#             结果列表.append("")

#     return 结果列表

# def bpformat_readutf8seek0(bp, 倍数, 列数):
#     # bin(bytebin) # TypeError: 'bytes' object cannot be interpreted as an integer
#     binchar8列表 = bpformat_readbin8seek0(bp, 倍数, 列数)
    
#     描述列表 = []
#     跳位 = 0
#     for i in range(len(binchar8列表)):
#         if i < 跳位: continue
#         length = len(binchar8列表[i:])
#         if length >= 2:
#             if binchar8列表[i].startswith("0b110") and binchar8列表[i+1].startswith("0b10"):
#                 跳位 += 2
#                 描述列表.append([i, 2])
#                 continue
#         if length >= 3:
#             if binchar8列表[i].startswith("0b1110") and binchar8列表[i+1].startswith("0b10") and binchar8列表[i+2].startswith("0b10"):
#                 跳位 += 3
#                 描述列表.append([i, 3])
#                 continue
#         if length >= 4:
#             if binchar8列表[i].startswith("0b11110") and binchar8列表[i+1].startswith("0b10") and binchar8列表[i+2].startswith("0b10") and binchar8列表[i+3].startswith("0b10"):
#                 跳位 += 4
#                 描述列表.append([i, 4])
#                 continue
#         跳位 += 1
#         描述列表.append([i, 1])

#     结果列表 = []
#     bx = bp.readsliceseek0(倍数*列数)
#     for index, size in 描述列表:
#         if size == 1: 
#             char = chr(bx.readuint8())
#             结果列表.append(char)
#         else:
#             结果列表.append(bx.readutf8(size))
#             for i in range(size-1): 结果列表.append("")

#     return 结果列表


# def __刷新__表格显示处理__(frametreeview, 数据显示列表):
#     treeviewitems = frametreeview.treeview.get_children("")
#     numitem, numdata = len(treeviewitems), len(数据显示列表)
#     for item, values in zip(treeviewitems, 数据显示列表): frametreeview.treeview.item(item, values=values)
#     for item in treeviewitems[numdata:]: frametreeview.treeview.item(item, text="", values=[])

# def __刷新__表格列数处理__(frametreeview, 范围列数列表):
#     列数最大值 = 0
#     for 列数 in 范围列数列表:
#         if 列数 > 列数最大值: 列数最大值 = 列数

#     字符列表 = [f"#{i+1}" for i in range(列数最大值)]
#     frametreeview.treeview["columns"] = 字符列表

# def __刷新__表格列宽处理__(frametreeview, 格式列表):
#     宽度最大值 = 0
#     for 格式 in 格式列表:
#         宽度 = __格式__宽度__字典__[格式]
#         if 宽度 > 宽度最大值: 宽度最大值 = 宽度

#     columns = frametreeview.treeview["columns"]
#     frametreeview.treeview.column("#0", width=240, stretch=0)
#     for column, columnstitle in zip(columns, columns):
#         frametreeview.treeview.column(column, width=宽度最大值, stretch=False, anchor='center')
#         frametreeview.treeview.heading(column, text=columnstitle)

# def __刷新__表格左边侧栏__(frametreeview, 数据列表, 显示列表, 索引列表, 格式列表):
#     items = frametreeview.treeview.get_children("")
#     数据数量, 格式数量 = len(数据列表), len(格式列表)
#     折算数量 = 数据数量 * 格式数量
#     if "char" in 格式列表:
#         index = 格式列表.index("char")
#         更新左侧栏(frametreeview, items, 显示列表, 数据数量, 折算数量, index)
#     if "gbk" in 格式列表:
#         index = 格式列表.index("gbk")
#         更新左侧栏(frametreeview, items, 显示列表, 数据数量, 折算数量, index)
#     if "utf8" in 格式列表:
#         index = 格式列表.index("utf8")
#         更新左侧栏(frametreeview, items, 显示列表, 数据数量, 折算数量, index)
#     更新偏移值(frametreeview, items, 显示列表, 索引列表, 数据数量, 折算数量, 0)

# def 更新左侧栏(frametreeview, items, 显示列表, 数据数量, 折算数量, index):
#     for i, [item, values] in enumerate(zip(items, 显示列表)):   
#         m = i  % 折算数量
#         n = m // 数据数量
#         if n == index : 
#             chars = ""
#             for 列表 in values:
#                 for char in 列表: chars += char
#             frametreeview.treeview.item(item, text=chars)

# def 更新偏移值(frametreeview, items, 显示列表, 索引列表, 数据数量, 折算数量, index):
#     for i, [item, values, [offset, slicetell]] in enumerate(zip(items, 显示列表, 索引列表)):   
#         m = i  % 折算数量
#         n = m // 数据数量
#         if n == index : frametreeview.treeview.item(item, text=f"0d{offset}_0d{slicetell}_")
#         # if n == index : frametreeview.treeview.item(item, text=f"0d{slicetell}")





# class __TreeFrame__(ttk.Widget):
#     def __init__(self, master=None, **kwargs):
#         ttk.Widget.__init__(self, master, "ttk::frame", kwargs)
#         self.frametreeview = master
#         self.__bars__ = self.__添加__滚动__区域__()
#         self.treeview = self.__添加__表格__区域__()
#         self.__绑定函数__()
#         self.leftpopup = None
#         self.rightmenu = None

#     def __添加__滚动__区域__(self):
#         xbar = ttk.Scrollbar(self, orient='horizontal')   # 水平
#         xbar.pack(side="bottom", fill="x")                # must be top, bottom, left, or right
#         ybar = ttk.Scrollbar(self, orient='vertical')     # 垂直
#         ybar.pack(side="right", fill="y")
#         ybar.set(str(0.0), str(0.1)) 
#         return xbar, ybar
    
#     def __添加__表格__区域__(self):
#         columns = [f"#{i+1}" for i in range(32)]
#         treeview = ttk.Treeview(self, columns=columns)
#         treeview.pack(side='left', fill='both', expand=1)
#         treeview.column("#0", width=200, stretch=0)
#         for column, columnstitle in zip(columns, columns): # 设置列宽
#             treeview.column(column, width=40, stretch=False, anchor='center')
#             treeview.heading(column, text=columnstitle)
#         return treeview


#     def __绑定函数__(self):
#         xbar, ybar = self.__bars__
#         xbar.config(command=self.treeview.xview)
#         ybar.config(command=self.__滚条点击函数__)
#         self.treeview.config(xscrollcommand=xbar.set, yscrollcommand=None)

#         self.treeview.bind("<Motion>", self.__隐藏状态弹窗__)  
#         # self.treeview.bind('<Button-1>', self.__左键单击函数__)
#         # self.treeview.bind('<Button-3>', self.__右键单击函数__)
#         self.treeview.bind("<ButtonRelease-1>", self.__显示状态弹窗__) # 左键单击离开
#         self.treeview.bind("<ButtonRelease-3>", self.__显示右键菜单__) # 右键单击离开

#     def __点击更新状态__(self, event):
#         item = self.treeview.identify_row(event.y)
#         if item == "": return None
#         rowindex = self.treeview.index(item)
#         column = self.treeview.identify_column(event.x)
#         columnindex = int(column[1:]) if column != "" else 0
#         self.frametreeview.item = item
#         self.frametreeview.column = column
#         self.frametreeview.frameoption.rowindex = rowindex
#         self.frametreeview.frameoption.columnindex = columnindex
#         return item


#     def __显示状态弹窗__(self, event):
#         if self.__点击更新状态__(event) == None: return None
#         self.leftpopup.显示弹窗(event)

#     def __隐藏状态弹窗__(self, event):
#         if self.frametreeview.item == None: return None
#         self.leftpopup.隐藏弹窗(event)


#     def __显示右键菜单__(self, event):
#         if self.__点击更新状态__(event) == None: return None
#         self.rightmenu.post(event.x_root, event.y_root)
        

#     def __绑定__滚条__函数__(self):
#         xbar, ybar = self.__bars__
#         ybar.config(command=self.__滚条点击函数__)


#     def __滚条点击函数__(self, *args, **kwargs):
#         frameoption = self.frametreeview.frameoption
#         if args[0] == "moveto": frameoption.ybarfloat = float(args[1])
#         if args[0] == "scroll": frameoption.ybarrange = int(args[1])
#         self.__运行__滚条__函数__(args, kwargs)


#     def __运行__滚条__函数__(self, args, kwargs):
#         # print("滚条输出参数：", args, kwargs)
#         # # 滚条输出参数： args ('moveto', '0.004434589800443459') kwargs {}
#         # # 滚条输出参数： args ('scroll', '4', 'units') kwargs {}
#         # # 滚条输出参数： args ('scroll', '1', 'pages') kwargs {}
#         frameoption = self.frametreeview.frameoption
#         if "分段长度" not in frameoption.__dict__: return None
#         if args[0] == "moveto": frameoption.__函数__拖动显示__()
#         if args[0] == "scroll": frameoption.__函数__换页显示__()
#         ybarfloat = frameoption.段索引*frameoption.分段长度 + frameoption.行索引*frameoption.分行长度
#         ybarlength = frameoption.有效行数*frameoption.分行长度
#         xbar, ybar = self.__bars__
#         ybar.set(str(ybarfloat), str(ybarfloat+ybarlength)) 


# class __EditPopup__(tk.Toplevel):
#     def __init__(self, master=None, frameoption=None):
#         tk.Toplevel.__init__(self)
#         self.frametreeview = master
#         self.overrideredirect(True)
#         self.withdraw()
#         self.__添加__部件__()
#         self.__是否显示弹窗__ = False
#         self.__设置__按钮__属性__(frameoption)
#         self.__文件端序__ = "<"
#         self.__设置__端序__属性__(frameoption)


#     def 显示弹窗(self, event):
#         if self.__是否显示弹窗__ == False: return None
#         x2, y2, x_root, y_root = event.x, event.y, event.x_root, event.y_root
#         item, column = self.frametreeview.item, self.frametreeview.column
#         x1, y1, width, height = self.frametreeview.treeview.bbox(item=item, column=column)
#         x = x_root - (x2 - x1) + width
#         y = y_root - (y2 - y1) + height
#         width, height = 500, 205 # +23
#         self.geometry(f"{width}x{height}+{x}+{y}")
#         self.deiconify() # 显示窗口
#         self.__生成__显示__数据__()
#         self.__更新__部件__显示__()




#     def 隐藏弹窗(self, event):
#         None if self.__是否区域内__(event) else self.withdraw()


#     def __是否区域内__(self, event):
#         bbox = self.geometry()
#         width_height, x_root, y_root = bbox.split("+")
#         width, height = width_height.split("x")
#         width, height, x_root, y_root = int(width), int(height), int(x_root), int(y_root)
#         item, column = self.frametreeview.item, self.frametreeview.column
#         x1, y1, width1, height1 = self.frametreeview.treeview.bbox(item=item, column=column)
#         include = max(width1, height1)
#         boolwidth, boolheight = False, False
#         if x_root - include< event.x_root < x_root + width + include: boolwidth = True
#         if y_root - include< event.y_root < y_root + height + include: boolheight = True
#         return True if boolwidth and boolheight else False


#     def __添加__部件__(self):
#         frame = ttk.Frame(self)
#         frame.pack(side="left") # 
#         width = 25
#         row, column = 0, 0
#         self.__entry__uint8__,   row = Tree基础显示.__标签输入部件__(frame, text="  uint8：", row=row, column=column, width=width), row + 1
#         self.__entry__uint16__,  row = Tree基础显示.__标签输入部件__(frame, text=" uint16：", row=row, column=column, width=width), row + 1
#         self.__entry__uint32__,  row = Tree基础显示.__标签输入部件__(frame, text=" uint32：", row=row, column=column, width=width), row + 1
#         self.__entry__uint64__,  row = Tree基础显示.__标签输入部件__(frame, text=" uint64：", row=row, column=column, width=width), row + 1
#         frame.grid_rowconfigure((4,), minsize=6, weight=0)
#         row = 5
#         self.__entry__int8__,    row = Tree基础显示.__标签输入部件__(frame, text="   int8：", row=row, column=column, width=width), row + 1
#         self.__entry__int16__,   row = Tree基础显示.__标签输入部件__(frame, text="  int16：", row=row, column=column, width=width), row + 1
#         self.__entry__int32__,   row = Tree基础显示.__标签输入部件__(frame, text="  int32：", row=row, column=column, width=width), row + 1
#         self.__entry__int64__,   row = Tree基础显示.__标签输入部件__(frame, text="  int64：", row=row, column=column, width=width), row + 1

#         row, column = 0, column+2
#         self.__entry__char__,    row = Tree基础显示.__标签输入部件__(frame, text="   char：", row=row, column=column, width=width), row + 1
#         self.__entry__utf8__,    row = Tree基础显示.__标签输入部件__(frame, text="   utf8：", row=row, column=column, width=width), row + 1
#         row = 6
#         self.__entry__float16__, row = Tree基础显示.__标签输入部件__(frame, text="float16：", row=row, column=column, width=width), row + 1
#         self.__entry__float32__, row = Tree基础显示.__标签输入部件__(frame, text="float32：", row=row, column=column, width=width), row + 1
#         self.__entry__float64__, row = Tree基础显示.__标签输入部件__(frame, text="float64：", row=row, column=column, width=width), row + 1


#     def __设置__按钮__属性__(self, frameoption):
#         frameoption.__按钮__弹窗开闭__.configure(text="开启弹窗", command=self.__弹窗__切换__函数__)


#     def __弹窗__切换__函数__(self):
#         self.__是否显示弹窗__ = not self.__是否显示弹窗__
#         if self.__是否显示弹窗__:
#             self.frametreeview.frameoption.__按钮__弹窗开闭__["text"] = "关闭弹窗"
#         else:
#             self.frametreeview.frameoption.__按钮__弹窗开闭__["text"] = "开启弹窗"


#     def __设置__端序__属性__(self, frameoption):
#         frameoption.__按钮__端序切换__.configure(text="切换大端", command=self.__弹窗__端序__函数__)


#     def __弹窗__端序__函数__(self):
#         frameoption = self.frametreeview.frameoption
#         if self.__文件端序__ == "<":
#             self.__文件端序__ = ">"
#             frameoption.__按钮__端序切换__["text"] = "切换小端"
#         else:
#             self.__文件端序__ = "<"
#             frameoption.__按钮__端序切换__["text"] = "切换大端"
#         for bp in frameoption.选择集列表: bp.endian = self.__文件端序__


#     def __简化__前置__偏移__(self):
#         frameoption = self.frametreeview.frameoption
#         折算数量 = len(frameoption.选择集列表) * len(frameoption.格式列表)
#         范围行数 = frameoption.rowindex // 折算数量
#         列数列表 = frameoption.列数列表列表[frameoption.段索引][frameoption.行索引: frameoption.行索引+范围行数]
#         前置偏移 = frameoption.前置大小
#         for 列数 in 列数列表: 前置偏移 += 列数*frameoption.单格大小
#         前置偏移 += (frameoption.columnindex-1)*frameoption.单格大小
#         return 前置偏移

#     def __获取__数据__对象__(self):
#         frameoption = self.frametreeview.frameoption
#         数据数量 = len(frameoption.选择集列表) # 格式数量 = len(frameoption.格式列表)
#         折算数量 = len(frameoption.选择集列表) * len(frameoption.格式列表)
#         等效余数 = frameoption.rowindex % 折算数量
#         数据索引 = 等效余数 % 数据数量
#         return frameoption.选择集列表[数据索引]
    

#     def __生成__显示__数据__(self):
#         of = self.__简化__前置__偏移__()
#         bp = self.__获取__数据__对象__()
#         bp.index = bp.mpleft
#         bp.seekclosecheck(of)
#         self.__bp__char__ = bp.readcharseek0()
#         self.__bp__int8__  = bp.readint8seek0()
#         self.__bp__int16__ = bp.readint16seek0()
#         self.__bp__int32__ = bp.readint32seek0()
#         self.__bp__int64__ = bp.readint64seek0()
#         self.__bp__uint8__  = bp.readuint8seek0()
#         self.__bp__uint16__ = bp.readuint16seek0()
#         self.__bp__uint32__ = bp.readuint32seek0()
#         self.__bp__uint64__ = bp.readuint64seek0()
#         self.__bp__float16__ = bp.readfloat16seek0()
#         self.__bp__float32__ = bp.readfloat32seek0()
#         self.__bp__float64__ = bp.readfloat64seek0()


#     def __更新__部件__显示__(self):
#         self.__entry__char__.delete(0, "end"), self.__entry__char__.insert(0, f"{self.__bp__char__}") 
#         self.__entry__int8__.delete(0, "end"),  self.__entry__int8__.insert(0, f"{self.__bp__int8__}") 
#         self.__entry__int16__.delete(0, "end"), self.__entry__int16__.insert(0, f"{self.__bp__int16__}") 
#         self.__entry__int32__.delete(0, "end"), self.__entry__int32__.insert(0, f"{self.__bp__int32__}") 
#         self.__entry__int64__.delete(0, "end"), self.__entry__int64__.insert(0, f"{self.__bp__int64__}") 
#         self.__entry__uint8__.delete(0, "end"),  self.__entry__uint8__.insert(0, f"{self.__bp__uint8__}") 
#         self.__entry__uint16__.delete(0, "end"), self.__entry__uint16__.insert(0, f"{self.__bp__uint16__}") 
#         self.__entry__uint32__.delete(0, "end"), self.__entry__uint32__.insert(0, f"{self.__bp__uint32__}") 
#         self.__entry__uint64__.delete(0, "end"), self.__entry__uint64__.insert(0, f"{self.__bp__uint64__}") 
#         self.__entry__float16__.delete(0, "end"), self.__entry__float16__.insert(0, f"{self.__bp__float16__}") 
#         self.__entry__float32__.delete(0, "end"), self.__entry__float32__.insert(0, f"{self.__bp__float32__}") 
#         self.__entry__float64__.delete(0, "end"), self.__entry__float64__.insert(0, f"{self.__bp__float64__}")    
