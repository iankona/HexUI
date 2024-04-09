
import tkinter as tk
import tkinter.ttk as ttk
import tkinter.filedialog

import os
import 底层文件

from . import Tree右键菜单


class __Tree右键菜单__:
    def __init__(self, frametreeview, 菜单):
        self.menu = Tree右键菜单.stream2tkinter.类(frametreeview, 菜单)


class TreeFrame(ttk.Widget):
    def __init__(self, master=None, **kwargs):
        ttk.Widget.__init__(self, master, "ttk::frame", kwargs)
        self.bpdict = {}
        self.__添加__部件__()

    def insertblock(self, fitem="", text="", bp=None):
        item = self.treeview.insert(fitem, index="end", text=text, values=[uint8 for uint8 in bp.readseek0(32)])
        self.bpdict[item] = bp
        return item

    def insertbytes(self, fitem="", text="", bp=None):
        item = self.treeview.insert(fitem, index="end", text=text, values=[uint8 for uint8 in bp.readseek0(32)])
        self.bpdict[item] = bp
        return item

    def insertvalue(self, fitem="", text="", values=[]):
        item = self.treeview.insert(fitem, index="end", text=text, values=values[0:32])
        self.bpdict[item] = None
        return item
    
    def itemblock(self, item="", text="", bp=None):
        if text == "": 
            self.treeview.item(item, values=[uint8 for uint8 in bp.readseek0(32)])
        else:
            self.treeview.item(item, text=text, values=[uint8 for uint8 in bp.readseek0(32)])
        self.bpdict[item] = bp
        return item

    def itembytes(self, item="", text="", bp=None):
        if text == "": 
            self.treeview.item(item, values=[uint8 for uint8 in bp.readseek0(32)])
        else:
            self.treeview.item(item, text=text, values=[uint8 for uint8 in bp.readseek0(32)])
        self.bpdict[item] = bp
        return item
    
    def itemvalue(self, item="", text="", values=[]):
        if text == "": 
            self.treeview.item(item, values=values)
        else:
            self.treeview.item(item, text=text, values=values)
        self.bpdict[item] = None
        return item



    def __添加__部件__(self):
        frameb = self.__添加__选项__区域__()
        framec = self.__添加__表格__区域__()
        rmenua = self.__添加__右键__菜单__()
        self.treeview = framec.treeview
        framec.rightmenu = rmenua
        self.frameoption = frameb
        self.__frametreeview__ = framec
        self.rightmenu = rmenua

    def __添加__选项__区域__(self):
        frame = __ItemFrame__(self)
        frame.pack(side='top', fill='x')
        return frame

    def __添加__表格__区域__(self):
        frame = __TreeFrame__(self, height=500)
        frame.pack_propagate(0) # 同时pack_propagate(0)，才能调整面板高度
        frame.pack(side='left', fill='both', expand=1)
        return frame

    def __添加__右键__菜单__(self):
        menu = tk.Menu(None, tearoff=False)
        menu.add_command(label="打开文件", command=lambda:批量打开文件(self))
        menu.add_command(label="打开文件夹", command=lambda:打开文件夹(self))
        menu.add_separator() # 添加分割线
        menu.add_command(label="删除行", command=lambda:删除节点(self))
        menu.add_command(label="复制行标题", command=self.__复制行标题__)
        menu.add_separator() # 添加分割线
        self.__format__menu__ = __Tree右键菜单__(self, menu)
        menu.add_separator() # 添加分割线
        menu.add_command(label="数据集to右边", command=None)
        return menu

    def __复制行标题__(self):
        item = self.treeview.identify_row(self.event.y)
        chars = self.treeview.item(item)["text"]
        self.clipboard_clear()
        self.clipboard_append(chars)



def 批量打开文件(frametreeview):
    filepaths = tkinter.filedialog.askopenfilenames()
    filepaths = [filepath for filepath in filepaths if filepath != "" and os.path.isfile(filepath) == True]
    for filepath in filepaths:
        basename = os.path.basename(filepath)           # 带扩展名
        frametreeview.insertblock("", text=basename, bp=底层文件.bpformat.bpnumpy.类().filepath(filepath))


def 打开文件夹(frametreeview):
    filedir = tkinter.filedialog.askdirectory()
    if filedir == "": return None # FileNotFoundError: [WinError 3] 系统找不到指定的路径。: ''
    
    filepaths = []
    for filename in os.listdir(filedir):
        filepath = os.path.join(filedir, filename)
        if os.path.isfile(filepath): filepaths.append(filepath)
        
    for filepath in filepaths:
        basename = os.path.basename(filepath)           # 带扩展名
        frametreeview.insertblock("", text=basename, bp=底层文件.bpformat.bpnumpy.类().filepath(filepath))





def 删除节点(frametreeview):
    selection = frametreeview.treeview.selection()
    rootitems = frametreeview.treeview.get_children()
    for item in selection:
        if item in rootitems: frametreeview.bpdict[item].close()
        try:
            frametreeview.treeview.delete(item)
            frametreeview.bpdict.pop(item)
        except Exception as e:
            pass # print(e)




def __标签输入部件__(self, text="", row=0, column=0, width=15, value=""):
    标签 = ttk.Label(self, text=text)
    标签.grid(row=row, column=column, sticky="e")
    输入 = ttk.Entry(self, width=width)
    输入.insert(index=0, string=value)
    输入.grid(row=row, column=column+1, sticky="we")
    return 输入


def __标签标签部件__(self, text="", row=0, column=0, width=15):
    标签 = ttk.Label(self, text=text)
    标签.grid(row=row, column=column, sticky="e")
    输出 = ttk.Label(self, width=width)
    输出.grid(row=row, column=column+1, sticky="we")
    return 输出

def __标签选取部件__(self, text="", row=0, column=0, width=15, values=[0, 1, 2, 3, 4, 5, 6, 7], index=0):
    标签 = ttk.Label(self, text=text)
    标签.grid(row=row, column=column, sticky="e")
    选取 = ttk.Combobox(self, width=width, values=values)
    选取.current(index) # 选择第1个
    选取.grid(row=row, column=column+1, sticky="we")
    return 选取

def __标签按钮部件__(self, text="", row=0, column=0, width=15, command=None):
    按钮 = ttk.Button(self, text=text, width=width, command=command)
    按钮.grid(row=row, column=column, sticky="we")
    return 按钮


class __ItemFrame__(ttk.Widget):
    def __init__(self, master=None, **kwargs):
        ttk.Widget.__init__(self, master, "ttk::frame", kwargs)
        self.frametreeview = master
        self.__添加__参数__部件__()

    def __添加__参数__部件__(self):
        width = 42
        row, column = 0, 0
        self.索引列表, row = __标签输入部件__(self, text="索引列表：", row=row, column=column, width=width, value="48/56/128"), row + 1
        self.大小列表, row = __标签输入部件__(self, text="大小列表：", row=row, column=column, width=width, value="24/48/96"), row + 1
        self.状态显示, row = __标签输入部件__(self, text="状态显示：", row=row, column=column, width=width, value=""), row + 1

        width = 10
        row, column = 0, column + 2
        self.索引分割, row = __标签按钮部件__(self, text="索引分割", row=row, column=column, width=width, command=self.__按索引新建子节点__), row + 1
        self.大小分割, row = __标签按钮部件__(self, text="大小分割", row=row, column=column, width=width, command=self.__按大小新建子节点__), row + 1
        self.层级分割, row = __标签按钮部件__(self, text="层级分割", row=row, column=column, width=width, command=self.__按索引新建层节点__), row + 1


    def __按索引新建子节点__(self):
        numchars = self.索引列表.get().split("/")
        numbers = [0] + [int(numchar) for numchar in numchars if numchar != ""]
        sizes = [ numbers[i+1]-numbers[i] for i in range(len(numbers)-1)]
        sizes = [ size for size in sizes if size > 0 ]
        for item in self.frametreeview.treeview.selection(): self.__新建子节点__(item, sizes)


    def __按大小新建子节点__(self):
        numchars = self.大小列表.get().split("/")
        sizes = [int(numchar) for numchar in numchars if numchar != ""]
        sizes = [ size for size in sizes if size > 0 ]
        for item in self.frametreeview.treeview.selection(): self.__新建子节点__(item, sizes)


    def __新建子节点__(self, item, sizes):
        bp = self.frametreeview.bpdict[item].copy()
        bps = [bp.readslice(size) for size in sizes]
        sizes.append(bp.remainsize())
        bps.append(bp.readremainslice())
        for size, bp in zip(sizes, bps): citem = self.frametreeview.insertblock(item, text=f"块_{size}", bp=bp)  
        char = self.frametreeview.treeview.item(citem)["text"] 
        self.frametreeview.treeview.item(citem, text="余"+char)


    def __新建层节点__(self, item, sizes):
        bp = self.frametreeview.bpdict[item].copy()
        bps = [bp.readslice(size) for size in sizes]
        sizes.append(bp.remainsize())
        bps.append(bp.readremainslice())

        fitem = self.frametreeview.treeview.parent(item)
        if fitem == "":
            for size, bp in zip(sizes, bps): citem = self.frametreeview.insertblock(item, text=f"块_{size}", bp=bp)
        else:
            citem = self.frametreeview.itemblock(item, text=f"块_{sizes[0]}", bp=bps[0])
            for size, bp in zip(sizes[1:], bps[1:]): citem = self.frametreeview.insertblock(fitem, text=f"块_{size}", bp=bp) 

        char = self.frametreeview.treeview.item(citem)["text"] 
        self.frametreeview.treeview.item(citem, text="余"+char)
        
    def __按索引新建层节点__(self):
        numchars = self.索引列表.get().split("/")
        numbers = [0] + [int(numchar) for numchar in numchars if numchar != ""]
        sizes = [ numbers[i+1]-numbers[i] for i in range(len(numbers)-1)]
        sizes = [ size for size in sizes if size > 0 ]
        for item in self.frametreeview.treeview.selection(): self.__新建层节点__(item, sizes)


    def __按大小新建层节点__(self):
        numchars = self.大小列表.get().split("/")
        sizes = [int(numchar) for numchar in numchars if numchar != ""]
        sizes = [ size for size in sizes if size > 0 ]
        for item in self.frametreeview.treeview.selection(): self.__新建层节点__(item, sizes)
  


class __TreeFrame__(ttk.Widget):
    def __init__(self, master=None, **kwargs):
        ttk.Widget.__init__(self, master, "ttk::frame", kwargs)
        self.frametreeview = master
        self.__bars__ = self.__添加__滚动__区域__()
        self.treeview = self.__添加__表格__区域__()
        self.__绑定函数__()
        self.leftpopup = None
        self.rightmenu = None

    def __添加__滚动__区域__(self):
        xbar = ttk.Scrollbar(self, orient='horizontal')   # 水平
        xbar.pack(side="bottom", fill="x")                # must be top, bottom, left, or right
        ybar = ttk.Scrollbar(self, orient='vertical')     # 垂直
        ybar.pack(side="right", fill="y")
        return xbar, ybar
    
    def __添加__表格__区域__(self):
        columns = [f"#{i+1}" for i in range(32)]
        treeview = ttk.Treeview(self, columns=columns)
        treeview.pack(side='left', fill='both')
        treeview.column("#0", width=200, stretch=0)
        for column, columnstitle in zip(columns, columns): # 设置列宽
            treeview.column(column, width=40, stretch=False, anchor='center')
            treeview.heading(column, text=columnstitle)
        return treeview


    def __绑定函数__(self):
        xbar, ybar = self.__bars__
        xbar.config(command=self.treeview.xview)
        ybar.config(command=self.treeview.yview)
        self.treeview.config(xscrollcommand=xbar.set, yscrollcommand=ybar.set)
        self.treeview.bind("<ButtonRelease-3>", self.__显示右键菜单__) # 右键单击离开


    def __显示右键菜单__(self, event):
        self.rightmenu.post(event.x_root, event.y_root)

