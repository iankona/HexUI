import tkinter as tk
import tkinter.ttk as ttk
import tkinter.filedialog
import tkinter.scrolledtext

import os
import 底层编辑

from . import Tree基础编辑



class __列表历史__:
    def __init__(self):
        self.临时列表 = None
        self.列表计数 = -1
        self.列表列表 = [None for i in range(50)]

    def 插入列表(self, 列表):
        self.列表计数 = (self.列表计数 + 1) % 50
        self.列表列表[self.列表计数] = 列表


    def 获取列表(self):
        if self.列表计数 < 0: return None
        return self.列表列表[self.列表计数]

    def 后退列表(self):
        self.列表计数 = (self.列表计数 - 1) % 50 # (-1) % 10 == 9
        return self.列表列表[self.列表计数]

    def 前进列表(self):
        self.列表计数 = (self.列表计数 + 1) % 50
        return self.列表列表[self.列表计数]

列表历史 = __列表历史__()


class __FileInfo__:
    def __init__(self):
        self.filepath = None
        self.basename = None
        self.filebnode = None
        self.nummaterial = None




class TreeFrame(ttk.Widget):
    def __init__(self, master=None, **kwargs):
        ttk.Widget.__init__(self, master, "ttk::frame", kwargs)
        self.pathdict = {}
        self.item = None
        self.path = None
        self.column = None
        self.__添加__部件__()


    def insertvalue(self, fitem="", fileinfo=None):
        item = self.treeview.insert(fitem, index="end", text=fileinfo.basename, values=[0, fileinfo.nummaterial, fileinfo.filepath])
        self.pathdict[item] = fileinfo
        return item


    def itemvalue(self, item="", index=0):
        fileinfo = self.pathdict[item]
        self.treeview.item(item, values=[index, fileinfo.nummaterial, fileinfo.filepath])
        return item


    def __添加__部件__(self):
        framea = self.__添加__顶部__菜单__()
        frameb = self.__添加__选项__区域__()
        panela = self.__添加__面板__区域__()
        framec = self.__添加__表格__区域__()
        framed = self.__添加__日志__区域__()
        panela.add(framec)
        panela.add(framed)
        popupa = self.__添加__左键__弹窗__()
        rmenua = self.__添加__右键__菜单__()
        self.treeview = framec.treeview
        framec.leftpopup = popupa
        framec.rightmenu = rmenua
        self.frametoolbar = framea
        self.frameoption = frameb
        self.__frametreeview__ = framec
        self.frametext = framed
        self.rightmenu = rmenua


    def __添加__顶部__菜单__(self):
        frame = __MenuFrame__(self)
        frame.pack(side="top", fill="x")
        return frame

    def __添加__选项__区域__(self):
        frame = __ItemFrame__(self)
        frame.pack(side='top', fill='x')
        return frame

    def __添加__面板__区域__(self):
        panel = ttk.PanedWindow(self, orient="vertical")
        panel.pack(side="top", fill='both', expand=1)
        return panel

    def __添加__表格__区域__(self):
        frame = __TreeFrame__(self, height=500)
        frame.pack_propagate(0) # 同时pack_propagate(0)，才能调整面板高度
        frame.pack(side='left', fill='both', expand=1)
        return frame

    def __添加__日志__区域__(self):
        frame = Tree基础编辑.__TextFrame__(self, height=100)
        frame.pack_propagate(0) # 阻止高宽向子级传播
        frame.pack(side='left', fill='both', expand=1)
        return frame
    
    def __添加__左键__弹窗__(self):
        popup = __EditPopup__(self)
        return popup

    def __添加__右键__菜单__(self):
        menu = tk.Menu(None, tearoff=False)
        menu.add_command(label="打开文件", command=lambda:批量打开文件(self))
        menu.add_separator() # 添加分割线
        menu.add_command(label='删除节点', command=lambda:删除节点(self))
        menu.add_separator() # 添加分割线
        menu.add_command(label='清空节点', command=lambda:清空节点(self))
        menu.add_separator() # 添加分割线
        menu.add_command(label='退回上次编辑', command=self.__退回上次编辑__)
        menu.add_command(label='返回下次编辑', command=self.__返回下次编辑__)
        menu.add_separator() # 添加分割线
        menu.add_command(label='文件至右侧显示', command=None)
        return menu

    def __退回上次编辑__(self):
        列表 = 列表历史.后退列表()
        if 列表 == None: return None
        items = self.treeview.get_children()
        for item, index in zip(items, 列表): self.itemvalue(item, index=index)
        self.frametext.插入文本(f"批量设置索引::退回上次编辑{列表}")

    def __返回下次编辑__(self):
        列表 = 列表历史.前进列表()
        if 列表 == None: return None
        items = self.treeview.get_children()
        for item, index in zip(items, 列表): self.itemvalue(item, index=index)
        self.frametext.插入文本(f"批量设置索引::返回下次编辑{列表}")



def 批量打开文件(frametreeview):
    filepaths = tkinter.filedialog.askopenfilenames(filetypes=[("All files", "*"),])
    filepaths = [filepath for filepath in filepaths if filepath != "" and os.path.isfile(filepath) == True]
    filepaths = [filepath for filepath in filepaths if filepath.endswith(".model")]
    if filepaths == []: return None
    for filepath in filepaths:
        fileinfo = __FileInfo__()
        fileinfo.filepath = filepath
        fileinfo.basename = os.path.basename(filepath)
        fileinfo.filebnode = 底层编辑.model.类().read(filepath)
        fileinfo.nummaterial = fileinfo.filebnode.name_object_dict["SRTM"]["材质数量"].uint32
        frametreeview.insertvalue(fitem="", fileinfo=fileinfo)
    
    frametreeview.frametext.插入列表(filepaths, 首行="打开文件::")

    nummaterial_list = []
    for item, fileinfo in frametreeview.pathdict.items(): nummaterial_list.append(fileinfo.nummaterial)
    frametreeview.frameoption.列表选取["values"] = [i for i in range(min(nummaterial_list))]
    __保存索引修改历史__(frametreeview)

def 批量保存文件(frametreeview, 后缀=""): # 保存浏览文件多选(dirname, basename):
    filedire = tkinter.filedialog.askdirectory(#initialdir=dirname, 
                                               title="保存文件夹"
                                              ) # print(filedire) E:/Program_StructFiles/Avatar_Kiana_C4_YN/Avatar_Kiana_C4_Ani_Attack_QTE_AS
    frametreeview.frametext.插入文本(f"保存文件夹::{filedire}")
    filepaths = []
    for item, fileinfo in frametreeview.pathdict.items():
        filepath = f"{filedire}/{fileinfo.basename[0:-6]}{后缀}.model"
        filepaths.append(filepath)
        fileinfo.filebnode.write(filepath)
    
    frametreeview.frametext.插入列表(filepaths, 首行="保存文件::")

def 删除节点(frametreeview):
    items = frametreeview.treeview.selection()
    for item in items:
        frametreeview.treeview.delete(item)
        fileinfo = frametreeview.pathdict.pop(item)
        frametreeview.frametext.插入文本(f"删除节点::{fileinfo.basename}")
    frametreeview.item = None
    frametreeview.path = None
    __保存索引修改历史__(frametreeview)

def 清空节点(frametreeview):
    items = frametreeview.treeview.get_children()
    for item in items: frametreeview.treeview.delete(item)
    frametreeview.item = None
    frametreeview.path = None
    frametreeview.pathdict = {}

def 清空文本(frametreeview):
    frametreeview.frametext.清空文本()


class __MenuFrame__(ttk.Widget):
    def __init__(self, master=None, **kwargs):
        ttk.Widget.__init__(self, master, "ttk::frame", kwargs)
        self.frametreeview = master
        self.添加部件()

    def 添加部件(self):
        self.menufile = self.__添加菜单__(label='文件', menu=self.__文件菜单区__())
        self.menuedit = self.__添加菜单__(label='编辑', menu=self.__编辑菜单区__())
        self.menutext = self.__添加菜单__(label='日志', menu=self.__日志菜单区__())


    def __添加菜单__(self, label='', menu=None):
        menubutton = ttk.Menubutton(self, text=label)
        menubutton.pack(side="left")
        menubutton.config(menu=menu)
        return menu


    def __文件菜单区__(self):
        menu=tk.Menu(None, tearoff=False) # 关闭菜单可拖动
        menu.add_command(label='打开文件',command=lambda:批量打开文件(self.frametreeview)) # menu.entryconfigure("打开", command=lambda :print('123'))
        return menu


    def __编辑菜单区__(self):       
        menu=tk.Menu(None, tearoff=False)
        menu.add_command(label='清空节点',command=lambda:清空节点(self.frametreeview))
        return menu


    def __日志菜单区__(self):
        menu=tk.Menu(None, tearoff=False)
        menu.add_command(label='清空文本',command=lambda:清空文本(self.frametreeview))
        return menu

class __ItemFrame__(ttk.Widget):
    def __init__(self, master=None, **kwargs):
        ttk.Widget.__init__(self, master, "ttk::frame", kwargs)
        self.frametreeview = master
        self.__添加__选项__区域__()

    def __添加__选项__区域__(self):
        row, column = 0, 0
        buttona = Tree基础编辑.__标签按钮部件__(self, text="批量设置索引:", row=row, column=column, width=12)
        column += 2
        comboxa = Tree基础编辑.__标签选取部件__(self, text="", row=row, column=column, width=13)
        column += 2
        buttonb = Tree基础编辑.__标签按钮部件__(self, text="批量替换节点", row=row, column=column, width=16)
        column += 2

        comboxa.bind('<<ComboboxSelected>>', self.__批量设置索引__)
        self.列表选取 = comboxa
        self.按钮替换 = buttonb

    def __批量设置索引__(self, event):
        index = self.列表选取.get()
        items = self.frametreeview.treeview.get_children()
        for item in items: self.frametreeview.itemvalue(item, index=index)
        __保存索引修改历史__(self.frametreeview)
        self.frametreeview.frametext.插入文本(f"批量设置索引::{index}")

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
        indexs = ["#1", "#2", "#3"]
        widths = [80, 80, 600]
        titles = ["替换索引", "材质数量", "文件路径"]
        treeview = ttk.Treeview(self, columns=indexs)
        treeview.pack(side='left', fill='both', expand=1)
        treeview.column("#0", width=250, stretch=0)
        treeview.heading("#0", text="文件名称")
        for column, width, title in zip(indexs, widths, titles): 
            treeview.column(column, width=width, stretch=False, anchor='center')
            treeview.heading(column, text=title)
        return treeview


    def __绑定函数__(self):
        xbar, ybar = self.__bars__
        xbar.config(command=self.treeview.xview)
        ybar.config(command=self.treeview.yview)
        self.treeview.config(xscrollcommand=xbar.set, yscrollcommand=ybar.set)

        # self.treeview.bind("<Button-1>", self.__左键单击函数__) 
        # self.treeview.bind("<Button-3>", self.__右键单击函数__) 
        self.treeview.bind('<Double-1>', self.__显示双击弹窗__)
        # self.treeview.bind("<ButtonRelease-1>", self.__显示左键弹窗__) # 左键单击离开
        self.treeview.bind("<ButtonRelease-3>", self.__显示右键菜单__) # 右键单击离开

    def __点击更新函数__(self, event):
        item = self.treeview.identify_row(event.y)
        if item == "": return None, None # 处理空表格
        column = self.treeview.identify_column(event.x)
        self.frametreeview.item = item
        self.frametreeview.path = self.frametreeview.pathdict[item]
        self.frametreeview.column = column
        return item, column

    def __显示右键菜单__(self, event):
        self.__点击更新函数__(event)
        self.rightmenu.post(event.x_root, event.y_root)

    def __显示双击弹窗__(self, event):
        item, column = self.__点击更新函数__(event)
        if item == None or column != "#1": return None
        self.leftpopup.显示弹窗(event)


class __EditPopup__(tk.Toplevel):
    def __init__(self, master=None):
        tk.Toplevel.__init__(self)
        self.frametreeview = master
        self.overrideredirect(True)  # 关闭外框 # 开启外框后，会自动隐藏到窗口后面，锁定窗口焦点无效，原因未明
        self.withdraw()
        self.输入框 = self.__添加__输入框__部件__()


    def 显示弹窗(self, event):
        x2, y2, x_root, y_root = event.x, event.y, event.x_root, event.y_root
        item, column = self.frametreeview.item, self.frametreeview.column
        x1, y1, width, height = self.frametreeview.treeview.bbox(item=item, column=column)
        x = x_root - (x2 - x1)
        y = y_root - (y2 - y1)
        self.geometry(f"{width}x{height}+{x}+{y}")
        self.deiconify() # 显示窗口
        self.grab_set() # 锁定焦点在窗口 # Python tkinter Misc类+Wm类详解

        默认值 = self.frametreeview.treeview.item(item)["values"][0]
        self.输入框.delete(0, "end")
        self.输入框.insert(index=0, string=默认值)


    def 隐藏弹窗(self):
        self.grab_release() # 释放焦点
        self.withdraw() # 隐藏窗口

    def __添加__输入框__部件__(self):
        输入框 = ttk.Entry(self)
        输入框.pack(side="left", fill="both", expand=1)
        输入框.bind("<Return>", self.__输入框内容保存到单元格__)
        return 输入框

    def __输入框内容保存到单元格__(self, event):
        self.隐藏弹窗()
        try:
            item, fileinfo = self.frametreeview.item, self.frametreeview.path
            index = int(self.输入框.get())
            if index < 0: index = 0
            if index >= fileinfo.nummaterial: index = fileinfo.nummaterial - 1
            self.frametreeview.itemvalue(item, index=index)
            __保存索引修改历史__(self.frametreeview)
        except Exception as e:
            print(e)

def __保存索引修改历史__(frametreeview):
    items = frametreeview.treeview.get_children()
    列表 = []
    for item in items: 
        values = frametreeview.treeview.item(item)["values"]
        列表.append(values[0])
    if 列表 != []: 列表历史.插入列表(列表)






