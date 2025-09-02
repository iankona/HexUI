import tkinter.ttk as ttk

from . import Model材质替换
from . import Model批量替换

class Frame(ttk.Widget):
    def __init__(self, master=None, **kw):
        ttk.Widget.__init__(self, master, "ttk::frame", kw)
        self.__notebook__ = ttk.Notebook(self, padding=5) 
        self.__notebook__.pack(fill="both", expand=1)
        self.__notebook__.add(__添加材质替换面板__(self.__notebook__), text=f'　材质替换　')
        self.__notebook__.add(__添加批量替换面板__(self.__notebook__), text=f'　批量替换　')


def __添加材质替换面板__(parent):
    frame = Model材质替换.PanelFrame(parent)
    frame.pack(side="left", fill='both')
    return frame

def __添加批量替换面板__(parent):
    frame = Model批量替换.PanelFrame(parent)
    frame.pack(side="left", fill='both')
    return frame


import tkinter as tk
import tkinter.ttk as ttk



from . import 编辑Model



class PanelFrame(ttk.Widget):
    def __init__(self, master=None, **kwargs):
        ttk.Widget.__init__(self, master, "ttk::frame", kwargs)
        self.__添加__部件__()

    def __添加__部件__(self):
        panela = self.__添加__面板__区域__() # Panel 实例在前，界面才能正常显示
        framea = self.__添加__左边__区域__()
        frameb = self.__添加__右边__区域__()
        panela.add(framea)
        panela.add(frameb)
        self.frametree = framea
        self.frameedit = frameb


    def __添加__面板__区域__(self):
        panel = ttk.PanedWindow(self, orient="horizontal")
        panel.pack(side="top", fill='both', expand=1)
        return panel

    def __添加__左边__区域__(self):
        frame = 编辑Model.FrameModel(self, width=450, height=600)
        frame.pack_propagate(0)
        frame.pack(side="left", fill='both')
        return frame
    
    def __添加__右边__区域__(self):
        frame = 编辑Model.FrameModel(self, width=550, height=600)
        frame.pack_propagate(0)
        frame.pack(side="left", fill='both')
        return frame




import tkinter as tk
import tkinter.ttk as ttk


from . import Tree基础编辑
from . import Tree基础路径
from . import 编辑Model






class PanelFrame(ttk.Widget):
    def __init__(self, master=None, **kwargs):
        ttk.Widget.__init__(self, master, "ttk::frame", kwargs)
        self.__添加__部件__()
        self.__绑定__按钮__函数__()
        self.__绑定__显示__函数__()



    def __添加__部件__(self):
        panela = self.__添加__面板__区域__() # Panel 实例在前，界面才能正常显示
        framea = self.__添加__左边__区域__()
        frameb = self.__添加__中间__区域__()
        framec = self.__添加__右边__区域__()
        panela.add(framea)
        panela.add(frameb)
        panela.add(framec)
        self.frameread = framea
        self.framepath = frameb
        self.frameedit = framec


    def __添加__面板__区域__(self):
        panel = ttk.PanedWindow(self, orient="horizontal")
        panel.pack(side="top", fill='both', expand=1)
        return panel

    def __添加__左边__区域__(self):
        frame = TreeFrame1(self, width=300, height=600)
        frame.pack_propagate(0)
        frame.pack(side="left", fill='both')
        return frame
    

    def __添加__中间__区域__(self):
        frame = Tree基础路径.TreeFrame(self, width=350, height=600)
        frame.pack_propagate(0)
        frame.pack(side="left", fill='both')
        return frame


    def __添加__右边__区域__(self):
        frame = TreeFrame3(self, width=300, height=600)
        frame.pack_propagate(0)
        frame.pack(side="left", fill='both')
        return frame


    def __绑定__按钮__函数__(self):
        # 以下三种修改按钮执行函数的方法，任选一种就行
        self.framepath.frameoption.按钮替换["command"] = self.__批量替换节点__
        # self.framepath.frameoption.按钮替换.config(command=self.__批量替换节点__)
        # self.framepath.frameoption.按钮替换.configure(command=self.__批量替换节点__)
        self.frameedit.frameoption.按钮保存.configure(command=self.__批量保存文件__)

    def __批量替换节点__(self):
        bnode_select = self.frameread.bnode_select
        if bnode_select == None: 
            self.framepath.frametext.插入文本(f"替换节点::请先在左侧选择节点...")
            return None

        items = self.framepath.treeview.get_children()
        indexs = [] 
        for item in items:
            values = self.framepath.treeview.item(item)["values"]
            indexs.append(values[0])
        filebnodes = []
        for item in items:
            fileinfo = self.framepath.pathdict[item]
            filebnodes.append(fileinfo.filebnode)

        for filebnode, index in zip(filebnodes, indexs): __替换节点__(bnode_select, filebnode, index, self.framepath)


    def __批量保存文件__(self):
        后缀 = self.frameedit.frameoption.输入后缀.get()
        Tree基础路径.批量保存文件(self.framepath, 后缀)


    def __绑定__显示__函数__(self):
        # self.framepath.treeview.bind("<<TreeviewSelect>>", self.__刷新__显示__函数__) # 有event
        self.framepath.rightmenu.entryconfig("文件至右侧显示", command=self.__刷新__显示__函数__) # 没有event

        
    def __刷新__显示__函数__(self):
        item = self.framepath.treeview.selection()[0]
        fileinfo = self.framepath.pathdict[item]
        Tree基础编辑.清空节点(self.frameedit)
        Tree基础编辑.递归显示(self.frameedit, fitem="", fbnode=fileinfo.filebnode)
        self.framepath.frametext.插入文本(f"刷新显示::已选文件{fileinfo.basename}，右侧显示同步刷新")
        self.frameedit.frametext.插入文本(f"文件显示::{fileinfo.filepath}")


def __替换节点__(bnode_select, filebnode, index, frametreeview):
    srtmbg = filebnode.name_object_dict["SRTM"]
    keyname = f"lrtm_{index}"
    bnode_temp = srtmbg[keyname]
    srtmbg[keyname] = bnode_select
    frametreeview.frametext.插入文本(f"替换节点::{bnode_temp.name} --> {bnode_select.name}", f"{bnode_temp} --> {bnode_select}")



    

class TreeFrame1(编辑Model.FrameModel):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.__释放__文件__菜单__()
        self.__删除__右键__菜单__()
        self.__附加__选项__区域__()
        self.bnode_select = None

    def __释放__文件__菜单__(self):
        self.frametoolbar.buttonfile.destroy()


    def __删除__右键__菜单__(self):
        self.rightmenu.delete('复制节点')
        self.rightmenu.delete('粘贴节点')
        self.rightmenu.delete('回退粘贴')
        self.rightmenu.add_command(label="选择节点", command=self.__选择节点__)



    def __附加__选项__区域__(self):
        row, column = 0, 0
        buttona = Tree基础编辑.__标签按钮部件__(self.frameoption, text="材质节点已选索引:", row=row, column=column, width=15)
        column += 2
        comboxa = Tree基础编辑.__标签选取部件__(self.frameoption, text="", row=row, column=column, width=20)
        column += 2
        self.frameoption.combox_select = comboxa



    def __打开文件__(self):
        super().__打开文件__()
        srtmbg = self.filebnode.name_object_dict["SRTM"]
        材质数量 = srtmbg["材质数量"].uint32
        self.frameoption.combox_select["values"] = [i for i in range(材质数量)]


    def __选择节点__(self):
        if "lrtm_" not in self.bnode.name: 
            self.frametext.插入文本("选择节点::您所选择的节点不是lrtm节点")
            return None
        index = int(self.bnode.name.split("_")[-1])
        self.frameoption.combox_select.current(index)
        self.bnode_select = self.bnode
        self.frametext.插入文本(f"选择节点::{self.bnode.name}", f"{self.bnode}")


class TreeFrame3(编辑Model.FrameModel):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.__释放__文件__菜单__()
        self.__删除__右键__菜单__()
        self.__附加__选项__区域__()


    def __释放__文件__菜单__(self):
        self.frametoolbar.buttonfile.destroy()

    def __删除__右键__菜单__(self):
        self.rightmenu.delete(2) # 索引需先删除
        self.rightmenu.delete("打开文件") # 0
        # self.rightmenu.delete("保存文件") # 1
        self.rightmenu.delete('复制节点')
        self.rightmenu.delete('粘贴节点')
        self.rightmenu.delete('回退粘贴')


    def __附加__选项__区域__(self):
        row, column = 0, 0
        button = Tree基础编辑.__标签按钮部件__(self.frameoption, text="批量后缀:", row=row, column=column, width=8)
        column += 2
        entrya = Tree基础编辑.__标签输入部件__(self.frameoption, text="", row=row, column=column, width=13)
        column += 2
        labela = Tree基础编辑.__标签标签部件__(self.frameoption, text=".model", row=row, column=column, width=0)
        column += 2
        button = Tree基础编辑.__标签按钮部件__(self.frameoption, text="批量保存", row=row, column=column, width=13)
        column += 2
        self.frameoption.输入后缀 = entrya
        self.frameoption.按钮保存 = button

        
import tkinter as tk
import tkinter.ttk as ttk

import tkinter.filedialog
import tkinter.scrolledtext


class TreeFrame(ttk.Widget):
    def __init__(self, master=None, **kwargs):
        ttk.Widget.__init__(self, master, "ttk::frame", kwargs)
        self.filebnode = None        
        self.item = None
        self.bnode = None
        self.bnodedict = {}
        self.__添加__部件__()


    def insertblock(self, fitem="", bnode=None):
        item = self.treeview.insert(fitem, index="end", text=bnode.name, values=[uint8 for uint8 in bnode.slice0b[0:32]], open=True)
        self.bnodedict[item] = bnode
        return item


    def itemblock(self, item="", bnode=None):
        self.treeview.item(item, text=bnode.name, values=[uint8 for uint8 in bnode.slice0b[0:32]], open=True)
        self.bnodedict[item] = bnode
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
        self.leftpopup = popupa
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
        frame = __TextFrame__(self, height=100)
        frame.pack_propagate(0) # 阻止高宽向子级传播
        frame.pack(side='left', fill='both', expand=1)
        return frame

    def __添加__左键__弹窗__(self):
        popup = __EditPopup__(self)
        return popup

    def __添加__右键__菜单__(self):
        menu = tk.Menu(None, tearoff=False)
        menu.add_command(label="打开文件", command=None)
        menu.add_command(label="保存文件", command=None)
        return menu



def 递归显示(frametreeview, fitem="", fbnode=None):
    for key, cbnode in fbnode.name_object_dict.items():
        citem = frametreeview.insertblock(fitem, bnode=cbnode)
        if "name_object_dict" in cbnode.__dict__: 递归显示(frametreeview, citem, cbnode)

def 清空节点(frametreeview):
    items = frametreeview.treeview.get_children("")
    for item in items: frametreeview.treeview.delete(item)
    frametreeview.filepath = None
    frametreeview.item = None
    frametreeview.bnode = None
    frametreeview.bnodedict = {}

def 清空文本(frametreeview):
    frametreeview.frametext.清空文本()



class __MenuFrame__(ttk.Widget):
    def __init__(self, master=None, **kwargs):
        ttk.Widget.__init__(self, master, "ttk::frame", kwargs)
        self.frametreeview = master
        self.添加部件()


    def 添加部件(self):
        self.menufile, self.buttonfile = self.__添加菜单__(label='文件', menu=self.__文件菜单区__())
        self.menuedit, self.buttonedit = self.__添加菜单__(label='编辑', menu=self.__编辑菜单区__())
        self.menutext, self.buttontext = self.__添加菜单__(label='日志', menu=self.__日志菜单区__())
        return self


    def __添加菜单__(self, label='', menu=None):
        menubutton = ttk.Menubutton(self, text=label)
        menubutton.pack(side="left")
        menubutton.config(menu=menu)
        return menu, menubutton


    def __文件菜单区__(self):
        menu=tk.Menu(None, tearoff=False) # 关闭菜单可拖动
        menu.add_command(label='打开文件', command=None) # menu.entryconfigure("打开", command=lambda :print('123'))
        menu.add_command(label="保存文件", command=None)
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

        # self.treeview.bind("<Button-1>", self.__左键单击函数__) 
        # self.treeview.bind("<Button-3>", self.__右键单击函数__) 
        self.treeview.bind('<Double-1>', self.__显示双击弹窗__)
        self.treeview.bind("<ButtonRelease-3>", self.__显示右键菜单__) # 右键单击离开
        # self.treeview.bind("<ButtonRelease-1>", self.__显示左键弹窗__) # 左键单击离开


    def __点击更新函数__(self, event):
        item = self.treeview.identify_row(event.y)
        if item == "": return None
        self.frametreeview.item = item
        self.frametreeview.bnode = self.frametreeview.bnodedict[item]
        return item

    def __显示右键菜单__(self, event):
        self.__点击更新函数__(event)
        # if self.__点击更新函数__(event) == None: return None
        self.rightmenu.post(event.x_root, event.y_root)

    def __显示双击弹窗__(self, event):
        self.__点击更新函数__(event)
        # if self.__点击更新函数__(event) == None: return None
        self.leftpopup.显示弹窗(event)


class __TextFrame__(ttk.Widget):
    def __init__(self, master=None, **kwargs):
        ttk.Widget.__init__(self, master, "ttk::frame", kwargs)
        self.frametreeview = master
        self.text = self.__添加__文本__区域__()

    def __添加__文本__区域__(self):
        text = tkinter.scrolledtext.ScrolledText(self)
        text.pack(side='left', fill='both', expand=1)
        return text

    def 插入文本(self, *args):
        for char in args:
            if type(char) != str:
                try:
                    char = str(char)
                except:
                    char = ""
            self.text.insert("end", f"{char}\n")
        self.text.insert("end", f"\n")    
        self.text.see("end")


    def 插入列表(self, args:list, 前缀="", 后缀="", 首行="", 末行=""):
        if 首行 != "": self.text.insert("end", f"{首行}\n")
        for element in args:
            chars = f"{前缀}{element}{后缀}\n"
            self.text.insert("end", chars)
        if 末行 != "": self.text.insert("end", f"{末行}\n")
        self.text.insert("end", "\n")    
        self.text.see("end")


    def 清空文本(self):
        # self.tk.call(self._w, 'delete', index1, index2)
        self.text.delete(0.0, "end") # 0.0 表示第0行第0个下标

class __EditPopup__(tk.Toplevel):
    def __init__(self, master=None, **kwargs):
        tk.Toplevel.__init__(self)
        self.frametreeview = master
        self.overrideredirect(True)  # 关闭外框 # 开启外框后，会自动隐藏到窗口后面，锁定窗口焦点无效，原因未明
        self.withdraw()
        self.__添加__部件__()


    def 显示弹窗(self, event):
        bnode = self.frametreeview.bnode
        classtype = str(type(bnode)) # bnode: bgmultipletree.类, bbyte: bgmultipletree.bbyte, bchar: bgmultipletree.bchar
        if "bgmultipletree.bchar" not in classtype: return None

        x_root, y_root = event.x_root, event.y_root
        width, height = 600, 100 # +23
        self.geometry(f"{width}x{height}+{x_root}+{y_root}")
        self.deiconify() # 显示窗口
        self.grab_set() # 锁定焦点在窗口 # Python tkinter Misc类+Wm类详解
        self.__更新__BCHAR__信息__()


    def 隐藏弹窗(self):
        self.grab_release() # 释放焦点
        self.withdraw() # 隐藏窗口



    def 保存修改(self):
        self.隐藏弹窗()
        try:
            bnode = self.frametreeview.bnode
            classtype = str(type(bnode)) # bnode: bgmultipletree.类, bbyte: bgmultipletree.bbyte, bchar: bgmultipletree.bchar
            if "bgmultipletree.bchar" not in classtype: return None
            bnode.string = self.entry_strdata.get()
            self.frametreeview.itemblock(self.frametreeview.item, self.frametreeview.bnode)
            name1, name2 = self.label_strdata["text"], bnode.string
            self.frametreeview.frametext.插入文本(f"修改文本::{name1} --> {name2}")
        except Exception as e:
            print(e)


    def __添加__部件__(self):
        frame = ttk.Frame(self)
        frame.pack()
        self.__添加__编辑__区域__(frame)
        frame = ttk.Frame(self)
        frame.pack()
        self.__添加__按钮__区域__(frame)


    def __添加__编辑__区域__(self, frame):
        row = 0
        widtha, widthb = 8, 60
        column = 0
        self.label_inttype = __标签标签部件__(frame, text="数值类型：", row=row, column=column, width=widtha)
        column += 2
        self.label_intdata = __标签标签部件__(frame, text="读取：", row=row, column=column, width=widthb)
        row += 1

        # column = 0
        # __标签标签部件__(frame, text="", row=row, column=column, width=widtha)
        # column += 2
        # __标签输入部件__(frame, text="编辑：", row=row, column=column, width=widthb)
        # row += 1

        column = 0
        self.label_strtype = __标签标签部件__(frame, text="字符类型：", row=row, column=column, width=widtha)
        column += 2
        self.label_strdata = __标签标签部件__(frame, text="读取：", row=row, column=column, width=widthb)
        row += 1

        column = 0
        # self.entry_strtype = __标签标签部件__(frame, text="", row=row, column=column, width=widtha)
        column += 2
        self.entry_strdata = __标签输入部件__(frame, text="编辑：", row=row, column=column, width=widthb)
        row += 1


    def __添加__按钮__区域__(self, frame):
        frame.grid_rowconfigure((0,), minsize=4, weight=0)
        frame.grid_columnconfigure((1, 3), minsize=18, weight=0) # 由于您使用的是grid，一个简单的解决方案是在按钮之间保留空列，然后给这些列一个minsize等于您想要的空间
        width = 25
        row, column = 1, 0
        __标签按钮部件__(frame, text="", row=row, column=column, width=width, command=None)
        column += 2
        __标签按钮部件__(frame, text="退出修改", row=row, column=column, width=width, command=self.隐藏弹窗)
        column += 2
        __标签按钮部件__(frame, text="保存修改", row=row, column=column, width=width, command=self.保存修改)
        

    def __更新__BCHAR__信息__(self):
        bnode = self.frametreeview.bnode
        self.label_inttype["text"] = bnode.__ibyte__.__type__
        self.label_intdata["text"] = bnode.__ibyte__.__data__
        self.label_strtype["text"] = bnode.__sbyte__.__type__
        self.label_strdata["text"] = bnode.__sbyte__.__data__
        self.entry_strdata.delete(0, "end")
        self.entry_strdata.insert(0, bnode.__sbyte__.__data__)



def __标签标签部件__(self, text="", row=0, column=0, width=15):
    标签 = ttk.Label(self, text=text)
    标签.grid(row=row, column=column, sticky="e")
    输出 = ttk.Label(self, width=width)
    输出.grid(row=row, column=column+1, sticky="we")
    return 输出

def __标签输入部件__(self, text="", row=0, column=0, width=15, value=""):
    标签 = ttk.Label(self, text=text)
    标签.grid(row=row, column=column, sticky="e")
    输入 = ttk.Entry(self, width=width)
    输入.insert(index=0, string=value)
    输入.grid(row=row, column=column+1, sticky="we")
    return 输入

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



import tkinter as tk
import tkinter.ttk as ttk
import tkinter.filedialog

import os
import 底层编辑

from . import Tree基础编辑


class __节点历史__:
    def __init__(self):
        self.临时节点 = None
        self.节点计数 = -1
        self.节点列表 = [None for i in range(10)]

    def 插入节点(self, 节点):
        self.节点计数 = (self.节点计数 + 1) % 10
        self.节点列表[self.节点计数] = 节点


    def 获取节点(self):
        if self.节点计数 < 0: return None
        return self.节点列表[self.节点计数]

    def 后退节点(self):
        self.节点计数 = (self.节点计数 - 1) % 10 # (-1) % 10 == 9
        return self.节点列表[self.节点计数]

    def 前进节点(self):
        self.节点计数 = (self.节点计数 + 1) % 10
        return self.节点列表[self.节点计数]


节点历史 = __节点历史__()


class FrameModel(Tree基础编辑.TreeFrame):
    def __init__(self, master=None, **kwargs):
        Tree基础编辑.TreeFrame.__init__(self, master, **kwargs)
        self.__更新__文件__函数__()
        self.__附加__编辑__菜单__()
        self.__附加__右键__菜单__()

    def __更新__文件__函数__(self):   
        menu = self.frametoolbar.menufile
        menu.entryconfig("打开文件", command=lambda:__打开文件__(self))
        menu.entryconfig("保存文件", command=lambda:__保存文件__(self))
        menu = self.rightmenu
        menu.entryconfig("打开文件", command=lambda:__打开文件__(self))
        menu.entryconfig("保存文件", command=lambda:__保存文件__(self))


    def __附加__编辑__菜单__(self):   
        menu = self.frametoolbar.menuedit
        menu.add_separator() # 添加分割线
        菜单=tk.Menu(menu, tearoff=False)
        menu.add_cascade(label="添加贴图栏", menu=菜单)
        菜单.add_command(label='第4~第6栏',command=self.__添加__第3至第6栏贴图__)
        菜单.add_command(label='第7~第9栏',command=self.__添加__第7至第9栏贴图__)
        menu.add_separator() # 添加分割线
        菜单=tk.Menu(menu, tearoff=False)
        menu.add_cascade(label="删除贴图栏", menu=菜单)
        菜单.add_command(label='第4~第6栏',command=self.__删除__第3至第6栏贴图__)
        菜单.add_command(label='第7~第9栏',command=self.__删除__第7至第9栏贴图__)
        


    def __添加__第3至第6栏贴图__(self):
        __添加__第3至第6栏贴图__(self)

    def __添加__第7至第9栏贴图__(self):
        __添加__第7至第9栏贴图__(self)

    def __删除__第3至第6栏贴图__(self):
        __删除__第3至第6栏贴图__(self)

    def __删除__第7至第9栏贴图__(self):
        __删除__第7至第9栏贴图__(self)


    def __附加__右键__菜单__(self):
        menu = self.rightmenu
        menu.add_separator() # 添加分割线
        menu.add_command(label='复制节点', command=self.__复制节点__)
        menu.add_command(label='粘贴节点', command=self.__粘贴节点__)
        menu.add_command(label='回退粘贴', command=self.__回退粘贴__)


    def __复制节点__(self):
        if "lrtm_" not in self.bnode.name:
            self.frametext.插入文本(f"复制节点::请选择lrtm节点复制")
            return None
        节点历史.插入节点(self.bnode)
        self.frametext.插入文本(f"复制节点::{self.bnode.name}", f"{self.bnode}")


    def __粘贴节点__(self):
        if self.item == "": return None
        if "lrtm_" not in self.bnode.name:
            self.frametext.插入文本(f"粘贴节点::请先选择lrtm节点")
            return None

        节点历史.临时节点 = self.bnode
        bnode = 节点历史.获取节点()
        classtype = str(type(bnode))
        if "bgmultipletree.类" in classtype:
            __替换节点__(bnode, self)
            self.frametext.插入文本(f"粘贴节点::{节点历史.临时节点.name} --> {bnode.name}", f"{节点历史.临时节点} --> {bnode}")
        else:
            self.frametext.插入文本(f"粘贴节点::类型{classtype}不支持")


    def __回退粘贴__(self):
        if self.item == "": return None
        tempbnode = self.bnode
        bnode = 节点历史.临时节点
        classtype = str(type(bnode))
        if "bgmultipletree.类" in classtype:
            __替换节点__(bnode, self)
            self.frametext.插入文本(f"回退粘贴::{tempbnode.name} --> {bnode.name}", f"{tempbnode} --> {bnode}")
        else:
            self.frametext.插入文本(f"回退粘贴::类型{classtype}不支持")



def __打开文件__(frametreeview):
    Tree基础编辑.清空节点(frametreeview)
    filepath = tkinter.filedialog.askopenfilename(filetypes=[("古剑model", ".model"), ("All files", "*")])
    if filepath == "" or os.path.isfile(filepath) == False: return None
    if filepath.endswith(".model") == False: return None
    filebnode = 底层编辑.model.类().read(filepath)
    Tree基础编辑.递归显示(frametreeview, fitem="", fbnode=filebnode)
    frametreeview.filebnode = filebnode
    frametreeview.frametext.插入文本(f"打开文件::{filepath}")


def __保存文件__(frametreeview): # def 保存浏览文件单选():
    filepath = frametreeview.filebnode.filepath
    dirname = os.path.dirname(filepath)
    basename = os.path.basename(filepath)
    filepath = tkinter.filedialog.asksaveasfilename(initialdir=dirname, 
                                                    initialfile=basename[0:-6]+"_new.model",
                                                    filetypes=[("古剑model", ".model"), ("All files", "*")]
                                                    )
    if filepath == "": return None
    frametreeview.filebnode.write(filepath)
    frametreeview.frametext.插入文本(f"保存文件::{filepath}")


def __添加__贴图栏__(keyname, bnode, i):
    if bnode.uint32 == 3: return None
    lrtmbnode = bnode.parent
    lrtmbnode.addstart(keyname)
    lrtmbnode.addstring("", name=f"{i}_贴图名称_0", inttype="uint32", strtype="utf8")
    lrtmbnode.addstring("", name=f"{i}_贴图名称_1", inttype="uint32", strtype="utf8")
    lrtmbnode.addstring("", name=f"{i}_贴图名称_2", inttype="uint32", strtype="utf8")
    lrtmbnode.addfinal()
    bnode.uint32 = 3


def __删除__贴图栏__(keyname, bnode, i):
    if bnode.uint32 == 0: return None
    keynames = list(bnode.parent.keys())
    index = keynames.index(keyname)
    for name in keynames[index+1: index+4]: 
        if f"{i}_贴图名称_" in name: bnode.parent.pop(name)
    bnode.uint32 = 0



def __添加__第3至第6栏贴图__(frametreeview):
    if frametreeview.item == "": return None
    srtmbg = frametreeview.filebnode.name_object_dict["SRTM"]
    for name, bnode in srtmbg.items(): 
        if "lrtm_" not in name: continue
        for keyname, cnode in list(bnode.items()): # RuntimeError: dictionary changed size during iteration
            if keyname == "第2栏贴图数量": __添加__贴图栏__(keyname, cnode, 2)

    Tree基础编辑.清空节点(frametreeview)
    Tree基础编辑.递归显示(frametreeview, fitem="", fbnode=frametreeview.filebnode)

def __添加__第7至第9栏贴图__(frametreeview):
    if frametreeview.item == "": return None
    srtmbg = frametreeview.filebnode.name_object_dict["SRTM"]
    for name, bnode in srtmbg.items():
        if "lrtm_" not in name: continue
        for keyname, cnode in list(bnode.items()):
            if keyname == "第3栏贴图数量": __添加__贴图栏__(keyname, cnode, 3)

    Tree基础编辑.清空节点(frametreeview)
    Tree基础编辑.递归显示(frametreeview, fitem="", fbnode=frametreeview.filebnode)

def __删除__第3至第6栏贴图__(frametreeview):
    if frametreeview.item == "": return None
    srtmbg = frametreeview.filebnode.name_object_dict["SRTM"]
    for name, bnode in srtmbg.items():
        if "lrtm_" not in name: continue
        for keyname, cnode in list(bnode.items()):
            if keyname == "第2栏贴图数量": __删除__贴图栏__(keyname, cnode, 2)

    Tree基础编辑.清空节点(frametreeview)
    Tree基础编辑.递归显示(frametreeview, fitem="", fbnode=frametreeview.filebnode)

def __删除__第7至第9栏贴图__(frametreeview):
    if frametreeview.item == "": return None
    srtmbg = frametreeview.filebnode.name_object_dict["SRTM"]
    for name, bnode in list(srtmbg.items()):
        if "lrtm_" not in name: continue
        for keyname, cnode in list(bnode.items()):
            if keyname == "第3栏贴图数量": __删除__贴图栏__(keyname, cnode, 3)

    Tree基础编辑.清空节点(frametreeview)
    Tree基础编辑.递归显示(frametreeview, fitem="", fbnode=frametreeview.filebnode)



def __替换节点__(anode, frametreeview): # a 替换 b
    item = frametreeview.item
    bnode = frametreeview.bnode

    keyname = bnode.parent.getname_fromobject(bnode)
    bnode.parent[keyname] = anode

    frametreeview.itemblock(item, anode)
    frametreeview.bnode = anode

    childitems = frametreeview.treeview.get_children(item)
    for citem in childitems: frametreeview.treeview.delete(citem)

    Tree基础编辑.递归显示(frametreeview, fitem=item, fbnode=anode)

























# def __前进节点__(self):
#     if self.item == "": return None
#     tempbnode = self.bnode
#     bnode = 节点历史.前进节点()
#     classtype = str(type(bnode))
#     if "bgmultipletree.类" in classtype:
#         __替换节点__(bnode, self)
#         self.frametext.插入文本(f"前进节点::{tempbnode} --> {bnode}")
#     else:
#         self.frametext.插入文本(f"前进节点::类型{classtype}不支持")

# def __后退节点__(self):
#     if self.item == "": return None
#     tempbnode = self.bnode
#     bnode = 节点历史.后退节点()
#     classtype = str(type(bnode))
#     if "bgmultipletree.类" in classtype:
#         __替换节点__(bnode, self)
#         self.frametext.插入文本(f"后退节点::{tempbnode} --> {bnode}")
#     else:
#         self.frametext.插入文本(f"后退节点::类型{classtype}不支持")


