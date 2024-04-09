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




