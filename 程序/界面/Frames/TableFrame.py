
import tkinter as tk
import tkinter.ttk as ttk
import os
import sys
import time


from .Table选项Frame import TableItemFrame
from .Table状态Toplevel import TableStatusToplevel
from .. import 缓存
from .  import Table索引
from .  import Table显示处理
from .  import Table右键Memu


class TableFrame(ttk.Widget):
    def __init__(self, master=None, **kw):
        ttk.Widget.__init__(self, master, "ttk::frame", kw)
        self.建立列()
        self.建立表()
        self.建立菜单()
        self.添加菜单()
        self.建立行()
        self.设置列()
        # self.隐藏左侧栏()

        # self.绑定右键单击离开函数() # 显示菜单
        Table显示处理.itemframe = self.itemframe
        Table显示处理.frametable = self
        self.是否弹窗 = True
        self.是否区域 = False


    def 建立列(self, numcolumn=16):
        charlist = [f"#{i+1}" for i in range(numcolumn)]
        self.columns = charlist[:]
        self.columnstitle = charlist[:]


    def 建立表(self):
        xbar = ttk.Scrollbar(self, orient='horizontal')   # 水平
        xbar.pack(side="bottom", fill="x")                # must be top, bottom, left, or right

        ybar = ttk.Scrollbar(self, orient='vertical')     # 垂直
        ybar.pack(side="right", fill="y")

        self.table = ttk.Treeview(self, columns=self.columns)
        self.itemframe = TableItemFrame(self)
        self.itemframe.pack(side='top', fill='x')
        self.table.pack(side='left', fill='both', expand=1)
        self.statustoplevel = TableStatusToplevel(self, self.itemframe)
        xbar.config(command=self.table.xview)
        ybar.config(command=self.table_yview)
        self.ybar = ybar
        self.ybar.set(str(0.0), str(0.1)) # 滑块位置 与 滑块长短
        self.table.config(xscrollcommand=xbar.set, yscrollcommand=self.ybar_set)


    def ybar_set(self, *args, **kwargs):
        # print("表格输出参数：", "args", args, "kwargs", kwargs)
        Table索引.ybar_set(self, self.itemframe, *args, **kwargs)


    def table_yview(self, *args, **kwargs):
        # print("滚条输出参数：", "args", args, "kwargs", kwargs)
        Table索引.table_yview(self, self.itemframe, *args, **kwargs)
        Table显示处理.回车显示()


    def 建立菜单(self):
        self.右键菜单 = tk.Menu(self.table, tearoff=False)

        self.table.bind("<Motion>", self.隐藏状态弹窗)            
        self.table.bind("<ButtonRelease-1>", self.显示状态弹窗) # 左键单击离开
        self.table.bind("<ButtonRelease-3>", self.显示右键菜单) # 右键单击离开


    def 显示右键菜单(self, event):
        self.statustoplevel.withdraw()
        self.右键菜单.post(event.x_root, event.y_root)


    def 显示状态弹窗(self, event):
        if 缓存.弹窗开闭状态 is False: return ""
        x_root, y_root = event.x_root, event.y_root
        width, height = 250, 92
        self.statustoplevel.geometry(f"{width}x{height}+{x_root}+{y_root}")
        self.statustoplevel.deiconify()
        self.statustoplevel.更新状态显示()


    def 隐藏状态弹窗(self, event):
        self.statustoplevel.隐藏弹窗(event)


    def 添加菜单(self):
        回车菜单 = Table右键Memu.TableRightMemu(self, self.右键菜单)
        # self.右键菜单.add_separator() # 添加分割线
        # self.右键菜单.add_command(label="显示所有列", command=self.显示所有列)
        # self.右键菜单.add_command(label="隐藏参数列", command=self.隐藏参数列)
        self.右键菜单.add_separator() # 添加分割线
        self.右键菜单.add_command(label="跳转首行", command=self.跳转首行)
        self.右键菜单.add_command(label="跳转末行", command=self.跳转末行)        
        self.右键菜单.add_separator() # 添加分割线
        self.右键菜单.add_command(label="显示左侧栏", command=self.显示左侧栏)
        self.右键菜单.add_command(label="隐藏左侧栏", command=self.隐藏左侧栏)
        self.右键菜单.add_separator() # 添加分割线
        self.右键菜单.add_command(label="复制行内容", command=self.复制行内容)
        self.右键菜单.add_command(label="复制行文本", command=self.复制行文本)
        

    def 建立行(self, numrow=128):
        for i in range(numrow): self.table.insert("", index="end", text=str(i)) 
    
    def 设置列(self):
        self.table.column("#0", width=120, stretch=0)
        for column, columnstitle in zip(self.columns, self.columnstitle):
            self.table.column(column, width=40, stretch=False, anchor='center')
            self.table.heading(column, text=columnstitle)


    def 跳转首行(self):
        item = self.table.get_children("")[0]
        self.table.see(item)

    def 跳转末行(self):
        item = self.table.get_children("")[-1]
        self.table.see(item)

    def 隐藏左侧栏(self):
        self.table["show"] = "headings"

    def 显示左侧栏(self):
        self.table["show"] = "tree headings"


    def 复制行内容(self):
        values = self.table.item(self.lastitem)["values"]
        缓存.tkinter.clipboard_clear()
        缓存.tkinter.clipboard_append(str(values))

    def 复制行文本(self):
        values = self.table.item(self.lastitem)["values"]
        chars = ""
        for char in values:
            if isinstance(char, int): char = str(char)
            chars += char
        缓存.tkinter.clipboard_clear()
        缓存.tkinter.clipboard_append(chars)


    # def 隐藏参数列(self):
    #     self.table["displaycolumns"] = self.table["columns"]

    # def 显示所有列(self):
    #     self.table["displaycolumns"] = self.table["columns"]