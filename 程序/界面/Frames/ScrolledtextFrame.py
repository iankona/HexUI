
import tkinter as tk
import tkinter.ttk as ttk
from tkinter.scrolledtext import ScrolledText




class ScrolledtextItemFrame(ttk.Widget):
    def __init__(self, master=None, **kw):
        ttk.Widget.__init__(self, master, "ttk::frame", kw)
        self.frametable = master

        row, column = 0, 0
        width = 15
        self.数据列表 = self.标签输入部件(text="单区大小：", row=row, column=column, width=width, value=10240)
        row += 1
        self.解析格式 = self.标签输入部件(text="单区大小：", row=row, column=column, width=width, value=10240)
        row += 1
        self.单区大小 = self.标签输入部件(text="单区大小：", row=row, column=column, width=width, value=10240)
        row += 1
        self.解析格式 = self.标签输入部件(text="单区大小：", row=row, column=column, width=width, value=10240)
        row += 1



        column = column + 2




    def 标签输入部件(self, text="", row=0, column=0, width=15, value=""):
        标签 = ttk.Label(self, text=text)
        标签.grid(row=row, column=column, sticky="e")
        输入 = ttk.Entry(self, width=width)
        输入.insert(index=0, string=value)
        输入.grid(row=row, column=column+1, sticky="we")
        return 输入

class ScrolledtextFrame(ttk.Widget):
    def __init__(self, master=None, **kw):
        ttk.Widget.__init__(self, master, "ttk::frame", kw)
        self.建立参数()
        self.建立文本框()


    def 建立参数(self):
        self.itemframe = ScrolledtextItemFrame(self)
        self.itemframe.pack(side='top', fill='x')


    def 建立文本框(self):
        self.text = ScrolledText(self)
        self.text.pack(side='left', fill='both', expand=1)