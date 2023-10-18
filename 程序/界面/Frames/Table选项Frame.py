import tkinter as tk
import tkinter.ttk as ttk

from .. import 缓存
from .  import Table显示处理


class TableItemFrame(ttk.Widget):
    def __init__(self, master=None, **kw):
        ttk.Widget.__init__(self, master, "ttk::frame", kw)
        self.frametable = master

        row, column = 0, 0
        width = 15

        self.解析格式 = self.标签选取部件(text="解析格式：", row=row, column=column, width=width, values=缓存.解析格式列表, index=8)
        row += 1
        self.单区大小 = self.标签输入部件(text="单区大小：", row=row, column=column, width=width, value=10240)
        row += 1
        self.有效行数 = self.标签输入部件(text="有效行数：", row=row, column=column, width=width, value=128)
        row += 1
        column = column + 2

        row, column = 0, column + 2
        width = 15
        self.偏移大小 = self.标签选取部件(text="偏移大小：", row=row, column=column, width=width)
        row += 1
        self.第1行列数 = self.标签输入部件(text="第1行列数：", row=row, column=column, width=width, value=16)
        row += 1
        self.第n行列数 = self.标签输入部件(text="第n行列数：", row=row, column=column, width=width, value=16)
        row += 1


        row, column = 0, column + 2
        width = 15
        self.弹窗开闭 = ttk.Button(self, text="开启弹窗", width=width, command=self.弹窗状态切换)
        self.弹窗开闭.grid(row=row, column=column, sticky="we")
        row += 1
        self.占用分割 = ttk.Button(self, text="", width=width, command=None)
        self.占用分割.grid(row=row, column=column, sticky="we")
        row += 1
        self.刷新按钮 = ttk.Button(self, text="刷新显示", command=Table显示处理.刷新显示)
        self.刷新按钮.grid(row=row, column=column, sticky="we")

        # row, column = 0, column + 1
        # width = 15
        # self.开启弹窗 = ttk.Checkbutton(self, text="开启弹窗")
        # self.开启弹窗.grid(row=row, column=column, sticky="we")
        # row += 1

        row, column = 0, column + 1
        width = 15
        self.表格行数 = self.标签标签部件(text="表格行数：", row=row, column=column, width=width)
        row += 1
        self.显示行数 = self.标签标签部件(text="显示行数：", row=row, column=column, width=width)
        row += 1
        self.显示限值 = self.标签标签部件(text="显示限值：", row=row, column=column, width=width)
        row += 1
        # self.显示限行 = self.标签标签部件(text="显示限行：", row=row, column=column, width=width)
        # row += 1

        row, column = 0, column + 2
        width = 15
        self.等效行数 = self.标签标签部件(text="等效行数：", row=row, column=column, width=width)
        row += 1
        self.等效行数 = self.标签标签部件(text="等效行数：", row=row, column=column, width=width)
        row += 1
        self.等效限值 = self.标签标签部件(text="等效限值：", row=row, column=column, width=width)
        row += 1
        # self.等效限行 = self.标签标签部件(text="等效限行：", row=row, column=column, width=width)
        # row += 1


    def 标签输入部件(self, text="", row=0, column=0, width=15, value=""):
        标签 = ttk.Label(self, text=text)
        标签.grid(row=row, column=column, sticky="e")
        输入 = ttk.Entry(self, width=width)
        输入.insert(index=0, string=value)
        输入.grid(row=row, column=column+1, sticky="we")
        return 输入


    def 标签标签部件(self, text="", row=0, column=0, width=15):
        标签 = ttk.Label(self, text=text)
        标签.grid(row=row, column=column, sticky="e")
        输出 = ttk.Label(self, width=width)
        输出.grid(row=row, column=column+1, sticky="we")
        return 输出

    def 标签选取部件(self, text="", row=0, column=0, width=15, values=[0, 1, 2, 3, 4, 5, 6, 7], index=0):
        标签 = ttk.Label(self, text=text)
        标签.grid(row=row, column=column, sticky="e")
        选取 = ttk.Combobox(self, width=width, values=values)
        选取.current(index) # 选择第1个
        选取.grid(row=row, column=column+1, sticky="we")
        return 选取




    def 弹窗状态切换(self):
        缓存.弹窗开闭状态 = not 缓存.弹窗开闭状态
        if 缓存.弹窗开闭状态:
            self.弹窗开闭["text"] = "关闭弹窗"
        else:
            self.弹窗开闭["text"] = "开启弹窗"