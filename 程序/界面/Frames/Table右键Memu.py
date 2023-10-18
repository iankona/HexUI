

import os
import sys
import tkinter as tk
import tkinter.ttk as ttk
from .  import Table显示处理


class TableRightMemu():
    def __init__(self, frametable, 菜单):
        self.index = 0
        self.column = 0
        self.frametable = frametable

        菜单.add_separator() # 添加分割线
        菜单.add_command(label="回车分行", command=self.回车分行)
        菜单.add_command(label="回车换行", command=self.回车换行)
        菜单.add_command(label="回车末行", command=self.回车末行)
        # 菜单.add(tk.Label, text="122")
        # bad menu entry type "<class 'tkinter.Label'>": must be cascade, checkbutton, command, radiobutton, or separator

        self.frametable.table.bind('<Button-1>', self.左键单击函数)
        self.frametable.table.bind('<Button-3>', self.右键单击函数)


    def 左键单击函数(self, event):
        item = self.frametable.table.identify_row(event.y)
        index = self.frametable.table.index(item)
        column = self.frametable.table.identify_column(event.x)
        Table显示处理.Table索引.刷新表列索引(self.frametable, index, column)


    def 右键单击函数(self, event):
        item = self.frametable.table.identify_row(event.y)
        index = self.frametable.table.index(item)
        column = self.frametable.table.identify_column(event.x)
        # print("右键单击函数", item, column)
        # Table显示处理.列数列表操作(index, column)
        self.index = index
        self.column = column


    def 回车分行(self):
        Table显示处理.回车分行_列数列表操作(self.index, self.column)
        Table显示处理.回车显示()

    def 回车换行(self):
        Table显示处理.回车换行_列数列表操作(self.index, self.column)
        Table显示处理.回车显示()

    def 回车末行(self):
        Table显示处理.回车末行_列数列表操作(self.index, self.column)
        Table显示处理.回车显示()


