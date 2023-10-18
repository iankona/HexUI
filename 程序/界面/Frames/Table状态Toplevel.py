
import tkinter as tk
import tkinter.ttk as ttk
import time
import threading

from .. import 缓存
from .  import Table索引
from .  import Table显示处理


class TableStatusToplevel(tk.Toplevel):
    def __init__(self, frametable, itemframe, master=None, cnf={}, **kw):
        tk.Toplevel.__init__(self)
        self.frametable = frametable
        self.itemframe = itemframe
        self.overrideredirect(True)
        self.withdraw()

        self.是否隐藏 = False
        self.添加标签输入()



    def 是否区域内(self, event):
        bbox = self.geometry()
        width_height, x_root, y_root = bbox.split("+")
        width, height = width_height.split("x")
        width, height, x_root, y_root = int(width), int(height), int(x_root), int(y_root)

        include = 16
        boolwidth, boolheight = False, False
        if x_root - include< event.x_root < x_root + width + include: boolwidth = True
        if y_root - include< event.y_root < y_root + height + include: boolheight = True
        return True if boolwidth and boolheight else False


    def 隐藏弹窗(self, event):
        None if self.是否区域内(event) else self.withdraw()


    def 添加标签输入(self):
        row, column = 0, 0
        width = 45
        self.entryuint8 = self.标签输入部件(text="uint8：", row=row, column=column, width=width)
        row += 1
        self.entryuint16 = self.标签输入部件(text="uint16：", row=row, column=column, width=width)
        row += 1
        self.entryuint32 = self.标签输入部件(text="uint32：", row=row, column=column, width=width)
        row += 1
        self.entryfloat32 = self.标签输入部件(text="float32：", row=row, column=column, width=width)
        row += 1


    def 更新标签输入(self, bpuint8, bpuint16, bpuint32, bpfloat32):
        self.entryuint8.delete(0, "end")
        self.entryuint8.insert(0, f"{bpuint8}") 
        self.entryuint16.delete(0, "end")
        self.entryuint16.insert(0, f"{bpuint16}") 
        self.entryuint32.delete(0, "end")
        self.entryuint32.insert(0, f"{bpuint32}") 
        self.entryfloat32.delete(0, "end")
        self.entryfloat32.insert(0, f"{bpfloat32}") 


    def 添加标签标签(self):
        row, column = 0, 0
        width = 15
        self.labeluint8 = self.标签标签部件(text="uint8：", row=row, column=column, width=width)
        row += 1
        self.labeluint16 = self.标签标签部件(text="uint16：", row=row, column=column, width=width)
        row += 1
        self.labeluint32 = self.标签标签部件(text="uint32：", row=row, column=column, width=width)
        row += 1
        self.labelfloat32 = self.标签标签部件(text="float32：", row=row, column=column, width=width)
        row += 1


    def 更新标签标签(self, bpuint8, bpuint16, bpuint32, bpfloat32):
        self.labeluint8["text"] = f"{bpuint8}"
        self.labeluint16["text"] = f"{bpuint16}"
        self.labeluint32["text"] = f"{bpuint32}"
        self.labelfloat32["text"] = f"{bpfloat32}"


    def 更新状态显示(self):
        折算数量 = len(Table显示处理.数据列表) * len(Table显示处理.格式列表)
        前置偏移值 = self.简化前置偏移(
            Table显示处理.单区大小, 
            Table显示处理.分段列表, 
            Table显示处理.单元格字节数, 
            Table索引.列表索引, 
            Table索引.列数索引 + Table索引.rowindex//折算数量,
            )
        
        # 固定偏移值 = int(self.itemframe.偏移大小.get())
        列前偏移值 = Table显示处理.单元格字节数 * Table索引.三级索引
        字节偏移值 = 前置偏移值 + 列前偏移值
        bp = self.取得bp数据对象(Table索引.rowindex, Table显示处理.数据列表, Table显示处理.格式列表)
        bp.seek(字节偏移值)
        self.更新标签输入(
            bp.readuint8seek0(), 
            bp.readuint16seek0(), 
            bp.readuint32seek0(), 
            bp.readfloat32seek0(),
            )



    def 简化前置偏移(self, 单区大小, 分段列表, 单元格字节数, 列表索引, 列数索引):
        分段偏移值 = 单区大小 * 列表索引
        列段偏移值 = 0
        for 列数 in 分段列表[列表索引][0:列数索引]:
            列段偏移值 += 列数*单元格字节数
        if 列段偏移值 > 单区大小: 列段偏移值 = 单区大小
        return 分段偏移值 + 列段偏移值


    def 取得bp数据对象(self, rowindex, 数据列表, 格式列表):
        数据数量, 格式数量 = len(数据列表), len(格式列表)
        折算数量 = 数据数量 * 格式数量
        等效余数 = rowindex % 折算数量
        数据索引 = 等效余数 % 数据数量
        return 数据列表[数据索引].copy()



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
            
            
            
        # self.bind("<Enter>", self.进入弹窗函数)
        # # self.bind("<Motion>", self.鼠标离开时最小化)
        # self.bind("<Leave>", self.最小化)


    # def 进入弹窗函数(self, event):
    #     缓存.弹窗停留时间戳 = time.time()
    #     self.更新显示()


    # def 最小化(self, event):
    #     # print(time.time(), 缓存.弹窗停留时间戳)
    #     if time.time() - 缓存.弹窗停留时间戳 > 1.5: 
    #         self.withdraw()
    #     else:
    #         self.after(1500, self.withdraw)
    #         # threading.Thread(target=self.最小化_线程_函数体).start()

    # def 鼠标离开时最小化(self, event):
    #     # # 获取Toplevel窗口的位置
    #     # x = toplevel.winfo_x()
    #     # y = toplevel.winfo_y()
    #     bbox = self.geometry()
    #     width_height, x_root, y_root = bbox.split("+")
    #     width, height = width_height.split("x")
    #     width, height, x_root, y_root = int(width), int(height), int(x_root), int(y_root)

    #     include = 12
    #     boolw, boolh = True, True
    #     if x_root-include < event.x_root < x_root + width + include: boolw = False
    #     if y_root-include < event.y_root < y_root + height + include: boolh = False        
    #     if boolw and boolh: self.withdraw()

    #     self.bind("<Leave>", lambda event: threading.Thread(target=self.最小化_线程_函数体).start())



    # def 最小化_线程_函数体(self, event):
    #     time.sleep(10)
    #     self.布尔重新显示 = False
    #     self.withdraw()

    # def 弹窗延时隐藏(self, event):
    #     threading.Thread(target=self.最小化_线程_函数体, args=[event,]).start()


    # def 最小化_线程_函数体(self, event):
    #     time.sleep(1)
    #     None if self.是否区域内(event) else self.withdraw()