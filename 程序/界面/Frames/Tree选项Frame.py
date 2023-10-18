import tkinter as tk
import tkinter.ttk as ttk
import tkinter.filedialog
import os
import sys
import mmap



class TreeItemFrame(ttk.Widget):
    def __init__(self, master=None, **kw):
        ttk.Widget.__init__(self, master, "ttk::frame", kw)
        self.frametree = master

        row, column = 0, 0
        width = 42
        self.索引列表 = self.标签输入部件(text="索引列表：", row=row, column=column, width=width, value="48/56/128")
        row += 1
        self.大小列表 = self.标签输入部件(text="大小列表：", row=row, column=column, width=width, value="24/48/96")
        row += 1
        self.状态显示 = self.标签输入部件(text="状态显示：", row=row, column=column, width=width, value="")
        row += 1
        column = column + 2

        row, column = 0, column + 2
        width = 10
        self.索引分割 = ttk.Button(self, text="索引分割", width=width, command=self.按索引新建子节点)
        self.索引分割.grid(row=row, column=column, sticky="we")
        row += 1
        self.大小分割 = ttk.Button(self, text="索引分割", width=width, command=self.按大小新建子节点)
        self.大小分割.grid(row=row, column=column, sticky="we")
        row += 1
        self.层级分割 = ttk.Button(self, text="层级分割", width=width, command=self.按索引新建层节点)
        self.层级分割.grid(row=row, column=column, sticky="we")
        row += 1


    def 标签输入部件(self, text="", row=0, column=0, width=15, value=""):
        标签 = ttk.Label(self, text=text)
        标签.grid(row=row, column=column, sticky="e")
        输入 = ttk.Entry(self, width=width)
        输入.insert(index=0, string=value)
        输入.grid(row=row, column=column+1, sticky="we")
        return 输入


    def 按索引新建子节点(self):
        numchars = self.索引列表.get().split("/")
        numbers = [0] + [int(numchar) for numchar in numchars if numchar != ""]
        sizes = [ numbers[i+1]-numbers[i] for i in range(len(numbers)-1)]
        sizes = [ size for size in sizes if size > 0 ]
        for item in self.frametree.tree.selection(): self.新建子节点(item, sizes)


    def 按大小新建子节点(self):
        numchars = self.大小列表.get().split("/")
        sizes = [int(numchar) for numchar in numchars if numchar != ""]
        sizes = [ size for size in sizes if size > 0 ]
        for item in self.frametree.tree.selection(): self.新建子节点(item, sizes)


    def 新建子节点(self, item, sizes):
        bp = self.frametree.bpdict[item].copy()
        bps = [bp.readslice(size) for size in sizes]
        sizes.append(bp.remainsize())
        bps.append(bp.readremainslice())
        for size, bp in zip(sizes, bps): citem = self.frametree.insertblock(item, index="end", text=f"块_{size}", bp=bp)  
        char = self.frametree.tree.item(citem)["text"] 
        self.frametree.tree.item(citem, text="余"+char)


    def 新建层节点(self, item, sizes):
        bp = self.frametree.bpdict[item].copy()
        bps = [bp.readslice(size) for size in sizes]
        sizes.append(bp.remainsize())
        bps.append(bp.readremainslice())

        fitem = self.frametree.tree.parent(item)
        if fitem == "":
            for size, bp in zip(sizes, bps): citem = self.frametree.insertblock(item, index="end", text=f"块_{size}", bp=bp)
        else:
            citem = self.frametree.itemblock(item, text=f"块_{sizes[0]}", bp=bps[0])
            for size, bp in zip(sizes[1:], bps[1:]): citem = self.frametree.insertblock(fitem, index="end", text=f"块_{size}", bp=bp) 

        char = self.frametree.tree.item(citem)["text"] 
        self.frametree.tree.item(citem, text="余"+char)
        
    def 按索引新建层节点(self):
        numchars = self.索引列表.get().split("/")
        numbers = [0] + [int(numchar) for numchar in numchars if numchar != ""]
        sizes = [ numbers[i+1]-numbers[i] for i in range(len(numbers)-1)]
        sizes = [ size for size in sizes if size > 0 ]
        for item in self.frametree.tree.selection(): self.新建层节点(item, sizes)


    def 按大小新建层节点(self):
        numchars = self.大小列表.get().split("/")
        sizes = [int(numchar) for numchar in numchars if numchar != ""]
        sizes = [ size for size in sizes if size > 0 ]
        for item in self.frametree.tree.selection(): self.新建层节点(item, sizes)
  







    # def 绑定函数(self):
    #     self.索引列表输入.insert(index=0, string="48/56/128")
    #     self.分割按钮['command'] = self.按输入新建子节点