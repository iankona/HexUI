import tkinter as tk

from . import bsfunction as bs
import 程序.文件.bpformat.function_bpnumpy as by


class 类:
    def __init__(self, frametree, 菜单):
        self.frametree = frametree
        子菜单 = tk.Menu(菜单, tearoff=False)
        菜单.add_cascade(label = "古剑23/Navmesh", menu=子菜单)
        子菜单.add_command(label="文件分块", command=self.文件分块)

        # 子菜单.add_command(label="dataposition", command=self.dataposition)
        # 子菜单.add_separator() # 添加分割线


    def context(self, item=""):
        if item == "": item = self.frametree.tree.selection()[0]
        bs.bscontext.frametree = self.frametree
        bs.bscontext.item = item
        by.context_bpnumpy.data = self.frametree.bpdict[item].copy() # bp = self.frametree.bpdict[item]
        return item


    def 文件分块(self):
        for item in self.frametree.tree.selection():
            self.context(item)
            bs.insertblock(text="TESM", bp=by.readslice(40).bp)
            while True:
                if by.remainsize() < 12: break
                uint8 = by.readuint8(8), 
                size = by.readuint32()
                bx = by.readslice(size).bp
                bs.insertblock(text=f"{uint8}_{size}_VAND", bp=bx)
            bs.insertblock(text="余下", bp=by.readremainslice().bp)


    def TESM(self):
        pass


    def VAND(self):
        for item in self.frametree.tree.selection():
            self.context(item)
            bs.insertblock(text="head", bp=by.readslice(40).bp)