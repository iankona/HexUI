import tkinter as tk

from . import bsfunction as bs
import 程序.文件.bpformat.function_bpnumpy as by


class 类:
    def __init__(self, frametree, 菜单):
        self.frametree = frametree
        子菜单 = tk.Menu(菜单, tearoff=False)
        菜单.add_cascade(label = "古剑23/Anarchy", menu=子菜单)
        子菜单.add_command(label="文件分块", command=self.文件分块)
        子菜单.add_command(label="HSMV", command=self.HSMV)
        子菜单.add_command(label="MOEG", command=self.MOEG)
        子菜单.add_command(label="MESH", command=self.meshfile)
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
            bs.insertblock(text="head", bp=by.readslice(8).bp)
            while True:
                if by.remainsize() < 20: break
                leftflag, leftname, = by.readuint32(), by.readchar(4)
                size = by.readuint32()
                bx = by.readslice(size).bp
                rightflag, rightnname = by.readuint32(), by.readchar(4)
                bs.insertblock(text=str([leftflag, leftname, size, rightflag, rightnname]), bp=bx)
            bs.insertblock(text="余下", bp=by.readremainslice().bp)


    def HSMV(self):
        for item in self.frametree.tree.selection():
            bp = self.frametree.bpdict[item].copy()
            fitem = self.frametree.tree.parent(item)
            chars = self.frametree.tree.item(fitem)["text"] + "_HSMV_meshfile"
            item = self.frametree.insertblock("", index="end", text=chars, bp=bp)
            self.context(item)
            with bs.insertblock(text="head", bp=by.readsliceseek0(8).bp), by.readslice(8):
                bs.insertvalue(text="magic, version", values=[by.readchar(4), by.readuint8(4)])

            bs.insertvalue(text="meshfile", values=[])
            self.MESH()

            # size = by.size() - 8 - 28
            # with bs.insertblock(text="MESH", bp=by.readsliceseek0(size).bp), by.readslice(size):
            #     self.MESH()
            # bs.insertblock(text="tail_28", bp=by.readremainslice().bp)


    def MOEG(self):
        for item in self.frametree.tree.selection():
            bp = self.frametree.bpdict[item].copy()
            fitem = self.frametree.tree.parent(item)
            chars = self.frametree.tree.item(fitem)["text"] + "_MOGE_meshfile"
            item = self.frametree.insertblock("", index="end", text=chars, bp=bp)
            self.context(item)
            bs.insertblock(text="head", bp=by.readslice(4).bp)
            bs.insertvalue(text="meshfile", values=[])
            # bs.insertblock(text="mesh", bp=by.readremainslice().bp)
            self.MESH()
            # with bs.insertblock(text="meshfile", bp=by.readremainsliceseek0().bp), by.readremainslice():
            #     self.MESH()

    def meshfile(self):
        for item in self.frametree.tree.selection():  
            self.context(item)
            self.MESH()


    def MESH(self):
        with bs.insertblock(text="head_8", bp=by.readsliceseek0(8).bp), by.readslice(8):
            magicchar = by.readcharseek0(4)
            magic, version = by.readuint8(4), by.readuint32()
            bs.insertvalue(text="magic, version", values=[magicchar, magic, version])
        with bs.insertblock(text="decs_60", bp=by.readsliceseek0(60).bp), by.readslice(60):
            l4uint8, bx, r4uint8 = by.readuint8(4), by.readslice(by.readuint32()).bp, by.readuint8(4)
            bs.insertblock(text=str([l4uint8, "bp", r4uint8]), bp=bx)
            顶点大小 = bx.readuint16()

        match version:
            case 1: numvert, numloop = self.MESH_info_version_1() # 30
            case 3: numvert, numloop = self.MESH_info_version_3() # 33
            case 5: numvert, numloop = self.MESH_info_version_5() # 36
            case _: print(f"Anarchymeshfile::未知mesh_info_type_{version}")

        uint8列表 = by.readuint8seek0(2)
        match uint8列表:
            case [0, 0]: bs.insertblock(text="block", bp=by.readslice(2).bp)
            case [255, 255]: bs.insertblock(text="block", bp=by.readslice(66).bp)

       
        sizevert = numvert * 顶点大小
        bs.insertblock(text="vert", bp=by.readslice(sizevert).bp)
        sizeloop = numloop * 2
        if numvert > 0xFFFF: sizeloop = numloop * 4
        bs.insertblock(text="loop", bp=by.readslice(sizeloop).bp)
        bs.insertblock(text="余块", bp=by.readremainslice().bp)


    def MESH_info_version_1(self):
        size = 30
        with bs.insertblock(text=f"mesh_info_{size}", bp=by.readsliceseek0(size).bp), by.readslice(size):
            顶点总数, 未知1 = by.readuint32(), by.readslice(5).bp
            Loop总数, 未知2 = by.readuint32(), by.readslice(4).bp
            Face总数, 未知3 = by.readuint32(), by.readslice(9).bp
            bs.insertblock(text=f"numvert: {顶点总数}", bp=未知1)
            bs.insertblock(text=f"numloop: {Loop总数}", bp=未知2)
            bs.insertblock(text=f"numface: {Face总数}", bp=未知3)
        return 顶点总数, Loop总数
    

    def MESH_info_version_3(self):
        size = 33
        with bs.insertblock(text=f"mesh_info_{size}", bp=by.readsliceseek0(size).bp), by.readslice(size):
            顶点总数, 未知1 = by.readuint32(), by.readslice(8).bp
            Loop总数, 未知2 = by.readuint32(), by.readslice(4).bp
            Face总数, 未知3 = by.readuint32(), by.readslice(9).bp # 0xFFFF FFFF
            bs.insertblock(text=f"numvert: {顶点总数}", bp=未知1)
            bs.insertblock(text=f"numloop: {Loop总数}", bp=未知2)
            bs.insertblock(text=f"numface: {Face总数}", bp=未知3)
        return 顶点总数, Loop总数
    
    def MESH_info_version_5(self):
        size = 36
        with bs.insertblock(text=f"mesh_info_{size}", bp=by.readsliceseek0(size).bp), by.readslice(size):
            顶点总数, 未知1 = by.readuint32(), by.readslice(9).bp
            Loop总数, 未知2 = by.readuint32(), by.readslice(4).bp
            Face总数, 未知3 = by.readuint32(), by.readslice(11).bp
            bs.insertblock(text=f"numvert: {顶点总数}", bp=未知1)
            bs.insertblock(text=f"numloop: {Loop总数}", bp=未知2)
            bs.insertblock(text=f"numface: {Face总数}", bp=未知3)
        return 顶点总数, Loop总数

# from Tkinter import Tk
# r = Tk()
# r.withdraw()
# r.clipboard_clear()
# r.clipboard_append('i can has clipboardz?')
# r.destroy()