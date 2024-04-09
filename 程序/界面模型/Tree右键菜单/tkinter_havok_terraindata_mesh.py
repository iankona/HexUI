import tkinter as tk
import tkinter.ttk as ttk
import tkinter.filedialog

import 界面.bcontext.bsfunction as bs
import 文件.bpformat.byfunction as by
import 文件


from . import model2tkinter

class 类:
    def __init__(self, frametreeview, 菜单):
        self.frametreeview = frametreeview
        子菜单 = tk.Menu(None, tearoff=False)
        菜单.add_cascade(label = "古剑3/terraindata", menu=子菜单)
        子菜单.add_command(label="MESH文件分块", command=self.MESH文件分块Wrapper)
        # 子菜单.add_command(label="HKX文件解析", command=self.HKX文件解析)
        # 子菜单.add_command(label="HKS文件解析", command=self.HKS文件解析)
        # 子菜单.add_command(label="HKT文件解析", command=self.HKT文件解析)
        # 子菜单.add_separator() # 添加分割线


    def context(self, item="", bp=None):
        if item == "": item = self.frametreeview.treeview.selection()[0]
        by.context(bp)
        bs.context(item, self.frametreeview)
        return item
    

    def MESH文件解析(self):
        for item in self.frametreeview.treeview.selection():
            bp = self.frametreeview.bpdict[item].copy()
            self.context(item, bp)
            # hkafile = 文件.hka.类(bp)


    def MESH文件分块Wrapper(self):
        for item in self.frametreeview.treeview.selection():
            bp = self.frametreeview.bpdict[item].copy()
            self.context(item, bp)
            self.MESH文件分块()


    def MESH文件分块(self):
        bs.insertblock(text="块_49", bp=by.readslice(49).bp)
        self.MESH_Vert()
        self.MESH_Loop()
        bs.insertblock(text="块_6144", bp=by.readslice(6144).bp)
        self.MESH_Page()
        with bs.insertvalue(text=f"MESH_"):
            model2tkinter.MESH()
        with bs.insertvalue(text=f"块_Material"):
            self.MESH_Material()
        bs.insertblock(text=f"余下_{by.remainsize()}", bp=by.readremainslice().bp)


    def MESH_Vert(self): # only 1 result
        numb = by.readuint32seek0()
        with bs.insertvalue(text=f"Vert块_{numb}"):  
            bs.insertblock(text=f"numb_4", bp=by.readslice(4).bp)
            for i in range(numb):
                size = 60
                with by.readsliceseek0(1024):
                    start = by.tell()
                    bufferuint8 = by.readuint8(60)[-4:]
                    if bufferuint8 == [255,255,255,255]: 
                        numbchar = by.readuint8(4)[2]
                        filepath = by.readchar(numbchar)
                        someuint8 = by.readuint8(5)
                        numbchar = by.readuint32()
                        filepath = by.readchar(numbchar)
                        someuint8 = by.readuint8(48)
                    if bufferuint8 == [1,0,0,128]:
                        someuint8 = by.readuint8(5)
                        numbchar = by.readuint32()
                        filepath = by.readchar(numbchar)
                        someuint8 = by.readuint8(48)  
                    final = by.tell() 
                    size = final -start
                bs.insertblock(text=f"块_{i}_{size}", bp=by.readslice(size).bp)


    def MESH_Loop(self): # only 1 result
        numb = by.readuint32seek0()
        if numb == 0: 
            bs.insertblock(text=f"Loop_4", bp=by.readslice(4).bp)
        else:
            size = 4 + numb*4
            with bs.insertblock(text=f"Loop_{size}", bp=by.readsliceseek0(size).bp), by.readslice(size):
                size1 = 4
                bs.insertblock(text=f"块_{size1}", bp=by.readslice(size1).bp)
                size2 = numb*4
                bs.insertblock(text=f"块_{size2}", bp=by.readslice(size2).bp)

    def MESH_Page(self): # only 1 result
        bs.insertblock(text="Page_932", bp=by.readslice(932).bp)


    def MESH_Material(self):
        count = -1
        while True:
            count += 1
            if by.remainsize() < 20: break
            if by.readuint8seek0() not in [0, 3]: break
            flag = by.readuint8seek0()
            if flag == 0: 
                bs.insertblock(text=f"_{count}_", bp=by.readslice(1).bp)
            if flag == 3: 
                flag, size = by.readuint32(2)
                char, inde = by.readcharseek0(size), by.seek(-8)
                bs.insertblock(text=f"_{count}_{size+8}_{char}", bp=by.readslice(size+8).bp)

                
        numb = by.readuint32seek0()
        with bs.insertblock(text=f"numb_{numb}_4", bp=by.readslice(4).bp):
            for i in range(numb):
                bs.insertblock(text=f"_{i}_{16}", bp=by.readslice(16).bp)




    def MESH_find_Page(self): # only 1 result
        tells = []
        with by.readremainsliceseek0():
            while True:
                if by.remainsize() < 20: break
                if by.readuint8seek0() == 80:
                    if by.readuint8seek0(4) == [80, 97, 103, 101]: tells.append(by.tell())
                by.seek(1)

        sizes = []
        for i in  range(1, len(tells)):
            sizes.append(tells[i]-tells[i-1])

        for size in sizes:
            bs.insertblock(text=f"Page_{size}", bp=by.readslice(size).bp)
                




                
            
