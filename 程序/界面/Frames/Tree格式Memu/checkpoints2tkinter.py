import tkinter as tk

from . import bsfunction as bs
import 程序.文件.bpformat.function_bpnumpy as by
import 程序.文件.bpformat.function_bpnumpy_inverted as byed



class 类:
    def __init__(self, frametree, 菜单):
        self.frametree = frametree
        子菜单 = tk.Menu(菜单, tearoff=False)
        菜单.add_cascade(label = "Checkpoints/cpkt", menu=子菜单)
        子菜单.add_command(label="文件分块", command=self.文件分块)
        子菜单.add_command(label="PK66分块", command=self.PK66分块)
        # 子菜单.add_command(label="PK分块", command=self.PK分块)


        # 子菜单.add_command(label="dataposition", command=self.dataposition)
        # 子菜单.add_separator() # 添加分割线


    def contextinverted(self, item="", bped=None):
        bs.bscontext.frametree = self.frametree
        bs.bscontext.item = item
        byed.bycontextinverted.data = bped # bp = self.frametree.bpdict[item]
        return item

    def context(self, item="", bp=None):
        bs.bscontext.frametree = self.frametree
        bs.bscontext.item = item
        by.context_bpnumpy.data = bp # bp = self.frametree.bpdict[item]
        return item
    


    def bpinverted(self, filepath):
        bped = byed.filepath(filepath).bped
        return bped








    def 文件分块(self):
        for item in self.frametree.tree.selection():
            bped = self.bpinverted(self.frametree.bpdict[item].rdpath)
            self.contextinverted(item, bped)
            

            bs.insertblock(text=f"PK56_22", bp=byed.readslice(22).bp)
            bs.insertblock(text=f"PK67_20", bp=byed.readslice(20).bp)
            bs.insertblock(text=f"PK66_56", bp=byed.readslice(56).bp)
            with bs.insertvalue(text=f"PK12"):
                count = 0
                while True:
                    count += 1
                    size = self.findPK12()
                    if size == 0: break
                    bs.insertblock(text=f"{count}_PK12_{size}", bp=byed.readslice(size).bp)

            aitem = bs.insertvalue(text=f"PK78").citem
            bitem = bs.insertvalue(text=f"块表").citem
            citem = bs.insertvalue(text=f"PK34").citem
            count = 0
            while True:
                count += 1
                if byed.remainsize() < 16: break
                bp = byed.readsliceseek0(16).bp
                flag, some, size = bp.readuint8(4), bp.readuint8(4), bp.readuint32()
                bs.insertblock(item=aitem, text=f"{count}_PK78_16", bp=byed.readslice(16).bp)
                bs.insertblock(item=bitem, text=f"{count}_块_{size}", bp=byed.readslice(size).bp)
                size = self.findPK34()
                if size == 0: break
                bs.insertblock(item=citem, text=f"{count}_PK34_{size}", bp=byed.readslice(size).bp)



            bs.insertblock(text=f"余块_{byed.remainsize()}", bp=byed.readremainslice().bp)




    def PK66分块(self):
        for item in self.frametree.tree.selection():
            self.context(item, self.frametree.bpdict[item].copy())
            bs.insertblock(text=f"flag_4", bp=by.readslice(4).bp)
            bs.insertblock(text=f"size_4", bp=by.readslice(4).bp)
            bs.insertblock(text=f"some_16", bp=by.readslice(16).bp)
            num = by.readuint32seek0()
            bs.insertblock(text=f"numpk12_{num}_8", bp=by.readslice(8).bp)
            num = by.readuint32seek0()
            bs.insertblock(text=f"numpk78_{num}_8", bp=by.readslice(8).bp)
            size = by.readuint32seek0()
            bs.insertblock(text=f"sizepk12_{size}_8", bp=by.readslice(8).bp)
            size = by.readuint32seek0()
            bs.insertblock(text=f"sizepk12_{size}_8", bp=by.readslice(8).bp)
            bs.insertblock(text=f"余下_{by.remainsize()}", bp=by.readremainslice().bp)


    def findPK12(self):
        size = 0
        bp = byed.readsliceseek0(60).bp
        if bp.readuint8(4) == [80, 75, 1, 2]: size = 60

        bp = byed.readsliceseek0(61).bp
        if bp.readuint8(4) == [80, 75, 1, 2]: size = 61

        bp = byed.readsliceseek0(62).bp
        if bp.readuint8(4) == [80, 75, 1, 2]: size = 62

        bp = byed.readsliceseek0(63).bp
        if bp.readuint8(4) == [80, 75, 1, 2]: size = 63
        return size



    def findPK34(self):
        size = 0

        bp = byed.readsliceseek0(48).bp
        if bp.readuint8(4) == [80, 75, 3, 4]: size = 48

        bp = byed.readsliceseek0(62).bp
        if bp.readuint8(4) == [80, 75, 3, 4]: size = 62

        bp = byed.readsliceseek0(72).bp
        if bp.readuint8(4) == [80, 75, 3, 4]: size = 72

        bp = byed.readsliceseek0(80).bp
        if bp.readuint8(4) == [80, 75, 3, 4]: size = 80

        bp = byed.readsliceseek0(96).bp
        if bp.readuint8(4) == [80, 75, 3, 4]: size = 96

        bp = byed.readsliceseek0(100).bp
        if bp.readuint8(4) == [80, 75, 3, 4]: size = 100


        bp = byed.readsliceseek0(112).bp
        if bp.readuint8(4) == [80, 75, 3, 4]: size = 112

        return size




    def 文件分块0(self):
        for item in self.frametree.tree.selection():
            self.context(item)
            indexs = [
                0 , 
                171314 , 
                171330 ,
                217472 , 
                217488 ,
                218816 ,
                218832 ,
                628544 , 
                628560 ,
                631232 ,
                631248 ,
                634432 , 
                634448 ,
                2993856 , 
                2993872 ,
                2997056 ,
                2997072 ,
                5356480 , 
                5356496 ,
                5359680 ,
                5359696 ,
                7719104 , 
                7719120 ,
                7722304 ,
                7722320 ,
                10081728 , 
                10081744 ,
                10084928 ,
                10084944 ,
                10088128 ,
                10088144 ,
                10090816 , 
                10090832 ,
                10094016 ,
                10094032 ,
                19531328 , 
                19531344 ,
                19543744 , 
                19543760 ,
                28981056 , 
                28981072 ,
                28984256 ,
                28984272 ,
                28987456 ,
                28987472 ,
                28990656 , 
                28990672 ,
                31350080 , 
                31350096 ,
                31353280 ,
                31353296 ,
                33712704 , 
                33712720 ,
                33715392 ,
                33715408 ,
                33718592 , 
                33718608 ,
                36078016 , 
                36078032 ,
                36081216 ,
                36081232 ,
                38440640 , 
                38440656 ,
                38443840 ,
                38443856 ,
                38447040 ,
                38447056 ,
                38450240 , 
                38450256 ,
                47887552 , 
                47887568 ,
                47899968 , 
                47899984 ,
            ]
            sizes = [indexs[i+1]-indexs[i] for i in range(len(indexs)-1)]
            sizes = [ size for size in sizes if size > 0 ]
            for size in sizes: bs.insertblock(text=f"{by.readuint8seek0(4)}_{size}", bp=by.readslice(size).bp)
            bs.insertblock(text="余下", bp=by.readremainslice().bp)
