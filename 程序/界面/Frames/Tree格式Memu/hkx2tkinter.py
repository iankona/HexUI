import tkinter as tk
import tkinter.ttk as ttk
import tkinter.filedialog

from . import bsfunction as bs
import 程序.文件.bpformat.byfunction as by
func = by

import 程序.文件 as 文件




class 类:
    def __init__(self, frametree, 菜单):
        self.frametree = frametree
        子菜单 = tk.Menu(菜单, tearoff=False)
        菜单.add_cascade(label = "古剑3/HKX", menu=子菜单)
        子菜单.add_command(label="文件解析", command=self.文件解析)
        # 子菜单.add_command(label="文件分块", command=self.文件分块)

        # 子菜单.add_separator() # 添加分割线
    def context(self, item=""):
        if item == "": item = self.frametree.tree.selection()[0]
        bs.bscontext.frametree = self.frametree
        bs.bscontext.item = item
        global by
        by = func.context(self.frametree.bpdict[item].copy()) # bp = self.frametree.bpdict[item]
        return item


    def 文件解析(self):
        for item in self.frametree.tree.selection():
            self.context(item)
            bp = by.filepath(by.rdpath()).bp
            hkxfile = 文件.hkx.类(bp)
            for char, value in hkxfile.blocks.items():
                bs.insertblock(text=char, bp=value.copy())

















    def 文件解析222(self):
        blockitems, hkafile = self.文件分块()

        self.frametree.tree.item(blockitems[0], text="header")
        for varname, varvalue in hkafile.header.items():
            if not isinstance(varvalue, list): varvalue = [varvalue]
            self.frametree.insertvalue(blockitems[0], index="end", text=varname, displayvalues=varvalue)

        self.frametree.tree.item(blockitems[1], text="__type__")
        for varname, varvalue in hkafile.types.items():
            [typeflag, classname, layout, parantindex, numelement, elements] = varvalue
            fitem = self.frametree.insertvalue(blockitems[1], index="end", text=classname, displayvalues=[typeflag, "", layout, parantindex, numelement])
            for [varname, vartype, havoktype] in elements:
                self.frametree.insertvalue(fitem, index="end", text="", displayvalues=["", varname, vartype, havoktype])

        dataitem = ""
        for item, values in zip(blockitems[2:], hkafile.datas):
            [classname, typeflag, classnameindex, classflag, datainstance] = values
            self.frametree.tree.item(item, text=classname)
            self.frametree.insertvalue(item, index="end", text="", displayvalues=[typeflag, classnameindex, classflag])
            for varname, varvalue in datainstance.__dict__.items():
                if classname == "hkaSplineCompressedAnimation" and varname == "data":
                    dataitem = self.frametree.insertblock(item, index="end", text="data", bp=datainstance.data.copy())
                    continue
                if not isinstance(varvalue, list): varvalue = [varvalue]
                self.frametree.insertvalue(item, index="end", text=varname, displayvalues=varvalue)

        count = 0
        for datablock, [maskstream, uint8stream, float32stream], [maskvalues, masksizes, maskblocks] in zip(hkafile.datablocks, hkafile.maskuint8float32blocks, hkafile.maskuint8datas):
            count += 1
            fitem = self.frametree.insertblock(dataitem, index="end", text="datablock_"+str(count), bp=datablock.copy())
            self.frametree.insertblock(fitem, index="end", text="mask", bp=maskstream.copy())
            self.frametree.insertblock(fitem, index="end", text="uint8", bp=uint8stream.copy())
            self.frametree.insertblock(fitem, index="end", text="float32", bp=float32stream.copy())

            num = 0
            for mask, \
                size, \
                [positionstream, rotationstream, scalestream] in zip(maskvalues, masksizes, maskblocks):
                num += 1
                aitem = self.frametree.insertvalue(fitem, index="end", text=str(num)+str(mask), displayvalues=size)
                self.frametree.insertblock(aitem, index="end", text="position", bp=positionstream)
                self.frametree.insertblock(aitem, index="end", text="rotation", bp=rotationstream)
                self.frametree.insertblock(aitem, index="end", text="", bp=scalestream)






    def dataposition(self):
        blockitems, hkafile = self.文件分块()

        self.frametree.tree.item(blockitems[0], text="header")
        # for varname, varvalue in hkafile.header.items():
        #     if not isinstance(varvalue, list): varvalue = [varvalue]
        #     self.frametree.insertvalue(blockitems[0], index="end", text=varname, displayvalues=varvalue)

        self.frametree.tree.item(blockitems[1], text="__type__")
        # for varname, varvalue in hkafile.types.items():
        #     [typeflag, classname, layout, parantindex, numelement, elements] = varvalue
        #     fitem = self.frametree.insertvalue(blockitems[1], index="end", text=classname, displayvalues=[typeflag, "", layout, parantindex, numelement])
        #     for [varname, vartype, havoktype] in elements:
        #         self.frametree.insertvalue(fitem, index="end", text="", displayvalues=["", varname, vartype, havoktype])

        dataitem = ""
        for item, values in zip(blockitems[2:], hkafile.datas):
            [classname, typeflag, classnameindex, classflag, datainstance] = values
            self.frametree.tree.item(item, text=classname)
            self.frametree.insertvalue(item, index="end", text="", displayvalues=[typeflag, classnameindex, classflag])
            for varname, varvalue in datainstance.__dict__.items():
                if classname == "hkaSplineCompressedAnimation" and varname == "data":
                    dataitem = self.frametree.insertblock(item, index="end", text="data", bp=datainstance.data.copy())
                    continue
                if not isinstance(varvalue, list): varvalue = [varvalue]
                self.frametree.insertvalue(item, index="end", text=varname, displayvalues=varvalue)

        count = 0
        for datablock, [maskstream, uint8stream, float32stream], [maskvalues, masksizes, maskblocks] in zip(hkafile.datablocks, hkafile.maskuint8float32blocks, hkafile.maskuint8datas):
            count += 1
            fitem = self.frametree.insertblock(dataitem, index="end", text="datablock_"+str(count), bp=datablock.copy())
            # self.frametree.insertblock(fitem, index="end", text="mask", bp=maskstream.copy())
            # self.frametree.insertblock(fitem, index="end", text="uint8", bp=uint8stream.copy())
            # self.frametree.insertblock(fitem, index="end", text="float32", bp=float32stream.copy())

            num = 0
            for mask, \
                size, \
                [positionstream, rotationstream, scalestream] in zip(maskvalues, masksizes, maskblocks):
                num += 1
                if positionstream.size() < 13: continue
                maskhex = ["%02X" % uint8 for uint8 in mask]
                numtype, padenum = self.positiondatatype(mask[1])
                self.frametree.insertblock(fitem, index="end", text=f"{num}_mask[1]_0x{maskhex[1]}_{numtype}_{padenum}", bp=positionstream)


    def positiondatatype(self, positiontype):
        positionthex = "%02X" % positiontype
        if   positionthex[0] in ["1", "2", "4"]: numtype = 2
        elif positionthex[0] in ["3", "5", "6"]: numtype = 4
        elif positionthex[0] in ["7", "8"     ]: numtype = 6
        else: print("hka：readpositionfcurvesize：未知positiontype: ", positiontype)
        if numtype == 2:
            if   positionthex[1] in ["0"          ]: padenum = 4
            elif positionthex[1] in ["1", "2", "4"]: padenum = 6
            elif positionthex[1] in ["3", "5", "6"]: padenum = 8
            else: print("hka：readpositionfcurvesize：未知positiontype: ", positiontype)
        if numtype == 4:
            if   positionthex[1] in ["0"          ]: padenum = 4
            elif positionthex[1] in ["1", "2", "4"]: padenum = 5
            else: print("hka：readpositionfcurvesize：未知positiontype: ", positiontype)
        if numtype == 6:
            if   positionthex[1] in ["0"          ]: padenum = 4
            else: print("hka：readpositionfcurvesize：未知positiontype: ", positiontype)

        return numtype, padenum

