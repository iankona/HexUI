import tkinter as tk
import tkinter.ttk as ttk
import tkinter.filedialog

import 界面.bcontext.bsfunction as bs
import 文件.bpformat.byfunction as by
import 文件




class 类:
    def __init__(self, frametreeview, 菜单):
        self.frametreeview = frametreeview
        子菜单 = tk.Menu(菜单, tearoff=False)
        菜单.add_cascade(label = "古剑3/havok", menu=子菜单)
        子菜单.add_command(label="HKA文件解析", command=self.HKA文件解析)
        子菜单.add_command(label="HKX文件解析", command=self.HKX文件解析)
        子菜单.add_command(label="HKS文件解析", command=self.HKS文件解析)
        子菜单.add_command(label="HKT文件解析", command=self.HKT文件解析)
        # 子菜单.add_separator() # 添加分割线


    def context(self, item="", bp=None):
        if item == "": item = self.frametreeview.treeview.selection()[0]
        by.context(bp)
        bs.context(item, self.frametreeview)
        return item


    def __type__to__treeview__(self, types):
        with bs.insertvalue(text="__type__", values=[]):
            for i, [typeflag, classname, layout, parantindex, numelement, elements] in enumerate(types):
                with bs.insertvalue(text=f"_{i}_{classname}", values=[typeflag, classname, layout, parantindex, numelement]):
                    for varname, vartype, datatype, havoktype in elements:
                        bs.insertvalue(text=varname, values=[varname, vartype, datatype, havoktype])


    def __node__to__treeview__(self, name, result):
        if isinstance(result, dict):
            with bs.insertvalue(text=name, values=[]):
                for key, value in result.items(): self.__node__to__treeview__(key, value)
            return None
        if isinstance(result, list):
            with bs.insertvalue(text=name, values=result):
                for i, value in enumerate(result): self.__node__to__treeview__(f"_{i}", value)
            return None
        
        classchar = str(type(result))
        if "文件.havok." in classchar:
            with bs.insertvalue(text=name, values=[]):
                for key, value in result.__dict__.items(): self.__node__to__treeview__(key, value)
            return None
        if "文件.bpformat." in classchar:
            bs.insertblock(text=name, bp=result.copy())
            return None
        
        bs.insertvalue(text=name, values=[result])


    def __data__to__treeview__(self, datas):
        with bs.insertvalue(text="__data__", values=[]):
            for typeflag, classname, classflag, o in datas:
                with bs.insertvalue(text=classname, values=[typeflag, classname, classflag]):
                    for varname, varvalue in o.__dict__.items(): self.__node__to__treeview__(varname, varvalue)


    def __block__to__treeview__(self, blocks):
        with bs.insertvalue(text="__blocks__", values=[]):
            for char, value in blocks.items(): 
                if "_type_" in char: continue
                bs.insertblock(text=char, bp=value)


    def __type__to__file__(self, types, filepath="havoktype.py"):
        file = open(filepath, mode="w")
        for typeflag, classname, layout, parantindex, numelement, elements in types:
            file.write(f"{[typeflag, classname, layout, parantindex, numelement]}\n")
            for varname, vartype, datatype, havoktype in elements:
                file.write(f"{[varname, vartype, datatype, havoktype]}\n")
            file.write(f"\n")
        file.close()



    def HKA文件解析(self):
        for item in self.frametreeview.treeview.selection():
            bp = self.frametreeview.bpdict[item].copy()
            self.context(item, bp)
            hkafile = 文件.hka.类(bp)
            self.__block__to__treeview__(hkafile.blocks)
            self.__type__to__treeview__(hkafile.types)
            self.__data__to__treeview__(hkafile.datas)
            self.__type__to__file__(hkafile.types, "hkatype.py")


    def HKX文件解析(self):
        for item in self.frametreeview.treeview.selection():
            bp = self.frametreeview.bpdict[item].copy()
            self.context(item, bp)
            hkxfile = 文件.hkx.类(bp)
            self.__block__to__treeview__(hkxfile.blocks)
            self.__type__to__treeview__(hkxfile.types)
            self.__data__to__treeview__(hkxfile.datas)
            self.__type__to__file__(hkxfile.types, "hkxtype.py")


    def HKS文件解析(self):
        for item in self.frametreeview.treeview.selection():
            bp = self.frametreeview.bpdict[item].copy()
            self.context(item, bp)
            hksfile = 文件.hks.类(bp)
            self.__block__to__treeview__(hksfile.blocks)
            self.__type__to__treeview__(hksfile.types)
            self.__data__to__treeview__(hksfile.datas)
            # self.__type__to__file__(hksfile.types, "hkstype.py")


    def HKT文件解析(self):
        for item in self.frametreeview.treeview.selection():
            bp = self.frametreeview.bpdict[item].copy()
            self.context(item, bp)
            hktfile = 文件.hkt.类(bp)
            self.__block__to__treeview__(hktfile.blocks)
            self.__type__to__treeview__(hktfile.types)
            self.__data__to__treeview__(hktfile.datas)
            # self.__type__to__file__(hktfile.types, "hkttype.py")



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

