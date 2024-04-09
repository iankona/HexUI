import tkinter as tk


from . import bsfunction as bs
import 文件.bpformat.byfunction as by
func = by

import 文件


class 类:
    def __init__(self, frametreeview, 菜单):
        self.frametreeview = frametreeview
        子菜单 = tk.Menu(菜单, tearoff=False)
        菜单.add_cascade(label = "古剑3/jsonbin", menu=子菜单)
        子菜单.add_command(label="文件解析", command=self.文件解析)        
        # 子菜单.add_command(label="文件分块", command=self.文件分块)


    def context(self, item=""):
        if item == "": item = self.frametreeview.treeview.selection()[0]
        bs.bscontext.frametree = self.frametreeview
        bs.bscontext.item = item
        global by
        by = func.context(self.frametreeview.bpdict[item].copy()) # bp = self.frametree.bpdict[item]
        return item
    


    def 文件解析(self):
        for item in self.frametreeview.treeview.selection():
            self.context(item)
            bp = by.filepath(by.rdpath()).bp
            jsonbinfile = 文件.jsonbin.类(bp)
            self.__node__to__tree__(item, jsonbinfile.result)
            # file = open("bin.json", mode="w", encoding="utf-8")
            # self.__node__to__file__(file, "json", jsonbinfile.result)
            # file.close()



    def __node__to__tree__(self, fitem, result):
        if isinstance(result, dict):
            for key, value in result.items():
                classtype = str(type(value))
                if "bpnumpy" in classtype: 
                    citem = self.frametreeview.insertblock(fitem, text=key, bp=value)
                    continue
                if "bpbytes" in classtype: 
                    citem = self.frametreeview.insertbytes(fitem, text=key, bp=value)
                    continue
                if isinstance(value, dict) or isinstance(value, list): 
                    citem = self.frametreeview.insertvalue(fitem, text=key, values=[])
                    self.__node__to__tree__(citem, value)
                    continue
                citem = self.frametreeview.insertvalue(fitem, text=key, values=[value])

        if isinstance(result, list):
            for i, value in enumerate(result):
                key = f"_{i}"
                classtype = str(type(value))
                if "bpnumpy" in classtype: 
                    citem = self.frametreeview.insertblock(fitem, text=key, bp=value)
                    continue
                if "bpbytes" in classtype: 
                    citem = self.frametreeview.insertbytes(fitem, text=key, bp=value)
                    continue
                if isinstance(value, dict) or isinstance(value, list): 
                    citem = self.frametreeview.insertvalue(fitem, text=key, values=[])
                    self.__node__to__tree__(citem, value)
                    continue
                citem = self.frametreeview.insertvalue(fitem, text=key, values=[value])


    def __node__to__file__(self, file, key, result):
        if isinstance(result, dict):
            file.write(f"\'{key}\':")
            file.write("{\n")
            for key, value in result.items():
                classtype = str(type(value))
                if "bpnumpy" in classtype: 
                    file.write(f"\'{key}\':{value.readuint8seek0(value.size())},\n")
                    continue
                if "bpbytes" in classtype: 
                    file.write(f"\'{key}\':{value.readuint8seek0(value.size())},\n")
                    continue
                if isinstance(value, dict) or isinstance(value, list): 
                    self.__node__to__file__(file, key, value)
                    continue
                if isinstance(value, str): 
                    file.write(f"\'{key}\':\'{value}\',\n")
                    continue
                file.write(f"\'{key}\':{value},\n")
            file.write("},\n")

        if isinstance(result, list):
            file.write(f"\'{key}\':")
            file.write("[\n")
            for i, value in enumerate(result):
                key = f"_{i}"
                classtype = str(type(value))
                if "bpnumpy" in classtype: 
                    file.write(f"\'{key}\':{value.readuint8seek0(value.size())}\n")
                    continue
                if "bpbytes" in classtype: 
                    file.write(f"\'{key}\':{value.readuint8seek0(value.size())}\n")
                    continue
                if isinstance(value, dict) or isinstance(value, list): 
                    self.__node__to__file__(file, key, value)
                    continue
                if isinstance(value, str): 
                    file.write(f"\'{key}\':\'{value}\',\n")
                    continue
                file.write(f"\'{key}\':{value},\n")
            file.write("],\n")

    def 文件分块(self):pass

