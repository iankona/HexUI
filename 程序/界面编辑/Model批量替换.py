
import tkinter as tk
import tkinter.ttk as ttk


from . import Tree基础编辑
from . import Tree基础路径
from . import 编辑Model






class PanelFrame(ttk.Widget):
    def __init__(self, master=None, **kwargs):
        ttk.Widget.__init__(self, master, "ttk::frame", kwargs)
        self.__添加__部件__()
        self.__绑定__按钮__函数__()
        self.__绑定__显示__函数__()



    def __添加__部件__(self):
        panela = self.__添加__面板__区域__() # Panel 实例在前，界面才能正常显示
        framea = self.__添加__左边__区域__()
        frameb = self.__添加__中间__区域__()
        framec = self.__添加__右边__区域__()
        panela.add(framea)
        panela.add(frameb)
        panela.add(framec)
        self.frameread = framea
        self.framepath = frameb
        self.frameedit = framec


    def __添加__面板__区域__(self):
        panel = ttk.PanedWindow(self, orient="horizontal")
        panel.pack(side="top", fill='both', expand=1)
        return panel

    def __添加__左边__区域__(self):
        frame = TreeFrame1(self, width=300, height=600)
        frame.pack_propagate(0)
        frame.pack(side="left", fill='both')
        return frame
    

    def __添加__中间__区域__(self):
        frame = Tree基础路径.TreeFrame(self, width=350, height=600)
        frame.pack_propagate(0)
        frame.pack(side="left", fill='both')
        return frame


    def __添加__右边__区域__(self):
        frame = TreeFrame3(self, width=300, height=600)
        frame.pack_propagate(0)
        frame.pack(side="left", fill='both')
        return frame


    def __绑定__按钮__函数__(self):
        # 以下三种修改按钮执行函数的方法，任选一种就行
        self.framepath.frameoption.按钮替换["command"] = self.__批量替换节点__
        # self.framepath.frameoption.按钮替换.config(command=self.__批量替换节点__)
        # self.framepath.frameoption.按钮替换.configure(command=self.__批量替换节点__)
        self.frameedit.frameoption.按钮保存.configure(command=self.__批量保存文件__)

    def __批量替换节点__(self):
        bnode_select = self.frameread.bnode_select
        if bnode_select == None: 
            self.framepath.frametext.插入文本(f"替换节点::请先在左侧选择节点...")
            return None

        items = self.framepath.treeview.get_children()
        indexs = [] 
        for item in items:
            values = self.framepath.treeview.item(item)["values"]
            indexs.append(values[0])
        filebnodes = []
        for item in items:
            fileinfo = self.framepath.pathdict[item]
            filebnodes.append(fileinfo.filebnode)

        for filebnode, index in zip(filebnodes, indexs): __替换节点__(bnode_select, filebnode, index, self.framepath)


    def __批量保存文件__(self):
        后缀 = self.frameedit.frameoption.输入后缀.get()
        Tree基础路径.批量保存文件(self.framepath, 后缀)


    def __绑定__显示__函数__(self):
        # self.framepath.treeview.bind("<<TreeviewSelect>>", self.__刷新__显示__函数__) # 有event
        self.framepath.rightmenu.entryconfig("文件至右侧显示", command=self.__刷新__显示__函数__) # 没有event

        
    def __刷新__显示__函数__(self):
        item = self.framepath.treeview.selection()[0]
        fileinfo = self.framepath.pathdict[item]
        Tree基础编辑.清空节点(self.frameedit)
        Tree基础编辑.递归显示(self.frameedit, fitem="", fbnode=fileinfo.filebnode)
        self.framepath.frametext.插入文本(f"刷新显示::已选文件{fileinfo.basename}，右侧显示同步刷新")
        self.frameedit.frametext.插入文本(f"文件显示::{fileinfo.filepath}")


def __替换节点__(bnode_select, filebnode, index, frametreeview):
    srtmbg = filebnode.name_object_dict["SRTM"]
    keyname = f"lrtm_{index}"
    bnode_temp = srtmbg[keyname]
    srtmbg[keyname] = bnode_select
    frametreeview.frametext.插入文本(f"替换节点::{bnode_temp.name} --> {bnode_select.name}", f"{bnode_temp} --> {bnode_select}")



    

class TreeFrame1(编辑Model.FrameModel):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.__释放__文件__菜单__()
        self.__删除__右键__菜单__()
        self.__附加__选项__区域__()
        self.bnode_select = None

    def __释放__文件__菜单__(self):
        self.frametoolbar.buttonfile.destroy()


    def __删除__右键__菜单__(self):
        self.rightmenu.delete('复制节点')
        self.rightmenu.delete('粘贴节点')
        self.rightmenu.delete('回退粘贴')
        self.rightmenu.add_command(label="选择节点", command=self.__选择节点__)



    def __附加__选项__区域__(self):
        row, column = 0, 0
        buttona = Tree基础编辑.__标签按钮部件__(self.frameoption, text="材质节点已选索引:", row=row, column=column, width=15)
        column += 2
        comboxa = Tree基础编辑.__标签选取部件__(self.frameoption, text="", row=row, column=column, width=20)
        column += 2
        self.frameoption.combox_select = comboxa



    def __打开文件__(self):
        super().__打开文件__()
        srtmbg = self.filebnode.name_object_dict["SRTM"]
        材质数量 = srtmbg["材质数量"].uint32
        self.frameoption.combox_select["values"] = [i for i in range(材质数量)]


    def __选择节点__(self):
        if "lrtm_" not in self.bnode.name: 
            self.frametext.插入文本("选择节点::您所选择的节点不是lrtm节点")
            return None
        index = int(self.bnode.name.split("_")[-1])
        self.frameoption.combox_select.current(index)
        self.bnode_select = self.bnode
        self.frametext.插入文本(f"选择节点::{self.bnode.name}", f"{self.bnode}")


class TreeFrame3(编辑Model.FrameModel):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.__释放__文件__菜单__()
        self.__删除__右键__菜单__()
        self.__附加__选项__区域__()


    def __释放__文件__菜单__(self):
        self.frametoolbar.buttonfile.destroy()

    def __删除__右键__菜单__(self):
        self.rightmenu.delete(2) # 索引需先删除
        self.rightmenu.delete("打开文件") # 0
        # self.rightmenu.delete("保存文件") # 1
        self.rightmenu.delete('复制节点')
        self.rightmenu.delete('粘贴节点')
        self.rightmenu.delete('回退粘贴')


    def __附加__选项__区域__(self):
        row, column = 0, 0
        button = Tree基础编辑.__标签按钮部件__(self.frameoption, text="批量后缀:", row=row, column=column, width=8)
        column += 2
        entrya = Tree基础编辑.__标签输入部件__(self.frameoption, text="", row=row, column=column, width=13)
        column += 2
        labela = Tree基础编辑.__标签标签部件__(self.frameoption, text=".model", row=row, column=column, width=0)
        column += 2
        button = Tree基础编辑.__标签按钮部件__(self.frameoption, text="批量保存", row=row, column=column, width=13)
        column += 2
        self.frameoption.输入后缀 = entrya
        self.frameoption.按钮保存 = button

        
