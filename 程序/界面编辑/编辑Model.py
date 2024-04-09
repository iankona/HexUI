import tkinter as tk
import tkinter.ttk as ttk
import tkinter.filedialog

import os
import 底层编辑

from . import Tree基础编辑


class __节点历史__:
    def __init__(self):
        self.临时节点 = None
        self.节点计数 = -1
        self.节点列表 = [None for i in range(10)]

    def 插入节点(self, 节点):
        self.节点计数 = (self.节点计数 + 1) % 10
        self.节点列表[self.节点计数] = 节点


    def 获取节点(self):
        if self.节点计数 < 0: return None
        return self.节点列表[self.节点计数]

    def 后退节点(self):
        self.节点计数 = (self.节点计数 - 1) % 10 # (-1) % 10 == 9
        return self.节点列表[self.节点计数]

    def 前进节点(self):
        self.节点计数 = (self.节点计数 + 1) % 10
        return self.节点列表[self.节点计数]


节点历史 = __节点历史__()


class FrameModel(Tree基础编辑.TreeFrame):
    def __init__(self, master=None, **kwargs):
        Tree基础编辑.TreeFrame.__init__(self, master, **kwargs)
        self.__更新__文件__函数__()
        self.__附加__编辑__菜单__()
        self.__附加__右键__菜单__()

    def __更新__文件__函数__(self):   
        menu = self.frametoolbar.menufile
        menu.entryconfig("打开文件", command=lambda:__打开文件__(self))
        menu.entryconfig("保存文件", command=lambda:__保存文件__(self))
        menu = self.rightmenu
        menu.entryconfig("打开文件", command=lambda:__打开文件__(self))
        menu.entryconfig("保存文件", command=lambda:__保存文件__(self))


    def __附加__编辑__菜单__(self):   
        menu = self.frametoolbar.menuedit
        menu.add_separator() # 添加分割线
        菜单=tk.Menu(menu, tearoff=False)
        menu.add_cascade(label="添加贴图栏", menu=菜单)
        菜单.add_command(label='第4~第6栏',command=self.__添加__第3至第6栏贴图__)
        菜单.add_command(label='第7~第9栏',command=self.__添加__第7至第9栏贴图__)
        menu.add_separator() # 添加分割线
        菜单=tk.Menu(menu, tearoff=False)
        menu.add_cascade(label="删除贴图栏", menu=菜单)
        菜单.add_command(label='第4~第6栏',command=self.__删除__第3至第6栏贴图__)
        菜单.add_command(label='第7~第9栏',command=self.__删除__第7至第9栏贴图__)
        


    def __添加__第3至第6栏贴图__(self):
        __添加__第3至第6栏贴图__(self)

    def __添加__第7至第9栏贴图__(self):
        __添加__第7至第9栏贴图__(self)

    def __删除__第3至第6栏贴图__(self):
        __删除__第3至第6栏贴图__(self)

    def __删除__第7至第9栏贴图__(self):
        __删除__第7至第9栏贴图__(self)


    def __附加__右键__菜单__(self):
        menu = self.rightmenu
        menu.add_separator() # 添加分割线
        menu.add_command(label='复制节点', command=self.__复制节点__)
        menu.add_command(label='粘贴节点', command=self.__粘贴节点__)
        menu.add_command(label='回退粘贴', command=self.__回退粘贴__)


    def __复制节点__(self):
        if "lrtm_" not in self.bnode.name:
            self.frametext.插入文本(f"复制节点::请选择lrtm节点复制")
            return None
        节点历史.插入节点(self.bnode)
        self.frametext.插入文本(f"复制节点::{self.bnode.name}", f"{self.bnode}")


    def __粘贴节点__(self):
        if self.item == "": return None
        if "lrtm_" not in self.bnode.name:
            self.frametext.插入文本(f"粘贴节点::请先选择lrtm节点")
            return None

        节点历史.临时节点 = self.bnode
        bnode = 节点历史.获取节点()
        classtype = str(type(bnode))
        if "bgmultipletree.类" in classtype:
            __替换节点__(bnode, self)
            self.frametext.插入文本(f"粘贴节点::{节点历史.临时节点.name} --> {bnode.name}", f"{节点历史.临时节点} --> {bnode}")
        else:
            self.frametext.插入文本(f"粘贴节点::类型{classtype}不支持")


    def __回退粘贴__(self):
        if self.item == "": return None
        tempbnode = self.bnode
        bnode = 节点历史.临时节点
        classtype = str(type(bnode))
        if "bgmultipletree.类" in classtype:
            __替换节点__(bnode, self)
            self.frametext.插入文本(f"回退粘贴::{tempbnode.name} --> {bnode.name}", f"{tempbnode} --> {bnode}")
        else:
            self.frametext.插入文本(f"回退粘贴::类型{classtype}不支持")



def __打开文件__(frametreeview):
    Tree基础编辑.清空节点(frametreeview)
    filepath = tkinter.filedialog.askopenfilename(filetypes=[("古剑model", ".model"), ("All files", "*")])
    if filepath == "" or os.path.isfile(filepath) == False: return None
    if filepath.endswith(".model") == False: return None
    filebnode = 底层编辑.model.类().read(filepath)
    Tree基础编辑.递归显示(frametreeview, fitem="", fbnode=filebnode)
    frametreeview.filebnode = filebnode
    frametreeview.frametext.插入文本(f"打开文件::{filepath}")


def __保存文件__(frametreeview): # def 保存浏览文件单选():
    filepath = frametreeview.filebnode.filepath
    dirname = os.path.dirname(filepath)
    basename = os.path.basename(filepath)
    filepath = tkinter.filedialog.asksaveasfilename(initialdir=dirname, 
                                                    initialfile=basename[0:-6]+"_new.model",
                                                    filetypes=[("古剑model", ".model"), ("All files", "*")]
                                                    )
    if filepath == "": return None
    frametreeview.filebnode.write(filepath)
    frametreeview.frametext.插入文本(f"保存文件::{filepath}")


def __添加__贴图栏__(keyname, bnode, i):
    if bnode.uint32 == 3: return None
    lrtmbnode = bnode.parent
    lrtmbnode.addstart(keyname)
    lrtmbnode.addstring("", name=f"{i}_贴图名称_0", inttype="uint32", strtype="utf8")
    lrtmbnode.addstring("", name=f"{i}_贴图名称_1", inttype="uint32", strtype="utf8")
    lrtmbnode.addstring("", name=f"{i}_贴图名称_2", inttype="uint32", strtype="utf8")
    lrtmbnode.addfinal()
    bnode.uint32 = 3


def __删除__贴图栏__(keyname, bnode, i):
    if bnode.uint32 == 0: return None
    keynames = list(bnode.parent.keys())
    index = keynames.index(keyname)
    for name in keynames[index+1: index+4]: 
        if f"{i}_贴图名称_" in name: bnode.parent.pop(name)
    bnode.uint32 = 0



def __添加__第3至第6栏贴图__(frametreeview):
    if frametreeview.item == "": return None
    srtmbg = frametreeview.filebnode.name_object_dict["SRTM"]
    for name, bnode in srtmbg.items(): 
        if "lrtm_" not in name: continue
        for keyname, cnode in list(bnode.items()): # RuntimeError: dictionary changed size during iteration
            if keyname == "第2栏贴图数量": __添加__贴图栏__(keyname, cnode, 2)

    Tree基础编辑.清空节点(frametreeview)
    Tree基础编辑.递归显示(frametreeview, fitem="", fbnode=frametreeview.filebnode)

def __添加__第7至第9栏贴图__(frametreeview):
    if frametreeview.item == "": return None
    srtmbg = frametreeview.filebnode.name_object_dict["SRTM"]
    for name, bnode in srtmbg.items():
        if "lrtm_" not in name: continue
        for keyname, cnode in list(bnode.items()):
            if keyname == "第3栏贴图数量": __添加__贴图栏__(keyname, cnode, 3)

    Tree基础编辑.清空节点(frametreeview)
    Tree基础编辑.递归显示(frametreeview, fitem="", fbnode=frametreeview.filebnode)

def __删除__第3至第6栏贴图__(frametreeview):
    if frametreeview.item == "": return None
    srtmbg = frametreeview.filebnode.name_object_dict["SRTM"]
    for name, bnode in srtmbg.items():
        if "lrtm_" not in name: continue
        for keyname, cnode in list(bnode.items()):
            if keyname == "第2栏贴图数量": __删除__贴图栏__(keyname, cnode, 2)

    Tree基础编辑.清空节点(frametreeview)
    Tree基础编辑.递归显示(frametreeview, fitem="", fbnode=frametreeview.filebnode)

def __删除__第7至第9栏贴图__(frametreeview):
    if frametreeview.item == "": return None
    srtmbg = frametreeview.filebnode.name_object_dict["SRTM"]
    for name, bnode in list(srtmbg.items()):
        if "lrtm_" not in name: continue
        for keyname, cnode in list(bnode.items()):
            if keyname == "第3栏贴图数量": __删除__贴图栏__(keyname, cnode, 3)

    Tree基础编辑.清空节点(frametreeview)
    Tree基础编辑.递归显示(frametreeview, fitem="", fbnode=frametreeview.filebnode)



def __替换节点__(anode, frametreeview): # a 替换 b
    item = frametreeview.item
    bnode = frametreeview.bnode

    keyname = bnode.parent.getname_fromobject(bnode)
    bnode.parent[keyname] = anode

    frametreeview.itemblock(item, anode)
    frametreeview.bnode = anode

    childitems = frametreeview.treeview.get_children(item)
    for citem in childitems: frametreeview.treeview.delete(citem)

    Tree基础编辑.递归显示(frametreeview, fitem=item, fbnode=anode)

























# def __前进节点__(self):
#     if self.item == "": return None
#     tempbnode = self.bnode
#     bnode = 节点历史.前进节点()
#     classtype = str(type(bnode))
#     if "bgmultipletree.类" in classtype:
#         __替换节点__(bnode, self)
#         self.frametext.插入文本(f"前进节点::{tempbnode} --> {bnode}")
#     else:
#         self.frametext.插入文本(f"前进节点::类型{classtype}不支持")

# def __后退节点__(self):
#     if self.item == "": return None
#     tempbnode = self.bnode
#     bnode = 节点历史.后退节点()
#     classtype = str(type(bnode))
#     if "bgmultipletree.类" in classtype:
#         __替换节点__(bnode, self)
#         self.frametext.插入文本(f"后退节点::{tempbnode} --> {bnode}")
#     else:
#         self.frametext.插入文本(f"后退节点::类型{classtype}不支持")