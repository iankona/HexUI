import tkinter.ttk as ttk


from . import Tree基础显示
from . import View快速显示



class Frame(ttk.Widget):
    def __init__(self, master=None, **kwargs):
        ttk.Widget.__init__(self, master, "ttk::frame", kwargs)
        self.__添加__部件__()
        self.__绑定__显示__函数__()



    def __添加__部件__(self):
        panela = self.__添加__面板__区域__() # Panel 实例在前，界面才能正常显示
        framea = self.__添加__左边__区域__()
        framec = self.__添加__右边__区域__()
        panela.add(framea)
        panela.add(framec)
        self.frametree = framea
        self.frameview = framec


    def __添加__面板__区域__(self):
        panel = ttk.PanedWindow(self, orient="horizontal")
        panel.pack(side="top", fill='both', expand=1)
        return panel

    def __添加__左边__区域__(self):
        frame = Tree基础显示.TreeFrame(self, width=450, height=600)
        frame.pack_propagate(0)
        frame.pack(side="left", fill='both')
        return frame
    

    def __添加__右边__区域__(self):
        frame = View快速显示.ViewFrame(self, width=550, height=600)
        frame.pack_propagate(0)
        frame.pack(side="left", fill='both')
        return frame

    def __绑定__显示__函数__(self):
        self.frametree.rightmenu.entryconfig("数据集to右边", command=self.__数据集to右边__) # 没有event


    def __数据集to右边__(self):
        选择集列表 = []
        items = self.frametree.treeview.selection()
        for item in items:
            if self.frametree.bpdict[item] is None: continue
            bp = self.frametree.bpdict[item].copy()
            选择集列表.append(bp)
        self.frameview.frameoption.选择集列表 = 选择集列表