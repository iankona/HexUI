
import tkinter as tk
import tkinter.ttk as ttk



from . import 编辑Model



class PanelFrame(ttk.Widget):
    def __init__(self, master=None, **kwargs):
        ttk.Widget.__init__(self, master, "ttk::frame", kwargs)
        self.__添加__部件__()

    def __添加__部件__(self):
        panela = self.__添加__面板__区域__() # Panel 实例在前，界面才能正常显示
        framea = self.__添加__左边__区域__()
        frameb = self.__添加__右边__区域__()
        panela.add(framea)
        panela.add(frameb)
        self.frametree = framea
        self.frameedit = frameb


    def __添加__面板__区域__(self):
        panel = ttk.PanedWindow(self, orient="horizontal")
        panel.pack(side="top", fill='both', expand=1)
        return panel

    def __添加__左边__区域__(self):
        frame = 编辑Model.FrameModel(self, width=450, height=600)
        frame.pack_propagate(0)
        frame.pack(side="left", fill='both')
        return frame
    
    def __添加__右边__区域__(self):
        frame = 编辑Model.FrameModel(self, width=550, height=600)
        frame.pack_propagate(0)
        frame.pack(side="left", fill='both')
        return frame



