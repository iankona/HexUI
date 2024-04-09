import tkinter.ttk as ttk

from . import Model材质替换
from . import Model批量替换

class Frame(ttk.Widget):
    def __init__(self, master=None, **kw):
        ttk.Widget.__init__(self, master, "ttk::frame", kw)
        self.__notebook__ = ttk.Notebook(self, padding=5) 
        self.__notebook__.pack(fill="both", expand=1)
        self.__notebook__.add(__添加材质替换面板__(self.__notebook__), text=f'　材质替换　')
        self.__notebook__.add(__添加批量替换面板__(self.__notebook__), text=f'　批量替换　')


def __添加材质替换面板__(parent):
    frame = Model材质替换.PanelFrame(parent)
    frame.pack(side="left", fill='both')
    return frame

def __添加批量替换面板__(parent):
    frame = Model批量替换.PanelFrame(parent)
    frame.pack(side="left", fill='both')
    return frame