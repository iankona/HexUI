import tkinter as tk
import tkinter.ttk as ttk

import 界面文件
import 界面编辑


def 运行():
    app = tk.Tk()
    frame = NoteBookFrame()
    frame.pack(side="top", fill='both', expand=1)
    app.mainloop()



class NoteBookFrame(ttk.Widget):
    def __init__(self, master=None, **kw):
        ttk.Widget.__init__(self, master, "ttk::frame", kw)
        self.__notebook__ = ttk.Notebook(self, padding=5) 
        self.__notebook__.pack(fill="both", expand=1)
        self.__notebook__.add(__添加格式面板__(self.__notebook__), text='　Hex格式　')
        self.__notebook__.add(__添加编辑面板__(self.__notebook__), text='　Hex编辑　')
        # self.__notebook__.add(__添加绘图面板__(self.__notebook__), text='　SDC绘图　')
        # self.__notebook__.add(__添加训练面板__(self.__notebook__), text='　SDC训练　')



def __添加格式面板__(parent):
    frame = 界面文件.Frame.Frame(parent)
    frame.pack(side="left", fill="both", expand=1)
    return frame



def __添加编辑面板__(parent=None):
    notebook = 界面编辑.Frame.Frame(parent)
    notebook.pack(fill="both", expand=1)
    return notebook


def __添加绘图面板__(parent):
    def __添加Tree基础显示__(parent):
        frametree = 界面文件.Tree基础显示.TreeFrame(parent, width=450, height=600)
        frametree.pack_propagate(0)
        frametree.pack(side="left", fill='both')
        return frametree
    def __添加Table快速显示__(parent):
        frametree = 界面文件.View快速显示.ViewFrame(parent, width=550, height=600)
        frametree.pack_propagate(0)
        frametree.pack(side="left", fill='both')
        return frametree

    水平面板 = ttk.Panedwindow(parent, orient='horizontal') 
    水平面板.pack(side="left", fill='both', expand=1)
    # 水平面板.add(__添加Tree基础显示__(水平面板))
    # 水平面板.add(__添加Table快速显示__(水平面板))
    return 水平面板

def __添加训练面板__(parent=None):
    水平面板 = ttk.Panedwindow(parent, orient='horizontal')
    水平面板.pack(side="left", fill='both', expand=1)
    return 水平面板


# def __添加界面编辑__(parent=None):
#     def __添加Tree基础编辑__(parent):
#         frametree = 界面编辑.Tree基础编辑.TreeFrame(parent, width=450, height=600)
#         frametree.pack_propagate(0)
#         frametree.pack(side="left", fill='both')
#         return frametree
#     水平面板 = ttk.Panedwindow(parent, orient='horizontal')
#     水平面板.pack(side="left", fill='both', expand=1)
#     水平面板.add(__添加Tree基础编辑__(水平面板))
#     水平面板.add(__添加Tree基础编辑__(水平面板))
#     return 水平面板