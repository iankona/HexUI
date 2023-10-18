import tkinter.ttk as ttk

from .Frames.TreeFrame import TreeFrame
from .Frames.TableFrame import TableFrame
from .Frames.ScrolledtextFrame import ScrolledtextFrame
from . import 缓存


class 面板Frame(ttk.Widget):
    def __init__(self, master=None, **kw):
        ttk.Widget.__init__(self, master, "ttk::frame", kw)

        self.面板 = ttk.Panedwindow(self, orient='horizontal') # 垂直
        self.面板.pack(side="left", fill='both', expand=1)
        self.添加Tree区()
        # self.添加文本区()
        self.添加表格区()
        
        # 缓存.frame节点.打开默认文件()

    def 添加Tree区(self):
        frametree = TreeFrame(self.面板, width=450, height=600)
        frametree.pack_propagate(0)
        frametree.pack(side="left", fill='both')
        self.面板.add(frametree)
        缓存.frame节点 = frametree

    def 添加表格区(self):
        frametable = TableFrame(self.面板, width=500, height=600)
        frametable.pack_propagate(0)
        frametable.pack(side="left", fill='both')
        self.面板.add(frametable)
        缓存.frame表格 = frametable

    def 添加文本区(self):
        frametext = ScrolledtextFrame(self.面板, width=100, height=600)
        frametext.pack_propagate(0)
        frametext.pack(side="left", fill='y')
        self.面板.add(frametext)
        缓存.frame文本 = frametext