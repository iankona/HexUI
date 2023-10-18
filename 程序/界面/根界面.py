import tkinter as tk
import tkinter.ttk as ttk

from . import 缓存
from .面板Frame import 面板Frame

def 运行():
    缓存.根界面 = app = tk.Tk()
    面板frame0 = 面板Frame()
    面板frame0.pack(fill='both', expand=1)
    app.mainloop()