try:
    from . import tkfunction as ui
    from .tkfunction import contextitem, contexttext, PopPack
    from .tkexcanvas import CanvasCAD
    from .tkextreeprocess import TreeProcess
except:
    import tkfunction as ui
    from tkfunction import contextitem, contexttext, PopPack
    from tkexcanvas import CanvasCAD
    from tkextreeprocess import TreeProcess

class CanvasCADPack(contextitem):
    def __init__(self, **kwargs):
        side, fill, expand = PopPack(kwargs)
        self.create_item = CanvasCAD(ui.record_item)
        for key, value in kwargs.items(): self.create_item[key] = value
        self.create_item.pack(side=side, fill=fill, expand=expand)


class TreeProcessPack():
    def __init__(self):
        self.create_item = TreeProcess()
