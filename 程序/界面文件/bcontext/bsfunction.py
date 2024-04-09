from . import bscontext


def context(item="", frametreeview=None):
    bscontext.item = item
    bscontext.frametreeview = frametreeview


class insertblock(bscontext.类):
    def __init__(self, item="", text="", bp=None):
        if item == "": item = bscontext.item
        self.citem = bscontext.frametreeview.insertblock(fitem=item, text=text, bp=bp)


class insertbytes(bscontext.类):
    def __init__(self, item="", text="", bp=None):
        if item == "": item = bscontext.item
        self.citem = bscontext.frametreeview.insertbytes(fitem=item, text=text, bp=bp)


class insertvalue(bscontext.类):
    def __init__(self, item="", text="", values=[]):
        if item == "": item = bscontext.item
        self.citem = bscontext.frametreeview.insertvalue(fitem=item, text=text, values=values)


