from . import bscontext

class insertblock(bscontext.类):
    def __init__(self, item="", text="", bp=None):
        if item == "": item = bscontext.item
        self.citem = bscontext.frametree.insertblock(fitem=item, index="end", text=text, bp=bp)


class insertbytes(bscontext.类):
    def __init__(self, item="", text="", bp=None):
        if item == "": item = bscontext.item
        self.citem = bscontext.frametree.insertbytes(fitem=item, index="end", text=text, bp=bp)


class insertvalue(bscontext.类):
    def __init__(self, item="", text="", values=[]):
        if item == "": item = bscontext.item
        self.citem = bscontext.frametree.insertvalue(fitem=item, index="end", text=text, values=values)


