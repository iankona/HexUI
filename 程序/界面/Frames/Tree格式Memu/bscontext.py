
frametree = ""
item = ""

class 类:
    def __enter__(self):
        global item
        self.fitem = item
        item = self.citem
        return self.citem

    def __exit__(self, type, value, traceback):
        global item
        item = self.fitem

