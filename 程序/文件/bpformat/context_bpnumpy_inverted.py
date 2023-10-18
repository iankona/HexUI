

data = ""

class 类:
    def __enter__(self):
        global data
        self.item = data
        data = self.bped
        return self.bped

    def __exit__(self, type, value, traceback):
        global data
        data = self.item

