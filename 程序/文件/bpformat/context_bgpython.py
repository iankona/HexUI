

node = ""

class 类:
    def __enter__(self):
        global node
        self.item = node
        node = self.node
        self.node.parent = self.item
        return self.node

    def __exit__(self, type, value, traceback):
        global node
        node = self.item

