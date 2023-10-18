

data = ""

class 类:
    def __enter__(self):
        global data
        self.item = data
        data = self.bp
        return self.bp

    def __exit__(self, type, value, traceback):
        global data
        data = self.item

