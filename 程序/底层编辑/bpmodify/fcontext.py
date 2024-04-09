

bg = None

class 类:
    def __enter__(self):
        global bg
        self.buff = bg
        bg = self.bg
        return self.bg

    def __exit__(self, type, value, traceback):
        global bg
        bg = self.buff

