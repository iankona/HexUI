

bp = ""

class 类:
    def __enter__(self):
        global bp
        self.buff = bp
        bp = self.bp
        return self.bp

    def __exit__(self, type, value, traceback):
        global bp
        bp = self.buff

