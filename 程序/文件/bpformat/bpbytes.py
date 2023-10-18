from . import underlying

class 类(underlying.类):
    def __init__(self, mpbyte=None, endian="<"):
        underlying.类.__init__(self, mpbyte=mpbyte, endian=endian)


    def frombyte(self, stream:bytes):
        self.mpbyte = stream
        self.mpleft, self.index, self.mpright = 0, 0, len(stream)
        return self
    
