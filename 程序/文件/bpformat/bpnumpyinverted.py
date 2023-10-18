import os
import mmap


from . import underlying
from . import bpnumpy





class 类(underlying.类):
    def __init__(self, mpbyte=None, endian="<"):
        underlying.类.__init__(self, mpbyte=mpbyte, endian=endian)


    def __calc_size__(self, size):
        left, right = self.index-size, self.index
        if left < self.mpleft: left = self.mpleft
        self.index = left
        return left, right


    def __calc_size_no_seek__(self, size):
        left, right = self.index-size, self.index
        if left < self.mpleft: left = self.mpleft
        return left, right



    def filepath(self, filepath):
        self.rdpath = filepath
        self.rdfile = open(filepath, "rb")
        self.mpbyte = mmap.mmap(self.rdfile.fileno(), 0, access=mmap.ACCESS_READ)
        self.mpleft, self.index, self.mpright = 0, os.path.getsize(filepath), os.path.getsize(filepath)
        return self


    def close(self):
        self.mpbyte.close()
        self.rdfile.close()


    def copy(self):
        bp = 类(mpbyte=self.mpbyte, endian=self.endian)
        bp.mpleft, bp.mpright = self.mpleft, self.mpright
        bp.index = bp.mpleft
        bp.rdpath, bp.rdfile = self.rdpath, self.rdfile
        return bp
    

    def readslice(self, size):
        left, right = self.__calc_size__(size)
        bp = 类(mpbyte=self.mpbyte, endian=self.endian)
        bp.mpleft, bp.mpright = left, right
        bp.index = bp.mpleft
        bp.rdpath, bp.rdfile = self.rdpath, self.rdfile
        return bp


    def readsliceseek0(self, size):
        left, right = self.__calc_size_no_seek__(size)
        bp = 类(mpbyte=self.mpbyte, endian=self.endian)
        bp.mpleft, bp.mpright = left, right
        bp.index = bp.mpleft
        bp.rdpath, bp.rdfile = self.rdpath, self.rdfile
        return bp



    def tobpnumpy(self):
        bp = bpnumpy.类(mpbyte=self.mpbyte, endian=self.endian)
        bp.mpleft, bp.mpright = self.mpleft, self.mpright
        bp.index = bp.mpleft
        bp.rdpath, bp.rdfile = self.rdpath, self.rdfile
        return bp


