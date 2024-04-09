import os
import mmap

from . import stacke
from . import struct_python

class 类(stacke.类):
    def __init__(self, mpbyte=None, endian="<"):
        stacke.类.__init__(self, mpbyte=mpbyte, endian=endian, module=struct_python)


    def filepath(self, filepath):
        self.rdpath = filepath
        self.rdfile = open(filepath, "rb")
        self.mpbyte = mmap.mmap(self.rdfile.fileno(), 0, access=mmap.ACCESS_READ) # mpfile
        self.mpleft, self.index, self.mpright = 0, 0, os.path.getsize(filepath)
        return self

    def close(self):
        self.mpbyte.close()
        self.rdfile.close()


    def copy(self):
        bp = 类(mpbyte=self.mpbyte, endian=self.endian)
        bp.script = self.script
        bp.mpleft, bp.mpright = self.mpleft, self.mpright
        bp.index = bp.mpleft
        bp.rdpath, bp.rdfile = self.rdpath, self.rdfile
        return bp
    

    def readslice(self, size):
        left, right = self.__calc__size__(size)
        bp = 类(mpbyte=self.mpbyte, endian=self.endian)
        bp.script = self.script
        bp.mpleft, bp.mpright = left, right
        bp.index = bp.mpleft
        bp.rdpath, bp.rdfile = self.rdpath, self.rdfile
        return bp


    def readsliceseek0(self, size):
        left, right = self.__calc__size__no__seek__(size)
        bp = 类(mpbyte=self.mpbyte, endian=self.endian)
        bp.script = self.script
        bp.mpleft, bp.mpright = left, right
        bp.index = bp.mpleft
        bp.rdpath, bp.rdfile = self.rdpath, self.rdfile
        return bp


    def readremainslice(self):
        left, right = self.__calc__size__(self.remainsize())
        bp = 类(mpbyte=self.mpbyte, endian=self.endian)
        bp.script = self.script
        bp.mpleft, bp.mpright = left, right
        bp.index = bp.mpleft
        bp.rdpath, bp.rdfile = self.rdpath, self.rdfile
        return bp


    def readremainsliceseek0(self):
        left, right = self.__calc__size__no__seek__(self.remainsize())
        bp = 类(mpbyte=self.mpbyte, endian=self.endian)
        bp.script = self.script
        bp.mpleft, bp.mpright = left, right
        bp.index = bp.mpleft
        bp.rdpath, bp.rdfile = self.rdpath, self.rdfile
        return bp
