import os
import mmap


from . import stacke
from . import struct_bnumpy

class 类(stacke.类):
    def __init__(self, mpbyte=None, endian="<"):
        stacke.类.__init__(self, mpbyte=mpbyte, endian=endian, module=struct_bnumpy)


    def __assert__number__invert__(self, num, step):
        left, right = self.index - num * step, self.index
        if left < self.mpleft: left = self.mpleft
        count = (right - left) // step
        left = self.index - count * step
        return left, right, count


    def __calc__size__invert__(self, size): 
        left, right, count = self.__assert__number__invert__(size, 1)
        self.index = left
        return left, right

    def __calc__size__invert__no__seek__(self, size):
        left, right, count = self.__assert__number__invert__(size, 1)
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






    def readsliceinvert(self, size):
        left, right = self.__calc__size__invert__(size)
        bp = 类(mpbyte=self.mpbyte, endian=self.endian)
        bp.script = self.script
        bp.mpleft, bp.mpright = left, right
        bp.index = bp.mpleft
        bp.rdpath, bp.rdfile = self.rdpath, self.rdfile
        return bp


    def readsliceinvertseek0(self, size):
        left, right = self.__calc__size__invert__no__seek__(size)
        bp = 类(mpbyte=self.mpbyte, endian=self.endian)
        bp.script = self.script
        bp.mpleft, bp.mpright = left, right
        bp.index = bp.mpleft
        bp.rdpath, bp.rdfile = self.rdpath, self.rdfile
        return bp



    def remainsizeinvert(self):
        return self.mpleft



    def readremainsliceinvert(self):
        left, right = self.__calc__size__invert__(self.remainsizeinvert())
        bp = 类(mpbyte=self.mpbyte, endian=self.endian)
        bp.script = self.script
        bp.mpleft, bp.mpright = left, right
        bp.index = bp.mpleft
        bp.rdpath, bp.rdfile = self.rdpath, self.rdfile
        return bp


    def readremainsliceinvertseek0(self):
        left, right = self.__calc__size__invert__no__seek__(self.remainsizeinvert())
        bp = 类(mpbyte=self.mpbyte, endian=self.endian)
        bp.script = self.script
        bp.mpleft, bp.mpright = left, right
        bp.index = bp.mpleft
        bp.rdpath, bp.rdfile = self.rdpath, self.rdfile
        return bp



