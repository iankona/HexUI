from . import context_bpbytes
from . import bpbytes


class filepath(context_bpbytes.类):
    def __init__(self, filepath):
        self.bp = bpbytes.类().filepath(filepath)
        self.uint8 = self.bp.readuint8seek0(16)

def close(): return context_bpbytes.data.close()
def left(): return context_bpbytes.data.left()
def right(): return context_bpbytes.data.right()
def tell(): return context_bpbytes.data.tell()
def slicetell(): return context_bpbytes.data.slicetell()
def sliceEOF(): return context_bpbytes.data.sliceEOF()
def size(): return context_bpbytes.data.size()
def remainsize(): return context_bpbytes.data.remainsize()
def seek(size): return context_bpbytes.data.seek(size)


class copy(context_bpbytes.类):
    def __init__(self):
        self.bp = context_bpbytes.data.copy()
        self.uint8 = self.bp.readuint8seek0(16)
class readslice(context_bpbytes.类):
    def __init__(self, num=1):
        self.bp = context_bpbytes.data.readslice(num)
        self.uint8 = self.bp.readuint8seek0(16)
class readremainslice(context_bpbytes.类):
    def __init__(self):
        self.bp = context_bpbytes.data.readremainslice()
        self.uint8 = self.bp.readuint8seek0(16)


def read(num=1): return context_bpbytes.data.read(num)
def remainread(num=1): return context_bpbytes.data.remainread(num)

def readuint8(num=1): return context_bpbytes.data.readuint8(num)
def readuint16(num=1): return context_bpbytes.data.readuint16(num)
def readuint32(num=1): return context_bpbytes.data.readuint32(num)
def readuint64(num=1): return context_bpbytes.data.readuint64(num)

def readint8(num=1): return context_bpbytes.data.readint8(num)
def readint16(num=1): return context_bpbytes.data.readint16(num)
def readint32(num=1): return context_bpbytes.data.readint32(num)
def readint64(num=1): return context_bpbytes.data.readint64(num)

def readfloat16(num=1): return context_bpbytes.data.readfloat16(num)
def readfloat32(num=1): return context_bpbytes.data.readfloat16(num)
def readfloat64(num=1): return context_bpbytes.data.readfloat16(num)

def readu8float32(num=1): return context_bpbytes.data.readu8float32(num)
def readi8float32(num=1): return context_bpbytes.data.readi8float32(num)
def readu16float32(num=1): return context_bpbytes.data.readu16float32(num)
def readi16float32(num=1): return context_bpbytes.data.readi16float32(num)

def read5u8uint64(): return context_bpbytes.data.read5u8uint64()
def readhex(num=1): return context_bpbytes.data.readhex(num)
def readchar(num=1): return context_bpbytes.data.readchar(num)
def readgbk(size): return context_bpbytes.data.readgbk(size)
def readutf8(size): return context_bpbytes.data.readutf8(size)

class readsliceseek0(context_bpbytes.类):
    def __init__(self, num=1):
        self.bp = context_bpbytes.data.readsliceseek0(num)
        self.uint8 = self.bp.readuint8seek0(16)
class readremainsliceseek0(context_bpbytes.类):
    def __init__(self):
        self.bp = context_bpbytes.data.readremainsliceseek0()
        self.uint8 = self.bp.readuint8seek0(16)

def readseek0(num=1): return context_bpbytes.data.readseek0(num)
def remainreadseek0(num=1): return context_bpbytes.data.remainreadseek0(num)

def readuint8seek0(num=1): return context_bpbytes.data.readuint8seek0(num)
def readuint16seek0(num=1): return context_bpbytes.data.readuint16seek0(num)
def readuint32seek0(num=1): return context_bpbytes.data.readuint32seek0(num)
def readuint64seek0(num=1): return context_bpbytes.data.readuint64seek0(num)

def readint8seek0(num=1): return context_bpbytes.data.readint8seek0(num)
def readint16seek0(num=1): return context_bpbytes.data.readint16seek0(num)
def readint32seek0(num=1): return context_bpbytes.data.readint32seek0(num)
def readint64seek0(num=1): return context_bpbytes.data.readint64seek0(num)

def readfloat16seek0(num=1): return context_bpbytes.data.readfloat16seek0(num)
def readfloat32seek0(num=1): return context_bpbytes.data.readfloat16seek0(num)
def readfloat64seek0(num=1): return context_bpbytes.data.readfloat16seek0(num)

def readu8float32seek0(num=1): return context_bpbytes.data.readu8float32seek0(num)
def readi8float32seek0(num=1): return context_bpbytes.data.readi8float32seek0(num)
def readu16float32seek0(num=1): return context_bpbytes.data.readu16float32seek0(num)
def readi16float32seek0(num=1): return context_bpbytes.data.readi16float32seek0(num)

def read5u8uint64seek0(): return context_bpbytes.data.read5u8uint64seek0()
def readhexseek0(num=1): return context_bpbytes.data.readhexseek0(num)
def readcharseek0(num=1): return context_bpbytes.data.readcharseek0(num)
def readgbkseek0(size): return context_bpbytes.data.readgbkseek0(size)
def readutf8seek0(size): return context_bpbytes.data.readutf8seek0(size)


