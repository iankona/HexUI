from . import context_bpnumpy
from . import bpnumpy


def context(bp): context_bpnumpy.data = bp

def rdpath(): return context_bpnumpy.data.rdpath

class filepath(context_bpnumpy.类):
    def __init__(self, filepath): self.bp = bpnumpy.类().filepath(filepath)

def close(): return context_bpnumpy.data.close()
def left(): return context_bpnumpy.data.left()
def right(): return context_bpnumpy.data.right()
def tell(): return context_bpnumpy.data.tell()
def slicetell(): return context_bpnumpy.data.slicetell()
def sliceEOF(): return context_bpnumpy.data.sliceEOF()
def size(): return context_bpnumpy.data.size()
def remainsize(): return context_bpnumpy.data.remainsize()
def seek(size): return context_bpnumpy.data.seek(size)

class copy(context_bpnumpy.类):
    def __init__(self): self.bp = context_bpnumpy.data.copy()
class readslice(context_bpnumpy.类):
    def __init__(self, num=1): self.bp = context_bpnumpy.data.readslice(num)
class readremainslice(context_bpnumpy.类):
    def __init__(self): self.bp = context_bpnumpy.data.readremainslice()

def read(num=1): return context_bpnumpy.data.read(num)
def remainread(num=1): return context_bpnumpy.data.remainread(num)

def readuint8(num=1): return context_bpnumpy.data.readuint8(num)
def readuint16(num=1): return context_bpnumpy.data.readuint16(num)
def readuint32(num=1): return context_bpnumpy.data.readuint32(num)
def readuint64(num=1): return context_bpnumpy.data.readuint64(num)

def readint8(num=1): return context_bpnumpy.data.readint8(num)
def readint16(num=1): return context_bpnumpy.data.readint16(num)
def readint32(num=1): return context_bpnumpy.data.readint32(num)
def readint64(num=1): return context_bpnumpy.data.readint64(num)

def readfloat16(num=1): return context_bpnumpy.data.readfloat16(num)
def readfloat32(num=1): return context_bpnumpy.data.readfloat16(num)
def readfloat64(num=1): return context_bpnumpy.data.readfloat16(num)

def readu8float32(num=1): return context_bpnumpy.data.readu8float32(num)
def readi8float32(num=1): return context_bpnumpy.data.readi8float32(num)
def readu16float32(num=1): return context_bpnumpy.data.readu16float32(num)
def readi16float32(num=1): return context_bpnumpy.data.readi16float32(num)

def read5u8uint64(): return context_bpnumpy.data.read5u8uint64()
def readhex(num=1): return context_bpnumpy.data.readhex(num)
def readchar(num=1): return context_bpnumpy.data.readchar(num)
def readgbk(size): return context_bpnumpy.data.readgbk(size)
def readutf8(size): return context_bpnumpy.data.readutf8(size)

class readsliceseek0(context_bpnumpy.类):
    def __init__(self, num=1): self.bp = context_bpnumpy.data.readsliceseek0(num)

class readremainsliceseek0(context_bpnumpy.类):
    def __init__(self): self.bp = context_bpnumpy.data.readremainsliceseek0()

def readseek0(num=1): return context_bpnumpy.data.readseek0(num)
def remainreadseek0(num=1): return context_bpnumpy.data.remainreadseek0(num)

def readuint8seek0(num=1): return context_bpnumpy.data.readuint8seek0(num)
def readuint16seek0(num=1): return context_bpnumpy.data.readuint16seek0(num)
def readuint32seek0(num=1): return context_bpnumpy.data.readuint32seek0(num)
def readuint64seek0(num=1): return context_bpnumpy.data.readuint64seek0(num)

def readint8seek0(num=1): return context_bpnumpy.data.readint8seek0(num)
def readint16seek0(num=1): return context_bpnumpy.data.readint16seek0(num)
def readint32seek0(num=1): return context_bpnumpy.data.readint32seek0(num)
def readint64seek0(num=1): return context_bpnumpy.data.readint64seek0(num)

def readfloat16seek0(num=1): return context_bpnumpy.data.readfloat16seek0(num)
def readfloat32seek0(num=1): return context_bpnumpy.data.readfloat16seek0(num)
def readfloat64seek0(num=1): return context_bpnumpy.data.readfloat16seek0(num)

def readu8float32seek0(num=1): return context_bpnumpy.data.readu8float32seek0(num)
def readi8float32seek0(num=1): return context_bpnumpy.data.readi8float32seek0(num)
def readu16float32seek0(num=1): return context_bpnumpy.data.readu16float32seek0(num)
def readi16float32seek0(num=1): return context_bpnumpy.data.readi16float32seek0(num)

def read5u8uint64seek0(): return context_bpnumpy.data.read5u8uint64seek0()
def readhexseek0(num=1): return context_bpnumpy.data.readhexseek0(num)
def readcharseek0(num=1): return context_bpnumpy.data.readcharseek0(num)
def readgbkseek0(size): return context_bpnumpy.data.readgbkseek0(size)
def readutf8seek0(size): return context_bpnumpy.data.readutf8seek0(size)

def tobpbytes(stream): return context_bpnumpy.data.tobpbytes(stream)


