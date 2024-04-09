from . import fcontext

from . import bpbytes
from . import bpfile
from . import bpnumpy
from . import bpstruct
from . import bpinvert


def bp(): return fcontext.bp

def endian(endian=">"): 
    if endian not in [">", "<"]: raise ValueError(f"endian must be in ['>', '<'], not {endian}")
    fcontext.bp.endian = endian


实例类 = None
def context(bp):
    fcontext.bp = bp
    global 实例类
    classname = str(type(bp))
    if "bpbytes.类" in classname: 实例类 = bpbytes.类
    if "bpfile.类" in classname: 实例类 = bpfile.类
    if "bpnumpy.类" in classname: 实例类 = bpnumpy.类
    if "bpstruct.类" in classname: 实例类 = bpstruct.类
    if "bpinvert.类" in classname: 实例类 = bpinvert.类


class filepath(fcontext.类):
    def __init__(self, filepath): self.bp = 实例类().filepath(filepath)




def close(): return fcontext.bp.close()
def left(): return fcontext.bp.left()
def right(): return fcontext.bp.right()
def tell(): return fcontext.bp.tell()
def slicetell(): return fcontext.bp.slicetell()
def sliceEOF(): return fcontext.bp.sliceEOF()
def size(): return fcontext.bp.size()
def remainsize(): return fcontext.bp.remainsize()
def seek(size): return fcontext.bp.seek(size)
def movetell(index): fcontext.bp.movetell(index)

class copy(fcontext.类):
    def __init__(self): self.bp = fcontext.bp.copy()
class readslice(fcontext.类):
    def __init__(self, size=0): self.bp = fcontext.bp.readslice(size)
class readremainslice(fcontext.类):
    def __init__(self): self.bp = fcontext.bp.readremainslice()
class fromstream(fcontext.类):
    def __init__(self, mpbyte): self.bp = fcontext.bp.fromstream(mpbyte)

def readuint8(num=1): return fcontext.bp.readuint8(num)
def readuint16(num=1): return fcontext.bp.readuint16(num)
def readuint32(num=1): return fcontext.bp.readuint32(num)
def readuint64(num=1): return fcontext.bp.readuint64(num)

def readint8(num=1): return fcontext.bp.readint8(num)
def readint16(num=1): return fcontext.bp.readint16(num)
def readint32(num=1): return fcontext.bp.readint32(num)
def readint64(num=1): return fcontext.bp.readint64(num)

def readfloat16(num=1): return fcontext.bp.readfloat16(num)
def readfloat32(num=1): return fcontext.bp.readfloat32(num)
def readfloat64(num=1): return fcontext.bp.readfloat64(num)

def readu8float32(num=1): return fcontext.bp.readu8float32(num)
def readi8float32(num=1): return fcontext.bp.readi8float32(num)
def readu16float32(num=1): return fcontext.bp.readu16float32(num)
def readi16float32(num=1): return fcontext.bp.readi16float32(num)


def read(size=0): return fcontext.bp.read(size)
def readremain(): return fcontext.bp.readremain()

def readbin(size=0): return fcontext.bp.readbin(size)
def readbin8(size=0): return fcontext.bp.readbin8(size)
def readhex(size=0): return fcontext.bp.readhex(size)
def readchar(size=0): return fcontext.bp.readchar(size)

def readcharend0(): return fcontext.bp.readcharend0()

def readbinqueue(size=0): return fcontext.bp.readbinqueue(size)
def readbin8queue(size=0): return fcontext.bp.readbin8queue(size)
def readhexqueue(size=0): return fcontext.bp.readhexqueue(size)
def readcharqueue(size=0): return fcontext.bp.readcharqueue(size)

def readgbk(size=0): return fcontext.bp.readgbk(size)
def readutf8(size=0): return fcontext.bp.readutf8(size)
def readascii(size=0): return fcontext.bp.readascii(size)



class readsliceseek0(fcontext.类):
    def __init__(self, size=0): self.bp = fcontext.bp.readsliceseek0(size)
class readremainsliceseek0(fcontext.类):
    def __init__(self): self.bp = fcontext.bp.readremainsliceseek0()

def readuint8seek0(num=1): return fcontext.bp.readuint8seek0(num)
def readuint16seek0(num=1): return fcontext.bp.readuint16seek0(num)
def readuint32seek0(num=1): return fcontext.bp.readuint32seek0(num)
def readuint64seek0(num=1): return fcontext.bp.readuint64seek0(num)

def readint8seek0(num=1): return fcontext.bp.readint8seek0(num)
def readint16seek0(num=1): return fcontext.bp.readint16seek0(num)
def readint32seek0(num=1): return fcontext.bp.readint32seek0(num)
def readint64seek0(num=1): return fcontext.bp.readint64seek0(num)

def readfloat16seek0(num=1): return fcontext.bp.readfloat16seek0(num)
def readfloat32seek0(num=1): return fcontext.bp.readfloat32seek0(num)
def readfloat64seek0(num=1): return fcontext.bp.readfloat64seek0(num)

def readu8float32seek0(num=1): return fcontext.bp.readu8float32seek0(num)
def readi8float32seek0(num=1): return fcontext.bp.readi8float32seek0(num)
def readu16float32seek0(num=1): return fcontext.bp.readu16float32seek0(num)
def readi16float32seek0(num=1): return fcontext.bp.readi16float32seek0(num)



def readseek0(size=0): return fcontext.bp.readseek0(size)
def readremainseek0(): return fcontext.bp.readremainseek0()

def readbinseek0(size=0): return fcontext.bp.readbinseek0(size)
def readbin8seek0(size=0): return fcontext.bp.readbin8seek0(size)
def readhexseek0(size=0): return fcontext.bp.readhexseek0(size)
def readcharseek0(size=0): return fcontext.bp.readcharseek0(size)

def readbinqueueseek0(size=0): return fcontext.bp.readbinqueueseek0(size)
def readbin8queueseek0(size=0): return fcontext.bp.readbin8queueseek0(size)
def readhexqueueseek0(size=0): return fcontext.bp.readhexqueueseek0(size)
def readcharqueueseek0(size=0): return fcontext.bp.readcharqueueseek0(size)

def readcharend0seek0(): return fcontext.bp.readcharend0seek0()
def readcharend0sizeseek0(): return fcontext.bp.readcharend0sizeseek0()

def readgbkseek0(size=0): return fcontext.bp.readgbkseek0(size)
def readutf8seek0(size=0): return fcontext.bp.readutf8seek0(size)
def readasciiseek0(size=0): return fcontext.bp.readasciiseek0(size)

