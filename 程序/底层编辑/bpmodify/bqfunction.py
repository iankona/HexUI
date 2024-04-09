from . import fcontext
from . import bgmultipletree

def bg(): return fcontext.bg

实例类 = None
def context(bg): 
    fcontext.bg = bg
    global 实例类
    实例类 = bgmultipletree.类

def rdpath(): return fcontext.bg.rdpath

class filepath(fcontext.类):
    def __init__(self, filepath): self.bg = 实例类().filepath(filepath)

def close(): return fcontext.bg.close()
def left(): return fcontext.bg.left()
def right(): return fcontext.bg.right()
def tell(): return fcontext.bg.tell()
def slicetell(): return fcontext.bg.slicetell()
def sliceEOF(): return fcontext.bg.sliceEOF()
def size(): return fcontext.bg.size()
def remainsize(): return fcontext.bg.remainsize()
def seek(size): return fcontext.bg.seek(size)

def readuint8(num=1): return fcontext.bg.readuint8(num)
def readuint16(num=1): return fcontext.bg.readuint16(num)
def readuint32(num=1): return fcontext.bg.readuint32(num)
def readuint64(num=1): return fcontext.bg.readuint64(num)

def readint8(num=1): return fcontext.bg.readint8(num)
def readint16(num=1): return fcontext.bg.readint16(num)
def readint32(num=1): return fcontext.bg.readint32(num)
def readint64(num=1): return fcontext.bg.readint64(num)

def readfloat16(num=1): return fcontext.bg.readfloat16(num)
def readfloat32(num=1): return fcontext.bg.readfloat16(num)
def readfloat64(num=1): return fcontext.bg.readfloat16(num)

def read(size): return fcontext.bg.read(size)
def readremain(): return fcontext.bg.readremain()

def readhex(size): return fcontext.bg.readhex(size)
def readgbk(size): return fcontext.bg.readgbk(size)
def readutf8(size): return fcontext.bg.readutf8(size)
def readchar(size): return fcontext.bg.readchar(size)
def readascii(size): return fcontext.bg.readascii(size)

class readsliceseek0(fcontext.类):
    def __init__(self, num=1): self.bg = fcontext.bg.readsliceseek0(num)

class readremainsliceseek0(fcontext.类):
    def __init__(self): self.bg = fcontext.bg.readremainsliceseek0()

def readseek0(size): return fcontext.bg.readseek0(size)
def readremainseek0(): return fcontext.bg.readremainseek0()

def readuint8seek0(num=1): return fcontext.bg.readuint8seek0(num)
def readuint16seek0(num=1): return fcontext.bg.readuint16seek0(num)
def readuint32seek0(num=1): return fcontext.bg.readuint32seek0(num)
def readuint64seek0(num=1): return fcontext.bg.readuint64seek0(num)

def readint8seek0(num=1): return fcontext.bg.readint8seek0(num)
def readint16seek0(num=1): return fcontext.bg.readint16seek0(num)
def readint32seek0(num=1): return fcontext.bg.readint32seek0(num)
def readint64seek0(num=1): return fcontext.bg.readint64seek0(num)

def readfloat16seek0(num=1): return fcontext.bg.readfloat16seek0(num)
def readfloat32seek0(num=1): return fcontext.bg.readfloat16seek0(num)
def readfloat64seek0(num=1): return fcontext.bg.readfloat16seek0(num)

def readhexseek0(size): return fcontext.bg.readhexseek0(size)
def readgbkseek0(size): return fcontext.bg.readgbkseek0(size)
def readutf8seek0(size): return fcontext.bg.readutf8seek0(size)
def readcharseek0(size): return fcontext.bg.readcharseek0(size)
def readasciiseek0(size): return fcontext.bg.readasciiseek0(size)


class copy(fcontext.类):
    def __init__(self): self.bg = fcontext.bg.copy()
class bnodeslice(fcontext.类):
    def __init__(self, size, name=""): self.bg = fcontext.bg.bnodeslice(size, name)
class bnoderemainslice(fcontext.类):
    def __init__(self, name=""): self.bg = fcontext.bg.bnoderemainslice(name)

def bnodeint8(num=1, name=""): return fcontext.bg.bnodeint8(num, name)
def bnodeint16(num=1, name=""): return fcontext.bg.bnodeint16(num, name)
def bnodeint32(num=1, name=""): return fcontext.bg.bnodeint32(num, name)
def bnodeint64(num=1, name=""): return fcontext.bg.bnodeint64(num, name)

def bnodeuint8(num=1, name=""): return fcontext.bg.bnodeuint8(num, name)
def bnodeuint16(num=1, name=""): return fcontext.bg.bnodeuint16(num, name)
def bnodeuint32(num=1, name=""): return fcontext.bg.bnodeuint32(num, name)
def bnodeuint64(num=1, name=""): return fcontext.bg.bnodeuint64(num, name)

def bnodefloat16(num=1, name=""): return fcontext.bg.bnodefloat16(num, name)
def bnodefloat32(num=1, name=""): return fcontext.bg.bnodefloat32(num, name)
def bnodefloat64(num=1, name=""): return fcontext.bg.bnodefloat64(num, name)


def bnodegbk(size, name=""): return fcontext.bg.bnodegbk(size, name)
def bnodeutf8(size, name=""): return fcontext.bg.bnodeutf8(size, name)
def bnodeascii(size, name=""): return fcontext.bg.bnodeascii(size, name)
def bnodestring(size, name="", inttype="uint32", strtype="utf8"): return fcontext.bg.bnodestring(size, name, inttype, strtype)


