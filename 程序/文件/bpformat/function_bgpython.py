from . import context_bgpython
from . import bgpython


def node(): return context_bgpython.node

class copy(context_bgpython.类):
    def __init__(self):
        self.node = context_bgpython.node.copy()


class addnode(context_bgpython.类):
    def __init__(self, varname, varnode):
        self.node = varnode
        context_bgpython.node.addnode(varname, varnode)


class modifynode(context_bgpython.类):
    def __init__(self, iname, varnode):
        self.node = varnode
        context_bgpython.node.modifynode(iname, varnode)


class modifynamenode(context_bgpython.类):
    def __init__(self, iname, varname, varnode):
        self.node = varnode
        context_bgpython.node.modifynamenode(iname, varname, varnode)


class findnode(context_bgpython.类):
    def __init__(self, iname):
        name, node = context_bgpython.node.find(iname)
        self.node = node


def size(): return context_bgpython.node.size()
def find(iname): return context_bgpython.node.find(iname)
def index(varvalue): return context_bgpython.node.index(varvalue)
def emptynode(): return context_bgpython.node.emptynode()

def adduint8(varname, varvalue): return context_bgpython.node.adduint8(varname, varvalue)
def adduint16(varname, varvalue): return context_bgpython.node.adduint16(varname, varvalue)
def adduint32(varname, varvalue): return context_bgpython.node.adduint32(varname, varvalue)
def adduint64(varname, varvalue): return context_bgpython.node.adduint64(varname, varvalue)

def addint8(varname, varvalue): return context_bgpython.node.addint8(varname, varvalue)
def addint16(varname, varvalue): return context_bgpython.node.addint16(varname, varvalue)
def addint32(varname, varvalue): return context_bgpython.node.addint32(varname, varvalue)
def addint64(varname, varvalue): return context_bgpython.node.addint64(varname, varvalue)

# def addfloat16(varname, varfloat): return bgcontext.node.addfloat16(varname, varfloat)
# def addfloat32(varname, varfloat): return bgcontext.node.addfloat16(varname, varfloat)
# def addfloat64(varname, varfloat): return bgcontext.node.addfloat16(varname, varfloat)

# def addu8float32(varname, varfloat): return bgcontext.node.addu8float32(varname, varfloat)
# def addi8float32(varname, varfloat): return bgcontext.node.addi8float32(varname, varfloat)
# def addu16float32(varname, varfloat): return bgcontext.node.addu16float32(varname, varfloat)
# def addi16float32(varname, varfloat): return bgcontext.node.addi16float32(varname, varfloat)

def addslice0b(varname, varbyte): return context_bgpython.node.add0b(varname, varbyte)

def addchar(varname, varchars): return context_bgpython.node.addchar(varname, varchars)
def addgbk(varname, varchars): return context_bgpython.node.addgbk(varname, varchars)
def addutf8(varname, varchars): return context_bgpython.node.addutf8(varname, varchars)

def addnumu8char(varname, varchars): return context_bgpython.node.addnumu8char(varname, varchars)
def addnumu16char(varname, varchars): return context_bgpython.node.addnumu16char(varname, varchars)
def addnumu32char(varname, varchars): return context_bgpython.node.addnumu32char(varname, varchars)
def addnumu32gbk(varname, varchars): return context_bgpython.node.addnumu32gbk(varname, varchars)
def addnumu32utf8(varname, varchars): return context_bgpython.node.addnumu32utf8(varname, varchars)

def updateslice0b(): return context_bgpython.node.updateslice0b()
def updatenodetree0b(): return context_bgpython.node.updatenodetree0b()


# 修改变量值
def modifyuint8(iname, varvalue): return context_bgpython.node.modifyuint8(iname, varvalue)
def modifyuint16(iname, varvalue): return context_bgpython.node.modifyuint16(iname, varvalue)
def modifyuint32(iname, varvalue): return context_bgpython.node.modifyuint32(iname, varvalue)
def modifyuint64(iname, varvalue): return context_bgpython.node.modifyuint64(iname, varvalue)

def modifyint8(iname, varvalue): return context_bgpython.node.modifyint8(iname, varvalue)
def modifyint16(iname, varvalue): return context_bgpython.node.modifyint16(iname, varvalue)
def modifyint32(iname, varvalue): return context_bgpython.node.modifyint32(iname, varvalue)
def modifyint64(iname, varvalue): return context_bgpython.node.modifyint64(iname, varvalue)

def modifyslice0b(iname, varbyte): return context_bgpython.node.modifyslice0b(iname, varbyte)


def modifychar(iname, varchars): return context_bgpython.node.modifychar(iname, varchars)
def modifygbk(iname, varchars): return context_bgpython.node.modifygbk(iname, varchars)
def modifyutf8(iname, varchars): return context_bgpython.node.modifyutf8(iname, varchars)

def modifynumu8char(iname, varchars): return context_bgpython.node.modifynumu8char(iname, varchars)
def modifynumu16char(iname, varchars): return context_bgpython.node.modifynumu16char(iname, varchars)
def modifynumu32char(iname, varchars): return context_bgpython.node.modifynumu32char(iname, varchars)
def modifynumu32gbk(iname, varchars): return context_bgpython.node.modifynumu32gbk(iname, varchars)
def modifynumu32utf8(iname, varchars): return context_bgpython.node.modifynumu32utf8(iname, varchars)


# 修改变量名称，变量值
def modifynameuint8(iname, varname, varvalue): return context_bgpython.node.modifynameuint8(iname, varname, varvalue)
def modifynameuint16(iname, varname, varvalue): return context_bgpython.node.modifynameuint16(iname, varname, varvalue)
def modifynameuint32(iname, varname, varvalue): return context_bgpython.node.modifynameuint32(iname, varname, varvalue)
def modifynameuint64(iname, varname, varvalue): return context_bgpython.node.modifynameuint64(iname, varname, varvalue)

def modifynameint8(iname, varname, varvalue): return context_bgpython.node.modifynameint8(iname, varname, varvalue)
def modifynameint16(iname, varname, varvalue): return context_bgpython.node.modifynameint16(iname, varname, varvalue)
def modifynameint32(iname, varname, varvalue): return context_bgpython.node.modifynameint32(iname, varname, varvalue)
def modifynameint64(iname, varname, varvalue): return context_bgpython.node.modifynameint64(iname, varname, varvalue)

def modifynameslice0b(iname, varname, varbyte): return context_bgpython.node.modifynameslice0b(iname, varname, varbyte)

def modifynamechar(iname, varname, varchars): return context_bgpython.node.modifynamechar(iname, varname, varchars)
def modifynamegbk(iname, varname, varchars): return context_bgpython.node.modifynamegbk(iname, varname, varchars)
def modifynameutf8(iname, varname, varchars): return context_bgpython.node.modifynameutf8(iname, varname, varchars)

def modifynamenumu8char(iname, varname, varchars): return context_bgpython.node.modifynamenumu8char(iname, varname, varchars)
def modifynamenumu16char(iname, varname, varchars): return context_bgpython.node.modifynamenumu16char(iname, varname, varchars)
def modifynamenumu32char(iname, varname, varchars): return context_bgpython.node.modifynamenumu32char(iname, varname, varchars)
def modifynamenumu32gbk(iname, varname, varchars): return context_bgpython.node.modifynamenumu32gbk(iname, varname, varchars)
def modifynamenumu32utf8(iname, varname, varchars): return context_bgpython.node.modifynamenumu32utf8(iname, varname, varchars)



def readuint80b(varbyte): return context_bgpython.node.readuint80b(varbyte)
def readuint160b(varbyte): return context_bgpython.node.readuint160b(varbyte)
def readuint320b(varbyte): return context_bgpython.node.readuint320b(varbyte)
def readuint640b(varbyte): return context_bgpython.node.readuint640b(varbyte)

def readint80b(varbyte): return context_bgpython.node.readint80b(varbyte)
def readint160b(varbyte): return context_bgpython.node.readint160b(varbyte)
def readint320b(varbyte): return context_bgpython.node.readint320b(varbyte)
def readint640b(varbyte): return context_bgpython.node.readint640b(varbyte)