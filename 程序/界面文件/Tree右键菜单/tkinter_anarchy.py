import 界面文件.bcontext.bsfunction as bs
import 底层文件.bpformat.byfunction as by
import 底层文件

from . import tkinter_file


class 类(tkinter_file.类):
    def __init__(self, frametreeview):
        self.frametreeview = frametreeview

        self.文件分块_wrapper = self.wrappercontext(self.文件分块)
        self.HSMV_wrapper = self.wrappercontext(self.HSMV)
        self.MOEG_wrapper = self.wrappercontext(self.MOEG)
        self.MESH_wrapper = self.wrappercontext(self.MESH)


    def 文件分块(self):
        bs.insertblock(text=f"Head_8", bp=by.readslice(8).bp)
        while True:
            if by.remainsize() < 20: break
            with by.readsliceseek0(12):
                flag, name, size = by.readuint32(), by.readchar(4), by.readuint32() # leftflag, leftname, size, bp, rightflag, rightname
            bs.insertblock(text=f"_{[flag, name, size, ]}_{size+20}", bp=by.readslice(size+20).bp)
        bs.insertblock(text=f"余下_{by.remainsize()}", bp=by.readremainslice().bp)


    def HSMV(self):
        bs.insertblock(text="Flag_12", bp=by.readslice(12).bp)
        bs.insertblock(text="Head_ 8", bp=by.readslice( 8).bp)
        with bs.insertvalue(text=f"MESH_"):
            self.MESH()
        bs.insertblock(text=f"余下_{by.remainsize()}", bp=by.readremainslice().bp)


    def MOEG(self):
        bs.insertblock(text="Flag_12", bp=by.readslice(12).bp)
        bs.insertblock(text="Head_ 4", bp=by.readslice( 4).bp)
        with bs.insertvalue(text=f"MESH_"):
            self.MESH()
        bs.insertblock(text=f"余下_{by.remainsize()}", bp=by.readremainslice().bp)

    def MESH(self):
        MESH()



def MESH():
    with bs.insertblock(text="head_8", bp=by.readsliceseek0(8).bp), by.readslice(8):
        magicchar = by.readcharseek0(4)
        magic, version = by.readuint8(4), by.readuint32()
        bs.insertvalue(text="magic, version", values=[magicchar, magic, version])
    with bs.insertblock(text="decs_60", bp=by.readsliceseek0(60).bp), by.readslice(60):
        l4uint8, bx, r4uint8 = by.readuint8(4), by.readslice(by.readuint32()).bp, by.readuint8(4)
        bs.insertblock(text=str([l4uint8, "bp", r4uint8]), bp=bx)
        顶点大小 = bx.readuint16()

    match version:
        case 1: numvert, numloop = MESH_info_version_1() # 30
        case 3: numvert, numloop = MESH_info_version_3() # 33
        case 5: numvert, numloop = MESH_info_version_5() # 36
        case _: print(f"Anarchymeshfile::未知mesh_info_type_{version}")

    uint8列表 = by.readuint8seek0(2)
    match uint8列表:
        case [0, 0]: bs.insertblock(text="block", bp=by.readslice(2).bp)
        case [255, 255]: bs.insertblock(text="block", bp=by.readslice(66).bp)

    
    sizevert = numvert * 顶点大小
    bs.insertblock(text="vert", bp=by.readslice(sizevert).bp)
    sizeloop = numloop * 2
    if numvert > 0xFFFF: sizeloop = numloop * 4
    bs.insertblock(text="loop", bp=by.readslice(sizeloop).bp)


def MESH_info_version_1():
    size = 30
    with bs.insertblock(text=f"mesh_info_{size}", bp=by.readsliceseek0(size).bp), by.readslice(size):
        顶点总数, 未知1 = by.readuint32(), by.readslice(5).bp
        Loop总数, 未知2 = by.readuint32(), by.readslice(4).bp
        Face总数, 未知3 = by.readuint32(), by.readslice(9).bp
        bs.insertblock(text=f"numvert: {顶点总数}", bp=未知1)
        bs.insertblock(text=f"numloop: {Loop总数}", bp=未知2)
        bs.insertblock(text=f"numface: {Face总数}", bp=未知3)
    return 顶点总数, Loop总数


def MESH_info_version_3():
    size = 33
    with bs.insertblock(text=f"mesh_info_{size}", bp=by.readsliceseek0(size).bp), by.readslice(size):
        顶点总数, 未知1 = by.readuint32(), by.readslice(8).bp
        Loop总数, 未知2 = by.readuint32(), by.readslice(4).bp
        Face总数, 未知3 = by.readuint32(), by.readslice(9).bp # 0xFFFF FFFF
        bs.insertblock(text=f"numvert: {顶点总数}", bp=未知1)
        bs.insertblock(text=f"numloop: {Loop总数}", bp=未知2)
        bs.insertblock(text=f"numface: {Face总数}", bp=未知3)
    return 顶点总数, Loop总数

def MESH_info_version_5():
    size = 36
    with bs.insertblock(text=f"mesh_info_{size}", bp=by.readsliceseek0(size).bp), by.readslice(size):
        顶点总数, 未知1 = by.readuint32(), by.readslice(9).bp
        Loop总数, 未知2 = by.readuint32(), by.readslice(4).bp
        Face总数, 未知3 = by.readuint32(), by.readslice(11).bp
        bs.insertblock(text=f"numvert: {顶点总数}", bp=未知1)
        bs.insertblock(text=f"numloop: {Loop总数}", bp=未知2)
        bs.insertblock(text=f"numface: {Face总数}", bp=未知3)
    return 顶点总数, Loop总数

# from Tkinter import Tk
# r = Tk()
# r.withdraw()
# r.clipboard_clear()
# r.clipboard_append('i can has clipboardz?')
# r.destroy()