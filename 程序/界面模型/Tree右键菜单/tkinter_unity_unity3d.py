
import 界面.bcontext.bsfunction as bs
import 文件.bpformat.byfunction as by
import 文件

import lz4.block

from . import tkinter_file
from . import tkinter_unity_asserts
from . import tkinter_unity_unity3d_asserts

flag_name_dict = tkinter_unity_asserts.flag_name_dict


class CompressionType: # https://github.com/HearthSim/UnityPack
    NONE = 0
    LZMA = 1
    LZ4 = 2
    LZ4HC = 3
    LZHAM = 4


class 类(tkinter_file.类):
    def __init__(self, frametreeview):
        self.frametreeview = frametreeview
        self.unity3d_文件分块_wrapper = self.wrappercontext(self.unity3d_文件分块)


    def unity3d_文件分块(self):
        bs.insertvalue(text=f"是大端")
        by.endian(">")

        self.wrapperinsert(function=unity3d_head)(  self, label="unity3d_head")
        offsets = []
        self.wrapperinsert(function=unity3d_uncompress_offset)(self, label="unity3d_uncompress_offset", offsets=offsets)
        self.wrapperinsert(function=unity3d_uncompress_data)(  self, label="unity3d_uncompress_data", offsets=offsets)

        bs.insertblock(text=f"余下_{by.remainsize()}", bp=by.readremainslice().bp)




def unity3d_head(*args, **kwargs):
    bs.insertblock(text=f"Head_8_", bp=by.readslice(8).bp) # 'UnityFS' end0
    bs.insertblock(text=f"Head_4_", bp=by.readslice(4).bp) # 6 int32 or uint32
    bs.insertblock(text=f"Head_6_", bp=by.readslice(6).bp) # '5.x.x'
    bs.insertblock(text=f"Head_8_", bp=by.readslice(8).bp) # '5.6.4p4'
    bs.insertblock(text=f"Head_sizefile_8_{by.readuint64seek0()}", bp=by.readslice(8).bp) # 30967414



def unity3d_uncompress_offset(*args, **kwargs):
    self = args[0]
    offsets = kwargs.get("offsets", [])

    解压前大小, 解压后大小, 解压标记 = by.readuint32seek0(3)
    bs.insertblock(text=f"Head_compsize_12_{[解压前大小, 解压后大小, 解压标记]}", bp=by.readslice(12).bp) # 2133, 4761, 67
    压缩块 = by.read(解压前大小)

    # 67    0b 01000011 # flag
    # 0x3F  0b 00111111 # compression
    # 0x80  0b 10000000 # eof_metadata # if True buf.seek(-self.ciblock_size, 2)
    解压类型 = 解压标记 & 0x3F
    if 解压类型 in [2, 3]: 解压块 = lz4.block.decompress(压缩块, 解压后大小)
    with bs.insertblock(text=f"块_解压前_{解压前大小}", bp=by.fromstream(压缩块).bp): pass
    with bs.insertblock(text=f"块_解压后_{解压后大小}", bp=by.fromstream(解压块).bp), by.fromstream(解压块):
        self.wrapperinsert(function=unity3d_offset_read_offset)(self, label="unity3d_offset", offsets=offsets)
        self.wrapperinsert(function=unity3d_offset_read_node)(  self, label="unity3d_node")
        bs.insertblock(text=f"余下_{by.remainsize()}", bp=by.readremainslice().bp)


def unity3d_offset_read_offset(*args, **kwargs):
    offsets = kwargs.get("offsets", [])

    bs.insertblock(text=f"块_guid_16", bp=by.readslice(16).bp)
    numbe = by.readuint32seek0()
    bs.insertblock(text=f"块_info_4_{numbe}", bp=by.readslice(4).bp)
    for i in range(numbe):
        with by.readsliceseek0(10): 解压后大小, 解压前大小, 解压类型 = by.readuint32(), by.readuint32(), by.readuint16()
        offsets.append([解压后大小, 解压前大小, 解压类型])
        bs.insertblock(text=f"块_{i}_后前_{[解压后大小, 解压前大小, 解压类型]}", bp=by.readslice(10).bp)


def unity3d_offset_read_node(*args, **kwargs):
    numbe = by.readuint32seek0()   
    bs.insertblock(text=f"块_node_4_{numbe}", bp=by.readslice(4).bp)
    for i in range(numbe):
        start = by.tell()
        offset, size, statu, name = by.readuint64(), by.readuint64(), by.readuint32(), by.readcharend0()
        final = by.tell()
        sizeb = final -start
        by.seek(-sizeb)
        bs.insertblock(text=f"块_{i}_{[offset, size, statu, name]}", bp=by.readslice(sizeb).bp)


def unity3d_uncompress_data(*args, **kwargs):
    self = args[0]
    offsets = kwargs.get("offsets", [])

    with bs.insertvalue(text=f"文件分块解压缩"):
        file = b""
        for i, [解压后大小, 解压前大小, 解压类型] in enumerate(offsets):
            压缩块 = by.read(解压前大小)
            if 解压类型 in [2, 3]: 解压块 = lz4.block.decompress(压缩块, 解压后大小)
            file += 解压块
            bs.insertblock(text=f"_{i}_解压前_{解压前大小}", bp=by.fromstream(压缩块).bp)
            bs.insertblock(text=f"_{i}_解压后_{解压后大小}", bp=by.fromstream(解压块).bp)


    with bs.insertblock(text=f"解压出来的文件_{len(file)}", bp=by.fromstream(file).bp), by.fromstream(file):
        tkinter_unity_unity3d_asserts.unity3d_asserts_文件分块(self)






