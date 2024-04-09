import 界面文件.bcontext.bsfunction as bs
import 底层文件.bpformat.byfunction as by
import 底层文件

from . import tkinter_file


import lz4.block

from . import tkinter_file
from . import tkinter_unity_asserts

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
        self.unity3d_asserts_文件分块_wrapper = self.wrappercontext(self.unity3d_asserts_文件分块)

    def unity3d_asserts_文件分块(self):
        unity3d_asserts_文件分块(self)


def unity3d_asserts_文件分块(self):
    bs.insertvalue(text=f"是小端")
    by.endian("<")

    self.wrapperinsert(function=tkinter_unity_asserts.asserts_stream_head)(       self, label="stream_head")
    typeflags, offsets = [], []
    self.wrapperinsert(function=asserts_stream_type_list)(                        self, label="stream_type", typeflags=typeflags)
    self.wrapperinsert(function=tkinter_unity_asserts.asserts_stream_offset_list)(self, label="stream_offset", offsets=offsets)
    self.wrapperinsert(function=tkinter_unity_asserts.asserts_stream_unknow_list)(self, label="stream_unknow",)
    self.wrapperinsert(function=tkinter_unity_asserts.asserts_stream_name_list)(  self, label="stream_name",)
    self.wrapperinsert(function=tkinter_unity_asserts.asserts_stream_pad0)(       self, label="stream_pad0",)

    typenames = []
    for typeflag in typeflags:
        try:
            typenames.append(flag_name_dict[typeflag])
        except:
            raise ValueError(f"{[typeflag]}, 有1个或多个未识别的typeflag！")
    with bs.insertvalue(text=f"typenames"): 
        for name in typenames: bs.insertvalue(text=name)

    self.wrapperinsert(function=tkinter_unity_asserts.asserts_stream_file_list)(  self, label="stream_file", offsets=offsets, typenames=typenames)

    bs.insertblock(text=f"余下_{by.remainsize()}", bp=by.readremainslice().bp)



def asserts_stream_type_list(*args, **kwargs):
    typeflags = kwargs.get("typeflags", [])

    numbe = by.readuint32seek0()
    bs.insertblock(text=f"_{numbe}_4", bp=by.readslice(4).bp)
    for i in range(numbe):
        flag = by.readuint32seek0()
        typeflags.append(flag)
        sizeb = 23
        if flag == 114: sizeb = 39
        bs.insertblock(text=f"_{i}_{sizeb}", bp=by.readslice(sizeb).bp)
        numbinfo, sizechar = by.readuint32seek0(2)
        bs.insertblock(text=f"块_{i}_{[numbinfo, sizechar]}", bp=by.readslice(8).bp)
        sizeinfo = numbinfo * 24
        bs.insertblock(text=f"块_{i}_info_{sizeinfo}", bp=by.readslice(sizeinfo).bp)
        bs.insertblock(text=f"块_{i}_char_{sizechar}", bp=by.readslice(sizechar).bp)




















