import os
import copy
import mmap

from .bgtype import bbyte
from .bgtype import bchar

from . import stacke_python



class 类0: pass 




class 类(stacke_python.类):
    def __init__(self, mpbyte=None, endian="<"):
        stacke_python.类.__init__(self, mpbyte=mpbyte, endian=endian)
        self.name = None
        self.parent = None
        self.slice0b = None
        self.upinput = []
        self.downinput = []


    def filepath(self, filepath):
        self.name = os.path.basename(filepath)
        self.parent = None
        self.rdpath = filepath
        self.rdfile = open(filepath, "rb")
        self.mpbyte = mmap.mmap(self.rdfile.fileno(), 0, access=mmap.ACCESS_READ) # mpfile
        self.mpleft, self.index, self.mpright = 0, 0, os.path.getsize(filepath)
        return self

    def close(self):
        self.mpbyte.close()
        self.rdfile.close()

    def copy(self):
        bg = 类(mpbyte=self.mpbyte, endian=self.endian)
        bg.mpleft, bg.mpright = self.mpleft, self.mpright
        bg.index = bg.mpleft
        bg.rdpath, bg.rdfile = self.rdpath, self.rdfile

        bg.name = self.name
        bg.parent = self.parent
        bg.slice0b = bg.mpbyte[bg.mpleft: bg.mpright]
        bg.name_object_dict = copy.deepcopy(self.name_object_dict)
        return bg

    def readsliceseek0(self, size):
        left, right = self.__calc__size__no__seek__(size)
        bg = 类(mpbyte=self.mpbyte, endian=self.endian)
        bg.mpleft, bg.mpright = left, right
        bg.index = bg.mpleft
        bg.rdpath, bg.rdfile = self.rdpath, self.rdfile
        return bg

    def readremainsliceseek0(self):
        bg = self.readsliceseek0(self.remainsize())
        return bg




    def __len__(self):
        return len(self.name_object_dict)
    

    def __find__iname__(self, iname:str|int):
        for i, name in enumerate(self.name_object_dict):
            if iname == i: return name
            if iname == name: return name
        raise ValueError(f"bnode::key::{iname}未找到...")


    def __getitem__(self, iname:int|str):
        name = self.__find__iname__(iname)
        return self.name_object_dict[name]

    def __placeholder__name__(self):
        for i in range(1000):
            name = "collection" + ".%03d"%i
            if name not in self.name_object_dict: return name
        raise ValueError(f"bnode::key::占位命名超过1000个...")

    def __look__iname__(self, iname:str|int):
        findname = None
        for i, name in enumerate(self.name_object_dict):
            if iname == i: findname = name
            if iname == name: findname = name
        if findname != None: return findname
        if isinstance(iname, str):
            return self.__placeholder__name__()
        if isinstance(iname, int):
            raise ValueError(f"bnode::key::索引{iname}不存在，请使用name方式写入...")

    def __setitem__(self, iname, value):
        name = self.__look__iname__(iname)
        self.name_object_dict[name] = value

    def __delitem__(self, iname):
        for i, name in enumerate(self.name_object_dict):
            if iname == i: 
                del self.name_object_dict[name] # 删除键值对
                break
            if iname == name: 
                del self.name_object_dict[name] # 删除键值对
                break

    def __iter__(self):
        return self.name_object_dict

    def insert(self, iname, bnodes):
        name = self.__find__iname__(iname)
        keys = list(self.name_object_dict.keys())
        i = keys.index(name)
        delete_keys = keys[i+1: ]
        end_key_value_list = [[key, self.name_object_dict[key]] for key in delete_keys]
        for key in delete_keys: self.name_object_dict.pop(key)  #  ==  del self.name_object_dict[key]
        if isinstance(bnodes, list):
            for bnode in bnodes:  
                name = self.__check__name__(bnode.name)
                self.name_object_dict[name] = bnode
        else:
            name = self.__check__name__(bnodes.name)
            self.name_object_dict[name] = bnodes
        for key, value in end_key_value_list: self.name_object_dict[key] = value


    def keys(self):
        return self.name_object_dict.keys()        

    def values(self):
        return self.name_object_dict.values()    

    def items(self):
        return self.name_object_dict.items()

    def pop(self, name:str):
        name = self.__find__iname__(name)
        return self.name_object_dict.pop(name)

    def getname_fromobject(self, bnode):
        for name, o in self.name_object_dict.items():
            if o == bnode: return name
        raise ValueError(f"bnode::key::所获取的对象{bnode}不存在")





    def __check__name__(self, checkname):
        if checkname == "": checkname = "collection"
        findname = False
        for i, name in enumerate(self.name_object_dict):
            if checkname == name: findname = True
        if findname == False: return checkname
        for i in range(1000):
            name = checkname + ".%03d"%i
            if name not in self.name_object_dict: return name
        raise ValueError(f"bnode::key::占位命名超过1000个...")


    def bnodeslice(self, size, name=""):
        left, right = self.__calc__size__(size)
        bg = 类(mpbyte=self.mpbyte, endian=self.endian)
        bg.mpleft, bg.mpright = left, right
        bg.index = bg.mpleft
        bg.rdpath, bg.rdfile = self.rdpath, self.rdfile

        bg.slice0b = bg.mpbyte[bg.mpleft: bg.mpright]
        bg.parent = self
        bg.name = self.__check__name__(name)
        self.name_object_dict[bg.name] = bg
        return bg
        

    def bnoderemainslice(self, name=""):
        bg = self.bnodeslice(self.remainsize(), name)
        return bg



    def bnodeuint8(self, num=1, name=""):
        left, right = self.__calc__size__(num)
        name = self.__check__name__(name)
        bnode = bbyte(name=name, endian=self.endian).uint8_frombytes(self.mpbyte[left:right])
        bnode.parent = self
        self.name_object_dict[name] = bnode
        return bnode

    def bnodeuint16(self, num=1, name=""):
        left, right = self.__calc__size__(2*num)
        name = self.__check__name__(name)
        bnode = bbyte(name=name, endian=self.endian).uint16_frombytes(self.mpbyte[left:right])
        bnode.parent = self
        self.name_object_dict[name] = bnode
        return bnode
    
    def bnodeuint32(self, num=1, name=""):
        left, right = self.__calc__size__(4*num)
        name = self.__check__name__(name)
        bnode = bbyte(name=name, endian=self.endian).uint32_frombytes(self.mpbyte[left:right])
        bnode.parent = self
        self.name_object_dict[name] = bnode
        return bnode
    
    def bnodeuint64(self, num=1, name=""):
        left, right = self.__calc__size__(8*num)
        name = self.__check__name__(name)
        bnode = bbyte(name=name, endian=self.endian).uint64_frombytes(self.mpbyte[left:right])
        bnode.parent = self
        self.name_object_dict[name] = bnode
        return bnode

    def bnodeint8(self, num=1, name=""):
        left, right = self.__calc__size__(num)
        name = self.__check__name__(name)
        bnode = bbyte(name=name, endian=self.endian).int8_frombytes(self.mpbyte[left:right])
        bnode.parent = self
        self.name_object_dict[name] = bnode
        return bnode
    
    def bnodeint16(self, num=1, name=""):
        left, right = self.__calc__size__(2*num)
        name = self.__check__name__(name)
        bnode = bbyte(name=name, endian=self.endian).int16_frombytes(self.mpbyte[left:right])
        bnode.parent = self
        self.name_object_dict[name] = bnode
        return bnode
    
    def bnodeint32(self, num=1, name=""):
        left, right = self.__calc__size__(4*num)
        name = self.__check__name__(name)
        bnode = bbyte(name=name, endian=self.endian).int32_frombytes(self.mpbyte[left:right])
        bnode.parent = self
        self.name_object_dict[name] = bnode
        return bnode
    
    def bnodeint64(self, num=1, name=""):
        left, right = self.__calc__size__(8*num)
        name = self.__check__name__(name)
        bnode = bbyte(name=name, endian=self.endian).int64_frombytes(self.mpbyte[left:right])
        bnode.parent = self
        self.name_object_dict[name] = bnode
        return bnode

    def bnodefloat16(self, num=1, name=""):
        left, right = self.__calc__size__(2*num)


    def bnodefloat32(self, num=1, name=""):
        left, right = self.__calc__size__(4*num)


    def bnodefloat64(self, num=1, name=""):
        left, right = self.__calc__size__(8*num)

    def bnodechar(self, bytenum, name=""):
        return self.bnodeascii(bytenum, name)


    def bnodeascii(self, bytenum, name=""):
        left, right = self.__calc__size__(bytenum)
        name = self.__check__name__(name)
        bnode = bbyte(name=name, endian=self.endian).ascii_frombytes(self.mpbyte[left:right])
        bnode.parent = self
        self.name_object_dict[name] = bnode
        return bnode
    
    def bnodeutf8(self, bytenum, name=""):
        left, right = self.__calc__size__(bytenum)
        name = self.__check__name__(name)
        bnode = bbyte(name=name, endian=self.endian).utf8_frombytes(self.mpbyte[left:right])
        bnode.parent = self
        self.name_object_dict[self.__check__name__(name)] = bnode
        return bnode
        
    def bnodegbk(self, bytenum, name=""):
        left, right = self.__calc__size__(bytenum)
        name = self.__check__name__(name)
        bnode = bbyte(name=name, endian=self.endian).gbk_frombytes(self.mpbyte[left:right])
        bnode.parent = self
        self.name_object_dict[name] = bnode
        return bnode

    def bnodestring(self, size, name="", inttype="uint8", strtype="utf8"):
        left, right = self.__calc__size__(size)
        slice0b = self.mpbyte[left:right]
        if inttype not in ['uint8', 'uint16', 'uint32']: raise ValueError(f"bchar::inttype::接受类型为[\"uint8\", \"uint16\", \"uint32\"]，not {inttype}")
        bnode = bchar(inttype=inttype, strtype=strtype, endian=self.endian).fromslice0b(slice0b)
        bnode.parent = self
        self.name_object_dict[self.__check__name__(name)] = bnode
        return bnode


    def updatesize(self):
        slice0b = self.updateslice0b()
        return len(slice0b)
    
    def iteratesize(self):
        slice0b = self.iterateslice0b()
        return len(slice0b)

    def updateslice0b(self):
        slice0b = b""
        for name, bnode in self.name_object_dict.items():
            slice0b = slice0b + bnode.slice0b
        self.slice0b = slice0b
        return slice0b
    
    def iterateslice0b(self):
        slice0b = b""
        for name, bnode in self.name_object_dict.items():
            if isinstance(bnode, 类): bnode.iterateslice0b()
            slice0b = slice0b + bnode.slice0b
        self.slice0b = slice0b
        return slice0b


    def addstart(self, iname):
        name = self.__find__iname__(iname)
        keys = list(self.name_object_dict.keys())
        i = keys.index(name)
        delete_keys = keys[i+1: ]
        self.__end__key__value__list__ = [[key, self.name_object_dict[key]] for key in delete_keys]
        for key in delete_keys: self.name_object_dict.pop(key)  #  ==  del self.name_object_dict[key]

    def addstring(self, strs, name="", inttype="uint8", strtype="utf8"):
        bnode = bchar(inttype=inttype, strtype=strtype, endian=self.endian).fromstring(strs)
        bnode.parent = self
        self.name_object_dict[self.__check__name__(name)] = bnode
        return bnode


    def addfinal(self):
        for key, value in self.__end__key__value__list__: self.name_object_dict[key] = value






    # def bnodesliceseek0(self, size):
    #     left, right = self.__calc__size__no__seek__(size)
    #     bg = 类(mpbyte=self.mpbyte, endian=self.endian)
    #     bg.mpleft, bg.mpright = left, right
    #     bg.index = bg.mpleft
    #     bg.rdpath, bg.rdfile = self.rdpath, self.rdfile

    #     bg.slice0b = bg.mpbyte[bg.mpleft: bg.mpright]
    #     return bg


    # def bnoderemainsliceseek0(self):
    #     bg = self.bnodesliceseek0(self.remainsize())
    #     return bg


    # def bnodeuint8seek0(self, num=1):
    #     left, right = self.__calc__size__no__seek__(num)
    #     bnode = bbyte(self.endian).uint8_frombytes(self.mpbyte[left:right])

    # def bnodeuint16seek0(self, num=1):
    #     left, right = self.__calc__size__no__seek__(2*num)
    #     bnode = bbyte(self.endian).uint16_frombytes(self.mpbyte[left:right])

    # def bnodeuint32seek0(self, num=1):
    #     left, right = self.__calc__size__no__seek__(4*num)
    #     bnode = bbyte(self.endian).uint32_frombytes(self.mpbyte[left:right])


    # def bnodeuint64seek0(self, num=1):
    #     left, right = self.__calc__size__no__seek__(8*num)
    #     bnode = bbyte(self.endian).uint64_frombytes(self.mpbyte[left:right])


    # def bnodeint8seek0(self, num=1):
    #     left, right = self.__calc__size__no__seek__(num)
    #     bnode = bbyte(self.endian).int8_frombytes(self.mpbyte[left:right])

    # def bnodeint16seek0(self, num=1):
    #     left, right = self.__calc__size__no__seek__(2*num)
    #     bnode = bbyte(self.endian).int16_frombytes(self.mpbyte[left:right])

    # def bnodeint32seek0(self, num=1):
    #     left, right = self.__calc__size__no__seek__(4*num)
    #     bnode = bbyte(self.endian).int32_frombytes(self.mpbyte[left:right])

    # def bnodeint64seek0(self, num=1):
    #     left, right = self.__calc__size__no__seek__(8*num)
    #     bnode = bbyte(self.endian).int64_frombytes(self.mpbyte[left:right])


    # def bnodefloat16seek0(self, num=1):
    #     left, right = self.__calc__size__no__seek__(2*num)


    # def bnodeloat32seek0(self, num=1):
    #     left, right = self.__calc__size__no__seek__(4*num)


    # def bnodefloat64seek0(self, num=1):
    #     left, right = self.__calc__size__no__seek__(8*num)

    # def bnodecharseek0(self, bytenum):
    #     return self.bnodeasciiseek0(bytenum)



    # def bnodeasciiseek0(self, bytenum):
    #     left, right = self.__calc__size__no__seek__(bytenum)
    #     bnode = bbyte(self.endian).ascii_frombytes(self.mpbyte[left:right])
    
    # def bnodeutf8seek0(self, bytenum):
    #     left, right = self.__calc__size__no__seek__(bytenum)
    #     bnode = bbyte(self.endian).utf8_frombytes(self.mpbyte[left:right])
    
    # def bnodegbkseek0(self, bytenum):
    #     left, right = self.__calc__size__no__seek__(bytenum)
    #     bnode = bbyte(self.endian).gbk_frombytes(self.mpbyte[left:right])




            # if iname < 0: raise ValueError(f"bnode::key::您输入的索引为{iname}, 请输入正整数...")
            # for m in range(iname-len(self.name_object_dict)):
            #     name = self.__placeholder__name__()
            #     self.name_object_dict[name] = None
            # return self.__placeholder__name__()