import copy

# 在python3.5及以上版本中int长度理论上是无限的
class bbyte:
    def __init__(self, endian:str="<", type:str="", data:int|float|list=0, slice0b:bytes=b""):
        self.endian = endian
        self.type = type
        self.data = data
        self.slice0b = slice0b

class bchar:
    def __init__(self, endian:str="<", type:str="", char:str="", slice0b:bytes=b""):
        self.endian = endian
        self.type = type
        self.char = char
        self.slice0b = slice0b


class bnode:
    def __init__(self, endian="<"):
        self.bp = None
        self.parent = None
        self.ibytedict = {}
        self.inamedict = {}
        self.count = -1
        self.endian = endian
        self.slice0b = b""
  

    def __enter__(self):
        # 主要用个格式缩进
        return self

    def __exit__(self, type, value, traceback):
        return ""

    def __len__(self):
        return self.count + 1
    
    def __getitem__(self, iname:int|str):
        i = self.__find__iname__(iname)
        return self.ibytedict[i]

    def __setitem__(self, iname, value):
        match value:
            case value if isinstance(value, bytes): self.addslice0b(iname, value)
            case value if isinstance(value, bnode): self.addnode(iname, value)
            case _: raise ValueError(f"bgpython::bnode::元素成员类型错误，类型为{type(value)}，但其应为bytes或者bnode类型...")


    def __delitem__(self, iname):
        pass

    def __iter__(self):
        pass


    def __find__name__(self, chars:str=""):
        for i, name in self.inamedict.items():
            if chars == name: return i
        return None


    def __find__index__(self, index:int):
        number = self.count + 1
        if index < 0: index = index + number
        if index >= 0 and index < number: 
            return index
        else:
            return None


    def __find__iname__(self, iname:str|int):
        if isinstance(iname, str):
            i = self.__find__name__(iname)
            if i == None : raise ValueError(f"变量名称{iname}未找到...")
        if isinstance(iname, int):
            i = self.__find__index__(iname)
            if i == None : raise ValueError(f"索引值{iname}未找到，索引最大值为{self.count}...")
        return i


    def __calc__index__(self):
        self.count += 1
        return self.count


    def __calc__size__(self):
        size = 0
        for i, value in self.ibytedict.items(): 
            match value:
                case value if isinstance(value, bytes): size += len(value)
                case value if isinstance(value, bbyte): size += len(value.slice0b)
                case value if isinstance(value, bchar): size += len(value.slice0b)                
                case value if isinstance(value, bnode): size += len(value.slice0b)
                case _: raise ValueError(f"bgpython::bnode::元素成员类型错误，类型为{type(value)}，但其应为bytes类型...")
        return size


    def __recursion__size__(self):
        size = 0
        for i, value in self.ibytedict.items(): 
            match value:
                case value if isinstance(value, bytes): size += len(value)
                case value if isinstance(value, bbyte): size += len(value.slice0b)
                case value if isinstance(value, bchar): size += len(value.slice0b)             
                case value if isinstance(value, bnode): size += value.__recursion__size__()
                case _: raise ValueError(f"bgpython::bnode::元素成员类型错误，类型为{type(value)}，但其应为bytes类型...")
        return size



    def __calc__slice0b__(self):
        self.slice0b = b""
        for i, value in self.ibytedict.items(): 
            match value:
                case value if isinstance(value, bytes): self.slice0b += value
                case value if isinstance(value, bbyte): self.slice0b += value.slice0b
                case value if isinstance(value, bchar): self.slice0b += value.slice0b               
                case value if isinstance(value, bnode): self.slice0b += value.slice0b
                case _: raise ValueError(f"bgpython::bnode::元素成员类型错误，类型为{type(value)}，但其应为bytes类型...")
        return self.slice0b


    def __recursion__0b__(self):
        self.slice0b = b""
        for i, value in self.ibytedict.items(): 
            match value:
                case value if isinstance(value, bytes): self.slice0b += value
                case value if isinstance(value, bbyte): self.slice0b += value.slice0b
                case value if isinstance(value, bchar): self.slice0b += value.slice0b               
                case value if isinstance(value, bnode): self.slice0b += value.__recursion__0b__()
                case _: raise ValueError(f"bgpython::bnode::元素成员类型错误，类型为{type(value)}，但其应为bytes类型...")
        return self.slice0b


    def __check__range__(self, numbyte, signed, varvalue):
        match [numbyte, signed]:
            case [1, True]: 
                if varvalue < 0 or varvalue > 255: raise ValueError(f"值{varvalue}超过uint8[0, 255]范围...")
            case [1, False]: 
                if varvalue < -128 or varvalue > 127: raise ValueError(f"值{varvalue}超过int8[-128, 127]范围...")
            case [2, True]:
                if varvalue < 0 or varvalue > 65535: raise ValueError(f"值{varvalue}超过uint16[0, 65535]范围...")
            case [2, False]:
                if varvalue < -32768 or varvalue > 32767: raise ValueError(f"值{varvalue}超过int16[-32768, 32767]范围...")
            case [4, True]:
                if varvalue < 0 or varvalue > 4294967295: raise ValueError(f"值{varvalue}超过uint32[0, 4294967295]范围...")
            case [4, False]:
                if varvalue < -2147483648 or varvalue > 2147483647: raise ValueError(f"值{varvalue}超过int32[-2147483648, 2147483647]范围...")
            case [8, True]:
                if varvalue < 0 or varvalue > 18446744073709551615: raise ValueError(f"值{varvalue}超过uint64[0, 18446744073709551615]范围...")
            case [8, False]:
                if varvalue < -9223372036854775808 or varvalue > 9223372036854775807: raise ValueError(f"值{varvalue}超过int64[-9223372036854775808, 9223372036854775807]范围...")


    def __convert_int2bytes__(self, numbyte, signed, varvalue):
        if self.endian == ">": 
            return varvalue.to_bytes(numbyte, byteorder="big", signed=signed)
        else: 
            return varvalue.to_bytes(numbyte, byteorder="little", signed=signed)


    def size(self):
        return self.__recursion__size__()


    def find(self, iname:int|str=-1):
        if iname == -1: return self.inamedict[self.count], self.ibytedict[self.count]
        i = self.__find__iname__(iname)
        return self.inamedict[i], self.ibytedict[i]


    def index(self, varvalue=None):
        for [i, name], [j, value] in zip(self.inamedict.items(), self.ibytedict.items()):
            if varvalue == value: return i, name
        return None, None

    def getslice0b(self):
        return self.__calc__slice0b__()
    
    def getnodetree0b(self):
        return self.__recursion__0b__()
    
    def setslice(self, bp):
        self.bp = bp
        self.slice0b = bp.readslice0b()
        return self



    def copy(self):
        类例 = bnode(self.endian)
        类例.bp = self.bp
        类例.parent = self.parent
        类例.ibytedict = copy.deepcopy(self.ibytedict)
        类例.inamedict = copy.deepcopy(self.inamedict)
        类例.count = self.count
        类例.slice0b = self.slice0b
        return 类例
    

    def emptynode(self):
        return bnode(self.endian)


    def __addint8__(self, varvalue:int):
        numbyte, signed = 1, True
        self.__check__range__(numbyte, signed, varvalue)
        return self.__convert_int2bytes__(numbyte, signed, varvalue)


    def __adduint8__(self, varvalue:int):
        numbyte, signed = 1, False
        self.__check__range__(numbyte, signed, varvalue)
        return self.__convert_int2bytes__(numbyte, signed, varvalue)


    def __addint16__(self, varvalue:int):
        numbyte, signed = 2, True
        self.__check__range__(numbyte, signed, varvalue)
        return self.__convert_int2bytes__(numbyte, signed, varvalue)


    def __adduint16__(self, varvalue:int):
        numbyte, signed = 2, False
        self.__check__range__(numbyte, signed, varvalue)
        return self.__convert_int2bytes__(numbyte, signed, varvalue)


    def __addint32__(self, varvalue:int):
        numbyte, signed = 4, True
        self.__check__range__(numbyte, signed, varvalue)
        return self.__convert_int2bytes__(numbyte, signed, varvalue)


    def __adduint32__(self,  varvalue:int):
        numbyte, signed = 4, False
        self.__check__range__(numbyte, signed, varvalue)
        return self.__convert_int2bytes__(numbyte, signed, varvalue)
    

    def __addint64__(self,  varvalue:int):
        numbyte, signed = 8, True
        self.__check__range__(numbyte, signed, varvalue)
        return self.__convert_int2bytes__(numbyte, signed, varvalue)


    def __adduint64__(self, varvalue:int):
        numbyte, signed = 8, False
        self.__check__range__(numbyte, signed, varvalue)
        return self.__convert_int2bytes__(numbyte, signed, varvalue)



    def addint8(self, varname:str="", varvalue:int|list=0):
        i = self.__calc__index__()
        if isinstance(varvalue, list):
            binvalue = b""
            for value in self.unpack_list_depth2(varvalue): binvalue += self.__addint8__(value)
            varbyte = bbyte(self.endian, "int8list", varvalue, binvalue)
        else:
            binvalue = self.__addint8__(varvalue)
            varbyte = bbyte(self.endian, "int8", varvalue, binvalue)
        self.ibytedict[i] = varbyte
        self.inamedict[i] = varname
        return self.ibytedict[i]


    def adduint8(self, varname:str="", varvalue:int|list=0):
        i = self.__calc__index__()
        if isinstance(varvalue, list):
            binvalue = b""
            for value in self.unpack_list_depth2(varvalue): binvalue += self.__adduint8__(value)
            varbyte = bbyte(self.endian, "uint8list", varvalue, binvalue)
        else:
            binvalue = self.__adduint8__(varvalue)
            varbyte = bbyte(self.endian, "uint8", varvalue, binvalue)
        self.ibytedict[i] = varbyte
        self.inamedict[i] = varname
        return self.ibytedict[i]


    def addint16(self, varname:str="", varvalue:int|list=0):
        i = self.__calc__index__()
        if isinstance(varvalue, list):
            binvalue = b""
            for value in self.unpack_list_depth2(varvalue): binvalue += self.__addint16__(value)
            varbyte = bbyte(self.endian, "int16list", varvalue, binvalue)
        else:
            binvalue = self.__addint16__(varvalue)
            varbyte = bbyte(self.endian, "int16", varvalue, binvalue)
        self.ibytedict[i] = varbyte
        self.inamedict[i] = varname
        return self.ibytedict[i]


    def adduint16(self, varname:str="", varvalue:int|list=0):
        i = self.__calc__index__()
        if isinstance(varvalue, list):
            binvalue = b""
            for value in self.unpack_list_depth2(varvalue): binvalue += self.__adduint16__(value)
            varbyte = bbyte(self.endian, "uint16list", varvalue, binvalue)
        else:
            binvalue = self.__adduint16__(varvalue)
            varbyte = bbyte(self.endian, "uint16", varvalue, binvalue)
        self.ibytedict[i] = varbyte
        self.inamedict[i] = varname
        return self.ibytedict[i]


    def addint32(self, varname:str="", varvalue:int|list=0):
        i = self.__calc__index__()
        if isinstance(varvalue, list):
            binvalue = b""
            for value in self.unpack_list_depth2(varvalue): binvalue += self.__addint32__(value)
            varbyte = bbyte(self.endian, "int32list", varvalue, binvalue)
        else:
            binvalue = self.__addint32__(varvalue)
            varbyte = bbyte(self.endian, "int32", varvalue, binvalue)
        self.ibytedict[i] = varbyte
        self.inamedict[i] = varname
        return self.ibytedict[i]


    def adduint32(self, varname:str="", varvalue:int|list=0):
        i = self.__calc__index__()
        if isinstance(varvalue, list):
            binvalue = b""
            for value in self.unpack_list_depth2(varvalue): binvalue += self.__adduint32__(value)
            varbyte = bbyte(self.endian, "uint32list", varvalue, binvalue)
        else:
            binvalue = self.__adduint32__(varvalue)
            varbyte = bbyte(self.endian, "uint32", varvalue, binvalue)
        self.ibytedict[i] = varbyte
        self.inamedict[i] = varname
        return self.ibytedict[i]
    

    def addint64(self, varname:str="", varvalue:int|list=0):
        i = self.__calc__index__()
        if isinstance(varvalue, list):
            binvalue = b""
            for value in self.unpack_list_depth2(varvalue): binvalue += self.__addint64__(value)
            varbyte = bbyte(self.endian, "int64list", varvalue, binvalue)
        else:
            binvalue = self.__addint64__(varvalue)
            varbyte = bbyte(self.endian, "int64", varvalue, binvalue)
        self.ibytedict[i] = varbyte
        self.inamedict[i] = varname
        return self.ibytedict[i]


    def adduint64(self, varname:str="", varvalue:int|list=0):
        i = self.__calc__index__()
        if isinstance(varvalue, list):
            binvalue = b""
            for value in self.unpack_list_depth2(varvalue): binvalue += self.__adduint64__(value)
            varbyte = bbyte(self.endian, "uint64list", varvalue, binvalue)
        else:
            binvalue = self.__addint64__(varvalue)
            varbyte = bbyte(self.endian, "uint64", varvalue, binvalue)
        self.ibytedict[i] = varbyte
        self.inamedict[i] = varname
        return self.ibytedict[i]


    def addslice0b(self, varname:str="", varbyte:bytes=b""):
        i = self.__calc__index__()
        self.ibytedict[i] = varbyte
        self.inamedict[i] = varname
        return self.ibytedict[i]


    def addnode(self, varname:str="", varnode=None):
        if not isinstance(varnode, bnode): return ""
        i = self.__calc__index__()
        self.ibytedict[i] = varnode
        self.inamedict[i] = varname
        return self.ibytedict[i]


    def addchar(self, varname:str="", varchars:str=""):
        i = self.__calc__index__()
        varbyte = varchars.encode("ASCII")
        self.ibytedict[i] = bchar(self.endian, "ascii", varchars, varbyte)
        self.inamedict[i] = varname
        return self.ibytedict[i]


    def addutf8(self, varname:str="", varchars:str=""):
        i = self.__calc__index__()
        varbyte = varchars.encode("utf-8")
        self.ibytedict[i] = bchar(self.endian, "utf8", varchars, varbyte)
        self.inamedict[i] = varname
        return self.ibytedict[i]


    def addgbk(self, varname:str="", varchars:str=""):
        i = self.__calc__index__()
        varbyte = varchars.encode("GBK")
        self.ibytedict[i] = bchar(self.endian, "gbk", varchars, varbyte)
        self.inamedict[i] = varname
        return self.ibytedict[i]


    def addnumu8char(self, varname:str="", varchars:str=""):
        binchars = varchars.encode("ASCII")
        size = len(binchars)
        if self.endian == ">":
            binsize = size.to_bytes(1, byteorder="big", signed=False)
        else:
            binsize = size.to_bytes(1, byteorder="little", signed=False)
        i = self.__calc__index__()
        varbyte = binsize + binchars
        self.ibytedict[i] = bchar(self.endian, "ascii", varchars, varbyte)
        self.inamedict[i] = varname
        return self.ibytedict[i]


    def addnumu16char(self, varname:str="", varchars:str=""):
        binchars = varchars.encode("ASCII")
        size = len(binchars)
        if self.endian == ">":
            binsize = size.to_bytes(2, byteorder="big", signed=False)
        else:
            binsize = size.to_bytes(2, byteorder="little", signed=False)
        i = self.__calc__index__()
        varbyte = binsize + binchars
        self.ibytedict[i] = bchar(self.endian, "ascii", varchars, varbyte)
        self.inamedict[i] = varname
        return self.ibytedict[i]


    def addnumu32char(self, varname:str="", varchars:str=""):
        binchars = varchars.encode("ASCII")
        size = len(binchars)
        if self.endian == ">":
            binsize = size.to_bytes(4, byteorder="big", signed=False)
        else:
            binsize = size.to_bytes(4, byteorder="little", signed=False)
        i = self.__calc__index__()
        varbyte = binsize + binchars
        self.ibytedict[i] = bchar(self.endian, "ascii", varchars, varbyte)
        self.inamedict[i] = varname
        return self.ibytedict[i]


    def addnumu32utf8(self, varname:str="", varchars:str=""):
        binchars = varchars.encode("utf-8")
        size = len(binchars)
        if self.endian == ">":
            binsize = size.to_bytes(4, byteorder="big", signed=False)
        else:
            binsize = size.to_bytes(4, byteorder="little", signed=False)
        i = self.__calc__index__()
        varbyte = binsize + binchars
        self.ibytedict[i] = bchar(self.endian, "utf8", varchars, varbyte)
        self.inamedict[i] = varname
        return self.ibytedict[i]


    def addnumu32gbk(self, varname:str="", varchars:str=""):
        binchars = varchars.encode("gbk")
        size = len(binchars)
        if self.endian == ">":
            binsize = size.to_bytes(4, byteorder="big", signed=False)
        else:
            binsize = size.to_bytes(4, byteorder="little", signed=False)
        i = self.__calc__index__()
        varbyte = binsize + binchars
        self.ibytedict[i] = bchar(self.endian, "gbk", varchars, varbyte)
        self.inamedict[i] = varname
        return self.ibytedict[i]


    def updateslice0b(self):
        return self.__calc__slice0b__()


    def updatenodetree0b(self):
        return self.__recursion__0b__()

 
    # 修改变量值
    def modifyint8(self, iname:str|int="", varvalue:int|list=0):
        i = self.__find__iname__(iname)
        if isinstance(varvalue, list):
            binvalue = b""
            for value in self.unpack_list_depth2(varvalue): binvalue += self.__addint8__(value)
            varbyte = bbyte(self.endian, "int8list", varvalue, binvalue)
        else:
            binvalue = self.__addint8__(varvalue)
            varbyte = bbyte(self.endian, "int8", varvalue, binvalue)
        self.ibytedict[i] = varbyte
        return self.ibytedict[i]


    def modifyuint8(self, iname:str|int="", varvalue:int|list=0):
        i = self.__find__iname__(iname)
        if isinstance(varvalue, list):
            binvalue = b""
            for value in self.unpack_list_depth2(varvalue): binvalue += self.__adduint8__(value)
            varbyte = bbyte(self.endian, "uint8list", varvalue, binvalue)
        else:
            binvalue = self.__adduint8__(varvalue)
            varbyte = bbyte(self.endian, "uint8", varvalue, binvalue)
        self.ibytedict[i] = varbyte
        return self.ibytedict[i]


    def modifyint16(self, iname:str|int="", varvalue:int|list=0):
        i = self.__find__iname__(iname)
        if isinstance(varvalue, list):
            binvalue = b""
            for value in self.unpack_list_depth2(varvalue): binvalue += self.__addint16__(value)
            varbyte = bbyte(self.endian, "int16list", varvalue, binvalue)
        else:
            binvalue = self.__addint16__(varvalue)
            varbyte = bbyte(self.endian, "int16", varvalue, binvalue)
        self.ibytedict[i] = varbyte
        return self.ibytedict[i]


    def modifyuint16(self, iname:str|int="", varvalue:int|list=0):
        i = self.__find__iname__(iname)
        if isinstance(varvalue, list):
            binvalue = b""
            for value in self.unpack_list_depth2(varvalue): binvalue += self.__adduint16__(value)
            varbyte = bbyte(self.endian, "uint16list", varvalue, binvalue)
        else:
            binvalue = self.__adduint16__(varvalue)
            varbyte = bbyte(self.endian, "uint16", varvalue, binvalue)
        self.ibytedict[i] = varbyte
        return self.ibytedict[i]


    def modifyint32(self, iname:str|int="", varvalue:int|list=0):
        i = self.__find__iname__(iname)
        if isinstance(varvalue, list):
            binvalue = b""
            for value in self.unpack_list_depth2(varvalue): binvalue += self.__addint32__(value)
            varbyte = bbyte(self.endian, "int32list", varvalue, binvalue)
        else:
            binvalue = self.__addint32__(varvalue)
            varbyte = bbyte(self.endian, "int32", varvalue, binvalue)
        self.ibytedict[i] = varbyte
        return self.ibytedict[i]


    def modifyuint32(self, iname:str|int="", varvalue:int|list=0):
        i = self.__find__iname__(iname)
        if isinstance(varvalue, list):
            binvalue = b""
            for value in self.unpack_list_depth2(varvalue): binvalue += self.__adduint32__(value)
            varbyte = bbyte(self.endian, "uint32", varvalue, binvalue)
        else:
            binvalue = self.__adduint32__(varvalue)
            varbyte = bbyte(self.endian, "uint32", varvalue, binvalue)
        self.ibytedict[i] = varbyte
        return self.ibytedict[i]


    def modifyint64(self, iname:str|int="", varvalue:int|list=0):
        i = self.__find__iname__(iname)
        if isinstance(varvalue, list):
            binvalue = b""
            for value in self.unpack_list_depth2(varvalue): binvalue += self.__addint64__(value)
            varbyte = bbyte(self.endian, "int64list", varvalue, binvalue)
        else:
            binvalue = self.__addint64__(varvalue)
            varbyte = bbyte(self.endian, "int64", varvalue, binvalue)
        self.ibytedict[i] = varbyte
        return self.ibytedict[i]


    def modifyuint64(self, iname:str|int="", varvalue:int|list=0):
        i = self.__find__iname__(iname)
        if isinstance(varvalue, list):
            binvalue = b""
            for value in self.unpack_list_depth2(varvalue): binvalue += self.__adduint64__(value)
            varbyte = bbyte(self.endian, "uint64list", varvalue, binvalue)
        else:
            binvalue = self.__adduint64__(varvalue)
            varbyte = bbyte(self.endian, "uint64", varvalue, binvalue)
        self.ibytedict[i] = varbyte
        return self.ibytedict[i]


    def modifyslice0b(self, iname:str|int="", varbyte:bytes=b""):
        i = self.__find__iname__(iname)
        self.ibytedict[i] = varbyte
        return self.ibytedict[i]


    def modifynode(self, iname:str|int="", varnode=None):
        if not isinstance(varnode, bnode): return ""
        i = self.__find__iname__(iname)
        self.ibytedict[i] = varnode
        return self.ibytedict[i]


    def modifychar(self, iname:str="", varchars:str=""):
        i = self.__find__iname__(iname)
        varbyte = varchars.encode("ASCII")
        self.ibytedict[i] = bchar(self.endian, "ascii", varchars, varbyte)
        return self.ibytedict[i]


    def modifyutf8(self, iname:str="", varchars:str=""):
        i = self.__find__iname__(iname)
        varbyte = varchars.encode("utf-8")
        self.ibytedict[i] = bchar(self.endian, "utf8", varchars, varbyte)
        return self.ibytedict[i]


    def modifygbk(self, iname:str="", varchars:str=""):
        i = self.__find__iname__(iname)
        varbyte = varchars.encode("GBK")
        self.ibytedict[i] = bchar(self.endian, "gbk", varchars, varbyte)
        return self.ibytedict[i]


    def modifyname(self, iname:str="", varname:str=""):
        i = self.__find__iname__(iname)
        self.inamedict[i] = varname
        return self.ibytedict[i]


    def modifynumu8char(self, iname:str="", varchars:str=""):
        binchars = varchars.encode("ASCII")
        size = len(binchars)
        if self.endian == ">":
            binsize = size.to_bytes(1, byteorder="big", signed=False)
        else:
            binsize = size.to_bytes(1, byteorder="little", signed=False)
        i = self.__find__iname__(iname)
        varbyte = binsize + binchars
        self.ibytedict[i] = bchar(self.endian, "ascii", varchars, varbyte)
        return self.ibytedict[i]


    def modifynumu16char(self, iname:str="", varchars:str=""):
        binchars = varchars.encode("ASCII")
        size = len(binchars)
        if self.endian == ">":
            binsize = size.to_bytes(2, byteorder="big", signed=False)
        else:
            binsize = size.to_bytes(2, byteorder="little", signed=False)
        i = self.__find__iname__(iname)
        varbyte = binsize + binchars
        self.ibytedict[i] = bchar(self.endian, "ascii", varchars, varbyte)
        return self.ibytedict[i]


    def modifynumu32char(self, iname:str="", varchars:str=""):
        binchars = varchars.encode("ASCII")
        size = len(binchars)
        if self.endian == ">":
            binsize = size.to_bytes(4, byteorder="big", signed=False)
        else:
            binsize = size.to_bytes(4, byteorder="little", signed=False)
        i = self.__find__iname__(iname)
        varbyte = binsize + binchars
        self.ibytedict[i] = bchar(self.endian, "ascii", varchars, varbyte)
        return self.ibytedict[i]


    def modifynumu32utf8(self, iname:str="", varchars:str=""):
        binchars = varchars.encode("utf-8")
        size = len(binchars)
        if self.endian == ">":
            binsize = size.to_bytes(4, byteorder="big", signed=False)
        else:
            binsize = size.to_bytes(4, byteorder="little", signed=False)
        i = self.__find__iname__(iname)
        varbyte = binsize + binchars
        self.ibytedict[i] = bchar(self.endian, "utf8", varchars, varbyte)
        return self.ibytedict[i]


    def modifynumu32gbk(self, iname:str="", varchars:str=""):
        binchars = varchars.encode("gbk")
        size = len(binchars)
        if self.endian == ">":
            binsize = size.to_bytes(4, byteorder="big", signed=False)
        else:
            binsize = size.to_bytes(4, byteorder="little", signed=False)
        i = self.__find__iname__(iname)
        varbyte = binsize + binchars
        self.ibytedict[i] = bchar(self.endian, "gbk", varchars, varbyte)
        return self.ibytedict[i]


    # 修改变量名称，变量值
    def modifynameint8(self, iname:str|int="", varname:str="", varvalue:int|list=0):
        i = self.__find__iname__(iname)
        if isinstance(varvalue, list):
            binvalue = b""
            for value in self.unpack_list_depth2(varvalue): binvalue += self.__addint8__(value)
            varbyte = bbyte(self.endian, "int8list", varvalue, binvalue)
        else:
            binvalue = self.__addint8__(varvalue)
            varbyte = bbyte(self.endian, "int8", varvalue, binvalue)
        self.ibytedict[i] = varbyte
        self.inamedict[i] = varname
        return self.ibytedict[i]


    def modifynameuint8(self, iname:str|int="", varname:str="", varvalue:int|list=0):
        i = self.__find__iname__(iname)
        if isinstance(varvalue, list):
            binvalue = b""
            for value in self.unpack_list_depth2(varvalue): binvalue += self.__adduint8__(value)
            varbyte = bbyte(self.endian, "uint8list", varvalue, binvalue)
        else:
            binvalue = self.__adduint8__(varvalue)
            varbyte = bbyte(self.endian, "uint8", varvalue, binvalue)
        self.ibytedict[i] = varbyte
        self.inamedict[i] = varname
        return self.ibytedict[i]


    def modifynameint16(self, iname:str|int="", varname:str="", varvalue:int|list=0):
        i = self.__find__iname__(iname)
        if isinstance(varvalue, list):
            binvalue = b""
            for value in self.unpack_list_depth2(varvalue): binvalue += self.__addint16__(value)
            varbyte = bbyte(self.endian, "int16list", varvalue, binvalue)
        else:
            binvalue = self.__addint16__(varvalue)
            varbyte = bbyte(self.endian, "int16", varvalue, binvalue)
        self.ibytedict[i] = varbyte
        self.inamedict[i] = varname
        return self.ibytedict[i]


    def modifynameuint16(self, iname:str|int="", varname:str="", varvalue:int|list=0):
        i = self.__find__iname__(iname)
        if isinstance(varvalue, list):
            binvalue = b""
            for value in self.unpack_list_depth2(varvalue): binvalue += self.__adduint16__(value)
            varbyte = bbyte(self.endian, "uint16list", varvalue, binvalue)
        else:
            binvalue = self.__adduint16__(varvalue)
            varbyte = bbyte(self.endian, "uint16", varvalue, binvalue)
        self.ibytedict[i] = varbyte
        self.inamedict[i] = varname
        return self.ibytedict[i]


    def modifynameint32(self, iname:str|int="", varname:str="", varvalue:int|list=0):
        i = self.__find__iname__(iname)
        if isinstance(varvalue, list):
            binvalue = b""
            for value in self.unpack_list_depth2(varvalue): binvalue += self.__addint32__(value)
            varbyte = bbyte(self.endian, "int32list", varvalue, binvalue)
        else:
            binvalue = self.__addint32__(varvalue)
            varbyte = bbyte(self.endian, "int32", varvalue, binvalue)
        self.ibytedict[i] = varbyte
        self.inamedict[i] = varname
        return self.ibytedict[i]


    def modifynameuint32(self, iname:str|int="", varname:str="", varvalue:int|list=0):
        i = self.__find__iname__(iname)
        if isinstance(varvalue, list):
            binvalue = b""
            for value in self.unpack_list_depth2(varvalue): binvalue += self.__adduint32__(value)
            varbyte = bbyte(self.endian, "uint32list", varvalue, binvalue)
        else:
            binvalue = self.__adduint32__(varvalue)
            varbyte = bbyte(self.endian, "uint32", varvalue, binvalue)
        self.ibytedict[i] = varbyte
        self.inamedict[i] = varname
        return self.ibytedict[i]


    def modifyint64(self, iname:str|int="", varname:str="", varvalue:int|list=0):
        i = self.__find__iname__(iname)
        if isinstance(varvalue, list):
            binvalue = b""
            for value in self.unpack_list_depth2(varvalue): binvalue += self.__addint64__(value)
            varbyte = bbyte(self.endian, "int64list", varvalue, binvalue)
        else:
            binvalue = self.__addint64__(varvalue)
            varbyte = bbyte(self.endian, "int64", varvalue, binvalue)
        self.ibytedict[i] = varbyte
        self.inamedict[i] = varname
        return self.ibytedict[i]


    def modifynameuint64(self, iname:str|int="", varname:str="", varvalue:int|list=0):
        i = self.__find__iname__(iname)
        if isinstance(varvalue, list):
            binvalue = b""
            for value in self.unpack_list_depth2(varvalue): binvalue += self.__adduint64__(value)
            varbyte = bbyte(self.endian, "uint64list", varvalue, binvalue)
        else:
            binvalue = self.__adduint64__(varvalue)
            varbyte = bbyte(self.endian, "uint64", varvalue, binvalue)
        self.ibytedict[i] = varbyte
        self.inamedict[i] = varname
        return self.ibytedict[i]
    

    def modifynameslice0b(self, iname:str|int="", varname:str="", varvalue:bytes=b""):
        i = self.__find__iname__(iname)
        self.ibytedict[i] = varvalue
        self.inamedict[i] = varname
        return self.ibytedict[i]


    def modifynamenode(self, iname:str|int="", varname:str="", varnode=None):
        if not isinstance(varnode, bnode): return ""
        i = self.__find__iname__(iname)
        self.ibytedict[i] = varnode
        self.inamedict[i] = varname
        return self.ibytedict[i]


    def modifynamechar(self, iname:str="", varname:str="", varchars:str=""):
        i = self.__find__iname__(iname)
        varbyte = varchars.encode("ASCII")
        self.ibytedict[i] = bchar(self.endian, "ascii", varchars, varbyte)
        self.inamedict[i] = varname
        return self.ibytedict[i]


    def modifynameutf8(self, iname:str="", varname:str="", varchars:str=""):
        i = self.__find__iname__(iname)
        varbyte = varchars.encode("utf-8")
        self.ibytedict[i] = bchar(self.endian, "utf8", varchars, varbyte)
        self.inamedict[i] = varname
        return self.ibytedict[i]


    def modifynamegbk(self, iname:str="", varname:str="", varchars:str=""):
        i = self.__find__iname__(iname)
        varbyte = varchars.encode("GBK")
        self.ibytedict[i] = bchar(self.endian, "gbk", varchars, varbyte)
        self.inamedict[i] = varname
        return self.ibytedict[i]


    def modifynamenumu8char(self, iname:str="", varname:str="", varchars:str=""):
        binchars = varchars.encode("ASCII")
        size = len(binchars)
        if self.endian == ">":
            binsize = size.to_bytes(1, byteorder="big", signed=False)
        else:
            binsize = size.to_bytes(1, byteorder="little", signed=False)
        i = self.__find__iname__(iname)
        varbyte = binsize + binchars
        self.ibytedict[i] = bchar(self.endian, "ascii", varchars, varbyte)
        self.inamedict[i] = varname
        return self.ibytedict[i]


    def modifynamenumu16char(self, iname:str="", varname:str="", varchars:str=""):
        binchars = varchars.encode("ASCII")
        size = len(binchars)
        if self.endian == ">":
            binsize = size.to_bytes(2, byteorder="big", signed=False)
        else:
            binsize = size.to_bytes(2, byteorder="little", signed=False)
        i = self.__find__iname__(iname)
        varbyte = binsize + binchars
        self.ibytedict[i] = bchar(self.endian, "ascii", varchars, varbyte)
        self.inamedict[i] = varname
        return self.ibytedict[i]


    def modifynamenumu32char(self, iname:str="", varname:str="", varchars:str=""):
        binchars = varchars.encode("ASCII")
        size = len(binchars)
        if self.endian == ">":
            binsize = size.to_bytes(4, byteorder="big", signed=False)
        else:
            binsize = size.to_bytes(4, byteorder="little", signed=False)
        i = self.__find__iname__(iname)
        varbyte = binsize + binchars
        self.ibytedict[i] = bchar(self.endian, "ascii", varchars, varbyte)
        self.inamedict[i] = varname
        return self.ibytedict[i]


    def modifynamenumu32utf8(self, iname:str="", varname:str="", varchars:str=""):
        binchars = varchars.encode("utf-8")
        size = len(binchars)
        if self.endian == ">":
            binsize = size.to_bytes(4, byteorder="big", signed=False)
        else:
            binsize = size.to_bytes(4, byteorder="little", signed=False)
        i = self.__find__iname__(iname)
        varbyte = binsize + binchars
        self.ibytedict[i] = bchar(self.endian, "utf8", varchars, varbyte)
        self.inamedict[i] = varname
        return self.ibytedict[i]


    def modifynamenumu32gbk(self, iname:str="", varname:str="", varchars:str=""):
        binchars = varchars.encode("gbk")
        size = len(binchars)
        if self.endian == ">":
            binsize = size.to_bytes(4, byteorder="big", signed=False)
        else:
            binsize = size.to_bytes(4, byteorder="little", signed=False)
        i = self.__find__iname__(iname)
        varbyte = binsize + binchars
        self.ibytedict[i] = bchar(self.endian, "gbk", varchars, varbyte)
        self.inamedict[i] = varname
        return self.ibytedict[i]


    def insert(self, ):
        pass

    def move(self,):
        pass

    def delete(self, *args, **kwargs):
        pass

    def pop(self, ):
        pass


    def remove(self, ):
        pass


    def lists(self):
        namevaluelist = []
        for [i, name], [i, value] in zip(self.inamedict.items(), self.ibytedict.items()): namevaluelist.append([name, value])
        return namevaluelist


    def items(self):
        namevaluedict = {}
        for [i, name], [i, value] in zip(self.inamedict.items(), self.ibytedict.items()): namevaluedict[name] = value
        return namevaluedict.items()


    def unpack_list_depth1(self, values:list=[]):
        newvalues = []
        for value in values: newvalues.append(value)
        # 类型检查
        if newvalues == []: return []
        for value in newvalues:
            if not isinstance(value, type(newvalues[0])): raise ValueError(f"列表里的元素类型不一致，请检查列表里元素类型...")
        return newvalues



    def unpack_list_depth2(self, values:list=[]):
        newvalues = []
        for value in values:
            if isinstance(value, list):
                for o in value: newvalues.append(o)
            else:
                newvalues.append(value)

        # 类型检查
        if newvalues == []: return []
        for value in newvalues:
            if not isinstance(value, type(newvalues[0])): raise ValueError(f"列表里的元素类型不一致，请检查列表里元素类型...")
        return newvalues
    

    def __split__bytes__(self, step, varbyte):
        length = len(varbyte)
        numint = length // step
        values = [varbyte[i*step: i+step] for i in range(numint)]
        return values


    def __convert_bytes2int__(self, varbyte, signed):
        if self.endian == ">": 
            return int.from_bytes(varbyte, byteorder="big", signed=signed)
        else: 
            return int.from_bytes(varbyte, byteorder="little", signed=signed)


    def readint80b(self, varbyte:bytes):
        numbyte, signed = 1, True
        values = self.__split__bytes__(numbyte, varbyte)
        ints = [self.__convert_bytes2int__(value, signed) for value in values]
        if len(ints) == 1:
            return ints[0]
        else:
            return ints


    def readuint80b(self, varbyte:bytes):
        numbyte, signed = 1, False
        values = self.__split__bytes__(numbyte, varbyte)
        ints = [self.__convert_bytes2int__(value, signed) for value in values]
        if len(ints) == 1:
            return ints[0]
        else:
            return ints

    def readint160b(self, varbyte:bytes):
        numbyte, signed = 2, True
        values = self.__split__bytes__(numbyte, varbyte)
        ints = [self.__convert_bytes2int__(value, signed) for value in values]
        if len(ints) == 1:
            return ints[0]
        else:
            return ints


    def readuint160b(self, varbyte:bytes):
        numbyte, signed = 2, False
        values = self.__split__bytes__(numbyte, varbyte)
        ints = [self.__convert_bytes2int__(value, signed) for value in values]
        if len(ints) == 1:
            return ints[0]
        else:
            return ints
            
    def readint320b(self, varbyte:bytes):
        numbyte, signed = 4, True
        values = self.__split__bytes__(numbyte, varbyte)
        ints = [self.__convert_bytes2int__(value, signed) for value in values]
        if len(ints) == 1:
            return ints[0]
        else:
            return ints


    def readuint320b(self, varbyte:bytes):
        numbyte, signed = 4, False
        values = self.__split__bytes__(numbyte, varbyte)
        ints = [self.__convert_bytes2int__(value, signed) for value in values]
        if len(ints) == 1:
            return ints[0]
        else:
            return ints


    def readint640b(self, varbyte:bytes):
        numbyte, signed = 8, True
        values = self.__split__bytes__(numbyte, varbyte)
        ints = [self.__convert_bytes2int__(value, signed) for value in values]
        if len(ints) == 1:
            return ints[0]
        else:
            return ints


    def readuint640b(self, varbyte:bytes):
        numbyte, signed = 8, False
        values = self.__split__bytes__(numbyte, varbyte)
        ints = [self.__convert_bytes2int__(value, signed) for value in values]
        if len(ints) == 1:
            return ints[0]
        else:
            return ints


    def __get__int__(self, iname):
        i = self.__find__iname__(iname)
        varbyte = self.ibytedict[i]
        if isinstance(varbyte, bbyte): return varbyte.data
        raise ValueError(f"输入的索引或变量名称，找到的对象类型为{type(varbyte)}，不是bbyte类型[int|float|list]...") 

    def getint8(self, iname:int|str):
        return self.__get__int__(iname)

    def getuint8(self, iname:int|str):
        return self.__get__int__(iname)
    
    def getint16(self, iname:int|str):
        return self.__get__int__(iname)

    def getuint16(self, iname:int|str):
        return self.__get__int__(iname)
    
    def getint32(self, iname:int|str):
        return self.__get__int__(iname)
    
    def getuint32(self, iname:int|str):
        return self.__get__int__(iname)

    def getint64(self, iname:int|str):
        return self.__get__int__(iname)

    def getuint64(self, iname:int|str):
        return self.__get__int__(iname)
    

    def getslice0b(self, iname:int|str):
        i = self.__find__iname__(iname)
        varbyte = self.ibytedict[i]
        if isinstance(varbyte, bytes): return varbyte
        raise ValueError(f"输入的索引或变量名称，找到的对象类型为{type(varbyte)}，不是bytes类型...") 

    def getnode(self, iname:int|str):
        i = self.__find__iname__(iname)
        varnode = self.ibytedict[i]
        if isinstance(varnode, bnode): return varnode
        raise ValueError(f"输入的索引或变量名称，找到的对象类型为{type(varnode)}，不是bnode类型...") 


    def __get__unicode__(self, iname):
        i = self.__find__iname__(iname)
        varchar = self.ibytedict[i]
        if isinstance(varchar, bchar): return varchar.char
        raise ValueError(f"输入的索引或变量名称，找到的对象类型为{type(varchar)}，不是bchar类型...") 

    def getchar(self, iname:int|str):
        return self.__get__unicode__(iname)

    def getgbk(self, iname:int|str):
        return self.__get__unicode__(iname)

    def getutf8(self, iname:int|str):
        return self.__get__unicode__(iname)



if __name__ == "__main__":
    node = bnode()
    values = [0, 1, [0, 1 ,1.5]]
    node.unpack_list_depth2(values)


