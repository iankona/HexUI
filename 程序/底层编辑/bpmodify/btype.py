
from . import struct_python

class bbyte:
    def __init__(self, name="", endian:str="<"):
        self.name = name
        self.endian = endian
        self.allow = True
        self.type = None
        self.parent = None
        self.slice0b = None



    def __assert__int__(self, varvalue:int|list, function):
        if isinstance(varvalue, int): varvalue = [varvalue]   
        varbyte = b""
        for varvalue in varvalue: varbyte += function(varvalue, self.endian)
        return varbyte
    
    def __assert__float__(self, varvalue:float|list, function):
        if isinstance(varvalue, float): varvalue = [varvalue]   
        varbyte = b""
        for varvalue in varvalue: varbyte += function(varvalue, self.endian)
        return varbyte

    def __assert__byte__(self, varbyte:bytes, step:int, function):
        count = count = len(varbyte) // step
        buffer = []
        for i in range(count): buffer.append( varbyte[i*step: (i+1)*step] )
        varvalue = []
        for bufferstep in buffer: varvalue.append( function(bufferstep, self.endian) )
        if count == 1: return varvalue[0]
        return varvalue



    @property
    def int8(self):
        assert self.type == "int8", f"读取类型: int8 与 实例类型: {self.type}不一致"
        return self.data

    @int8.setter
    def int8(self, varvalue:int|list):
        assert self.type == "int8", f"赋值类型: int8 与 实例类型: {self.type}不一致"
        self.slice0b = self.__assert__int__(varvalue, func=struct_python.int8_2_byte)
        self.data = varvalue

    def int8_fromint(self, varvalue:int|list):
        assert self.allow, f"请通过self.int8属性重新赋值"
        self.slice0b = self.__assert__int__(varvalue, func=struct_python.int8_2_byte)
        self.type = "int8"
        self.data = varvalue
        self.allow = False
        return self
    
    def int8_frombyte(self, varbyte:bytes):
        assert self.allow, f"请通过self.int8属性重新赋值"
        self.data = self.__assert__byte__(varbyte, step=1, func=struct_python.byte_2_int8)
        self.type = "int8"
        self.slice0b = varbyte 
        self.allow = False
        return self
    


    @property
    def int16(self):
        assert self.type == "int16", f"读取类型: int16 与 实例类型: {self.type}不一致"
        return self.data

    @int16.setter
    def int16(self, varvalue:int|list):
        assert self.type == "int16", f"赋值类型: int16 与 实例类型: {self.type}不一致"
        self.slice0b = self.__assert__int__(varvalue, func=struct_python.int16_2_byte)
        self.data = varvalue

    def int16_fromint(self, varvalue:int|list):
        assert self.allow, f"请通过self.int16属性重新赋值"
        self.slice0b = self.__assert__int__(varvalue, func=struct_python.int16_2_byte)
        self.type = "int16"
        self.data = varvalue
        self.allow = False
        return self
    
    def int16_frombyte(self, varbyte:bytes):
        assert self.allow, f"请通过self.int16属性重新赋值"
        self.data = self.__assert__byte__(varbyte, step=2, func=struct_python.byte_2_int16)
        self.type = "int16"
        self.slice0b = varbyte 
        self.allow = False
        return self
    


    @property
    def int32(self):
        assert self.type == "int32", f"读取类型: int32 与 实例类型: {self.type}不一致"
        return self.data

    @int32.setter
    def int32(self, varvalue:int|list):
        assert self.type == "int32", f"赋值类型: int32 与 实例类型: {self.type}不一致"
        self.slice0b = self.__assert__int__(varvalue, func=struct_python.int32_2_byte)
        self.data = varvalue

    def int32_fromint(self, varvalue:int|list):
        assert self.allow, f"请通过self.int32属性重新赋值"
        self.slice0b = self.__assert__int__(varvalue, func=struct_python.int32_2_byte)
        self.type = "int32"
        self.data = varvalue
        self.allow = False
        return self
    
    def int32_frombyte(self, varbyte:bytes):
        assert self.allow, f"请通过self.int32属性重新赋值"
        self.data = self.__assert__byte__(varbyte, step=4, func=struct_python.byte_2_int32)
        self.type = "int32"
        self.slice0b = varbyte 
        self.allow = False
        return self
    


    @property
    def int64(self):
        assert self.type == "int64", f"读取类型: int64 与 实例类型: {self.type}不一致"
        return self.data

    @int64.setter
    def int64(self, varvalue:int|list):
        assert self.type == "int64", f"赋值类型: int64 与 实例类型: {self.type}不一致"
        self.slice0b = self.__assert__int__(varvalue, func=struct_python.int64_2_byte)
        self.data = varvalue

    def int64_fromint(self, varvalue:int|list):
        assert self.allow, f"请通过self.int64属性重新赋值"
        self.slice0b = self.__assert__int__(varvalue, func=struct_python.int64_2_byte)
        self.type = "int64"
        self.data = varvalue
        self.allow = False
        return self
    
    def int64_frombyte(self, varbyte:bytes):
        assert self.allow, f"请通过self.int64属性重新赋值"
        self.data = self.__assert__byte__(varbyte, step=8, func=struct_python.byte_2_int64)
        self.type = "int64"
        self.slice0b = varbyte 
        self.allow = False
        return self



    @property
    def uint8(self):
        assert self.type == "uint8", f"读取类型: uint8 与 实例类型: {self.type}不一致"
        return self.data

    @uint8.setter
    def uint8(self, varvalue:int|list):
        assert self.type == "uint8", f"赋值类型: uint8 与 实例类型: {self.type}不一致"
        self.slice0b = self.__assert__int__(varvalue, func=struct_python.uint8_2_byte)
        self.data = varvalue

    def uint8_fromint(self, varvalue:int|list):
        assert self.allow, f"请通过self.uint8属性重新赋值"
        self.slice0b = self.__assert__int__(varvalue, func=struct_python.uint8_2_byte)
        self.type = "uint8"
        self.data = varvalue
        self.allow = False
        return self
    
    def uint8_frombyte(self, varbyte:bytes):
        assert self.allow, f"请通过self.uint8属性重新赋值"
        self.data = self.__assert__byte__(varbyte, step=1, func=struct_python.byte_2_uint8)
        self.type = "uint8"
        self.slice0b = varbyte 
        self.allow = False
        return self



    @property
    def uint16(self):
        assert self.type == "uint16", f"读取类型: uint16 与 实例类型: {self.type}不一致"
        return self.data

    @uint16.setter
    def uint16(self, varvalue:int|list):
        assert self.type == "uint16", f"赋值类型: uint16 与 实例类型: {self.type}不一致"
        self.slice0b = self.__assert__int__(varvalue, func=struct_python.uint16_2_byte)
        self.data = varvalue

    def uint16_fromint(self, varvalue:int|list):
        assert self.allow, f"请通过self.uint16属性重新赋值"
        self.slice0b = self.__assert__int__(varvalue, func=struct_python.uint16_2_byte)
        self.type = "uint16"
        self.data = varvalue
        self.allow = False
        return self
    
    def uint16_frombyte(self, varbyte:bytes):
        assert self.allow, f"请通过self.uint16属性重新赋值"
        self.data = self.__assert__byte__(varbyte, step=2, func=struct_python.byte_2_uint16)
        self.type = "uint16"
        self.slice0b = varbyte 
        self.allow = False
        return self



    @property
    def uint32(self):
        assert self.type == "uint32", f"读取类型: uint32 与 实例类型: {self.type}不一致"
        return self.data

    @uint32.setter
    def uint32(self, varvalue:int|list):
        assert self.type == "uint32", f"赋值类型: uint32 与 实例类型: {self.type}不一致"
        self.slice0b = self.__assert__int__(varvalue, func=struct_python.uint32_2_byte)
        self.data = varvalue

    def uint32_fromint(self, varvalue:int|list):
        assert self.allow, f"请通过self.uint32属性重新赋值"
        self.slice0b = self.__assert__int__(varvalue, func=struct_python.uint32_2_byte)
        self.type = "uint32"
        self.data = varvalue
        self.allow = False
        return self
    
    def uint32_frombyte(self, varbyte:bytes):
        assert self.allow, f"请通过self.uint32属性重新赋值"
        self.data = self.__assert__byte__(varbyte, step=4, func=struct_python.byte_2_uint32)
        self.type = "uint32"
        self.slice0b = varbyte 
        self.allow = False
        return self



    @property
    def uint64(self):
        assert self.type == "uint64", f"读取类型: uint64 与 实例类型: {self.type}不一致"
        return self.data

    @uint64.setter
    def uint64(self, varvalue:int|list):
        assert self.type == "uint64", f"赋值类型: uint64 与 实例类型: {self.type}不一致"
        self.slice0b = self.__assert__int__(varvalue, func=struct_python.uint64_2_byte)
        self.data = varvalue

    def uint64_fromint(self, varvalue:int|list):
        assert self.allow, f"请通过self.uint64属性重新赋值"
        self.slice0b = self.__assert__int__(varvalue, func=struct_python.uint64_2_byte)
        self.type = "uint64"
        self.data = varvalue
        self.allow = False
        return self
    
    def uint64_frombyte(self, varbyte:bytes):
        assert self.allow, f"请通过self.uint64属性重新赋值"
        self.data = self.__assert__byte__(varbyte, step=8, func=struct_python.byte_2_uint64)
        self.type = "uint64"
        self.slice0b = varbyte 
        self.allow = False
        return self


    @property
    def float16(self):
        assert self.type == "float16", f"读取类型: float16 与 实例类型: {self.type}不一致"
        return self.data

    @float16.setter
    def float16(self, varvalue:float|list):
        assert self.type == "float16", f"赋值类型: float16 与 实例类型: {self.type}不一致"
        self.slice0b = self.__assert__float__(varvalue, func=struct_python.float16_2_byte)
        self.data = varvalue

    def float16_fromint(self, varvalue:float|list):
        assert self.allow, f"请通过self.float16属性重新赋值"
        self.slice0b = self.__assert__float__(varvalue, func=struct_python.float16_2_byte)
        self.type = "float16"
        self.data = varvalue
        self.allow = False
        return self
    
    def float16_frombyte(self, varbyte:bytes):
        assert self.allow, f"请通过self.float16属性重新赋值"
        self.data = self.__assert__byte__(varbyte, step=2, func=struct_python.byte_2_float16)
        self.type = "float16"
        self.slice0b = varbyte 
        self.allow = False
        return self
    


    @property
    def float32(self):
        assert self.type == "float32", f"读取类型: float32 与 实例类型: {self.type}不一致"
        return self.data

    @float32.setter
    def float32(self, varvalue:float|list):
        assert self.type == "float32", f"赋值类型: float32 与 实例类型: {self.type}不一致"
        self.slice0b = self.__assert__float__(varvalue, func=struct_python.float32_2_byte)
        self.data = varvalue

    def float32_fromint(self, varvalue:float|list):
        assert self.allow, f"请通过self.float32属性重新赋值"
        self.slice0b = self.__assert__float__(varvalue, func=struct_python.float32_2_byte)
        self.type = "float32"
        self.data = varvalue
        self.allow = False
        return self
    
    def float32_frombyte(self, varbyte:bytes):
        assert self.allow, f"请通过self.float32属性重新赋值"
        self.data = self.__assert__byte__(varbyte, step=4, func=struct_python.byte_2_float32)
        self.type = "float32"
        self.slice0b = varbyte 
        self.allow = False
        return self
    

    @property
    def float64(self):
        assert self.type == "float64", f"读取类型: float64 与 实例类型: {self.type}不一致"
        return self.data

    @float64.setter
    def float64(self, varvalue:float|list):
        assert self.type == "float64", f"赋值类型: float64 与 实例类型: {self.type}不一致"
        self.slice0b = self.__assert__float__(varvalue, func=struct_python.float64_2_byte)
        self.data = varvalue

    def float64_fromint(self, varvalue:float|list):
        assert self.allow, f"请通过self.float64属性重新赋值"
        self.slice0b = self.__assert__float__(varvalue, func=struct_python.float64_2_byte)
        self.type = "float64"
        self.data = varvalue
        self.allow = False
        return self
    
    def float64_frombyte(self, varbyte:bytes):
        assert self.allow, f"请通过self.float64属性重新赋值"
        self.data = self.__assert__byte__(varbyte, step=8, func=struct_python.byte_2_float64)
        self.type = "float64"
        self.slice0b = varbyte 
        self.allow = False
        return self



    @property
    def ascii(self):
        assert self.type == "ascii", f"读取类型: ascii 与 实例类型: {self.type}不一致"
        return self.data

    @ascii.setter
    def ascii(self, char:str=""):
        assert self.type == "ascii", f"赋值类型: ascii 与 实例类型: {self.type}不一致"
        self.slice0b = char.encode(encoding="ascii")
        self.name = char
        self.data = char

    def ascii_fromstr(self, char:str=""):
        assert self.allow, f"请通过self.ascii属性重新赋值"
        self.slice0b = char.encode(encoding="ascii")
        self.type = "ascii"
        self.name = char
        self.data = char
        return self
    
    def ascii_frombyte(self, strbyte:bytes):
        assert self.allow, f"请通过self.ascii属性重新赋值"
        self.data = strbyte.decode(encoding="ascii")
        self.type = "ascii"
        self.name = self.data
        self.slice0b = strbyte
        return self



    @property
    def utf8(self):
        assert self.type == "utf8", f"读取类型: utf8 与 实例类型: {self.type}不一致"
        return self.data

    @utf8.setter
    def utf8(self, char:str=""):
        assert self.type == "utf8", f"赋值类型: utf8 与 实例类型: {self.type}不一致"
        self.slice0b = char.encode(encoding="utf8")
        self.name = char
        self.data = char

    def utf8_fromstr(self, char:str=""):
        assert self.allow, f"请通过self.utf8属性重新赋值"
        self.slice0b = char.encode(encoding="utf8")
        self.type = "utf8"
        self.name = char
        self.data = char
        return self
    
    def utf8_frombyte(self, strbyte:bytes):
        assert self.allow, f"请通过self.utf8属性重新赋值"
        self.data = strbyte.decode(encoding="utf8")
        self.type = "utf8"
        self.name = self.data
        self.slice0b = strbyte
        return self


    @property
    def gbk(self):
        assert self.type == "gbk", f"读取类型: gbk 与 实例类型: {self.type}不一致"
        return self.data

    @gbk.setter
    def gbk(self, char:str=""):
        assert self.type == "gbk", f"赋值类型: gbk 与 实例类型: {self.type}不一致"
        self.slice0b = char.encode(encoding="gbk")
        self.name = char
        self.data = char

    def gbk_fromstr(self, char:str=""):
        assert self.allow, f"请通过self.gbk属性重新赋值"
        self.slice0b = char.encode(encoding="gbk")
        self.type = "gbk"
        self.name = char
        self.data = char
        return self
    
    def gbk_frombyte(self, strbyte:bytes):
        assert self.allow, f"请通过self.gbk属性重新赋值"
        self.data = strbyte.decode(encoding="gbk")
        self.type = "gbk"
        self.name = self.data
        self.slice0b = strbyte
        return self




class bchar:
    def __init__(self, inttype="uint32", strtype="utf8", endian:str="<"):
        self.name = None
        self.parent = None
        self.slice0b = None
        self.inttype = inttype
        self.strtype = strtype
        self.intbnode = bbyte(endian)
        self.strbnode = bbyte(endian)



    @property
    def string(self):
        match self.strtype:
            case "gbk"  : return self.gbk
            case "utf8" : return self.utf8
            case "ascii": return self.ascii
            case _ : raise ValueError(f"bchar:: can't support {self.strtype}, must be in {['ascii', 'utf8', 'gbk']}")

    @string.setter
    def string(self, char:str):
        match self.strtype:
            case "gbk"  : self.gbk = char
            case "utf8" : self.utf8 = char
            case "ascii": self.ascii = char
            case _ : raise ValueError(f"bchar:: can't support {self.strtype}, must be in {['ascii', 'utf8', 'gbk']}")

    def fromstring(self, char:str):
        match self.strtype:
            case "gbk"  : self.gbk_fromstring(char)
            case "utf8" : self.utf8_fromstring(char)
            case "ascii": self.ascii_fromstring(char)
            case _ : raise ValueError(f"bchar:: can't support {self.strtype}, must be in {['ascii', 'utf8', 'gbk']}")
        return self

    def fromslice0b(self, slice0b:bytes):
        match self.strtype:
            case "gbk"  : self.gbk_fromslice0b(slice0b)
            case "utf8" : self.utf8_fromslice0b(slice0b)
            case "ascii": self.ascii_fromslice0b(slice0b)
            case _ : raise ValueError(f"bchar:: can't support {self.strtype}, must be in {['ascii', 'utf8', 'gbk']}")
        return self



    def __set__varvalue__(self, varvalue):
        match self.inttype:
            case "uint8" : self.intbnode.uint8_fromint(varvalue)
            case "uint16": self.intbnode.uint16_fromint(varvalue)
            case "uint32": self.intbnode.uint32_fromint(varvalue)
            case _ : raise ValueError(f"bchar:: can't support {self.inttype}, must be in {['uint8', 'uint16', 'uint32']}")

    def __set__varbyte__(self, varbyte):
        match self.inttype:
            case "uint8" : self.intbnode.uint8_frombyte(varbyte)
            case "uint16": self.intbnode.uint16_frombyte(varbyte)
            case "uint32": self.intbnode.uint32_frombyte(varbyte)
            case _ : raise ValueError(f"bchar:: can't support {self.inttype}, must be in {['uint8', 'uint16', 'uint32']}")

    def __split__slice0b__(self, slice0b):
        match self.inttype:
            case "uint8" : return slice0b[0:1], slice0b[1:]
            case "uint16": return slice0b[0:2], slice0b[2:]
            case "uint32": return slice0b[0:4], slice0b[4:]
            case _ : raise ValueError(f"bchar:: can't support {self.inttype}, must be in {['uint8', 'uint16', 'uint32']}")



    @property
    def ascii(self):
        return self.strbnode.ascii

    @ascii.setter
    def ascii(self, char:str=""):
        self.strbnode.ascii = char
        self.__set__varvalue__(len(self.strbnode.slice0b))
        self.name = char
        self.slice0b = self.intbnode.slice0b + self.strbnode.slice0b

    def ascii_fromstring(self, char:str=""):
        self.strbnode.ascii_fromstr(char)
        self.__set__varvalue__(len(self.strbnode.slice0b))
        self.name = self.strbnode.ascii
        self.slice0b = self.intbnode.slice0b + self.strbnode.slice0b
        return self

    def ascii_fromslice0b(self, slice0b):
        varbyte, strbyte = self.__split__slice0b__(slice0b)
        self.__set__varbyte__(varbyte)
        self.strbnode.ascii_frombyte(strbyte)
        self.name = self.strbnode.ascii
        self.slice0b = slice0b
        return self



    @property
    def utf8(self):
        return self.strbnode.utf8
    
    @utf8.setter
    def utf8(self, char:str=""):
        self.strbnode.utf8 = char
        self.__set__varvalue__(len(self.strbnode.slice0b))
        self.name = char
        self.slice0b = self.intbnode.slice0b + self.strbnode.slice0b

    def utf8_fromstring(self, char:str=""):
        self.strbnode.utf8_fromstr(char)
        self.__set__varvalue__(len(self.strbnode.slice0b))
        self.name = char
        self.slice0b = self.intbnode.slice0b + self.strbnode.slice0b
        return self

    def utf8_fromslice0b(self, slice0b):
        varbyte, strbyte = self.__split__slice0b__(slice0b)
        self.__set__varbyte__(varbyte)
        self.strbnode.utf8_frombyte(strbyte)
        self.name = self.strbnode.utf8
        self.slice0b = slice0b
        return self



    @property
    def gbk(self):
        return self.strbnode.gbk

    @gbk.setter
    def gbk(self, char:str=""):
        self.strbnode.gbk = char
        self.__set__varvalue__(len(self.strbnode.slice0b))
        self.name = char
        self.slice0b = self.intbnode.slice0b + self.strbnode.slice0b

    def gbk_fromstring(self, char:str=""):
        self.strbnode.gbk_fromstr(char)
        self.__set__varvalue__(len(self.strbnode.slice0b))
        self.name = char
        self.slice0b = self.intbnode.slice0b + self.strbnode.slice0b
        return self

    def gbk_fromslice0b(self, slice0b):
        varbyte, strbyte = self.__split__slice0b__(slice0b)
        self.__set__varbyte__(varbyte)
        self.strbnode.gbk_frombyte(strbyte)
        self.name = self.strbnode.gbk
        self.slice0b = slice0b
        return self






