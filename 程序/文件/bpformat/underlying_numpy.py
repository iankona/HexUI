import os
import numpy as np

class 类:
    def __init__(self, mpbyte=None, endian="<"):
        self.mpbyte = mpbyte
        self.endian = endian


    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        return ""

    def __calc_size__(self, size):
        left, right = self.index, self.index+size
        if right > self.mpright: right = self.mpright
        self.index = right
        return left, right

    def __calc_size_no_seek__(self, size):
        left, right = self.index, self.index + size
        if right > self.mpright: right = self.mpright
        return left, right


    def left(self):
        return self.mpleft

    def right(self):
        return self.mpright

    def tell(self):
        return self.index

    def slicetell(self):
        return self.index - self.mpleft

    def sliceEOF(self):
        if self.index >= self.mpright:
            return True
        else:
            return False

    def size(self):
        return self.mpright - self.mpleft

    def remainsize(self):
        return self.mpright - self.index

    def seek(self, size):
        self.__calc_size__(size)


    def changeleft(self, offset):
        if offset < 0: offset = 0
        if offset > self.mpright: offset = self.mpright
        self.mpleft = self.index = offset
        

    def copy(self):
        bp = 类(mpbyte=self.mpbyte, endian=self.endian)
        bp.mpleft, bp.mpright = self.mpleft, self.mpright
        bp.index = bp.mpleft
        return bp
    
    
    def readslice0b(self):
        return self.mpbyte[self.mpleft: self.mpright]
    

    def readslice(self, size):
        left, right = self.__calc_size__(size)
        bp = 类(mpbyte=self.mpbyte, endian=self.endian)
        bp.mpleft, bp.mpright = left, right
        bp.index = bp.mpleft
        return bp

    def readremainslice(self):
        bp = self.readslice(self.remainsize())
        return bp


    def read(self, num=1):
        left, right = self.__calc_size__(num)
        return self.mpbyte[left:right]

    def remainread(self):
        return self.read(self.remainsize())


    def readuint8(self, num=1):
        left, right = self.__calc_size__(num)
        result = np.frombuffer(self.mpbyte[left:right], dtype=self.endian+"u1", count=num)
        if num == 1: return int(result[0])
        return list(result)

    def readuint16(self, num=1):
        left, right = self.__calc_size__(2*num)
        result = np.frombuffer(self.mpbyte[left:right], dtype=self.endian+"u2", count=num)
        if num == 1: return int(result[0])
        return list(result)

    def readuint32(self, num=1):
        left, right = self.__calc_size__(4*num)
        result = np.frombuffer(self.mpbyte[left:right], dtype=self.endian+"u4", count=num)
        if num == 1: return int(result[0])
        return list(result)

    def readuint64(self, num=1):
        left, right = self.__calc_size__(8*num)
        result = np.frombuffer(self.mpbyte[left:right], dtype=self.endian+"u8", count=num)
        if num == 1: return int(result[0])
        return list(result)


    def readint8(self, num=1):
        left, right = self.__calc_size__(num)
        result = np.frombuffer(self.mpbyte[left:right], dtype=self.endian+"i1", count=num)
        if num == 1: return int(result[0])
        return list(result)

    def readint16(self, num=1):
        left, right = self.__calc_size__(2*num)
        result = np.frombuffer(self.mpbyte[left:right], dtype=self.endian+"i2", count=num)
        if num == 1: return int(result[0])
        return list(result)

    def readint32(self, num=1):
        left, right = self.__calc_size__(4*num)
        result = np.frombuffer(self.mpbyte[left:right], dtype=self.endian+"i4", count=num)
        if num == 1: return int(result[0])
        return list(result)

    def readint64(self, num=1):
        left, right = self.__calc_size__(8*num)
        result = np.frombuffer(self.mpbyte[left:right], dtype=self.endian+"i8", count=num)
        if num == 1: return int(result[0])
        return list(result)


    def readfloat16(self, num=1):
        left, right = self.__calc_size__(2*num)
        result = np.frombuffer(self.mpbyte[left:right], dtype=self.endian+"f2", count=num)
        if num == 1: return result[0]
        return list(result)

    def readfloat32(self, num=1):
        left, right = self.__calc_size__(4*num)
        result = np.frombuffer(self.mpbyte[left:right], dtype=self.endian+"f4", count=num)
        if num == 1: return result[0]
        return list(result)

    def readfloat64(self, num=1):
        left, right = self.__calc_size__(8*num)
        result = np.frombuffer(self.mpbyte[left:right], dtype=self.endian+"f8", count=num)
        if num == 1: return result[0]
        return list(result)

    def readu8float32(self, num=1):
        uint8s = self.readuint8(num)
        if num == 1: return uint8s/255.0
        else: return [uint8/255.0 for uint8 in uint8s]

    def readi8float32(self, num=1):
        int8s = self.readint8(num)
        if num == 1: return int8s/128.0
        else: return [int8/128.0 for int8 in int8s]

    def readu16float32(self, num=1):
        uint16s = self.readuint16(num)
        if num == 1: return uint16s/65535.0
        else: return [uint16/65535.0 for uint16 in uint16s]

    def readi16float32(self, num=1):
        int16s = self.readint16(num)
        if num == 1: return int16s/32768.0
        else: return [int16/32768.0 for int16 in int16s]


    def read5u8uint64(self): # 5 uint8 to uint64
        left, right = self.__calc_size__(5)
        if self.endian == "<" : buffer = self.mpbyte[left:right] + b"\x00\x00\x00"
        if self.endian == ">" : buffer = b"\x00\x00\x00" + self.mpbyte[left:right]
        return int(np.frombuffer(buffer, dtype=self.endian+"u8")[0])

    def readbin(self, num=1):
        if num == 1: return bin(self.readuint8(num))
        return [bin(uint8) for uint8 in self.readuint8(num)]
    
    def readhex(self, num=1):
        if num == 1: return "%02X"%self.readuint8(num)
        return ["%02X"%uint8 for uint8 in self.readuint8(num)]

    def readchar(self, num=1):
        if num <= 0: return ""
        if num == 1: return chr(self.readuint8())
        chars = ""
        for uint8 in self.readuint8(num):
            if 31 < uint8 < 128: chars += chr(uint8)
        return chars

    def readgbk(self, bytenum):
        if bytenum <= 0: return ""
        left, right = self.__calc_size__(bytenum)
        try:
            return self.mpbyte[left:right].decode("GBK")
        except: 
            chars = ""
            for uint8 in self.mpbyte[left:right]: chars += chr(uint8)
            return chars


    def readutf8(self, bytenum):
        if bytenum <= 0: return ""
        left, right = self.__calc_size__(bytenum)
        try: 
            return self.mpbyte[left:right].decode("utf-8")
        except: 
            chars = ""
            for uint8 in self.mpbyte[left:right]: chars += chr(uint8)
            return chars


    def readsliceseek0(self, size):
        left, right = self.__calc_size_no_seek__(size)
        bp = 类(mpbyte=self.mpbyte, endian=self.endian)
        bp.mpleft, bp.mpright = left, right
        bp.index = bp.mpleft
        return bp

    def readremainsliceseek0(self):
        bp = self.readsliceseek0(self.remainsize())
        return bp


    def readseek0(self, num=1):
        left, right = self.__calc_size_no_seek__(num)
        return self.mpbyte[left:right]

    def remainreadseek0(self):
        return self.readseek0(self.remainsize())


    def readuint8seek0(self, num=1):
        left, right = self.__calc_size_no_seek__(num)
        result = np.frombuffer(self.mpbyte[left:right], dtype=self.endian+"u1", count=(right-left))
        if num == 1: return int(result[0])
        return list(result)

    def readuint16seek0(self, num=1):
        left, right = self.__calc_size_no_seek__(2*num)
        result = np.frombuffer(self.mpbyte[left:right], dtype=self.endian+"u2", count=(right-left)//2)
        if num == 1: return int(result[0])
        return list(result)

    def readuint32seek0(self, num=1):
        left, right = self.__calc_size_no_seek__(4*num)
        result = np.frombuffer(self.mpbyte[left:right], dtype=self.endian+"u4", count=(right-left)//4)
        if num == 1: return int(result[0])
        return list(result)

    def readuint64seek0(self, num=1):
        left, right = self.__calc_size_no_seek__(8*num)
        result = np.frombuffer(self.mpbyte[left:right], dtype=self.endian+"u8", count=(right-left)//8)
        if num == 1: return int(result[0])
        return list(result)


    def readint8seek0(self, num=1):
        left, right = self.__calc_size_no_seek__(num)
        result = np.frombuffer(self.mpbyte[left:right], dtype=self.endian+"i1", count=(right-left))
        if num == 1: return int(result[0])
        return list(result)

    def readint16seek0(self, num=1):
        left, right = self.__calc_size_no_seek__(2*num)
        result = np.frombuffer(self.mpbyte[left:right], dtype=self.endian+"i2", count=(right-left)//2)
        if num == 1: return int(result[0])
        return list(result)

    def readint32seek0(self, num=1):
        left, right = self.__calc_size_no_seek__(4*num)
        result = np.frombuffer(self.mpbyte[left:right], dtype=self.endian+"i4", count=(right-left)//4)
        if num == 1: return int(result[0])
        return list(result)

    def readint64seek0(self, num=1):
        left, right = self.__calc_size_no_seek__(8*num)
        result = np.frombuffer(self.mpbyte[left:right], dtype=self.endian+"i8", count=(right-left)//8)
        if num == 1: return int(result[0])
        return list(result)


    def readfloat16seek0(self, num=1):
        left, right = self.__calc_size_no_seek__(2*num)
        result = np.frombuffer(self.mpbyte[left:right], dtype=self.endian+"f2", count=(right-left)//2)
        if num == 1: return result[0]
        return list(result)

    def readfloat32seek0(self, num=1):
        left, right = self.__calc_size_no_seek__(4*num)
        result = np.frombuffer(self.mpbyte[left:right], dtype=self.endian+"f4", count=(right-left)//4)
        if num == 1: return result[0]
        return list(result)

    def readfloat64seek0(self, num=1):
        left, right = self.__calc_size_no_seek__(8*num)
        result = np.frombuffer(self.mpbyte[left:right], dtype=self.endian+"f8", count=(right-left)//8)
        if num == 1: return result[0]
        return list(result)


    def readu8float32seek0(self, num=1):
        uint8s = self.readuint8seek0(num)
        if num == 1: return uint8s/255.0
        else: return [uint8/255.0 for uint8 in uint8s]

    def readi8float32seek0(self, num=1):
        int8s = self.readint8seek0(num)
        if num == 1: return int8s/128.0
        else: return [int8/128.0 for int8 in int8s]

    def readu16float32seek0(self, num=1):
        uint16s = self.readuint16seek0(num)
        if num == 1: return uint16s/65535.0
        else: return [uint16/65535.0 for uint16 in uint16s]

    def readi16float32seek0(self, num=1):
        int16s = self.readint16seek0(num)
        if num == 1: return int16s/32767.0
        else: return [int16/32767.0 for int16 in int16s]


    def read5u8uint64seek0(self):
        left, right = self.__calc_size_no_seek__(5)
        if self.endian == "<" : buffer = self.mpbyte[left:right] + b"\x00\x00\x00"
        if self.endian == ">" : buffer = b"\x00\x00\x00" + self.mpbyte[left:right]
        return int(np.frombuffer(buffer, dtype=self.endian+"u8")[0])

    def readbinseek0(self, num=1):
        if num == 1: return bin(self.readuint8seek0(num))
        return [bin(uint8) for uint8 in self.readuint8seek0(num)]

    def readhexseek0(self, num=1):
        if num == 1: return "%02X"%self.readuint8seek0(num)
        return ["%02X"%uint8 for uint8 in self.readuint8seek0(num)]

    def readcharseek0(self, num=1):
        if num <= 0: return ""
        left, right = self.__calc_size_no_seek__(num)
        chars = ""
        for uint8 in self.mpbyte[left:right]: chars += chr(uint8)
        return chars

    def readgbkseek0(self, bytenum):
        if bytenum <= 0: return ""
        left, right = self.__calc_size_no_seek__(bytenum)
        try:
            return self.mpbyte[left:right].decode("GBK")
        except: 
            chars = ""
            for uint8 in self.mpbyte[left:right]: chars += chr(uint8)
            return chars


    def readutf8seek0(self, bytenum):
        if bytenum <= 0: return ""
        left, right = self.__calc_size_no_seek__(bytenum)
        try: 
            return self.mpbyte[left:right].decode("utf-8")
        except: 
            chars = ""
            for uint8 in self.mpbyte[left:right]: chars += chr(uint8)
            return chars
        

# # list里的 u8, u16, u32，转python int, 由上层负责int()转换，本文件不处理
# # 注意u8乘法超出u8显示范围的错误，计算结果超出u8范围请上层文件int()转换类型在计算

# int8 = 127  # -128, 127
# uint8 = 255  # 0, 255
# int16 = 32767 # -32768, 32767
# uint16 = 65535 # 0, 65535