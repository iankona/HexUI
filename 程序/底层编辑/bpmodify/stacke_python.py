import os
from . import struct_python



class 类:
    def __init__(self, mpbyte=None, endian="<"):
        self.mpbyte = mpbyte # self.index = 0
        self.endian = endian


    def __assert__number__(self, num, step):
        assert num >= 0, f"assert::__assert__number__::num must be >= 0, not < 0 !" # 读末尾时，会一直读零，需保持正常读取
        left, right = self.index, self.index + num * step
        if right > self.mpright: right = self.mpright
        count = (right - left) // step
        right = self.index + count * step
        return left, right, count

    def __calc__count__(self, num, step): 
        left, right, count = self.__assert__number__(num, step)
        self.index = right
        return left, right, count

    def __calc__count__no__seek__(self, num, step):
        left, right, count = self.__assert__number__(num, step)
        return left, right, count

    def __calc__size__(self, size): 
        left, right, count = self.__assert__number__(size, 1)
        self.index = right
        return left, right

    def __calc__size__no__seek__(self, size):
        left, right, count = self.__assert__number__(size, 1)
        return left, right



    def __assert__function__(self, varbyte, count, step, function):
        buffer = []
        for i in range(count): buffer.append( varbyte[i*step: (i+1)*step] )
        varvalue = []
        for bufferstep in buffer: varvalue.append( function(bufferstep, self.endian) )
        if count == 1: return varvalue[0]
        return varvalue

    def __read__function__(self, num, step=1, function=None):
        left, right, count = self.__assert__number__(num, step)
        varbyte = self.mpbyte[left: right]
        varvalue = self.__assert__function__(varbyte, count, step, function)
        self.index = right
        return varvalue

    def __read__function__no__seek__(self, num, step=1, function=None):
        left, right, count = self.__assert__number__(num, step)
        varbyte = self.mpbyte[left: right]
        varvalue = self.__assert__function__(varbyte, count, step, function)
        return varvalue



    def readint8(self, num=1):  
        return self.__read__function__(num, step=1, func=struct_python.byte_2_int8)
    
    def readint16(self, num=1): 
        return self.__read__function__(num, step=2, func=struct_python.byte_2_int16)
    
    def readint32(self, num=1): 
        return self.__read__function__(num, step=4, func=struct_python.byte_2_int32)
    
    def readint64(self, num=1): 
        return self.__read__function__(num, step=8, func=struct_python.byte_2_int64)
    


    def readuint8(self, num=1): 
        return self.__read__function__(num, step=1, func=struct_python.byte_2_uint8)
    
    def readuint16(self, num=1): 
        return self.__read__function__(num, step=2, func=struct_python.byte_2_uint16)
    
    def readuint32(self, num=1): 
        return self.__read__function__(num, step=4, func=struct_python.byte_2_uint32)
    
    def readuint64(self, num=1): 
        return self.__read__function__(num, step=8, func=struct_python.byte_2_uint64)



    def readfloat16(self, num=1): 
        return self.__read__function__(num, step=2, func=struct_python.byte_2_float16)
    
    def readfloat32(self, num=1): 
        return self.__read__function__(num, step=4, func=struct_python.byte_2_float32)
    
    def readfloat64(self, num=1): 
        return self.__read__function__(num, step=8, func=struct_python.byte_2_float64)
    


    def readint8seek0(self, num=1): 
        return self.__read__function__no__seek__(num, step=1, func=struct_python.byte_2_int8)
    
    def readint16seek0(self, num=1): 
        return self.__read__function__no__seek__(num, step=2, func=struct_python.byte_2_int16)
    
    def readint32seek0(self, num=1): 
        return self.__read__function__no__seek__(num, step=4, func=struct_python.byte_2_int32)
    
    def readint64seek0(self, num=1): 
        return self.__read__function__no__seek__(num, step=8, func=struct_python.byte_2_int64)
    


    def readuint8seek0(self, num=1): 
        return self.__read__function__no__seek__(num, step=1, func=struct_python.byte_2_uint8)
    
    def readuint16seek0(self, num=1): 
        return self.__read__function__no__seek__(num, step=2, func=struct_python.byte_2_uint16)
    
    def readuint32seek0(self, num=1): 
        return self.__read__function__no__seek__(num, step=4, func=struct_python.byte_2_uint32)
    
    def readuint64seek0(self, num=1): 
        return self.__read__function__no__seek__(num, step=8, func=struct_python.byte_2_uint64)
    


    def readfloat16seek0(self, num=1): 
        return self.__read__function__no__seek__(num, step=2, func=struct_python.byte_2_float16)
    
    def readfloat32seek0(self, num=1): 
        return self.__read__function__no__seek__(num, step=4, func=struct_python.byte_2_float32)
    
    def readfloat64seek0(self, num=1): 
        return self.__read__function__no__seek__(num, step=8, func=struct_python.byte_2_float64)



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
        resultindex = self.index + size
        if resultindex < self.mpleft : raise ValueError(f"stacke::seek::回跳的索引值超出slice范围...")
        if resultindex > self.mpright: raise ValueError(f"stacke::seek::往前的索引值超出slice范围...")
        self.index = self.index + size

    def seekclosecheck(self, size):
        resultindex = self.index + size
        if resultindex < self.mpleft : resultindex = self.mpleft
        if resultindex > self.mpright: resultindex = self.mpright
        self.index = resultindex
        

    def readslice0b(self):
        return self.mpbyte[self.mpleft: self.mpright]
    
    def read(self, size=1):
        left, right = self.__calc__size__(size)
        return self.mpbyte[left: right]

    def readremain(self):
        return self.read(self.remainsize())



    def readseek0(self, size=1):
        left, right = self.__calc__size__no__seek__(size)
        return self.mpbyte[left:right]

    def readremainseek0(self):
        return self.readseek0(self.remainsize())



    def readascii(self, bytenum):
        left, right = self.__calc__size__(bytenum)
        strbyte = self.mpbyte[left:right]
        return strbyte.decode(encoding="ASCII")

    def readutf8(self, bytenum):
        left, right = self.__calc__size__(bytenum)
        strbyte = self.mpbyte[left:right]
        return strbyte.decode(encoding="utf-8")
    
    def readgbk(self, bytenum):
        left, right = self.__calc__size__(bytenum)
        strbyte = self.mpbyte[left:right]
        return strbyte.decode(encoding="gbk")



    def readasciiseek0(self, bytenum):
        left, right = self.__calc__size__no__seek__(bytenum)
        strbyte = self.mpbyte[left:right]
        return strbyte.decode(encoding="ASCII")
    
    def readutf8seek0(self, bytenum):
        left, right = self.__calc__size__no__seek__(bytenum)
        strbyte = self.mpbyte[left:right]
        return strbyte.decode(encoding="utf-8")
    
    def readgbkseek0(self, bytenum):
        left, right = self.__calc__size__no__seek__(bytenum)
        strbyte = self.mpbyte[left:right]
        return strbyte.decode(encoding="gbk")


