from . import struct_bnumpy


# 对于num
# if num == 0: return []
# if num == 1: return int or float
# if num >  1: return [int] or [float]
# 对于size 
# size == 0 or size == 1 or size > 1: return [], [byte, char] or bp 
class 类:
    def __init__(self, mpbyte=None, endian="<", module:struct_bnumpy=None):
        self.script = module
        self.mpbyte = mpbyte 
        self.endian = endian


    def fromstream(self, mpbyte:bytes):
        bp = 类(mpbyte=mpbyte, endian=self.endian)
        bp.script = self.script
        bp.mpleft, bp.index, bp.mpright = 0, 0, len(mpbyte)
        return bp


    def copy(self):
        bp = 类(mpbyte=self.mpbyte, endian=self.endian)
        bp.script = self.script
        bp.mpleft, bp.mpright = self.mpleft, self.mpright
        bp.index = bp.mpleft
        return bp
    

    def readslice(self, size):
        left, right = self.__calc__size__(size)
        bp = 类(mpbyte=self.mpbyte, endian=self.endian)
        bp.script = self.script
        bp.mpleft, bp.mpright = left, right
        bp.index = bp.mpleft
        return bp


    def readsliceseek0(self, size):
        left, right = self.__calc__size__no__seek__(size)
        bp = 类(mpbyte=self.mpbyte, endian=self.endian)
        bp.script = self.script
        bp.mpleft, bp.mpright = left, right
        bp.index = bp.mpleft
        return bp


    def readremainslice(self):
        left, right = self.__calc__size__(self.remainsize())
        bp = 类(mpbyte=self.mpbyte, endian=self.endian)
        bp.script = self.script
        bp.mpleft, bp.mpright = left, right
        bp.index = bp.mpleft
        return bp


    def readremainsliceseek0(self):
        left, right = self.__calc__size__no__seek__(self.remainsize())
        bp = 类(mpbyte=self.mpbyte, endian=self.endian)
        bp.script = self.script
        bp.mpleft, bp.mpright = left, right
        bp.index = bp.mpleft
        return bp


    def __assert__count__(self, num, step):
        assert num >= 0, f"assert::__assert__count__::读取个数最小为 0 , not {num}" 
        left, right = self.index, self.index + num * step
        if right > self.mpright: right = self.mpright
        count = (right - left) // step
        right = self.index + count * step
        return left, right, count

    def __calc__count__(self, num, step): 
        left, right, count = self.__assert__count__(num, step)
        self.index = right
        return left, right, count

    def __calc__count__no__seek__(self, num, step):
        left, right, count = self.__assert__count__(num, step)
        return left, right, count



    def __assert__size__(self, size):
        assert size >= 0, f"assert::__assert__size__::读取个数最小为 0 , not {size}" # 读末尾时，会一直读零，需保持正常读取
        left, right = self.index, self.index + size
        if right > self.mpright: right = self.mpright
        return left, right
    

    def __calc__size__(self, size): 
        left, right = self.__assert__size__(size)
        self.index = right
        return left, right

    def __calc__size__no__seek__(self, size):
        left, right = self.__assert__size__(size)
        return left, right



    def __assert__function__(self, varbyte, count, step, function):
        buffer = []
        for i in range(count): buffer.append( varbyte[i*step: (i+1)*step] )
        varvalue = []
        for bufferstep in buffer: varvalue.append( function(bufferstep, self.endian) )
        if count == 1: return varvalue[0]
        return varvalue

    def __read__function__(self, num, step=1, function=None):
        left, right, count = self.__calc__count__(num, step)
        varbyte = self.mpbyte[left: right]
        varvalue = self.__assert__function__(varbyte, count, step, function)
        return varvalue

    def __read__function__no__seek__(self, num, step=1, function=None):
        left, right, count = self.__calc__count__no__seek__(num, step)
        varbyte = self.mpbyte[left: right]
        varvalue = self.__assert__function__(varbyte, count, step, function)
        return varvalue



    def readint8( self, num=1): return self.__read__function__(num, step=1, function=self.script.byte_2_int8 )
    def readint16(self, num=1): return self.__read__function__(num, step=2, function=self.script.byte_2_int16)
    def readint32(self, num=1): return self.__read__function__(num, step=4, function=self.script.byte_2_int32)
    def readint64(self, num=1): return self.__read__function__(num, step=8, function=self.script.byte_2_int64)
    
    def readuint8( self, num=1): return self.__read__function__(num, step=1, function=self.script.byte_2_uint8 )
    def readuint16(self, num=1): return self.__read__function__(num, step=2, function=self.script.byte_2_uint16)
    def readuint32(self, num=1): return self.__read__function__(num, step=4, function=self.script.byte_2_uint32)
    def readuint64(self, num=1): return self.__read__function__(num, step=8, function=self.script.byte_2_uint64)

    def readfloat16(self, num=1): return self.__read__function__(num, step=2, function=self.script.byte_2_float16)
    def readfloat32(self, num=1): return self.__read__function__(num, step=4, function=self.script.byte_2_float32)
    def readfloat64(self, num=1): return self.__read__function__(num, step=8, function=self.script.byte_2_float64)
    

    def readi8float32(self, num=1):
        buffervalue, convert = self.readint8(num), 0xFF
        if num == 1: return buffervalue / convert
        return [value / convert for value in buffervalue]

    def readu8float32(self, num=1):
        buffervalue, convert = self.readuint8(num), 0xFF
        if num == 1: return buffervalue / convert
        return [value / convert for value in buffervalue]
        
    def readi16float32(self, num=1):
        buffervalue, convert = self.readint16(num), 0xFFFF
        if num == 1: return buffervalue / convert
        return [value / convert for value in buffervalue]

    def readu16float32(self, num=1):
        buffervalue, convert = self.readuint16(num), 0xFFFF
        if num == 1: return buffervalue / convert
        return [value / convert for value in buffervalue]



    def readint8seek0( self, num=1): return self.__read__function__no__seek__(num, step=1, function=self.script.byte_2_int8 )
    def readint16seek0(self, num=1): return self.__read__function__no__seek__(num, step=2, function=self.script.byte_2_int16)
    def readint32seek0(self, num=1): return self.__read__function__no__seek__(num, step=4, function=self.script.byte_2_int32)
    def readint64seek0(self, num=1): return self.__read__function__no__seek__(num, step=8, function=self.script.byte_2_int64)
    
    def readuint8seek0( self, num=1): return self.__read__function__no__seek__(num, step=1, function=self.script.byte_2_uint8 )
    def readuint16seek0(self, num=1): return self.__read__function__no__seek__(num, step=2, function=self.script.byte_2_uint16)
    def readuint32seek0(self, num=1): return self.__read__function__no__seek__(num, step=4, function=self.script.byte_2_uint32)
    def readuint64seek0(self, num=1): return self.__read__function__no__seek__(num, step=8, function=self.script.byte_2_uint64)
    
    def readfloat16seek0(self, num=1): return self.__read__function__no__seek__(num, step=2, function=self.script.byte_2_float16)
    def readfloat32seek0(self, num=1): return self.__read__function__no__seek__(num, step=4, function=self.script.byte_2_float32)
    def readfloat64seek0(self, num=1): return self.__read__function__no__seek__(num, step=8, function=self.script.byte_2_float64)

    def readi8float32seek0(self, num=1):
        buffervalue, convert = self.readint8seek0(num), 0xFF
        if num == 1: return buffervalue / convert
        return [value / convert for value in buffervalue]

    def readu8float32seek0(self, num=1):
        buffervalue, convert = self.readuint8seek0(num), 0xFF
        if num == 1: return buffervalue / convert
        return [value / convert for value in buffervalue]
        
    def readi16float32seek0(self, num=1):
        buffervalue, convert = self.readint16seek0(num), 0xFFFF
        if num == 1: return buffervalue / convert
        return [value / convert for value in buffervalue]

    def readu16float32seek0(self, num=1):
        buffervalue, convert = self.readuint16seek0(num), 0xFFFF
        if num == 1: return buffervalue / convert
        return [value / convert for value in buffervalue]



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


    def movetell(self, index):
        offset = index
        if index < self.mpleft : offset = self.mpleft
        if index > self.mpright: offset = self.mpright
        self.index = offset



    def readslice0b(self):
        return self.mpbyte[self.mpleft: self.mpright]
    

    def read(self, size=0):
        left, right = self.__calc__size__(size)
        return self.mpbyte[left: right]

    def readremain(self):
        return self.read(self.remainsize())



    def readbin(self, size=0):
        bufferbyte = self.read(size)
        result = ""
        for uint8 in bufferbyte: result += bin(uint8)[2:]
        return result

    def readbinqueue(self, size=0):
        bufferbyte = self.read(size)
        return [bin(uint8)[2:] for uint8 in bufferbyte]

    def readbin8(self, size=0):
        bufferbyte = self.read(size)
        result = ""
        for uint8 in bufferbyte: 
            charbin = bin(uint8)[2:]
            numpad0 = 8 - len(charbin)
            charbin8 = numpad0*"0" + charbin
            result += charbin8
        return result

    def readbin8queue(self, size=0):
        bufferbyte = self.read(size)
        result = []
        for uint8 in bufferbyte: 
            bufferbin = bin(uint8)[2:]
            numpad0 = 8 - len(bufferbin)
            result.append(numpad0*"0" + bufferbin)
        return result


    def readhex(self, size=0):
        bufferbyte = self.read(size)
        result = ""
        for uint8 in bufferbyte: result += "%02X"%uint8
        return result

    def readhexqueue(self, size=0):
        bufferbyte = self.read(size)
        result = []
        for uint8 in bufferbyte: result .append("%02X"%uint8)
        return result

    def readchar(self, size=0):
        bufferbyte = self.read(size)
        result = ""
        for uint8 in bufferbyte: result += chr(uint8)
        return result

    def readcharqueue(self, size=0):
        bufferbyte = self.read(size)
        result = []
        for uint8 in bufferbyte: result.append(chr(uint8))
        return result
    

    def readcharend0(self):
        bufferchar = ""
        while True:
            bufferuint8 = self.readuint8()
            if bufferuint8 == 0: break
            bufferchar += chr(bufferuint8)
        return bufferchar


    def readgbk(self, size=0):
        bufferbyte = self.read(size)
        return bufferbyte.decode(encoding="gbk")

    def readutf8(self, size=0):
        bufferbyte = self.read(size)
        return bufferbyte.decode(encoding="utf-8")
    
    def readascii(self, size=0):
        bufferbyte = self.read(size)
        return bufferbyte.decode(encoding="ASCII")
    




    def readseek0(self, size=0):
        left, right = self.__calc__size__no__seek__(size)
        return self.mpbyte[left:right]

    def readremainseek0(self):
        return self.readseek0(self.remainsize())



    def readbinseek0(self, size=0):
        bufferbyte = self.readseek0(size)
        result = ""
        for uint8 in bufferbyte: result += bin(uint8)[2:]
        return result

    def readbinqueueseek0(self, size=0):
        bufferbyte = self.readseek0(size)
        return [bin(uint8)[2:] for uint8 in bufferbyte]

    def readbin8seek0(self, size=0):
        bufferbyte = self.readseek0(size)
        result = ""
        for uint8 in bufferbyte: 
            bufferbin = bin(uint8)[2:]
            bufferbin8 = (8-len(bufferbin))*"0" + bufferbin
            result += bufferbin8
        return result

    def readbin8queueseek0(self, size=0):
        bufferbyte = self.readseek0(size)
        result = []
        for uint8 in bufferbyte: 
            bufferbin = bin(uint8)[2:]
            result.append((8-len(bufferbin))*"0" + bufferbin)
        return result


    def readhexseek0(self, size=0):
        bufferbyte = self.readseek0(size)
        result = ""
        for uint8 in bufferbyte: result += "%02X"%uint8
        return result

    def readhexqueueseek0(self, size=0):
        bufferbyte = self.readseek0(size)
        result = []
        for uint8 in bufferbyte: result.append("%02X"%uint8)
        return result

    def readcharseek0(self, size=0):
        bufferbyte = self.readseek0(size)
        result = ""
        for uint8 in bufferbyte: result += chr(uint8)
        return result

    def readcharqueueseek0(self, size=0):
        bufferbyte = self.readseek0(size)
        result = []
        for uint8 in bufferbyte: result.append(chr(uint8))
        return result

    def readcharend0seek0(self):
        bp = self.readremainsliceseek0()
        bufferchar = ""
        while True:
            bufferuint8 = bp.readuint8()
            if bufferuint8 == 0: break
            bufferchar += chr(bufferuint8)
        return bufferchar
    
    def readcharend0sizeseek0(self):
        bp = self.readremainsliceseek0()
        start = bp.tell()
        while True:
            bufferuint8 = bp.readuint8()
            if bufferuint8 == 0: break
        final = bp.tell()
        return final - start
    



    def readgbkseek0(self, size=0):
        bufferbyte = self.readseek0(size)
        return bufferbyte.decode(encoding="gbk")
    
    def readutf8seek0(self, size=0):
        bufferbyte = self.readseek0(size)
        return bufferbyte.decode(encoding="utf-8")

    def readasciiseek0(self, size=0):
        bufferbyte = self.readseek0(size)
        return bufferbyte.decode(encoding="ASCII")
    




