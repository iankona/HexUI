import math
import sys



def havokcarry(bp):
    if ((bp.readuint8seek0(1)      & 0b10000000) >> 7) == 0: return [ 1, [bp.readuint8seek0(1)]]
    if ((bp.readuint8seek0(2)[-1]  & 0b10000000) >> 7) == 0: return [ 2,  bp.readuint8seek0(2) ]
    if ((bp.readuint8seek0(3)[-1]  & 0b10000000) >> 7) == 0: return [ 3,  bp.readuint8seek0(3) ]
    if ((bp.readuint8seek0(4)[-1]  & 0b10000000) >> 7) == 0: return [ 4,  bp.readuint8seek0(4) ]
    if ((bp.readuint8seek0(5)[-1]  & 0b10000000) >> 7) == 0: return [ 5,  bp.readuint8seek0(5) ]
    if ((bp.readuint8seek0(6)[-1]  & 0b10000000) >> 7) == 0: return [ 6,  bp.readuint8seek0(6) ]
    if ((bp.readuint8seek0(7)[-1]  & 0b10000000) >> 7) == 0: return [ 7,  bp.readuint8seek0(7) ]
    if ((bp.readuint8seek0(8)[-1]  & 0b10000000) >> 7) == 0: return [ 8,  bp.readuint8seek0(8) ]
    if ((bp.readuint8seek0(9)[-1]  & 0b10000000) >> 7) == 0: return [ 9,  bp.readuint8seek0(9) ]
    if ((bp.readuint8seek0(10)[-1] & 0b10000000) >> 7) == 0: return [10,  bp.readuint8seek0(10) ]
    if ((bp.readuint8seek0(11)[-1] & 0b10000000) >> 7) == 0: return [11,  bp.readuint8seek0(11) ]
    if ((bp.readuint8seek0(12)[-1] & 0b10000000) >> 7) == 0: return [12,  bp.readuint8seek0(12) ]



def havok_int_from_uint8(auint8):
    result = 0
    for i, uint8 in enumerate(auint8):
        if i == 0: 
            value = (uint8 & 0b01111110) >> 1
            minus = (uint8 & 0b00000001) 
            result += value
        if i == 1:
            value = int(uint8 & 0b01111111)
            result += value * 64
        if i == 2:
            value = int(uint8 & 0b01111111)
            result += value * 64 * 128
        if i == 3:
            value = int(uint8 & 0b01111111)
            result += value * 64 * 128 * 128
        if i == 4:
            value = int(uint8 & 0b01111111)
            result += value * 64 * 128 * 128 * 128
        if i == 5:
            value = int(uint8 & 0b01111111)
            result += value * 64 * 128 * 128 * 128 * 128
        if i == 6:
            value = int(uint8 & 0b01111111)
            result += value * 64 * 128 * 128 * 128 * 128 * 128
        if i == 7:
            value = int(uint8 & 0b01111111)
            result += value * 64 * 128 * 128 * 128 * 128 * 128 * 128
        if i == 8:
            value = int(uint8 & 0b01111111)
            result += value * 64 * 128 * 128 * 128 * 128 * 128 * 128 * 128
        if i == 9:
            value = int(uint8 & 0b01111111)
            result += value * 64 * 128 * 128 * 128 * 128 * 128 * 128 * 128 * 128
        if i == 10:
            value = int(uint8 & 0b01111111)
            result += value * 64 * 128 * 128 * 128 * 128 * 128 * 128 * 128 * 128 * 128
        if i == 11:
            value = int(uint8 & 0b01111111)
            result += value * 64 * 128 * 128 * 128 * 128 * 128 * 128 * 128 * 128 * 128 * 128

    if minus == 1: result = -result
    return result



def havok_uint_from_uint8(auint8):
    result = havok_int_from_uint8(auint8)
    if result >= 0: return result
    intbytes = result.tobytes()
    match len(intbytes):
        case 1:  result += (0xFF + 1)
        case 2:  result += (0xFFFF + 1)
        case 3:  result += (0xFFFFFF + 1)
        case 4:  result += (0xFFFFFFFF + 1)
        case 5:  result += (0xFFFFFFFFFF + 1)
        case 6:  result += (0xFFFFFFFFFFFF + 1)
        case 7:  result += (0xFFFFFFFFFFFFFF + 1)
        case 8:  result += (0xFFFFFFFFFFFFFFFF + 1)
        case 9:  result += (0xFFFFFFFFFFFFFFFFFF + 1)
        case 10: result += (0xFFFFFFFFFFFFFFFFFFFF + 1)

    return result




def readhavokint(bp):
    numuint8, auint8 = havokcarry(bp)
    result = havok_int_from_uint8(auint8)
    bp.readuint8(numuint8)
    return result

def readhavokuint(bp):
    numuint8, auint8 = havokcarry(bp)
    result = havok_uint_from_uint8(auint8)
    bp.readuint8(numuint8)
    return result


def readhavokintseek0(bp):
    numuint8, auint8 = havokcarry(bp)
    result = havok_int_from_uint8(auint8)
    return result



def readhavokuintseek0(bp):
    numuint8, auint8 = havokcarry(bp)
    result = havok_uint_from_uint8(auint8)
    return result





def readhavokquat(bp):
    # 左移位和右移位运算符，其格式为a<<或b>>，含义为将数字a的二进制为数左移或右移b位。
    # a=0b10101
    # a<<2，表示将a左移2位，即a=0b1010100
    # a>>1，表示将a右移1位，即a=0b1010

    # (1)负数的移位计数为非法操作，其可能导致ValueError错误
    # (2)左移位，底位空缺补零，高位溢出舍弃∶右移位，高位空缺补零，低位溢出舍弃。
    # (3)左移N位相当于将数乘以2的N次幂；右移N位相当于将数除以2的N次幂。

    # 按位与运算符的步骤：对应的二个二进位都为 1 时，结果位就为 1，否则为 0。

    # 每字节/每个uint8, 8位bit
    # 定义左端为低位，计算机的内部处理都是小端字节序, 人类还是习惯读写大端字节序
    # 0x123456在内存中的存储方式
    # - 大端模式，低位存大数为大端
    #   低地址 ---> 高地址
    #   0x12 | 0x34 | 0x56 
    # - 小端模式，低位存小数为小端
    #   低地址 ---> 高地址
    #   0x56 | 0x34 | 0x12 

    # [0b 0011 0111     1111 1111     0111 1111     1111 1111     1111 1110, ]
    #    00 11 0111     1111 1111     0111 1111     1111 1111     1111 1110

    #                                              x, 0b 1111     1111 1110
    #                           y, 0b 0111 1111     1111
    #    z, 0b 0111     1111 1111

    #  4096 0b 0001 0000 0000 0000, 13bit
    #  4095 0b      1111 1111 1111, 12bit1
    #  2047 0b      0111 1111 1111, 11bit1

    # x, y, z, 4094 2047 4094
    #    x, 0b 1111 1111 1110
    # 2047, 0b 0111 1111 1111

    #    y, 0b 0111 1111 1111
    # 2047, 0b 0111 1111 1111

    #    z, 0b 0111 1111 1111
    # 2047, 0b 0111 1111 1111

    byte5 = bp.read(5)
    order = "big" if bp.endian == ">" else "little"
    value = int.from_bytes(byte5, byteorder=order, signed=False)
    xdigi =  value        & 4095  #  4095, 0b 1111 1111 1111, 12bit1  
    ydigi = (value >> 12) & 4095  #  4095, 0b 1111 1111 1111, 12bit1   
    zdigi = (value >> 24) & 4095  #  4095, 0b 1111 1111 1111, 12bit1        
    shift = (value >> 36) & 3     #     3, 0b 11
    minus = (value >> 38) & 1     #     1, 0b 1

    xint = xdigi - 2047          
    yint = ydigi - 2047        
    zint = zdigi - 2047          

    fractal = 0.000345436
    rx = xint * fractal
    ry = yint * fractal
    rz = zint * fractal

    rw = 1 - rx*rx - ry*ry - rz*rz
    if rw < 0.0: rw = 0.0
    rw = math.sqrt(rw)
    if minus > 0: rw = -rw

    result = [rx, ry, rz]
    result.insert(shift, rw)
    qx, qy, qz, qw = result
    # if qw < 0: qx, qy, qz, qw = -qx, -qy, -qz, -qw   # 预计blender会自动转换 # blender中共轭四元数不用转换，其行为是一样的

    return [qx, qy, qz, qw]




# def readhavokint(bp):
#     result = 0
#     # 0
#     uint8 = bp.readuint8()
#     carry = (uint8 & 0b10000000) >> 7
#     value = (uint8 & 0b01111110) >> 1
#     minus = (uint8 & 0b00000001) 
#     result += value
#     if carry == 0:
#         if minus == 1: result = -result
#         return result
#     # 1
#     uint8 = bp.readuint8()
#     carry = (uint8 & 0b10000000) >> 7
#     value = (uint8 & 0b01111111)
#     result += value * 64
#     if carry == 0:
#         if minus == 1: result = -result
#         return result
#     # 2
#     uint8 = bp.readuint8()
#     carry = (uint8 & 0b10000000) >> 7
#     value = (uint8 & 0b01111111)
#     result += value * 64 * 128
#     if carry == 0:
#         if minus == 1: result = -result
#         return result
#     # 3
#     uint8 = bp.readuint8()
#     carry = (uint8 & 0b10000000) >> 7
#     value = (uint8 & 0b01111111)
#     result += value * 64 * 128 * 128
#     if carry == 0:
#         if minus == 1: result = -result
#         return result



# def havokuint(bp):
#     cint8 = bp.readuint8()
#     if cint8//2 >= 64: result = cint8//2 + havokuint(bp)
#     else:              result = cint8//2
#     if cint8 % 2 == 0: result = (result*2-1) * 64
#     else:              result = (result*2  ) * 64
#     return result


# def readhavokint(bp): # 整数的压缩及解压缩
#     aint8 = bp.readuint8()
#     if aint8//2 >= 64: result = aint8//2 + havokint(bp)
#     else:              result = aint8//2
#     if aint8 % 2 == 1: result = -result
#     return result


# def readhavokuint(bp, datatype=4): # 整数的压缩及解压缩
#     aint8 = bp.readuint8()
#     if aint8//2 >= 64: result = aint8//2 + havokuint(bp)
#     else:              result = aint8//2

#     if aint8 % 2 == 1:
#         if   datatype ==  6: result = -result & 0xFFFF
#         elif datatype ==  8: result = -result & 0xFFFFFFFF
#         elif datatype == 10: result = -result & 0xFFFFFFFFFFFFFFFF
#         else :               result = -result
#     return result





