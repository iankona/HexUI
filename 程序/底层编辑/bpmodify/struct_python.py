# 函数内部获取函数名，sys._getframe().f_code.co_name

# 浮点数的精度损失
# float： 2^23 = 8388608，         一共 7位， float的精度为 6~ 7位十进制有效数字，有隐含位"1”
# double：2^52 = 4503599627370496，一共16位，double的精度为15~16位十进制有效数字，有隐含位"1”
# long double：占用内存10个字节，精度最高19位。1， 15,  64, long double 有16字节，12字节，10字节。 其中16字节占大多数，无隐含位

#          4       5       0     3     5     9     9    6    2     7     3    7   0   4   9   5
# 亿亿  千万亿  百万亿  十万亿  万亿  千亿  百亿  十亿  亿  千万  百万  十万  万  仟  百  十  个

# 宏int8  = [-128, 127]
# 宏int16 = [-32768, 32767]
# 宏int32 = [-2147483648, 2147483647]
# 宏int64 = [-9223372036854775808, 9223372036854775807]

# 宏uint8  = [0, 255]
# 宏uint16 = [0, 65535]
# 宏uint32 = [0, 4294967295]
# 宏uint64 = [0, 18446744073709551615]


def __convert__endian__(varbyte:bytes, endian:str="<"):
    match endian:
        case ">": bigbyte = varbyte
        case "<": bigbyte = varbyte[::-1]
        case  _ : raise ValueError(f"bgstruct::__convert__endian__::错误的endian值:{endian}，must be in ['>', '<']")
    return bigbyte



def byte_2_int8(varbyte:bytes, endian:str="<"):
    numbyte = 1
    assert len(varbyte) == numbyte, f"bgstruct::byte_2_int8::输入的字节长度!={numbyte}"
    bigbyte = __convert__endian__(varbyte, endian)
    return int.from_bytes(bigbyte, byteorder="big", signed=True)


def byte_2_int16(varbyte:bytes, endian:str="<"):
    numbyte = 2
    assert len(varbyte) == numbyte, f"bgstruct::byte_2_int16::输入的字节长度!={numbyte}"
    bigbyte = __convert__endian__(varbyte, endian)
    return int.from_bytes(bigbyte, byteorder="big", signed=True)


def byte_2_int32(varbyte:bytes, endian:str="<"):
    numbyte = 4
    assert len(varbyte) == numbyte, f"bgstruct::byte_2_int32::输入的字节长度!={numbyte}"
    bigbyte = __convert__endian__(varbyte, endian)
    return int.from_bytes(bigbyte, byteorder="big", signed=True)


def byte_2_int64(varbyte:bytes, endian:str="<"):
    numbyte = 8
    assert len(varbyte) == numbyte, f"bgstruct::byte_2_int64::输入的字节长度!={numbyte}"
    bigbyte = __convert__endian__(varbyte, endian)
    return int.from_bytes(bigbyte, byteorder="big", signed=True)


def byte_2_uint8(varbyte:bytes, endian:str="<"):
    numbyte = 1
    assert len(varbyte) == numbyte, f"bgstruct::byte_2_uint8::输入的字节长度!={numbyte}"
    bigbyte = __convert__endian__(varbyte, endian)
    return int.from_bytes(bigbyte, byteorder="big", signed=False)




def byte_2_uint16(varbyte:bytes, endian:str="<"):
    numbyte = 2
    assert len(varbyte) == numbyte, f"bgstruct::byte_2_uint16::输入的字节长度!={numbyte}"
    bigbyte = __convert__endian__(varbyte, endian)
    return int.from_bytes(bigbyte, byteorder="big", signed=False)


def byte_2_uint32(varbyte:bytes, endian:str="<"):
    numbyte = 4
    assert len(varbyte) == numbyte, f"bgstruct::byte_2_uint32::输入的字节长度!={numbyte}"
    bigbyte = __convert__endian__(varbyte, endian)
    return int.from_bytes(bigbyte, byteorder="big", signed=False)


def byte_2_uint64(varbyte:bytes, endian:str="<"):
    numbyte = 8
    assert len(varbyte) == numbyte, f"bgstruct::byte_2_uint64::输入的字节长度!={numbyte}"
    bigbyte = __convert__endian__(varbyte, endian)
    return int.from_bytes(bigbyte, byteorder="big", signed=False)


def int8_2_byte(intvalue:int, endian:str="<"):
    numbyte = 1
    bigbyte = intvalue.to_bytes(numbyte, byteorder="big", signed=True) # OverflowError: int too big to convert
    varbyte = __convert__endian__(bigbyte, endian)
    return varbyte


def int16_2_byte(intvalue:int, endian:str="<"):
    numbyte = 2
    bigbyte = intvalue.to_bytes(numbyte, byteorder="big", signed=True) # OverflowError: int too big to convert
    varbyte = __convert__endian__(bigbyte, endian)
    return varbyte


def int32_2_byte(intvalue:int, endian:str="<"):
    numbyte = 4
    bigbyte = intvalue.to_bytes(numbyte, byteorder="big", signed=True) # OverflowError: int too big to convert
    varbyte = __convert__endian__(bigbyte, endian)
    return varbyte


def int64_2_byte(intvalue:int, endian:str="<"):
    numbyte = 8
    bigbyte = intvalue.to_bytes(numbyte, byteorder="big", signed=True) # OverflowError: int too big to convert
    varbyte = __convert__endian__(bigbyte, endian)
    return varbyte



def uint8_2_byte(intvalue:int, endian:str="<"):
    numbyte = 1
    bigbyte = intvalue.to_bytes(numbyte, byteorder="big", signed=False) # OverflowError: int too big to convert
    varbyte = __convert__endian__(bigbyte, endian)
    return varbyte


def uint16_2_byte(intvalue:int, endian:str="<"):
    numbyte = 2
    bigbyte = intvalue.to_bytes(numbyte, byteorder="big", signed=False) # OverflowError: int too big to convert
    varbyte = __convert__endian__(bigbyte, endian)
    return varbyte


def uint32_2_byte(intvalue:int, endian:str="<"):
    numbyte = 4
    bigbyte = intvalue.to_bytes(numbyte, byteorder="big", signed=False) # OverflowError: int too big to convert
    varbyte = __convert__endian__(bigbyte, endian)
    return varbyte


def uint64_2_byte(intvalue:int, endian:str="<"):
    numbyte = 8
    bigbyte = intvalue.to_bytes(numbyte, byteorder="big", signed=False) # OverflowError: int too big to convert
    varbyte = __convert__endian__(bigbyte, endian)
    return varbyte



# IEEE-754
class struct_float:
    exponentzero = ""
    exponentoneo = ""
    fractionzero = ""
    signedsfmask = ""
    exponentmask = ""
    fractionmask = ""

    @staticmethod
    def from_bytes(bigbyte:bytes, numsignedsfbit:int=1, numexponentbit:int=5, numfractionbit:int=10):
        bin8none = struct_float.__assert__floatbyte__(bigbyte, numsignedsfbit, numexponentbit, numfractionbit)
        bin8char = struct_float.__convert__byte2bin__(bigbyte)
        signedsf = struct_float.按位与(bin8char, struct_float.signedsfmask)[0: 1]
        exponent = struct_float.按位与(bin8char, struct_float.exponentmask)[1: 1+numexponentbit]
        fraction = struct_float.按位与(bin8char, struct_float.fractionmask)[numsignedsfbit+numexponentbit: ]
        if exponent == struct_float.exponentzero and fraction == struct_float.fractionzero: return struct_float.正负零(signedsf)
        if exponent == struct_float.exponentzero and fraction != struct_float.fractionzero: return struct_float.非规格化小数(signedsf, fraction)
        if exponent == struct_float.exponentoneo and fraction == struct_float.fractionzero: return struct_float.正负无穷大(signedsf)
        if exponent == struct_float.exponentoneo and fraction != struct_float.fractionzero: return float("nan")
        return struct_float.规格化小数(signedsf, exponent, fraction)


    def __assert__floatbyte__(bigbyte:bytes, numsignedsfbit:int=1, numexponentbit:int=5, numfractionbit:int=10):
        numinputbit = len(bigbyte) * 8
        numfinfobit = numsignedsfbit + numexponentbit + numfractionbit
        if numinputbit != numfinfobit: raise ValueError(f"bgstruct::struct_float::__assert__floatbyte__::字节{bigbyte}与bit位数[{numsignedsfbit}, {numexponentbit}, {numfractionbit}]不对应]")
        struct_float.exponentzero = numexponentbit*"0"
        struct_float.exponentoneo = numexponentbit*"1"
        struct_float.fractionzero = numfractionbit*"0"
        struct_float.signedsfmask = numsignedsfbit*"1" + numexponentbit*"0" + numfractionbit*"0"
        struct_float.exponentmask = numsignedsfbit*"0" + numexponentbit*"1" + numfractionbit*"0"
        struct_float.fractionmask = numsignedsfbit*"0" + numexponentbit*"0" + numfractionbit*"1"


    @staticmethod
    def __convert__byte2bin__(bigbyte:bytes):
        buffer = [bin(uint8)[2:] for uint8 in bigbyte]
        bin8buffer = []
        for binchar in buffer:
            number0 = 8 - len(binchar)
            padchar = number0 * "0"
            bin8buffer.append( padchar + binchar )
        bin8char = ""
        for binchar in bin8buffer: bin8char += binchar
        assert bin8char != "", "bgstruct::struct_float::__convert__byte2bin__::字节转二进制出错"
        return bin8char


    @staticmethod
    def 按位与(bin8char:str, maskchar:str):
        assert len(bin8char) == len(maskchar), "bgstruct::struct_float::按位与::数据位数与掩码位数不一致"
        result = ""
        for a, b in zip(bin8char, maskchar):
            buffer = "0"
            if a == "1" and b == "1": buffer = "1"
            result += buffer
        return result
    

    @staticmethod
    def 正负零(signedsf:str):
        result = float("+0.0")
        if signedsf == "1": result = float("-0.0")
        return result


    @staticmethod
    def 正负无穷大(signedsf:str):
        result = float("+inf")
        if signedsf == "1": result = float("-inf")
        return result


    @staticmethod
    def 二进制尾数to十进制小数(fraction:str):
        number = len(fraction)
        result = 0.0
        for i in range(number):
            j = i + 1
            if fraction[i] == "1": result += (2**(-j))
        return result


    @staticmethod
    def 非规格化小数(signedsf:str, fraction:str):
        小数 = struct_float.二进制尾数to十进制小数(fraction)
        if signedsf == "1": 小数 = -小数
        return 小数
        

    @staticmethod
    def 规格化小数(signedsf:str, exponent:str, fraction:str):
        减数 = 2**(len(exponent)-1) - 1
        指数 = int(exponent, 2) - 减数
        小数 = struct_float.二进制尾数to十进制小数(fraction)
        底数 = 1.0 + 小数
        结果 = 底数 * 2 ** 指数
        if signedsf == "1": 结果 = -结果
        return 结果


    @staticmethod
    def to_bytes(floatvalue:float, numbyte:int, numsignedsfbit:int=1, numexponentbit:int=5, numfractionbit:int=10):
        struct_float.__assert__floatvalue__(numbyte, numsignedsfbit, numexponentbit, numfractionbit)
        singed, intbinchar, fntbinchar = struct_float.__convert__float2bin__(floatvalue, numexponentbit, numfractionbit)
        fraction, 指数 = struct_float.二进制位to二进制指数(intbinchar, fntbinchar, numfractionbit)
        减数 = 2**(numexponentbit-1) - 1
        存数 = 指数 + 减数
        exponent = bin(存数)[2:]
        padchar = ( numexponentbit - len(exponent) ) * "0"
        floatbinchar = singed + padchar + exponent + fraction
        bufferuint8 = []
        for i in range(0, len(floatbinchar), 8): bufferuint8.append(int(floatbinchar[i:i+8], 2))
        bigbyte = bytes(bufferuint8)
        return bigbyte


    @staticmethod
    def __assert__floatvalue__(numbyte:int, numsignedsfbit:int=1, numexponentbit:int=5, numfractionbit:int=10):
        if numbyte not in [2, 4, 8]: raise OverflowError(f"can't support numbyte:{numbyte}, must be in {[2, 4, 8]}")
        numinputbit = numbyte * 8
        numfinfobit = numsignedsfbit + numexponentbit + numfractionbit
        if numinputbit != numfinfobit: raise ValueError(f"bgstruct::struct_float::__assert__floatvalue__::字节数{numbyte}与bit位数[{numsignedsfbit}, {numexponentbit}, {numfractionbit}]不对应]")
        

    @staticmethod
    def 十进制小数to二进制小数(fntvalue:float, numexponentbit:int=5, numfractionbit:int=10):
        fntbinchar = ""
        小数位数 = 2**numexponentbit + numfractionbit
        for i in range(小数位数):
            buffer = "0"
            fntvalue = fntvalue * 2
            if int(fntvalue) >= 1: 
                buffer = "1"
                fntvalue = fntvalue % 1
            fntbinchar += buffer
        return fntbinchar


    @staticmethod
    def __convert__float2bin__(floatvalue:float, numexponentbit:int=5, numfractionbit:int=10):
        singedchar = "0"
        if floatvalue < 0: 
            singedchar = "1"
            floatvalue = -floatvalue
        intvalue, fntvalue = int(floatvalue), floatvalue % 1
        intbinchar, fntbinchar = bin(intvalue)[2:], struct_float.十进制小数to二进制小数(fntvalue, numexponentbit, numfractionbit)
        # 减数 = 2**(numexponentbit-1) - 1
        # if len(intbinchar) > 减数: raise OverflowError(f"float too big to convert, 超过了{(2**减数)-1}")
        if len(intbinchar) > numfractionbit: raise OverflowError(f"floatX too big to convert, 超过了{(2**numfractionbit)-1}")
        return singedchar, intbinchar, fntbinchar


    @staticmethod
    def 二进制位to二进制指数(intbinchar:str, fntbinchar:str, numfractionbit:int=10):
        if intbinchar == "0":
            buffer = 0
            fraction = numfractionbit * "0"
            for i in range(len(fntbinchar)):
                if fntbinchar[i] == "1": 
                    buffer = -(i+1)
                    fraction = fntbinchar[i+1:i+1+numfractionbit]
                    break
            指数 = buffer
        else:
            binchar = intbinchar + fntbinchar
            fraction = binchar[1:1+numfractionbit]
            指数 = len(intbinchar) - 1
        return fraction, 指数



def byte_2_float16(varbyte:bytes, endian:str="<"):
    numbyte = 2
    assert len(varbyte) == numbyte, f"bgstruct::byte_2_float16::输入的字节长度!={numbyte}"
    bigbyte = __convert__endian__(varbyte, endian)
    return struct_float.from_bytes(bigbyte, 1, 5, 10)


def byte_2_float32(varbyte:bytes, endian:str="<"):
    numbyte = 4
    assert len(varbyte) == numbyte, f"bgstruct::byte_2_float32::输入的字节长度!={numbyte}"
    bigbyte = __convert__endian__(varbyte, endian)
    return struct_float.from_bytes(bigbyte, 1, 8, 23)


def byte_2_float64(varbyte:bytes, endian:str="<"):
    numbyte = 8
    assert len(varbyte) == numbyte, f"bgstruct::byte_2_float64::输入的字节长度!={numbyte}"
    bigbyte = __convert__endian__(varbyte, endian)
    return struct_float.from_bytes(bigbyte, 1, 11, 52)



# varhex = "0000803F" # 0, 0, 128, 63  # 1.0 # 0b00000000, 0b00000000 0b10000000, 0b00111111
# varhex = "1E0DB0CA" # 30, 13, 176, 202, # -5768847.0  # '0b11001010' , '0b10110000', '0b00001101', '0b00011110' # '0b00011110', '0b00001101', '0b10110000', '0b11001010'
# varbyte = bytes.fromhex(varhex)
# a = byte_2_float32(varbyte)

# varhex = "1E0DB0CACEFA11D0" 
# varbyte = bytes.fromhex(varhex)
# b = byte_2_float64(varbyte) # -5.204772746436576e+77

# varhex = "1E0D" 
# varhex = "B0CA" 
# varbyte = bytes.fromhex(varhex)
# c = byte_2_float16(varbyte) # 0.0003123283386230469, -13.375
# pass



def float16_2_byte(floatvalue:float, endian:str="<"): 
    numbyte = 2
    bigbyte = struct_float.to_bytes(floatvalue, numbyte, 1, 5, 10) 
    varbyte = __convert__endian__(bigbyte, endian)
    return varbyte


def float32_2_byte(floatvalue:float, endian:str="<"): 
    numbyte = 4
    bigbyte = struct_float.to_bytes(floatvalue, numbyte, 1, 8, 23) 
    varbyte = __convert__endian__(bigbyte, endian)
    return varbyte


def float64_2_byte(floatvalue:float, endian:str="<"): 
    numbyte = 8
    bigbyte = struct_float.to_bytes(floatvalue, numbyte, 1, 11, 52) 
    varbyte = __convert__endian__(bigbyte, endian)
    return varbyte


# aa = float16_2_byte(0.0003123283386230469)
# ab = float16_2_byte(-13.375)
# ac = float16_2_byte(0.00000000005123283386230469)
# ad = float32_2_byte(1.0)
# ae = float32_2_byte(-5768847.0)
# af = float64_2_byte(-5.204772746436576e+77) # 
# pass


