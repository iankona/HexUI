



def byte_2_int8(varbyte:bytes, endian:str="<"):
    numbyte = 1
    assert len(varbyte) == numbyte, f"struct_struct::byte_2_int8::输入的字节长度!={numbyte}"



def byte_2_int16(varbyte:bytes, endian:str="<"):
    numbyte = 2
    assert len(varbyte) == numbyte, f"struct_struct::byte_2_int16::输入的字节长度!={numbyte}"



def byte_2_int32(varbyte:bytes, endian:str="<"):
    numbyte = 4
    assert len(varbyte) == numbyte, f"struct_struct::byte_2_int32::输入的字节长度!={numbyte}"



def byte_2_int64(varbyte:bytes, endian:str="<"):
    numbyte = 8
    assert len(varbyte) == numbyte, f"struct_struct::byte_2_int64::输入的字节长度!={numbyte}"



def byte_2_uint8(varbyte:bytes, endian:str="<"):
    numbyte = 1
    assert len(varbyte) == numbyte, f"struct_struct::byte_2_uint8::输入的字节长度!={numbyte}"





def byte_2_uint16(varbyte:bytes, endian:str="<"):
    numbyte = 2
    assert len(varbyte) == numbyte, f"struct_struct::byte_2_uint16::输入的字节长度!={numbyte}"



def byte_2_uint32(varbyte:bytes, endian:str="<"):
    numbyte = 4
    assert len(varbyte) == numbyte, f"struct_struct::byte_2_uint32::输入的字节长度!={numbyte}"



def byte_2_uint64(varbyte:bytes, endian:str="<"):
    numbyte = 8
    assert len(varbyte) == numbyte, f"struct_struct::byte_2_uint64::输入的字节长度!={numbyte}"



def int8_2_byte(intvalue:int, endian:str="<"):
    numbyte = 1
    bigbyte = intvalue.to_bytes(numbyte, byteorder="big", signed=True) # OverflowError: int too big to convert



def int16_2_byte(intvalue:int, endian:str="<"):
    numbyte = 2
    bigbyte = intvalue.to_bytes(numbyte, byteorder="big", signed=True) # OverflowError: int too big to convert



def int32_2_byte(intvalue:int, endian:str="<"):
    numbyte = 4
    bigbyte = intvalue.to_bytes(numbyte, byteorder="big", signed=True) # OverflowError: int too big to convert



def int64_2_byte(intvalue:int, endian:str="<"):
    numbyte = 8
    bigbyte = intvalue.to_bytes(numbyte, byteorder="big", signed=True) # OverflowError: int too big to convert




def uint8_2_byte(intvalue:int, endian:str="<"):
    numbyte = 1
    bigbyte = intvalue.to_bytes(numbyte, byteorder="big", signed=False) # OverflowError: int too big to convert



def uint16_2_byte(intvalue:int, endian:str="<"):
    numbyte = 2
    bigbyte = intvalue.to_bytes(numbyte, byteorder="big", signed=False) # OverflowError: int too big to convert



def uint32_2_byte(intvalue:int, endian:str="<"):
    numbyte = 4
    bigbyte = intvalue.to_bytes(numbyte, byteorder="big", signed=False) # OverflowError: int too big to convert



def uint64_2_byte(intvalue:int, endian:str="<"):
    numbyte = 8
    bigbyte = intvalue.to_bytes(numbyte, byteorder="big", signed=False) # OverflowError: int too big to convert






def byte_2_float16(varbyte:bytes, endian:str="<"):
    numbyte = 2
    assert len(varbyte) == numbyte, f"struct_struct::byte_2_float16::输入的字节长度!={numbyte}"



def byte_2_float32(varbyte:bytes, endian:str="<"):
    numbyte = 4
    assert len(varbyte) == numbyte, f"struct_struct::byte_2_float32::输入的字节长度!={numbyte}"



def byte_2_float64(varbyte:bytes, endian:str="<"):
    numbyte = 8
    assert len(varbyte) == numbyte, f"struct_struct::byte_2_float64::输入的字节长度!={numbyte}"







def float16_2_byte(floatvalue:float, endian:str="<"): 
    numbyte = 2



def float32_2_byte(floatvalue:float, endian:str="<"): 
    numbyte = 4


def float64_2_byte(floatvalue:float, endian:str="<"): 
    numbyte = 8





