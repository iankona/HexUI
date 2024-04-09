
import numpy as np


def byte_2_int8(varbyte:bytes, endian:str="<"):
    numbyte = 1
    assert len(varbyte) == numbyte, f"struct_bnumpy::byte_2_int8::输入的字节长度!={numbyte}"
    return list(np.frombuffer(varbyte, dtype=endian+"i1", count=1))[0] # value([1], dtype="int8")


def byte_2_int16(varbyte:bytes, endian:str="<"):
    numbyte = 2
    assert len(varbyte) == numbyte, f"struct_bnumpy::byte_2_int16::输入的字节长度!={numbyte}"
    return list(np.frombuffer(varbyte, dtype=endian+"i2", count=1))[0]


def byte_2_int32(varbyte:bytes, endian:str="<"):
    numbyte = 4
    assert len(varbyte) == numbyte, f"struct_bnumpy::byte_2_int32::输入的字节长度!={numbyte}"
    return list(np.frombuffer(varbyte, dtype=endian+"i4", count=1))[0]


def byte_2_int64(varbyte:bytes, endian:str="<"):
    numbyte = 8
    assert len(varbyte) == numbyte, f"struct_bnumpy::byte_2_int64::输入的字节长度!={numbyte}"
    return list(np.frombuffer(varbyte, dtype=endian+"i8", count=1))[0]


def byte_2_uint8(varbyte:bytes, endian:str="<"):
    numbyte = 1
    assert len(varbyte) == numbyte, f"struct_bnumpy::byte_2_uint8::输入的字节长度!={numbyte}"
    return list(np.frombuffer(varbyte, dtype=endian+"u1", count=1))[0]


def byte_2_uint16(varbyte:bytes, endian:str="<"):
    numbyte = 2
    assert len(varbyte) == numbyte, f"struct_bnumpy::byte_2_uint16::输入的字节长度!={numbyte}"
    return list(np.frombuffer(varbyte, dtype=endian+"u2", count=1))[0]


def byte_2_uint32(varbyte:bytes, endian:str="<"):
    numbyte = 4
    assert len(varbyte) == numbyte, f"struct_bnumpy::byte_2_uint32::输入的字节长度!={numbyte}"
    return list(np.frombuffer(varbyte, dtype=endian+"u4", count=1))[0]


def byte_2_uint64(varbyte:bytes, endian:str="<"):
    numbyte = 8
    assert len(varbyte) == numbyte, f"struct_bnumpy::byte_2_uint64::输入的字节长度!={numbyte}"
    return list(np.frombuffer(varbyte, dtype=endian+"u8", count=1))[0]



def byte_2_float16(varbyte:bytes, endian:str="<"):
    numbyte = 2
    assert len(varbyte) == numbyte, f"struct_bnumpy::byte_2_float16::输入的字节长度!={numbyte}"
    return list(np.frombuffer(varbyte, dtype=endian+"f2", count=1))[0]


def byte_2_float32(varbyte:bytes, endian:str="<"):
    numbyte = 4
    assert len(varbyte) == numbyte, f"struct_bnumpy::byte_2_float32::输入的字节长度!={numbyte}"
    return list(np.frombuffer(varbyte, dtype=endian+"f4", count=1))[0]


def byte_2_float64(varbyte:bytes, endian:str="<"):
    numbyte = 8
    assert len(varbyte) == numbyte, f"struct_bnumpy::byte_2_float64::输入的字节长度!={numbyte}"
    return list(np.frombuffer(varbyte, dtype=endian+"f8", count=1))[0]


