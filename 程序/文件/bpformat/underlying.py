
from . import underlying_numpy
from . import underlying_struct
from . import underlying_python



class 类(underlying_numpy.类):
    def __init__(self, mpbyte=None, endian="<"):
        # super().__init__(mpbyte, endian)
        underlying_numpy.类.__init__(self, mpbyte=mpbyte, endian=endian)


# class 类(typestruct.类):
#     def __init__(self, mpbyte=None, endian="<"):
#         super().__init__(self, mpbyte=mpbyte, endian=endian)


# class 类(typepython.类):
#     def __init__(self, mpbyte=None, endian="<"):
#         super().__init__(self, mpbyte=mpbyte, endian=endian)