from . import stacke
from . import struct_python

class 类(stacke.类):
    def __init__(self, mpbyte=None, endian="<"):
        stacke.类.__init__(self, mpbyte=mpbyte, endian=endian, module=struct_python)



    
