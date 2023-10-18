from . import context_bpnumpy_inverted
from . import bpnumpyinverted

class filepath(context_bpnumpy_inverted.类):
    def __init__(self, filepath):
        self.bped = bpnumpyinverted.类().filepath(filepath)
        self.bp = self.bped.tobpnumpy()


def close(): return context_bpnumpy_inverted.data.close()
def size(): return context_bpnumpy_inverted.data.size()
def remainsize(): return context_bpnumpy_inverted.data.remainsize()


class copy(context_bpnumpy_inverted.类):
    def __init__(self):
        self.bped = context_bpnumpy_inverted.data.copy()
        self.bp = self.bped.tobpnumpy()

class readslice(context_bpnumpy_inverted.类):
    def __init__(self, num=1):
        self.bped = context_bpnumpy_inverted.data.readslice(num)
        self.bp = self.bped.tobpnumpy()

class readremainslice(context_bpnumpy_inverted.类):
    def __init__(self):
        self.bped = context_bpnumpy_inverted.data.readremainslice()
        self.bp = self.bped.tobpnumpy()


class readsliceseek0(context_bpnumpy_inverted.类):
    def __init__(self, num=1):
        self.bped = context_bpnumpy_inverted.data.readsliceseek0(num)
        self.bp = self.bped.tobpnumpy()


class readremainsliceseek0(context_bpnumpy_inverted.类):
    def __init__(self):
        self.bped = context_bpnumpy_inverted.data.readremainsliceseek0()
        self.bp = self.bped.tobpnumpy()


def tobpnumpy(): return context_bpnumpy_inverted.data.tobpnumpy()
