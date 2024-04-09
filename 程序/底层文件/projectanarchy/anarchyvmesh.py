from .anarchyfile import anarchyfile

# 古剑2
from .HSMV import HSMV
from .SRTM import SRTM
from .MBUS import MBUS
from .RPXE import RPXE
from .LOCV import LOCV
from .AVUV import AVUV


vmeshclasses = {
    'HSMV': HSMV,
    'SRTM': SRTM,
    'MBUS': MBUS,
    'RPXE': RPXE,
    'LOCV': LOCV,
    'AVUV': AVUV,
}


class vmeshfile(anarchyfile):
    def __init__(self, bp): super().__init__(bp, vmeshclasses)