from .anarchyfile import anarchyfile

from .SRVT import SRVT
from .FNOC import FNOC
from .MOEG import MOEG


vtcclasses = {
    'SRVT': SRVT,
    'FNOC': FNOC,
    'MOEG': MOEG,
}


class vtcfile(anarchyfile):
    def __init__(self, bp): super().__init__(bp, vtcclasses)