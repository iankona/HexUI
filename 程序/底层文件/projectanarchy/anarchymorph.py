from .anarchyfile import anarchyfile

from .EHCM import EHCM
from .STCM import STCM


morphclasses = {
    'EHCM': EHCM,
    'STCM': STCM,

}


class morphfile(anarchyfile):
    def __init__(self, bp): super().__init__(bp, morphclasses)