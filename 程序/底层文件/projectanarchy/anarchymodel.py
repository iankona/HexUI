from .anarchyfile import anarchyfile

# 古剑3
from .HSMV import HSMV
from .SRTM import SRTM
from .MBUS import MBUS
from .RPXE import RPXE
from .LEKS import LEKS
from .THGW import THGW
from .XBBB import XBBB
from .RPBC import RPBC
from .SDNB import SDNB
# 古剑2
from .HSMS import HSMS
from .RMSV import RMSV
from .RMST import RMST

modelclasses = {
    'HSMV': HSMV,
    'SRTM': SRTM,
    'MBUS': MBUS,
    'RPXE': RPXE,
    'LEKS': LEKS,
    'THGW': THGW,
    'XBBB': XBBB,
    'RPBC': RPBC,
    'SDNB': SDNB,

    'HSMS': HSMS,
    'RMSV': RMSV,
    'RMST': RMST,
}


class modelfile(anarchyfile):
    def __init__(self, bp): super().__init__(bp, modelclasses)

