from .anarchyfile import anarchyfile

from .HMCO import HMCO
from .VMCO import VMCO
from .IMCO import IMCO


ocmclasses = {
    'HMCO': HMCO,
    'VMCO': VMCO,
    'IMCO': IMCO,
}


class ocmfile(anarchyfile):
    def __init__(self, bp): super().__init__(bp, ocmclasses)