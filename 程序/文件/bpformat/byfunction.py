from . import bpbytes
from . import bpnumpy
from . import bpnumpyinverted
from . import bgpython


from . import function_bpbytes
from . import function_bpnumpy
from . import function_bpnumpy_inverted

from . import function_bgpython


def context(o):
    classname = str(type(o))
    if "bpbytes.类" in classname:
        function = function_bpbytes
        function_bpbytes.context_bpbytes.data = o
        
    if "bpnumpy.类" in classname:
        function = function_bpnumpy
        function_bpnumpy.context_bpnumpy.data = o

    if "bpnumpyinverted.类" in classname:
        function = function_bpnumpy_inverted
        function_bpnumpy_inverted.context_bpnumpy_inverted.data = o

    if "bgpython.类" in classname:
        function = function_bgpython
        function_bgpython.context_bgpython.node = o
    return function