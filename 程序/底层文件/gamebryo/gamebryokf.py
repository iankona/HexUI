from  .gamebryofile import gamebryofile


from .NiSequenceData import NiSequenceData
from .NiConstFloatEvaluator import NiConstFloatEvaluator
from .NiTransformEvaluator import NiTransformEvaluator
from .NiTransformData import NiTransformData
from .NiConstTransformEvaluator import NiConstTransformEvaluator
from .NiTextKeyExtraData import NiTextKeyExtraData
from .NiBSplineCompTransformEvaluator import NiBSplineCompTransformEvaluator
from .NiBSplineData import NiBSplineData
from .NiBSplineBasisData import NiBSplineBasisData

kfclasses = {
    'NiSequenceData': NiSequenceData,
    'NiConstFloatEvaluator': NiConstFloatEvaluator,
    'NiTransformEvaluator': NiTransformEvaluator,
    'NiTransformData': NiTransformData,
    'NiConstTransformEvaluator': NiConstTransformEvaluator,
    'NiTextKeyExtraData': NiTextKeyExtraData,
    'NiBSplineCompTransformEvaluator': NiBSplineCompTransformEvaluator,
    "NiBSplineData": NiBSplineData,
    'NiBSplineBasisData': NiBSplineBasisData,

}


class kffile(gamebryofile):
    def __init__(self, bp): super().__init__(bp, kfclasses)