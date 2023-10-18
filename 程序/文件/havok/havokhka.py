from  .havokfile import havokfile

from .hkRootLevelContainer import hkRootLevelContainer
from .hkaAnimationContainer import hkaAnimationContainer
from .hkaAnimationBinding import hkaAnimationBinding
from .hkaSkeleton import hkaSkeleton
from .hkaSplineCompressedAnimation import hkaSplineCompressedAnimation
from .hkaDefaultAnimatedReferenceFrame import hkaDefaultAnimatedReferenceFrame
from .hkaInterleavedUncompressedAnimation import hkaInterleavedUncompressedAnimation
from .hkSimpleLocalFrame import hkSimpleLocalFrame



hkaclasses = {
    'hkRootLevelContainer': hkRootLevelContainer,
    'hkaAnimationContainer': hkaAnimationContainer,
    'hkaAnimationBinding': hkaAnimationBinding,
    'hkaSkeleton': hkaSkeleton,
    'hkaSplineCompressedAnimation': hkaSplineCompressedAnimation,
    'hkaDefaultAnimatedReferenceFrame': hkaDefaultAnimatedReferenceFrame,
    'hkaInterleavedUncompressedAnimation':hkaInterleavedUncompressedAnimation,
    'hkSimpleLocalFrame': hkSimpleLocalFrame,
}





class hkafile(havokfile):
    def __init__(self, bp): super().__init__(bp, hkaclasses)






