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


from . import hkaSplineCompressedAnimationData
from . import hkaInterleavedUncompressedAnimationData



class hkafile(havokfile):
    def __init__(self, bp): 
        super().__init__(bp, hkaclasses)
        self.readhkaSplineCompressedAnimationData()
        self.readhkaInterleavedUncompressedAnimationData()


    def readhkaSplineCompressedAnimationData(self):
        for [typeflag, classname, classflag, datainstance] in self.datas:
            if classname != "hkaSplineCompressedAnimation": continue
            hkaSplineCompressedAnimationData.函数(self, datainstance)


    def readhkaInterleavedUncompressedAnimationData(self):
        for [typeflag, classname, classflag, datainstance] in self.datas:
            if classname != "hkaInterleavedUncompressedAnimation": continue
            hkaInterleavedUncompressedAnimationData.函数(self, datainstance)

