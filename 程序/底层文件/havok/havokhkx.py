from .havokfile import havokfile


from .hkRootLevelContainer import hkRootLevelContainer
from .hkaAnimationContainer import hkaAnimationContainer
from .hkaBoneAttachment import hkaBoneAttachment
from .hkaMeshBinding import hkaMeshBinding
from .hkxSkinBinding import hkxSkinBinding
from .hkxMesh import hkxMesh
from .hkxMeshSection import hkxMeshSection
from .hkxMaterial import hkxMaterial
from .hkxTextureFile import hkxTextureFile
from .hkxIndexBuffer import hkxIndexBuffer
from .hkxVertexBuffer import hkxVertexBuffer
from .hkaAnimationBinding import hkaAnimationBinding
from .hkaSkeleton import hkaSkeleton
from .hkaSplineCompressedAnimation import hkaSplineCompressedAnimation
from .hkaDefaultAnimatedReferenceFrame import hkaDefaultAnimatedReferenceFrame
from .hkaSkeleton import hkaSkeleton
from .hkxScene import hkxScene
from .hkxNode import hkxNode

hkxclasses = {
    'hkRootLevelContainer': hkRootLevelContainer,
    'hkaAnimationContainer': hkaAnimationContainer,
    'hkaMeshBinding': hkaMeshBinding,
    'hkaBoneAttachment': hkaBoneAttachment,
    'hkxSkinBinding': hkxSkinBinding,
    'hkxMesh': hkxMesh,
    'hkxMeshSection': hkxMeshSection,
    'hkxMaterial': hkxMaterial,
    'hkxTextureFile': hkxTextureFile,
    'hkxIndexBuffer': hkxIndexBuffer,
    'hkxVertexBuffer': hkxVertexBuffer,
    'hkaAnimationBinding': hkaAnimationBinding,
    'hkaSkeleton': hkaSkeleton,
    'hkaSplineCompressedAnimation': hkaSplineCompressedAnimation,
    'hkaDefaultAnimatedReferenceFrame': hkaDefaultAnimatedReferenceFrame,
    'hkaSkeleton': hkaSkeleton,
    'hkxScene': hkxScene,
    'hkxNode': hkxNode,

    # 'hkaInterleavedUncompressedAnimation':hkaInterleavedUncompressedAnimation,
    # 'hkSimpleLocalFrame': hkSimpleLocalFrame,
}


from . import hkaSplineCompressedAnimationData
from . import hkaInterleavedUncompressedAnimationData


class hkxfile(havokfile):
    def __init__(self, bp): 
        super().__init__(bp, hkxclasses)
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
