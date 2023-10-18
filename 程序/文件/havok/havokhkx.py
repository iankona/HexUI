from .havokfile import havokfile


from .hkRootLevelContainer import hkRootLevelContainer
from .hkaAnimationContainer import hkaAnimationContainer
from .hkaBoneAttachment import hkaBoneAttachment
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
# from .hkaInterleavedUncompressedAnimation import hkaInterleavedUncompressedAnimation
# from .hkSimpleLocalFrame import hkSimpleLocalFrame

from . import hkaSplineCompressedAnimationData
# from . import hkaInterleavedUncompressedAnimationData

hkxclasses = {
    'hkRootLevelContainer': hkRootLevelContainer,
    'hkaAnimationContainer': hkaAnimationContainer,
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

    # 'hkaInterleavedUncompressedAnimation':hkaInterleavedUncompressedAnimation,
    # 'hkSimpleLocalFrame': hkSimpleLocalFrame,
}



class hkxfile(havokfile):
    def __init__(self, bp): super().__init__(bp, hkxclasses)
