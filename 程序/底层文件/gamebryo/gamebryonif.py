from  .gamebryofile import gamebryofile

from .NiNode import NiNode
from .NiZBufferProperty import NiZBufferProperty
from .NiVertexColorProperty import NiVertexColorProperty
from .NiStringExtraData import NiStringExtraData
from .NiFloatExtraData import NiFloatExtraData
from .NiTransformController import NiTransformController
from .NiIntegerExtraData import NiIntegerExtraData
from .NiMesh import NiMesh
from .NiMorphWeightsController import NiMorphWeightsController
from .NiTexturingProperty import NiTexturingProperty
from .NiSourceTexture import NiSourceTexture
from .NiSpecularProperty import NiSpecularProperty
from .NiMaterialProperty import NiMaterialProperty
from .NiDataStream import NiDataStream
from .NiMorphMeshModifier import NiMorphMeshModifier
from .NiAlphaProperty import NiAlphaProperty
from .NiCamera import NiCamera
from .NiPointLight import NiPointLight
from .NiSkinningMeshModifier import NiSkinningMeshModifier
from .NiAmbientLight import NiAmbientLight
from .NiFloatsExtraData import NiFloatsExtraData
from .NiIntegersExtraData import NiIntegersExtraData
from .NiExtraData import NiExtraData
from .NiTextKeyExtraData import NiTextKeyExtraData
from .NiTransformInterpolator import NiTransformInterpolator
from .NiTransformData import NiTransformData
from .NiInstancingMeshModifier import NiInstancingMeshModifier
from .NiMeshHWInstance import NiMeshHWInstance
from .NiStencilProperty import NiStencilProperty

nifclasses = {
    "NiNode": NiNode,
    "NiZBufferProperty": NiZBufferProperty,
    "NiVertexColorProperty": NiVertexColorProperty,
    "NiStringExtraData": NiStringExtraData,
    "NiFloatExtraData": NiFloatExtraData,
    "NiTransformController": NiTransformController,
    "NiIntegerExtraData": NiIntegerExtraData,
    "NiMesh": NiMesh,
    "NiMorphWeightsController": NiMorphWeightsController,
    "NiTexturingProperty": NiTexturingProperty,
    "NiSourceTexture": NiSourceTexture,
    "NiSpecularProperty": NiSpecularProperty,
    "NiMaterialProperty": NiMaterialProperty,
    "NiDataStream": NiDataStream,
    "NiMorphMeshModifier": NiMorphMeshModifier,
    "NiAlphaProperty": NiAlphaProperty,
    "NiCamera": NiCamera,
    "NiPointLight": NiPointLight,
    "NiSkinningMeshModifier": NiSkinningMeshModifier,
    "NiAmbientLight": NiAmbientLight,
    "NiFloatsExtraData": NiFloatsExtraData,
    "NiIntegersExtraData": NiIntegersExtraData,
    "NiExtraData": NiExtraData,
    "NiTextKeyExtraData": NiTextKeyExtraData,
    "NiTransformInterpolator": NiTransformInterpolator,
    "NiTransformData": NiTransformData,
    "NiInstancingMeshModifier": NiInstancingMeshModifier,
    "NiMeshHWInstance": NiMeshHWInstance,
    "NiStencilProperty": NiStencilProperty,
}


class niffile(gamebryofile):
    def __init__(self, bp): super().__init__(bp, nifclasses)