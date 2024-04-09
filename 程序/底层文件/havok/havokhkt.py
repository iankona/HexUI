from .havokfile import havokfile


from .hkRootLevelContainer import hkRootLevelContainer
from .hkpPhysicsData import hkpPhysicsData
from .hkpPhysicsSystem import hkpPhysicsSystem
from .hkpRigidBody import hkpRigidBody
from .hkpBvCompressedMeshShape import hkpBvCompressedMeshShape


hktclasses = {
    'hkRootLevelContainer': hkRootLevelContainer,
    'hkpPhysicsData': hkpPhysicsData,
    "hkpPhysicsSystem": hkpPhysicsSystem,
    "hkpRigidBody": hkpRigidBody,
    "hkpBvCompressedMeshShape": hkpBvCompressedMeshShape,
}



class hktfile(havokfile):
    def __init__(self, bp): super().__init__(bp, hktclasses)
