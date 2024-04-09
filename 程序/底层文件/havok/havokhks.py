from .havokfile import havokfile


from .hkRootLevelContainer import hkRootLevelContainer
from .hkaAnimationContainer import hkaAnimationContainer
from .hkaAnimationBinding import hkaAnimationBinding
from .hkaSkeleton import hkaSkeleton
from .hkSimpleLocalFrame import hkSimpleLocalFrame

from .hkaSkeletonMapper import hkaSkeletonMapper
from .hkaMeshBinding import hkaMeshBinding

from .hkpPhysicsData import hkpPhysicsData
from .hkpPhysicsSystem import hkpPhysicsSystem
from .hkpRigidBody import hkpRigidBody
from .hkpBvCompressedMeshShape import hkpBvCompressedMeshShape

hksclasses = {
    'hkRootLevelContainer': hkRootLevelContainer,
    'hkaAnimationContainer': hkaAnimationContainer,
    'hkaAnimationBinding': hkaAnimationBinding,
    'hkaSkeleton': hkaSkeleton,
    'hkSimpleLocalFrame': hkSimpleLocalFrame,
    
    'hkaSkeletonMapper': hkaSkeletonMapper,
    'hkaMeshBinding': hkaMeshBinding,

    'hkpPhysicsData': hkpPhysicsData,
    "hkpPhysicsSystem": hkpPhysicsSystem,
    "hkpRigidBody": hkpRigidBody,
    "hkpBvCompressedMeshShape": hkpBvCompressedMeshShape,
}



class hksfile(havokfile):
    def __init__(self, bp): super().__init__(bp, hksclasses)
