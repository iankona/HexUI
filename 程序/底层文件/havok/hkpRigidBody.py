from .havokbinary import readhavokint, readhavokintseek0


def hkpRigidBody(bp, flag, havokfile):
    if flag == 3228: return hkpRigidBody3228(bp, havokfile)


def hkpRigidBody3228(bp, havokfile):
    hkpEntity_flag = readhavokint(bp) # [224,16], 1072
    if hkpEntity_flag == 1072: return hkpEntity1072(bp, havokfile)


class hkpEntity1072:
    def __init__(self, bp, havokfile):
        char1 = havokfile.readname(bp) # collidable_flag = readhavokint(bp) # -148 # 'numShapeKeysInContactPointProperties'
        shape_index = readhavokint(bp) - 1 # 10, 5-1, 4
        shapeKey = bp.readuint32() + 1 # 4294967294, [254,255,255,255] + 1
        char2 = havokfile.readname(bp) # systems # 31
        forceCollideOntoPpu = bp.readuint8() # 8
        num1, num2 = readhavokint(bp), readhavokint(bp) # [2,2]
        num3 = bp.readuint16() # [238,255], 65518
        num4, num5 = bp.readuint8(), bp.readuint8() # 127, 127
        num6 = bp.readuint8() # 0

        self.name = havokfile.readname(bp) # Bip01#17

        # material
        char2 = havokfile.readname(bp) # 'name'
        num21 = readhavokint(bp) # 1
        friction = bp.readfloat32() # 0.5
        restitution = bp.readfloat32() # 0.4
        # 

        self.damageMultiplier = bp.readfloat32() # 1.0

        num31, char31 = bp.readuint16(), havokfile.readname(bp) # [254,255], 7 # 65534, 'hkRootLevelContainer'
        num32, char32 = bp.readuint16(), havokfile.readname(bp) # [254,255], 7 # 65534, 'hkRootLevelContainer'
        index33, char33 = bp.readuint32(), havokfile.readname(bp) # [254,255,255,255], 31 # 4294967294, 'systems'


        # spuCollisionCallback # [12, 3, 1]
        spuCollisionCallback_index = readhavokint(bp) - 1 # 12
        eventFilter = bp.readuint8() # 3
        userFilter = bp.readuint8() # 1


        # motion
        num41, char41 = readhavokint(bp), havokfile.readname(bp) # [254,19], 5
        deactivationIntegrateCounter = bp.readuint8() # 15
        deactivationNumInactiveFrame_type, deactivationNumInactiveFrames = bp.readuint8(), [readhavokint(bp) for i in range(2)] # 8, [128,128,6], 49152
        char43 = havokfile.readname(bp) # [239,3], -247 # 'numPackedVertices'

        # motionState
        transform = [bp.readfloat32(4), bp.readfloat32(4), bp.readfloat32(4), bp.readfloat32(4)]
        sweptTransform_type, sweptTransform = bp.readuint8(), [bp.readfloat32(4) for i in range(5)]
        deltaAngle = bp.readfloat32(4)
        objectRadius = bp.readfloat32()
        angularDamping = bp.readfloat32()
        timeFactor = bp.readfloat32()
        maxLinearVelocity = bp.readuint8(2) # [1,127] value=127
        maxAngularVelocity = bp.readuint8(2)  # [1,127] value=127
        deactivationClass = bp.readuint8()


        self.inertiaAndMassInv = bp.readfloat32(4)
        self.linearVelocity = bp.readfloat32(4)
        self.angularVelocity = bp.readfloat32(4)
        deactivationRefPosition_type, self.deactivationRefPosition = bp.readuint8(), [bp.readfloat32(3) for i in range(2)]
        deactivationRefOrientation_type, self.deactivationRefOrientation = bp.readuint8(), [bp.readuint8() for i in range(2)]
        self.gravityFactor = bp.readfloat32()
        self.npData, char53 = bp.readuint32(), havokfile.readname(bp) # [254,255,255,255], 31 # 4294967294, 'systems'
        pass