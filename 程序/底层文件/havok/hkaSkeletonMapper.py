from .havokbinary import readhavokint, readhavokintseek0


def hkaSkeletonMapper(bp, flag, havokfile):
    if flag == 2: return hkaSkeletonMapper2(bp, havokfile)



class hkaSkeletonMapper2:
    def __init__(self, bp, havokfile):
        hkaSkeletonMapperData_flag = bp.readuint16() 
        # print(hkaSkeletonMapperData_flag)
        if hkaSkeletonMapperData_flag == 803: self.mapping = hkaSkeletonMapperData803(bp, havokfile) # [35, 3]
        if hkaSkeletonMapperData_flag == 867: self.mapping = hkaSkeletonMapperData867(bp, havokfile) # [99, 3]
        if hkaSkeletonMapperData_flag == 899: self.mapping = hkaSkeletonMapperData899(bp, havokfile) # [131,3]
        if hkaSkeletonMapperData_flag == 995: self.mapping = hkaSkeletonMapperData995(bp, havokfile) # [227,3]




class hkaSkeletonMapperData803:
    def __init__(self, bp, havokfile):
        self.skeletonA = readhavokint(bp) - 1
        self.skeletonB = readhavokint(bp) - 1

        numsimpleMapping = readhavokint(bp)
        data1 = bp.readuint8()
        boneAtype, self.boneAs = readhavokint(bp), [readhavokint(bp) for i in range(numsimpleMapping)]
        boneBtype, self.boneBs = readhavokint(bp), [readhavokint(bp) for i in range(numsimpleMapping)]
        self.aFromBTransforms = [ [bp.readfloat32(4), bp.readfloat32(4), bp.readfloat32(4)] for i in range(numsimpleMapping)]

        self.extractedMotionMapping = [bp.readfloat32(4), bp.readfloat32(4), bp.readfloat32(4)]
        self.keepUnmappedLocal = bp.readuint8()



class hkaSkeletonMapperData867:
    def __init__(self, bp, havokfile):
        self.skeletonA = readhavokint(bp) - 1
        self.skeletonB = readhavokint(bp) - 1

        numsimpleMapping = readhavokint(bp)
        name3 = havokfile.readname(bp)
        boneAtype, self.boneAs = readhavokint(bp), [readhavokint(bp) for i in range(numsimpleMapping)]
        boneBtype, self.boneBs = readhavokint(bp), [readhavokint(bp) for i in range(numsimpleMapping)]
        self.aFromBTransforms = [ [bp.readfloat32(4), bp.readfloat32(4), bp.readfloat32(4)] for i in range(numsimpleMapping)]

        number2 = bp.readuint8()
        data2 = bp.readuint8()
        boneCtype, self.boneCs = readhavokint(bp), [readhavokint(bp) for i in range(number2+1)]
        boneDtype, self.boneDs = readhavokint(bp), [readhavokint(bp) for i in range(number2+1)]
        self.cFromDTransforms = [ [bp.readfloat32(4), bp.readfloat32(4), bp.readfloat32(4)] for i in range(number2)]

        self.extractedMotionMapping = [bp.readfloat32(4), bp.readfloat32(4), bp.readfloat32(4)]
        self.keepUnmappedLocal = bp.readuint8()


class hkaSkeletonMapperData899:
    def __init__(self, bp, havokfile):
        self.skeletonA = readhavokint(bp) - 1
        self.skeletonB = readhavokint(bp) - 1

        numsimpleMapping = readhavokint(bp)
        boneAtype, self.boneAs = readhavokint(bp), [readhavokint(bp) for i in range(numsimpleMapping)]

        self.extractedMotionMapping = [bp.readfloat32(4), bp.readfloat32(4), bp.readfloat32(4)]
        self.keepUnmappedLocal = bp.readuint8()


class hkaSkeletonMapperData995:
    def __init__(self, bp, havokfile):
        self.skeletonA = readhavokint(bp) - 1
        self.skeletonB = readhavokint(bp) - 1

        numsimpleMapping = readhavokint(bp)
        data1 = bp.readuint8()
        boneAtype, self.boneAs = readhavokint(bp), [readhavokint(bp) for i in range(numsimpleMapping)]
        boneBtype, self.boneBs = readhavokint(bp), [readhavokint(bp) for i in range(numsimpleMapping)]
        self.aFromBTransforms = [ [bp.readfloat32(4), bp.readfloat32(4), bp.readfloat32(4)] for i in range(numsimpleMapping)]

        number2 = bp.readuint8()
        data2 = bp.readuint8()
        boneCtype, self.boneCs = readhavokint(bp), [readhavokint(bp) for i in range(number2+1)]
        boneDtype, self.boneDs = readhavokint(bp), [readhavokint(bp) for i in range(number2+1)]
        self.cFromDTransforms = [ [bp.readfloat32(4), bp.readfloat32(4), bp.readfloat32(4)] for i in range(number2)]

        number3 = readhavokint(bp)
        boneEtype, self.boneEs = readhavokint(bp), [readhavokint(bp) for i in range(number3)]

        self.extractedMotionMapping = [bp.readfloat32(4), bp.readfloat32(4), bp.readfloat32(4)]
        self.keepUnmappedLocal = bp.readuint8()