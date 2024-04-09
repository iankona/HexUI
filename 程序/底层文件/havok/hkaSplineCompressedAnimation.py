# https://github.com/CucFlavius/Zee-010-Templates/blob/main/HKSplineCompressedAnimation.bt
# https://github.com/CucFlavius/Zee-010-Templates/blob/main/FFXIV_PAP.bt_bak
# https://github.com/Meowmaritus/SoulsAssetPipeline/blob/master/SoulsAssetPipeline/Animation/HKX/SplineCompressedAnimation.cs

from .havokbinary import readhavokint


def hkaSplineCompressedAnimation(bp, flag, hkafile):
    if flag == 49127: return hkaSplineCompressedAnimation49127(bp, hkafile)
    if flag == 40951: return hkaSplineCompressedAnimation40951(bp, hkafile)
    if flag == 40935: return hkaSplineCompressedAnimation40935(bp, hkafile)


class hkaSplineCompressedAnimation49127:
    def __init__(self, bp, hkafile):
        # hkaAnimation
        self.type = readhavokint(bp)
        self.duration = bp.readfloat32()
        self.numberOfTransformTracks = readhavokint(bp)

        annotationTrackNumber = readhavokint(bp) # 骨骼名称列表
        annotationTrackType, self.annotationTracks = readhavokint(bp), [hkafile.readname(bp) for i in range(annotationTrackNumber) ]
        # hkaSplineCompressedAnimation
        self.numFrames = readhavokint(bp)
        self.numBlocks = readhavokint(bp)
        self.maxFramesPerBlock = readhavokint(bp)
        self.maskAndQuantizationSize = readhavokint(bp)
        self.blockDuration = bp.readfloat32()
        self.blockInverseDuration = bp.readfloat32()
        self.frameDuration = bp.readfloat32()

        blockOffsetNumber = readhavokint(bp)
        blockOffsetType, self.blockOffsets = readhavokint(bp), [ readhavokint(bp) for i in range(blockOffsetNumber) ]

        floatBlockOffsetNumber = readhavokint(bp)
        floatBlockOffsetType, self.floatBlockOffsets = readhavokint(bp), [ readhavokint(bp) for i in range(floatBlockOffsetNumber) ]

        transformOffsetNumber = readhavokint(bp)
        transformOffsetType, self.transformOffsets = readhavokint(bp), [ readhavokint(bp) for i in range(transformOffsetNumber) ]

        self.data  = bp.readslice( readhavokint(bp) )


class hkaSplineCompressedAnimation40951:
    def __init__(self, bp, hkafile):
        # hkaAnimation
        self.type = readhavokint(bp)
        self.duration = bp.readfloat32()
        self.numberOfTransformTracks = readhavokint(bp)
        self.extractedMotion = readhavokint(bp) - 1

        annotationTrackNumber = readhavokint(bp) # 骨骼名称列表
        annotationTrackType, self.annotationTracks = readhavokint(bp), [hkafile.readname(bp) for i in range(annotationTrackNumber) ]
        # hkaSplineCompressedAnimation
        self.numFrames = readhavokint(bp)
        self.numBlocks = readhavokint(bp)
        self.maxFramesPerBlock = readhavokint(bp)
        self.maskAndQuantizationSize = readhavokint(bp)
        self.blockDuration = bp.readfloat32()
        self.blockInverseDuration = bp.readfloat32()
        self.frameDuration = bp.readfloat32()

        blockOffsetNumber = readhavokint(bp)
        blockOffsetType, self.blockOffsets = readhavokint(bp), [ readhavokint(bp) for i in range(blockOffsetNumber) ]

        floatBlockOffsetNumber = readhavokint(bp)
        floatBlockOffsetType, self.floatBlockOffsets = readhavokint(bp), [ readhavokint(bp) for i in range(floatBlockOffsetNumber) ]

        self.data  = bp.readslice( readhavokint(bp) )


class hkaSplineCompressedAnimation40935:
    def __init__(self, bp, hkafile):
        # hkaAnimation
        self.type = readhavokint(bp)
        self.duration = bp.readfloat32()
        self.numberOfTransformTracks = readhavokint(bp)

        annotationTrackNumber = readhavokint(bp) # 骨骼名称列表
        annotationTrackType, self.annotationTracks = readhavokint(bp), [hkafile.readname(bp) for i in range(annotationTrackNumber) ]
        # hkaSplineCompressedAnimation
        self.numFrames = readhavokint(bp)
        self.numBlocks = readhavokint(bp)
        self.maxFramesPerBlock = readhavokint(bp)
        self.maskAndQuantizationSize = readhavokint(bp)
        self.blockDuration = bp.readfloat32()
        self.blockInverseDuration = bp.readfloat32()
        self.frameDuration = bp.readfloat32()

        blockOffsetNumber = readhavokint(bp)
        blockOffsetType, self.blockOffsets = readhavokint(bp), [ readhavokint(bp) for i in range(blockOffsetNumber) ]

        floatBlockOffsetNumber = readhavokint(bp)
        floatBlockOffsetType, self.floatBlockOffsets = readhavokint(bp), [ readhavokint(bp) for i in range(floatBlockOffsetNumber) ]

        self.data  = bp.readslice( readhavokint(bp) )





