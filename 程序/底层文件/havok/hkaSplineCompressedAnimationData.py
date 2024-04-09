# https://github.com/CucFlavius/Zee-010-Templates/blob/main/HKSplineCompressedAnimation.bt
# https://github.com/CucFlavius/Zee-010-Templates/blob/main/FFXIV_PAP.bt_bak
# https://github.com/Meowmaritus/SoulsAssetPipeline/blob/master/SoulsAssetPipeline/Animation/HKX/SplineCompressedAnimation.cs
# https://github.com/System10111/BB_To_DS3/blob/master/Assets/Converter.cs
# https://github.com/OpenAWE-Project/OpenAWE/blob/master/src/awe/havokfile.cpp

# https://github.com/dlunch/FFXIVTools/blob/master/libs/havok_parser/src/spline_compressed_animation.rs
# https://github.com/Meowmaritus/MVDX2/blob/master/MVDX2/Havok/SplineCompressedAnimation.cs


import traceback
import math
from .havokbinary import readhavokquat
from .hkaSplineCompressedAnimationDataBlockValues import uint8_to_fcurves

class Object: pass
def 函数(hkfile, 实例):
    hkfile.action = Object()
    hkaSplineCompressedAnimationSizes(hkfile, 实例, hkfile.action)
    hkaSplineCompressedAnimationBlocks(hkfile, 实例, hkfile.action)
    hkaSplineCompressedAnimationFcurves(hkfile, 实例, hkfile.action)
    hkaSplineCompressedAnimationFcurvesHandle(hkfile, 实例, hkfile.action)


def hkaSplineCompressedAnimationSizes(hkfile, 实例, action):
    action.blocks = []
    block_offsets = 实例.blockOffsets[:] + [实例.data.size()]
    block_sizes = []
    for i in range(实例.numBlocks):
        size = block_offsets[i+1] - block_offsets[i]
        block_sizes.append(size)
    action.blocks = [实例.data.readslice(size) for size in block_sizes]


def hkaSplineCompressedAnimationBlocks(hkfile, 实例, action):
    action.mask4_uint8_float_list = []
    for floatoffset, bp in zip(实例.floatBlockOffsets, action.blocks):
        mask4block = bp.readslice(实例.maskAndQuantizationSize)
        uint8block = bp.readslice(floatoffset-实例.maskAndQuantizationSize)
        floatblock = bp.readremainslice()
        action.mask4_uint8_float_list.append( [mask4block, uint8block, floatblock] )


def hkaSplineCompressedAnimationFcurves(hkfile, 实例, action):
    action.block_fcurves = []
    for mask4block, uint8block, floatblock in action.mask4_uint8_float_list:
        tracks = uint8_to_fcurves(实例.annotationTracks, mask4block.copy(), uint8block.copy())
        action.block_fcurves.append(tracks)


def hkaSplineCompressedAnimationFcurvesHandle(hkfile, 实例, action):
    for i, tracks in enumerate(action.block_fcurves):
        for track in tracks:
            l_keyframe_dict, r_keyframe_dict, s_keyframe_dict = {}, {}, {}
            for time, value in track.lkeyframes: l_keyframe_dict[time+i*255] = value
            for time, value in track.rkeyframes: r_keyframe_dict[time+i*255] = value
            for time, value in track.skeyframes: s_keyframe_dict[time+i*255] = value
            track.channels = [l_keyframe_dict, r_keyframe_dict, s_keyframe_dict]

    action.tracknodes = {}
    for trackname in 实例.annotationTracks:
        action.tracknodes[trackname] = [trackname, {}, {}, {}]

    for tracks in action.block_fcurves:
        for track in tracks:
            trackname, node_l_keyframe_dict, node_r_keyframe_dict, node_s_keyframe_dict = action.tracknodes[track.name]
            l_keyframe_dict, r_keyframe_dict, s_keyframe_dict = track.channels
            for time, value in l_keyframe_dict.items(): node_l_keyframe_dict[time] = value
            for time, value in r_keyframe_dict.items(): node_r_keyframe_dict[time] = value
            for time, value in s_keyframe_dict.items(): node_s_keyframe_dict[time] = value





