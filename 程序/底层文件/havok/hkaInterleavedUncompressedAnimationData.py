
class Object: pass
def 函数(hkfile, 实例):
    hkfile.action = Object()
    hkaInterleavedUncompressedAnimation(hkfile, 实例, hkfile.action)


def hkaInterleavedUncompressedAnimation(hkafile, 实例, action):
    # 要求 position 和 rotation 都是 basis 空间, 插件暂时不支持
    # fps = 60.0
    # frameCount = 1 + round(fps * 实例.duration)
    # 骨骼总数 = 实例.numberOfTransformTracks # 187
    # 骨骼名称列表 = 实例.annotationTracks # 187
    # 实例.transforms # 28611 # [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.9999999], [1.0, 1.0, 1.0, 0.0]]

    action.tracknodes = {}
    for i, trackname in enumerate(实例.annotationTracks):
        action.tracknodes[trackname] = [trackname, {}, {}, {}]

    numframe = len(实例.transforms) // 实例.numberOfTransformTracks

    # # 按trackname节点存 # 错误
    # action.tracktransforms = {}
    # for i, trackname in enumerate(实例.annotationTracks):
    #     action.tracktransforms[trackname] = 实例.transforms[i*numframe: (i+1)*numframe]

    # for i, trackname in enumerate(实例.annotationTracks):
    #     tracktransforms = action.tracktransforms[trackname]
    #     骨骼名称, 位置帧字典, 旋转帧字典, 缩放帧字典 = action.tracknodes[trackname]
    #     for j, [position, rotation, scale] in enumerate(tracktransforms):
    #         vx, vy, vz, vw = position
    #         qx, qy, qz, qw = rotation
    #         sx, sy, sz, sw = scale
    #         位置帧字典[j+1] = [vx, vy, vz]
    #         旋转帧字典[j+1] = [qw, qx, qy, qz]
    #         缩放帧字典[j+1] = [sx, sy, sz]


    # 按frame节点存
    action.frametransforms = {}
    for i in range(numframe):
        action.frametransforms[i] = 实例.transforms[i*实例.numberOfTransformTracks: (i+1)*实例.numberOfTransformTracks]

    for i, frametransforms in action.frametransforms.items():
        for trackname, [position, rotation, scale] in zip(实例.annotationTracks, frametransforms):
            骨骼名称, 位置帧字典, 旋转帧字典, 缩放帧字典 = action.tracknodes[trackname]
            vx, vy, vz, vw = position
            qx, qy, qz, qw = rotation
            sx, sy, sz, sw = scale
            位置帧字典[i+1] = [vx, vy, vz]
            旋转帧字典[i+1] = [qw, qx, qy, qz]
            缩放帧字典[i+1] = [sx, sy, sz]












