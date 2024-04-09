def 函数(flag, size, version, srtm):
    材质数量 = 0
    match version:
        case 1: 材质数量 = SRTM_version1(srtm)
        case 2: 材质数量 = SRTM_version2(srtm)  # 古剑2
    return 材质数量

def SRTM_version1(bp):
    列表 = bp.readuint32(3) # [num, 0, num]
    材质数量 = max(列表)
    return 材质数量


def SRTM_version2(bp):  # 古剑2
    列表 = bp.readuint32(4) # [num, 0, num, 0] # [0, num, num, 0]
    材质数量 = max(列表)
    return 材质数量
