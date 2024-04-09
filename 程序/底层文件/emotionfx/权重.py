def 函数(flag, size, version, thgw):
    权重值Loop列表, 区权重信息列表 = [], []
    match version:
        case 3: 权重值Loop列表, 区权重信息列表 = THGW_version3(thgw)
        case 4: 权重值Loop列表, 区权重信息列表 = THGW_version4(thgw)  # 古剑2
    return 权重值Loop列表, 区权重信息列表


def THGW_version3(bp):
    模型编号, 未知数, 权重总数, 未知数 = bp.readuint32(4)
    权重值Loop列表, 区权重信息列表 = THGW(权重总数, bp)
    return 权重值Loop列表, 区权重信息列表


def THGW_version4(bp):
    模型编号, 未知数, 未知数, 权重总数, 未知数 = bp.readuint32(5)
    权重值Loop列表, 区权重信息列表 = THGW(权重总数, bp)
    return 权重值Loop列表, 区权重信息列表


def THGW(权重总数, bp):
    权重值Loop列表 = [[bp.readfloat32(), bp.readuint32()] for i in range(权重总数)] # [权重值, 骨骼ID]
    区权重信息列表 = [ bp.readuint32(2) for j in range(bp.remainsize()//8)] # [索引左值, 权重个数]
    return 权重值Loop列表, 区权重信息列表