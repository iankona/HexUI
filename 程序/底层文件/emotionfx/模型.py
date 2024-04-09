def 函数(flag, size, version, hsmv):
    顶点列表, 顶点UV列表, 区网格息列表, 区权重索引Loop列表 = [], [], [], []
    match version:
        case 1: 区权重索引Loop列表, 顶点列表, 顶点UV列表, 区网格息列表 = HSMV_version1(hsmv)
        case 2: 区权重索引Loop列表, 顶点列表, 顶点UV列表, 区网格息列表 = HSMV_version2(hsmv)  # 古剑2
    return 顶点列表, 顶点UV列表, 区网格息列表, 区权重索引Loop列表


def HSMV_version1(bp):
    未知数, 区权重总数, 顶点总数, 顶点Loop总数, 区网格总数, 数据块总数, 未知数 = bp.readuint32(7)
    区权重索引Loop列表, 顶点列表, 顶点UV列表, 区网格息列表 = HSMV(数据块总数, 顶点总数, 区网格总数, bp)
    return 区权重索引Loop列表, 顶点列表, 顶点UV列表, 区网格息列表


def HSMV_version2(bp): # 古剑2
    未知数, 未知数, 区权重总数, 顶点总数, 顶点Loop总数, 区网格总数, 数据块总数, 未知数 = bp.readuint32(8)
    区权重索引Loop列表, 顶点列表, 顶点UV列表, 区网格息列表 = HSMV(数据块总数, 顶点总数, 区网格总数, bp)
    return 区权重索引Loop列表, 顶点列表, 顶点UV列表, 区网格息列表


def HSMV(数据块总数, 顶点总数, 区网格总数, bp):
    区权重索引Loop列表, 顶点列表, 顶点UV列表 = [], [], []
    for i in range(数据块总数):
        标识, 步长, 未知 = bp.readuint32(3)
        match 标识:
            case 5: 区权重索引Loop列表 = bp.readuint32(顶点总数)
            case 0: 顶点列表 = [bp.readfloat32(3) for x in range(顶点总数)]
            case 1: 顶点法线列表 = [bp.readfloat32(3) for x in range(顶点总数)]
            case 3: 顶点UV列表 = [bp.readfloat32(2) for x in range(顶点总数)]
            case 2: 顶点颜色列表 = [bp.readfloat32(4) for x in range(顶点总数)]
            case _: 列表 = [bp.read(步长) for x in range(顶点总数)]

    区网格息列表 = []
    for j in range(区网格总数):
        Loop个数, 顶点个数, 材质索引, 骨骼个数  = bp.readuint32(4)
        顶点Loop列表 = bp.readuint32(Loop个数)
        骨骼映射列表 = bp.readuint32(骨骼个数)
        区网格息列表.append( [顶点个数, Loop个数, 材质索引, 顶点Loop列表, 骨骼个数, 骨骼映射列表] )

    return 区权重索引Loop列表, 顶点列表, 顶点UV列表, 区网格息列表
