def 函数(flag, size, version, leks):
    骨骼节点列表 = []
    match version:
        case 1: 骨骼节点列表 = LEKS_version1(leks) # 古剑2
    return 骨骼节点列表


def LEKS_version1(bp):   # 古剑2
    骨骼数量, 根骨骼数量 = bp.readuint32(2)
    骨骼节点列表 = []
    for i in range(骨骼数量):
        rotation, scalerotation, position, scale, unfloat000 = bp.readfloat32(4), bp.readfloat32(4), bp.readfloat32(3), bp.readfloat32(3), bp.readfloat32(3)
        number8f = bp.readhex(8)
        parentid, numchild, number11  = bp.readint32(3)
        transformmatrix = [bp.readfloat32(4), bp.readfloat32(4), bp.readfloat32(4), bp.readfloat32(4)]
        unfloat0 = bp.readfloat32()
        bonename = bp.readchar(bp.readuint32())
        骨骼节点列表.append([position, rotation, scale, parentid, bonename])
    return 骨骼节点列表