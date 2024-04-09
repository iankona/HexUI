def 函数(flag, size, version, stcm):
    形态键列表 = []
    match version:
        case 1: 形态键列表 = STCM_version1(stcm)  # 古剑2
    return 形态键列表


def STCM_version1(bp):
    形态键数量, 未知数 = bp.readuint32(2)
    形态键列表 = []
    for i in range(形态键数量):
        rangemin, rangemax = bp.readfloat32(2)
        lodlevel, numdeformation, numtransformation, phonemesetbitmask = bp.readuint32(4)
        形态键名称 = bp.readchar(bp.readuint32())
        positionoffset, positionindex = [], []
        for j in range(numdeformation):
            nodeid, minvalue, maxvalue, numvertice = bp.readuint32(), bp.readfloat32(), bp.readfloat32(), bp.readuint32()
            # fXOffset = fMinValue + (fMaxValue - fMinValue)*(x / 65535);
            # vecDeformedPos.fX = vecPos.fX + fXOffset*fMorphAmount)
            position_offsets = [[minvalue+(maxvalue-minvalue)*(bp.readuint16()/65535),
                                    minvalue+(maxvalue-minvalue)*(bp.readuint16()/65535),
                                    minvalue+(maxvalue-minvalue)*(bp.readuint16()/65535) ] for m in range(numvertice)]
            # fXOffset = x/127.5 - 1.0;
            # vecDeformedNormal.fX = vecNormal.fX + fXOffset * fMorphAmount)
            normal_offsets   = [[bp.readuint8()/127.5-1.0,
                                    bp.readuint8()/127.5-1.0,
                                    bp.readuint8()/127.5-1.0 ]  for m in range(numvertice)]

            tangent_offsets  = [ bp.readuint8(3)  for m in range(numvertice)]
            vertice_indices  = [ bp.readuint32()  for m in range(numvertice)]
            positionoffset += position_offsets
            positionindex  += vertice_indices
        for j in range(numtransformation):
            nodeid, rotation, scaleRotation, position, scale = bp.readuint32(), bp.readfloat32(4), bp.readfloat32(4), bp.readuint32(3), bp.readuint32(3)

        形态键列表.append([形态键名称, positionoffset, positionindex])

    return 形态键列表