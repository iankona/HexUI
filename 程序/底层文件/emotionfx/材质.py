def 函数(flag, size, version, lrtm):
    材质属性列表, 贴图数量, 材质名称 = [], 0, ""
    match version:
        case 2: 材质属性列表, 贴图数量, 材质名称 = LRTM_version2(lrtm)
        case 3: 材质属性列表, 贴图数量, 材质名称 = LRTM_version3(lrtm)  # 古剑2

    return 材质属性列表, 贴图数量, 材质名称


def LRTM_version2(bp):
    with bp.readslice(80) as data:
        ambient, diffuse, specular, emissive = data.readfloat32(4), data.readfloat32(4), data.readfloat32(4), data.readfloat32(4)
        shine, shinestrength, opacity, ior   = data.readfloat32(4)
    材质属性 = [ambient, diffuse, specular, emissive, shine, shinestrength, opacity, ior]
    贴图数量 = bp.readuint8(4)[-1]    # 0, 0, 70, num
    材质名称 = bp.readchar(bp.readuint32())
    return 材质属性, 贴图数量, 材质名称


def LRTM_version3(bp):  # 古剑2
    with bp.readslice(84) as data:
        ambient, diffuse, specular, emissive = data.readfloat32(4), data.readfloat32(4), data.readfloat32(4), data.readfloat32(4)
        shine, shinestrength, opacity, ior   = data.readfloat32(4)
        unsed = data.readfloat32()
    材质属性 = [ambient, diffuse, specular, emissive, shine, shinestrength, opacity, ior]
    贴图数量 = bp.readuint8(4)[-1]    # 0, 0, 70, num
    材质名称 = bp.readchar(bp.readuint32())
    return 材质属性, 贴图数量, 材质名称