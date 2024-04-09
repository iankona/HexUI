def 函数(flag, size, version, bp):
    index, result = 0, None
    match version:
        case 1: index, result = FILE_version1(bp)
    return index, result


def FILE_version1(bp):
    index, size = bp.readuint32(2)
    result = bp.readslice(size)
    return index, result