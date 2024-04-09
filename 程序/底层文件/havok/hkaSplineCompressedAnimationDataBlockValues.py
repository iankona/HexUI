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


class Object: pass
def uint8_to_fcurves(annotationtracks, mask4, uint8):
    tracks = []
    for trackname in annotationtracks:
        track = Object()
        track.name = trackname
        track.mask = mask4.readuint8(4)
        position_data, position_fcurve_datas = position_type_size(track.mask[1], uint8)
        rotation_data, rotation_fcurve_datas = rotation_type_size(track.mask[2], uint8)
        scale_data,    scale_fcurve_datas    = position_type_size(track.mask[3], uint8)
        track.lblock = position_data
        track.rblock = rotation_data
        track.sblock = scale_data
        track.ldatas = position_fcurve_datas
        track.rdatas = rotation_fcurve_datas
        track.sdatas = scale_fcurve_datas
        track.lkeyframes = position_keyframes(track.mask[1], track.lblock, track.ldatas)
        track.rkeyframes = rotation_keyframes(track.mask[2], track.rblock, track.rdatas)
        track.skeyframes = position_keyframes(track.mask[3], track.sblock, track.sdatas)
        tracks.append(track)
    return tracks


def position_type_size(typeflag, uint8):
    position_data, position_fcurve_datas = None, []

    if 0 < typeflag < 8:
        if int(typeflag) in [1, 2, 4]: position_size = 4   # float32,  x or y or z,     other 0.0
        if int(typeflag) in [3, 5, 6]: position_size = 8   # float32,  xy or xz or yz,  other 0.0
        if int(typeflag) in [7,     ]: position_size = 12  # float32,  [x, y, z]
        position_data = uint8.readslice(position_size)

    if typeflag >= 8:
        # uint16 to float32
        # 注：不存在 11，22，33，44，55，66，77 typeflag
        # [xmin, xmax, y, z],          [x, ymin, ymax, z],          [x, y, zmin, zmax]
        # [xmin, xmax, ymin, ymax, z], [xmin, xmax, y, zmin, zmax], [x, ymin, ymax, zmin, zmax]
        # [xmin, xmax, ymin, ymax, zmin, zmax]
        hexs = "%02X" % typeflag
        numfloat, numuint16, dnumtime = 0, 0, 0
        if hexs[0] in ["1", "2", "4"]:   # [xmin, xmax, y, z], [x, ymin, ymax, z], [x, y, zmin, zmax]
            if hexs[1] in ["0"          ]: numfloat, numuint16 = 2, 1
            if hexs[1] in ["1", "2", "4"]: numfloat, numuint16 = 3, 1
            if hexs[1] in ["3", "5", "6"]: numfloat, numuint16 = 4, 1
        if hexs[0] in ["3", "5", "6"]:   # [xmin, xmax, ymin, ymax, z], [xmin, xmax, y, zmin, zmax], [x, ymin, ymax, zmin, zmax]
            if hexs[1] in ["0"          ]: numfloat, numuint16 = 4, 2
            if hexs[1] in ["1", "2", "4"]: numfloat, numuint16 = 5, 2
        if hexs[0] in ["7",         ]:   # [xmin, xmax, ymin, ymax, zmin, zmax]
            if hexs[1] in ["0"          ]: numfloat, numuint16 = 6, 3

        number, skip, degree = uint8.readuint8seek0(3)
        if skip != 0: return 0, 0
        numhead = 3 + degree
        numtime = number + 1
        numtail = 1
        numpad1 = 0 if (numhead+numtime+numtail)%4 == 0 else 4 - (numhead+numtime+numtail)%4
        sizeflag = numhead + numtime + numtail + numpad1
        sizehead = numfloat * 4
        sizedata = numtime * numuint16 * 2
        sizepad2 = 0 if (sizeflag+sizehead+sizedata)%4 == 0 else 4 - (sizeflag+sizehead+sizedata)%4
        datasize = sizeflag + sizehead + sizedata + sizepad2

        with uint8.readslice(datasize) as position_data:
            head = position_data.readslice(numhead)
            time = position_data.readslice(numtime)
            tail = position_data.readslice(numtail+numpad1)
            stat = position_data.readslice(sizehead)
            data = position_data.readslice(sizedata)
        return position_data.copy(), [numtime, time, stat, data]

    return position_data, position_fcurve_datas

def rotation_type_size(typeflag, uint8):
    rotation_data, rotation_fcurve_datas = None, []

    if 0 < typeflag < 16:
        rotation_size = 8 # 5+3
        rotation_data = uint8.readslice(rotation_size)

    if typeflag >= 16:
        number, skip, degree = uint8.readuint8seek0(3)
        if skip != 0: return 0, 0
        numhead = 3 + degree
        numtime = number + 1
        numtail = 1
        sizeflag = numhead + numtime + numtail
        sizedata = numtime * 5
        sizepad1 = 0 if (sizeflag+sizedata)%4 == 0 else 4 - (sizeflag+sizedata)%4
        datasize = sizeflag + sizedata + sizepad1

        with uint8.readslice(datasize) as rotation_data:
            head = rotation_data.readslice(numhead)
            time = rotation_data.readslice(numtime)
            tail = rotation_data.readslice(numtail)
            data = rotation_data.readslice(sizedata)
        return rotation_data.copy(), [numtime, time, data]

    return rotation_data, rotation_fcurve_datas

def position_keyframes(typeflag, block, datas):
    result = []
    hexs = "%02X" % typeflag
    match hexs:
        case "00": pass
        case "01": result = location_01(block)
        case "02": result = location_02(block)
        case "04": result = location_04(block)
        case "03": result = location_03(block)
        case "05": result = location_05(block)
        case "06": result = location_06(block)
        case "07": result = location_07(block)
        case "10": result = location_10(datas)
        case "12": result = location_12(datas)
        case "14": result = location_14(datas)
        case "16": result = location_16(datas)
        case "20": result = location_20(datas)
        case "21": result = location_21(datas)
        case "24": result = location_24(datas)
        case "25": result = location_25(datas)
        case "40": result = location_40(datas)
        case "41": result = location_41(datas)
        case "42": result = location_42(datas)
        case "43": result = location_43(datas)
        case "30": result = location_30(datas)
        case "34": result = location_34(datas)
        case "50": result = location_50(datas)
        case "52": result = location_52(datas)
        case "60": result = location_60(datas)
        case "61": result = location_61(datas)
        case "70": result = location_70(datas)
        case _: print("未知", hexs)
    return result

def rotation_keyframes(typeflag, block, datas):
    result = []
    if 0 < typeflag < 16: result = rotation_01(block)
    if typeflag >= 16: result = rotation_16(datas)
    return result


def location_01(block):
    x = block.readfloat32()
    return [[0, [x, 0, 0]], ]

def location_02(block):
    y = block.readfloat32()
    return [[0, [0, y, 0]], ]

def location_04(block):
    z = block.readfloat32()
    return [[0, [0, 0, z]], ]

def location_03(block):
    x, y = block.readfloat32(2)
    return [[0, [x, y, 0]], ]

def location_05(block):
    x, z = block.readfloat32(2)
    return [[0, [x, 0, z]], ]

def location_06(block):
    y, z = block.readfloat32(2)
    return [[0, [0, y, z]], ]

def location_07(block):
    x, y, z = block.readfloat32(3)
    return [[0, [x, y, z]], ]


def location_10(datas):
    numtime, time, stat, data = datas
    times = time.readuint8(numtime)
    xmin, xmax = stat.readfloat32(2)
    datas = [data.readu16float32() for i in range(numtime)]
    xseg = xmax - xmin
    values = [xmin + xseg*xnum for xnum in datas]

    result = []
    for time, x in zip(times, values):
        result.append([time, [x, 0, 0]])
    return result


def location_12(datas):
    numtime, time, stat, data = datas
    times = time.readuint8(numtime)
    xmin, xmax, y = stat.readfloat32(3)
    datas = [data.readu16float32() for i in range(numtime)]
    xseg = xmax - xmin
    values = [xmin + xseg*xnum for xnum in datas]

    result = []
    for time, x in zip(times, values):
        result.append([time, [x, y, 0]])
    return result


def location_14(datas):
    numtime, time, stat, data = datas
    times = time.readuint8(numtime)
    xmin, xmax, z = stat.readfloat32(3)
    datas = [data.readu16float32() for i in range(numtime)]
    xseg = xmax - xmin
    values = [xmin + xseg*xnum for xnum in datas]

    result = []
    for time, x in zip(times, values):
        result.append([time, [x, 0, z]])
    return result


def location_16(datas):
    numtime, time, stat, data = datas
    times = time.readuint8(numtime)
    xmin, xmax, y, z = stat.readfloat32(4)
    datas = [data.readu16float32() for i in range(numtime)]
    xseg = xmax - xmin
    values = [xmin + xseg*xnum for xnum in datas]

    result = []
    for time, x in zip(times, values):
        result.append([time, [x, y, z]])
    return result


def location_20(datas):
    numtime, time, stat, data = datas
    times = time.readuint8(numtime)
    ymin, ymax = stat.readfloat32(2)
    datas = [data.readu16float32() for i in range(numtime)]
    yseg = ymax - ymin
    values = [ymin + yseg*ynum for ynum in datas]

    result = []
    for time, y in zip(times, values):
        result.append([time, [0, y, 0]])
    return result


def location_21(datas):
    numtime, time, stat, data = datas
    times = time.readuint8(numtime)
    x, ymin, ymax = stat.readfloat32(3)
    datas = [data.readu16float32() for i in range(numtime)]
    yseg = ymax - ymin
    values = [ymin + yseg*ynum for ynum in datas]

    result = []
    for time, y in zip(times, values):
        result.append([time, [x, y, 0]])
    return result


def location_24(datas):
    numtime, time, stat, data = datas
    times = time.readuint8(numtime)
    ymin, ymax, z = stat.readfloat32(3)
    datas = [data.readu16float32() for i in range(numtime)]
    yseg = ymax - ymin
    values = [ymin + yseg*ynum for ynum in datas]

    result = []
    for time, y in zip(times, values):
        result.append([time, [0, y, z]])
    return result


def location_25(datas):
    numtime, time, stat, data = datas
    times = time.readuint8(numtime)
    x, ymin, ymax, z = stat.readfloat32(4)
    datas = [data.readu16float32() for i in range(numtime)]
    yseg = ymax - ymin
    values = [ymin + yseg*ynum for ynum in datas]

    result = []
    for time, y in zip(times, values):
        result.append([time, [x, y, z]])
    return result


def location_40(datas):
    numtime, time, stat, data = datas
    times = time.readuint8(numtime)
    zmin, zmax = stat.readfloat32(2)
    datas = [data.readu16float32() for i in range(numtime)]
    zseg = zmax - zmin
    values = [zmin + zseg*znum for znum in datas]

    result = []
    for time, z in zip(times, values):
        result.append([time, [0, 0, z]])
    return result


def location_41(datas):
    numtime, time, stat, data = datas
    times = time.readuint8(numtime)
    x, zmin, zmax = stat.readfloat32(3)
    datas = [data.readu16float32() for i in range(numtime)]
    zseg = zmax - zmin
    values = [zmin + zseg*znum for znum in datas]

    result = []
    for time, z in zip(times, values):
        result.append([time, [x, 0, z]])
    return result


def location_42(datas):
    numtime, time, stat, data = datas
    times = time.readuint8(numtime)
    y, zmin, zmax = stat.readfloat32(3)
    datas = [data.readu16float32() for i in range(numtime)]
    zseg = zmax - zmin
    values = [zmin + zseg*znum for znum in datas]

    result = []
    for time, z in zip(times, values):
        result.append([time, [0, y, z]])
    return result


def location_43(datas):
    numtime, time, stat, data = datas
    times = time.readuint8(numtime)
    x, y, zmin, zmax = stat.readfloat32(4)
    datas = [data.readu16float32() for i in range(numtime)]
    zseg = zmax - zmin
    values = [zmin + zseg*znum for znum in datas]

    result = []
    for time, z in zip(times, values):
        result.append([time, [x, y, z]])
    return result


def location_30(datas):
    numtime, time, stat, data = datas
    times = time.readuint8(numtime)
    xmin, xmax, ymin, ymax = stat.readfloat32(4)
    datas = [data.readu16float32(2) for i in range(numtime)]
    xseg = xmax - xmin
    yseg = ymax - ymin

    values = []
    for xnum, ynum in datas:
        x = xmin + xseg*xnum
        y = ymin + yseg*ynum
        values.append([x, y])

    result = []
    for time, [x, y] in zip(times, values):
        result.append([time, [x, y, 0]])
    return result


def location_34(datas):
    numtime, time, stat, data = datas
    times = time.readuint8(numtime)
    xmin, xmax, ymin, ymax, z = stat.readfloat32(5)
    datas = [data.readu16float32(2) for i in range(numtime)]
    xseg = xmax - xmin
    yseg = ymax - ymin

    values = []
    for xnum, ynum in datas:
        x = xmin + xseg*xnum
        y = ymin + yseg*ynum
        values.append([x, y])

    result = []
    for time, [x, y] in zip(times, values):
        result.append([time, [x, y, z]])
    return result


def location_50(datas):
    numtime, time, stat, data = datas
    times = time.readuint8(numtime)
    xmin, xmax, zmin, zmax = stat.readfloat32(4)
    datas = [data.readu16float32(2) for i in range(numtime)]
    xseg = xmax - xmin
    zseg = zmax - zmin

    values = []
    for xnum, znum in datas:
        x = xmin + xseg*xnum
        z = zmin + zseg*znum
        values.append([x, z])

    result = []
    for time, [x, z] in zip(times, values):
        result.append([time, [x, 0, z]])
    return result


def location_52(datas):
    numtime, time, stat, data = datas
    times = time.readuint8(numtime)
    xmin, xmax, y, zmin, zmax = stat.readfloat32(5)
    datas = [data.readu16float32(2) for i in range(numtime)]
    xseg = xmax - xmin
    zseg = zmax - zmin

    values = []
    for xnum, znum in datas:
        x = xmin + xseg*xnum
        z = zmin + zseg*znum
        values.append([x, z])

    result = []
    for time, [x, z] in zip(times, values):
        result.append([time, [x, y, z]])
    return result


def location_60(datas):
    numtime, time, stat, data = datas
    times = time.readuint8(numtime)
    ymin, ymax, zmin, zmax = stat.readfloat32(4)
    datas = [data.readu16float32(2) for i in range(numtime)]
    yseg = ymax - ymin
    zseg = zmax - zmin

    values = []
    for ynum, znum in datas:
        y = ymin + yseg*ynum
        z = zmin + zseg*znum
        values.append([y, z])

    result = []
    for time, [y, z] in zip(times, values):
        result.append([time, [0, y, z]])
    return result


def location_61(datas):
    numtime, time, stat, data = datas
    times = time.readuint8(numtime)
    x, ymin, ymax, zmin, zmax = stat.readfloat32(5)
    datas = [data.readu16float32(2) for i in range(numtime)]
    yseg = ymax - ymin
    zseg = zmax - zmin

    values = []
    for ynum, znum in datas:
        y = ymin + yseg*ynum
        z = zmin + zseg*znum
        values.append([x, z])

    result = []
    for time, [y, z] in zip(times, values):
        result.append([time, [x, y, z]])
    return result


def location_70(datas):
    numtime, time, stat, data = datas
    times = time.readuint8(numtime)
    xmin, xmax, ymin, ymax, zmin, zmax = stat.readfloat32(6)
    datas = [data.readu16float32(3) for i in range(numtime)]
    xseg = xmax - xmin
    yseg = ymax - ymin
    zseg = zmax - zmin

    values = []
    for xnum, ynum, znum in datas:
        x = xmin + xseg*xnum
        y = ymin + yseg*ynum
        z = zmin + zseg*znum
        values.append([x, y, z])

    result = []
    for time, [x, y, z] in zip(times, values):
        result.append([time, [x, y, z]])
    return result


def rotation_01(block):
    qx, qy, qz, qw = readhavokquat(block)
    return [[0, [qx, qy, qz, qw]], ]


def rotation_16(datas):
    numtime, time, data = datas
    times = time.readuint8(numtime)
    datas = [readhavokquat(data) for i in range(numtime)]

    result = []
    for time, quat in zip(times, datas): result.append([time, quat])
    return result

