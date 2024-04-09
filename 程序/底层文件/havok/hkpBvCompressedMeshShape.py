


from .havokbinary import readhavokint, readhavokuint


def hkpBvCompressedMeshShape(bp, flag, havokfile):
    if flag == 25668: return hkpBvCompressedMeshShape25668(bp, havokfile) # [34, 50], 25668
    if flag == 16452: return hkpBvCompressedMeshShape16452(bp, havokfile) # [34, 32], 16452

class hkpBvCompressedMeshShape25668:
    def __init__(self, bp, havokfile):
        num2, num3, num4 = readhavokint(bp), readhavokint(bp), readhavokint(bp) # [100,4,3,1]
        num_userStringPalette = readhavokint(bp)
        self.userStringPalette = [havokfile.readname(bp) for i in range(num_userStringPalette)]
        num01 = readhavokint(bp) # 3

        # tree

        num_node = readhavokint(bp)
        list_flag = bp.readuint8() # 7
        node_xyzs = [bp.readuint8(3) for i in range(num_node)]
        node_hiDatas = [bp.readuint8() for i in range(num_node)]
        node_loData = [bp.readuint8() for i in range(num_node)]

        # 
        num21 = bp.readuint8()
        domain_min = bp.readfloat32(4)
        domain_max = bp.readfloat32(4)

        self.numPrimitiveKeys = readhavokint(bp)
        self.bitsPerKey = readhavokint(bp)
        self.maxKeyValue = readhavokint(bp)
        num_section = readhavokint(bp)
        char11 = havokfile.readname(bp) # -255
        section_nodes = [readsectionnode(bp) for i in range(num_section)]

        num21, num22 = readhavokint(bp), readhavokint(bp) # [3, 6]
        section_domain_mins = [bp.readfloat32(3) for i in range(num_section)]
        num33 = readhavokint(bp) # 6
        section_domain_maxs = [bp.readfloat32(3) for i in range(num_section)]
        section_codecParms = [bp.readfloat32(6) for i in range(num_section)]
        num41 = bp.readuint8(2) # [8, 0]
        section_firstPackedVertexs = [readhavokint(bp) for i in range(num_section)]
        num51 = bp.readuint8() # [8]
        section_sharedVertices = [readhavokint(bp) for i in range(num_section)]

        num61 = bp.readuint8(2) # [1, 8]
        section_primitives = [readhavokint(bp) for i in range(num_section)]

        num71 = bp.readuint8(2) # [1, 8]
        section_dataRuns = [readhavokint(bp) for i in range(num_section)]

        section_numPackedVertices = [bp.readuint8() for i in range(num_section)]
        section_numSharedIndices = [bp.readuint8() for i in range(num_section)]

        num81 = bp.readuint8() # [8]
        section_leafIndexs = [readhavokint(bp) for i in range(num_section)]

        num_primitive = readhavokint(bp)
        num91 = bp.readuint8() # [1]
        self.primitives = [bp.readuint8(4) for i in range(num_primitive)]

        num_sharedVerticesIndex = readhavokint(bp)
        num10 = bp.readuint8() # [8]
        self.sharedVerticesIndex = [readhavokint(bp) for i in range(num_sharedVerticesIndex)]

        num_packedVertice = readhavokint(bp)
        num13 = bp.readuint8() # [8]
        self.packedVertices = [readhavokuint(bp) for i in range(num_packedVertice)]

        num_sharedVertice = readhavokint(bp)
        num13 = bp.readuint8() # [16]
        self.sharedVertices = [readhavokuint(bp) for i in range(num_sharedVertice)]

        num_primitiveDataRun = readhavokint(bp)
        num14 = bp.readuint8(2) # [16]
        primitiveDataRun_values = [readhavokint(bp) for i in range(num_primitiveDataRun)]
        primitiveDataRun_indexs = [bp.readuint8() for i in range(num_primitiveDataRun)]
        primitiveDataRun_counts = [bp.readuint8() for i in range(num_primitiveDataRun)]


class hkpBvCompressedMeshShape16452:
    def __init__(self, bp, havokfile):
        num2, num3, num4 = readhavokint(bp), readhavokint(bp), readhavokint(bp) # [64,4,3, ]

        # tree
        num_node = readhavokint(bp)
        list_flag = bp.readuint8() # 7
        node_xyzs = [bp.readuint8(3) for i in range(num_node)]
        node_hiDatas = [bp.readuint8() for i in range(num_node)]
        node_loData = [bp.readuint8() for i in range(num_node)]

        # 
        num21 = bp.readuint8()
        domain_min = bp.readfloat32(4)
        domain_max = bp.readfloat32(4)

        self.numPrimitiveKeys = readhavokint(bp)
        self.bitsPerKey = readhavokint(bp)
        self.maxKeyValue = readhavokint(bp)
        num_section = readhavokint(bp)
        char11 = havokfile.readname(bp) # -255
        section_nodes = [readsectionnode(bp) for i in range(num_section)]

        num21, num22 = readhavokint(bp), readhavokint(bp) # [3, 6]
        section_domain_mins = [bp.readfloat32(3) for i in range(num_section)]
        num33 = readhavokint(bp) # 6
        section_domain_maxs = [bp.readfloat32(3) for i in range(num_section)]
        section_codecParms = [bp.readfloat32(6) for i in range(num_section)]
        num41 = bp.readuint8(2) # [8, 0]
        section_firstPackedVertexs = [readhavokint(bp) for i in range(num_section)]
        num51 = bp.readuint8() # [8]
        section_sharedVertices = [readhavokint(bp) for i in range(num_section)]

        num61 = bp.readuint8(2) # [1, 8]
        section_primitives = [readhavokint(bp) for i in range(num_section)]

        num71 = bp.readuint8(2) # [1, 8]
        section_dataRuns = [readhavokint(bp) for i in range(num_section)]

        section_numPackedVertices = [bp.readuint8() for i in range(num_section)]
        section_numSharedIndices = [bp.readuint8() for i in range(num_section)]

        num81 = bp.readuint8() # [8]
        section_leafIndexs = [readhavokint(bp) for i in range(num_section)]

        num_primitive = readhavokint(bp)
        num91 = bp.readuint8() # [1]
        self.primitives = [bp.readuint8(4) for i in range(num_primitive)]

        num_sharedVerticesIndex = readhavokint(bp)
        num10 = bp.readuint8() # [8]
        self.sharedVerticesIndex = [readhavokint(bp) for i in range(num_sharedVerticesIndex)]

        num_packedVertice = readhavokint(bp)
        num13 = bp.readuint8() # [8]
        self.packedVertices = [readhavokuint(bp) for i in range(num_packedVertice)]

        num_sharedVertice = readhavokint(bp)
        num13 = bp.readuint8() # [16]
        self.sharedVertices = [readhavokuint(bp) for i in range(num_sharedVertice)]

        num_primitiveDataRun = readhavokint(bp)
        num14 = bp.readuint8(2) # [16]
        primitiveDataRun_values = [readhavokint(bp) for i in range(num_primitiveDataRun)]
        primitiveDataRun_indexs = [bp.readuint8() for i in range(num_primitiveDataRun)]
        primitiveDataRun_counts = [bp.readuint8() for i in range(num_primitiveDataRun)]














def readsectionnode(bp):
    num_node = readhavokint(bp) # 181
    num_elem = bp.readuint8() # 3 or list_flag
    node_xyzs = [bp.readuint8(3) for i in range(num_node)]
    node_datas = [bp.readuint8() for i in range(num_node)]
    return [num_node, num_elem, node_xyzs, node_datas]





ShapeInfoCodecTypeEnum = {
    0: "NULL_CODEC",
    1: "UFM358",
    16: "MAX_NUM_CODECS",
    }


ShapeDispatchTypeEnum = {
	0: "CONVEX_IMPLICIT",
    1: "CONVEX",
    2: "HEIGHT_FIELD",
    3: "COMPOSITE",
    4: "USER",
    5: "NUM_DISPATCH_TYPES",
    }

BvTreeType ={
    0:"BVTREE_MOPP",
    1:"BVTREE_TRISAMPLED_HEIGHTFIELD",
    2:"BVTREE_STATIC_COMPOUND",
    3:"BVTREE_COMPRESSED_MESH",
    4:"BVTREE_USER",
    5:"BVTREE_MAX",
    }

WeldingType = {
    0: "WELDING_TYPE_ANTICLOCKWISE",
    4: "WELDING_TYPE_CLOCKWISE",
    5: "WELDING_TYPE_TWO_SIDED",
    6: "WELDING_TYPE_NONE",
    }


class hkBaseObject: pass
[4, 'hkBaseObject', 0, 0, 0]


class hkReferencedObject(hkBaseObject): pass
[4, 'hkReferencedObject', 0, 3, 1]
['memSizeAndRefCount', 0, None, '']


class hkcdShape(hkReferencedObject): pass
[4, 'hkcdShape', 0, 4, 4]
['type', 0, None, '']
['dispatchType', 2, None, '']
['bitsPerKey', 2, None, '']
['shapeInfoCodecType', 2, None, '']


class hkpShapeBase(hkcdShape): pass
[4, 'hkpShapeBase', 0, 20, 0]


class hkpShape(hkpShapeBase): pass
[4, 'hkpShape', 2, 21, 1]
['userData', 4, None, '']


class hkpBvTreeShape(hkpShape): pass
[4, 'hkpBvTreeShape', 2, 28, 1]
['bvTreeType', 2, None, '']


class _hkpBvCompressedMeshShape(hkpBvTreeShape): pass
[4, 'hkpBvCompressedMeshShape', 4, 51, 8]
['convexRadius', 6, None, '']
['weldingType', 2, None, '']
['hasPerPrimitiveCollisionFilterInfo', 2, None, '']
['hasPerPrimitiveUserData', 2, None, '']
['collisionFilterInfoPalette', 36, None, '']
['userDataPalette', 36, None, '']
['userStringPalette', 52, None, '']
['tree', 18, None, 'hkpBvCompressedMeshShapeTree']





