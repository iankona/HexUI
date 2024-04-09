class NiDataStream:
    def __init__(self, bp, niffile):
        self.描述列表列表 = []
        self.读取成员(bp, niffile)

    def 读取成员(self, bp, niffile):
        数据区大小, 克隆数, 区域数 = bp.readuint32(3) # 20.6.0.0  区域数, 块标识，默认为1和0
        区域信息列表 = [[bp.readuint32(), bp.readuint32()] for i in range(区域数)]
        标识总数 = bp.readuint32()
        标识列表 = [ bp.readuint8(4) for x in range(标识总数)]
        for 索引左值, 索引个数 in 区域信息列表:
            描述列表 = [[] for x in range(标识总数)]
            # 数据区大小, 开始
            for n in range(索引个数):
                for 列表, [datatype, size , count,  pad] in zip(描述列表, 标识列表):
                    if   datatype == 21: # 21, 2, 1, 0   # 'INDEX', 'BONE_PALETTE'
                        列表.append( bp.readuint16(count) )
                    elif datatype == 37: # 37, 4, 1, 0   # 'INDEX'
                        列表.append( bp.readuint32(count) )
                    elif datatype == 55: # 55, 4, 3, 0   # 'POSITION', 'NORMAL', 'BINORMAL', 'TANGENT', 'MORPH_POSITION', 'POSITION_BP', 'NORMAL_BP', 'BINORMAL_BP', 'TANGENT_BP', 'BLENDWEIGHT'
                        列表.append( bp.readfloat32(count) )
                    elif datatype == 16: # 16, 1, 4, 0   # 'COLOR'
                        列表.append( bp.readuint8(count) )
                    elif datatype == 54: # 54, 4, 2, 0   # 'TEXCOORD', 'TEXCOORD'
                        列表.append( bp.readfloat32(count) )
                    elif datatype ==  8: #  8, 1, 4, 0   # 'BLENDINDICES',
                        列表.append( bp.readuint8(count) )
                    elif datatype == 53: # 53, 4, 1, 0   # 'MORPHWEIGHTS'
                        列表.append( bp.readfloat32(count) )
                    elif datatype == 56: # 56, 4, 4, 0   # 'TRANSFORMS', 'TRANSFORMS', 'TRANSFORMS', 'TRANSFORMS', 'INSTANCETRANSFORMS', 'INSTANCETRANSFORMS', 'INSTANCETRANSFORMS', 'INSTANCETRANSFORMS'
                        列表.append( bp.readfloat32(count) )
                    else:
                        # print( [datatype, size , count,  pad])
                        列表.append( bp.read(size*count) )
            self.描述列表列表.append(描述列表)
        # 数据区大小, 结束
        # pad = bp.readuint8()  # 末尾固定为1


