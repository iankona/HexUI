


class LRTM:
    def __init__(self, bp):
        self.version = 9
        self.材质名称 = ""
        self.相对贴图地址列表 = []
        self.read_block_data(bp)


    def read_block_data(self, bp):
        version, 材质名称 = bp.readuint16(), bp.readchar(bp.readuint32())
        match version:
            case 3: 材质属性块 = bp.readslice(22) # 古剑2，model
            case 5: 材质属性块 = bp.readslice(30) # 古剑2，vmesh
            case 9: 材质属性块 = bp.readslice(35) # 古剑3，model

        相对贴图地址列表 = []
        for i in range(3):
            charnum = bp.readuint32()
            if charnum > 0: 相对贴图地址列表.append( bp.readchar(charnum))
        for i in range(bp.readuint32()):
            charnum = bp.readuint32()
            if charnum > 0: 相对贴图地址列表.append( bp.readchar(charnum))
        for i in range(bp.readuint32()):
            charnum = bp.readuint32()
            if charnum > 0: 相对贴图地址列表.append( bp.readchar(charnum))

        # match version:
        #     case 3: 末尾块 = bp.readslice(40) # 古剑2，model
        #     case 5: 末尾块 = bp.readslice(48) # 古剑2，vmesh
        #     case 9: 末尾块 = bp.readslice(52) # 古剑3，model
        self.version = version
        self.材质名称 = 材质名称
        self.相对贴图地址列表 = 相对贴图地址列表

