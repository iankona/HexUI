


class NiTransformData:
    def __init__(self, bp, kffile):
        self.位置帧列表 = []
        self.旋转帧列表 = []
        self.缩放帧列表 = []
        self.旋转曲线帧列表 = []
        self.读取成员(bp, kffile)

    def 读取成员(self, bp, kffile):
        self.读取旋转帧(bp)
        self.读取位置帧(bp)
        self.读取缩放帧(bp)

    def 读取位置帧(self, bp):
        位置帧数 = bp.readuint32()
        if 位置帧数 == 0: return ""
        位置模式 = bp.readuint32()
        if 位置模式 == 1:
            self.位置帧列表 = [ [30*bp.readfloat32(), bp.readfloat32(3)] for i in range(位置帧数) ] # [frame, position]


    def 读取旋转帧(self, bp):
        旋转帧数 = bp.readuint32()
        if 旋转帧数 == 0: return ""
        旋转模式 = bp.readuint32()
        if 旋转模式 == 1:
            self.旋转帧列表 = [ [30*bp.readfloat32(), bp.readfloat32(4)] for i in range(旋转帧数) ] # [frame, quaternion]
        if 旋转模式 == 4:  # 前面旋转帧数固定为1
            self.读取旋转子帧(bp)


    def 读取旋转子帧(self, bp):
        self.旋转曲线帧列表 = [ [], [], [] ]
        for i in range(3):
            子帧数, 旋转模式 = bp.readuint32(2)
            if 旋转模式 == 2:
                self.旋转曲线帧列表[i] = [ [30*bp.readfloat32(), bp.readfloat32(), bp.readfloat32(2)][0:2] for n in range(子帧数) ]# [frame, quaternion]


    def 读取缩放帧(self, bp):
        缩放帧数 = bp.readuint32()
        if 缩放帧数 == 0: return ""
        缩放模式 = bp.readuint32()
        if 缩放模式 == 1:
            self.缩放帧列表 = [ [30*bp.readfloat32(), bp.readfloat32(3)] for i in range(缩放帧数) ] # [frame, scale]


