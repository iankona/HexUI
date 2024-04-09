class NiTransformEvaluator:
    def __init__(self, bp, kffile):
        self.name = None
        self.position = None
        self.quaternion = None
        self.scale = None
        self.NiTransformDataID = None
        self.读取成员(bp, kffile)

    def 读取成员(self, bp, kffile):
        # Evaluator
        self.name = kffile.names[bp.readuint32()]
        proptype, ctrltype, ctrlid, evalid  = bp.readint32(4) # nameindex
        chantypes = bp.readuint8(4)
        # NiTransformEvaluator
        self.position = bp.readfloat32(3)
        self.quaternion = bp.readfloat32(4)
        self.scale = bp.readfloat32()
        #
        self.NiTransformDataID = bp.readint32()

