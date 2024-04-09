
class NiBSplineCompTransformEvaluator:
    def __init__(self, bp, kffile):
        pass

        # 名称 = kffile.名称列表[bp.readu32()]
        # PropType   = self.名称列表[数据块.bp.readi32()]
        # CtrlType   = self.名称列表[数据块.bp.readi32()]
        # CtrlID     = self.名称列表[数据块.bp.readi32()]
        # EvalID     = self.名称列表[数据块.bp.readi32()]
        # ChanTypes = 数据块.bp.readu8(4)

        # # NiTransformEvaluator
        # 数据块.position = 数据块.bp.readf32(3)
        # 数据块.quaternion = 数据块.bp.readf32(4) # w, x, y, z,  w = -w
        # 数据块.scale = 数据块.bp.readf32()
        # # 
        # unsed = bp.readf32(4)
        # 数据块.parentID = bp.readi32()