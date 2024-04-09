class NiConstTransformEvaluator:

    def __init__(self, bp, kffile):
        pass
    #     for 数据块 in self.数据块列表:
    #         if 数据块.类型 != "NiConstTransformEvaluator": continue
    #         # Evaluator
    #         数据块.名称 = self.名称列表[数据块.bp.readu32()]
    #         PropType   = self.名称列表[数据块.bp.readi32()]
    #         CtrlType   = self.名称列表[数据块.bp.readi32()]
    #         CtrlID     = self.名称列表[数据块.bp.readi32()]
    #         EvalID     = self.名称列表[数据块.bp.readi32()]
    #         ChanTypes = 数据块.bp.readu8(4)
    #         # NiTransformEvaluator
    #         数据块.position = 数据块.bp.readf32(3)
    #         数据块.quaternion = 数据块.bp.readf32(4) # w, x, y, z,  w = -w
    #         数据块.scale = 数据块.bp.readf32()