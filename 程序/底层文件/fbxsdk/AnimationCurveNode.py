from . import P

class AnimationCurveNode: 
    def __init__(self): 
        # section2
        self.uuid = None
        self.type = None # RAnimCurveNode, TAnimCurveNode, SAnimCurveNode
        self.basetype = None
        # section3
        self.Properties70 = []


    def fromnode(self, fbxnode):
        # mysetting
        self.classname = "AnimationCurveNode"
        # self.uuid = None
        # self.type = None # RAnimCurveNode, TAnimCurveNode, SAnimCurveNode
        # self.basetype = None
        self.iteration_values_to_properties(fbxnode)
        # self.Properties70 = []
        self.iteration_attributes_to_properties(fbxnode)
        return self


    def iteration_values_to_properties(self, fbxnode):
        self.uuid = fbxnode.__values__[0]
        self.type = fbxnode.__values__[1]
        self.basetype = fbxnode.__values__[2]


    def iteration_attributes_to_properties(self, fbxnode):
        for propnode in fbxnode.__attributes__: 
            match propnode.blockname:
                case 'Properties70': self.Properties70 = self.__Properties70__(propnode)


    def __Properties70__(self, fbxnode):
        result = []
        for propnode in fbxnode.__attributes__: result.append(P.P().fromnode(propnode))
        return result

