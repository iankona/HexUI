class P: 
    def __init__(self): 
        self.name = None
        self.type = None
        self.basetype = None
        self.padchars = None
        self.value = None


    def fromnode(self, fbxnode):
        # self.name = None
        # self.type = None
        # self.basetype = None
        # self.padchars = None
        self.iteration_values_to_properties(fbxnode)
        # self.value = None
        self.matchvalue(fbxnode)
        return self
    

    def iteration_values_to_properties(self, fbxnode):
        self.name = fbxnode.__values__[0]
        self.type = fbxnode.__values__[1]
        self.basetype = fbxnode.__values__[2]
        self.padchars = fbxnode.__values__[3]


    def matchvalue(self, fbxnode):
        match self.name:
            # Model
            case "ScalingMax": self.value = [fbxnode.__values__[4], fbxnode.__values__[5], fbxnode.__values__[6]]
            case "DefaultAttributeIndex": self.value = fbxnode.__values__[4]
            case "PreferedAngleX": self.value = fbxnode.__values__[4]
            case "PreferedAngleY": self.value = fbxnode.__values__[4]
            case "PreferedAngleZ": self.value = fbxnode.__values__[4]
            case "Lcl Translation": self.value = [fbxnode.__values__[4], fbxnode.__values__[5], fbxnode.__values__[6]]
            case "Lcl Rotation": self.value = [fbxnode.__values__[4], fbxnode.__values__[5], fbxnode.__values__[6]]
            case "Lcl Scaling": self.value = [fbxnode.__values__[4], fbxnode.__values__[5], fbxnode.__values__[6]]
            # AnimationCurveNode
            case "d|X": self.value = fbxnode.__values__[4]
            case "d|Y": self.value = fbxnode.__values__[4]
            case "d|Z": self.value = fbxnode.__values__[4]


