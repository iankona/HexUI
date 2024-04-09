from . import P

class Model: 
    def __init__(self): 
        # section2
        self.uuid = None
        self.name = None
        self.type = None # LimbNode, Null, Mesh, 
        # section3
        self.Version = None
        self.Properties70 = []
        self.Shading = None
        self.Culling = None


    def fromnode(self, fbxnode):
        # mysetting
        self.classname = "Model"
        # self.uuid = None
        # self.name = None
        # self.type = None # LimbNode, Null, Mesh, 
        self.iteration_values_to_properties(fbxnode)
        # self.Version = None
        # self.Properties70 = []
        # self.Shading = None
        # self.Culling = None
        self.iteration_attributes_to_properties(fbxnode)
        return self


    def iteration_values_to_properties(self, fbxnode):
        self.uuid = fbxnode.__values__[0]
        self.name = fbxnode.__values__[1]
        self.type = fbxnode.__values__[2]


    def iteration_attributes_to_properties(self, fbxnode):
        for propnode in fbxnode.__attributes__: 
            match propnode.blockname:
                case 'Version': self.Version = self.__Version__(propnode)
                case 'Properties70': self.Properties70 = self.__Properties70__(propnode)
                case 'Shading': self.Shading = self.__Shading__(propnode)
                case 'Culling': self.Culling = self.__Culling__(propnode)


    def __Version__(self, fbxnode):
        return fbxnode.__values__[0]

    def __Properties70__(self, fbxnode):
        result = []
        for propnode in fbxnode.__attributes__: result.append(P.P().fromnode(propnode))
        return result

    def __Shading__(self, fbxnode):
        if fbxnode.__values__[0] == 1: return True
        return False

    def __Culling__(self, fbxnode):
        return fbxnode.__values__[0]


# 'Version'
# 'Properties70'
# 'Shading'
# 'Culling'