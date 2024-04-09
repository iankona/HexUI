
class Geometry: 
    def __init__(self): 
        # section2
        self.uuid = None
        self.name = None
        self.type = None # Mesh
        # section3
        self.Vertices = []
        self.PolygonVertexIndex = []
        self.GeometryVersion = None
        self.LayerElementNormal = None
        self.LayerElementTangent = None
        self.LayerElementColor = None
        self.LayerElementUV = None
        self.LayerElementMaterial = None
        self.Layer = None
        

    def fromnode(self, fbxnode):
        # mysetting
        self.classname = "Geometry"
        # self.uuid = None
        # self.name = None
        # self.type = None # Mesh
        self.iteration_values_to_properties(fbxnode)
        # self.Vertices = []
        # self.PolygonVertexIndex = []
        # self.GeometryVersion = None
        # self.LayerElementNormal = None
        # self.LayerElementTangent = None
        # self.LayerElementColor = None
        # self.LayerElementUV = None
        # self.LayerElementMaterial = None
        # self.Layer = None 
        self.iteration_attributes_to_properties(fbxnode)
        return self
    

    def iteration_values_to_properties(self, fbxnode):
        self.uuid = fbxnode.__values__[0]
        self.name = fbxnode.__values__[1]
        self.type = fbxnode.__values__[2]




    def iteration_attributes_to_properties(self, fbxnode):
        for propnode in fbxnode.__attributes__: 
            match propnode.blockname:
                case 'Vertices': self.Vertices = self.__Vertices__(propnode)
                case 'PolygonVertexIndex': self.PolygonVertexIndex = self.__PolygonVertexIndex__(propnode)
                case 'GeometryVersion': pass
                case 'LayerElementNormal': pass
                case 'LayerElementTangent': pass
                case 'LayerElementColor': pass
                case 'LayerElementUV': pass
                case 'LayerElementMaterial': pass
                case 'Layer': pass


    def __Vertices__(self, fbxnode):
        return fbxnode.__values__[0]

    def __PolygonVertexIndex__(self, fbxnode):
        return fbxnode.__values__[0]





# 'Vertices'
# 'PolygonVertexIndex'
# 'GeometryVersion'
# 'LayerElementNormal'
# 'LayerElementTangent'
# 'LayerElementColor'
# 'LayerElementUV'
# 'LayerElementMaterial'
# 'Layer'