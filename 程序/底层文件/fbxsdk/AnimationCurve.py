


class AnimationCurve: 
    def __init__(self): 
        # section2
        self.uuid = None
        self.type = None # AnimCurve
        self.basetype = None
        # section3
        self.Default = None
        self.KeyVer = None
        self.KeyTime = None
        self.KeyValueFloat = None
        self.KeyAttrFlags = None
        self.KeyAttrDataFloat = None
        self.KeyAttrRefCount = None


    def fromnode(self, fbxnode):
        # mysetting
        self.classname = "AnimationCurve"
        # self.uuid = None
        # self.type = None # AnimCurve
        # self.basetype = None
        self.iteration_values_to_properties(fbxnode)
        # self.Default = None
        # self.KeyVer = None
        # self.KeyTime = None
        # self.KeyValueFloat = None
        # self.KeyAttrFlags = None
        # self.KeyAttrDataFloat = None
        # self.KeyAttrRefCount = None
        self.iteration_attributes_to_properties(fbxnode)
        return self



    def iteration_values_to_properties(self, fbxnode):
        self.uuid = fbxnode.__values__[0]
        self.type = fbxnode.__values__[1]
        self.basetype = fbxnode.__values__[2]


    def iteration_attributes_to_properties(self, fbxnode):
        for propnode in fbxnode.__attributes__: 
            match propnode.blockname:
                case 'Default': self.Default = self.__Default__(propnode)
                case 'KeyVer': self.KeyVer = self.__KeyVer__(propnode)
                case 'KeyTime': self.KeyTime = self.__KeyTime__(propnode)
                case 'KeyValueFloat': self.KeyValueFloat = self.__KeyValueFloat__(propnode)
                case 'KeyAttrFlags': self.KeyAttrFlags = self.__KeyAttrFlags__(propnode)
                case 'KeyAttrDataFloat': self.KeyAttrDataFloat = self.__KeyAttrDataFloat__(propnode)
                case 'KeyAttrRefCount': self.KeyAttrRefCount = self.__KeyAttrRefCount__(propnode)


    def __Default__(self, fbxnode):
        return fbxnode.__values__[0]

    def __KeyVer__(self, fbxnode):
        return fbxnode.__values__[0]

    def __KeyTime__(self, fbxnode):
        return fbxnode.__values__[0]

    def __KeyValueFloat__(self, fbxnode):
        return fbxnode.__values__[0]
    
    def __KeyAttrFlags__(self, fbxnode):
        return fbxnode.__values__[0]
    
    def __KeyAttrDataFloat__(self, fbxnode):
        return fbxnode.__values__[0]
    
    def __KeyAttrRefCount__(self, fbxnode):
        return fbxnode.__values__[0]
    

# 'Default'
# 'KeyVer'
# 'KeyTime'
# 'KeyValueFloat'
# 'KeyAttrFlags'
# 'KeyAttrDataFloat'
# 'KeyAttrRefCount'
