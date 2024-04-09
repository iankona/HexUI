
class C: 
    def __init__(self): 
        self.name = None
        self.uuid_child = None
        self.uuid_parent = None


    def fromnode(self, fbxnode):
        # self.name = None
        # self.uuid_child = None
        # self.uuid_parent = None
        self.iteration_values_to_properties(fbxnode)
        return self

    def iteration_values_to_properties(self, fbxnode):
        self.name = fbxnode.__values__[0]
        self.uuid_child = fbxnode.__values__[1]
        self.uuid_parent = fbxnode.__values__[2]


