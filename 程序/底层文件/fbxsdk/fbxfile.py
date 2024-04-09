
from . import fbxnode
from . import Geometry
from . import Model
from . import AnimationCurveNode
from . import AnimationCurve

from . import C


class fbxfile:
    def __init__(self):
        self.FBXHeaderExtension = None
        self.FileId = None
        self.CreationTime = None
        self.Creator = None
        self.GlobalSettings = []
        self.Documents = []
        self.References = []
        self.Definitions = []
        self.Objects = []
        self.Connections = []
        self.Takes = []


    def frombp(self, bp):
        self.blocks = {}
        self.block_section1(bp)
        self.block_section2(bp)
        self.block_section3(bp)
        self.block_section4(bp)
        self.iteration_blocks_to_attributes()
        self.iteration_attributes_to_properties()
        return self
    
        
    def block_section1(self, bp):
        self.filepath = bp.rdpath
        self.blocks["b_head"] = bp.readslice(27)


    def block_section2(self, bp):
        pass


    def block_section3(self, bp):
        count = -1
        while True:
            count += 1
            if bp.readuint32seek0() == 0: break
            sliceleft, sliceright = bp.tell(), bp.readuint32seek0()
            self.blocks[f"b_prop{count}"] = bp.readslice(sliceright-sliceleft)


    def block_section4(self, bp):
        self.blocks["b_beof"] = bp.readremainslice()


    def iteration_blocks_to_attributes(self):
        self.__attributes__ = []
        for keychar, bp in self.blocks.items(): 
            if "b_prop" in keychar: self.__attributes__.append(fbxnode.fbxnode(bp))


    def iteration_attributes_to_properties(self):
        for propnode in self.__attributes__: 
            match propnode.blockname:
                case 'FBXHeaderExtension': pass
                case 'FileId': pass
                case 'CreationTime': pass
                case 'Creator': pass
                case 'GlobalSettings': pass
                case 'Documents': pass
                case 'References': pass
                case 'Definitions': pass
                case 'Objects': self.Objects = self.__Objects__(propnode)
                case 'Connections': self.Connections = self.__Connections__(propnode)
                case 'Takes': pass


    def __Objects__(self, fbxnode):
        result = []
        for propnode in fbxnode.__attributes__:
            match propnode.blockname:
                case 'NodeAttribute': pass
                case 'Geometry': result.append(Geometry.Geometry().fromnode(propnode))
                case 'Model': result.append(Model.Model().fromnode(propnode))
                case 'Pose': pass
                case 'Material': pass
                case 'Video': pass
                case 'Texture': pass
                case 'AnimationStack': pass
                case 'AnimationCurve': result.append(AnimationCurve.AnimationCurve().fromnode(propnode))
                case 'AnimationCurveNode': result.append(AnimationCurveNode.AnimationCurveNode().fromnode(propnode))
                case 'AnimationLayer': pass
        return result

    def __Connections__(self, fbxnode):
        result = []
        for propnode in fbxnode.__attributes__: result.append(C.C().fromnode(propnode))
        return result





# FBXHeaderExtension
# FileId
# CreationTime
# Creator
# GlobalSettings
# Documents
# References
# Definitions
# Objects
# Connections
# Takes


# Objects = [] # List
# 'NodeAttribute'
# 'Geometry'
# 'Model'
# 'Pose'
# 'Material'
# 'Video'
# 'Texture'
# 'AnimationStack'
# 'AnimationCurve'
# 'AnimationCurveNode'
# 'AnimationLayer'