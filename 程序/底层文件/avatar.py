import os
import traceback
import xml.etree.ElementTree as ET


class avatarnode:
    def __init__(self):
        self.type = None
        self.file = None
        self.tag = None

class avatarmaterial:
    def __init__(self):
        self.type = None
        self.surface = None
        self.data = None

        self.name = None
        self.materialtype = None
        self.IrisRadius = None
        self.HeightScale = None
        self.HeightThreshold = None
        self.PupilScale = None
        self.IrisBrightness = None
        self.LimbusDarkScale = None
        self.AOScale = None
        self.TexBaseIris = None
        self.TexBaseSclera = None
        self.TexAOMask = None
        self.FrontOffset = None
        self.Debug = None



# class 类:
#     def __init__(self, filepath):
#         self.filepath = filepath
#         self.root = self.__xmlET__(filepath)

class 类:
    def __init__(self, bp):
        self.filepath = bp.rdpath
        self.root = self.__xmlET__(bp.rdpath)
        self.路径节点列表 = self.__单独路径节点列表__()
        self.材质节点列表 = self.__单独材质节点列表__()


    def __xmlET__(self, filepath):
        tree = ET.parse(filepath)
        root = tree.getroot()
        return root


    def __单独路径节点列表__(self):
        路径节点列表 = [avatarnode() for child in self.root]
        for child, anode in zip(self.root, 路径节点列表):
            anode.type = child.tag
            if "file" in child.attrib: anode.file = child.attrib["file"]
            if "tag" in child.attrib: anode.tag = child.attrib["tag"]
        return 路径节点列表   

    def __单独材质节点列表__(self):
        materialchildren = None
        for child in self.root:
            if "tag" not in child.attrib: continue
            if child.attrib["tag"] == "morphface": materialchildren = child

        if materialchildren == None: return []
        材质节点列表 = [avatarmaterial() for child in materialchildren]
        for child, mnode in zip(materialchildren, 材质节点列表):
            mnode.type = child.tag
            if "surface" in child.attrib: mnode.surface = child.attrib["surface"]
            if "data" in child.attrib: mnode.data = child.attrib["data"]

        for mnode in 材质节点列表:
            mnode.name = mnode.surface
            for line in mnode.data.splitlines():
                if "=" not in line:
                    mnode.materialtype = line
                    continue

                varname, varvalue = line.split("=")
                match varname:
                    case 'IrisRadius': mnode.IrisRadius = float(varvalue)
                    case 'HeightScale': mnode.HeightScale = float(varvalue)
                    case 'HeightThreshold': mnode.HeightThreshold = float(varvalue)
                    case 'PupilScale': mnode.PupilScale = float(varvalue)
                    case 'IrisBrightness': mnode.IrisBrightness = float(varvalue)
                    case 'LimbusDarkScale': mnode.LimbusDarkScale = float(varvalue)
                    case 'AOScale': mnode.AOScale = float(varvalue)
                    case 'TexBaseIris': mnode.TexBaseIris = varvalue
                    case 'TexBaseSclera': mnode.TexBaseSclera = varvalue
                    case 'TexAOMask': mnode.TexAOMask = varvalue
                    case 'FrontOffset': mnode.FrontOffset = float(varvalue)
                    case 'Debug': mnode.Debug = float(varvalue)
        return 材质节点列表

if __name__ == "__main__":
    filepath = r"E:\Program_StructFiles\GuJianQT3\asset\avatar\actress1_coat.avatar"
    测试 = 类(filepath)
    pass


    # tree = ET.parse(filepath)
    # root = tree.getroot()
    # # for child in root:
    # #     print(child.tag)
    # #     print(child.attrib)
    # #     print(child.text)
    # #     print(child.tail)
    # #     # skin
    # #     # {'file': 'characters/actress1/models/actress1_default_luopan.model', 'tag': 'luopan'}
    # #     # None
    # #     # 
    # materialchildren = None
    # for child in root:
    #     if "tag" not in child.attrib: continue
    #     if child.attrib["tag"] == "morphface": materialchildren = child

    # for child in materialchildren:
    #     print(child.tag)
    #     print(child.attrib)
    #     print(child.text)
    #     print(child.tail)
    #     # mtl
    #     # {'surface': '<g3_eye>_right', 'data': 'Mesh_G3_Ch_Eye\nIrisRadius=0.126\nHeightScale=0.500\nHeightThreshold=1.700\nPupilScale=1.000\nIrisBrightness=0.600\nLimbusDarkScale=1.010\nAOScale=1.000\nTexBaseIris=materials/textures/eye/eye_iris_004.dds\nTexBaseSclera=materials\\textures\\eye\\eye_sclera_001.dds\nTexAOMask=materials\\textures\\eye\\T_FemaleEyeAOMask.dds\nFrontOffset=0.090\nDebug=1.000'}
    #     # None
    #     # 
    # pass

# Mesh_G3_Ch_Eye
# IrisRadius=0.126
# HeightScale=0.500
# HeightThreshold=1.700
# PupilScale=1.000
# IrisBrightness=0.600
# LimbusDarkScale=1.010
# AOScale=1.000
# TexBaseIris=materials/textures/eye/eye_iris_004.dds
# TexBaseSclera=materials\textures\eye\eye_sclera_001.dds
# TexAOMask=materials\textures\eye\T_FemaleEyeAOMask.dds
# FrontOffset=0.090
# Debug=1.000


# Mesh_G3_Ch_Eye
# IrisRadius=0.126
# HeightScale=0.500
# HeightThreshold=1.700
# PupilScale=1.000
# IrisBrightness=0.600
# LimbusDarkScale=1.010
# AOScale=1.000
# TexBaseIris=materials/textures/eye/eye_iris_004.dds
# TexBaseSclera=materials\textures\eye\eye_sclera_001.dds
# TexAOMask=materials\textures\eye\T_FemaleEyeAOMask.dds
# FrontOffset=0.090
# Debug=1.000


# 1 - 巩膜（sclera）  ：也称为“眼白”，通常非常湿润，包含少量的触感纹理、血丝等细节。
# 2 - 角膜缘（limbus）：角膜缘存在于虹膜和巩膜之间的深色环形。有些眼睛中的角膜缘更为明显，从侧面看时往往会消退。
# 3 - 虹膜（iris）    ：虹膜是围绕在眼睛中心周围的一圈色环。如果某个人有“绿”眼睛，就是因为虹膜主要是绿色的。在真实的眼睛中，虹膜是类似肌肉的纤维结构，有扩张和收缩功能，以让更多光线进入瞳孔或者不让光线进入瞳孔。还需要注意的是，在真实世界中，虹膜实际上更像是圆盘或锥形，不会向眼部其余部分突出。
# 4 - 瞳孔（pupil）   ：瞳孔是眼睛中心的黑点。这是一个孔，光线穿过这个孔后才会被视网膜的视杆和视锥捕捉到。
# 5 - 角膜（cornea）  : 角膜是位于虹膜表面上的一层透明的、充满液体的圆顶结构。


# 角膜的半透和光泽反射效果。
# 瞳孔的次表面散射。
# 瞳孔的缩放。最好根据整个场景的光照强度动态调整缩放大小。
# 虹膜的颜色变化。
# 其它眼球细节。