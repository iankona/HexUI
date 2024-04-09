# import bpmodify
# import bpmodify.bqfunction as bq
# func = bq

from . import bpmodify
from .bpmodify import bqfunction as bq




class 类:
    def __init__(self): 
        self.filepath = ""
        self.name_object_dict = {}


    def read(self, filepath): # filepath 的判断由上层执行
        bg = bpmodify.bgmultipletree.类().filepath(filepath) # bg = 编辑.bpmodify.bgmultipletree.类().filepath(filepath)
        self.__read__blocks__(bg)
        self.filepath = bg.rdpath
        self.name_object_dict = bg.name_object_dict
        self.__read__srtm__()
        self.__read__srtm__lrtm__()
        bg.close()
        return self


    def __read__blocks__(self, bg):
        bq.context(bg)
        bq.bnodeslice(8, "HEAD")
        while True:
            if bq.remainsize() < 20: break
            with bq.readsliceseek0(12): 
                flag, name, size = bq.readuint32(), bq.readchar(4), bq.readuint32()
            bq.bnodeslice(12+size+8, name)
        bq.bnoderemainslice(f"BEOF")



        
    def __read__srtm__(self):
        bg = self.name_object_dict["SRTM"]
        bq.context(bg)
        with bq.readsliceseek0(20): 
            flag, name, size, nummat = bq.readuint32(), bq.readchar(4), bq.readuint32(), bq.readuint32()
        bq.bnodeslice(size=8, name="块_info")
        bq.bnodeuint32(num=1, name="块_size")
        nummat = bq.bnodeuint32(num=1, name="材质数量").uint32
        for i in range(nummat):
            with bq.readsliceseek0(12): 
                flag, name, size = bq.readuint32(), bq.readchar(4), bq.readuint32()
            bq.bnodeslice(size=12+size+8, name=f"lrtm_{i}")
        bq.bnoderemainslice(f"块_余下")



    def __read__srtm__lrtm__(self):
        srtm = self.name_object_dict["SRTM"]
        for name, bg in srtm.name_object_dict.items(): 
            if "lrtm" in name: self.__lrtm__(bg)


    def __lrtm__(self, bg):
        bq.context(bg)
        with bq.readsliceseek0(18): 
            flag, name, size, version, numchar = bq.readuint32(), bq.readchar(4), bq.readuint32(), bq.readuint16(), bq.readuint32()
        bq.bnodeslice(size=8, name="块_info")
        bq.bnodeuint32(num=1, name="块_size")
        bq.bnodeuint16(num=1, name="块_version")
        bq.bnodestring(4+numchar, name="材质名称", numtype="uint32", strtype="utf8")
        match version:
            case 3: sizeprop = 22 # 古剑2，model
            case 5: sizeprop = 30 # 古剑2，vmesh
            case 9: sizeprop = 35 # 古剑3，model
        bq.bnodeslice(size=sizeprop, name="材质属性")
        for i in range(3):
            numchar = bq.readuint32seek0()
            bq.bnodestring(4+numchar, name=f"1_贴图名称_{i}", numtype="uint32", strtype="utf8")

        numtexture = bq.bnodeuint32(num=1, name="第2栏贴图数量").uint32
        for i in range(numtexture):
            numchar = bq.readuint32seek0()
            bq.bnodestring(4+numchar, name=f"2_贴图名称_{i}", numtype="uint32", strtype="utf8")

        numtexture = bq.bnodeuint32(num=1, name="第3栏贴图数量").uint32
        for i in range(numtexture):
            numchar = bq.readuint32seek0()
            bq.bnodestring(4+numchar, name=f"3_贴图名称_{i}", numtype="uint32", strtype="utf8")

        bq.bnoderemainslice(f"块_余下")



    def __update__lrtm__(self):
        srtm = self.name_object_dict["SRTM"]
        for name, lrtm in srtm.items():
            if "lrtm_" not in name: continue
            lrtm["块_size"].uint32 = lrtm.updatesize() - 20
            lrtm.updateslice0b()

    def __update__srtm__(self):
        srtm = self.name_object_dict["SRTM"]
        srtm["块_size"].uint32 = srtm.updatesize() - 20
        srtm.updateslice0b()


    def __update__file__(self):
        slice0b = b""
        for name, bg in self.name_object_dict.items():
            slice0b = slice0b + bg.slice0b
        return slice0b


    def write(self, filepath): # filepath 的判断由上层执行
        self.__update__lrtm__()
        self.__update__srtm__()
        slice0b = self.__update__file__()
        file = open(filepath, "wb") # b表示可以写二进制
        file.write(slice0b)
        file.close()
        return self



if __name__== "__main__":
    filepath = r"E:\Program_StructFiles\GuJianQT3\asset\models\m01\m01_stone_gate_01.model"
    bg = bpmodify.bgmultipletree.类().filepath(filepath)
    model = 类(bg)
    pass