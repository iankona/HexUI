# https://github.com/enenra/x4modding/wiki/Model-file-infomation-(_arc)

from . import 材质数量, 材质, 贴图属性, 字符名称
from . import 骨架, 模型, 权重, 形态键, 内嵌文件


class Object: pass
class 类:
    def __init__(self, bp):
        self.filepath = ""
        self.head = Object()
        self.blocks = {}
        self.blocks_lrtmlist = []
        self.blocks_hsmvlist = []
        self.blocks_thgwlist = []
        self.blocks_stcmlist = []
        self.blocks_filelist = []
        self.blocks_unknlist = []
        self.LEKS = Object()
        self.lrtmlist = []
        self.hsmvlist = []
        self.thgwlist = []
        self.stcmlist = []
        self.filelist = []
        self.unknlist = []
        self.read_blocks(bp)

        self.read_head()
        self.read_骨架()
        self.read_模型()
        self.read_权重()
        self.read_形态键()
        self.read_内嵌文件()


    def read_blocks(self, bp):
        self.filepath = bp.rdpath

        self.blocks["b_head"] = [0, 8, 0, bp.readslice(8)]
        while True:
            if bp.remainsize() < 12: break
            flag, size, version = bp.readuint32(3)
            match flag:
                case  7: self.blocks["b_file_info"] = [flag, size, version, bp.readslice(size)]
                case 11: self.blocks["b_leks"] = [flag, size, version, bp.readslice(size)]
                case 13: self.blocks_material(flag, size, version, bp)
                case  1: self.blocks_hsmvlist.append([flag, size, version, bp.readslice(size)])
                case  2: self.blocks_thgwlist.append([flag, size, version, bp.readslice(size)])
                case 12: self.blocks_stcmlist.append([flag, size, version, bp.readslice(size)])
                case  8: self.blocks_filelist.append([flag, size, version, bp.readslice(size)])
                case  _: self.blocks_unknlist.append([flag, size, version, bp.readslice(size)])
        self.blocks["b_endof"] = [0, 0, 0, bp.readremainslice()]


    def blocks_material(self, flag, size, version, bp):
        self.blocks["b_material"] = [flag, size, version, bp.readslice(size)]
        flag, size, version, srtm = self.blocks["b_material"]

        数量 = 材质数量.函数(flag, size, version, srtm)
        for i in range(数量):
            flag, size, version = bp.readuint32(3)
            lrtm = bp.readslice(size)

            材质属性列表, 贴图数量, 材质名称 = 材质.函数(flag, size, version, lrtm)

            列表1 = []
            列表2 = []
            贴图属性列表 = []
            贴图名称列表 = []
            for j in range(贴图数量):
                trtm = bp.readslice(28)
                crtm = bp.readslice(4 + bp.readuint32seek0())
                列表1.append(trtm)
                列表2.append(crtm)
                贴图属性列表.append(贴图属性.函数(trtm))
                贴图名称列表.append(字符名称.函数(crtm))

            self.lrtmlist.append([材质名称, 材质属性列表, 贴图名称列表, 贴图属性列表])
            self.blocks_lrtmlist.append([flag, size, version, lrtm, 列表1, 列表2])


    def read_head(self):
        flag, size, version, bx = self.blocks["b_head"]
        self.head.magic = bx.readchar(4) # 58 41 43 20 ("XAC ") # 1, 0, 0, 1 # ["XAC ", 1, 0, 0, 1]
        self.head.version = bx.readuint8(3) + [0]
        # if bx.readuint8(): bx.endian = ">"  # 小端，低位存大数


    def read_骨架(self):
        flag, size, version, leks = self.blocks["b_leks"]
        self.LEKS.骨骼节点列表 = 骨架.函数(flag, size, version, leks)


    def read_模型(self):
        for flag, size, version, hsmv in self.blocks_hsmvlist:
            顶点列表, 顶点UV列表, 区网格信息列表, 区权重索引Loop列表 = 模型.函数(flag, size, version, hsmv)
            HSMV = Object()
            HSMV.顶点列表 = 顶点列表
            HSMV.顶点UV列表 = 顶点UV列表
            HSMV.顶点划分信息列表 = 区网格信息列表
            HSMV.区权重索引Loop列表 = 区权重索引Loop列表
            self.hsmvlist.append(HSMV)


    def read_权重(self):
        for flag, size, version, thgw in self.blocks_thgwlist:
            权重值Loop列表, 区权重信息列表 = 权重.函数(flag, size, version, thgw)
            THGW = Object()
            THGW.权重值Loop列表 = 权重值Loop列表
            THGW.区权重信息列表 = 区权重信息列表
            self.thgwlist.append(THGW)


    def read_形态键(self):
        for flag, size, version, stcm in self.blocks_stcmlist:
            形态键列表 = 形态键.函数(flag, size, version, stcm)
            STCM = Object()
            STCM.形态键列表 = 形态键列表
            self.stcmlist.append(STCM)


    def read_内嵌文件(self):
        for flag, size, version, bp in self.blocks_filelist:
            index, bp = 内嵌文件.函数(flag, size, version, bp)
            xac = 类(bp) # filepath已被赋值
            xac.filepath = bp.rdpath[0:-4] + f"_{index}f.xac" # filepath与原文件进行区分
            self.filelist.append(xac)
