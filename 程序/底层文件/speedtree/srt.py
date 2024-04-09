from . import version0510
from . import version0520
from . import version0700


class Object: pass
class 类:
    def __init__(self, bp):
        self.filepath = ""
        self.head = Object()
        self.blocks = {}
        self.filelist = [] # 引用xac设置
        self.材质列表 = []
        self.网格列表 = []
        self.read_blocks(bp)
        pass

    def read_blocks(self, bp):
        head = bp.copy().readchar(10)
        match head:
            case "SRT 05.1.0": self.file_type05(o=version0510.类(bp))
            case "SRT 05.2.0": self.file_type05(o=version0520.类(bp))
            case "SRT 07.0.0": self.file_type07(o=version0700.类(bp))
            case _: print(f"SRT::FILE::未支持的srt(speedtree)文件格式!{bp.rdpath}")



    def file_type05(self, o):
        def 网格处理(类型网格列表=[]):
            网格列表 = []
            for 顶点列表, 顶点UV列表, 顶点Loop列表, mbuslist in 类型网格列表:
                for 材质索引, loop左值, loop个数, vert左值, vert个数 in mbuslist:
                    新顶点列表 = 顶点列表[vert左值: vert左值+vert个数]
                    新顶点UV列表 = 顶点UV列表[vert左值: vert左值+vert个数]
                    新顶点Loop列表 = [loop个值-vert左值 for loop个值 in 顶点Loop列表[loop左值: loop左值+loop个数]]
                    网格列表.append([新顶点列表, 新顶点UV列表, 新顶点Loop列表, 材质索引])
            return 网格列表


        树干网格列表 = 网格处理(o.树干网格列表)
        树枝网格列表 = 网格处理(o.树枝网格列表)
        树叶网格列表 = 网格处理(o.树叶网格列表)
        # self.o = o
        self.filepath = o.filepath
        self.head = o.head
        self.blocks = o.blocks
        self.材质列表 = o.材质列表
        self.网格列表 = 树干网格列表 + 树枝网格列表 + 树叶网格列表


        树干网格列表1 = 网格处理(o.树干网格列表1)
        树枝网格列表1 = 网格处理(o.树枝网格列表1)
        树叶网格列表1 = 网格处理(o.树叶网格列表1)
        lod1 = Object()
        lod1.filelist = []
        lod1.filepath = o.filepath[0:-4] + f"_1f.srt"
        lod1.head = o.head
        lod1.材质列表 = o.材质列表
        lod1.网格列表 = 树干网格列表1 + 树枝网格列表1 + 树叶网格列表1
        self.filelist.append(lod1)


    def file_type07(self, o):
        # self.o = o
        self.filepath = o.filepath
        self.head = o.head
        self.blocks = o.blocks
        self.blocks_mesh = o.blocks_mesh
        self.材质列表 = o.材质列表
        self.网格列表 = o.网格列表  # [[顶点列表, 顶点UV列表, 顶点Loop列表, i], ]

        lod1 = Object()
        lod1.filelist = []
        lod1.filepath = o.filepath[0:-4] + f"_1f.srt"
        lod1.head = o.head
        lod1.材质列表 = o.材质列表1
        lod1.网格列表 = o.网格列表1
        self.filelist.append(lod1)
