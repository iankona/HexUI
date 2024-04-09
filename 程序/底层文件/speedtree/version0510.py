from . import version0520

class Object: pass
class 类(version0520.类):
    def __init__(self, bp):
        self.filepath = ""
        self.head = Object()
        self.blocks = {}
        self.blocks_vert = []
        self.blocks_loop = []
        self.材质列表 = []
        self.网格列表 = []
        self.网格列表1 = []
        self.read_blocks(bp)
        self.read_head()
        self.read_material()
        self.read_mesh_data()
        # self.read_mesh_data_细枝()

    def blocks_section1_wind(self, bp):
        self.blocks["b_wind"] = bp.readslice(152)


