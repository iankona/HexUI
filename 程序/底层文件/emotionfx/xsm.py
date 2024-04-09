# https://github.com/enenra/x4modding/wiki/Model-file-infomation-(_arc)  as a reference

# Set a bones matrix to a custom matrix
# https://blender.stackexchange.com/questions/9318/set-a-bones-matrix-to-a-custom-matrix/38337#38337  as a reference

# How can I manually calculate bpy.types.PoseBone.matrix using Blender's Python API?
# https://blender.stackexchange.com/questions/44637/how-can-i-manually-calculate-bpy-types-posebone-matrix-using-blenders-python-ap   def get_mat_offs(bone):   as a reference

# calculate bone location/rotation from fcurve-animation-data
# https://blenderartists.org/t/calculate-bone-location-rotation-from-fcurve-animation-data/499045


class Object: pass
class 类:
    def __init__(self, bp):
        self.head = Object()
        self.blocks = {}
        self.blocks_somelist = []
        self.ACTI = Object()
        self.read_blocks(bp)
        self.read_动画数据()

    def read_blocks(self, bp):
        self.filepath = bp.rdpath

        self.blocks["b_head"] = [0, 8, 0, bp.readslice(8)]
        while True:
            if bp.remainsize() < 12: break
            flag, size, version = bp.readuint32(3)
            match flag:
                case 201: self.blocks["b_file_info"] = [flag, size, version, bp.readslice(size)]
                case 202: self.blocks["b_action_data"] = [flag, size, version, bp.readslice(size)]
                case   _: self.blocks_somelist.append([flag, size, version, bp.readslice(size)])
        self.blocks["b_endof"] = [0, 0, 0, bp.readremainslice()]


    def read_动画数据(self):
        flag, size, version, bp = self.blocks["b_action_data"]

        bindings = []
        骨骼节点列表 = []
        for i in range(bp.readuint32()): # 骨骼数量
            # rotation, rotation_bind, rotation_scale, rotation_scale_bind, position, scale, position_bind, scale_bind = self.读取_绑定姿态(bp.readslice(80))
            bindings.append(self.读取_绑定姿态(bp.readslice(80)))
            [位置帧数, 旋转帧数, 缩放帧数, 旋转缩放帧数], f_max_error, 骨骼名称 = bp.readuint32(4), bp.readfloat32(), bp.readchar(bp.readuint32())

            位置帧列表 = [[bp.readfloat32(3), 30*bp.readfloat32()] for i in range(位置帧数)] # [[x, y, z], frame] #  [x for x in range(0)] == []
            旋转帧列表 = []
            for i in range(旋转帧数): # [[x, y, z, w], frame]
                frotation = [ int16/32767 for int16 in bp.readint16(4)]
                frame = 30*bp.readfloat32()
                旋转帧列表.append([frotation, frame])
            缩放帧列表 = [[bp.readfloat32(3), 30*bp.readfloat32()] for i in range(缩放帧数)] # [[x, y, z], frame]

            旋转缩放帧列表 = []
            for i in range(旋转缩放帧数): # [[x, y, z, w], frame]
                frotation_scale = [ int16/32767 for int16 in bp.readint16(4)]
                frame = 30*bp.readfloat32()
                旋转帧列表.append([frotation_scale, frame])

            骨骼节点列表.append( [骨骼名称, 位置帧列表, 旋转帧列表, 缩放帧列表, 旋转缩放帧列表] )

        self.ACTI.bindings = bindings
        self.ACTI.骨骼节点列表 = 骨骼节点列表


    def 读取_绑定姿态(self, bp): # 80字节
        rotation = [int16/32767 for int16 in bp.readint16(4)]
        rotation_bind = [int16/32767 for int16 in bp.readint16(4)]
        rotation_scale = [int16/32767 for int16 in bp.readint16(4)]
        rotation_scale_bind = [int16/32767 for int16 in bp.readint16(4)]
        position = bp.readfloat32(3)
        scale = bp.readfloat32(3)
        position_bind = bp.readfloat32(3)
        scale_bind = bp.readfloat32(3)
        return [rotation, rotation_bind, rotation_scale, rotation_scale_bind, position, scale, position_bind, scale_bind]

