# use blender内置fbx插件 as a reference
# # https://github.com/mrdoob/three.js/blob/dev/examples/js/loaders/FBXLoader.js
from . import fbxsdk


class 类(fbxsdk.fbxfile.fbxfile):
    def __init__(self, bp): 
        super().__init__(), self.frombp(bp)
        self.uuid_dict = {}
        self.update_uuid_dict()
        self.OO_OP_link_OO_OP()
        self.Model_Handle_Armature()
        self.Model_Handle_Action()


    def update_uuid_dict(self):
        for o in self.Objects:
            if "uuid" not in o.__dict__: continue
            o.parent = None
            o.children = []
            self.uuid_dict[o.uuid] = o


    def OO_OP_link_OO_OP(self):
        for c in self.Connections:
            if c.uuid_child not in self.uuid_dict: continue
            if c.uuid_parent not in self.uuid_dict: continue
            ochild, oparent = self.uuid_dict[c.uuid_child], self.uuid_dict[c.uuid_parent]
            if ochild not in oparent.children: 
                oparent.children.append(ochild)
                ochild.parent = oparent
            

    def Model_Handle_Armature(self):
        self.bone_dict = {}
        for uuid, o in self.uuid_dict.items():
            # if o.type == "LimbNode": self.bone_dict[o.name] = o
            if o.classname == "Model": self.bone_dict[o.name] = o

        for name, o in self.bone_dict.items():
            o.translation, o.rotation, o.scale = [], [], []  # 警告：此处只能为[], 填入任何初始数值，会造成后续矩阵生成出现错误！！！,尤其是填入0，后续矩阵乘法，所有数值都失效，0乘以任何数都会变成0！！！
            for p in o.Properties70:
                if p.name == "Lcl Translation": o.translation = p.value
                if p.name == "Lcl Rotation": o.rotation = p.value
                if p.name == "Lcl Scaling": o.scale = p.value


    def Model_Handle_Action(self):
        # Model -> AnimCurveNode -> AnimCurve
        for name, o in self.bone_dict.items():
            o.childnode_translation, o.childnode_rotation, o.childnode_scale = None, None, None
            for c in o.children:
                match c.type:
                    case "TAnimCurveNode": o.childnode_translation = c
                    case "RAnimCurveNode": o.childnode_rotation = c
                    case "SAnimCurveNode": o.childnode_scale = c

        for name, o in self.bone_dict.items():
            o.action_translation, o.action_rotation, o.action_scale = {}, {}, {}

            if o.childnode_translation != None and o.childnode_translation.children != []:
                vx_times, vx_values = o.childnode_translation.children[0].KeyTime, o.childnode_translation.children[0].KeyValueFloat
                vy_times, vy_values = o.childnode_translation.children[1].KeyTime, o.childnode_translation.children[1].KeyValueFloat
                vz_times, vz_values = o.childnode_translation.children[2].KeyTime, o.childnode_translation.children[2].KeyValueFloat
                for vx_time, vy_time, vz_time, vx_value, vy_value, vz_value in zip(vx_times, vy_times, vz_times, vx_values, vy_values, vz_values):
                    vx_time = round(vx_time/46186158000*60)
                    o.action_translation[vx_time] = [vx_value, vy_value, vz_value]

            if o.childnode_rotation != None and o.childnode_rotation.children != []:
                rx_times, rx_values = o.childnode_rotation.children[0].KeyTime, o.childnode_rotation.children[0].KeyValueFloat
                ry_times, ry_values = o.childnode_rotation.children[1].KeyTime, o.childnode_rotation.children[1].KeyValueFloat
                rz_times, rz_values = o.childnode_rotation.children[2].KeyTime, o.childnode_rotation.children[2].KeyValueFloat
                for rx_time, ry_time, rz_time, rx_value, ry_value, rz_value in zip(rx_times, ry_times, rz_times, rx_values, ry_values, rz_values):
                    rx_time = round(rx_time/46186158000*60)
                    o.action_rotation[rx_time] = [rx_value, ry_value, rz_value]

            if o.childnode_scale != None and o.childnode_scale.children != []:
                sx_times, sx_values = o.childnode_scale.children[0].KeyTime, o.childnode_scale.children[0].KeyValueFloat
                sy_times, sy_values = o.childnode_scale.children[1].KeyTime, o.childnode_scale.children[1].KeyValueFloat
                sz_times, sz_values = o.childnode_scale.children[2].KeyTime, o.childnode_scale.children[2].KeyValueFloat
                for sx_time, sy_time, sz_time, sx_value, sy_value, sz_value in zip(sx_times, sy_times, sz_times, sx_values, sy_values, sz_values):
                    sx_time = round(sx_time/46186158000*60)
                    o.action_scale[sx_time] = [sx_value, sy_value, sz_value]


    # def __local__translation__(self, t_anim_curve_node):
    #     result = [0.0, 0.0, 0.0]
    #     for p in t_anim_curve_node.Properties70:
    #         match p.name:
    #             case "d|X": result[0] = p.value
    #             case "d|Y": result[1] = p.value
    #             case "d|Z": result[2] = p.value
    #     return result


    # def __local__rotation__(self, r_anim_curve_node):
    #     angles = [0.0, 0.0, 0.0]
    #     for p in r_anim_curve_node.Properties70:
    #         match p.name:
    #             case "d|X": angles[0] = p.value
    #             case "d|Y": angles[1] = p.value
    #             case "d|Z": angles[2] = p.value
    #     return angles
    

    # def __local__scale__(self, s_anim_curve_node):
    #     result = [0.0, 0.0, 0.0]
    #     for p in s_anim_curve_node.Properties70:
    #         match p.name:
    #             case "d|X": result[0] = p.value
    #             case "d|Y": result[1] = p.value
    #             case "d|Z": result[2] = p.value
    #     return result


# # https://github.com/mrdoob/three.js/blob/dev/examples/js/loaders/FBXLoader.js
# # // Converts FBX ticks into real time seconds.
# #     function convertFBXTimeToSeconds( time ) {
# #         return time / 46186158000;
# #     }

# <Matrix 4x4 (0.0000, 0.0000, 0.0000, 0.0000)
#             (0.0000, 0.0000, 0.0000, 0.0000)
#             (0.0000, 0.0000, 0.0000, 0.0000)
#             (0.0000, 0.0000, 0.0000, 1.0000)>
# @ 任何矩阵都变为零，所以为[]就好，不要设[0,0,0]