import tkinter as tk
import tkinter.ttk as ttk
import tkinter.filedialog

import 界面文件.bcontext.bsfunction as bs
import 底层文件.bpformat.byfunction as by
import 底层文件

class 类:
    def __init__(self, frametreeview, 菜单):
        self.菜单 = 菜单
        self.frametreeview = frametreeview

        self.添加古剑1菜单(label="古剑1/Gamebryo")
        self.添加古剑2EmotionFX菜单(label="古剑2/EmotionFX")
        self.添加古剑2HavokAnarchy菜单(label="古剑2/HavokAnarchy")
        self.添加古剑2SpeedTree菜单(label="古剑2/SpeedTree")
        self.添加古剑3HavokAnarchy菜单(label="古剑3/HavokAnarchy")
        self.添加古剑3Havok菜单(label="古剑3/Havok")
        self.添加古剑3SpeedTree菜单(label="古剑3/SpeedTree")
        self.添加古剑3JsonBin菜单(label="古剑3/JsonBin")
        self.添加fbxsdk菜单("FBX SDK")
        self.添加checkpoints菜单("Ckpt")
        self.添加Unity菜单("Unity/AssertsBundle")


    @staticmethod
    def 菜单装饰函数(function):
        def 菜单生成函数(self, label=""):
            菜单 = tk.Menu(None, tearoff=False)
            self.菜单.add_cascade(label=label, menu=菜单)
            function(self, 菜单)
        return 菜单生成函数

    @菜单装饰函数
    def 添加古剑1菜单(self, 菜单): 
        菜单.add_command(label=".nif_文件解析", command=None)
        菜单.add_command(label=".kf_文件解析", command=None)
        菜单.add_command(label="Gamebryo_文件分块", command=None)

    @菜单装饰函数
    def 添加古剑2EmotionFX菜单(self, 菜单): 
        菜单.add_command(label=".xac_文件解析", command=None)
        菜单.add_command(label=".xsm_文件解析", command=None)
        菜单.add_command(label="EmotionFX_文件分块", command=None)

    @菜单装饰函数
    def 添加古剑2HavokAnarchy菜单(self, 菜单): self.__HavokAnarchy菜单__(菜单)

    @菜单装饰函数
    def 添加古剑2SpeedTree菜单(self, 菜单):  self.__SpeedTree菜单__(菜单)

    @菜单装饰函数
    def 添加古剑3HavokAnarchy菜单(self, 菜单):  self.__HavokAnarchy菜单__(菜单)

    @菜单装饰函数
    def 添加古剑3Havok菜单(self, 菜单): 
        菜单.add_command(label="HKA文件解析", command=None)
        菜单.add_command(label="HKX文件解析", command=None)
        菜单.add_command(label="HKS文件解析", command=None)
        菜单.add_command(label="HKT文件解析", command=None)

    @菜单装饰函数
    def 添加古剑3SpeedTree菜单(self, 菜单):  self.__SpeedTree菜单__(菜单)

    @菜单装饰函数
    def 添加古剑3JsonBin菜单(self, 菜单):
        菜单.add_command(label=".bin_文件解析", command=None)        

    @菜单装饰函数
    def 添加checkpoints菜单(self, 菜单):
        pass

 
    @菜单装饰函数
    def 添加fbxsdk菜单(self, 菜单):
        菜单.add_command(label="文件解析", command=None)
        菜单.add_command(label="文件分块", command=None)
        菜单.add_command(label="单独节点分块", command=None)
        菜单.add_command(label="递归节点分块", command=None)


    @菜单装饰函数
    def 添加Unity菜单(self, 菜单):
        from . import tkinter_unity_asserts
        from . import tkinter_unity_unity3d
        from . import tkinter_unity_unity3d_asserts
        from . import tkinter_unity_wmv
        asserts = tkinter_unity_asserts.类(self.frametreeview)
        unity3d = tkinter_unity_unity3d.类(self.frametreeview)
        unity3d_asserts = tkinter_unity_unity3d_asserts.类(self.frametreeview)
        wmv = tkinter_unity_wmv.类(self.frametreeview)

        菜单.add_command(label=".asserts_文件分块", command=asserts.asserts_文件分块_wrapper)
        菜单.add_command(label=".unity3d_文件分块", command=unity3d.unity3d_文件分块_wrapper)
        菜单.add_command(label=".unity3d_asserts_文件分块", command=unity3d_asserts.unity3d_asserts_文件分块_wrapper)
        菜单.add_command(label=".wmv_Find_UnityFS", command=wmv.Find_UnityFS_wapper)





    def __HavokAnarchy菜单__(self, 菜单): 
        from . import tkinter_anarchy_model
        model = tkinter_anarchy_model.类(self.frametreeview)
        菜单.add_command(label="Anarchy_文件分块", command=model.文件分块_wrapper)
        菜单.add_command(label="HSMV", command=model.HSMV_wrapper)
        菜单.add_command(label="MOEG", command=model.MOEG_wrapper)
        菜单.add_command(label="MESH", command=model.MESH_wrapper)


    def __SpeedTree菜单__(self, 菜单):
        菜单.add_command(label=".srt_文件解析", command=None) 
        菜单.add_command(label="SRT 05.0.4_文件分块", command=None) 
        菜单.add_command(label="SRT 07.0.0_文件分块", command=None) 


# def context(self, item="", bp=None):
#     if item == "": item = self.frametreeview.treeview.selection()[0]
#     by.context(bp)
#     bs.context(item, self.frametreeview)
#     return item

# @staticmethod
# def 执行函数装饰函数(function):
#     def 执行函数(self):
#         for item in self.frametreeview.treeview.selection():
#             bp = self.frametreeview.bpdict[item].copy()
#             self.context(item, bp)
#             function(self)
#     return 执行函数