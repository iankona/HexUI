import tkinter as tk
import tkinter.ttk as ttk
import tkinter.filedialog


from . import Tree格式Memu

class TreeRightMemu:
    def __init__(self, frametree, 菜单):
        # self.memublock = Tree格式Memu.block2tkinter.类(frametree, 菜单)
        self.memufbx = Tree格式Memu.fbx2tkinter.类(frametree, 菜单)
        # self.memumodel = Tree格式Memu.model2tkinter.类(frametree, 菜单)
        self.memuhka = Tree格式Memu.hka2tkinter.类(frametree, 菜单)
        self.memuhkx = Tree格式Memu.hkx2tkinter.类(frametree, 菜单)

        # self.memunavmesh = Tree格式Memu.navmesh2tkinter.类(frametree, 菜单)
        # self.memuckpt = Tree格式Memu.checkpoints2tkinter.类(frametree, 菜单)
        self.memusrt = Tree格式Memu.speedtree2tkinter.类(frametree, 菜单)

        self.memubin = Tree格式Memu.zhulongbin2tkinter.类(frametree, 菜单)


# import mmap
# import os

# # 测试文件
# filename = "test.txt"

# # 首先打开文件
# file = open(filename, "r+")
# # 获得文件大小，我们将整个文件都映射到 mmap
# size = os.path.getsize(filename)

# # 创建mmap 对象
# data = mmap.mmap(file.fileno(), size)

# # 打印整个文件，注意它是 byte 类型，需要 decode
# print(data[:].decode("utf-8"))

# # 修改 'hello' 为 ‘12345’
# data[:5] = b'12345'

# # 关闭 mmap 对象
# data.close()
# # 关闭文件
# file.close()