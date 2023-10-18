import os
dirpath = os.path.dirname(__file__)
import sys
sys.path.append(dirpath+"\\ttks")
sys.path.append(dirpath+"\\libs")
sys.path.append(dirpath+"\\程序")

# print(sys.path)

import 界面
界面.根界面.运行()


# import mathutils
# print(mathutils.Quaternion())