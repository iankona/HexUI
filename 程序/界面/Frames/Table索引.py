# 专门处理 table_yview() 和 ybar_set() 逻辑


显示上行 = 0
显示下行 = 32
def ybar_set(frametable, itemframe, *args, **kwargs):
    # print("表格输出参数：", "args", args, "kwargs", kwargs)
    # 表格输出参数： args ('0.0', '0.1796875') kwargs {}
    global 显示上行, 显示下行
    表格行数 = len(frametable.table.get_children(""))
    显示左值, 显示右值 = float(args[0]), float(args[1])
    显示上行, 显示下行 = round(显示左值/1.0*表格行数), round(显示右值/1.0*表格行数)

    # 显示行数 = 显示下行 - 显示上行
    # itemframe.表格行数["text"] = 表格行数
    # itemframe.显示行数["text"] = 显示行数
    # itemframe.显示限值["text"] = f"{args[0]}，{args[1]}"
    # itemframe.显示限行["text"] = f"[{显示上行}，{显示下行})"

    # 格式文本 = itemframe.解析格式.get()
    # 格式列表 = 格式文本.split("+")
    # 格式数量 = len(格式列表)

    # 数据列表 = 缓存.选择集列表
    # 数据数量 = len(数据列表)
    # itemframe.数据列表["values"] = 数据列表

    # 折算行数 = 数据数量*格式数量
    # 等效行数 = round(显示行数/格式数量)
    # 等效上行, 等效下行 = round(显示上行/()), round(显示下行/(数据数量*格式数量))
    # itemframe.等效行数["text"] = 等效行数
    # itemframe.等效限值["text"] = f"未计算"
    # itemframe.等效限行["text"] = f"[{等效上行}，{等效下行})"


列表索引 = 0
列数索引 = 0
def table_yview(frametable, itemframe, *args, **kwargs):
    # print("滚条输出参数：", "args", args, "kwargs", kwargs)
    # 滚条输出参数： args ('moveto', '0.23308270676691728') kwargs {}
    # 滚条输出参数： args ('scroll', '±2', 'pages') kwargs {} 
    # 滚条输出参数： args ('scroll', '±4', 'units') kwargs {} # 在 滚动条区域向上向下滚动 和 点击上下三角形 输出的参数一致
    match args[0]:
        case "scroll": scroll(int(args[1]), frametable)
        case "moveto": moveto(float(args[1]), frametable)


分段列表 = []
有效行数 = 128
def moveto(value, frametable):
    print(value)
    global 列表索引, 列数索引
    if value < 0: value = 0.0
    if value > 1: value = 1.0
    滑块起点 = value
    # 列表位置
    分段数量 = len(分段列表)
    分段长度 = 1.0/分段数量
    列表索引 = int(value/分段长度)
    if 列表索引 >= 分段数量: 列表索引 = 分段数量-1

    # 列数位置
    value = value - 列表索引*分段长度
    列数数量 = len(分段列表[列表索引])
    列段长度 = 分段长度/列数数量
    列数索引 = int(value/列段长度)
    if 列数索引 >= (列数数量-有效行数): 列数索引 = 列数数量-有效行数 # -1 +1 对应 分段列表[列表索引][列数索引: 列数索引+有效行数]
    if 列数索引 < 0: 列数索引 = 0

    # print("moveto", 列表索引, 列数索引)
    滑块长度 = min(有效行数, 列数数量) * 列段长度
    if 滑块起点 > 1.0-滑块长度: 滑块起点 = 1.0-滑块长度
    frametable.ybar.set(str(滑块起点), str(滑块起点+滑块长度))
    # print(列表索引, 列数索引, 滑块起点, 滑块起点+滑块长度)


def scroll(value, frametable):
    global 列表索引, 列数索引

    分段数量 = len(分段列表)
    if 分段数量 == 1:
        上段数量 = 0
        本段数量 = len(分段列表[列表索引])
        下段数量 = 0
    if 分段数量 == 2 and 列表索引 == 0:
        上段数量 = 0
        本段数量 = len(分段列表[列表索引])
        下段数量 = len(分段列表[列表索引+1])
    if 分段数量 == 2 and 列表索引 == 1:
        上段数量 = len(分段列表[列表索引-1])
        本段数量 = len(分段列表[列表索引])
        下段数量 = 0
    if 分段数量 >= 3 and 列表索引 == 0:
        上段数量 = 0
        本段数量 = len(分段列表[列表索引])
        下段数量 = len(分段列表[列表索引+1])
    if 分段数量 >= 3 and 列表索引 >= 1 and 列表索引 < 分段数量-1:
        上段数量 = len(分段列表[列表索引-1])
        本段数量 = len(分段列表[列表索引])
        下段数量 = len(分段列表[列表索引+1])
    if 分段数量 >= 3 and 列表索引 == 分段数量-1:
        上段数量 = len(分段列表[列表索引-1])
        本段数量 = len(分段列表[列表索引])
        下段数量 = 0

    # print("------------------------------")
    # print("本段数量, 列表索引, 列数索引：", 本段数量, 列表索引, 列数索引)
    上次索引 = 列数索引
    当前索引 = 列数索引
    当前索引 += value*有效行数
    列数索引 = 当前索引
    if 当前索引 < 0 and 上段数量 == 0: 列数索引 = 0
    if 当前索引 < 0 and 上段数量 >  0: 
        列表索引 -= 1
        列数索引 = 上段数量 + 当前索引
        if 列数索引 < 0: 列数索引 = 0
             
    if 当前索引 > (本段数量-1) and 下段数量 == 0: 
        列数索引 = 上次索引 
    if 当前索引 > (本段数量-1) and 下段数量 > 0: 
        列表索引 += 1
        列数索引 = 0

    # print("scroll", 列表索引, 列数索引)
    分段长度 = 1.0/分段数量
    列数数量 = len(分段列表[列表索引])
    列段长度 = 分段长度/列数数量
    if 列数数量 > 有效行数:
        滑块长度 =  有效行数 * 列段长度
    else:
        滑块长度 =  列数数量 * 列段长度
    段位置值 = 列表索引*分段长度 + 列数索引*列段长度
    frametable.ybar.set(str(段位置值), str(段位置值+滑块长度))

    # print("本段数量, 列表索引, 列数索引：", 本段数量, 列表索引, 列数索引)
# def 表格区域与滚条滑块复位(self, frametable):
#     frametable.ybar.set(str(0.0), str(0.1))
#     frametable.table.yview("moveto", "0.0")

rowindex = ""
三级索引 = 0
def 刷新表列索引(frametable, index, column):
    global rowindex, 三级索引
    rowindex = index
    match column:
        case "#0": 三级索引 = 0
        case "": 三级索引 = 0
        case _: 三级索引 = int(column[1:])-1





