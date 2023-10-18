
索引列表 = []
显示列表 = []


def 生成数据显示列表(数据列表, 格式列表, 字节偏移值, 单元格字节数, 倍数列表, 范围列数列表):
    global 显示列表, 索引列表

    显示列表 = []
    索引列表 = []

    for bp in 数据列表: 
        bp.index = bp.mpleft
        bp.seek(字节偏移值)
        # bp.changeleft(bp.mpleft+字节偏移值)

    for 列数 in 范围列数列表:
        for 格式, 倍数 in zip(格式列表, 倍数列表):
            for bp in 数据列表: 
                结果列表 = 读取格式seek0(bp, 格式, 倍数, 列数)
                显示列表.append(结果列表)
                索引列表.append([bp.tell(), bp.slicetell()])

        for bp in 数据列表: bp.seek(单元格字节数*列数)

    return 显示列表

def 读取格式seek0(bp, 格式, 倍数, 列数):
    # tkinter 会自动处理 [[], [], [], [], [], , ]
    列表 = []
    match 格式:
        case "bin": 列表 = bp.readbinseek0(倍数*列数)            
        case "hex": 列表 = bp.readhexseek0(倍数*列数)      
        case "char": 列表 = bp.readcharseek0(倍数*列数)

        case "int8": 列表 = bp.readint8seek0(倍数*列数)
        case "int16": 列表 = bp.readint16seek0(倍数*列数)   
        case "int32": 列表 = bp.readint32seek0(倍数*列数) 
        case "int64": 列表 = bp.readint64seek0(倍数*列数) 

        case "uint8": 列表 = bp.readuint8seek0(倍数*列数)
        case "uint16": 列表 = bp.readuint16seek0(倍数*列数)
        case "uint32": 列表 = bp.readuint32seek0(倍数*列数)
        case "uint64": 列表 = bp.readuint64seek0(倍数*列数)

        case "float16": 列表 = bp.readfloat16seek0(倍数*列数)
        case "float32": 列表 = bp.readfloat32seek0(倍数*列数)
        case "float64": 列表 = bp.readfloat64seek0(倍数*列数)
        case "i8float32": 列表 = bp.readi8float32seek0(倍数*列数)
        case "u8float32": 列表 = bp.readu8float32seek0(倍数*列数)
        case "i16float32": 列表 = bp.readi16float32seek0(倍数*列数)
        case "u16float32": 列表 = bp.readu16float32seek0(倍数*列数)
        
    结果列表 = []
    if 倍数*列数 == 1: 列表 = [列表]
    if 格式 == "bin8": 列表 = bpformat_readbin8seek0(bp, 倍数, 列数)
    for i in range(0, len(列表), 倍数): 结果列表.append(列表[i:i+倍数])
    if 格式 == "gbk": 结果列表 = bpformat_readgbkseek0(bp, 倍数, 列数)    
    if 格式 == "utf8": 结果列表 = bpformat_readutf8seek0(bp, 倍数, 列数)
    return 结果列表


def bpformat_readbin8seek0(bp, 倍数, 列数):
    列表 = []
    列表 = bp.readbinseek0(倍数*列数)
    if 倍数*列数 == 1: 列表 = [列表]  
    binchar8列表 = []
    for binchar in 列表:
        前缀, 后列表 = binchar[0:2], binchar[2:]
        前零 = ""
        for i in range(8-len(后列表)): 前零 += "0"
        binchar8列表.append(前缀+前零+后列表)
    return binchar8列表


def bpformat_readgbkseek0(bp, 倍数, 列数): #
    # bin(bytebin) # TypeError: 'bytes' object cannot be interpreted as an integer
    binchar8列表 = bpformat_readbin8seek0(bp, 倍数, 列数)
    
    # 结果列表 = []
    # for i in range(0, len(binchar8列表), 倍数): 结果列表.append(binchar8列表[i:i+倍数])

    描述列表 = []
    跳位 = 0
    for i in range(len(binchar8列表)):
        if i < 跳位: continue
        length = len(binchar8列表[i:])
        if length >= 2:
            if binchar8列表[i].startswith("0b1") and binchar8列表[i+1].startswith("0b1"):
                跳位 += 2
                描述列表.append([i, 2])
                continue
        跳位 += 1
        描述列表.append([i, 1])

    结果列表 = []
    bx = bp.readsliceseek0(倍数*列数)
    for index, size in 描述列表:
        if size == 1: 
            char = chr(bx.readuint8())
            结果列表.append(char)
        else:
            结果列表.append(bx.readgbk(size))
            结果列表.append("")

    return 结果列表


def bpformat_readutf8seek0(bp, 倍数, 列数):
    # bin(bytebin) # TypeError: 'bytes' object cannot be interpreted as an integer
    binchar8列表 = bpformat_readbin8seek0(bp, 倍数, 列数)
    
    # 结果列表 = []
    # for i in range(0, len(binchar8列表), 倍数): 结果列表.append(binchar8列表[i:i+倍数])

    描述列表 = []
    跳位 = 0
    for i in range(len(binchar8列表)):
        if i < 跳位: continue
        length = len(binchar8列表[i:])
        if length >= 2:
            if binchar8列表[i].startswith("0b110") and binchar8列表[i+1].startswith("0b10"):
                跳位 += 2
                描述列表.append([i, 2])
                continue
        if length >= 3:
            if binchar8列表[i].startswith("0b1110") and binchar8列表[i+1].startswith("0b10") and binchar8列表[i+2].startswith("0b10"):
                跳位 += 3
                描述列表.append([i, 3])
                continue
        if length >= 4:
            if binchar8列表[i].startswith("0b11110") and binchar8列表[i+1].startswith("0b10") and binchar8列表[i+2].startswith("0b10") and binchar8列表[i+3].startswith("0b10"):
                跳位 += 4
                描述列表.append([i, 4])
                continue
        跳位 += 1
        描述列表.append([i, 1])

    结果列表 = []
    bx = bp.readsliceseek0(倍数*列数)
    for index, size in 描述列表:
        if size == 1: 
            char = chr(bx.readuint8())
            结果列表.append(char)
        else:
            结果列表.append(bx.readutf8(size))
            for i in range(size-1): 结果列表.append("")

    return 结果列表


def 表格显示刷新处理(frametable):
    tableitems = frametable.table.get_children("")
    numitem, numdata = len(tableitems), len(显示列表)
    if numitem >= numdata:
        for item, values in zip(tableitems, 显示列表): frametable.table.item(item, values=values)
        for item in tableitems[numdata:]: frametable.table.item(item, text="", values=[])
        # for item in tableitems[numdata:][::-1]: frametable.table.delete(item)
    else:
        for item, values in zip(tableitems, 显示列表): frametable.table.item(item, values=values)
        for i, values in enumerate(显示列表[numitem:]) : frametable.table.insert("", index="end", text=str(numitem+i), values=values) 


def 计算列数最大值与重设列(frametable, 范围列数列表, 格式列表, 格式宽度字典):
    列数最大值 = 0
    for 列数 in 范围列数列表:
        if 列数 > 列数最大值: 列数最大值 = 列数

    宽度最大值 = 0
    for 格式 in 格式列表:
        宽度 = 格式宽度字典[格式]
        if 宽度 > 宽度最大值: 宽度最大值 = 宽度


    字符列表 = [f"#{i+1}" for i in range(列数最大值)]
    frametable.table["columns"] = 字符列表

    frametable.table.column("#0", width=240, stretch=0)
    for column, columnstitle in zip(字符列表, 字符列表):
        frametable.table.column(column, width=宽度最大值, stretch=False, anchor='center')
        frametable.table.heading(column, text=columnstitle)


def 表格左侧栏显示处理(frametable, 数据列表, 格式列表):
    tableitems = frametable.table.get_children("")
    数据数量, 格式数量 = len(数据列表), len(格式列表)
    折算数量 = 数据数量 * 格式数量
    if "char" in 格式列表:
        index = 格式列表.index("char")
        更新左侧栏(frametable, tableitems, 数据数量, 折算数量, index)
    if "gbk" in 格式列表:
        index = 格式列表.index("gbk")
        更新左侧栏(frametable, tableitems, 数据数量, 折算数量, index)
    if "utf8" in 格式列表:
        index = 格式列表.index("utf8")
        更新左侧栏(frametable, tableitems, 数据数量, 折算数量, index)
    更新偏移值(frametable, tableitems, 数据数量, 折算数量, 0)

def 更新左侧栏(frametable, tableitems, 数据数量, 折算数量, index):
    for i, [item, values] in enumerate(zip(tableitems, 显示列表)):   
        m = i  % 折算数量
        n = m // 数据数量
        if n == index : 
            chars = ""
            for 列表 in values:
                for char in 列表: chars += char
            frametable.table.item(item, text=chars)

def 更新偏移值(frametable, tableitems, 数据数量, 折算数量, index):
    for i, [item, values, [offset, slicetell]] in enumerate(zip(tableitems, 显示列表, 索引列表)):   
        m = i  % 折算数量
        n = m // 数据数量
        if n == index : frametable.table.item(item, text=f"0d{offset}_0d{slicetell}_")
        # if n == index : frametable.table.item(item, text=f"0d{slicetell}")