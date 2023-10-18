# 逻辑
选择集列表 = []


# 界面

根界面 = None

frame面板 = None

frame节点 = None
frame表格 = None
frame文本 = None

节点itemframe = None
表格itemframe = None




# 节点选项


# 表格选项
弹窗开闭状态 = False
弹窗停留时间戳 = 0


# 常量



格式步长字典 = {
    "bin":1, 
    "bin8":1,        
    "hex":1,
    "gbk":1,    
    "char":1,
    "utf8":1, # 可变字节，步长只能为1，为其他步长，会造成少读字符
    "int8": 1,
    "int16": 2,
    "int32": 4,
    "int64": 8,
    "uint8": 1,
    "uint16": 2,
    "uint32": 4,
    "uint64": 8,
    # "5u8uint64": 5, 
    "float16": 2,
    "float32": 4,
    "float64": 8, 
    "i8float32": 4,
    "u8float32": 4,
    "i16float32": 4,
    "u16float32": 4,          
    }


格式宽度字典 = {
    "bin":80, 
    "bin8":80,        
    "hex":40,
    "gbk":40,    
    "char":40,
    "utf8":40,
    "int8": 40,
    "uint8": 40,

    "int16": 60,
    "uint16": 60,

    "int32": 80,
    "uint32": 80,

    "int64": 120,
    "uint64": 120,
    # "5u8uint64": 120,

    "float16": 100,
    "float32": 100,
    "float64": 100, 
    "i8float32": 100,
    "u8float32": 100,
    "i16float32": 100,
    "u16float32": 100,                
    }



解析格式列表 = [
    "bin",  
    "bin8",        
    "hex",
    "gbk",
    "char",
    "utf8",
    "hex+char",
    "hex+uint8",
    "uint8+char",
    "bin8+gbk",    
    "bin8+utf8",   
    "uint8+gbk",    
    "uint8+utf8",  
    "uint8+bin8+gbk",    
    "uint8+bin8+utf8",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "int8",
    "int16",
    "int32",
    "int64",
    "float16",
    "float32",
    "float64",
    "uint8+uint16",
    "uint8+uint32",    
    "i8float32",
    "u8float32",
    "i16float32",
    "u16float32",
    "int8+i8float32",
    "uint8+u8float32",
    "int16+i16float32",
    "uint16+u16float32",
    # "5u8uint64",
    # "uint8+5u8uint64",    
    ]


