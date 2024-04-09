
# json 解析存在递归深度问题
# import sys
# sys.setrecursionlimit(3600) # 设置递归深度





class 类:
    def __init__(self, bp): 
        self.filepath = bp.rdpath
        self.result = None
        self.block_iteration_file(bp)


    def block_iteration_file(self, bp):
        head = bp.readslice(28)
        datatype, result = self.__json__type__(bp)
        self.result = self.__json__type__11__(datatype, result, bp)


    def __json__type__(self, bp):
        tell = bp.tell()
        datatype = bp.readuint8()
        match datatype:
            case  0: result = datatype       
            case  1: result = datatype 
            case  2: result = datatype            
            case  3: result = bp.readuint8() 
            case  4: result = bp.readuint16()   
            case  5: result = bp.readuint32() 
            case  6: result = bp.readslice(8)     
            case  7: result = bp.readslice(8)
            case  8: result = bp.readutf8(bp.readuint8())    
            case  9: result = bp.readutf8(bp.readuint16())  
            case 10: result = None   
            case 11: result = bp.readuint32()
            case 12: result = None    
            case 13: result = datatype
            case 14: result = None    
            case 15: result = bp.readslice(2) 
            case 16: result = bp.readslice(4) 
        try:
            if result == None: print(datatype, bp.readuint8seek0(8), tell)
        except:
            print(datatype, bp.readuint8seek0(8), tell)

        return datatype, result
    

    def __json__type__11__(self, datatype, numenum, bp):
        result_list = []
        for i in range(numenum):
            datatype0, result0 = self.__json__type__(bp)
            if datatype0 == 11: result0 = self.__json__type__11__(datatype0, result0, bp)
            result_list.append(result0)

        result_dict = {}

        while True:
            if bp.remainsize() < 2: break
            datatype0, result0 = self.__json__type__(bp)
            if datatype0 == 0: break
            datatype1, result1 = self.__json__type__(bp)
            if datatype1 == 11: result1 = self.__json__type__11__(datatype1, result1, bp)
            result_dict[result0] = result1

        if result_list != [] and result_dict == {}: return result_list
        if result_list == [] and result_dict != {}: return result_dict
        result = {}
        for i, value in enumerate(result_list): result[f"_{i}"] = value
        for key, value in result_dict.items(): result[key] = value
        return result


    # def __json__slice__(self, bp):
    #     tell = bp.tell()
    #     datatype = bp.readuint8()
    #     match datatype:
    #         case  0: result = bp.readslice(0)       
    #         case  1: result = bp.readslice(0) 
    #         case  2: result = bp.readslice(0)            
    #         case  3: result = bp.readslice(1)
    #         case  4: result = bp.readslice(2)  
    #         case  5: result = bp.readslice(4)
    #         case  6: result = bp.readslice(8)     
    #         case  7: result = bp.readslice(8)
    #         case  8: result = bp.readutf8(bp.readuint8())    
    #         case  9: result = bp.readutf8(bp.readuint16())  
    #         case 10: result = None   
    #         case 11: result = bp.readslice(4)
    #         case 12: result = None    
    #         case 13: result = bp.readslice(0)
    #         case 14: result = None    
    #         case 15: result = bp.readslice(2)
    #         case 16: result = bp.readslice(4)
    #     try:
    #         if result == None: print(datatype, bp.readuint8seek0(8), tell)
    #     except:
    #         print(datatype, bp.readuint8seek0(8), tell)
    #     return datatype, result