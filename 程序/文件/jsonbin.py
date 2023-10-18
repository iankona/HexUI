
# json 解析存在递归深度问题
# import sys
# sys.setrecursionlimit(3600) # 设置递归深度





class 类:
    def __init__(self, bp): 
        self.filepath = bp.rdpath
        self.type_result_list = []
        self.block_iteration_file(bp)




    def block_iteration_file(self, bp):
        head = bp.readslice(28)
        while True:
            if bp.remainsize() < 16: break
            datatype, result = self.__json__type__(bp)
            self.type_result_list.append([datatype, result])


    # def __json__node__(self, bp):
    #     node_result = []
    #     num = bp.readuint8()
    #     node_result.append(["", num])
    #     node_result.append(["", bp.readuint8(3)])
    #     if num > 0:
    #         for i in range(num):
    #             datatype, result = self.__json__type__(bp)
    #             node_result.append([datatype, result])


    #     # while True:
    #     #     if bp.remainsize() < 16: break
    #     #     datatype1, result1 = self.__json__type__(bp)
    #     #     node_result.append([datatype1, result1])
    #     #     if datatype1 == 8:
    #     #         datatype2, result2 = self.__json__type__(bp)
    #     #         node_result.append([datatype2, result2])
    #     #         if datatype2 == 8: self.key_value_list.append([result1, result2])

    #     #     datatype3 = bp.readuint8seek0()
    #     #     if datatype3 == 11: break
    #     #     if datatype3 >  13: break
    #     return node_result

    def __json__type__(self, bp):
        tell = bp.tell()
        datatype = bp.readuint8()
        match datatype:
            case  0: result = bp.readslice(0)         
            case  1: result = bp.readslice(0) 
            case  2: result = bp.readslice(0)            
            case  3: result = bp.readslice(1)
            case  4: result = bp.readslice(2)  
            case  5: result = bp.readslice(4)
            case  6: result = bp.readslice(8)     
            case  7: result = bp.readslice(8)
            case  8: result = bp.readutf8(bp.readuint8())    
            case  9: result = bp.readutf8(bp.readuint16())  
            case 10: result = None   
            case 11: result = bp.readslice(4)
            case 12: result = None    
            case 13: result = bp.readslice(0)
            case 14: result = None    
            case 15: result = bp.readslice(2)
            case 16: result = bp.readslice(4)
        try:
            if result == None: print(datatype, bp.readuint8seek0(8), tell)
        except:
            print(datatype, bp.readuint8seek0(8), tell)
        return datatype, result
    
    def __json__float__(self, bp):
        datatype, typefloat = self.__json__type__(bp) # name
        return datatype, typefloat