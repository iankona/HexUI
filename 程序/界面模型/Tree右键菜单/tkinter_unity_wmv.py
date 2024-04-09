import tkinter as tk
import tkinter.ttk as ttk
import tkinter.filedialog

import 界面.bcontext.bsfunction as bs
import 文件.bpformat.byfunction as by
import 文件

import lz4.block

class CompressionType:
    NONE = 0
    LZMA = 1
    LZ4 = 2
    LZ4HC = 3
    LZHAM = 4

### https://github.com/HearthSim/UnityPack


class 类:
    def __init__(self, frametreeview):
        self.frametreeview = frametreeview
        self.Find_UnityFS_wapper = self.wrapper(self.Find_UnityFS)


    def wrapper(self, function=None):
        def 执行函数(): # *args, **kwargs == (), {}
            for item in self.frametreeview.treeview.selection():
                bp = self.frametreeview.bpdict[item].copy()
                by.context(bp)
                bs.context(item, self.frametreeview)
                function() # function(self) # TypeError: 类.Find_UnityFS() takes 1 positional argument but 2 were given
        return 执行函数
        


















    def UnityFS文件分块(self):
        bs.insertvalue(text=f"是大端")
        by.endian(">")
        bs.insertblock(text=f"Head_UnityFS_8", bp=by.readslice(8).bp) # 'UnityFS' end0
        bs.insertblock(text=f"Head_FileVersion_4", bp=by.readslice(4).bp) # 6 int32 or uint32
        bs.insertblock(text=f"Head_UnityVersion_6", bp=by.readslice(6).bp) # '5.x.x'
        bs.insertblock(text=f"Head_GeneratorVersion_8", bp=by.readslice(8).bp) # '5.6.4p4'
        bs.insertblock(text=f"Head_filesize_{by.readuint64seek0()}", bp=by.readslice(8).bp) # 30967414
        [解压前大小, 解压后大小] = by.readuint32seek0(2)
        bs.insertblock(text=f"Head_compsize_{[解压前大小, 解压后大小]}", bp=by.readslice(8).bp) # 2133, 4761
        flag = by.readuint32seek0()
        bs.insertblock(text=f"Head_compflag_{flag}", bp=by.readslice(4).bp) # # 67
        # 67    0b 01000011 # flag
        # 0x3F  0b 00111111 # compression
        # 0x80  0b 10000000 # eof_metadata # if True buf.seek(-self.ciblock_size, 2)
        sizeinfos = []

        compression = flag & 0x3F
        if compression in [2, 3]: 
            blockdata = by.readseek0(解压前大小)
            bs.insertblock(text=f"块_解压前_{解压前大小}", bp=by.readslice(解压前大小).bp)
            bs.insertvalue(text=f"filetell_{by.tell()}")
            blockdata = lz4.block.decompress(blockdata, 解压后大小)
            with bs.insertblock(text=f"块_解压后_{解压后大小}", bp=by.fromstream(blockdata).bp), by.fromstream(blockdata):
                self.UnityFS_Head_Info_List(sizeinfos)

        self.UnityFS_File_Data_List(sizeinfos)
        bs.insertblock(text=f"余下_{by.remainsize()}", bp=by.readremainslice().bp)


    def UnityFS_Head_Info_List(self, sizeinfos):
        bs.insertblock(text=f"块_guid_16", bp=by.readslice(16).bp)
        numb = by.readuint32seek0()
        累加解压后大小, 累加解压前大小 = 0, 0
        with bs.insertblock(text=f"块_info_{numb}", bp=by.readslice(4).bp):
            for i in range(numb):
                with by.readsliceseek0(10):
                    解压后大小, 解压前大小, compressiontype = by.readuint32(), by.readuint32(), by.readuint16()
                    sizeinfos.append([解压后大小, 解压前大小, compressiontype])
                    累加解压后大小 += 解压后大小
                    累加解压前大小 += 解压前大小
                bs.insertblock(text=f"块_{i}_后前_{[解压后大小, 解压前大小, compressiontype]}", bp=by.readslice(10).bp)

        bs.insertvalue(text=f"累加_解压后_解压前_{累加解压后大小}_{累加解压前大小}")


        numb = by.readuint32seek0()   
        with bs.insertblock(text=f"块_node_{numb}", bp=by.readslice(4).bp):
            for j in range(numb):
                start = by.tell()
                offset, size, statu, name = by.readuint64(), by.readuint64(), by.readuint32(), by.readcharend0()
                final = by.tell()
                size1 = final -start
                by.seek(-size1)
                bs.insertblock(text=f"块_{j}_{[offset, size, statu, name]}", bp=by.readslice(size1).bp)





    def UnityFS_File_Data_List(self, sizeinfos):
        file = b""
        with bs.insertvalue(text=f"文件列表"):
            for i, [解压后大小, 解压前大小, compressiontype] in enumerate(sizeinfos):
                blockdata = by.readseek0(解压前大小)
                blockdata = lz4.block.decompress(blockdata, 解压后大小)
                file += blockdata
                bs.insertblock(text=f"块_{i}_compress_{解压前大小}", bp=by.readslice(解压前大小).bp)
                bs.insertblock(text=f"块_{i}_解压后_{解压后大小}", bp=by.fromstream(blockdata).bp)

        with bs.insertblock(text=f"解压出来的_{len(file)}", bp=by.fromstream(file).bp), by.fromstream(file):
            self.UnityFSAsserts文件分块()




    def UnityFSAsserts文件分块Wrapper(self):
        for item in self.frametreeview.treeview.selection():
            bp = self.frametreeview.bpdict[item].copy()
            self.context(item, bp)
            self.UnityFSAsserts文件分块()


    def UnityFSAsserts文件分块(self):
        bs.insertvalue(text=f"是小端")
        by.endian("<")

        bs.insertblock(text=f"Head_GUID_20", bp=by.readslice(20).bp)
        bs.insertblock(text=f"块_8", bp=by.readslice(8).bp)
        bs.insertblock(text=f"块_5", bp=by.readslice(5).bp)
        typeflags = []
        numtype = by.readuint32seek0()
        with bs.insertblock(text=f"块_4", bp=by.readslice(4).bp):
            for i in range(numtype):
                sizetype = 23
                flagtype = by.readuint32seek0()
                typeflags.append(flagtype)
                if flagtype == 114: sizetype = 39
                bs.insertblock(text=f"块_{i}_type_{sizetype}", bp=by.readslice(sizetype).bp)
                numbinfo, sizechar = by.readuint32seek0(2)
                bs.insertblock(text=f"块_{i}_{[numbinfo, sizechar]}", bp=by.readslice(8).bp)
                sizeinfo = numbinfo * 24
                bs.insertblock(text=f"块_{i}_info_{sizeinfo}", bp=by.readslice(sizeinfo).bp)
                bs.insertblock(text=f"块_{i}_char_{sizechar}", bp=by.readslice(sizechar).bp)

        start = by.tell()
        numinfo = by.readuint32()
        while True:
            if by.readuint8() != 0: break
        final = by.tell()
        size1 = final - start
        by.seek(-size1)
        sizenumb = size1 - 1 # numinfo + numpad0

        offsets = []
        with bs.insertblock(text=f"块_{sizenumb}", bp=by.readslice(sizenumb).bp):
            for i in range(numinfo):
                path1, path2, offset, size, index = by.readuint32seek0(5)
                offsets.append([path1, path2, offset, size, index])
                bs.insertblock(text=f"_{i}_{[path1, path2, offset, size, index]}_{offset+size}", bp=by.readslice(20).bp)


        typenames = []
        for typeflag in typeflags:
            try:
                typenames.append(flag_name_dict[typeflag])
            except:
                raise ValueError(f"{[typeflag]}, 有1个或多个未识别的typeflag！")
        bs.insertvalue(text=f"typenames", values=typenames)

        self.Asserts_Numb_List_and_Name_List()


        maxright = 0
        for path1, path2, offset, size, index in offsets:
            right = offset + size
            if right > maxright: maxright = right
        numpad0 = by.remainsize() - maxright
        bs.insertblock(text=f"块_{numpad0}", bp=by.readslice(numpad0).bp)
        前置偏移 = by.tell()

        with bs.insertblock(text=f"Data_file_{maxright}", bp=by.readsliceseek0(maxright).bp), by.readslice(maxright):
            for index, zero, offset, size, typeindex in offsets:
                by.movetell(offset+前置偏移)
                with by.readsliceseek0(1024):
                    name = by.readchar(by.readuint32())
                bs.insertblock(text=f"{[index, offset, size, offset+size]}__{typenames[typeindex]}__{name}", bp=by.readslice(size).bp)
            bs.insertblock(text=f"余下_{by.remainsize()}", bp=by.readremainslice().bp)

        bs.insertblock(text=f"余下_{by.remainsize()}", bp=by.readremainslice().bp)








    def Asserts文件分块(self):
        bs.insertvalue(text=f"是小端")
        by.endian("<")

        maxright = self.Asserts_offsets()
        size1, size2 = by.right() - maxright, maxright

        offsets = []
        typeflags = []
        with bs.insertblock(text=f"Head_info_{size1}", bp=by.readsliceseek0(size1).bp), by.readslice(size1):
            bs.insertblock(text=f"Head_GUID_20", bp=by.readslice(20).bp)
            self.Asserts_Version()
            bs.insertblock(text=f"块_6", bp=by.readslice(6).bp)
            self.Asserts_Type_List_and_File_List(offsets, typeflags)
            self.Asserts_Numb_List_and_Name_List()
            bs.insertblock(text=f"余下_{by.remainsize()}", bp=by.readremainslice().bp)

        typenames = []
        for typeflag in typeflags:
            try:
                typenames.append(flag_name_dict[typeflag])
            except:
                raise ValueError(f"{[typeflag]}, 有1个或多个未识别的typeflag！")




        with bs.insertblock(text=f"Data_file_{size2}", bp=by.readsliceseek0(size2).bp), by.readslice(size2):
            for index, zero, offset, size, typeindex in offsets:
                by.movetell(offset+size1)
                with by.readsliceseek0(1024):
                    name = by.readchar(by.readuint32())
                bs.insertblock(text=f"{[index, offset, size, offset+size]}_{typenames[typeindex]}_{name}", bp=by.readslice(size).bp)
            bs.insertblock(text=f"余下_{by.remainsize()}", bp=by.readremainslice().bp)

        bs.insertblock(text=f"余下_{by.remainsize()}", bp=by.readremainslice().bp)



    def Asserts_offsets(self):
        with by.readremainsliceseek0():
            head = by.readuint8(20)
            bufferchar = by.readcharseek0(11)
            if "5.6.4p4" in bufferchar: versionchar = by.readchar(7)
            if "2017.4.18f1" in bufferchar: versionchar = by.readchar(11)
            buffer1 = by.readslice(6)

            numtype = by.readuint32()
            for i in range(numtype): 
                size = 23
                typeflag = by.readuint32seek0()
                if typeflag == 114: size = 39
                block = by.readslice(size)

            numfile = by.readuint32()
            numpad0 = 0
            if [by.readuint8seek0()] == [1]: numpad0 = 0
            if  by.readuint8seek0(2) == [0, 1]: numpad0 = 1
            if  by.readuint8seek0(3) == [0, 0, 1]: numpad0 = 2
            if  by.readuint8seek0(4) == [0, 0, 0, 1]: numpad0 = 3
            block = by.readslice(numpad0)


            maxright = 0
            for j in range(numfile): 
                index, zero, offset, size, typeindex = by.readuint32(5)
                right = offset + size
                if right > maxright: maxright = right
        return maxright



    def Asserts_Version(self):
        bufferchar = by.readcharseek0(11)
        if "5.6.4p4" in bufferchar: size = 7
        if "2017.4.18f1" in bufferchar: size = 11
        versionchar = by.readcharseek0(size)
        if versionchar not in ["5.6.4p4", "2017.4.18f1"]: raise ValueError(f"未知的unity版本{versionchar}")
        bs.insertblock(text=f"File_Version_{versionchar}", bp=by.readslice(size).bp)


    def Asserts_Type_List_and_File_List(self, offsets, typeflags):
        numtype = by.readuint32seek0()
        with bs.insertblock(text=f"numtype_{numtype}", bp=by.readslice(4).bp):
            for i in range(numtype): 
                size = 23
                typeflag = by.readuint32seek0()
                if typeflag == 114: size = 39
                typeflags.append(typeflag)
                bs.insertblock(text=f"块_{i}_{size}", bp=by.readslice(size).bp)


        numfile = by.readuint32seek0()
        with bs.insertblock(text=f"numfile_{numfile}_{4}", bp=by.readslice(4).bp):
            numpad0 = 0
            if [by.readuint8seek0()] == [1]: numpad0 = 0
            if  by.readuint8seek0(2) == [0, 1]: numpad0 = 1
            if  by.readuint8seek0(3) == [0, 0, 1]: numpad0 = 2
            if  by.readuint8seek0(4) == [0, 0, 0, 1]: numpad0 = 3
            bs.insertblock(text=f"numpad0_{numpad0}", bp=by.readslice(numpad0).bp)

            for j in range(numfile): 
                index, zero, offset, size, typeindex = by.readuint32seek0(5)
                offsets.append([index, zero, offset, size, typeindex])
                bs.insertblock(text=f"{[index, zero, offset, size, typeindex]}_{offset+size}", bp=by.readslice(20).bp)
        
    def Asserts_Numb_List_and_Name_List(self):
        numnumb = by.readuint32seek0()
        with bs.insertblock(text=f"numnumb_{numnumb}", bp=by.readslice(4).bp):
            for i in range(numnumb): 
                bs.insertblock(text=f"_{i}_{by.readuint32seek0(3)}", bp=by.readslice(12).bp)

        numname = by.readuint32seek0()
        with bs.insertblock(text=f"numname_{numname}", bp=by.readslice(4).bp):
            for i in range(numname):  
                start = by.tell()
                bufferuint8 = by.readuint8(21)
                bufferchars = by.readcharend0()
                final = by.tell()
                size1 = final - start
                by.seek(-size1)
                bs.insertblock(text=f"_{i}_{bufferchars}", bp=by.readslice(size1).bp)        
                        

        
     







    def Find_UnityFS(self): # only 1 result
        tells = []
        with by.readremainsliceseek0():
            while True:
                if by.remainsize() < 20: break
                if by.readuint8() == 85:
                    if by.readuint8seek0(6) == [110, 105, 116, 121, 70, 83]: tells.append(by.tell()-1)


        sizes = []
        for i in  range(1, len(tells)):
            sizes.append(tells[i]-tells[i-1])

        for size in sizes:
            bs.insertblock(text=f"UnityFS_{size}", bp=by.readslice(size).bp)


        bs.insertblock(text=f"余下_{by.remainsize()}_{by.tell()}_{by.remainsize()+by.tell()}", bp=by.readremainslice().bp)




    def 异或测试(self):
        bufferuint8 = by.readuint8(7)
        
        for i in range(256):
            resultchar8 = ""
            resultuint8 = []
            for uint8 in bufferuint8:
                a = uint8 ^ i
                resultchar8 += chr(a)
                resultuint8.append(a)
            print(i, resultchar8, resultuint8)
            






    def Wmv文件分块(self):
        bs.insertvalue(text=f"是大端")
        by.endian(">")

        count = -1
        while True:
            count += 1
            if by.remainsize() < 20: break
            with by.readsliceseek0(115):
                by.readslice(52)
                flag = by.readuint8(4)
                if flag == [1,0,178,1]:
                    解压后大小, 解压前大小 = by.readuint32seek0(2)
                    sizefile = 115 + 解压前大小
                if flag == [1,0,48,8]:
                    by.readslice(44)
                    解压后大小, 解压前大小 = by.readuint32seek0(2)
                    sizefile = 161 + 解压前大小

            bs.insertblock(text=f"File_{count}_{sizefile}", bp=by.readslice(sizefile).bp)

        bs.insertblock(text=f"余下_{by.remainsize()}", bp=by.readremainslice().bp)






    def WmvBundlec文件分块1(self):
        bs.insertvalue(text=f"是大端")
        by.endian(">")
        with bs.insertblock(text=f"UnityFS_File_115", bp=by.readsliceseek0(115).bp), by.readslice(115):
            bs.insertblock(text=f"Head_UnityFS_8", bp=by.readslice(8).bp) # 'UnityFS' end0
            bs.insertblock(text=f"Head_48", bp=by.readslice(48).bp) 
            解压后大小, 解压前大小 = by.readuint32seek0(2)
            bs.insertblock(text=f"Head_compsize后前_{[解压后大小, 解压前大小]}", bp=by.readslice(8).bp) # 2133, 4761
            bs.insertblock(text=f"Head_14", bp=by.readslice(14).bp) 
            bs.insertblock(text=f"余下_{by.remainsize()}", bp=by.readremainslice().bp)

        blockdata = by.readseek0(解压前大小)
        bs.insertblock(text=f"块_解压前_{解压前大小}", bp=by.readslice(解压前大小).bp)
        blockdata = lz4.block.decompress(blockdata, 解压后大小)
        bs.insertblock(text=f"块_解压后_{解压后大小}", bp=by.fromstream(blockdata).bp)

    def WmvBundlec文件分块2(self):
        bs.insertvalue(text=f"是大端")
        by.endian(">")
        with bs.insertblock(text=f"UnityFS_File_115", bp=by.readsliceseek0(115).bp), by.readslice(115):
            bs.insertblock(text=f"Head_UnityFS_8", bp=by.readslice(8).bp) # 'UnityFS' end0
            bs.insertblock(text=f"Head_48", bp=by.readslice(48).bp) 
            bs.insertblock(text=f"Head_44", bp=by.readslice(44).bp) 
            解压后大小, 解压前大小 = by.readuint32seek0(2)
            bs.insertblock(text=f"Head_compsize后前_{[解压后大小, 解压前大小]}", bp=by.readslice(8).bp) # 2133, 4761
            bs.insertblock(text=f"Head_14", bp=by.readslice(14).bp) 
            bs.insertblock(text=f"余下_{by.remainsize()}", bp=by.readremainslice().bp)

        blockdata = by.readseek0(解压前大小)
        bs.insertblock(text=f"块_解压前_{解压前大小}", bp=by.readslice(解压前大小).bp)
        blockdata = lz4.block.decompress(blockdata, 解压后大小)
        bs.insertblock(text=f"块_解压后_{解压后大小}", bp=by.fromstream(blockdata).bp)





    def WmvAsserts文件分块_2017_4_18f1(self):
        bs.insertvalue(text=f"是小端")
        by.endian("<")

        bs.insertblock(text=f"Head_GUID_20", bp=by.readslice(20).bp)
        bs.insertblock(text=f"块_14", bp=by.readslice(14).bp)
        bs.insertblock(text=f"块_5", bp=by.readslice(5).bp)

        offsets = []
        typeflags = []
        self.Asserts_Type_List_and_File_List(offsets, typeflags)
        self.Asserts_Numb_List_and_Name_List()
    
        typenames = []
        for typeflag in typeflags:
            try:
                typenames.append(flag_name_dict[typeflag])
            except:
                raise ValueError(f"{[typeflag]}, 有1个或多个未识别的typeflag！")

        tell = by.tell()
        if tell < 4096: sizepad0 = 4096 - tell
        bs.insertblock(text=f"块_{sizepad0}", bp=by.readslice(sizepad0).bp)

        bs.insertblock(text=f"余下_{by.remainsize()}", bp=by.readremainslice().bp)






