
import 界面文件.bcontext.bsfunction as bs
import 底层文件.bpformat.byfunction as by
import 底层文件

from . import tkinter_file

class 类:
    def __init__(self, frametreeview):
        self.frametreeview = frametreeview


    def context(self, item="", bp=None):
        if item == "": item = self.frametreeview.treeview.selection()[0]
        by.context(bp)
        bs.context(item, self.frametreeview)
        return item
    
    def wrapper(self, function=None):
        def 执行函数():
            for item in self.frametreeview.treeview.selection():
                bp = self.frametreeview.bpdict[item].copy()
                self.context(item, bp)
                function(self)
        return 执行函数

    # @staticmethod
    # def 执行函数装饰函数(function):
    #     def 执行函数(self):
    #         for item in self.frametreeview.treeview.selection():
    #             bp = self.frametreeview.bpdict[item].copy()
    #             self.context(item, bp)
    #             function(self)
    #     return 执行函数




    def 文件解析(self):
        for item in self.frametree.tree.selection():
            self.context(item)
            bp = by.filepath(by.rdpath()).bp
            fbxfile = 文件.fbx.类(bp)
            for char, value in fbxfile.blocks.items():
                bs.insertblock(text=char, bp=value.copy())
     


    def 文件分块(self):
        for item in self.frametree.tree.selection():
            self.context(item)
            bs.insertblock(text="head", bp=by.readslice(27).bp)
            while True:
                if by.remainsize() < 16: break
                if by.readuint32seek0() == 0: break
                sliceleft, sliceright = by.tell(), by.readuint32seek0()
                size = sliceright - sliceleft
                with by.readsliceseek0(size):
                    sliceright, numvalue, datasize, blockname = by.readuint32(), by.readuint32(), by.readuint32(), by.readchar(by.readuint8())
                    # print(blockname)
                bs.insertblock(text=str([size, sliceleft, sliceright, numvalue, datasize, blockname]), bp=by.readslice(size).bp)
            bs.insertblock(text="余下", bp=by.readremainslice().bp)


    def 单独节点分块(self):
        for item in self.frametree.tree.selection():
            self.context(item)
            self.__单独节点__()


    def __单独节点__(self):
        with by.copy():
            sliceleft = by.tell()
            sliceright, numvalue, datasize, numchar = by.readuint32(), by.readuint32(), by.readuint32(), by.readuint8()
            blockname = by.readchar(numchar)

        asize = 13 + numchar
        bsize = datasize
        csize = sliceright - sliceleft - datasize - 13 - numchar
        bs.insertblock(text=f"块_{asize}", bp=by.readslice(asize).bp)
        bs.insertblock(text=f"块_{bsize}", bp=by.readslice(bsize).bp)
        with bs.insertblock(text=f"块_{csize}", bp=by.readsliceseek0(csize).bp), by.readslice(csize):
            while True:
                if by.remainsize() < 16: break
                sliceleft, sliceright = by.tell(), by.readuint32seek0()
                size = sliceright - sliceleft
                with by.readsliceseek0(size):
                    sliceright, numvalue, datasize, blockname = by.readuint32(), by.readuint32(), by.readuint32(), by.readchar(by.readuint8())
                    print(blockname)
                bs.insertblock(text=str([size, sliceleft, sliceright, numvalue, datasize, blockname]), bp=by.readslice(size).bp)



    def 递归节点分块(self):
        for item in self.frametree.tree.selection():
            self.context(item)
            self.__递归节点__()


    def __递归节点__(self):
        with by.copy():
            sliceleft = by.tell()
            sliceright, numvalue, datasize, numchar = by.readuint32(), by.readuint32(), by.readuint32(), by.readuint8()
            blockname = by.readchar(numchar)

        asize = 13 + numchar
        bsize = datasize
        csize = sliceright - sliceleft - datasize - 13 - numchar

        bs.insertblock(text=f"块_{asize}", bp=by.readslice(asize).bp)

        # values
        with bs.insertblock(text=f"块_{bsize}", bp=by.readsliceseek0(bsize).bp), by.readslice(bsize):
            for i in range(numvalue):
                valuetype = by.readchar()
                match valuetype:
                    case "S" : 
                        bx = by.readsliceseek0(4+by.readuint32seek0()).bp
                        bs.insertblock(text=by.readchar(by.readuint32()), bp=bx)
                    case "R" : bs.insertblock(text=valuetype, bp=by.readslice(1).bp) # 
                    case "B" : bs.insertblock(text=valuetype, bp=by.readslice(1).bp) # 字节
                    case "C" : bs.insertblock(text=valuetype, bp=by.readslice(1).bp) # bool
                    case "Y" : bs.insertblock(text=valuetype, bp=by.readslice(2).bp) # int16
                    case "I" : bs.insertblock(text=valuetype, bp=by.readslice(4).bp) # int32
                    case "L" : bs.insertblock(text=valuetype, bp=by.readslice(8).bp) # int64
                    case "F" : bs.insertblock(text=valuetype, bp=by.readslice(4).bp) # float32
                    case "D" : bs.insertblock(text=valuetype, bp=by.readslice(8).bp) # float64
                    case "i" :
                        numelem, elemtype, datasize = by.readuint32seek0(3)
                        bs.insertblock(text=[valuetype, numelem, elemtype, datasize], bp=by.readslice(12).bp)
                        bs.insertblock(text=[numelem, elemtype, datasize], bp=by.readsliceseek0(datasize).bp)
                        with by.readslice(datasize):
                            if elemtype == 1:
                                bpbytes = zlib.decompress(by.read(datasize)) # TypeError: a bytes-like object is required, not 'str'
                                bs.insertbytes(text=[numelem, elemtype, len(bpbytes)], bp=by.tobpbytes(bpbytes))
                    case "l" :
                        numelem, elemtype, datasize = by.readuint32seek0(3)
                        bs.insertblock(text=[valuetype, numelem, elemtype, datasize], bp=by.readslice(12).bp)
                        bs.insertblock(text=[numelem, elemtype, datasize], bp=by.readsliceseek0(datasize).bp)
                        with by.readslice(datasize):
                            if elemtype == 1:
                                bpbytes = zlib.decompress(by.read(datasize)) # TypeError: a bytes-like object is required, not 'str'
                                bs.insertbytes(text=[numelem, elemtype, len(bpbytes)], bp=by.tobpbytes(bpbytes))
                    case "f" :
                        numelem, elemtype, datasize = by.readuint32seek0(3)
                        bs.insertblock(text=[valuetype, numelem, elemtype, datasize], bp=by.readslice(12).bp)
                        bs.insertblock(text=[numelem, elemtype, datasize], bp=by.readsliceseek0(datasize).bp)
                        with by.readslice(datasize):
                            if elemtype == 1:
                                bpbytes = zlib.decompress(by.read(datasize)) # TypeError: a bytes-like object is required, not 'str'
                                bs.insertbytes(text=[numelem, elemtype, len(bpbytes)], bp=by.tobpbytes(bpbytes))
                    case "d" :
                        numelem, elemtype, datasize = by.readuint32seek0(3)
                        bs.insertblock(text=[valuetype], bp=by.readslice(12).bp)
                        bs.insertblock(text=[numelem, elemtype, datasize], bp=by.readsliceseek0(datasize).bp)
                        with by.readslice(datasize):
                            if elemtype == 1:
                                bpbytes = zlib.decompress(by.read(datasize)) # TypeError: a bytes-like object is required, not 'str'
                                bs.insertbytes(text=[numelem, elemtype, len(bpbytes)], bp=by.tobpbytes(bpbytes))
                    case  _  :
                        bs.insertblock(text="余下", bp=by.readremainslice().bp)
                        break

        # if blockname == "P": print()
        # childs
        with bs.insertblock(text=f"块_{csize}", bp=by.readsliceseek0(csize).bp), by.readslice(csize):
            while True:
                if by.remainsize() < 16: break
                sliceleft, sliceright = by.tell(), by.readuint32seek0()
                size = sliceright - sliceleft
                with by.readsliceseek0(size):
                    sliceright, numvalue, datasize, blockname = by.readuint32(), by.readuint32(), by.readuint32(), by.readchar(by.readuint8())
                with bs.insertblock(text=str([size, sliceleft, sliceright, numvalue, datasize, blockname]), bp=by.readsliceseek0(size).bp), by.readslice(size):
                    self.__递归节点__()


    