
import 界面.bcontext.bsfunction as bs
import 文件.bpformat.byfunction as by
import 文件



class 类:
    def wrappercontext(self, function=None):
        def 执行函数(): # *args, **kwargs == (), {}
            for item in self.frametreeview.treeview.selection():
                bp = self.frametreeview.bpdict[item].copy()
                by.context(bp)
                bs.context(item, self.frametreeview)
                function() # function(self) # TypeError: 类.Find_UnityFS() takes 1 positional argument but 2 were given
        return 执行函数
    

    def wrapperfile(self, function=None):
        def 执行函数(): # *args, **kwargs == (), {}
            for item in self.frametreeview.treeview.selection():
                bp = self.frametreeview.bpdict[item].copy()
                by.context(bp)
                bs.context(item, self.frametreeview)
                function() # function(self) # TypeError: 类.Find_UnityFS() takes 1 positional argument but 2 were given
        return 执行函数


    def wrapperinsert(self, function=None, nextfunc=None):
        def 执行函数(*args, **kwargs): 
            self = args[0]
            label = kwargs.get("label", "")
            with bs.insertvalue() as fitem:
                start = by.tell()
                function(*args, **kwargs)
                final = by.tell()
                if nextfunc != None: nextfunc(*args, **kwargs)
            sizeb = final - start
            by.seek(-sizeb)
            self.frametreeview.itemblock(fitem, text=f"{label}_{sizeb}", bp=by.readslice(sizeb).bp)
        return 执行函数