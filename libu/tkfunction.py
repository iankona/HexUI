import tkinter as tk
import tkinter.ttk as ttk



record_item = None
class contextitem:
    def __enter__(self):
        self.parent_item = record_item
        self.setter_item(self.create_item)

    def __exit__(self, type=None, value=None, traceback=None):
        self.setter_item(self.parent_item)
        AddToParent(self.parent_item, self.create_item)

    def setter_item(self, attrib_item):
        global record_item
        record_item = attrib_item


record_text = ""
class contexttext:
    def __enter__(self):
        # self.parent_text = record_text
        self.setter_item(self.create_text)

    def __exit__(self, type=None, value=None, traceback=None):
        pass
        # self.setter_item(self.parent_text)

    def setter_item(self, attrib_item):
        global record_text
        record_text = attrib_item




record_item_buff = None
record_text_buff = ""
def GetRecord():
    global record_item_buff, record_text_buff
    record_item_buff = record_item
    record_text_buff = record_text


def ChangeRecord(record_item_this, record_text_this=""):
    global record_item, record_text
    record_item = record_item_this
    if record_text_this != "": record_text = record_text_this


def SetRecord():
    global record_item, record_text
    record_item = record_item_buff
    record_text = record_text_buff




class Title(contexttext):
    def __init__(self, text="标题1"):
        self.create_text = text    



class Tk(contextitem):
    def __init__(self, **kwargs):
        self.create_item = tk.Tk()

    def __exit__(self, type, value, traceback):
        contextitem.__exit__(self)
        self.create_item.mainloop()


class Toplevel(contextitem):
    def __init__(self, **kwargs):
        self.create_item = tk.Toplevel()
        geometry = kwargs.pop("geometry", None)
        overrideredirect = kwargs.pop("overrideredirect", None)
        if geometry != None: self.create_item.geometry(geometry)
        if overrideredirect != None: self.create_item.overrideredirect(overrideredirect)
        # self.create_item.overrideredirect(True) # 不显示最小化和关闭按钮
        # self.create_item.withdraw() # 隐藏窗口
    def __exit__(self, type, value, traceback):
        self.setter_item(self.parent_item)






def AddToParent(parent, child):
    match type(parent):
        case ttk.Frame: pass
        case ttk.Notebook: parent.add(child, text=record_text)
        case ttk.PanedWindow: parent.add(child)



def PopPlace(kwargs):
    x = kwargs.pop("x", None) # 绝对定位
    y = kwargs.pop("y", None)
    # width = kwargs.pop("width", None) 
    # height = kwargs.pop("height", None)
    # relx = kwargs.pop("relx", None)
    # rely = kwargs.pop("rely", None)
    # relwidth = kwargs.pop("relwidth", None) 
    # relheight = kwargs.pop("relheight", None)
    # anchor = kwargs.pop("anchor", None) # 'center', 'n', 'ne', 'e', 'se', 's', 'sw', 'w', 'nw'
    return x, y

def PopGrid(kwargs):
    # print(item.grid_info())
    row = kwargs.pop("row", None)
    column = kwargs.pop("column", None)
    # rowspan = kwargs.pop("rowspan", None)
    # columnspan = kwargs.pop("columnspan", None)
    sticky = kwargs.pop("sticky", '')  # 默认 'n', 'ne', 'e', 'se', 's', 'sw', 'w', 'nw'
    # ipadx = kwargs.pop("ipadx", None) # 水平方向内边距
    # ipady = kwargs.pop("ipady", None) # 垂直方向内边距
    # padx = kwargs.pop("padx", None) # 水平方向外边距
    # pady = kwargs.pop("pady", None) # 垂直方向外边距
    return row, column, sticky
    # return row, column, rowspan, columnspan, sticky

def PopPack(kwargs):
    side = kwargs.pop("side", 'left') # 'left', 'right', 'top', 'bottom'
    fill = kwargs.pop("fill", 'none') # 默认 'none', 'both', 'x', 'y'
    expand = kwargs.pop("expand", 0) # true or false
    # anchor = kwargs.pop("anchor", None) # 'center', 'n', 'ne', 'e', 'se', 's', 'sw', 'w', 'nw'
    # ipadx = kwargs.pop("ipadx", None) # 水平方向内边距
    # ipady = kwargs.pop("ipady", None) # 垂直方向内边距
    # padx = kwargs.pop("padx", None) # 水平方向外边距
    # pady = kwargs.pop("pady", None) # 垂直方向外边距
    return side, fill, expand



class FramePack(contextitem):
    def __init__(self, **kwargs):
        side, fill, expand = PopPack(kwargs)
        self.create_item = ttk.Frame(record_item)
        pack_propagate = kwargs.pop("pack_propagate", None)
        if pack_propagate != None: self.create_item.pack_propagate(pack_propagate)
        for key, value in kwargs.items(): self.create_item[key] = value
        self.create_item.pack(side=side, fill=fill, expand=expand)




class PanedWindowVPack(contextitem):
    def __init__(self, **kwargs):
        side, fill, expand = PopPack(kwargs)
        self.create_item = ttk.PanedWindow(orient="vertical")
        for key, value in kwargs.items(): self.create_item[key] = value
        self.create_item.pack(side="left", fill="y", expand=1)


class PanedWindowHPack(contextitem):
    def __init__(self, **kwargs):
        side, fill, expand = PopPack(kwargs)
        self.create_item = ttk.PanedWindow(orient="horizontal")
        for key, value in kwargs.items(): self.create_item[key] = value
        self.create_item.pack(side="top", fill="x", expand=1)


def LabelPack(**kwargs):
    side, fill, expand = PopPack(kwargs)    
    child = ttk.Label(record_item)
    for key, value in kwargs.items(): child[key] = value
    child.pack(side=side, fill=fill, expand=expand)
    return child

def LabelGrid(**kwargs):
    row, column, sticky = PopGrid(kwargs)    
    child = ttk.Label(record_item)
    for key, value in kwargs.items(): child[key] = value
    child.grid(row=row, column=column, sticky=sticky)
    return child

def EntryGrid(**kwargs):
    row, column, sticky = PopGrid(kwargs)    
    child = ttk.Entry(record_item)
    for key, value in kwargs.items(): child[key] = value
    child.grid(row=row, column=column, sticky=sticky)
    return child

def ButtonGrid(**kwargs):
    row, column, sticky = PopGrid(kwargs)    
    child = ttk.Button(record_item)
    for key, value in kwargs.items(): child[key] = value
    child.grid(row=row, column=column, sticky=sticky)
    return child


def ScrollBarHPack(**kwargs):
    side, fill, expand = PopPack(kwargs)    
    child = ttk.Scrollbar(record_item, orient='horizontal')
    for key, value in kwargs.items(): child[key] = value
    child.pack(side=side, fill=fill, expand=expand)
    return child

def ScrollBarVPack(**kwargs):
    side, fill, expand = PopPack(kwargs)    
    child = ttk.Scrollbar(record_item, orient='vertical')
    for key, value in kwargs.items(): child[key] = value
    child.pack(side=side, fill=fill, expand=expand)
    return child


def TreeViewPack(**kwargs):
    side, fill, expand = PopPack(kwargs)    
    child = ttk.Treeview(record_item)
    for key, value in kwargs.items(): child[key] = value
    child.pack(side=side, fill=fill, expand=expand)

    columns = [f"#{i+1}" for i in range(16)]
    child["columns"] = columns
    child.column("#0", width=200, stretch=0)
    for column in [f"#{i+1}" for i in range(16)]: # 设置列宽
        child.column(column, width=40, stretch=False, anchor='center')
        child.heading(column, text=column)
    return child



    def __init__(self, **kwargs):
        side, fill, expand = PopPack(kwargs)
        self.create_item = ttk.Treeview(record_item)
        for key, value in kwargs.items(): self.create_item[key] = value
        self.create_item.pack(side=side, fill=fill, expand=expand)





class NoteBookPack(contextitem):
    def __init__(self, **kwargs):
        side, fill, expand = PopPack(kwargs)
        self.create_item = ttk.Notebook(record_item)
        for key, value in kwargs.items(): self.create_item[key] = value
        self.create_item.pack(side="top", fill="both", expand=1)



class CanvasPack(contextitem):
    def __init__(self, **kwargs):
        side, fill, expand = PopPack(kwargs)
        self.create_item = tk.Canvas(record_item)
        for key, value in kwargs.items(): self.create_item[key] = value
        self.create_item.pack(side="top", fill="both", expand=1)   


def CanvasAddRect(x1,y1,x2,y2):
    record_item.create_rectangle(x1,y1,x2,y2)
    record_item.create_line(30,30,100,200,200,200,300,200,300,300)
    pass

def CanvasScale():
    record_item.scale("all", 0, 0, 0.1, 0.1)

def draw():
    excel.Excel()
    regionlist = excel.GetRegionRowDirect(2, 1, 18, 9)
    x, y = 0, 0
    for valuelist in regionlist: 
        length = valuelist[1]
        width = valuelist[2]
        CanvasAddRect(x,y,x+length,y-width)
        y += (width + 20)
    CanvasScale()
if __name__ == '__main__':
    with Tk():
        with NoteBookPack():
            with CanvasPack(), Title("画布1"):
                pass
                # draw()
            with PanedWindowVPack(), Title("标题1"):
                with FramePack(width=45, height=60, side="top", expand=1):
                    LabelPack(text="鹅鹅鹅1")
                    LabelPack(text="鹅鹅鹅")
                    LabelPack(text="鹅鹅鹅")
                with PanedWindowHPack():
                    with FramePack(width=55, height=70, side="top", expand=1):
                        LabelPack(text="鹅鹅鹅21")
                        LabelPack(text="鹅鹅鹅")
                        LabelPack(text="鹅鹅鹅")
                    with FramePack(width=65, height=80, side="top", expand=1):
                        LabelPack(text="鹅鹅鹅22")
                        LabelPack(text="鹅鹅鹅")
                        LabelPack(text="鹅鹅鹅")
                with FramePack(width=75, height=90, side="top", expand=1):
                    LabelPack(text="鹅鹅鹅3")
                    LabelPack(text="鹅鹅鹅")
                    LabelPack(text="鹅鹅鹅")
            with FramePack(width=450, height=600, side="top", expand=1), Title("标题2"):
                    LabelPack(text="鹅鹅鹅1")
                    LabelPack(text="鹅鹅鹅")
                    LabelPack(text="鹅鹅鹅")