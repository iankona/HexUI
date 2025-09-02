import tkinter as tk
import random

class CanvasCAD(tk.Frame):
    def __init__(self, root):
        tk.Frame.__init__(self, root)
        self.canvas = tk.Canvas(self, width=400, height=400, background="bisque")
        # self.xsb = tk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        # self.ysb = tk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        # self.canvas.configure(yscrollcommand=self.ysb.set, xscrollcommand=self.xsb.set)
        # self.canvas.configure(scrollregion=(0,0,1000,1000))

        # self.xsb.grid(row=1, column=0, sticky="ew")
        # self.ysb.grid(row=0, column=1, sticky="ns")
        self.canvas.pack(side="top", fill='both', expand=1)
        # self.grid_rowconfigure(0, weight=1)
        # self.grid_columnconfigure(0, weight=1)

        #Plot some rectangles
        for n in range(50):
            x0 = random.randint(0, 900)
            y0 = random.randint(50, 900)
            x1 = x0 + random.randint(50, 100)
            y1 = y0 + random.randint(50,100)
            color = ("red", "orange", "yellow", "green", "blue")[random.randint(0,4)]
            self.canvas.create_rectangle(x0,y0,x1,y1, outline="black", fill=color, activefill="black", tags="图层0")
        textid = self.canvas.create_text(50,10, anchor="nw", text="Click and drag to move the canvas\nScroll to zoom.")
        textstr = self.canvas.itemcget(textid, "text")
        print("文本输出", textstr)
        self.canvas.itemconfig(textid, text="鹅鹅鹅", width=10)


        entry_widget1 = tk.Label(self, text="标签1")
        self.canvas.create_window(100, 200, width=20, window=entry_widget1)


        entry_widget2 = tk.Entry(self)
        self.canvas.create_window(300, 200, width=10, window=entry_widget2)


        # This is what enables using the mouse:
        self.canvas.bind("<ButtonPress-1>", self.move_start)
        self.canvas.bind("<B1-Motion>", self.move_move)
        # #linux scroll
        # self.canvas.bind("<Button-4>", self.zoomerP)
        # self.canvas.bind("<Button-5>", self.zoomerM)
        #windows scroll
        self.canvas.bind("<MouseWheel>",self.zoomer)

        # # 为鼠标按钮绑定画布
        # canvas.bind("<Button-1>", on_button_pressed)
        # canvas.bind("<Button1-Motion>", on_button_motion)


    def 获取对象坐标(self):
        rect = self.canvas.create_rectangle(50, 50, 150, 150, fill="blue")
        oval = self.canvas.create_oval(200, 200, 300, 300, fill="red")
        text = self.canvas.create_text(200, 100, text="Hello World")    

        rect_coords = self.canvas.coords(rect)
        print(rect_coords)  # 输出 [50.0, 50.0, 150.0, 150.0]

        oval_bbox = self.canvas.bbox(oval)
        text_bbox = self.canvas.bbox(text)
        print(oval_bbox)  # 输出 [200.0, 200.0, 300.0, 300.0]
        print(text_bbox)  # 输出 [187.5, 86.5, 212.5, 113.5]

        all_objects = self.canvas.find_all()

        for obj in all_objects:
            obj_coords = self.canvas.coords(obj)
            print(obj_coords)

    def 获取坐标附近对象(self, event):  # 定义一个显示控件ID的函数
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        widget_id = self.canvas.find_closest(x, y)
        self.canvas.focus(widget_id)
        print(f"当前控件的ID是: {widget_id}")  # 输出控件ID


    #move
    def move_start(self, event):
        self.获取坐标附近对象(event)
        self.on_button_pressed(event)
        self.canvas.scan_mark(event.x, event.y)

    def move_move(self, event):
        self.on_button_motion(event)
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    #windows zoom
    def zoomer(self,event):
        if (event.delta > 0):
            self.canvas.scale("all", event.x, event.y, 1.1, 1.1)
        elif (event.delta < 0):
            self.canvas.scale("all", event.x, event.y, 0.9, 0.9)
        # self.canvas.configure(scrollregion = self.canvas.bbox("all"))


    def 移动绘图对象(self):
        # 绘制圆形
        circle_id = self.canvas.create_oval(50, 50, 70, 70, fill="red")
        # 事件函数def move_circle(event):
        self.canvas.move(circle_id, 10, 0)  # 向右移动 10 像素



    def on_button_pressed(self,event):
        start_x = self.canvas.canvasx(event.x)
        start_y = self.canvas.canvasy(event.y)
        print("start_x, start_y = ", start_x, start_y)

    def on_button_motion(self,event):
        end_x = self.canvas.canvasx(event.x)
        end_y = self.canvas.canvasy(event.y)
        print("end_x, end_y = ", end_x, end_y)

    # #linux zoom
    # def zoomerP(self,event):
    #     self.canvas.scale("all", event.x, event.y, 1.1, 1.1)
    #     self.canvas.configure(scrollregion = self.canvas.bbox("all"))
    # def zoomerM(self,event):
    #     self.canvas.scale("all", event.x, event.y, 0.9, 0.9)
    #     self.canvas.configure(scrollregion = self.canvas.bbox("all"))



if __name__ == "__main__":
    root = tk.Tk()
    Canvas(root).pack(fill="both", expand=True)
    root.mainloop()



# import tkinter as tk

# # 创建窗口
# window = tk.Tk()

# # 创建Canvas控件
# canvas = tk.Canvas(window, width=400, height=400)
# canvas.pack()

# # 在Canvas控件中绘制矩形
# rect = canvas.create_rectangle(50, 50, 150, 150, fill="blue")

# # 定义处理点击事件的回调函数
# def on_rect_click(event):
#     print("矩形被点击了！")

# # 绑定矩形的点击事件
# canvas.tag_bind(rect, "<Button-1>", on_rect_click)

# # 启动窗口的消息循环
# window.mainloop()


# find_above(item)
# -- 返回在 item 参数指定的画布对象之上的 ID
# -- 如果有多个画布对象符合要求，那么返回最顶端的那个
# -- 如果 item 参数指定的是最顶层的画布对象，那么返回一个空元组
# -- item 可以是单个画布对象的 ID，也可以是某个 Tag

# find_all()
# -- 返回 Canvas 组件上所有的画布对象
# -- 返回格式是一个元组，包含所有画布对象的 ID
# -- 按照显示列表的顺序返回
# -- 该方法相当于 find_withtag("all")

# find_below(item)
# -- 返回在 item 参数指定的画布对象之下的 ID
# -- 如果有多个画布对象符合要求，那么返回最底端的那个
# -- 如果 item 参数指定的是最底层的画布对象，那么返回一个空元组
# -- item 可以是单个画布对象的 ID，也可以是某个 Tag

# find_closest(x, y, halo=None, start=None)
# -- 返回一个元组，包含所有靠近点（x, y）的画布对象的 ID
# -- 如果没有符合的画布对象，则返回一个空元组
# -- 可选参数 halo 用于增加点（x, y）的辐射范围
# -- 可选参数 start 指定一个画布对象，该方法仅返回在显示列表中低于但最接近的一个 ID
# -- 注意，点（x, y）的坐标是采用画布坐标系来表示

# find_enclosed(x1, y1, x2, y2)
# -- 返回完全包含在限定矩形内所有画布对象的 ID

# find_overlapping(x1, y1, x2, y2)
# -- 返回所有与限定矩形有重叠的画布对象的 ID（让然也包含在限定矩形内的画布对象）

# find_withtag(item)
# -- 返回 item 指定的所有画布对象的 ID
# -- item 可以是单个画布对象的 ID，也可以是某个 Tag

# focus(item=None)
# -- 将焦点移动到指定的 item
# -- 如果有多个画布对象匹配，则将焦点移动到显示列表中第一个可以接受光标输入的画布对象
# -- item 可以是单个画布对象的 ID，也可以是某个 Tag