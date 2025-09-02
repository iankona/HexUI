

import win32com.client 

excel = None
sheet = None
def Excel():
    global excel, sheet
    try:
        excel = win32com.client.Dispatch('Excel.Application')
    except:
        excel = win32com.client.Dispatch('Ket.Application')
    sheet = excel.ActiveSheet


def Cells(row, col):
    return sheet.Cells(row, col).Value

def Rows(row):
    return sheet.Rows(row).Value

def Columns(col):
    return sheet.Columns(col).Value


def GetCell(row, col):
    return sheet.Cells(row, col).Value


def GetRow(row, col, colcount):
    result = [GetCell(row, i) for i in range(col, col+colcount)]
    return result


def GetCol(row, col, rowcount):
    result = [GetCell(i, col) for i in range(row, row+rowcount)]
    return result


def GetRegionRowDirect(row, col, rowcount, colcount):
    result = []
    for i in range(row, row+rowcount):
        valuelist = GetRow(i, col, colcount)
        result.append(valuelist)
    return result

def GetRegionColDirect(row, col, rowcount, colcount):
    result = []
    for i in range(col, col+colcount):
        valuelist = GetCol(row, i, rowcount)
        result.append(valuelist)
    return result



def SetCell(row, col, value):
    sheet.Cells(row, col).Value = value


def SetRow(row, col, valuelist):
    i = col
    for value in valuelist: 
        SetCell(row, i, value)
        i += 1


def SetCol(row, col, valuelist):
    i = row
    for value in valuelist: 
        SetCell(i, col, value)
        i += 1

# 行自增为行方向
def SetRegionRowDirect(row, col, regionlist):
    i = row
    for valuelist in regionlist:
        SetRow(i, col, valuelist)
        i += 1

# 列自增为列方向
def SetRegionColDirect(row, col, regionlist):
    i = col
    for valuelist in regionlist:
        SetCol(row, i, valuelist)
        i += 1



if __name__ == '__main__':
    Excel()

    print(GetCell(28, 1)) # None

    print(GetCell(1, 1))
    print(GetCell(1, 2))
    print(GetRow(1, 1, 3))
    print(GetCol(1, 3, 3))
    print(GetRegionRowDirect(1, 1, 2, 3))
    print(GetRegionColDirect(1, 1, 2, 3))


    SetCell(5, 1, "清波")
    SetCell(6, 1, "测试")
    SetRow(8, 1, [11, 12, 13])
    SetCol(9, 1, [21, 22, 23])
    SetRegionRowDirect(15, 1, [[11, 12], [21, 22], [31, 32]])
    SetRegionColDirect(20, 1, [[11, 12], [21, 22], [31, 32]])








# sheet = excel.ActiveWorkbook.ActiveSheet
# workbook = excel.Workbooks.Add()
# workbook = excel.Workbooks.Open('path/to/file.xlsx')
# worksheet = workbook.Worksheets('Sheet1')
# value = worksheet.Range('A1').Value
# worksheet = workbook.Worksheets('Sheet1')
# worksheet.Range('A1').Value = 'Hello, world!'


# worksheet = workbook.Worksheets('Sheet1')


# print(cells(1, 1))

# cells(5,1).Value = "白毛浮绿水"
# cells(5,1).Value = "白毛浮绿水"

# row = 1
# while worksheet.Cells(row, 1).Value is not None:
# value = worksheet.Cells(row, 1).Value
# print(value)
# row += 1
# 格式化和样式

# 设置单元格格式

# worksheet = workbook.Worksheets('Sheet1')
# range = worksheet.Range('A1:B2')
# range.NumberFormat = '0.00%'
# 设置字体样式

# worksheet = workbook.Worksheets('Sheet1')
# range = worksheet.Range('A1:B2')
# range.Font.Name = 'Arial'
# range.Font.Size = 12
# range.Font.Bold = True
# 设置边框和背景色

# worksheet = workbook.Worksheets('Sheet1')
# range = worksheet.Range('A1:B2')
# range.Borders.LineStyle = 1 # 设置边框样式为实线
# range.Borders.Weight = 2 # 设置边框粗细为2
# range.Interior.ColorIndex = 6 # 设置背景色为黄色
# 插入和删除

# 插入行、列和单元格

# worksheet = workbook.Worksheets('Sheet1')
# worksheet.Rows(1).Insert() # 插入行
# worksheet.Columns(1).Insert() # 插入列
# worksheet.Cells(1, 1).Insert() # 插入单元格
# 删除行、列和单元格

# worksheet = workbook.Worksheets('Sheet1')
# worksheet.Rows(1).Delete() # 删除行
# worksheet.Columns(1).Delete() # 删除列
# worksheet.Cells(1, 1).Delete() # 删除单元格
# 图表和图形

# 创建图表

# worksheet = workbook.Worksheets('Sheet1')
# chart = worksheet.Shapes.AddChart2().Chart
# 添加数据到图表

# worksheet = workbook.Worksheets('Sheet1')
# chart = worksheet.Shapes.AddChart2().Chart
# chart.SetSourceData(worksheet.Range('A1:B5'))
# 设置图表样式和布局

# worksheet = workbook.Worksheets('Sheet1')
# chart = worksheet.Shapes.AddChart2().Chart
# chart.ChartStyle = 1 # 设置图表样式为第一个样式
# chart.Layout = 4 # 设置图表布局为第四种布局
# 自动化操作

# 自动保存 Excel 文件

# workbook.Save()
# 自动关闭 Excel 应用程序

# excel.Quit()
# 批量处理 Excel 文件

# import os

# folder_path = 'path/to/folder'
# files = os.listdir(folder_path)
# for file in files:
# if file.endswith('.xlsx'):
# workbook = excel.Workbooks.Open(os.path.join(folder_path, file))
# # 进行操作
# workbook.Close()
# 通过使用 win32com 库，我们可以方便地实