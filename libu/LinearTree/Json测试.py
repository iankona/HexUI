import json
import math
import random
import itertools

import networkx as nx
import matplotlib
matplotlib.rc("font", family='Microsoft YaHei')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = 'SimHei'
# SimSun :宋体；KaiTI：楷体；Microsoft YaHei：微软雅黑LiSu：隶书；FangSong：仿宋；Apple LiGothic Medium：苹果丽中黑；
 

# 从JSON文件中读取数据
itemdict = {}
with open("item.json", 'r', encoding='utf-8') as f:
    data_loaded = json.load(f)
    # print(data_loaded)
    for valuedict in data_loaded['dataArray']:
        id = valuedict["ID"]
        name = valuedict["Name"]
        # print(id, name)
        itemdict[id] = name


recipelist = []
recipedict = {}
with open("recipe.json", 'r', encoding='utf-8') as f:
    data_loaded = json.load(f)
    
    for valuedict in data_loaded['dataArray']:
        name = valuedict["Name"]
        timespend = valuedict["TimeSpend"]
        items = valuedict["Items"]
        itemnames = [itemdict[id] for id in items]
        itemcounts = valuedict["ItemCounts"]
        results = valuedict["Results"]
        resultnames = [itemdict[id] for id in results]
        resultcounts = valuedict["ResultCounts"]
        recipelist.append([name, timespend, itemnames, itemcounts, resultnames, resultcounts])



def hierarchy_pos_beautiful(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    '''
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.
    Licensed under Creative Commons Attribution-Share Alike

    If the graph is a tree this will return the positions to plot this in a
    hierarchical layout.

    G: the graph (must be a tree)

    root: the root node of current branch
    - if the tree is directed and this is not given,
      the root will be found and used
    - if the tree is directed and this is given, then
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given,
      then a random choice will be used.

    width: horizontal space allocated for this branch - avoids overlap with other branches

    vert_gap: gap between levels of hierarchy

    vert_loc: vertical location of root

    xcenter: horizontal location of root
    '''
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  # allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''

        if pos is None:
            # pos = nx.spring_layout(G)
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)
        if len(children) != 0:
            dx = width / len(children)
            nextx = xcenter - width / 2 - dx / 2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap,
                                     vert_loc=vert_loc - vert_gap, xcenter=nextx,
                                     pos=pos, parent=root)
        return pos

    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)






# 全链接
edges = []
for name, timespend, itemnames, itemcounts, resultnames, resultcounts in recipelist:
    name = "配方" + name
    for iname in itemnames:  edges.append([name, iname])
    for rname in resultnames:edges.append([rname, name])

# # 树形连接
# edges = []
# for name, timespend, itemnames, itemcounts, resultnames, resultcounts in recipelist:
#     rname = resultnames[0]
#     # print(rname)
#     for iname in itemnames:  edges.append([rname, iname])


G=nx.Graph()
G.add_edges_from(edges)
pos = nx.spring_layout(G) # 增加节点间距
# # 计算中心度
# deg_centrality = nx.degree_centrality(G)
# # print(deg_centrality)

# # 对于有向图，入度和出度的度量是不同的，计算无向图会出错
# # in_deg_centrality = nx.in_degree_centrality(G)
# # out_deg_centrality = nx.out_degree_centrality(G)

# # 计算接近中心度
# close_centrality = nx.closeness_centrality(G)

pass
# 寻找起点与终点之间的路径
# for path in nx.all_simple_paths(G, '宇宙矩阵', '铁矿'): # 寻找路径
#     print(path)



# 获取节点1的所有相连边
for node in G.nodes:
    links = G.edges(node)
    if len(links) <= 1: pass
        # print(links)
links = G.edges("铁矿")
print(links)

node = G.nodes["铁矿"]


degree = nx.degree(G)
pass


def remove_edges(G, node=0):
    desc = nx.descendants(G, node) # 子集 
    anc = nx.ancestors(G, node) # 父级
    # 对于循环图，父级和子集重复了，
    paths = []
    for p in desc:
        for path in nx.all_simple_paths(G, node, p): paths.append(path)

    for p in anc:
        for path in nx.all_simple_paths(G, p, node): paths.append(path)

    Q = nx.Graph()
    for p in paths:
        nx.add_path(Q, p)
    return Q 

q = remove_edges(G, node="宇宙矩阵")
qpos = nx.spring_layout(q)
nx.draw(q, pos=qpos, with_labels=True)
plt.show()
pass
# # 寻找树
# T = nx.minimum_spanning_tree(G,algorithm='kruskal')  
# # 提取字典数据(边:权重)
# w = nx.get_edge_attributes(T,'weight')
# # 计算最小生成树的长度（边对应权值相加）
# TL = sum(w.values())  
# print("最小生成树为:",w)
# print("最小生成树的长度为：",TL)
# # 最小生成树为: {(1, 5): 2, (2, 3): 4, (3, 5): 1, (3, 4): 2}
# # 最小生成树的长度为： 9
# # 绘图
# nx.draw(T,pos,with_labels=True)
# # # 绘制边的权重
# # nx.draw_networkx_edge_labels(T,pos,font_size=12,edge_labels=w)
# # # 绘制边，改变颜色
# # nx.draw_networkx_edges(T,pos,edgelist=w.keys(),
# #              edge_color='r',width=3)
# plt.show()



# pos = nx.spring_layout(G, k=1.0) # 增加节点间距

# 在networkx中，spring_layout布局的参数k控制了节点间的平衡距离。
# 默认情况下，k的值为1/sqrt(n)，其中n为节点的数量。
# 也就是说，默认节点间距与节点数量有关，节点越多，节点间距越小。
# pos = hierarchy_pos_beautiful(G,170) # not a tree
nx.draw(G, pos=pos, with_labels=True)
plt.savefig('hierarchy.png')


# p.write_png('example.png')
plt.show()


[('铜矿', '配方铜块')]
# [('风力涡轮机', '配方风力涡轮机')]
# [('原油萃取站', '配方原油萃取站')]
# [('原油精炼厂', '配方原油精炼厂')]
[('原油', '配方等离子精炼')]
# [('氢燃料棒', '配方氢燃料棒')]
[('可燃冰', '配方石墨烯（高效）')]
[('刺笋结晶', '配方碳纳米管（高效）')]
# [('微型粒子对撞机', '配方微型粒子对撞机')]
# [('人造恒星', '配方人造恒星')]
# [('采矿机', '配方采矿机')]
# [('抽水站', '配方抽水站')]
# [('木材', '配方有机晶体（原始）')]
# [('植物燃料', '配方有机晶体（原始）')]
# [('金伯利矿石', '配方金刚石（高效）')]
# [('分形硅石', '配方晶格硅（高效）')]
# [('火力发电厂', '配方火力发电厂')]
[('钛石', '配方钛块')]
# [('太阳能板', '配方太阳能板')]
# [('电磁轨道弹射器', '配方电磁轨道弹射器')]
# [('射线接收站', '配方射线接收站')]
# [('卫星配电站', '配方卫星配电站')]
# [('临界光子', '配方光子物质化')]
# [('宇宙矩阵', '配方宇宙矩阵')]
# [('蓄电器', '配方蓄电器')]
# [('能量枢纽', '配方能量枢纽')]
# [('垂直发射井', '配方垂直发射井')]
# [('小型运载火箭', '配方小型运载火箭')]
# [('小型储物仓', '配方小型储物仓')]
# [('四向分流器', '配方四向分流器')]
# [('大型储物仓', '配方大型储物仓')]
# [('极速传送带', '配方极速传送带')]
# [('物流运输机', '配方物流运输机')]
# [('星际物流运输船', '配方星际物流运输船')]
# [('增产剂 Mk.III', '配方增产剂 Mk.III')]
# [('喷涂机', '配方喷涂机')]
# [('分馏塔', '配方分馏塔')]
# [('蓄电器（满）', '配方轨道采集器')]
# [('轨道采集器', '配方轨道采集器')]
# [('地基', '配方地基')]
# [('微型聚变发电站', '配方微型聚变发电站')]
# [('储液罐', '配方储液罐')]
# [('流速器', '配方流速器')]
# [('地热发电站', '配方地热发电站')]
# [('大型采矿机', '配方大型采矿机')]
# [('自动集装机', '配方自动集装机')]
# [('物流配送器', '配方物流配送器')]
# [('配送运输机', '配方配送运输机')]
# [('化工厂 Mk.II', '配方化工厂 Mk.II')]
# [('高斯机枪塔', '配方高斯机枪塔')]
# [('高频激光塔', '配方高频激光塔')]
# [('聚爆加农炮', '配方聚爆加农炮')]
# [('磁化电浆炮', '配方磁化电浆炮')]
# [('导弹防御塔', '配方导弹防御塔')]
# [('干扰塔', '配方干扰塔')]
# [('信标', '配方信标')]
# [('护盾发生器', '配方护盾发生器')]
# [('超合金弹箱', '配方超合金弹箱')]
# [('晶石炮弹组', '配方晶石炮弹组')]
# [('反物质胶囊', '配方反物质胶囊')]
# [('引力导弹组', '配方引力导弹组')]
# [('地面战斗机-A型', '配方地面战斗机-A型')]
# [('地面战斗机-F型', '配方地面战斗机-F型')]
# [('太空战斗机-A型', '配方太空战斗机-A型')]
# [('太空战斗机-F型', '配方太空战斗机-F型')]
# [('战场分析基站', '配方战场分析基站')]
# [('硅基神经元', '配方矩阵研究站 Mk.II')]
# [('存储单元', '配方矩阵研究站 Mk.II')]
# [('矩阵研究站 Mk.II', '配方矩阵研究站 Mk.II')]
# [('物质重组器', '配方制造台 Mk.IV')]
# [('制造台 Mk.IV', '配方制造台 Mk.IV')]
# [('负熵奇点', '配方熔炉 Mk.III')]
# [('熔炉 Mk.III', '配方熔炉 Mk.III')]
# [('虚粒子', '配方金色燃料棒')]
# [('金色燃料棒', '配方金色燃料棒')]
# [('地面电浆炮', '配方地面电浆炮')]
# [('电磁压制胶囊', '配方电磁压制胶囊')]
# [('集装分拣器', '配方集装分拣器')]