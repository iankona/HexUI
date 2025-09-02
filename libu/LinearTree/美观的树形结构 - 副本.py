
# import networkx as nx
# import matplotlib.pyplot as plt

# # 创建一个空的无向图
# G = nx.Graph()

# # 添加节点
# G.add_node("A")
# G.add_node("B")
# G.add_node("C")

# # 添加边
# G.add_edge("A", "B")
# G.add_edge("B", "C")
# G.add_edge("C", "A")

# # 计算节点的度中心性
# centrality = nx.degree_centrality(G)
# print("度中心性:", centrality)

# # 绘制网络图
# nx.draw(G, with_labels=True, node_color='skyblue', node_size=700, edge_color='k')
# plt.show()

import math
import random

import networkx as nx
import matplotlib.pyplot as plt
 
 
def my_grap():
    G = nx.Graph(my_seq='first_graph')
    a=G.add_node('a', my_index='NO-1')
    G.add_nodes_from(['b', 'c'], my_index='NO-2/3')
    G.add_nodes_from(['d', 'e', 'f'])
    nx.add_cycle(G, ['d', 'e', 'f'])
 
    G.add_edge('c', 'd', weight=3)
    G.add_edges_from([('a', 'b', {'weight': 1}),
                      ('a', 'd', {'weight': 2}),
                      ('d', 'e', {'weight': 2}),
                      ('d', 'f', {'weight': 2}),
                      ('e', 'f', {'weight': 2})],
                     my_len=3)
 
    G.nodes['e']['my_index'] = 'NO-4'
    return G
#     print('所有节点', G.nodes(data=True))
#     print('所有边', G.edges(data=True))
#     print('节点数', G.number_of_nodes())
 
#     # 打印所有边
#     for u, v, w in G.edges(data='weight'):
#         print((u, v, w))
 
#     pos = nx.spring_layout(G)
#     nx.draw(G, pos,
#             node_color='red',
#             node_size=300,
#             font_size=10,
#             font_color='blue',
#             with_labels=True)
 
#     weights = nx.get_edge_attributes(G, 'weight')
#     nx.draw_networkx_edge_labels(G, pos, edge_labels=weights)
#     plt.show()
 
 


# my_grap()



# Beam Search
# 动态束宽的束搜索算法。

# 渐进式扩展束搜索通过不断增加束宽执行束搜索，直到找到目标节点。
# import math

# import matplotlib.pyplot as plt
# import networkx as nx

# # 定义一个渐进式扩展束搜索函数
# def progressive_widening_search(G, source, value, condition, initial_width=1):
#     """
#     渐进式扩展束搜索以找到一个节点。

#     渐进式扩展束搜索涉及重复的束搜索，从一个小的束宽开始，如果没有找到目标节点，则逐步扩大束宽。
#     这个实现简单地返回第一个找到的符合终止条件的节点。

#     `G` 是一个 NetworkX 图。

#     `source` 是图中的一个节点。从这里开始搜索感兴趣的节点，并且只扩展到这个节点的（弱）连通分量中的节点。

#     `value` 是一个函数，返回一个实数，表示在决定哪些邻居节点加入到宽度优先搜索队列时，一个潜在邻居节点的好坏。
#     在每一步中，只有当前束宽内最好的节点会被加入队列。

#     `condition` 是搜索的终止条件。这是一个函数，接受一个节点作为输入并返回一个布尔值，指示该节点是否为目标。
#     如果没有节点匹配终止条件，这个函数会抛出 :exc:`NodeNotFound`。

#     `initial_width` 是束搜索开始的束宽（默认为1）。如果在这个束宽下没有找到匹配`condition`的节点，
#     则从`source`节点重新开始束搜索，束宽增加一倍（因此束宽呈指数增长）。当束宽超过图中的节点数时，搜索终止。
#     """
#     # 检查一个特殊情况，即源节点满足终止条件。
#     if condition(source):
#         return source
#     # 在这个范围内，`i`的最大可能值产生的宽度至少等于图中的节点数，所以最后一次调用
#     # `bfs_beam_edges`相当于普通的宽度优先搜索。因此，最终所有节点都会被访问。
#     log_m = math.ceil(math.log2(len(G)))
#     for i in range(log_m):
#         width = initial_width * pow(2, i)
#         # 由于我们总是从同一个源节点开始搜索，这个搜索可能会多次访问相同的节点（取决于
#         # `value`函数的实现）。
#         for u, v in nx.bfs_beam_edges(G, source, value, width):
#             if condition(v):
#                 return v
#     # 此时，由于所有节点都已被访问，我们知道没有节点满足终止条件。
#     raise nx.NodeNotFound("no node satisfied the termination condition")

# # 寻找具有高中心性的节点。
# # 我们生成一个随机图，计算每个节点的中心性，然后执行渐进式扩展搜索以找到一个具有高中心性的节点。

# # 设置随机数生成的种子，以便示例可以复现
# seed = 89

# G = nx.gnp_random_graph(100, 0.5, seed=seed)
# centrality = nx.eigenvector_centrality(G)
# avg_centrality = sum(centrality.values()) / len(G)

# # 定义一个函数，判断一个节点是否具有高中心性
# def has_high_centrality(v):
#     return centrality[v] >= avg_centrality

# source = 0
# value = centrality.get
# condition = has_high_centrality

# found_node = progressive_widening_search(G, source, value, condition)
# c = centrality[found_node]
# print(f"found node {found_node} with centrality {c}")

# # 绘制图形
# pos = nx.spring_layout(G, seed=seed)
# options = {
#     "node_color": "blue",
#     "node_size": 20,
#     "edge_color": "grey",
#     "linewidths": 0,
#     "width": 0.1,
# }
# nx.draw(G, pos, **options)
# # 以大的红色节点绘制具有高中心性的节点
# nx.draw_networkx_nodes(G, pos, nodelist=[found_node], node_size=100, node_color="r")
# plt.show()






def hierarchy_pos_ugly(G, root, levels=None, width=1., height=1.):
    """If there is a cycle that is reachable from root, then this will see infinite recursion.
       G: the graph
       root: the root node
       levels: a dictionary
               key: level number (starting from 0)
               value: number of nodes in this level
       width: horizontal space allocated for drawing
       height: vertical space allocated for drawing"""
    TOTAL = "total"
    CURRENT = "current"

    def make_levels(levels, node=root, currentLevel=0, parent=None):
        """Compute the number of nodes for each level
        """
        if not currentLevel in levels:
            levels[currentLevel] = {TOTAL: 0, CURRENT: 0}
        levels[currentLevel][TOTAL] += 1
        neighbors = G.neighbors(node)
        for neighbor in neighbors:
            if not neighbor == parent:
                levels = make_levels(levels, neighbor, currentLevel + 1, node)
        return levels

    def make_pos(pos, node=root, currentLevel=0, parent=None, vert_loc=0):
        dx = 1 / levels[currentLevel][TOTAL]
        left = dx / 2
        pos[node] = ((left + dx * levels[currentLevel][CURRENT]) * width, vert_loc)
        levels[currentLevel][CURRENT] += 1
        neighbors = G.neighbors(node)
        for neighbor in neighbors:
            if not neighbor == parent:
                pos = make_pos(pos, neighbor, currentLevel + 1, node, vert_loc - vert_gap)
        return pos

    if levels is None:
        levels = make_levels({})
    else:
        levels = {l: {TOTAL: levels[l], CURRENT: 0} for l in levels}
    vert_gap = height / (max([l for l in levels]) + 1)
    return make_pos({})


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

# g = my_grap()
# pos = hierarchy_pos_beautiful(g, g.nodes["a"]) # //g是networkx的图，root是根节点
# node_labels = nx.get_node_attributes(g,"attr")
# nx.draw(g, pos, with_labels=True, labels=node_labels)
# plt.show()



import networkx as nx
import random

    
def hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):

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
            root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''
    
        if pos is None:
            pos = {root:(xcenter,vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)  
        if len(children)!=0:
            dx = width/len(children) 
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G,child, width = dx, vert_gap = vert_gap, 
                                    vert_loc = vert_loc-vert_gap, xcenter=nextx,
                                    pos=pos, parent = root)
        return pos

            
    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)



import matplotlib.pyplot as plt
import networkx as nx
G=nx.Graph()
G.add_edges_from([(1,2), (1,3), (1,4), (2,5), (2,6), (2,7), (3,8), (3,9), (4,10),
                  (5,11), (5,12), (6,13)])

# 单方向图
pos = hierarchy_pos(G,1)  

# 弹簧图
pos = nx.spring_layout(G)

# 单向图转圆向图
pos = hierarchy_pos(G, 1, width = 2*math.pi, xcenter=1)
new_pos = {u:(r*math.cos(theta),r*math.sin(theta)) for u, (theta, r) in pos.items()}
nx.draw(G, pos=new_pos, node_size = 50)
nx.draw_networkx_nodes(G, pos=new_pos, nodelist = [1], node_color = 'blue', node_size = 200)


# nx.draw(G, pos=pos, with_labels=True)
# plt.savefig('hierarchy.png')
# plt.show()

# 斜弯图
def hierarchy_pos(G, root, levels=None, width=1., height=1.):
    '''If there is a cycle that is reachable from root, then this will see infinite recursion.
       G: the graph
       root: the root node
       levels: a dictionary
               key: level number (starting from 0)
               value: number of nodes in this level
       width: horizontal space allocated for drawing
       height: vertical space allocated for drawing'''
    TOTAL = "total"
    CURRENT = "current"
    def make_levels(levels, node=root, currentLevel=0, parent=None):
        """Compute the number of nodes for each level
        """
        if not currentLevel in levels:
            levels[currentLevel] = {TOTAL : 0, CURRENT : 0}
        levels[currentLevel][TOTAL] += 1
        neighbors = G.neighbors(node)
        for neighbor in neighbors:
            if not neighbor == parent:
                levels =  make_levels(levels, neighbor, currentLevel + 1, node)
        return levels

    def make_pos(pos, node=root, currentLevel=0, parent=None, vert_loc=0):
        dx = 1/levels[currentLevel][TOTAL]
        left = dx/2
        pos[node] = ((left + dx*levels[currentLevel][CURRENT])*width, vert_loc)
        levels[currentLevel][CURRENT] += 1
        neighbors = G.neighbors(node)
        for neighbor in neighbors:
            if not neighbor == parent:
                pos = make_pos(pos, neighbor, currentLevel + 1, node, vert_loc-vert_gap)
        return pos
    if levels is None:
        levels = make_levels({})
    else:
        levels = {l:{TOTAL: levels[l], CURRENT:0} for l in levels}
    vert_gap = height / (max([l for l in levels])+1)
    return make_pos({})



import networkx as nx
G=nx.Graph()
G.add_edges_from([(1,2), (1,3), (1,4), (2,5), (2,6), (2,7), (3,8), (3,9), (4,10),
                  (5,11), (5,12), (6,13)])
pos = hierarchy_pos(G,1)    
nx.draw(G, pos=pos, with_labels=True)
plt.savefig('hierarchy.png')


# p.write_png('example.png')
plt.show()