
from igraph import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# # Create graph

# fp = "./data/socfb-Cal65.mtx"
# fp = "./data/rt_gmanews.edges"

# df = pd.read_csv(fp, sep=',', engine='python', header=None)
# df1 = df.head()
# for arr in df1.values[:,:-1]:
#     print(arr)




def graphs(fpath):
    df = pd.read_csv(fpath, sep =',', engine='python', header=None) # for number type
    # df = pd.read_csv(fpath, sep='\s+', engine='python', header=None)  # for string type
    tpl = [(int(x), int(y)) for arr in df.values for [x, y] in [arr[0].split()]]   ## two columns in fb
    # tpl = df.values[:,:-1]    ## three columns in tw

    g = Graph.TupleList(tpl, directed=False)
    return g


def graphs_tmx(fpath):
    df = pd.read_csv(fpath, sep=',', engine='python', header=None)
    tpl = [(int(x), int(y)) for arr in df.values for [x, y] in [arr[0].split()]]   ## two columns in fb
    # tpl = df.values[:,:-1]    ## three columns in tw

    g = Graph.TupleList(tpl, directed=False)
    return g



# plt.savefig("./result/xxx.pdf")

# Obtaining information on vertices and edges of the graph

def my_degree(g):
    num_node = g.vcount()
    # num_edge = g.ecount()
    deg = g.degree()    # degrees of all nodes
    uni_deg = np.unique(deg)  # del duplicated elements

    k = []  # degree sequences
    pk = []  # degree distribution

    for i in range(len(uni_deg)):
        nk = 0
        # print("====")
        for j in range(len(deg)):
            if uni_deg[i] == deg[j]:   ## count the num of each degree
                nk = nk + 1
        # print(nk)
        # print(num_node)
        pk.append(np.round(nk/num_node,4))  ## get the degree distribution
        k.append(uni_deg[i])     ## assign unique degree to k list

    return np.array(k), np.array(pk)





# k, pk = my_degree(g)
# print(k, pk)

# Visualizing a graph
#
# g = graphs(fp)
# k,pk = my_degree(g)
# print(k, pk)
#
# fig, ax = plt.subplots()
# plot(g,
#         target=ax,
#         # layout="circle",
#         vertex_frame_color="lightblue",
#         vertex_color="lightblue",
#         vertex_size=0.5,
#         edge_width=1,
#         edge_color="lightgray"
#         )
# plt.show()
