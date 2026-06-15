import numpy as np
# import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import csv
import igraph as ig
from scipy.io import mmread
# def adcy(fpath):
#     df = pd.read_csv("data/euro_mini.csv")
#     G = nx.from_pandas_edgelist(df, 'source', 'target', create_using=nx.Graph())
#     # ======================================plot fig==============================================
#     pos = nx.spring_layout(G)
#     nx.draw_networkx(G, pos, node_size=100, node_color="orange", edge_color="gray", with_labels=False)
#     plt.show()
#     # plt.savefig("")
#     # ===================================convert edgelist to adjacency matrix================================
#     E = G.edges()
#     # # Find the size of the matrix
#     size = max(max(E)) + 1
#     # # Create the empty adjacency matrix
#     adcy = [[0 for i in range(size)] for j in range(size)]
#     # # Fill in the matrix using the edgelist
#     for row, col in E:
#         adcy[row][col] = 1
#     return adcy


# ========================================================================================

# igraph python

def adj_matrix(fpath): ## read csv file

        df0 = pd.read_csv(fpath, header=None)
        df1 = df0[df0.columns[:2]]    # some files have weights other than node 1 and node 2
        df = df1[:960]
        g = ig.Graph.TupleList(df.itertuples(index=False), directed=False, weights=False)
        adj_ma = g.get_adjacency()
        adjacy = np.array(adj_ma.data)  # change the adj_ma to array datatype
        return adjacy

def adj_matrix2(fp): # read mtx file
        graph = ig.Graph.Read(fp, format="edgelist")   # generate a graph from a mtx file
        edgelist = graph.get_edgelist()  # get edgelist of the graph
        g = graph.subgraph_edges(edgelist[:885])  # get the list of a certain amount of edge
        adj_ma = g.get_adjacency()   # convert the subgraph to adjacency matrices
        adjacy = np.array(adj_ma.data)
        return adjacy

#
# fp = "./data/network_data/politician_edges.csv"
# g=adj_matrix(fp)
# # g = adj_matrix2(fp)
#
#
# #
# fig, ax = plt.subplots()
# # ig.plot(g,target=ax)
# ig.plot(g, main='g0')
# # ig.plot(g,"sample1.pdf")
# ig.plot(g,
#         target=ax,
#         # layout="circle",
#         vertex_frame_color="lightblue",
#         vertex_color="lightblue",
#         vertex_size=0.27,
#         edge_width=0.7,
#         edge_color="lightgray",
#         )
# plt.show()
# plt.savefig("./data/network_data/facebook.pdf")

