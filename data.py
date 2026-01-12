import networkx as nx
import numpy as np

def generate_graph(n=100):
    G = nx.erdos_renyi_graph(n, 0.05)
    for u,v in G.edges():
        G[u][v]['amount'] = np.random.rand()
    labels = {i: np.random.randint(0,2) for i in G.nodes()}
    return G, labels
