from scipy.spatial import Delaunay
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import numpy as np
import torch

import numpy as np
import torch

data = torch.randn((1, 6, 2))


def triange2points(t):
    edges = torch.tensor([
        [t[0], t[1]],
        [t[1], t[0]],
        [t[0], t[2]],
        [t[2], t[0]],
        [t[1], t[2]],
        [t[2], t[1]],
    ])
    return edges


for graph in data:
    print(graph)
    tri = Delaunay(graph)

    edges = torch.empty(0)
    # convert triangle to edges:
    for t in tri.simplices:
        edges = torch.cat((edges, triange2points(t)), -2)

    edge_index = torch.tensor(edges)

    d = Data(x=graph, edge_index=edge_index.t().contiguous())
    print(d)