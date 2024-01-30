import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

DG = nx.DiGraph(weight=1)

DG.add_nodes_from([
    (0, {"weight": 1.0}),
    (1, {"weight": 1.0}),
    (2, {"weight": 1.0}),
    (3, {"weight": 1.0})
])

DG.add_weighted_edges_from([
    (0, 1, 1.0),
    (1, 2, 2.0),
    (2, 3, 3.0),
    (3, 0, 4.0),
    (3, 1, 5.0)
])


nx.draw(DG, with_labels=True, node_color='orange', node_size=400, edge_color='black', linewidths=1, font_size=15)
plt.show()


#%%

'''SIMPLE CONVOLUTION'''

from torch_geometric.data import Data
import torch


#x = torch.tensor(list(DG.nodes), dtype=torch.float)
nodes = torch.tensor([list(dict(DG.nodes(data="weight", default=1)).values())], dtype=torch.float).t().contiguous()
edges = torch.tensor(list(DG.edges), dtype=torch.long).t().contiguous()
weights = torch.tensor([list(nx.get_edge_attributes(DG, 'weight').values())], dtype=torch.float).t().contiguous()

data = Data(nodes, edges, weights)

from torch_geometric.nn import GCNConv, global_mean_pool, EdgeConv, MLP, SimpleConv

class SmplConv(torch.nn.Module):
    def __init__(self):
        torch.manual_seed(1234567)
        super().__init__() #(GCN, self)
        self.conv1 = SimpleConv(aggr="sum")

    def forward(self, x, edge_index, edge_weight):
        print(x)
        x = self.conv1(x, edge_index, edge_weight)
        print(x)
        x = self.conv1(x, edge_index, edge_weight)
        print(x)
        x = x.relu()
        print(x)
        #x = global_mean_pool(x)
        return x

model = SmplConv()
print(model)

out = model(x=data.x, edge_index=data.edge_index, edge_weight=data.edge_attr)


#%%

'''GCNConv'''

import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, EdgeConv, MLP, SimpleConv
from torch_geometric.loader import DataLoader

from torch_geometric.data import Data


#x = torch.tensor(list(DG.nodes), dtype=torch.float)
nodes = torch.tensor([list(dict(DG.nodes(data="weight", default=1)).values())], dtype=torch.float).t().contiguous()
edges = torch.tensor(list(DG.edges), dtype=torch.long).t().contiguous()
weights = torch.tensor([list(nx.get_edge_attributes(DG, 'weight').values())], dtype=torch.float).t().contiguous()

data = Data(nodes, edges, weights)

class GCN(torch.nn.Module):
    def __init__(self):
        torch.manual_seed(1234567)
        super().__init__() #(GCN, self)
        self.conv1 = GCNConv(1, 1)

    def forward(self, x, edge_index, edge_weight):
        print(x)
        x = self.conv1(x, edge_index, edge_weight)
        print(x)
        x = x.relu()
        print(x)
        #x = global_mean_pool(x)
        return x

model = GCN()
print(model)

out = model(data.x, data.edge_index, data.edge_attr)
