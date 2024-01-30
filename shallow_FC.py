import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
import torch
from torch_geometric.nn import GCNConv, global_mean_pool, EdgeConv, MLP, SimpleConv



#%%

data_path = "C:\\Users\\andre\\Desktop\\WORKS\\Graph_Neural_Networks\\FC_Matthieu\\"

tot_corr = np.load(data_path + "FC.npy", allow_pickle=True)
label_sub = np.load(data_path + "lbl_sub.npy", allow_pickle=True)
label_task = np.load(data_path + "lbl_task.npy", allow_pickle=True)

tot_corr[np.isnan(tot_corr)] = 0

for i in range(0, tot_corr.shape[0]):
    np.fill_diagonal(tot_corr[i], 0)

#change the label_sub array to have classes from 0 to num_classes-1 without interruptions
label_sub = label_sub-1
# Get the unique values in the labels
unique_values = np.unique(label_sub)
# Create a dictionary that maps each unique value to an integer
value_to_int = {value: i for i, value in enumerate(unique_values)}
# Use the dictionary to replace the values in the original labels
new_labels_sub = np.array([value_to_int[value] for value in label_sub])

#%%

thresh = 0.7

dataset = []

for patient in range(0, tot_corr.shape[0]):
    print(f'----------------------------------', patient)

    means = []
    stds = []
    degrees = []

    single_corr = tot_corr[patient]
    single_corr_thresh = np.where(single_corr > thresh, single_corr, 0)

    matrix_df = pd.DataFrame(single_corr_thresh)
    links = matrix_df.stack().reset_index()
    links.columns = ['var1', 'var2', 'value']
    links_filtered = links.loc[links['value'] != 0]
    #map to have edges index in [0, #edge_attr] and not in [1,196]
    new_links_filtered = links_filtered.copy()
    unique_values = np.unique(links_filtered.var1)
    edge_map = {value: i for i, value in enumerate(unique_values)}
    new_links_filtered.var1 = [edge_map[value] for value in links_filtered.var1]
    new_links_filtered.var2 = [edge_map[value] for value in links_filtered.var2]

    weights = new_links_filtered.value
    edges = new_links_filtered.drop(labels="value", axis="columns")

    #nodes = np.zeros(shape=(len(pd.unique(links_filtered.var1)), 1))
    #nodes = np.array([])

    for node in np.unique(new_links_filtered.var1):

        node_mean = np.nanmean(new_links_filtered[new_links_filtered.var1 == node]['value'])
        node_std = np.nanstd(new_links_filtered[new_links_filtered.var1 == node]['value'])
        node_degree = len(new_links_filtered[new_links_filtered.var1 == node]['value'])

        means = np.append(means, node_mean)
        stds = np.append(stds, node_std)
        degrees = np.append(degrees, node_degree)

    nodes = np.column_stack((means, stds, degrees))

    nodes_all = torch.tensor(nodes, dtype=torch.float)
    edges_all = torch.tensor(edges.values.tolist(), dtype=torch.long).t().contiguous()
    weights_all = torch.tensor([list(weights)], dtype=torch.float).t().contiguous()
    label = torch.tensor([new_labels_sub[patient]], dtype=torch.float)

    data = Data(nodes_all, edges_all, weights_all, label)

    dataset.append(data)

#%%

data = dataset[3]

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
        #print(x)
        x = x.relu()
        #print(x)
        #x = global_mean_pool(x)
        return x

model = SmplConv()
print(model)

out = model(x=data.x, edge_index=data.edge_index, edge_weight=data.edge_attr)


