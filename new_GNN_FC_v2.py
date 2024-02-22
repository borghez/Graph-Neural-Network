import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from random import shuffle

from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch.optim.lr_scheduler import ExponentialLR


import networkx as nx
import matplotlib.pyplot as plt

import os

#%%

data_path = "C:\\Users\\andre\\Desktop\\WORKS\\Graph_Neural_Networks\\FC_Matthieu\\"

tot_corr = np.load(data_path + "FC.npy", allow_pickle=True)
label_sub = np.load(data_path + "lbl_sub.npy", allow_pickle=True)
label_task = np.load(data_path + "lbl_task.npy", allow_pickle=True)

tot_corr[np.isnan(tot_corr)] = 0

for i in range(0, tot_corr.shape[0]):
    np.fill_diagonal(tot_corr[i], 0)
    #np.nan_to_num(tot_corr[i], nan=0, posinf=0, neginf=0, copy=False)

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

for patient in range(0, tot_corr.shape[0]-400):
    print(f'----------------------------------', patient)

    means = []
    stds = []
    degrees = []

    single_corr = tot_corr[patient]
    single_corr_thresh = np.where(single_corr > thresh, single_corr, 0) #sigle_corr instead of 1

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

num_feat = dataset[0].x.shape[1]
#num_classes = len(np.unique(new_labels_sub))
num_classes = len(np.unique(new_labels_sub))

#%%

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        torch.manual_seed(1234567)
        super().__init__() #(GCN, self)
        self.conv1 = GCNConv(num_feat, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.conv5 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        #print(x)
        x = self.conv1(x, edge_index)
        #print(x)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        x = self.conv4(x, edge_index)
        x = self.conv5(x, edge_index)
        #print(x)
        #x = x.relu()
        #print(x)
        x = global_mean_pool(x, batch)

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x

model = GCN(hidden_channels=16)
print(model)

#%%

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)#, weight_decay=1e-5)
scheduler = ExponentialLR(optimizer, gamma=0.9)

loss_function = torch.nn.CrossEntropyLoss()
#criterion = F.nll_loss()

shuffle(dataset)

train_dataset = dataset[:280]   #70%
test_dataset = dataset[280:]    #30%

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

#%%

max_epochs = 50
val_interval = 5
epoch_loss_values = []
correct_test = 0
patience = 5
trigger_times = 0
last_loss=10000
eps = 0.1
go = True

torch.manual_seed(12345)

dir = "C:\\Users\\andre\\Desktop\\WORKS\\Graph_Neural_Networks"

# Training loop
for epoch in range(max_epochs):
    if go:
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        total_loss = 0
        correct_train = 0
        for data in train_loader:
            step += 1
            optimizer.zero_grad()
            outputs = model(x=data.x, edge_index=data.edge_index, batch=data.batch)
            loss = loss_function(outputs, torch.tensor(data.y, dtype=torch.long))   # Compute loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct_train += int((preds == torch.tensor(data.y, dtype=torch.int)).sum())

            print(f"{step}/{len(train_dataset) // train_loader.batch_size}, " f"train_loss: {loss.item():.4f}")

            #total_loss += loss.item()

            #pred = outputs.argmax(dim=1)
            #print(out)
            #print(data.y)
            #correct_train += int((pred == torch.tensor(data.y, dtype=torch.int)).sum())
        #print(f'Epoch: {epoch + 1}, Loss: {total_loss / len(train_loader)}')
        #print(f'Accuracy train: {correct_train / len(train_loader.dataset)}')
        scheduler.step()
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if np.abs(epoch_loss - last_loss) < eps:
            trigger_times += 1

            if trigger_times >= patience:
                torch.save(model.state_dict(), os.path.join(dir, "best_metric_model_FC.pth"))
                go = False
        else:
            trigger_times = 0

        last_loss = epoch_loss


# Testing

model.load_state_dict(torch.load(os.path.join(dir, "best_metric_model_FC.pth")))
model.eval()

for data_test in test_loader:
    out = model(x=data_test.x, edge_index=data_test.edge_index, batch=data_test.batch)
    pred = out.argmax(dim=1)
    correct_test += int((pred == torch.tensor(data_test.y, dtype=torch.int)).sum())
    #print(correct_test)
print(f'Accuracy test: {correct_test / len(test_loader.dataset)}')
