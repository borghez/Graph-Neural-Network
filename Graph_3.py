import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


#%%

data_path = "C:\\Users\\andre\\Desktop\\WORKS\\Graph_Neural_Networks\\FC_Matthieu\\"

tot_corr = np.load(data_path + "FC.npy", allow_pickle=True)
label_sub = np.load(data_path + "lbl_sub.npy", allow_pickle=True)
label_task = np.load(data_path + "lbl_task.npy", allow_pickle=True)

tot_corr[np.isnan(tot_corr)] = 0

for i in range(0, tot_corr.shape[0]):
    np.fill_diagonal(tot_corr[i], 1)

#change the label_sub array to have classes from 0 to num_classes-1 without interruptions
label_sub = label_sub-1
# Get the unique values in the labels
unique_values = np.unique(label_sub)
# Create a dictionary that maps each unique value to an integer
value_to_int = {value: i for i, value in enumerate(unique_values)}
# Use the dictionary to replace the values in the original labels
new_labels_sub = np.array([value_to_int[value] for value in label_sub])


#%%

#corr_thresh = np.where(tot_corr > 0.5, tot_corr, 0)
corr_thresh = np.where(tot_corr > 0.5, 1, 0)


graphs = []  # This will store all the graphs
for i in range(corr_thresh.shape[0]):
    # Get the correlation matrix for the i-th patient
    correlation_matrix = corr_thresh[i]

    # Convert the numpy array into a pandas DataFrame
    matrix_df = pd.DataFrame(correlation_matrix)

    # Transform it into a links data frame (3 columns only):
    links = matrix_df.stack().reset_index()
    links.columns = ['var1', 'var2', 'value']

    links_filtered = links.loc[links['value'] != 0]

    # Build the graph
    G = nx.from_pandas_edgelist(links_filtered, source='var1', target='var2', edge_attr='value')#, create_using=nx.DiGraph)

    # Add the graph to the list of graphs
    graphs.append(G)

#%%

from torch_geometric.data import Data
import torch

mask = np.tri(corr_thresh.shape[1], corr_thresh.shape[2], 0, dtype=bool)
corr = corr_thresh[:, mask]

#%%

dataset = []
x = []
edge_attr = []
edge_index = []
y = []
cnt = -1

for G, edge_values in zip(graphs, corr):

    cnt +=1
    edge_index_ = list(G.edges)
    edge_index = torch.tensor(edge_index_, dtype=torch.long)
    edge_attr_ = edge_values[edge_values != 0]
    edge_attr = torch.tensor(edge_attr_, dtype=torch.float).view(-1, 1)
    x_mean = np.nanmean(tot_corr[cnt], axis=1)
    x_std = np.std(tot_corr[cnt], axis=1)
    x_degrees = np.array([val for (node, val) in G.degree()])
    x_tot = np.stack((x_degrees, x_mean, x_std), axis=-1)
    #x = torch.tensor(np.array(x, ndmin=2).T, dtype=torch.float)
    x = torch.tensor(x_tot, dtype=torch.float)
    y = torch.tensor([new_labels_sub[cnt]], dtype=torch.float)

    single_data = Data(x=x, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr, y=y)#, train_mask=train_array,
                   #test_mask=test_array, y=label)

    dataset.append(single_data)


#%%

#access to the first graph and print infos

print(dataset[0].keys())

print(dataset[0]['x'])

for key, item in dataset[0]:
    print(f'{key} found in data')

print('edge_attr' in dataset)

print(dataset[0].num_nodes)
print(dataset[0].num_edges)
print(dataset[0].num_node_features)
print(dataset[0].has_isolated_nodes())
print(dataset[0].has_self_loops())
print(dataset[0].is_undirected())


#%%

total_size = len(label_sub)

num_ones_train = int(0.7 * total_size)
num_ones_test = total_size - num_ones_train

# Create train and test arrays with zeros
train_array = np.zeros(total_size)
test_array = np.ones(total_size)

# Randomly choose indices for ones in train array
one_indices_train = np.random.choice(total_size, size=num_ones_train, replace=False)
train_array[one_indices_train] = 1
test_array[one_indices_train] = 0

#%%

train_dataset = [element for idx, element in enumerate(dataset) if train_array[idx] == 1]
test_dataset = [element for idx, element in enumerate(dataset) if test_array[idx] == 1]

#%%

import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, EdgeConv, MLP
from torch_geometric.loader import DataLoader

class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super().__init__() #(GCN, self)
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        #self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.linear = Linear(hidden_channels, len(np.unique(new_labels_sub)))

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        #print('1')
        #print(x)
        x = x.relu()
        x = self.conv2(x, edge_index)
        #print('2')
        #print(x)
        x = x.relu()
        #x = self.conv3(x, edge_index)
        #x = x.relu()

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # Assuming all nodes belong to a single graph
        #print('3')
        #print(x)
        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear(x)
        #print('4')
        #print(x)

        return x#F.log_softmax(x, dim=1)

model = GCN(num_node_features=3, hidden_channels=64)
print(model)

#%%

model = GCN(num_node_features=3, hidden_channels=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)#, weight_decay=5e-4)

# Use a suitable loss function for your problem
criterion = torch.nn.CrossEntropyLoss()

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Training loop
model.train()
for epoch in range(200):
    total_loss = 0
    correct_train = 0
    for data in train_loader:
        #print(data.x)
        #print('k')
        out = model(data.x, data.edge_index, data.batch)                # Forward pass
        #print('k')
        loss = criterion(out, torch.tensor(data.y, dtype=torch.long))   # Compute loss
        #print('k')
        loss.backward()                                                 # Backward pass
        optimizer.step()                                                # Update weights
        optimizer.zero_grad()                                           # Clear gradients

        total_loss += loss.item()

        pred = out.argmax(dim=1)
        #print(pred)
        #print(data.y)
        correct_train += int((pred == torch.tensor(data.y, dtype=torch.int)).sum())
    print(f'Epoch: {epoch + 1}, Loss: {total_loss / len(train_loader)}')
    print(f'Accuracy: {correct_train / len(train_loader.dataset)}')


# Testing
model.eval()
correct_test = 0
for data in test_loader:
    out = model(data.x, data.edge_index, data.batch)                        # Forward pass
    pred = out.argmax(dim=1)                                                # Get the predicted classes
    correct_test += int((pred == torch.tensor(data.y, dtype=torch.int)).sum())   # Compute the number of correct predictions
    print(correct_test)
print(f'Accuracy: {correct_test / len(test_loader.dataset)}')
