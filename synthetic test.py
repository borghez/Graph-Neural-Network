import numpy as np
import networkx as nx

import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit

import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, global_mean_pool, EdgeConv, MLP, SimpleConv
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import ExponentialLR

from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset

from random import shuffle

import os

#%%

'''Constructing a list of undirected graphs with low/high degree'''

#number of nodes
n = 30                      #30

# number of graph samples
n_sample = 200              #100

# list of graphs and labels
list_G = []
labels = []
list_adjs = np.zeros([n_sample, n, n])

for i in range(n_sample):
    # adjust parameters depending on class
    if i < n_sample / 2:
        l = 0
        p = 0.2             #0.2
    else:
        l = 1
        p = 0.3             #0.3
    G = nx.fast_gnp_random_graph(n, p, directed=False)
    list_G.append(G)
    adj = nx.to_numpy_matrix(G)
    list_adjs[i, :, :] = adj
    labels.append(l)

labels = np.array(labels)

#%%

mean_deg = []

for G in list_G:
    mean_deg.append(np.mean(nx.degree(G), axis=0)[1])

mean_deg = np.array(mean_deg)

plt.figure()
plt.hist(mean_deg[labels == 0], histtype='step', color='red')
plt.hist(mean_deg[labels == 1], histtype='step', color='blue')
plt.show()


#%%

'''Logistic Regression'''

vect_G = list_adjs.reshape([n_sample, -1])

mlr = LogisticRegression(C=1.0)

cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2)

acc = []
acc_shuf = []
for train_idx, test_idx in cv.split(vect_G, labels):
    # perf on data
    mlr.fit(vect_G[train_idx,:], labels[train_idx])
    acc.append(mlr.score(vect_G[test_idx,:], labels[test_idx]))
    # surrogate by shuffling
    shuf_idx = np.random.permutation(train_idx)
    mlr.fit(vect_G[train_idx,:], labels[shuf_idx])
    acc_shuf.append(mlr.score(vect_G[test_idx,:], labels[test_idx]))

print('mean(std) of acc:{} ({})'.format(np.mean(acc), np.std(acc)))
print('shuf acc:{} ({})'.format(np.mean(acc_shuf), np.std(acc_shuf)))



#%%

'''GCNConv'''

dataset = []

for i, G in enumerate(list_G):
    nodes_ones = torch.tensor([np.ones(len(G.nodes))], dtype=torch.float).t().contiguous()
    edges = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
    label = torch.tensor([labels[i]], dtype=torch.float)

    data = Data(x=nodes_ones, edge_index=edges, y=label)
    dataset.append(data)

#%%

count=0

for pat in range(0, len(list_G)):
    if labels[pat] != int(dataset[pat].y):
        print("Error!")
    else:
        count += 1

print(f"Labels correct: {count} out of {len(list_G)}")

#%%

from torch_geometric.utils import to_networkx

G = to_networkx(dataset[0], to_undirected=True)
plt.figure(figsize=(7, 7))
plt.xticks([])
plt.yticks([])
nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False, cmap="Set2")
plt.show()

G1 = to_networkx(dataset[51], to_undirected=True)
plt.figure(figsize=(7, 7))
plt.xticks([])
plt.yticks([])
nx.draw_networkx(G1, pos=nx.spring_layout(G, seed=42), with_labels=False, cmap="Set2")
plt.show()

#%%

num_feat = 1#dataset[0].x.shape[1]
num_classes = len(np.unique(labels))

#%%

#dataset = TUDataset(root='data/TUDataset', name='MUTAG')

#num_feat = dataset.num_features
#num_classes = dataset.num_classes

#%%

#class GCN(torch.nn.Module):
#    def __init__(self, hidden_channels):
#        torch.manual_seed(1234567)
#        super().__init__() #(GCN, self)
#        self.conv1 = SAGEConv(num_feat, hidden_channels)         #SAGEConv; GCNConv
#        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
#        #self.conv3 = SAGEConv(hidden_channels, hidden_channels)
#        self.batch1 = BatchNorm(hidden_channels)
#        #self.conv4 = SAGEConv(hidden_channels, hidden_channels)
#        #self.conv5 = SAGEConv(hidden_channels, hidden_channels)
#        self.batch2 = BatchNorm(hidden_channels)
#        self.lin = Linear(hidden_channels, num_classes)
#
#    def forward(self, x, edge_index, batch):
#        #print(x)
#        x = self.conv1(x, edge_index)
#        x = F.elu(x)
#        x = self.batch1(x)
#        x = F.dropout(x, p=0.05, training=self.training)
#        x = self.conv2(x, edge_index)
#        x = self.batch2(x)
#        #x = x.relu()
#        #x = self.conv3(x, edge_index)
#        #x = x.relu()
#        #x = self.conv4(x, edge_index)
#        #x = x.relu()
#        #x = self.conv5(x, edge_index)
#        #print(x)
#        #x = x.relu()
#        #print(x)
#        x = F.dropout(x, p=0.05, training=self.training)
#        x = global_mean_pool(x, batch)
#
#        x = F.dropout(x, p=0.05, training=self.training)
#        x = self.lin(x)
#        return x

class GCN(torch.nn.Module):                                     #GCN; GAT
    def __init__(self, hidden_channels):
        super(GCN, self).__init__() #(GCN, self)
        self.conv1 = GCNConv(num_feat, hidden_channels)         #SAGEConv; GCNConv
        self.lin1 = Linear(num_feat, hidden_channels)
        self.batch1 = BatchNorm(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.batch2 = BatchNorm(hidden_channels)
        self.lin3 = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        #print(x)
        x = self.conv1(x, edge_index) + self.lin1(x)
        x = F.relu(x)
        x = self.batch1(x)
        x = F.dropout(x, p=0.05, training=self.training)
        x = self.conv2(x, edge_index) + self.lin2(x)
        x = self.batch2(x)
        x = F.dropout(x, p=0.05, training=self.training)

        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.05, training=self.training)
        x = self.lin3(x)
        return x



#%%

#print(dataset)
#dataset = dataset.shuffle()
shuffle(dataset)
#print(dataset)

train_dataset = dataset[:140]   #70% :140
val_dataset = dataset[140:180]  #20% 140:180
test_dataset = dataset[180:]    #10% 180:

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

#%%

model = GCN(hidden_channels=64)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)#, weight_decay=1e-5)
#scheduler = ExponentialLR(optimizer, gamma=0.09)

loss_function = torch.nn.CrossEntropyLoss()  #se uso questa, le classi devono essere 0, 1
#loss_function = torch.nn.BCEWithLogitsLoss()  #se uso questa, le classi devono essere [0,1]=0, [1,0]=1
#criterion = F.nll_loss()

max_epochs = 150
val_interval = 5
epoch_loss_values = []
correct_test = 0
best_metric_epoch = -1
accuracies_val = []
patience = 5
trigger_times = 0
last_loss = 10000
eps = 0.01
go = True
best_metric = -1

dir = "C:\\Users\\andre\\Desktop\\WORKS\\Graph_Neural_Networks"

torch.manual_seed(1234567)

# Training loop
for epoch in range(max_epochs):
    if go:
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        step = 0
        epoch_loss = 0
        total_loss = 0
        correct_train = 0
        for train_data in train_loader:
            optimizer.zero_grad()
            outputs = model(x=train_data.x, edge_index=train_data.edge_index, batch=train_data.batch)
            loss = loss_function(outputs, torch.tensor(train_data.y, dtype=torch.long))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct_train += int((preds == torch.tensor(train_data.y, dtype=torch.int)).sum())

            print(f"{step}/{round(len(train_dataset) // train_loader.batch_size)}, " f"train_loss: {loss.item():.4f}")
            step += 1

            #total_loss += loss.item()

            #pred = outputs.argmax(dim=1)
            #print(out)
            #print(data.y)
            #correct_train += int((pred == torch.tensor(data.y, dtype=torch.int)).sum())
        #print(f'Epoch: {epoch + 1}, Loss: {total_loss / len(train_loader)}')
        #print(f'Accuracy train: {correct_train / len(train_loader.dataset)}')
        #scheduler.step()
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        #if np.abs(epoch_loss - last_loss) < eps or epoch_loss > last_loss:
        #    trigger_times += 1
#
        #    if trigger_times >= patience:
        #        torch.save(model.state_dict(), os.path.join(dir, "best_metric_model.pth"))
        #        go = False
        #else:
        #    trigger_times = 0
#
        #last_loss = epoch_loss

        if (epoch + 1) % val_interval == 0:
            model.eval()
            correct_val = 0
            with torch.no_grad():
                for val_data in val_loader:
                    out_val = model(x=val_data.x, edge_index=val_data.edge_index, batch=val_data.batch)
                    pred_val = out_val.argmax(dim=1)
                    correct_val += int((pred_val == torch.tensor(val_data.y, dtype=torch.int)).sum())
                    acc_val = correct_val / len(val_loader.dataset)

                accuracies_val.append(acc_val)
                if acc_val > best_metric:
                    best_metric = acc_val
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(dir, "best_metric_model_wee_synt.pth"))
                print("saved new best metric model")
                print(f"current epoch: {epoch + 1} current accuracy: {acc_val:.4f}"
                      f"\nbest accuracy: {best_metric:.4f} "
                      f"at epoch: {best_metric_epoch}")

        print(f"train completed, best_accuracy: {best_metric:.4f} " f"at epoch: {best_metric_epoch}")

# Testing
model.load_state_dict(torch.load(os.path.join(dir, "best_metric_model_wee_synt.pth")))
model.eval()

with torch.no_grad():
    for test_data in test_loader:
        out = model(x=test_data.x, edge_index=test_data.edge_index, batch=test_data.batch)
        pred = out.argmax(dim=1)
        correct_test += int((pred == torch.tensor(test_data.y, dtype=torch.int)).sum())
        #print(correct_test)
print(f'Accuracy test: {correct_test / len(test_loader.dataset)}')

#%%

def train():
    model.train()
    total_loss = 0

    for data in train_loader:  # Iterate in batches over the training dataset.
        out = model(x=data.x, edge_index=data.edge_index, batch=data.batch)    # Perform a single forward pass.
        loss = loss_function(out, torch.tensor(data.y, dtype=torch.long))  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
        total_loss += loss.item()

def test(loader):
    model.eval()

    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(x=data.x, edge_index=data.edge_index, batch=data.batch)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        print(pred)
        correct += int((pred == torch.tensor(data.y, dtype=torch.int)).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.


for epoch in range(1, 50):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')



