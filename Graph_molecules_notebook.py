import torch
from torch_geometric.datasets import TUDataset

dataset = TUDataset(root='data/TUDataset', name='MUTAG')

print()
print(f'Dataset: {dataset}:')
print('====================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

data = dataset[0]  # Get the first graph object.

print()
print(data)
print('=============================================================')

# Gather some statistics about the first graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')

#%%

torch.manual_seed(12345)
dataset = dataset.shuffle()

train_dataset = dataset[:150]
test_dataset = dataset[150:]

print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')

#%%

from torch_geometric.loader import DataLoader

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

for step, data in enumerate(train_loader):
    print(f'Step {step + 1}:')
    print('=======')
    print(f'Number of graphs in the current batch: {data.num_graphs}')
    print(data)
    print()

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
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.linear = Linear(hidden_channels, dataset.num_classes)

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
        x = self.conv3(x, edge_index)
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

model = GCN(num_node_features=7, hidden_channels=64)
print(model)

#%%

model = GCN(num_node_features=7, hidden_channels=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)#, weight_decay=5e-4)

# Use a suitable loss function for your problem
criterion = torch.nn.CrossEntropyLoss()


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
