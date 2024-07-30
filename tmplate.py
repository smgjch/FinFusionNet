import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# Function to create a static graph from the full dataset
def create_static_graph(data, threshold=0.5):
    num_samples, num_nodes, num_features = data.shape
    flattened_data = data.reshape(-1, num_features)
    corr_matrix = np.corrcoef(flattened_data.T)
    edge_index = []
    edge_attr = []
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if abs(corr_matrix[i, j]) > threshold:
                edge_index.extend([[i, j], [j, i]])
                edge_attr.extend([corr_matrix[i, j], corr_matrix[i, j]])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    return edge_index, edge_attr

# Function to create graph from a batch of data (for dynamic mode)
def create_batch_graphs(batch_data, threshold=0.5):
    batch_size, num_nodes, num_features = batch_data.shape
    batch_graphs = []

    for i in range(batch_size):
        data = batch_data[i]
        corr_matrix = np.corrcoef(data.T)
        edge_index = []
        edge_attr = []
        for j in range(num_nodes):
            for k in range(j+1, num_nodes):
                if abs(corr_matrix[j, k]) > threshold:
                    edge_index.extend([[j, k], [k, j]])
                    edge_attr.extend([corr_matrix[j, k], corr_matrix[j, k]])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        x = torch.tensor(data, dtype=torch.float)
        
        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        batch_graphs.append(graph)
    
    return Batch.from_data_list(batch_graphs)

# GCN Model
class GCNRegression(nn.Module):
    def __init__(self, num_features, hidden_channels):
        super(GCNRegression, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.linear = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = self.linear(x)
        return x

# Training loop
def train(model, dataloader, optimizer, criterion, device, static_graph=None, num_epochs=200):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_data, batch_labels in dataloader:
            optimizer.zero_grad()
            
            if static_graph is None:
                # Dynamic graph mode
                batch_graphs = create_batch_graphs(batch_data)
                batch_graphs = batch_graphs.to(device)
                out = model(batch_graphs.x, batch_graphs.edge_index, batch_graphs.edge_attr, batch_graphs.batch)
            else:
                # Static graph mode
                x = batch_data.view(-1, batch_data.size(-1)).to(device)
                edge_index, edge_attr = static_graph
                edge_index, edge_attr = edge_index.to(device), edge_attr.to(device)
                batch = torch.repeat_interleave(torch.arange(batch_data.size(0)), batch_data.size(1)).to(device)
                out = model(x, edge_index, edge_attr, batch)
            
            batch_labels = batch_labels.view(-1, 1).to(device)
            loss = criterion(out, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}')

# Custom Dataset
class DynamicGraphDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float)
        self.labels = torch.tensor(labels, dtype=torch.float)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Main execution
if __name__ == "__main__":
    # Assuming you have your data as a numpy array of shape (N, h, w)
    # and labels as a numpy array of shape (N,)
    # N: number of samples, h: number of features, w: number of nodes
    data = np.random.randn(1000, 10, 20)  # Example data
    labels = np.random.randn(1000)  # Example labels

    # Create dataset and dataloader
    dataset = DynamicGraphDataset(data, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCNRegression(num_features=data.shape[1], hidden_channels=64).to(device)
    
    # Setup optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    # Create static graph (optional)
    use_static_graph = True  # Set to False for dynamic graph mode
    static_graph = create_static_graph(data) if use_static_graph else None
    
    # Train the model
    train(model, dataloader, optimizer, criterion, device, static_graph)
    
    print("Training completed!")

