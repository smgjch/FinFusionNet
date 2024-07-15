import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F

pred_len = 10
fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)
# %%
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx] , self.labels[idx]


# %%
data = pd.read_csv('dataset/btc/btc_t_v_withf.csv')
data = data.apply(pd.to_numeric, errors='coerce').values

# Normalize features


data = data[:, 2:]  # First column as labels

# Split data
df_transform, df_tmp = train_test_split(data, test_size=0.4, shuffle= False)


scaler = StandardScaler()
scaler.fit(df_transform)


df_train, df_temp = train_test_split(df_transform,test_size=0.4, shuffle= False)
df_val, df_test = train_test_split(df_temp, test_size=0.16, shuffle= False)

df_train = scaler.transform(df_train)
df_val = scaler.transform(df_val)
df_test = scaler.transform(df_test)



X_train, y_train = df_train[:,1: ], df_train[:, 0]  
X_val, y_val = df_val[:,1: ], df_val[:, 0]  
X_test, y_test = df_test[:,1: ], df_test[:, 0]  
# print(f"before slice {X_train.shape}, {X_val.shape}, {X_test.shape}")

num_samples = (len(y_train) // pred_len) * pred_len
y_train = y_train[:num_samples]
X_train = X_train[:num_samples]

num_samples = (len(y_val) // pred_len) * pred_len
y_val = y_val[:num_samples]
X_val = X_val[:num_samples]

num_samples = (len(y_test) // pred_len) * pred_len
X_test = X_test[:num_samples]
y_test = y_test[:num_samples]

X_train, X_val, X_test = X_train.reshape(-1, pred_len, 138), X_val.reshape(-1, pred_len, 138), X_test.reshape(-1, pred_len, 138)
y_train, y_val, y_test = y_train.reshape(-1, pred_len, 1), y_val.reshape(-1, pred_len, 1), y_test.reshape(-1, pred_len, 1)

train_dataset = CustomDataset(X_train, y_train)
val_dataset = CustomDataset(X_val, y_val)
test_dataset = CustomDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)



# %%

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.verbose = configs.verbose
        self.input_window_size = configs.seq_len
        self.kernel_size = configs.kernel_size
        self.num_filters = configs.num_kernels
        self.pred_len = configs.label_len

        self.conv1_layers = nn.Conv1d(in_channels = 1, out_channels = self.num_filters, 
                                      kernel_size = self.kernel_size, dilation=1,padding =0)
        
        self.conv2_layers = nn.Conv1d(in_channels = 1, out_channels = self.num_filters, 
                                      kernel_size = self.kernel_size, dilation=2,padding = 0)
        
        self.conv3_layers = nn.Conv1d(in_channels = 1, out_channels = self.num_filters, 
                                      kernel_size = self.kernel_size, dilation=4,padding = 0)

        self.dense1 = nn.Linear(2442 , 4096 * 4)
        self.dense2 = nn.Linear(4096 * 4, 1024 * 4)
        self.dense3 = nn.Linear(1024 * 4, 1024 * 4)
        self.dense4 = nn.Linear(1024 * 4, 64)
        self.output_layer = nn.Linear(64, self.pred_len)
        self.output_layerf = nn.Linear(16, 1)

    def process_feature(self, input_feature):
        conv1 = F.relu(self.conv1_layers(input_feature))
        conv2 = F.relu(self.conv2_layers(input_feature))
        conv3 = F.relu(self.conv3_layers(input_feature))
        # print(f"shapes {conv1.shape} {conv2.shape} {conv3.shape}")
        concatenated = torch.cat((conv1, conv2, conv3), dim=2)

        return concatenated

    def forward(self, inputs,_x,y,_y):
        # Split the input into separate features
        features = [inputs[:, i:i+1, :] for i in range(6)]

        # Process each feature separately
        processed_features = [self.process_feature(feature) for feature in features]

        concatenated_features = torch.cat(processed_features, dim=2)

        x = F.relu(self.dense1(concatenated_features))
        x = F.relu(self.dense2(x))
        x = F.relu(self.dense3(x))
        x = F.relu(self.dense4(x))

        # Output layer
        output = self.output_layer(x)
        output = output.permute(0,2,1)
        # print(f"output {output.shape}")
        output = self.output_layerf(output)
        # print(f"outputf {output.shape}")
        
        return output


# %%

# Configuration class
class Configs:
    def __init__(self):
        self.verbose = True
        self.seq_len = 96
        self.kernel_size = 2
        self.num_kernels = 16
        self.label_len = 10

configs = Configs()
model = Model(configs)
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training function
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = []
    for inputs, labels in train_loader:
        inputs, labels = inputs.float().to(device), labels.float().to(device)

        optimizer.zero_grad()
        outputs = model(inputs, None, None, None)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss.append(loss.item())

    return np.average(running_loss)

# Validation function
def evaluate(model, val_loader, criterion, device):
    model.eval()
    val_loss = []
    preds = torch.tensor([])
    label = torch.tensor([])
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.float().to(device), labels.float().to(device)

            optimizer.zero_grad()
            outputs = model(inputs, None, None, None)
            loss = criterion(outputs, labels)

            outputs = outputs.detach().cpu()
            labels = labels.detach().cpu()

            preds = torch.cat((preds, outputs.reshape(-1)[:-1]), dim=0)
            label = torch.cat((label, labels.reshape(-1)[:-1]), dim=0)
  
            val_loss.append(loss.item())
    # print(f"preds.shape {preds.shape}, labels shape {labels.shape}")
    ic = np.corrcoef(preds,label)
    print(f" ----------------ic---------------------- \n{ic[0][1]}")

    return np.average(val_loss) 




# %%

# Training loop
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_epochs = 50
total_loss = []
total_val = []
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, criterion, optimizer, device)
    total_loss.append(train_loss)
    val_loss = evaluate(model, val_loader, criterion, device)
    total_val.append(val_loss)
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
# print(f"train loss {np.average(total_loss)} , val loss {np.average(total_val)}")

# Testing function
def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            
            inputs, labels = inputs.float().to(device), labels.float().to(device)
            
            outputs = model(inputs, None, None, None)
            loss = criterion(outputs, labels)
            test_loss += loss.item() 

    return test_loss 

# Evaluate on test set
test_loss = test(model, test_loader, criterion, device)
print(f'Test Loss: {test_loss:.4f}')