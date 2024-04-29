import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init


def cleandata(data):
    data_ = data.copy()

    if 'Cabin' in data.columns:
        splits = data_['Cabin'].str.split('/', expand=True)
        data.drop('Cabin', axis=1, inplace=True)
        data['Cabin_Category'] = splits[0]
        data['Cabin_Number'] = splits[1]
        data['Cabin_Identifier'] = splits[2]

    column_names_num = data.select_dtypes(include=['number']).columns.tolist()
    column_str_num = data.select_dtypes(exclude=['number']).columns.tolist()
    for i in column_str_num:
        valu = data[i].unique()
        mapping = {val: idx for idx, val in enumerate(valu)}
        data[i] = data[i].map(mapping)

    medians = data[column_names_num].median()
    data[column_names_num] = data[column_names_num].fillna(medians)

    return data


space_test = pd.read_csv(
    r"C:\Users\86187\Desktop\神经网络（pytorch理论\kaggle_beginer_level\spaceship-titanic\test.csv")
space_train = pd.read_csv(
    r"C:\Users\86187\Desktop\神经网络（pytorch理论\kaggle_beginer_level\spaceship-titanic\train.csv")

space_test = cleandata(space_test)
space_train = cleandata(space_train)


class DenseNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(DenseNetwork, self).__init__()
        self.layers = nn.ModuleList()
        self.total_features = input_size
        current_size = input_size
        self.feature_sizes = [input_size]
        self.dropout = nn.Dropout(p=0.1)

        for size in hidden_sizes:
            self.layers.append(nn.Linear(current_size, size))
            current_size = size
            self.feature_sizes.append(self.feature_sizes[-1] + current_size)

        assert isinstance(current_size, object)
        self.output = nn.Linear(self.feature_sizes[-1], output_size)

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            x = self.dropout(F.relu(layer(x)))
            features.append(x)

        x = torch.cat(features, dim=1)
        x = self.output(x)
        return x

train_lables = space_train['Transported']
train_data = space_train.drop(columns=['PassengerId', 'Name', 'Transported'])

input_size = len(train_data.columns)
hidden_sizes = [input_size * 30 for _ in range(5)]
output_size = len(train_lables.unique())-1

module = DenseNetwork(input_size, hidden_sizes, output_size)
lossfuc = nn.CrossEntropyLoss()
optimization = torch.optim.SGD(module.parameters(), lr=0.01)

train_data_tensor = torch.tensor(train_data.values.astype(np.float32))  # Ensure data is in float32 format for neural network
train_labels_tensor = torch.tensor(train_lables.values, dtype=torch.float32).unsqueeze(1)
batch_size = 2

from torch.utils.data import TensorDataset, DataLoader
dataset = TensorDataset(train_data_tensor, train_labels_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

module.train()
num_epochs = 5
for epoch in range(num_epochs):
    for inputs, labels in dataloader:
        optimization.zero_grad()
        outputs = module(inputs)
        loss = lossfuc(outputs, labels)
        loss.backward()
        optimization.step()

test_data = space_test.drop(columns=['PassengerId', 'Name'])
module.eval()

test_data_tensor = torch.tensor(test_data.values.astype(np.float32))
with torch.no_grad():
    test_outputs = module(test_data_tensor)
    predicted_classes = torch.argmax(test_outputs, dim=1)

predicted_classes = predicted_classes.numpy()
print(predicted_classes)

