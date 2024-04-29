import torch as torch
import pandas as pd
import numpy as np
import torch.nn as nn

house_train = pd.read_csv(
    r"C:\Users\86187\Desktop\神经网络（pytorch理论\kaggle_beginer_level\house-prices-advanced-regression-techniques\train.csv")
house_test = pd.read_csv(
    r"C:\Users\86187\Desktop\神经网络（pytorch理论\kaggle_beginer_level\house-prices-advanced-regression-techniques\test.csv")


def clear_data(data):
    numeric_columns = data.select_dtypes(include=['number']).columns.tolist()
    medians = data[numeric_columns].median()
    data[numeric_columns] = data[numeric_columns].fillna(medians)

    str_columns = data.select_dtypes(exclude=['number']).columns.tolist()
    for column in str_columns:
        unique_values = data[column].unique()
        mapping = {val: idx for idx, val in enumerate(unique_values)}
        data[column] = data[column].map(mapping)

    return data  # Optionally return the modified data


clear_data(house_train)
clear_data(house_train)


class fully_connected_layer(nn.Module):
    def __init__(self, house_train_):
        super(fully_connected_layer, self).__init__()
        self.fc1 = nn.Linear(house_train_, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 500)
        self.fc4 = nn.Linear(500, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


model = fully_connected_layer(house_train.shape[1] - 1)

criteria = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
inputs = torch.tensor(house_train.iloc[:, :-1].values).float()
labels = torch.tensor(house_train.iloc[:, -1].values).float().unsqueeze(1)
print("Inputs:", inputs)

for epoch in range(5):  # 训练5个epochs
    total_loss = 0  # 初始化总损失
    for i in range(inputs.shape[0]):  # 遍历每个样本
        # 前向传播
        outputs = model(torch.tensor(inputs[i]).float().unsqueeze(0))
        label_tensor = torch.tensor(labels[i]).unsqueeze(0)
        loss = criteria(outputs, label_tensor)  # 计算损失

        # 累加损失
        total_loss += loss.item()

        # 后向传播和优化
        optimizer.zero_grad()  # 清除之前的梯度
        loss.backward()  # 后向传播，计算当前梯度
        optimizer.step()  # 根据梯度更新模型参数
