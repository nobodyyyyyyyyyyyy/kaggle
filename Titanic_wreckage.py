import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import d2l as d2l
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold

test_titanic = pd.read_csv("C:/Users/86187/Desktop/神经网络（pytorch理论/kaggle_beginer_level/titanic/test.csv")
train_titanic = pd.read_csv("C:/Users/86187/Desktop/神经网络（pytorch理论/kaggle_beginer_level/titanic/train.csv")
kf = KFold(n_splits=5, shuffle=True, random_state=42)


class DataFrameProcessor:
    def __init__(self, df):
        self.df = df

    def encode_columns(self, column_names):
        for column_name in column_names:
            unique_values = self.df[column_name].unique()
            mapping = {val: idx for idx, val in enumerate(unique_values)}
            self.df[column_name] = self.df[column_name].replace(mapping)
        return self.df  # 一般习惯返回处理后的df，以便链式调用

    def exclude_columns(self, label_indexes):
        columns_to_exclude = [self.df.columns[idx - 1] for idx in label_indexes]
        remaining_columns = self.df.columns.difference(columns_to_exclude)
        return remaining_columns

    def extract_matrix_and_labels(self, label_column_indexes):
        feature_columns = [col for col in self.df.columns if col not in self.df.columns[label_column_indexes]]
        features_df = self.df.loc[:, feature_columns]
        labels_df = self.df.loc[:, self.df.columns[label_column_indexes]]
        return features_df, labels_df


# 用法示例
processor_test = DataFrameProcessor(test_titanic)
processor_train = DataFrameProcessor(train_titanic)

# 编码 Sex 和 Embarked 列
processor_test.encode_columns(['Sex', 'Embarked', 'Age'])
processor_train.encode_columns(['Sex', 'Embarked', 'Age'])

# 提取特征和标签
features_df_test, labels_test = processor_test.extract_matrix_and_labels([0, 2, 7, 9])
features_df_train, labels_train = processor_train.extract_matrix_and_labels([0, 1, 3, 8, 10])

# 填充 Age 列的缺失值
test_titanic['Age'].fillna(test_titanic['Age'].mean(), inplace=True)
train_titanic['Age'].fillna(test_titanic['Age'].mean(), inplace=True)
features_df_test.to_excel('features_df_test_output.xlsx', index=False, sheet_name='Sheet1')
features_df_train.to_excel('features_df_train_output.xlsx', index=False, sheet_name='Sheet1')

features = features_df_train.to_numpy()
labels = labels_train['Survived'].to_numpy()


# 利用神经网络进行分类操作
class fully_connected_layer(nn.Module):
    def __init__(self):
        super(fully_connected_layer, self).__init__()
        self.fc1 = nn.Linear(7, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 50)
        self.fc4 = nn.Linear(50, 50)
        self.fc5 = nn.Linear(50, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x


model = fully_connected_layer()

criteria = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

inputs = features_df_train
labels = labels_train['Survived'].to_numpy()

model.train()

for epoch in range(5):  # 训练5个epochs
    total_loss = 0  # 初始化总损失
    for i in range(inputs.shape[0]):  # 遍历每个样本
        # 前向传播
        outputs = model(torch.tensor(inputs.iloc[i].values).float().unsqueeze(0))
        label_tensor = torch.tensor(labels[i]).unsqueeze(0)
        loss = criteria(outputs, label_tensor)  # 计算损失

        # 累加损失
        total_loss += loss.item()

        # 后向传播和优化
        optimizer.zero_grad()  # 清除之前的梯度
        loss.backward()  # 后向传播，计算当前梯度
        optimizer.step()  # 根据梯度更新模型参数

for train_index, val_index in kf.split(features):
    # 划分数据
    X_train, X_val = features[train_index], features[val_index]
    y_train, y_val = labels[train_index], labels[val_index]

    # 转换为torch张量
    X_train_tensor = torch.tensor(X_train, dtype=torch.float)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    # 验证模型
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criteria(val_outputs, y_val_tensor)
        print(f'Validation loss: {val_loss.item()}')

        _, predicted = torch.max(val_outputs.data, 1)  # 获取最大概率的索引，即预测的类别
        total = y_val_tensor.size(0)  # 总样本数
        correct = (predicted == y_val_tensor).sum().item()  # 正确预测的样本数
        accuracy = 100 * correct / total  # 计算准确率

        # 输出验证损失和准确率
        print(f'Validation loss: {val_loss.item()}')
        print(f'Validation accuracy: {accuracy}%')

input_data = torch.tensor(features_df_test.values)
model.eval()
with torch.no_grad():
    predictions = model(input_data.float())
predicted_labels = predictions.argmax(dim=1)
predicted_labels_np = predicted_labels.numpy()

predicted_labels_df = pd.DataFrame(predicted_labels_np, columns=['label'])
test_titanic['label'] = predicted_labels_df['label']
print(test_titanic.head())
