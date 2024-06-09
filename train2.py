import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.fc = nn.Linear(output_dim, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = torch.mean(x, dim=0)
        x = self.fc(x)
        return x

def prepare_data(aig_features, eval_score, feature_name):
    edge_index = aig_features['edge_index']
    node_features = torch.stack([aig_features['node_type'], aig_features['num_inverted_predecessors']], dim=1).float()
    data = Data(x=node_features, edge_index=edge_index, y=torch.tensor([eval_score], dtype=torch.float))
    data.feature_name = feature_name  # 将特征名称添加到Data对象中
    return data

# 读取所有特征文件
featureDir = './feature'
logDir = './log'
featureNames = []
for circuitName in os.listdir(featureDir):
    classDir = os.path.join(featureDir, circuitName)
    if os.path.isdir(classDir):
        for featureFile in os.listdir(classDir):
            featureName = featureFile.split('.')[0]
            featureNames.append(featureName)

# 读取特征和评估得分
dataset = []
scores = pickle.load(open('Scores.pkl', 'rb'))
for featureName in featureNames:
    circuitName = featureName.split('_')[0]
    with open(os.path.join(featureDir, circuitName, f'{featureName}.pkl'), 'rb') as f:
        feature = pickle.load(f)
    eval_score = 10 * scores[featureName]  # 正则化后的分数是真实分数的10倍
    data = prepare_data(feature, eval_score, featureName)
    dataset.append(data)
print(f"Loaded {len(dataset)} data")

# 划分数据集为训练集和测试集
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

# 加载模型
model = GNNModel(input_dim=2, hidden_dim=64, output_dim=32)
model.load_state_dict(torch.load('modelc.pth', map_location=device))
model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.002)

# 训练模型
for epoch in range(200):
    model.train()
    total_loss = 0
    for i, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i+1) % 1000 == 0:
            avg_loss = total_loss / 1000
            print(f'output: {output.item()} , target: {data.y.item()} loss: {loss.item()}')
            print(f'Epoch {epoch+1} , Iteration {i+1} , Loss {avg_loss}')
            total_loss = 0

    # 保存模型
    torch.save(model.state_dict(), f'./modelc.pth')

# 评估模型
model.eval()
test_scores = []
with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        output = model(data)
        real_score = data.y.item()
        predicted_score = output.item()  # 模型输出的预测分数
        score = 1 / (1 + np.exp(abs(predicted_score - real_score) / abs(real_score)))
        test_scores.append(score)
        print(f'Predicted: {predicted_score}, Real: {real_score}, Score: {score}')

# 计算测试集的平均分数
average_test_score = np.mean(test_scores)
print(f'Average test score: {average_test_score}')


