import torch
import torch.nn as nn
from res.myread import read_data


# 逻辑回归

class LR(nn.Module):
    def __init__(self):
        super(LR, self).__init__()
        self.features = nn.Linear(5, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.features(x)
        x = self.sigmoid(x)
        return x


train_features, train_labels, test_features, test_labels = read_data('lr')

net = LR().double()
loss_fc = nn.BCELoss()
lr = 0.01  # 学习率
opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)

iteration = 10000
max_acc = 0
for it in range(iteration):
    y_pred = net(train_features)
    loss = loss_fc(y_pred.squeeze(), train_labels)
    # print(loss)
    print_loss = loss.data.item()  # 得出损失函数值
    loss.backward()
    opt.step()
    opt.zero_grad()
    if it % 20 == 0:
        mask = y_pred.ge(0.5).float().squeeze()  # 以0.5为阈值进行分类
        correct = ((mask == train_labels).sum())  # 计算正确预测的样本个数
        acc = correct.item() / train_labels.size(0)  # 计算分类准确率
        if acc > max_acc:
            max_acc = acc
print('train acc: ', acc)

y_pred = net(test_features)
mask = y_pred.ge(0.5).float().squeeze()  # 以0.5为阈值进行分类
correct = ((mask == test_labels).sum())  # 计算正确预测的样本个数
acc = correct.item() / test_labels.size(0)  # 计算分类准确率
print('test acc: ', acc)

torch.save(net, './res/net.pkl')
net = torch.load('./res/net.pkl')
y_pred = net(test_features)
mask = y_pred.ge(0.5).float().squeeze()  # 以0.5为阈值进行分类
correct = ((mask == test_labels).sum())  # 计算正确预测的样本个数
acc = correct.item() / test_labels.size(0)  # 计算分类准确率
print('test acc: ', acc)
