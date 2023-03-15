import numpy as np
import torch
from res.myread import read_data
import matplotlib.pyplot as plt


# 逻辑回归

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


class MyLR:
    def __init__(self):
        self.w = torch.normal(0, 0.01, size=(5, 1), requires_grad=True, dtype=torch.float64)
        self.b = torch.zeros(1, requires_grad=True, dtype=torch.float64)
        self.lr = 0.005
        self.epoch = 1000
        self.batch_size = 100

    def logi_reg(self, X):
        return torch.sigmoid((torch.matmul(X, self.w) + self.b))

    def squared_loss(self, y_pred, y):
        return ((y_pred - y.reshape(y_pred.shape)) ** 2 / 2)

    def bcelose(self, y_pred, y):
        y_pred = torch.cat((1 - y_pred, y_pred), 1)
        return -torch.log(y_pred.gather(1, y.view(-1, 1).to(torch.int64)))

    def loss_fc(self, y_pred, y):
        return self.bcelose(y_pred, y)

    def optim(self):
        with torch.no_grad():
            for param in [self.w, self.b]:
                param -= self.lr * param.grad / self.batch_size
                param.grad.zero_()

    def test(self, X, y):
        y_pred = self.logi_reg(X)
        mask = y_pred.ge(0.5).float().squeeze()  # 以0.5为阈值进行分类
        correct = ((mask == y).sum())  # 计算正确预测的样本个数
        acc = correct.item() / y.size(0)  # 计算分类准确率
        return acc


train_features, train_labels, test_features, test_labels = read_data('lr')

net = MyLR()
max_acc = 0
mark_epoch = 0
loss_list, train_acc_list, test_acc_list = [], [], []
print('training')
for epoch in range(net.epoch):
    for X, y in data_iter(net.batch_size, train_features, train_labels):
        loss = net.loss_fc(net.logi_reg(X), y)
        loss.sum().backward()
        net.optim()
    with torch.no_grad():
        train_loss = net.loss_fc(net.logi_reg(train_features), train_labels)
        # print(f'epoch {epoch + 1}, loss {float(train_loss.mean()):f}')
        train_acc = net.test(train_features, train_labels)
        test_acc = net.test(test_features, test_labels)
        if test_acc > max_acc:
            max_acc = test_acc
            mark_epoch = epoch
        loss_list.append(train_loss.mean())
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)  
print('max test acc:', max_acc, '\nwitch epoch: ', mark_epoch)
draw = True
if draw:
    fig, axes = plt.subplots()
    axes.plot(np.array([i for i in range(len(loss_list))]), np.array(loss_list), label='loss')
    axes.plot(np.array([i for i in range(len(train_acc_list))]), np.array(train_acc_list), label='train acc')
    axes.plot(np.array([i for i in range(len(test_acc_list))]), np.array(test_acc_list), label='test acc')
    axes.legend()
    plt.show()
y_pred = net.logi_reg(test_features)
mask = y_pred.ge(0.5).float().squeeze()  # 以0.5为阈值进行分类
correct = ((mask == test_labels).sum())  # 计算正确预测的样本个数
acc = correct.item() / test_labels.size(0)  # 计算分类准确率
print('test acc: ', acc)

torch.save(net, './res/net.pkl')
net = torch.load('./res/net.pkl')
y_pred = net.logi_reg(test_features)
mask = y_pred.ge(0.5).float().squeeze()  # 以0.5为阈值进行分类
correct = ((mask == test_labels).sum())  # 计算正确预测的样本个数
acc = correct.item() / test_labels.size(0)  # 计算分类准确率
print('test acc: ', acc)
