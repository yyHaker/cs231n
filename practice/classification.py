# -*- coding: utf-8 -*-
"""
使用神经网络来训练手写数字，学习调节超参数
1.网络层数, 神经元选择(sigmoid, tanh, relu...), 损失函数，正则化，dropout
2.learning rate , batch_size, hidden_size等等
3. 优化方法 sgd, Adam, momentum....
4.网络参数的初始化等
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision   # database module
import numpy as np

import matplotlib.pyplot as plt

# Hyper parameters
EPOCH = 30
BATCH_SIZE = 8
LR = 0.0003
DOWNLOAD_MNIST = True


# MNIST
print("load data.........")
train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)
# print(train_data.train_data)  # ByteTensor (60000x28x28)
# print(train_data.train_labels)  # LongTensor (60000)
train_x = train_data.train_data[:5000].type(torch.FloatTensor)
# print(train_x)  # torch.FloatTensor of size 5000x28x28
train_y = train_data.train_labels[:5000]
# print(train_y)  # torch.LongTensor of size 5000
val_x = train_data.train_data[5000:6000].type(torch.FloatTensor)
# print(val_x)  # torch.FloatTensor of size 1000x28x28
val_y = train_data.train_labels[5000:6000]
# print(val_y)  # torch.LongTensor of size 1000

# training set先转换成 torch 能识别的 Dataset
train_datasets = Data.TensorDataset(data_tensor=train_x, target_tensor=train_y)
# 把 dataset 放入 DataLoader
train_loader = Data.DataLoader(dataset=train_datasets, batch_size=BATCH_SIZE, shuffle=True)
test_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=False
)
# print(test_data.test_data)  # ByteTensor (10000x28x28)
# print(test_data.test_labels)  # LongTensor (10000)
test_x = test_data.test_data[:1000].type(torch.FloatTensor)
# print(test_x)  # torch.FloatTensor of size 1000x28x28
test_y = test_data.test_labels[:1000]
# print(test_y)  # torch.LongTensor of size 1000

# sample some data to visualize
print("sample some data to visualize..............")
for i in range(3):
    plt.imshow(train_x[i].numpy(), cmap='gray')
    plt.title('%i' % train_y[i])
    plt.show()

# build the network
net = torch.nn.Sequential(
    torch.nn.Linear(28*28, 500),
   # torch.nn.Dropout(0.5),
    torch.nn.ReLU(),
    torch.nn.Linear(500, 300),
   # torch.nn.Dropout(0.5),
    torch.nn.ReLU(),
    torch.nn.Linear(300, 100),
   # torch.nn.Dropout(0.5),
    torch.nn.ReLU(),
    torch.nn.Linear(100, 10),
    #torch.nn.Softmax()
)
print("the net architecture.......")
print(net)


def initNetParams(net):
    '''Init net parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Linear):
            # b = np.sqrt(6.0 / (m.weight.size(0) + m.weight.size(1)))
            # nn.init.uniform(m.weight, -b, b)
            m.weight.data.normal_(mean=0, std=np.sqrt(2.0/m.weight.size(0)))
            if m.bias is not None:
                m.bias.data.zero_()
# initNetParams(net)

# training
print("training..........")
optimizer = torch.optim.SGD(net.parameters(), lr=LR, weight_decay=1e-5)
loss_func = nn.CrossEntropyLoss()

best_val_accuracy = 0.0
best_model = None
loss_history = []

val_x = Variable(val_x.view(-1, 28*28))
val_y = val_y.numpy()
test_x = Variable(test_x.view(-1, 28*28))
test_y = test_y.numpy()
for epoch in range(EPOCH):
    for step, (batch_x, batch_y) in enumerate(train_loader):
        b_x = Variable(batch_x.view(-1, 28*28))
        b_y = Variable(batch_y)

        output = net(b_x)
        loss = loss_func(output, b_y)

        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()           # backpropagation, compute gradients
        optimizer.step()           # apply gradients

        loss_history.append(loss.data.numpy())

        if step % 50 == 0:
            out = net(val_x)
            pred_y = torch.max(out, 1)[1].data.numpy().squeeze()
            val_accuracy = sum(pred_y == val_y) / float(val_y.size)
            print("Epoch: ", epoch, "| Step:", step, "| Train loss: %.4f " % loss.data.numpy(),
                  '| validation accuracy %.2f ' % val_accuracy)
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_model = net
print("training done......")
print("The best validation accuracy is %.2f" % best_val_accuracy)
print("use the best model to predict the test......")
out = best_model(test_x)
pred_y = torch.max(out, 1)[1].data.numpy().squeeze()
test_accuracy = sum(pred_y == test_y) / float(test_y.size)
print("the test accuracy is %.2f" % test_accuracy)

print("plotting the training losses...")
print("the loss history length is %i" % len(loss_history))
# plt.ylim(np.min(loss_history[:100]), np.max(loss_history[:100]))
plt.plot(np.arange(1, 201), loss_history[:200])
plt.xlabel('Iteration number')
plt.ylabel('Loss value')
plt.show()




