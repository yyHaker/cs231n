# -*- coding: utf-8 -*-
"""
实现两层神经网络，并对CIFIR-10数据集进行分类
"""
from __future__ import print_function
import numpy as np
from data_utils import load_CIFAR10
import matplotlib.pyplot as plt
import time

import torch
from  torch.autograd.variable import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000, num_dev=500):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the linear classifier. These are the same steps as we used for the
    SVM, but condensed to a single function.
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]
    mask = np.random.choice(num_training, num_dev, replace=False)
    X_dev = X_train[mask]
    y_dev = y_train[mask]

    # Preprocessing: reshape the image data into rows
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    X_dev -= mean_image

    # add bias dimension and transform into columns
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])

    return X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev


print("load data........")
# Invoke the above function to get our data.
X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev = get_CIFAR10_data()
print('Train data shape: ', X_train.shape)  # (49000, 3073)
print('Train labels shape: ', y_train.shape)  # (49000,)
print('Validation data shape: ', X_val.shape)  # (1000, 3073)
print('Validation labels shape: ', y_val.shape)  # (1000,)
print('Test data shape: ', X_test.shape)  # (1000, 3073)
print('Test labels shape: ', y_test.shape)  # (1000,)
print('dev data shape: ', X_dev.shape)  # (500, 3073)
print('dev labels shape: ', y_dev.shape)  # (500,)

# 将numpy 转换为FloatTensor
X_train = torch.FloatTensor(X_train)
y_train = torch.IntTensor(y_train)
X_val = torch.FloatTensor(X_val)
y_val = torch.IntTensor(y_val)
X_test = torch.FloatTensor(X_test)
y_test = torch.IntTensor(y_test)
X_dev = torch.FloatTensor(X_dev)
y_dev = torch.IntTensor(y_dev)

# parameters
BATCH_SIZE = 128
LR = 0.3
EPOCHS = 1000
print("preparing the data......")
# training set先转换成 torch 能识别的 Dataset
train_dataset = Data.TensorDataset(data_tensor=X_train, target_tensor=y_train)
# 把 dataset 放入 DataLoader
train_loader = Data.DataLoader(
    dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True
)
# validation set
X_val = Variable(X_val)
y_val = y_val.numpy().squeeze()   # convert to numpy array
# test set
X_test = Variable(X_test)
y_test = y_test.numpy().squeeze()  # convert to numpy array

print("X_val shape: ", X_val.data.shape)  # torch,  1000 x 3073
print("y_val shape: ", y_val.shape)  # (1000, )
print("X_test shape: ", X_test.data.shape)  # torch, 1000, 3073
print("y_test shape: ", y_test.shape)   # (1000, )


# build the networks
class NeuralNetworks(nn.Module):
    def __init__(self, n_features, hidden1, hidden2, n_output):
        super(NeuralNetworks, self).__init__()
        self.hidden_1 = torch.nn.Linear(n_features, hidden1)
        self.hidden_2 = torch.nn.Linear(hidden1, hidden2)
        self.out = torch.nn.Linear(hidden2, n_output)

    def forward(self, x):
        x = F.sigmoid(self.hidden_1(x))
        x = F.sigmoid(self.hidden_2(x))
        out = self.out(x)
        return out


# get the network
print("The network.....")
net = NeuralNetworks(n_features=3073, hidden1=1000, hidden2=100, n_output=10)
print(net)

print("training...........")
optimizer = torch.optim.SGD(net.parameters(), lr=LR)
loss_func = torch.nn.CrossEntropyLoss()

best_model = None
best_val_accuracy = 0.
loss_history = []
for epoch in range(EPOCHS):
    for step, (batch_x, batch_y) in enumerate(train_loader):
        b_x = Variable(batch_x)
        b_y = Variable(batch_y)

        out = net(b_x)  # forward
        loss = loss_func(out, b_y)  # calculate loss
        loss_history.append(loss)
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients
        if step % 50 == 0:
            val_out = net(X_val)
            pred_y = torch.max(val_out, 1)[1].data.numpy().squeeze()  # val prediction result
            val_accuracy = sum(pred_y == y_val) / float(y_val.size)
            print('Epoch: ', epoch, '| Step: ', step, '| Train loss: %.4f' % loss.data.numpy(),
                  '| Validation accuracy %.2f' % val_accuracy)
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_model = net   # save

print("training done......")

print("plotting the training losses...")
plt.plot(loss_history)
plt.xlabel('Iteration number')
plt.ylabel('Loss value')
plt.show()

print("use the best model to predict the test data......")
out = best_model(X_test)
pred_y = torch.max(out, 1)[1].data.numpy().squeeze()
test_accuracy = sum(pred_y == X_test) / float(X_test.size)
print("the test accuracy is: %.4f" % test_accuracy)












