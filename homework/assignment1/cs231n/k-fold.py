# -*- coding: utf-8 -*-
"""
test the k_nearest_neighbor classifier on the CIFAR-10 dataset.
推荐一步步的运算执行！
"""
from __future__ import print_function
import random
import numpy as np
from data_utils import load_CIFAR10
import matplotlib.pyplot as plt

from classifiers import KNearestNeighbor

# Load the raw CIFAR-10 data.
print("Loading data.....")
cifar10_dir = 'datasets/cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# As a sanity check, we print out the size of the training and test data.
print('Training data shape: ', X_train.shape)  # (50000, 32, 32, 3)
print('Training labels shape: ', y_train.shape)  # (50000,)
print('Test data shape: ', X_test.shape)  # (10000, 32, 32, 3)
print('Test labels shape: ', y_test.shape)  # (10000,)
# print(y_train[:10])

# Visualize some examples from the dataset.
# We show a few examples of training images from each class.
print("visualize the data....")
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)
samples_per_class = 7
for y, cls in enumerate(classes):
    idxs = np.flatnonzero(y_train == y)  # 找到对应类别y的索引
    idxs = np.random.choice(idxs, samples_per_class, replace=False)  # 不重复采样
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(X_train[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls)
plt.show()

print("subsample the data.....")
# Subsample the data for more efficient code execution in this exercise
num_training = 5000
mask = list(range(num_training))
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 500
mask = list(range(num_test))
X_test = X_test[mask]
y_test = y_test[mask]

print("reshape....")
# Reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))  # (5000, 3072)
X_test = np.reshape(X_test, (X_test.shape[0], -1))   # (500, 3072)
print(X_train.shape, X_test.shape)

print("using 5-fold cross-validation to train and test.......")
num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

X_train_folds = []
y_train_folds = []
################################################################################
# TODO:                                                                        #
# Split up the training data into folds. After splitting, X_train_folds and    #
# y_train_folds should each be lists of length num_folds, where                #
# y_train_folds[i] is the label vector for the points in X_train_folds[i].     #
# Hint: Look up the numpy array_split function.                                #
################################################################################
X_train_folds = np.array_split(X_train, num_folds)
y_train_folds = np.array_split(y_train, num_folds)
################################################################################
#                                 END OF YOUR CODE                             #
################################################################################

# A dictionary holding the accuracies for different values of k that we find
# when running cross-validation. After running cross-validation,
# k_to_accuracies[k] should be a list of length num_folds giving the different
# accuracy values that we found when using that value of k.
k_to_accuracies = {}
################################################################################
# TODO:                                                                        #
# Perform k-fold cross validation to find the best value of k. For each        #
# possible value of k, run the k-nearest-neighbor algorithm num_folds times,   #
# where in each case you use all but one of the folds as training data and the #
# last fold as a validation set. Store the accuracies for all fold and all     #
# values of k in the k_to_accuracies dictionary.                               #
################################################################################
for k in k_choices:
    k_to_accuracies[k] = np.zeros(num_folds)
    for i in range(num_folds):
        xtr = np.array(X_train_folds[:i] + X_train_folds[(i+1):])
        ytr = np.array(y_train_folds[:i] + y_train_folds[(i+1):])
        xdev = np.array(X_train_folds[i])
        ydev = np.array(y_train_folds[i])

        xtr = np.reshape(xtr, ((X_train.shape[0] * (num_folds-1)) / num_folds, -1))  # (4000, 3072)
        ytr = np.reshape(ytr, ((y_train.shape[0] * (num_folds-1)) / num_folds, -1))  # (4000, 1)
        xdev = np.reshape(xdev, (X_train.shape[0] / num_folds, -1))  # (1000, 3072)
        ydev = np.reshape(ydev, (y_train.shape[0] / num_folds, -1))  # (1000, 1)

        nn = KNearestNeighbor()
        nn.train(xtr, ytr)
        y_predict = nn.predict(xdev, k=k, num_loops=0)  # (1000, )
        # print(np.shape(y_predict))  注意维度匹配
        y_predict = np.reshape(y_predict, (y_predict.shape[0], -1))
        num_correct = np.sum(y_predict == ydev)
        accuracy = num_correct / float(xdev.shape[0])
        k_to_accuracies[k][i] = accuracy
################################################################################
#                                 END OF YOUR CODE                             #

################################################################################

# Print out the computed accuracies
print("the computed accuracies.....")
for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
        print('k = %d, accuracy = %f' % (k, accuracy))

print("plot the raw observations....")
# plot the raw observations
for k in k_choices:
    accuracies = k_to_accuracies[k]
    plt.scatter([k] * len(accuracies), accuracies)  # 画出相应的k值和accuracy

# plot the trend line with error bars that correspond to standard deviation
accuracies_mean = np.array([np.mean(v) for k,v in sorted(k_to_accuracies.items())])
accuracies_std = np.array([np.std(v) for k,v in sorted(k_to_accuracies.items())])
plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
plt.title('Cross-validation on k')
plt.xlabel('k')
plt.ylabel('Cross-validation accuracy')
plt.show()
