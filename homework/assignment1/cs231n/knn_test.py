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

print("training.....")
# Create a kNN classifier instance.
# Remember that training a kNN classifier is a noop:
# the Classifier simply remembers the data and does no further processing
classifier = KNearestNeighbor()
classifier.train(X_train, y_train)

print("test compute_distances_two_loops implementation......")
# Test your implementation:
dists = classifier.compute_distances_two_loops(X_test)
print(dists.shape)  # (num_test x num_train)
# We can visualize the distance matrix: each row is a single test example and
# its distances to training examples
plt.imshow(dists, interpolation='none')
plt.show()

print("set k=1 and test the data......")
# Now implement the function predict_labels and run the code below:
# We use k = 1 (which is Nearest Neighbor).
y_test_pred = classifier.predict_labels(dists, k=1)
# Compute and print the fraction of correctly predicted examples
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))

print("use compute_distances_one_loop to calculate the dists.....")
# Now lets speed up distance matrix computation by using partial vectorization
# with one loop. Implement the function compute_distances_one_loop and run the
# code below:
dists_one = classifier.compute_distances_one_loop(X_test)
# To ensure that our vectorized implementation is correct, we make sure that it
# agrees with the naive implementation. There are many ways to decide whether
# two matrices are similar; one of the simplest is the Frobenius norm. In case
# you haven't seen it before, the Frobenius norm of two matrices is the square
# root of the squared sum of differences of all elements; in other words, reshape
# the matrices into vectors and compute the Euclidean distance between them.
print("test the difference between two loop and one loop........")
difference = np.linalg.norm(dists - dists_one, ord='fro')
print('Difference was: %f' % (difference, ))
if difference < 0.001:
    print('Good! The distance matrices are the same')
else:
    print('Uh-oh! The distance matrices are different')

print("use compute_distances_no_loop to calculate the dists.....")
# Now implement the fully vectorized version inside compute_distances_no_loops
# and run the code
dists_two = classifier.compute_distances_no_loops(X_test)
# check that the distance matrix agrees with the one we computed before:
print("test the difference between two loop and no loop........")
difference = np.linalg.norm(dists - dists_two, ord='fro')
print('Difference was: %f' % (difference, ))
if difference < 0.001:
    print('Good! The distance matrices are the same')
else:
    print('Uh-oh! The distance matrices are different')


# Let's compare how fast the implementations are
def time_function(f, *args):
    """
    Call a function f with args and return the time (in seconds) that it took to execute.
    """
    import time
    tic = time.time()
    f(*args)
    toc = time.time()
    return toc - tic

two_loop_time = time_function(classifier.compute_distances_two_loops, X_test)
print('Two loop version took %f seconds' % two_loop_time)

one_loop_time = time_function(classifier.compute_distances_one_loop, X_test)
print('One loop version took %f seconds' % one_loop_time)

no_loop_time = time_function(classifier.compute_distances_no_loops, X_test)
print('No loop version took %f seconds' % no_loop_time)

"""
Two loop version took 46.924758 seconds
One loop version took 110.905241 seconds
No loop version took 1.839480 seconds
所以看似简单直观的one loop速度最慢！而numpy矩阵操作最快！
"""
"""
小结： 
1. 学会灵活利用numpy矩阵操作，计算速度确实大大加快了
2. 学会控制运算过程中的矩阵维度，灵活使用np.reshape()和numpy slice操作、broadcast、dim参数等
3. 灵活使用np.bincount(), np.argsort(), np.flatnonzero(), np.random.choice(), np.array_split()等库
"""