#!/usr/bin/env python
"""Chainer example: train a multi-layer perceptron on MNIST

This is a minimal example to write a feed-forward net. It requires scikit-learn
to load MNIST dataset.

"""
import argparse

import numpy as np
import six

import chainer
from chainer import computational_graph as c
from chainer import cuda
import chainer.functions as F
from chainer import optimizers

import load_images

import cv2

parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()
if args.gpu >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if args.gpu >= 0 else np

batchsize = 100
n_epoch = 5

resize = 50
predict_no = 80

#load train data
print('load Sunshine dataset')
x_train, x_test, y_train, y_test = load_images.load("./image",11,10,resize,resize)

cv2.imshow("predict",x_test[predict_no])
cv2.waitKey(0)


#Normalization 0~1
x_train = np.array(x_train).astype(np.float32).reshape((len(x_train),3, resize, resize)) / 255
y_train = np.array(y_train).astype(np.int32)
x_test = np.array(x_test).astype(np.float32).reshape((len(x_test),3, resize, resize)) / 255
y_test = np.array(y_test).astype(np.int32)

N = len(y_train)
print(N)
print(len(x_test))
N_test = y_test.size

# Prepare multi-layer perceptron model
model = chainer.FunctionSet(cv1=F.Convolution2D(3,20, 3),
                            bn2 = F.BatchNormalization(20),
                            ln3=F.Linear(11520, 1000),
                            ln4=F.Linear(1000, 11))

if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()

def modelFunction(x,train):
    h = F.max_pooling_2d(F.dropout(F.relu(model.bn2(model.cv1(x))),  train=train),2)
    h = F.dropout(F.relu(model.ln3(h)), train=train)
    return model.ln4(h)
 
def forward(x_data, y_data, train=True):
    # Neural net architecture
    x, t = chainer.Variable(x_data), chainer.Variable(y_data)
    y = modelFunction(x,train)
    return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

   

def predict(x_data, train=False):
    # Neural net architecture
    x = chainer.Variable(x_data)
    y = modelFunction(x,train)
    return  F.softmax(y)


# Setup optimizer
optimizer = optimizers.Adam()
optimizer.setup(model)

# Learning loop
for epoch in six.moves.range(1, n_epoch + 1):
    print('epoch', epoch)

    # training
    perm = np.random.permutation(N)
    sum_accuracy = 0
    sum_loss = 0
    for i in six.moves.range(0, N, batchsize):
        x_batch = xp.asarray(x_train[perm[i:i + batchsize]])
        y_batch = xp.asarray(y_train[perm[i:i + batchsize]])

        optimizer.zero_grads()
        loss, acc = forward(x_batch, y_batch)
        loss.backward()
        optimizer.update()

        if epoch == 1 and i == 0:
            with open("graph.dot", "w") as o:
                o.write(c.build_computational_graph((loss, )).dump())
            with open("graph.wo_split.dot", "w") as o:
                g = c.build_computational_graph((loss, ),
                                                remove_split=True)
                o.write(g.dump())
            print('graph generated')

        sum_loss += float(loss.data) * len(y_batch)
        sum_accuracy += float(acc.data) * len(y_batch)

    print('train mean loss={}, accuracy={}'.format(
        sum_loss / N, sum_accuracy / N))

    # evaluation
    sum_accuracy = 0
    sum_loss = 0
    for i in six.moves.range(0, N_test, batchsize):
        x_batch = xp.asarray(x_test[i:i + batchsize])
        y_batch = xp.asarray(y_test[i:i + batchsize])

        loss, acc = forward(x_batch, y_batch, train=False)

        sum_loss += float(loss.data) * len(y_batch)
        sum_accuracy += float(acc.data) * len(y_batch)

    print('test  mean loss={}, accuracy={}'.format(
        sum_loss / N_test, sum_accuracy / N_test))


    score = predict(xp.asarray(x_test[predict_no:predict_no+1]))
    print(y_test[predict_no])
    print(score.data)

