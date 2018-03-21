import numpy as np
import os, csv, random, pickle, time

class DataIterator():
    def __init__(self, X, Y, batch_size=64):
        self.num_splits = len(X) // batch_size
        self.X = np.array_split(X, self.num_splits)
        self.Y = np.array_split(Y, self.num_splits)
        self.cursor = 0

    def __iter__(self):
        return zip(self.X, self.Y)

    def next(self, step=1):
        X, Y = self.X[self.cursor], self.X[self.cursor]
        self.cursor = (self.cursor + step) % self.num_splits
        return X, Y

def read_cifar(fname):
    with open(fname, 'rb') as f:
        d = pickle.load(f, encoding='bytes')
    return d[b'data'], np.asarray(d[b'labels'])

def check_class_balance(y_train, y_test):
    classes = np.unique(np.concatenate((y_train, y_test)))
    class_balance_train, class_balance_test = np.empty((len(classes)), dtype=np.int32), np.empty((len(classes)), dtype=np.int32)
    for idx, c in enumerate(classes):
        class_balance_train[idx] = (y_train == c).sum()
        class_balance_test[idx] = (y_test == c).sum()
    return classes, class_balance_train, class_balance_test
