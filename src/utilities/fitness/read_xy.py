from algorithm.parameters import params
from utilities.stats.logger import Logger
import cv2 as cv
import numpy as np
import os, csv, random, pickle

class DataReader():
    @classmethod
    def read_data(cls, data_id):
        if data_id in range(1, 5):
            return cls.read_csv(cls)
        if data_id is 5:
            return cls.read_cifar(cls)

    def read_csv(self):
        Logger.log("Reading images from {0} ...".format(params['DATASET']), info=False)

        X, Y = np.array([]), np.array([])
        datapath = os.path.join(params['DATASET'], 'dataset.csv')
        with open(datapath) as f:
            fnames = csv.reader(f)
            next(fnames) #Skip header
            for row in fnames:
                fname = os.path.join(params['DATASET'], row[0])
                img = cv.imread(fname) if os.path.isfile(fname) else print(fname+" does not exist")
                X = np.append(X, img)
                Y = np.append(Y, int(row[1]))
            Logger.log("Done reading dataset with {0} images...".format(len(X)), info=False)
        return X, Y

    def read_cifar(self):
        Logger.log("Reading images from {} ...".format(params['DATASET']), info=False)

        X, Y = np.array([]), np.array([])
        for num in range(1, 6):
            datapath = os.path.join(params['DATASET'], 'data_batch_'+str(num))
            with open(datapath, 'rb') as f:
                d = pickle.load(f, encoding='bytes')
                x, y = d[b'data'], np.asarray(d[b'labels'])
                X = np.append(X, x)
                Y = np.append(Y, y)
        X = X.reshape((-1, 32, 32, 3))
        Logger.log("Done reading dataset with {} images...".format(len(X)), info=False)
        return X, Y
