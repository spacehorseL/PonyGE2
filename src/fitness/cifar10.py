from algorithm.parameters import params
from fitness.base_ff_classes.base_ff import base_ff
from utilities.stats.logger import Logger
from utilities.stats.individual_stat import stats
from utilities.fitness.image_data import ImageData, ImageProcessor
from utilities.fitness.network import ClassificationNet
from sklearn.model_selection import train_test_split, KFold
import cv2 as cv
import numpy as np
import os, csv, random, pickle

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

class cifar10(base_ff):
    maximise = True  # True as it ever was.
    def __init__(self):
        # Initialise base fitness function class.
        super().__init__()
        self.layers = [16384, 256, 10]
        self.resize = params['RESIZE']

        # Read images from dataset
        Logger.log("Reading images from {0} ...".format(params['DATASET']), info=False)

        X, Y = np.array([]), np.array([])
        for num in range(1, 6):
            datapath = os.path.join(params['DATASET'], 'data_batch_'+str(num))
            x, y = self.read_cifar(datapath)
            # print(x, y)
            X = np.append(X, x)
            Y = np.append(Y, y)
        X = np.reshape(X, (-1, 32,32,3))
        Logger.log("Done reading dataset with {0} images...".format(len(X)), info=False)

        # Train & test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
        Logger.log("Training & Test split: {0}/{1} with size {2}".format(len(self.X_train), len(self.X_test), self.X_train[0].shape), info=False)
        Logger.log("CUDA ENABLED = {}".format(params['CUDA_ENABLED']), info=False)
        Logger.log("Using grammar = {}".format(params['GRAMMAR_FILE'], info=False))
        Logger.log("Resizing after processing = {}".format(self.resize, info=False))
        Logger.log("Network structure = \n{}".format(ClassificationNet(self.layers).model, info=False))

    def read_cifar(self, fname):
        with open(fname, 'rb') as f:
            d = pickle.load(f, encoding='bytes')
        return d[b'data'], np.asarray(d[b'labels'])

    def evaluate(self, ind, **kwargs):
        # ind.phenotype will be a string, including function definitions etc.
        # When we exec it, it will create a value XXX_output_XXX, but we exec
        # inside an empty dict for safety.

        p, d = ind.phenotype, {}

        genome, output, invalid, max_depth, nodes = ind.tree.get_tree_info(params['BNF_GRAMMAR'].non_terminals.keys(),[], [])
        Logger.log("Depth: {0}\tGenome: {1}".format(max_depth, genome))

        # Exec the phenotype.
        Logger.log("Processing Pipeline Start: {} images...".format(len(self.X_train)+len(self.X_test)))
        processed_train = ImageProcessor.process_images(self.X_train, ind, resize=self.resize)
        processed_test = ImageProcessor.process_images(self.X_test, ind, resize=self.resize)

        image = ImageProcessor.image
        init_size = image.shape[0]*image.shape[1]*image.shape[2]

        train_loss = stats('mse')
        test_loss = stats('accuracy')
        kf, freq = KFold(n_splits=params['CROSS_VALIDATION_SPLIT']), params["EPOCH_FREQ"]
        net = ClassificationNet(self.layers)
        fitness, early_stop, fold = 0, 0, 1

        Logger.log("Training Start: ")
        for train_index, val_index in kf.split(processed_train):
            X_train, X_val = processed_train[train_index], processed_train[val_index]
            y_train, y_val = self.y_train[train_index], self.y_train[val_index]
            data_train = DataIterator(X_train, y_train, params['BATCH_SIZE'])
            prev, early_stop = 0, 0
            for epoch in range(1, params['NUM_EPOCHS'] + 1):
                batch = 0
                for x, y in data_train:
                    net.train(epoch, x, y, train_loss)
                    batch += 1
                    # if batch % 10 == 0:
                    #     Logger.log("Batch {}/{}".format(batch, data_train.num_splits))
                if epoch % freq == 0:
                    Logger.log("Epoch {}\tTraining loss (NLL): {:.6f}".format(epoch, train_loss.getLoss('mse')))
                if abs(prev - train_loss.getLoss('mse')) < 1e-6:
                    early_stop += 1
                    if early_stop > 10:
                        Logger.log("Early stopping at epoch {}".format(epoch))
                        break
                else:
                    early_stop = 0
            net.test(X_val, y_val, test_loss)
            fitness += test_loss.getLoss('accuracy')
            Logger.log("Cross Validation [Fold {}/{}] (MSE/Accuracy): {:.6f} {:.6f}".format(fold, kf.get_n_splits(), test_loss.getLoss('mse'), test_loss.getLoss('accuracy')))
            fold = fold + 1
        fitness /= kf.get_n_splits()

        net.test(processed_test, self.y_test, test_loss)
        ind.net = net
        Logger.log("Generalization Loss (MSE/Accuracy): {:.6f} {:.6f}".format(test_loss.getLoss('mse'), test_loss.getLoss('accuracy')))
        params['CURRENT_EVALUATION'] += 1
        return fitness
