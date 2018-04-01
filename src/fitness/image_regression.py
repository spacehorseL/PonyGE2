from algorithm.parameters import params
from fitness.base_ff_classes.base_ff import base_ff
from utilities.stats.logger import Logger
from utilities.stats.individual_stat import stats
from utilities.fitness.image_processor import ImageProcessor
from utilities.fitness.network import RegressionNet
from utilities.fitness.read_xy import DataReader
from sklearn.model_selection import train_test_split, KFold
import cv2 as cv
import numpy as np
import os, csv, random

class image_regression(base_ff):
    maximise = False  # True as it ever was.
    def __init__(self):
        # Initialise base fitness function class.
        super().__init__()

        # Read images from dataset
        X, y = DataReader.read_data(params['DATASET_ID'])

        # Train & test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        # Normalize label
        mean, std = np.mean(self.y_train), np.std(self.y_train)
        self.y_train = np.divide(np.subtract(self.y_train, mean), std)
        self.y_test = np.divide(np.subtract(self.y_test, mean), std)

        Logger.log("Training & Test split: {0}/{1} with size {2}".format(len(self.X_train), len(self.X_test), self.X_train[0].shape), info=False)
        Logger.log("CUDA ENABLED = {}".format(params['CUDA_ENABLED']), info=False)
        Logger.fcreate("rankHist", "log.rank")

    def evaluate(self, ind, **kwargs):
        # ind.phenotype will be a string, including function definitions etc.
        # When we exec it, it will create a value XXX_output_XXX, but we exec
        # inside an empty dict for safety.

        p, d = ind.phenotype, {}

        genome, output, invalid, max_depth, nodes = ind.tree.get_tree_info(params['BNF_GRAMMAR'].non_terminals.keys(),[], [])
        Logger.log("Depth: {0}\tGenome: {1}".format(max_depth, genome))

        # Exec the phenotype.
        processed_train = ImageProcessor.process_images(self.X_train, ind.tree)
        processed_test = ImageProcessor.process_images(self.X_test, ind.tree)

        init_size = ImageProcessor.image.shape[0]*ImageProcessor.image.shape[1]*ImageProcessor.image.shape[2]

        train_loss = stats('mse')
        test_loss = stats('mse_rnk')
        kf = KFold(n_splits=params['CROSS_VALIDATION_SPLIT'])
        net = RegressionNet([init_size, 9600, 1200, 1])
        fitness, fold = 0, 1

        for train_index, val_index in kf.split(processed_train):
            X_train, X_val = processed_train[train_index], processed_train[val_index]
            y_train, y_val = self.y_train[train_index], self.y_train[val_index]
            for epoch in range(1, params['NUM_EPOCHS'] + 1):
                net.train(epoch, X_train, y_train, train_loss)
                if epoch % 5 == 0:
                    Logger.log("Epoch {}\tTraining loss (MSE): {:.6f}".format(epoch, train_loss.getLoss('mse')))
            net.test(X_val, y_val, test_loss)
            fitness += test_loss.getLoss('mse_rnk')
            Logger.log("Cross Validation [Fold {}/{}] (MSE/MSE_RNK): {:.6f} {:.6f}".format(fold, kf.get_n_splits(), test_loss.getLoss('mse'), test_loss.getLoss('mse_rnk')))
            fold = fold + 1
        fitness /= kf.get_n_splits()
        ind.stats = test_loss

        net.test(processed_test, self.y_test, test_loss)
        Logger.log("Generalization Loss (MSE/MSE_RNK): {:.6f} {:.6f}".format(test_loss.getLoss('mse'), test_loss.getLoss('mse_rnk')))
        params['CURRENT_EVALUATION'] += 1
        return fitness

    def cleanup(self, inds):
        best_ind, best_fitness = None, float("inf")
        for ind in inds:
            if best_fitness > ind.fitness:
                best_fitness = ind.fitness
                best_ind = ind
        best_rnk = ",".join([str(i) for i in ind.stats.getList("rankHist")])
        Logger.fwrite("rankHist", "{},{},{}".format(params['CURRENT_GENERATION'], best_fitness, best_rnk))
