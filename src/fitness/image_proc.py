from algorithm.parameters import params
from fitness.base_ff_classes.base_ff import base_ff
from utilities.stats.logger import Logger
from utilities.fitness.image_data import ImageData, ImageProcessor
from utilities.fitness.network import Network
from sklearn.model_selection import train_test_split
import cv2 as cv
import numpy as np
import os, csv, random

class image_proc(base_ff):
    maximise = False  # True as it ever was.
    def __init__(self):
        # Initialise base fitness function class.
        super().__init__()

        # Read images from dataset
        X, Y = [], []
        datapath = os.path.join(params['DATASET'], 'dataset.csv')
        Logger.log("Reading images from {0}...".format(params['DATASET']), info=False)
        with open(datapath) as f:
            fnames = csv.reader(f)
            next(fnames) #Skip header
            for row in fnames:
                fname = os.path.join(params['DATASET'], row[0])
                img = cv.imread(fname) if os.path.isfile(fname) else print(fname+" does not exist")
                X.append(cv.resize(img, (80, 80)))
                Y.append(int(row[1]))
            Logger.log("Done reading dataset with {0} images...".format(len(X)), info=False)

        # Normalize label
        mean, std = np.mean(Y), np.std(Y)
        Y = np.divide(np.subtract(Y, mean), std)
        # Train & test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
        Logger.log("Training & Test split: {0}/{1} with size {2}".format(len(self.X_train), len(self.X_test), self.X_train[0].shape), info=False)
        Logger.log("CUDA ENABLED = {}".format(params['CUDA_ENABLED']), info=False)

    def process_image(self, imgs, ind):
        processed = []
        for img in imgs:
            ImageProcessor.image = img
            processed.append(ind.tree.evaluate_tree())
        return np.asarray(processed)

    def evaluate(self, ind, **kwargs):
        # ind.phenotype will be a string, including function definitions etc.
        # When we exec it, it will create a value XXX_output_XXX, but we exec
        # inside an empty dict for safety.

        p, d = ind.phenotype, {}

        genome, output, invalid, max_depth, nodes = ind.tree.get_tree_info(params['BNF_GRAMMAR'].non_terminals.keys(),[], [])
        Logger.log("Depth: {0}\tGenome: {1}".format(max_depth, genome))

        # Exec the phenotype.
        processed_train = self.process_image(self.X_train, ind)
        processed_test = self.process_image(self.X_test, ind)

        init_size = ImageProcessor.image.shape[0]*ImageProcessor.image.shape[1]*ImageProcessor.image.shape[2]

        net = Network([init_size, 9600, 1200, 1])
        for epoch in range(1, params['NUM_EPOCHS'] + 1):
            net.train(epoch, processed_train, self.y_train)
            fitness = net.test(processed_test, self.y_test)

        # print("Processed {0} images...".format(len(processed)))
        params['CURRENT_EVALUATION'] = params['CURRENT_EVALUATION'] + 1
        return fitness
