from algorithm.parameters import params
from fitness.base_ff_classes.base_ff import base_ff
from utilities.stats.logger import Logger
from utilities.stats.individual_stat import stats
from utilities.fitness.image_processor import ImageProcessor
from utilities.fitness.network import ClassificationNet
from utilities.fitness.preprocess import DataIterator, check_class_balance, read_cifar
from sklearn.model_selection import train_test_split, KFold
import cv2 as cv
import numpy as np
import os, csv, random, pickle, time

class cifar10(base_ff):
    maximise = True  # True as it ever was.
    def __init__(self):
        # Initialise base fitness function class.
        super().__init__()
        self.layers = params['FCN_LAYERS']#[3072, 8192, 8192, 10]
        self.resize = params['RESIZE']

        # Read images from dataset
        Logger.log("Reading images from {} ...".format(params['DATASET']), info=False)

        X, Y = np.array([]), np.array([])
        for num in range(1, 6):
            datapath = os.path.join(params['DATASET'], 'data_batch_'+str(num))
            x, y = read_cifar(datapath)
            X = np.append(X, x)
            Y = np.append(Y, y)
        X = np.reshape(X, (-1, 32,32,3))
        Logger.log("Done reading dataset with {} images...".format(len(X)), info=False)

        # Train & test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

        # Check class balance between splits
        classes, class_balance_train, class_balance_test = check_class_balance(self.y_train, self.y_test)
        Logger.log("---------------------------------------------------", info=False)
        Logger.log("Class Balance --", info=False)
        Logger.log("\tClass: \t{}".format("\t".join([str(c) for c in classes])), info=False)
        Logger.log("\tTrain: \t{}".format("\t".join([str(n) for n in class_balance_train])), info=False)
        Logger.log("\tTest: \t{}".format("\t".join([str(n) for n in class_balance_test])), info=False)
        Logger.log("\tTotal: \t{}\t{}".format("\t".join([str(n) for n in class_balance_train + class_balance_test]), (class_balance_train + class_balance_test).sum()), info=False)

        Logger.log("---------------------------------------------------", info=False)
        Logger.log("General Setup --", info=False)
        Logger.log("\tCUDA enabled: \t{}".format(params['CUDA_ENABLED']), info=False)
        Logger.log("\tDebug network enabled: \t{}".format(params['DEBUG_NET']), info=False)

        Logger.log("---------------------------------------------------", info=False)
        Logger.log("Data Preprocess --", info=False)
        Logger.log("\tNumber of samples: \t{}".format(len(X)), info=False)
        Logger.log("\tTraining / Test split: \t{}/{}".format(len(self.X_train), len(self.X_test)), info=False)
        Logger.log("\tImage size: \t{}".format(self.X_train[0].shape), info=False)
        Logger.log("\tNormalize after preprocessing: \t{}".format(params['NORMALIZE']), info=False)

        Logger.log("---------------------------------------------------", info=False)
        Logger.log("GP Setup --", info=False)
        Logger.log("\tGrammar file: \t{}".format(params['GRAMMAR_FILE']), info=False)
        Logger.log("\tPoupulation size: \t{}".format(params['POPULATION_SIZE']), info=False)
        Logger.log("\tGeneration num: \t{}".format(params['GENERATIONS']), info=False)
        Logger.log("\tImage resizing (after proc): \t{}".format(self.resize), info=False)
        Logger.log("\tTree depth init (Min/Max): \t{}/{}".format(params['MIN_INIT_TREE_DEPTH'], params['MAX_INIT_TREE_DEPTH']), info=False)
        Logger.log("\tTree depth Max: \t\t{}".format(params['MAX_TREE_DEPTH']), info=False)

        Logger.log("---------------------------------------------------", info=False)
        Logger.log("Neural Network Setup --", info=False)
        Logger.log("\tEpochs / CV fold: \t{} * {} ({} total)".format(params['NUM_EPOCHS'], params['CROSS_VALIDATION_SPLIT'], params['NUM_EPOCHS']*params['CROSS_VALIDATION_SPLIT']), info=False)
        Logger.log("\tBatch size: \t\t{}".format(params['BATCH_SIZE']), info=False)
        Logger.log("\tLearning rate / Momentum: \t{} / {}".format(params['LEARNING_RATE'], params['MOMENTUM']), info=False)
        Logger.log("\tNetwork structure: \n{}".format(ClassificationNet(self.layers, []).model), info=False)

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

        if params['NORMALIZE']:
            Logger.log("Normalizing processed images...")
            processed_train, mean, std = ImageProcessor.normalize(processed_train)
            processed_test, _, _ = ImageProcessor.normalize(processed_test, mean=mean, std=std)
            Logger.log("Mean / Std of training set (by channel): {} / {}".format(mean, std))

        X_test, y_test = processed_test, self.y_test
        image = ImageProcessor.image
        init_size = image.shape[0]*image.shape[1]*image.shape[2]

        train_loss = stats('mse')
        test_loss = stats('accuracy')
        kf = KFold(n_splits=params['CROSS_VALIDATION_SPLIT'])
        net = ClassificationNet(self.layers, [])
        fitness, fold = 0, 1

        Logger.log("Training Start: ")

        # Cross validation
        s_time = np.empty((kf.get_n_splits()))
        validation_acc = np.empty((kf.get_n_splits()))
        test_acc = np.empty((kf.get_n_splits()))
        for train_index, val_index in kf.split(processed_train):
            X_train, X_val = processed_train[train_index], processed_train[val_index]
            y_train, y_val = self.y_train[train_index], self.y_train[val_index]
            data_train = DataIterator(X_train, y_train, params['BATCH_SIZE'])
            early_ckpt, early_stop, early_crit, epsilon = 4, [], params['EARLY_STOP_FREQ'], params['EARLY_STOP_EPSILON']
            s_time[fold-1] = time.time()

            # Train model
            net.model.reinitialize_params()
            for epoch in range(1, params['NUM_EPOCHS'] + 1):
                # mini-batch training
                for x, y in data_train:
                    net.train(epoch, x, y, train_loss)

                # log training loss
                if epoch % params['TRAIN_FREQ'] == 0:
                    Logger.log("Epoch {} Training loss (NLL): {:.6f}".format(epoch, train_loss.getLoss('mse')))

                # log validation/test loss
                if epoch % params['VALIDATION_FREQ'] == 0:
                    net.test(X_val, y_val, test_loss)
                    Logger.log("Epoch {} Validation loss (NLL/Accuracy): {:.6f} {:.6f}".format(epoch, test_loss.getLoss('mse'), test_loss.getLoss('accuracy')))
                    net.test(X_test, y_test, test_loss)
                    Logger.log("Epoch {} Test loss (NLL/Accuracy): {:.6f} {:.6f}".format(epoch, test_loss.getLoss('mse'), test_loss.getLoss('accuracy')))

                # check for early stop
                if epoch == early_ckpt:
                    accuracy = net.test(X_test, y_test, test_loss, print_confusion=True)
                    early_stop.append(accuracy)
                    if len(early_stop) > 3:
                        latest_acc = early_stop[-early_crit:]
                        latest_acc = np.subtract(latest_acc, latest_acc[1:]+[0])
                        if (abs(latest_acc[:-1]) < epsilon).all() == True:
                            Logger.log("Early stopping at epoch {} (latest {} ckpts): {}".format(epoch, early_crit, " ".join(["{:.4f}".format(x) for x in early_stop[-early_crit:]])))
                            break
                    early_ckpt = min(early_ckpt+300, early_ckpt*2)

            # Validate model
            net.test(X_val, y_val, test_loss)
            validation_acc[fold-1] = test_loss.getLoss('accuracy')
            Logger.log("Cross Validation [Fold {}/{}] Validation (NLL/Accuracy): {:.6f} {:.6f}".format(fold, kf.get_n_splits(), test_loss.getLoss('mse'), test_loss.getLoss('accuracy')))

            # Test model
            net.test(processed_test, self.y_test, test_loss)
            test_acc[fold-1] = test_loss.getLoss('accuracy')
            Logger.log("Cross Validation [Fold {}/{}] Test (NLL/Accuracy): {:.6f} {:.6f}".format(fold, kf.get_n_splits(), test_loss.getLoss('mse'), test_loss.getLoss('accuracy')))

            # Calculate time
            s_time[fold-1] = time.time() - s_time[fold-1]
            Logger.log("Cross Validation [Fold {}/{}] Training Time (m / m per epoch): {:.3f} {:.3f}".format(fold, kf.get_n_splits(), s_time[fold-1]/60, s_time[fold-1]/60/epoch))

            fold = fold + 1

        fitness = validation_acc.mean()

        for i in range(0, kf.get_n_splits()):
            Logger.log("STAT -- Model[{}/{}] #{:.3f}m Validation / Generalization accuracy (%): {:.4f} {:.4f}".format(i, kf.get_n_splits(), s_time[i]/60, validation_acc[i]*100, test_acc[i]*100))
        Logger.log("STAT -- Mean Validation / Generatlization accuracy (%): {:.4f} {:.4f}".format(validation_acc.mean()*100, test_acc.mean()*100))
        # ind.net = net
        params['CURRENT_EVALUATION'] += 1
        return fitness
