from algorithm.parameters import params
from utilities.stats.logger import Logger
from utilities.stats.individual_stat import stats
from utilities.stats.network_visualizer import Visualizer
import collections, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from utilities.fitness.models.plain_nets import *
from utilities.fitness.models.res_nets import *
global prenet, theirnet

class Network():
    def __init__(self, batch_size=32):
        if params['RANDOM_SEED']:
            torch.manual_seed(int(params['RANDOM_SEED']))
        self.batch_size = batch_size
        self.measures = ['mse']
        self.train_loss, self.test_loss = stats('mse'), stats('mse')
        self.validation_log, self.test_log = [], []

    def reinitialize_params(self):
        self.model.reinitialize_params()

    def train(self, epoch, x, y):
        # Set model to training mode for Dropout and BatchNorm operations
        self.model.train()
        x, y = self.load_xy(x, y)
        if params['CUDA_ENABLED']:
            x, y = x.cuda(), y.cuda()
        x, y = Variable(x), Variable(y)

        output = self.model(x)
        loss_fcn = self.criterion(output, y)
 
        self.optimizer.zero_grad()
        loss_fcn.backward()
        self.optimizer.step()
        self.train_loss.setLoss('mse', loss_fcn.data.item())


    def test(self, x, y, print_confusion=False):
        self.model.eval()
        x, y = self.load_xy(x, y)
        if params['CUDA_ENABLED']:
            x, y = x.cuda(), y.cuda()
        x, y = Variable(x), Variable(y)

        output = self.model(x)
        loss_fcn = self.criterion(output, y)

        return self.calc_loss(output, y, loss_fcn, self.test_loss, print_confusion=print_confusion)

    def visualize(self, layers, image_size, canvas_size):
        Logger.log("Visualizing layers: " + ", ".join([n[0] for n in layers]))
        model = [m for m in self.model.modules()][1] if params['CUDA_ENABLED'] else self.model
        for name, fname in layers:
            Visualizer.from_torch(model.__getattr__(name)).visualize_fcn(fname+'.png', image_size, canvas_size)

    @classmethod
    def calc_conv_output(cls, conv_layers, img_shape):
        output, sizes = img_shape, []
        for i, l in enumerate(conv_layers):
            output_channel, kernel, stride, pool = l[0], l[1], l[3], l[4]
            factor = 1 if not pool else 2
            factor *= stride
            output = (output[0]//factor, output[1]//factor, output_channel)
            sizes.append(output)
        return sizes

    @classmethod
    def assert_net(cls, conv_layers, fcn_layers, img_shape):
        output = cls.calc_conv_output(conv_layers, img_shape)[-1]
        assert fcn_layers[0] == output[0]*output[1]*output[2]
        return output

    def log_model(self):
        Logger.log("---------------------------------------------------", info=False)
        Logger.log("Neural Network Setup --", info=False)
        Logger.log("\tEpochs / CV fold: \t{} * {} ({} total)".format(params['NUM_EPOCHS'], params['CROSS_VALIDATION_SPLIT'], params['NUM_EPOCHS']*params['CROSS_VALIDATION_SPLIT']), info=False)
        Logger.log("\tBatch size: \t\t{}".format(params['BATCH_SIZE']), info=False)
        Logger.log("\tLearning rate / Momentum: \t{} / {}".format(params['LEARNING_RATE'], params['MOMENTUM']), info=False)
        Logger.log("\tNetwork structure = \n{}".format(self.model), info=False)
        Logger.log("---------------------------------------------------", info=False)

    def get_test_loss(self):
        return ([self.test_loss[m] for m in self.measures])

    def get_test_loss_str(self):
        loss = self.get_test_loss()
        return " ".join(["{:.6f}".format(l) for l in loss])

    def save_validation_loss(self):
        self.validation_log += [np.array(self.get_test_loss())]

    def save_test_loss(self):
        self.test_log += [np.array(self.get_test_loss())]

    def get_validation_log(self):
        return np.array(self.validation_log)

    def get_test_log(self):
        return np.array(self.test_log)

class RegressionNet(Network):
    def __init__(self, fcn_layers, conv_layers, batch_size=32):
        super(RegressionNet, self).__init__(batch_size=batch_size)
        self.model = eval(params['NETWORK_MODEL'])(fcn_layers=fcn_layers, conv_layers=conv_layers)
        self.criterion = F.mse_loss
        self.optimizer = optim.SGD(self.model.parameters(), lr=params['LEARNING_RATE'], momentum=params['MOMENTUM'])
        self.measures = ['mse', 'mse_rnk']
        self.train_loss, self.test_loss = stats('mse'), stats('mse_rnk')
        self.log_model()

    def calc_msernk(self, x, y=None):
        if y is not None:
            x = x.sub(y)
        return x.float().pow(2).sum() / len(x)**3

    def load_xy(self, x, y):
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()

    def calc_loss(self, output, y, loss_fcn, loss, print_confusion=False):
        o_sorted, o_idx = torch.sort(output.data.view(-1))
        y_sorted, y_idx = torch.sort(y.data)
        o_idx = o_idx.view_as(y_idx)
        diff = o_idx.sub(y_idx).abs()

        loss['mse'] = loss_fcn.data.item()
        loss['mse_rnk'] = self.calc_msernk(diff)
        loss.setList('rankHist', diff.tolist())
        return loss['mse_rnk']

    def get_fitness(self):
        # return mean of mse_rnk (at index 1)
        return np.array(self.validation_log).mean(axis=0)[1]

class ClassificationNet(Network):
    def __init__(self, fcn_layers, conv_layers, input_channel=params['INPUT_CHANNEL']):
        super(ClassificationNet, self).__init__()
        self.model = eval(params['NETWORK_MODEL'])(fcn_layers=fcn_layers, conv_layers=conv_layers, input_channel=input_channel)
        self.criterion = F.cross_entropy
        self.optimizer = optim.SGD(self.model.parameters(), lr=params['LEARNING_RATE'], momentum=params['MOMENTUM'])
        self.measures = ['nll', 'accuracy', 'top5']
        self.train_loss, self.test_loss = stats('nll'), stats('accuracy')
        self.log_model()

    def load_xy(self, x, y):
        return torch.from_numpy(x).float(), torch.from_numpy(y).long()

    def print_confusion_matrix(self, pred, y, num_classes):
        num_pred = pred.size()[0]
        pred, y = pred.view(-1).cpu().numpy(), y.view(-1).cpu().numpy()
        def dot(v1, v2):
            return np.dot(v1.astype(np.int), v2.astype(np.int))
        Logger.log("CM ({} cls): TP\tTN\tFP\tFN\tAccurcay\tTotal: {}".format(num_classes, num_pred))
        for i in range(0, num_classes):
            p_, y_ = pred==i, y==i
            p_neg, y_neg = p_==0, y_==0
            tp, tn, fp, fn = dot(p_, y_), dot(p_neg, y_neg), dot(p_, y_neg), dot(p_neg, y_)
            num_c = p_.sum()
            weighted = tp / num_c + tn / (num_pred - num_c)
            Logger.log("Class [{:02d}]: {}\t{}\t{}\t{}\t{:.6f}\t{}".format(i, tp, tn, fp, fn, weighted, num_c))

    def calc_accuracy(self, output, y):
        # get the index of the max log-probability
        # pred = output.data.max(1, keepdim=True)[1]
        return self.calc_topk(output, y, 1)

    def calc_topk(self, output, y, k):
        # topk(): returns tuple (values, indices)
        return sum([a in b for a,b in zip(y.data, output.data.topk(k)[1])]) / len(output.data)

    def calc_loss(self, output, y, loss_fcn, loss, print_confusion=False):
        loss['nll'] = loss_fcn.data.item()
        loss['accuracy'] = self.calc_accuracy(output, y)
        loss['top5'] = self.calc_topk(output, y, 5)
        # if print_confusion:
        #     self.print_confusion_matrix(pred, y.data, output.size()[1])
        return loss['accuracy']

    def get_fitness(self):
        # return mean of accuracy (at index 1)
        return np.array(self.validation_log).mean(axis=0)[1]

class EvoClassificationNet(ClassificationNet):
    def __init__(self, fcn_layers, conv_layers):
        self.model = ConvModel(fcn_layers, conv_layers)
        self.criterion = F.cross_entropy
        self.optimizer = optim.SGD(self.model.parameters(), lr=params['LEARNING_RATE'], momentum=params['MOMENTUM'])
        self.log_model()

class EvoPretrainedClassificationNet(ClassificationNet):
    def __init__(self, fcn_layers, conv_layers):
        super(EvoPretrainedClassificationNet, self).__init__(fcn_layers, conv_layers)

        pretrained_model_path = params['PRETRAINED_MODEL']
        prenet = ClassificationNet(params['FCN_LAYERS'], params['CONV_LAYERS'], input_channel=3)
        if not os.path.isfile(pretrained_model_path):
            Logger.log("Training Pretrained Network: ")
            Logger.log("Saving Pretrained Network to: {}".format(pretrained_model_path))
            # torch.save(prenet.model.state_dict(), pretrained_model_path)
        prenet.model.load_state_dict(torch.load(pretrained_model_path))

        self.model = ConvModel(fcn_layers=fcn_layers, conv_layers=conv_layers)

        Logger.log('Transfer parameters start...')

        # Transfer the weights
        prenet_m = iter(prenet.model.named_parameters())
        p_name, p_params = next(prenet_m)

        for q_name, q_params in self.model.named_parameters():
            if p_name == q_name:
                Logger.log('Set parameters: {}'.format(p_name))
                # Skip initialize if 1D bias
                if p_name.find('bias') < 0:
                    nn.init.xavier_uniform_(q_params, gain=np.sqrt(2))
                # Set extra parameters to zero for debugging
                if params['DEBUG_NET']:
                    Logger.log('Debug network :: Fill 0')
                    q_params.data.fill_(0)

                # Transfer convolution params
                if p_name.find('conv') >= 0 and p_name.find('weight') >= 0:
                    q_params.data[:p_params.data.shape[0],:p_params.data.shape[1],:,:] = p_params.data.clone()
                    # template = p_params.data.view((-1, p_params.data.shape[2], p_params.data.shape[3]))
                    # for sub in patch[:, p_params.data.shape[1]:]:
                    #     sub = template[torch.randperm(len(template))]
                    # for sub in patch[p_params.data.shape[0]:,:]:
                    #     sub = template[torch.randperm(len(template))]
                # Transfer fully connected params
                elif p_name.find('fcn') >= 0 and p_name.find('weight') >= 0:
                    q_params.data[:p_params.data.shape[0],:p_params.data.shape[1]] = p_params.data.clone()
                # Transfer bias params
                elif p_name.find('bias') >= 0:
                    q_params.data[:p_params.data.shape[0]] = p_params.data.clone()

                # with torch.no_grad():
                #     q_params.copy_(patch)
                # Set to next layer of pretrained
                try:
                    p_name, p_params = next(prenet_m)
                except StopIteration:
                    break
            else:
                print("WARNING: Module {0} is not initialized.".format(q_name))

        for q_name, q_params in self.model.named_parameters():
            print("Model parameters: ", q_name)
        for p_name, p_params in prenet.model.named_parameters():
            print("Prenet parameters: ", p_name)

        # Release pretrained model
        prenet.model.cpu()
        del prenet
        torch.cuda.empty_cache()
        self.optimizer = optim.SGD(self.model.parameters(), lr=params['LEARNING_RATE'], momentum=params['MOMENTUM'])
