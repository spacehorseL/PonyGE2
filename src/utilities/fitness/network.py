from algorithm.parameters import params
from utilities.stats.logger import Logger
from utilities.stats.network_visualizer import Visualizer
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class Model(nn.Module):
    def __init__(self, layers):
        super(Model, self).__init__()
        self.layers = layers
        self.conv_layers = self.set_conv()
        self.fcn_layers = self.set_fcn(layers)

        if params['CUDA_ENABLED']:
            if self.conv_layers:
                self.conv_layers = nn.DataParallel(self.conv_layers).cuda()
            if self.fcn_layers:
                self.fcn_layers = nn.DataParallel(self.fcn_layers).cuda()

    def set_conv(self):
        return

    def set_fcn(self, layers):
        fcn = collections.OrderedDict()
        for l in range(1, len(layers)):
            linear = nn.Linear(layers[l-1], layers[l])
            nn.init.xavier_uniform(linear.weight, gain=np.sqrt(2))
            fcn["fcn"+str(l)] = linear
            if l != len(layers) - 1:
                fcn["dp"+str(l)] = nn.Dropout(p=0.7)
                fcn["reluf"+str(l)] = nn.ReLU(inplace=True)
        return nn.Sequential(fcn)

    def reinitialize_params(self):
        for l in self.conv_layers.module if params['CUDA_ENABLED'] else self.conv_layers:
            if hasattr(l, 'weight'):
                nn.init.xavier_uniform(l.weight, gain=np.sqrt(2))
        for l in self.fcn_layers.module if params['CUDA_ENABLED'] else self.fcn_layers:
            if hasattr(l, 'weight'):
                nn.init.xavier_uniform(l.weight, gain=np.sqrt(2))
        if params['DEBUG_NET']:
            for l in self.conv_layers.module if params['CUDA_ENABLED'] else self.conv_layers:
                if hasattr(l, 'weight'):
                    print("Layer {}: \t\t{}".format(str(l), l.weight.mean().data.cpu().numpy()))
            for l in self.fcn_layers.module if params['CUDA_ENABLED'] else self.fcn_layers:
                if hasattr(l, 'weight'):
                    print("Layer {}: \t\t{}".format(str(l), l.weight.mean().data.cpu().numpy()))

    def get_module(self):
        return self.model.modules()

class FCNModel(Model):
    def __init__(self, layers):
        super(FCNModel, self).__init__(layers)

    def forward(self, x):
        x = x.view(-1, self.layers[0])
        x = self.fcn_layers(x)
        return x

class Conv1Model(Model):
    def __init__(self, layers):
        super(Conv1Model, self).__init__(layers)

    def set_conv(self):
        fcn = collections.OrderedDict()
        fcn['conv1'] = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        fcn['relu1'] = nn.ReLU(inplace=True)
        fcn['pool1'] = nn.MaxPool2d(kernel_size=2, stride=2)
        return nn.Sequential(fcn)

    def forward(self, x):
        x = x.view(x.size(0), x.size(3), x.size(1), x.size(2))
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fcn_layers(x)
        return x

class Conv2Model(Conv1Model):
    def __init__(self, layers):
        super(Conv2Model, self).__init__(layers)

    def set_conv(self):
        fcn = collections.OrderedDict()
        fcn['conv1'] = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        fcn['relu1'] = nn.ReLU(inplace=True)
        fcn['pool1'] = nn.MaxPool2d(kernel_size=2, stride=2)
        fcn['conv2'] = nn.Conv2d(64, 256, kernel_size=3, padding=1)
        fcn['relu2'] = nn.ReLU(inplace=True)
        fcn['pool2'] = nn.MaxPool2d(kernel_size=2, stride=2)
        return nn.Sequential(fcn)

class AlexNetModel(Conv1Model):
    def __init__(self, layers):
        super(AlexNetModel, self).__init__([256, 10])

    def set_conv(self):
        fcn = collections.OrderedDict()
        fcn['conv1'] = nn.Conv2d(params['INPUT_CHANNEL'], 64, kernel_size=11, stride=4, padding=5)
        fcn['relu1'] = nn.ReLU(inplace=True)
        fcn['pool1'] = nn.MaxPool2d(kernel_size=2, stride=2)
        fcn['conv2'] = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        fcn['relu2'] = nn.ReLU(inplace=True)
        fcn['pool2'] = nn.MaxPool2d(kernel_size=2, stride=2)
        fcn['conv3'] = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        fcn['relu3'] = nn.ReLU(inplace=True)
        fcn['conv4'] = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        fcn['relu4'] = nn.ReLU(inplace=True)
        fcn['conv5'] = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        fcn['relu5'] = nn.ReLU(inplace=True)
        fcn['pool5'] = nn.MaxPool2d(kernel_size=2, stride=2)
        return nn.Sequential(fcn)

class AlexNetModel2(Conv1Model):
    def __init__(self, layers):
        super(AlexNetModel2, self).__init__([320, 10])

    def set_conv(self):
        fcn = collections.OrderedDict()
        fcn['conv1'] = nn.Conv2d(params['INPUT_CHANNEL'], 80, kernel_size=11, stride=4, padding=5)
        fcn['relu1'] = nn.ReLU(inplace=True)
        fcn['pool1'] = nn.MaxPool2d(kernel_size=2, stride=2)
        fcn['conv2'] = nn.Conv2d(80, 240, kernel_size=5, padding=2)
        fcn['relu2'] = nn.ReLU(inplace=True)
        fcn['pool2'] = nn.MaxPool2d(kernel_size=2, stride=2)
        fcn['conv3'] = nn.Conv2d(240, 480, kernel_size=3, padding=1)
        fcn['relu3'] = nn.ReLU(inplace=True)
        fcn['conv4'] = nn.Conv2d(480, 320, kernel_size=3, padding=1)
        fcn['relu4'] = nn.ReLU(inplace=True)
        fcn['conv5'] = nn.Conv2d(320, 320, kernel_size=3, padding=1)
        fcn['relu5'] = nn.ReLU(inplace=True)
        fcn['pool5'] = nn.MaxPool2d(kernel_size=2, stride=2)
        return nn.Sequential(fcn)

class Network():
    def __init__(self, batch_size=32):
        if params['RANDOM_SEED']:
            torch.manual_seed(int(params['RANDOM_SEED']))
        self.batch_size = batch_size

    def reinitialize_params(self):
        self.model.reinitialize_params()

    def train(self, epoch, x, y, loss):
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
        loss.setLoss('mse', loss_fcn.data[0])

    def test(self, x, y, loss, print_confusion=False):
        self.model.eval()
        x, y = self.load_xy(x, y)
        if params['CUDA_ENABLED']:
            x, y = x.cuda(), y.cuda()
        x, y = Variable(x, volatile=True), Variable(y)

        output = self.model(x)
        loss_fcn = self.criterion(output, y)

        return self.calc_loss(output, y, loss_fcn, loss, print_confusion=print_confusion)

    def visualize(self, layers, image_size, canvas_size):
        Logger.log("Visualizing layers: " + ", ".join([n[0] for n in layers]))
        model = [m for m in self.model.modules()][1] if params['CUDA_ENABLED'] else self.model
        for name, fname in layers:
            Visualizer.from_torch(model.__getattr__(name)).visualize_fcn(fname+'.png', image_size, canvas_size)

class RegressionNet(Network):
    def __init__(self, layers, batch_size=32):
        super(RegressionNet, self).__init__(batch_size=batch_size)
        self.model = FCNModel(layers)
        self.criterion = F.mse_loss
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.00015, momentum=0.8)

    def mse_rnk(self, x, y=None):
        if y is not None:
            x = x.sub(y)
        return x.float().pow(2).sum() / len(x)**3

    def load_xy(self, x, y):
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()

    def calc_loss(self, output, y, loss_fcn, stats):
        o_sorted, o_idx = torch.sort(output.data.view(-1))
        y_sorted, y_idx = torch.sort(y.data)
        o_idx = o_idx.view_as(y_idx)
        diff = o_idx.sub(y_idx).abs()
        mse_rnk = self.mse_rnk(diff)
        mse = loss_fcn.data[0]

        stats.setLoss('mse', mse)
        stats.setLoss('mse_rnk', mse_rnk)
        stats.setList('rankHist', diff.tolist())
        return mse_rnk

class ClassificationNet(Network):
    def __init__(self, layers):
        super(ClassificationNet, self).__init__()
        self.model = eval(params['NETWORK_MODEL'])(layers)
        self.criterion = F.cross_entropy
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.00015, momentum=0.8)

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

    def calc_loss(self, output, y, loss_fcn, loss, print_confusion=False):
        accuracy = 0
        mse = loss_fcn.data[0]
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        accuracy += pred.eq(y.data.view_as(pred)).cpu().sum()
        accuracy /= len(output.data)
        loss.setLoss('mse', mse)
        loss.setLoss('accuracy', accuracy)
        # if print_confusion:
        #     self.print_confusion_matrix(pred, y.data, output.size()[1])
        return accuracy
