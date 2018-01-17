from algorithm.parameters import params
from utilities.stats.logger import Logger
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
        fcn = collections.OrderedDict()
        for l in range(1, len(layers)):
            linear = nn.Linear(layers[l-1], layers[l])
            nn.init.xavier_uniform(linear.weight, gain=np.sqrt(2))
            fcn["fcn"+str(l)] = linear
        self.parameters = nn.Sequential(fcn)

    def forward(self, x):
        # x = x.view(-1, 19200)
        for l in self.parameters:
            x = nn.Tanh(l(x))
        return x

class Network():
    def __init__(self, layers, data_size, batch_size=32):
        fcn = collections.OrderedDict()
        for l in range(1, len(layers)):
            linear = nn.Linear(layers[l-1], layers[l])
            nn.init.xavier_uniform(linear.weight, gain=np.sqrt(2))
            fcn["fcn"+str(l)] = linear
            if l != len(layers) - 1:
                fcn["dp"+str(l)] = nn.Dropout(p=0.7)
                fcn["tanh"+str(l)] = nn.Tanh()
        self.model = nn.Sequential(fcn)
        self.criterion = F.mse_loss
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.00015, momentum=0.7)
        self.batch_size = batch_size
        self.data_size = data_size

        if params['CUDA_ENABLED']:
            self.model.cuda()

    def mse_rnk(self, x, y):
        return x.sub(y).float().pow(2).sum() / len(x)**3

    def load_xy(self, x, y):
        return torch.from_numpy(x).float().view(-1, self.data_size), torch.from_numpy(y).float()

    def calc_loss(self, loss_fcn, output, y):
        o_sorted, o_idx = torch.sort(output.data)
        y_sorted, y_idx = torch.sort(y.data)
        mse_rnk = self.mse_rnk(o_idx, y_idx)
        mse = loss_fcn.data[0]

        loss.setLoss('mse', mse)
        loss.setLoss('mse_rnk', mse_rnk)
        return mse_rnk

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

    def test(self, x, y, loss):
        self.model.eval()
        x, y = self.load_xy(x, y)
        if params['CUDA_ENABLED']:
            x, y = x.cuda(), y.cuda()
        x, y = Variable(x, volatile=True), Variable(y)

        output = self.model(x).view(-1)
        loss_fcn = self.criterion(output, y)

        return calc_loss(output, y, loss_fcn)

class ClassificationNet(Network):
    def __init__(self, layers, data_size):
        super(ClassificationNet, self).__init__(layers, data_size)
        self.criterion = F.nll_loss

    def load_xy(self, x, y):
        return torch.from_numpy(x).float().view(-1, self.data_size), torch.from_numpy(y).long()

    def calc_loss(self, loss_fcn, output, y):
        mse = loss_fcn.data[0]
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        accuracy += pred.eq(y.data.view_as(pred)).cpu().sum()

        loss.setLoss('mse', mse)
        loss.setLoss('accuracy', accuracy)
        return accuracy
