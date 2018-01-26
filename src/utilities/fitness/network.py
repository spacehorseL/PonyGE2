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
        
        Logger.log("Network initialized with layers: ")
        for l in [n for n in self.model.named_modules()][1:]:
            Logger.log("\t{}".format(l))
        
        if params['CUDA_ENABLED']:
            self.model = nn.DataParallel(self.model).cuda()

    def mse_rnk(self, x, y=None):
        if y is not None:
            x = x.sub(y)
        return x.float().pow(2).sum() / len(x)**3

    def load_xy(self, x, y):
        return torch.from_numpy(x).float().view(-1, self.data_size), torch.from_numpy(y).float()

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

        output = self.model(x) #.view(-1)
        loss_fcn = self.criterion(output, y)
        
        return self.calc_loss(output, y, loss_fcn, loss)
    
    def visualize(self, layers, image_size, canvas_size):
        Logger.log("Visualizing layers: " + ", ".join([n[0] for n in layers]))
        for name, fname in layers:
            Visualizer.from_torch(self.model.__getattr__(name)).visualize_fcn(fname+'.png', image_size, canvas_size)

class ClassificationNet(Network):
    def __init__(self, layers, data_size):
        super(ClassificationNet, self).__init__(layers, data_size)
        self.criterion = F.nll_loss

    def load_xy(self, x, y):
        return torch.from_numpy(x).float().view(-1, self.data_size), torch.from_numpy(y).long()

    def calc_loss(self, output, y, loss_fcn, loss):
        accuracy = 0
        mse = loss_fcn.data[0]
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        accuracy += pred.eq(y.data.view_as(pred)).cpu().sum()
        accuracy /= len(output.data)
        loss.setLoss('mse', mse)
        loss.setLoss('accuracy', accuracy)
        return accuracy
