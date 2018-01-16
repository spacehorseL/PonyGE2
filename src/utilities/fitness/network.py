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

if params['CUDA_ENABLED']:
    model.cuda()

class Network():
    def __init__(self, layers, batch_size=32):
        fcn = collections.OrderedDict()
        for l in range(1, len(layers)):
            linear = nn.Linear(layers[l-1], layers[l])
            nn.init.xavier_uniform(linear.weight, gain=np.sqrt(2))
            fcn["fcn"+str(l)] = linear
            if l != len(layers) - 1:
                fcn["dp"+str(l)] = nn.Dropout(p=0.7)
                fcn["tanh"+str(l)] = nn.Tanh()
        self.model = nn.Sequential(fcn)
        self.criterion = nn.MSELoss(size_average=False)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.00015, momentum=0.7)
        self.batch_size = batch_size

    def train(self, epoch, x, y):
        # Set model to training mode for Dropout and BatchNorm operations
        self.model.train()

        x, y = torch.from_numpy(x).float().view(-1, 19200), torch.from_numpy(y).float()

        # for name, param in self.model.state_dict().items():
        #     print(name, param.size())
        if params['CUDA_ENABLED']:
            x, y = x.cuda(), y.cuda()
        x, y = Variable(x), Variable(y)
        output = self.model(x)
        loss = self.criterion(output, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        Logger.log('Train Epoch({}): {:.6f}'.format(epoch, loss.data[0]))

    def test(self, x, y):
        self.model.eval()
        test_loss = 0
        correct = 0
        x, y = torch.from_numpy(x).float().view(-1, 19200), torch.from_numpy(y).float()
        if params['CUDA_ENABLED']:
            x, y = x.cuda(), y.cuda()
        x, y = Variable(x, volatile=True), Variable(y)
        output = self.model(x).view(-1)
        loss = nn.MSELoss()
        loss = loss(output, y)

        # diff = output.data.sub(y.data)
        o_sorted, o_idx = torch.sort(output.data)
        y_sorted, y_idx = torch.sort(y.data)
        mse_rnk = self.mse_rnk(o_idx, y_idx)

        Logger.log('Test loss (MSE/MSE-RNK): {:.6f} {:.6f}'.format(loss.data[0], mse_rnk))
        return mse_rnk
    def mse_rnk(self, x, y):
        return x.sub(y).float().pow(2).sum() / len(x)**3
