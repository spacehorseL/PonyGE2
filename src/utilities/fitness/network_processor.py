import random
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from utilities.stats.logger import Logger
from algorithm.parameters import params

class NetworkProcessor:
    @classmethod
    def process_network(cls, ind, input_size):
        cls.input_size = input_size
        instance = ind.tree.get_terminals([])
        split = np.where(np.array(instance) == 'layer')[0]
        cleaned_instance = [instance[split[i-1]+1:split[i]] for i, t in enumerate(split) if i > 0]

        cls.processed = [input_size[0]//3//2*2+1]
        for i in cleaned_instance:
            # (output, kernel, padding, stride, pool?)
            cls.processed.append(cls.process(i[0], i[1:]))
        return cleaned_instance, cls.processed

    @classmethod
    def process(cls, operator, arguments, processed=[]):
        return cls.__getattribute__(cls, operator)(cls, arguments)

    def grow(self, args):
        return min(self.processed[-1] + np.random.poisson(params['KERNEL_GROW_RATE'])*2, self.input_size[0]//2)

    def shrink(self, args):
        return max(self.processed[-1] - np.random.poisson(params['KERNEL_GROW_RATE'])*2, 3)

    def same(self, args):
        return self.processed[-1]

    def random(self, args):
        return self.input_size[0]//np.random.randint(2, 10)//2*2+1
