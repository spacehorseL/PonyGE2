import random
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from utilities.stats.logger import Logger
from algorithm.parameters import params

class NetworkProcessor:
    @classmethod
    def process_network(cls, tree, input_size, fixed):
        cls.input_size = input_size
        # Obtain sequence of terminals
        instance = tree.get_terminals([])
        # Define split points (the first element is 'fixedlayer')
        split = [i for i, v in enumerate(instance) if v in ['layer', 'onebyonelayer', 'scalinglayer', 'fixedlayer']]
        # Split list into sublists, each sublist represent one layer
        cleaned_instance = [instance[split[i-1]:split[i]] for i, t in enumerate(split) if i > 0] + instance[split[-1]:]

        cls.processed, j = [], 0
        for i in cleaned_instance:
            if 'fixedlayer' in i:
                # Append fixed layer (pre-defined layers)
                cls.processed.append(fixed[j])
                j+=1
            elif 'onebyonelayer' in i:
                # Changes only output size => used to reduce/increase dimensionality
                output_channel = cls.process(i[1], None)
                cls.processed.append((output_channel, 1, 0, 1, False))
            elif 'scalinglayer' in i:
                # Changes nothing => used to simulate a missing layer
                output_channel = cls.process('same_o', None)
                cls.processed.append((output_channel, 1, 0, 1, False))
            else:
                # Cleaned instance: 'layer', 'output', 'kernel'
                # Layer definition: (output, kernel, padding, stride, pooling?)
                output_channel = cls.process(i[1], None)
                kernel_size = cls.process(i[2], None)
                padding, stride, pool = kernel_size // 2, 1, False
                cls.processed.append((output_channel, kernel_size, padding, stride, pool))
        return cleaned_instance, cls.processed

    @classmethod
    def process(cls, operator, arguments, processed=[]):
        return cls.__getattribute__(cls, operator)(cls, arguments)

    def grow_k(self, args):
        return min(self.processed[-1][1] + np.random.poisson(params['KERNEL_GROW_RATE'])*2, self.input_size[0]//2)

    def shrink_k(self, args):
        return max(self.processed[-1][1] - np.random.poisson(params['KERNEL_GROW_RATE'])*2, 3)

    def same_k(self, args):
        return self.processed[-1][1]

    def random_k(self, args):
        return self.input_size[0]//np.random.randint(2, 10)//2*2+1

    def grow_o(self, args):
        return int(self.processed[-1][0] * (1 + abs(np.random.normal(0, params['FILTER_GROW_RATE']))))

    def shrink_o(self, args):
        return max(int(self.processed[-1][0] * (1 - abs(np.random.normal(0, params['FILTER_GROW_RATE'])))), 16)

    def same_o(self, args):
        return self.processed[-1][0]
