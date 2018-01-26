import numpy as np
import matplotlib.pyplot as plt

class Visualizer():
    def __init__(self, arr):
        self.arr = arr
        
    @classmethod
    def from_torch(cls, layer):
        return cls(layer.weight.data.numpy())
    
    def normalize(self, array):
        array = ((array - array.mean()) / array.std() + 0.5)
        array /= array.max()
        return array.clip(0, 1)
        
    def visualize_fcn(self, fname, image_size, canvas_size, line_weight=1):
        array = self.arr.reshape((-1,) + image_size)
        vertical_line = np.ones((image_size[0], line_weight, image_size[2]))
        horizontal_line = np.ones((line_weight, image_size[1]*canvas_size[0] + line_weight*(canvas_size[0]-1), image_size[2]))
        
        result = np.array([]).reshape((0, horizontal_line.shape[1], image_size[2]))
        for i in range(0, canvas_size[1]-1):
            row = np.array([]).reshape((image_size[0], 0, image_size[2]))
            idx = i*canvas_size[1]
            for j in range(0, canvas_size[0]):
                img = self.normalize(array[idx+j]) if idx + j < len(array) else np.ones(image_size)
                row = np.concatenate((row, img), axis=1)
                if j != canvas_size[0] - 1:
                    row = np.concatenate((row, vertical_line), axis=1)
            result = np.concatenate((result, row, horizontal_line), axis=0)
        result = np.concatenate((result, row), axis=0)
        
        plt.imsave(fname, result, cmap='plasma')
        return result
    