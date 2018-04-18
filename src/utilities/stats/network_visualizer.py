import numpy as np
import matplotlib.pyplot as plt

class Visualizer():
    @classmethod
    def from_torch(cls, data):
        return data.cpu().numpy()

    @classmethod
    def normalize(cls, array):
        array = ((array - array.mean()) / array.std() + 0.5)
        array /= array.max()
        return array.clip(0, 1)*255

    @classmethod
    def visualize_fcn(self, data, fname, image_size, canvas_size, line_weight=1):
        array = data.reshape((-1,) + image_size)
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

    @classmethod
    def visualize_conv(cls, data, fname):
        patches = data.reshape((-1, data.shape[2], data.shape[3]))
        # Inplace modification of ndarrays
        for patch in patches:
            patch[:,:] = cls.normalize(patch)
        result = cls.patch_to_grid(patches, (data.shape[0], data.shape[1]))
        plt.imsave(fname, result, cmap='gray')


    @classmethod
    def patch_to_grid(cls, patches, image_shape, lw=1):
        patch_size = patches[0].shape
        result = np.ones((image_shape[0]*(patch_size[0]+lw)-lw, image_shape[1]*(patch_size[1]+lw)-lw))
        for i in range(len(patches)):
            r, c = i // image_shape[1], i % image_shape[1]
            r_i, c_i = r * (patch_size[0]+lw), c * (patch_size[1]+lw)
            result[r_i:r_i+patch_size[0], c_i:c_i+patch_size[1]] = patches[i]
        return result
