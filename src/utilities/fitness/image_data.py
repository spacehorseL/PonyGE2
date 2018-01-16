import random
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

class ImageData:
    def __init__(self, data):
        self.data = data
    def __str__(self):
        return "image d"
    def setData(self, data):
        self.data = data
        return self
    def getData(self):
        return self.data

class ImageProcessor:
    @classmethod
    def process(cls, operator, arguments):
        return cls.__getattribute__(cls, operator)(cls, arguments)

    def partOfImageERC(self, args):
        return random.random() * 80

    def intensityERC(self, args):
        return random.random() * 255

    def scaleERC(self, args):
        return random.random() + 1;

    def randomERC(self, args):
        return random.random()

    def merge(self, args):
        src = [arg.getData() for arg in args]
        dest = cv.merge(tuple(src))
        # cv.imshow('dest', dest)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        return dest

    def channel(self, args):
        src = np.uint8(args[0].getData())
        choice = int(args[1] * 7)
        if choice == 0:
            dest = cv.cvtColor(src, cv.COLOR_RGB2GRAY)
        elif choice <= 3:
            dest = cv.split(src)[choice-1]
        elif choice <= 6:
            src_hsv = cv.cvtColor(src, cv.COLOR_RGB2HSV)
            dest = cv.split(src_hsv)[choice-4]
        return args[0].setData(dest)

    def canny(self, args):
        src = args[0].getData()
        return args[0].setData(cv.Canny(src, args[1], args[2]))

    def contrast(self, args):
        src = args[0].getData()
        return args[0].setData(cv.equalizeHist(src))

    def filter(self, args):
        src = args[0].getData()
        kernel = int(args[1]) // 2 if args[1] > 3 else 1
        std = kernel * args[2]
        kernel = kernel * 2 + 1
        return args[0].setData(cv.GaussianBlur(src, (kernel, kernel), std))

    def noise(self, args):
        src = args[0].getData()
        return args[0].setData(cv.equalizeHist(src))

    def threshold(self, args):
        src = args[0].getData()
        thres_type = int(args[2] * 5)
        _, t = cv.threshold(src, args[1], 255, thres_type)
        return args[0].setData(t)

    def adaptiveThreshold(self, args):
        src = args[0].getData()
        thres_type = cv.THRESH_BINARY if args[1] > .5 else cv.THRESH_BINARY_INV
        block_size = int(args[2]) // 2 if args[2] > 3 else 1
        block_size = block_size * 2 + 1
        C = args[3]
        return args[0].setData(cv.adaptiveThreshold(src, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, thres_type, block_size, C))

    def meanShiftFilter(self, args):
        src = args[0].getData()
        # print(type(src))
        # src = np.uint8(src)
        return args[0].setData(cv.pyrMeanShiftFiltering(src, args[1], args[2]))

    def gammaCorrection(self, args):
        src = args[0].getData()
        lut = np.zeros(256)
        for i in range(0, len(lut)):
            lut[i] = np.power(i / 255.0, args[1]) * 255.0
        return args[0].setData(np.uint8(cv.LUT(src, lut)))

    def imgx(self, args):
        return ImageData(self.image)
