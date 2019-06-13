from os.path import join, exists, split
from os import makedirs
import argparse
import numpy as np
from scipy.signal import convolve2d
import cv2


class Ratliff(object):
    """Class that describes pixelated images"""

    def __init__(self, path_in):
        """Method for initialization"""
        image = cv2.imread(path_in, -1)
        self.raw = image
        self.quad = np.vstack((np.hstack((image[0::2, 0::2],
                                          image[0::2, 1::2])),
                               np.hstack((image[1::2, 1::2],
                                          image[1::2, 0::2]))))
        self.images = []  # list of images I0, I45, I90, I135

    def process(self):
        """ setter method for the method parameter
        Interpolation strategies for reducing IFOV artifacts in miccrogrid ...
        Ratliff 2009"""

        kernels = [np.array([[1, 0], [0, 0.]]),
                   np.array([[0, 1], [0, 0.]]),
                   np.array([[0, 0], [0, 1.]]),
                   np.array([[0, 0], [1, 0.]])]

        Is = []
        for k in kernels:
            Is.append(convolve2d(self.raw, k, mode='same'))

        offsets = [[(0, 0), (0, 1), (1, 1), (1, 0)],
                   [(0, 1), (0, 0), (1, 0), (1, 1)],
                   [(1, 1), (1, 0), (0, 0), (0, 1)],
                   [(1, 0), (1, 1), (0, 1), (0, 0)]]

        self.images = []
        for (j, o) in enumerate(offsets):
            self.images.append(np.zeros(self.raw.shape))
            for ide in range(4):
                self.images[j][o[ide][0]::2, o[ide][1]                               ::2] = Is[ide][o[ide][0]::2, o[ide][1]::2]

        return self.images
