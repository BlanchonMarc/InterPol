from os.path import join, exists, split
from os import makedirs
import argparse
import numpy as np
from scipy.signal import convolve2d
import cv2
import scipy


# Definition of the Polaim class


class Bicubic(object):
    """Class that describes pixelated images"""

    def __init__(self, path_in):
        """Method for initialization"""
        image = cv2.imread(path_in, -1)
        self.raw = image
        self.images = []
        self.images.append(image[0::2, 0::2])
        self.images.append(image[0::2, 1::2])
        self.images.append(image[1::2, 1::2])
        self.images.append(image[1::2, 0::2])

    def process(self):

        for indx in range(len(self.images)):
            self.images[indx] = scipy.misc.imresize(
                self.images[indx], self.raw.shape, 'bicubic')

        return self.images
